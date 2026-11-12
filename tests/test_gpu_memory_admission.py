import importlib.util
import json
import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
from subprocess import CompletedProcess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from manipulation_agent.batch_lifecycle import preferred_gpu_worker


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / 'scripts' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


remote = load_script('gpu_admission_remote', 'behavior100_remote.py')
batch = load_script('gpu_admission_batch', 'run_behavior100.py')


class SimulatorGpuMemoryAdmissionTests(unittest.TestCase):
    @contextmanager
    def host(self, *, driver='580.173.02', present=True, disk_gib=100):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            encoder = root / 'ffmpeg'
            encoder.write_bytes(b'fixture encoder')
            original_read = Path.read_text

            def read(path, *args, **kwargs):
                if path == Path('/proc/meminfo'):
                    return 'MemAvailable: 104857600 kB\n'
                if path.name == 'memory.current':
                    return '0'
                if path.name == 'memory.max':
                    return 'max'
                if path.name == 'memory.stat':
                    return 'file 0\n'
                return original_read(path, *args, **kwargs)

            def command(arguments):
                output = ''
                if arguments[0] == 'nvidia-smi' and arguments[1].startswith('--query-gpu='):
                    output = f'0, GPU-fixture, 23000, 24564, {driver}\n' if present else ''
                elif arguments[0] == 'df':
                    output = 'Filesystem\n/dev/fixture\n'
                return {'exit_code': 0, 'stdout': output, 'stderr': ''}

            stack.enter_context(patch.object(Path, 'read_text', autospec=True, side_effect=read))
            stack.enter_context(patch.object(remote, 'command', side_effect=command))
            stack.enter_context(patch.object(remote, 'local_quota', return_value={'stdout': ''}))
            stack.enter_context(patch.object(remote.shutil, 'disk_usage',
                                            return_value=SimpleNamespace(free=disk_gib * 1024**3)))
            stack.enter_context(patch.object(remote, 'source_import_readiness',
                                            return_value={'status': 'passed', 'simulator_imported': False}))
            stack.enter_context(patch('manipulation_agent.video.ffmpeg_executable', return_value=str(encoder)))
            socket = stack.enter_context(patch.object(remote.socket, 'socket'))
            socket.return_value.__enter__.return_value.connect_ex.return_value = 111
            yield root

    def test_full_and_light_checks_skip_only_capacity_when_explicitly_disabled(self):
        for light in (False, True):
            with self.subTest(light=light), self.host() as root:
                normal = remote.preflight(0, 60400, root, 14336, light=light)
                disabled = remote.preflight(0, 60400, root, 14336, light=light,
                                            gpu_memory_admission=False)
                self.assertEqual(normal['status'], 'blocked')
                self.assertFalse(normal['checks']['gpu_capacity'])
                self.assertTrue(normal['gpu_policy']['memory_admission_enabled'])
                self.assertEqual(disabled['status'], 'passed')
                self.assertNotIn('gpu_capacity', disabled['checks'])
                self.assertFalse(disabled['gpu_policy']['memory_admission_enabled'])
                self.assertEqual(disabled['gpu_policy']['minimum_free_mib'], 14336)
                self.assertEqual(disabled['gpu_uuid'], 'GPU-fixture')
                self.assertEqual(disabled['checks'],
                                 {key: value for key, value in normal['checks'].items() if key != 'gpu_capacity'})

    def test_disabling_bypasses_both_the_shared_floor_and_legacy_idle_requirement(self):
        with self.host() as root:
            with self.assertRaisesRegex(ValueError, 'at least 12 GiB'):
                remote.preflight(0, 60400, root, 10240, light=True)
            for minimum in (0, 10240):
                self.assertEqual(remote.preflight(0, 60400, root, minimum, light=True,
                                                 gpu_memory_admission=False)['status'], 'passed')
            self.assertFalse(remote.preflight(0, 60400, root, 0, light=True)['checks']['gpu_capacity'])

    def test_missing_gpu_bad_driver_and_low_storage_are_still_rejected(self):
        with self.host(present=False) as root:
            with self.assertRaisesRegex(RuntimeError, 'GPU is unavailable'):
                remote.preflight(0, 60400, root, gpu_memory_admission=False, light=True)
        with self.host(driver='570.0.0') as root:
            result = remote.preflight(0, 60400, root, gpu_memory_admission=False, light=True)
            self.assertEqual(result['status'], 'blocked')
            self.assertFalse(result['checks']['validated_driver_floor'])
        with self.host(disk_gib=8) as root:
            result = remote.preflight(0, 60400, root, gpu_memory_admission=False, light=True,
                                      runtime_storage_paths=[root])
            self.assertEqual(result['status'], 'blocked')
            self.assertFalse(result['checks']['data_disk_free_40GiB'])
            self.assertFalse(result['checks']['runtime_storage_headroom_16GiB'])

    def test_preference_ignores_memory_threshold_only_for_opted_out_workers(self):
        worker = {'id': 'local', 'gpu': 0, 'ssh': ['ssh', 'host']}
        snapshot = '0, GPU-fixture, 23000, 24564, 580.173.02'
        for minimum in (0, 14336):
            enabled = {**worker, 'minimum_free_gpu_mib': minimum}
            disabled = {**enabled, 'simulator_gpu_memory_admission': False}
            self.assertIsNone(preferred_gpu_worker([enabled], {}, 'host', snapshot))
            self.assertEqual(preferred_gpu_worker([disabled], {}, 'host', snapshot), 'local')
            self.assertIsNone(preferred_gpu_worker([disabled], {'local': {'stage': 'running'}}, 'host', snapshot))
            self.assertIsNone(preferred_gpu_worker([disabled], {}, 'host', snapshot, gpu_available=lambda _: False))
            self.assertIsNone(preferred_gpu_worker([disabled], {}, 'host', ''))

    def test_worker_setting_requires_boolean_and_can_be_overridden_per_lane(self):
        config = {'workers': [{'id': 'local', 'gpu': 0, 'ssh': ['ssh', 'host'], 'base_port': 60400}]}
        self.assertTrue(batch.worker_configs(config)[0].get('simulator_gpu_memory_admission', True))
        for value in (0, 1, 'false', None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'must be boolean'):
                batch.worker_configs({**config, 'simulator_gpu_memory_admission': value})
        config['workers'][0]['simulator_gpu_memory_admission'] = False
        self.assertFalse(batch.worker_configs({**config, 'simulator_gpu_memory_admission': True})[0]
                         ['simulator_gpu_memory_admission'])

    def test_scheduler_sends_opt_out_to_worker_preflight(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = batch.Batch.__new__(batch.Batch)
            runner.root = runner.source = Path(folder)
            (runner.root / 'preflight').mkdir()
            runner._worker_local = threading.local()
            runner._config = {'id': 'local', 'gpu': 0, 'base_port': 60400,
                              'data_root': folder, 'sim_python': sys.executable,
                              'simulator_gpu_memory_admission': False, 'prefer_gpu_headroom': False}
            runner.worker_state = Mock()
            runner.memory_policy_args = Mock(return_value=[])
            runner.ssh = Mock(return_value=CompletedProcess([], 0,
                              json.dumps({'status': 'passed', 'checks': {}}), ''))
            self.assertTrue(runner.resource_admission())
            self.assertIn('--skip-gpu-memory-admission', runner.ssh.call_args.args[0][-1])

    def test_remote_cli_applies_opt_out_in_preflight_and_allocated_simulator_unit(self):
        for mode in ('preflight', 'simulate'):
            for disabled in (False, True):
                with self.subTest(mode=mode, disabled=disabled), tempfile.TemporaryDirectory() as folder:
                    root = Path(folder)
                    (root / 'preflight').mkdir()
                    manifest = root / 'manifest.json'
                    manifest.write_text(json.dumps({'tasks': [{'index': 0, 'task': 'fixture',
                        'run_id': 'fixture_r1', 'instance': 301, 'seed': 0, 'instruction': 'Fixture task'}],
                        'agent_profile': 'skills', 'model': 'fixture', 'max_actions': 50,
                        'max_sim_steps': 20000, 'episode_budget_policy': 'unbounded'}))
                    argv = ['behavior100_remote.py', mode, '--manifest', str(manifest),
                            '--data-root', folder, '--gpu', '0', '--port', '60400',
                            '--index', '0', '--unit', 'fixture.service']
                    if disabled:
                        argv.append('--skip-gpu-memory-admission')
                    with patch.object(sys, 'argv', argv), patch.dict(os.environ), \
                            patch.object(remote, 'preflight', return_value={'status': 'passed', 'gpu_uuid': 'GPU-fixture'}) as check, \
                            patch.object(remote, 'assets', return_value={'status': 'passed'}), \
                            patch.object(remote.os, 'execv') as execute, patch('builtins.print'):
                        remote.main()
                    self.assertIs(check.call_args.kwargs['gpu_memory_admission'], not disabled)
                    self.assertEqual(execute.call_count, int(mode == 'simulate'))


if __name__ == '__main__':
    unittest.main()
