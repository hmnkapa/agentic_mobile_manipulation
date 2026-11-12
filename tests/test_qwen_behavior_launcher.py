"""Exercise launch settings through prepare-only; no model or simulator is loaded."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import pytest


GPU_UUIDS = ['GPU-0000-aaaa', 'GPU-1111-bbbb']


@pytest.fixture
def launcher(tmp_path):
    repo = tmp_path / 'checkout'
    scripts = repo / 'scripts'
    scripts.mkdir(parents=True)
    script = scripts / 'run_qwen_behavior.sh'
    shutil.copyfile(Path(__file__).resolve().parents[1] / 'scripts' / script.name, script)
    subprocess.run(['git', 'init', '-q', str(repo)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(repo), 'add', 'scripts'], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Fixture',
                    '-c', 'user.email=fixture@example.invalid', '-c', 'commit.gpgsign=false',
                    'commit', '-qm', 'Synthetic launcher fixture'], check=True, capture_output=True)

    source = tmp_path / 'source-batch'
    source.mkdir()
    instruction = 'Synthetic launcher fixture.'
    (source / 'manifest.json').write_text(json.dumps({'tasks': [{
        'index': 1, 'task': 'turning_on_radio', 'name': 'Radio', 'scene': 'fixture',
        'instance': 301, 'seed': 0, 'instruction': instruction,
        'instruction_sha256': hashlib.sha256(instruction.encode()).hexdigest(),
        'instruction_source': 'synthetic_fixture', 'bddl_sha256': 'fixture',
    }]}))
    template = repo / 'configs/inference/behavior_local.json'
    template.parent.mkdir(parents=True)
    template.write_text(json.dumps({
        'batch': str(source), 'shared_lease_dir': str(tmp_path / 'leases'),
        'workers': [{'id': 'local-gpu0', 'gpu': 0, 'expected_gpu_uuid': GPU_UUIDS[0]}],
    }))
    model = tmp_path / 'model'
    model.mkdir()
    lora = tmp_path / 'adapter'
    lora.mkdir()
    (lora / 'adapter_config.json').write_text('{}')

    commands = tmp_path / 'bin'
    commands.mkdir()
    query = commands / 'nvidia-smi'
    query.write_text('#!/bin/sh\n'
                     'printf "%s\\n" "$@" >> "$FIXTURE_GPU_QUERY_LOG"\n'
                     'if [ "${FIXTURE_GPU_QUERY_FAIL:-0}" = 1 ]; then\n'
                     '  printf "Fixture driver unavailable\\n" >&2; exit 1\n'
                     'fi\n'
                     f'printf "%s\\n" "0, {GPU_UUIDS[0]}" "1, {GPU_UUIDS[1]}"\n')
    query.chmod(0o755)
    query_log = tmp_path / 'gpu-query.log'
    env = {key: value for key, value in os.environ.items()
           if not key.startswith('QWEN_')}
    env.update(PATH=str(commands) + os.pathsep + env['PATH'], CUDA_VISIBLE_DEVICES='1,0',
               FIXTURE_GPU_QUERY_LOG=str(query_log))
    output = tmp_path / 'prepared'

    def run(*args, overrides=None):
        return subprocess.run([
            'bash', str(script), '--controller-python', sys.executable,
            '--qwen-python', sys.executable, '--model', str(model), '--lora', str(lora),
            '--output', str(output), '--prepare-only', *args,
        ], env={**env, **(overrides or {})}, capture_output=True, text=True, timeout=30)

    return run, output, template, query_log


@pytest.mark.parametrize(('inference', 'simulator'), [(0, 1), (1, 0)])
def test_physical_indices_route_both_roles_and_preserve_template(launcher, inference, simulator):
    run, output, template, query_log = launcher
    before = template.read_bytes()
    result = run('--inference-gpu', str(inference), '--simulator-gpu', str(simulator))
    assert result.returncode == 0, result.stderr
    qwen = tomllib.loads((output / 'qwen.toml').read_text())
    runner = json.loads((output / 'runner.json').read_text())
    provenance = json.loads((output / 'launch.json').read_text())
    assert qwen['device'] == 'cuda:0'  # Logical index after UUID isolation.
    assert qwen['gpu_uuid'] == GPU_UUIDS[inference]
    assert runner['workers'] == [{'id': f'local-gpu{simulator}', 'gpu': simulator,
                                 'expected_gpu_uuid': GPU_UUIDS[simulator]}]
    assert provenance['inference_gpu_index'] == inference
    assert provenance['simulator_workers'][0]['gpu_uuid'] == GPU_UUIDS[simulator]
    assert json.loads((output / 'manifest.json').read_text())['episode_budget_policy'] == 'unbounded'
    assert f'Inference: GPU {inference}' in result.stdout
    assert f'Simulator: local-gpu{simulator}' in result.stdout
    assert query_log.read_text().splitlines() == ['--query-gpu=index,uuid', '--format=csv,noheader,nounits']
    assert template.read_bytes() == before


def test_default_roles_keep_gpu0_inference_gpu1_simulation(launcher):
    run, output, _, _ = launcher
    result = run()
    assert result.returncode == 0, result.stderr
    assert tomllib.loads((output / 'qwen.toml').read_text())['gpu_uuid'] == GPU_UUIDS[0]
    assert json.loads((output / 'runner.json').read_text())['workers'][0]['gpu'] == 1


@pytest.mark.parametrize(('args', 'env', 'expected'), [
    ([], {}, 256),
    (['--image-max-token-num', '64'], {}, 64),
    (['--image-max-token-num', '256'], {}, 256),
    ([], {'QWEN_IMAGE_MAX_TOKEN_NUM': '64'}, 64),
    (['--image-max-token-num', '256'], {'QWEN_IMAGE_MAX_TOKEN_NUM': '64'}, 256),
])
def test_image_token_selection_is_recorded_in_new_run(launcher, args, env, expected):
    run, output, _, _ = launcher
    result = run(*args, overrides=env)
    assert result.returncode == 0, result.stderr
    assert tomllib.loads((output / 'qwen.toml').read_text())['image_max_token_num'] == expected
    assert json.loads((output / 'launch.json').read_text())['image_max_token_num'] == expected
    assert f'{expected} visual tokens per image' in result.stdout


@pytest.mark.parametrize('value', ['0', '128', '064', '64.0', 'true', 'arbitrary'])
def test_invalid_image_token_selection_creates_no_run(launcher, value):
    run, output, _, query_log = launcher
    result = run('--image-max-token-num', value)
    assert result.returncode != 0
    assert 'must be 256 or 64' in result.stderr
    assert not output.exists() and not query_log.exists()


def test_resume_cannot_override_image_token_selection(launcher, tmp_path):
    run, output, _, _ = launcher
    existing = tmp_path / 'existing-run'
    existing.mkdir()
    config = existing / 'qwen.toml'
    config.write_text('image_max_token_num = 256\n')
    before = config.read_bytes()
    result = run('--resume', str(existing), '--image-max-token-num', '64')
    assert result.returncode != 0
    assert 'existing run settings are retained' in result.stderr
    assert config.read_bytes() == before and not output.exists()


@pytest.mark.parametrize('cli_override', [False, True])
def test_environment_roles_and_cli_override_of_environment_uuid(launcher, cli_override):
    run, output, _, _ = launcher
    env = {'QWEN_INFERENCE_GPU': '0', 'QWEN_SIMULATOR_GPU': '1'}
    args = []
    if cli_override:
        env['QWEN_GPU_UUID'] = GPU_UUIDS[0]
        args = ['--inference-gpu', '1', '--simulator-gpu', '0']
    result = run(*args, overrides=env)
    assert result.returncode == 0, result.stderr
    assert tomllib.loads((output / 'qwen.toml').read_text())['gpu_uuid'] == GPU_UUIDS[int(cli_override)]
    assert json.loads((output / 'runner.json').read_text())['workers'][0]['gpu'] == int(not cli_override)


@pytest.mark.parametrize(('args', 'error'), [
    (['--inference-gpu', '0', '--simulator-gpu', '0'], 'must be distinct'),
    (['--gpu-uuid', GPU_UUIDS[0], '--simulator-gpu', '0'], 'must be distinct'),
    (['--inference-gpu', '2'], 'Physical GPU 2 absent'),
    (['--simulator-gpu', '-1'], 'non-negative physical GPU index'),
    (['--inference-gpu', 'gpu0'], 'non-negative physical GPU index'),
    (['--inference-gpu', '0', '--gpu-uuid', GPU_UUIDS[0]], 'mutually exclusive'),
    (['--simulator-gpu', '1', '--worker', 'local-gpu0'], 'cannot combine'),
])
def test_invalid_gpu_selection_fails_before_creating_run(launcher, args, error):
    run, output, _, _ = launcher
    result = run(*args)
    assert result.returncode != 0
    assert error in result.stderr
    assert not output.exists()


def test_explicit_uuid_and_existing_worker_need_no_local_gpu_query(launcher, tmp_path):
    run, output, template, query_log = launcher
    remote = json.loads(template.read_text())
    remote['workers'] = [{'id': 'remote-gpu5', 'gpu': 5, 'expected_gpu_uuid': 'GPU-5555-cccc'}]
    remote_template = tmp_path / 'remote.json'
    remote_template.write_text(json.dumps(remote))
    result = run('--runner-template', str(remote_template), '--worker', 'remote-gpu5',
                 '--gpu-uuid', GPU_UUIDS[0], overrides={'FIXTURE_GPU_QUERY_FAIL': '1'})
    assert result.returncode == 0, result.stderr
    assert json.loads((output / 'runner.json').read_text())['workers'] == remote['workers']
    assert not query_log.exists()


def test_simulator_index_cannot_retarget_remote_template(launcher, tmp_path):
    run, output, template, query_log = launcher
    remote_template = tmp_path / 'remote.json'
    shutil.copyfile(template, remote_template)
    result = run('--runner-template', str(remote_template), '--simulator-gpu', '1')
    assert result.returncode != 0
    assert 'requires the built-in local runner' in result.stderr
    assert not output.exists()
    assert not query_log.exists()


def test_driver_failure_does_not_write_partial_configuration(launcher):
    run, output, _, _ = launcher
    result = run(overrides={'FIXTURE_GPU_QUERY_FAIL': '1'})
    assert result.returncode != 0
    assert 'Fixture driver unavailable' in result.stderr
    assert not output.exists()
