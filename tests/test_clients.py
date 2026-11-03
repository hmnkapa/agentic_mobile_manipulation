import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from manipulation_agent.clients import ClientConfig, get_adapter
from manipulation_agent.clients.base import run_process
from manipulation_agent.clients.events import parse_events, write_derived_events, event_path
from manipulation_agent.clients.types import PreparedProject, ClientCapability
from manipulation_agent.tools import tool_specs
from manipulation_agent.transcript import build_transcript
from manipulation_agent.vision_policy import system_prompt

FIXTURES = Path(__file__).parent / 'fixtures' / 'clients'


class ClientAdapterTests(unittest.TestCase):
    def config(self, root, client='codex', **kwargs):
        return ClientConfig(client, 'provider/model', 'Inspect and finish.', '/usr/bin/python',
                            ['-m', 'manipulation_agent.mcp_server'], root, **kwargs)

    def test_configuration_rejects_non_argv_and_nonpositive_timeout(self):
        with self.assertRaises(ValueError):
            self.config(Path('/unused'), timeout=0)
        with self.assertRaises(ValueError):
            ClientConfig('kimi', 'model', 'task', 'python', 'shell command', Path('/unused'))
        with self.assertRaises(ValueError):
            get_adapter('unknown')

    def test_cli_absence_and_old_cli_are_different_outcomes(self):
        with patch('manipulation_agent.clients.base.shutil.which', return_value=None):
            self.assertEqual(get_adapter('kimi').probe().status, 'unavailable')
        with patch('manipulation_agent.clients.base.shutil.which', return_value='/bin/kimi'), \
             patch('manipulation_agent.clients.base.subprocess.check_output', side_effect=['old', '--model']):
            cap = get_adapter('kimi').probe()
        self.assertEqual(cap.status, 'unsupported')
        self.assertIn('--agent-file', cap.reason)

    def test_codex_preparation_keeps_profile_and_non_policy_tools_disabled(self):
        with tempfile.TemporaryDirectory() as folder:
            prepared = get_adapter('codex').prepare_project(self.config(Path(folder)), 'Task-only policy', ['initialize', 'finish'])
            argv = prepared.argv
            self.assertEqual(argv[:6], ['codex','exec','--ignore-user-config','--skip-git-repo-check','--json','--sandbox'])
            configs = [argv[i+1] for i, v in enumerate(argv[:-1]) if v == '-c']
            self.assertIn('mcp_servers.manipulation.enabled_tools=["initialize", "finish"]', configs)
            self.assertIn('mcp_servers.manipulation.required=true', configs)
            self.assertIn('features.code_mode.direct_only_tool_namespaces=["mcp__manipulation"]', configs)
            self.assertIn('mcp_servers.manipulation.tool_timeout_sec=1020', configs)
            for tool in ['shell_tool', 'view_image', 'multi_agent', 'plugins']:
                self.assertIn(tool, argv)
            self.assertEqual(prepared.env_overlay, {})
            self.assertIsNone(prepared.cwd)
            self.assertTrue(prepared.stdin.endswith('Inspect and finish.'))

    @unittest.skipUnless(shutil.which('bwrap'), 'bubblewrap is required for the mount regression')
    def test_codex_session_symlink_is_isolated_by_actual_mount(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            user_home = root / 'home'
            client_root = user_home / '.codex'
            client_root.mkdir(parents=True)
            shared_sessions = root / 'shared_sessions'
            shared_sessions.mkdir()
            (shared_sessions / 'prior-session').write_text('unchanged')
            (client_root / 'sessions').symlink_to(shared_sessions, target_is_directory=True)
            output = root / 'episode'
            output.mkdir()
            with patch('pathlib.Path.home', return_value=user_home), patch.dict(
                    os.environ, {'MAS_CLIENT_STATE_ROOT': str(root / 'state')}, clear=True):
                prepared = get_adapter('codex').prepare_project(
                    self.config(output, isolate_client_storage=True), 'Policy', ['finish'])
            marker = client_root / 'sessions' / 'new-session'
            process = subprocess.run([*prepared.mcp_prefix, '--', sys.executable, '-c',
                'from pathlib import Path; import sys; Path(sys.argv[1]).write_text("isolated")',
                str(marker)], capture_output=True, text=True, timeout=10)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual((prepared.native_sessions / 'new-session').read_text(), 'isolated')
            self.assertFalse((shared_sessions / 'new-session').exists())
            self.assertEqual((shared_sessions / 'prior-session').read_text(), 'unchanged')

    def test_codex_run_records_explicit_effort_and_only_selected_provider(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'CODEX_HOME': folder}):
            Path(folder, 'config.toml').write_text('''[model_providers.example]
name = "Private endpoint"
base_url = "https://example.invalid/v1"
requires_openai_auth = true
wire_api = "responses"
[mcp_servers.unrelated]
command = "ignored"
''')
            prepared = get_adapter('codex').prepare_project(
                self.config(Path(folder), reasoning_effort='low', model_provider_profile='example'),
                'Task-only policy', ['initialize', 'finish'])
            configs = [prepared.argv[i+1] for i, token in enumerate(prepared.argv[:-1]) if token == '-c']
            self.assertIn('model_reasoning_effort="low"', configs)
            self.assertIn('model_provider="example"', configs)
            self.assertIn('model_providers.example.base_url="https://example.invalid/v1"', configs)
            self.assertNotIn('cli_auth_credentials_store="ephemeral"', configs)
            self.assertFalse(any('unrelated' in value for value in configs))

    def test_api_provider_uses_env_key_without_subscription_fallback(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,
                {'CODEX_HOME':folder,'TEST_RELAY_KEY':'credential-not-for-argv'}):
            Path(folder,'config.toml').write_text('[model_providers.relay]\nname="Relay"\n'
                'base_url="https://example.invalid/v1"\nwire_api="responses"\nenv_key="TEST_RELAY_KEY"\n')
            prepared=get_adapter('codex').prepare_project(
                self.config(Path(folder),model_provider_profile='relay'),'Policy',['finish'])
            self.assertIn('model_providers.relay.env_key="TEST_RELAY_KEY"',prepared.argv)
            self.assertIn('model_providers.relay.requires_openai_auth=false',prepared.argv)
            self.assertIn('cli_auth_credentials_store="ephemeral"',prepared.argv)
            self.assertNotIn('credential-not-for-argv',str(prepared))
            del os.environ['TEST_RELAY_KEY']
            second=Path(folder,'second');second.mkdir()
            with self.assertRaisesRegex(ValueError,'credential environment absent'):
                get_adapter('codex').prepare_project(
                    self.config(second,model_provider_profile='relay'),'Policy',['finish'])

    def test_codex_resume_keeps_model_tool_boundary_and_workspace(self):
        with tempfile.TemporaryDirectory() as folder:
            adapter = get_adapter('codex')
            original = adapter.prepare_project(
                self.config(Path(folder), reasoning_effort='low'),
                'Task-only policy', ['initialize', 'look', 'act', 'finish'])
            thread_id = '01a12027-d1b5-7c40-a929-4936344a9ad5'
            resumed = adapter.prepare_resume(original, thread_id, 'Continue from RGB.')
            self.assertEqual(resumed.argv[:3], ['codex', 'exec', 'resume'])
            self.assertEqual(resumed.argv[-2:], [thread_id, '-'])
            self.assertEqual(resumed.cwd, Path(folder, 'empty_workspace'))
            self.assertEqual(resumed.stdin, 'Continue from RGB.')
            self.assertNotIn('--cd', resumed.argv)
            self.assertNotIn('--sandbox', resumed.argv)
            self.assertIn('sandbox_mode="read-only"', resumed.argv)
            self.assertIn('model_reasoning_effort="low"', resumed.argv)
            self.assertIn('mcp_servers.manipulation.enabled_tools=["initialize", "look", "act", "finish"]',
                          resumed.argv)
            for feature in ('shell_tool', 'plugins', 'multi_agent'):
                self.assertIn(feature, resumed.argv)

    def test_codex_resume_retains_isolated_native_session_mount(self):
        with tempfile.TemporaryDirectory() as folder:
            adapter = get_adapter('codex')
            original = adapter.prepare_project(self.config(Path(folder)), 'Policy', ['act', 'finish'])
            prefix = ['/usr/bin/bwrap', '--bind', '/episode/sessions', '/native/sessions']
            original.argv = [*prefix, '--', *original.argv]
            original.mcp_prefix = prefix
            original.native_sessions = Path('/episode/sessions')
            resumed = adapter.prepare_resume(original, '01a12027-d1b5-7c40-a929-4936344a9ad5', 'Continue.')
            self.assertEqual(resumed.argv[:len(prefix)+4], [*prefix, '--', 'codex', 'exec', 'resume'])
            self.assertEqual(resumed.native_sessions, original.native_sessions)
            self.assertEqual(resumed.mcp_prefix, prefix)
            self.assertEqual(resumed.cwd, Path(folder, 'empty_workspace'))

    def test_provider_gateway_changes_transport_only(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True), \
                patch('pathlib.Path.home', return_value=Path(folder)):
            root = Path(folder, '.codex'); root.mkdir()
            (root/'config.toml').write_text('[model_providers.relay]\nname="Relay"\n'
                'base_url="https://example.invalid/v1"\nwire_api="responses"\nstream_max_retries=8\n')
            prepared = get_adapter('codex').prepare_project(self.config(Path(folder),
                model_provider_profile='relay', model_provider_gateway='http://127.0.0.1:18081'), 'Policy', ['act'])
            self.assertIn('model_providers.relay.base_url="http://127.0.0.1:18081"', prepared.argv)
            self.assertIn('model_providers.relay.stream_max_retries=8', prepared.argv)
            self.assertIn('provider/model', prepared.argv)
            self.assertEqual(prepared.stdin, self.config(Path(folder)).prompt)

    def test_codex_accepts_private_http_provider_without_url_credentials(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'CODEX_HOME': folder}):
            config_file = Path(folder, 'config.toml')
            config_file.write_text('''[model_providers.local]
name = "Internal endpoint"
base_url = "http://10.130.136.133/v1"
requires_openai_auth = true
wire_api = "responses"
''')
            prepared = get_adapter('codex').prepare_project(
                self.config(Path(folder), model_provider_profile='local'), 'Task-only policy', ['finish'])
            self.assertIn('model_providers.local.base_url="http://10.130.136.133/v1"', prepared.argv)
            config_file.write_text(config_file.read_text().replace('/v1"', '/v1?token=secret"'))
            second = Path(folder, 'second')
            second.mkdir()
            with self.assertRaisesRegex(ValueError, 'without embedded credentials'):
                get_adapter('codex').prepare_project(
                    self.config(second, model_provider_profile='local'), 'Task-only policy', ['finish'])

    def test_non_codex_projects_have_explicit_tool_boundary_without_auth_copy(self):
        for client in ['opencode', 'kimi']:
            with self.subTest(client=client), tempfile.TemporaryDirectory() as folder:
                prepared = get_adapter(client).prepare_project(self.config(Path(folder), client), 'Task-only policy', ['initialize','finish'])
                self.assertNotIn('HOME', prepared.env_overlay)
                self.assertNotIn('CODEX_HOME', prepared.env_overlay)
                if client == 'opencode':
                    cfg = json.loads((prepared.cwd/'opencode.json').read_text())
                    self.assertEqual(cfg['permission'], {'*':'deny','manipulation_initialize':'allow','manipulation_finish':'allow'})
                    self.assertEqual(cfg['agent']['manipulation']['prompt'], 'Task-only policy')
                else:
                    agent = (prepared.cwd/'agent.yaml').read_text()
                    self.assertIn('tools: []', agent)
                    self.assertNotIn('extend:', agent)
                    self.assertIn('--mcp-config-file', prepared.argv)
                    self.assertEqual(set(json.loads((prepared.cwd/'mcp.json').read_text())['mcpServers']), {'manipulation'})

    def test_each_client_receives_the_selected_profile_prompt_and_catalog(self):
        for profile in ['minimal', 'skills', 'workflow']:
            instructions = system_prompt(profile)
            names = [tool['name'] for tool in tool_specs(profile)]
            self.assertIn('initialize', names)
            for client in ['codex', 'opencode', 'kimi']:
                with self.subTest(profile=profile, client=client), tempfile.TemporaryDirectory() as folder:
                    prepared = get_adapter(client).prepare_project(
                        self.config(Path(folder), client, agent_profile=profile), instructions, names)
                    if client == 'codex':
                        configs = dict(value.split('=', 1) for index, value in enumerate(prepared.argv)
                                       if index > 0 and prepared.argv[index - 1] == '-c')
                        actual = json.loads(configs['developer_instructions'])
                        self.assertEqual(json.loads(configs['mcp_servers.manipulation.enabled_tools']), names)
                    elif client == 'opencode':
                        config = json.loads((prepared.cwd / 'opencode.json').read_text())
                        actual = config['agent']['manipulation']['prompt']
                        self.assertEqual(config['permission'],
                                         {'*': 'deny', **{'manipulation_' + name: 'allow' for name in names}})
                    else:
                        actual = (prepared.cwd / 'system.md').read_text().removesuffix('\n')
                        self.assertIn('system_prompt_path: ./system.md',
                                      (prepared.cwd / 'agent.yaml').read_text())
                    self.assertEqual(actual, instructions)
                    self.assertIn('First call initialize({})', actual)

    def test_default_prompts_and_skills_do_not_require_hidden_observation_tools(self):
        hidden_tools = r'\b(?:observe|start_observation|get_observation|cancel_observation)\b'
        for profile in ['minimal', 'skills']:
            with self.subTest(profile=profile):
                instructions = system_prompt(profile)
                self.assertNotRegex(instructions, hidden_tools)
                self.assertIn('four views returned by act/look', instructions)
                self.assertIn('No additional observation call is required before finish.', instructions)
                self.assertIn('wait_seconds', instructions)
        skill_root = Path(__file__).resolve().parents[1] / 'skills'
        for resource in sorted(skill_root.glob('*/SKILL.md')) + sorted(skill_root.glob('*/references/*.md')):
            with self.subTest(resource=resource.relative_to(skill_root)):
                self.assertNotRegex(resource.read_text(), hidden_tools)

    def test_recorded_codex_finish_preserved_byte_for_byte(self):
        source = FIXTURES/'codex_recorded.jsonl'
        before = source.read_bytes()
        result = parse_events(source, 'codex')
        self.assertEqual(result.compat, result.raw_events)
        self.assertTrue(result.formal_finish_observed)
        self.assertEqual(source.read_bytes(), before)
        self.assertTrue(any(e['kind'] == 'tool_result' for e in result.canonical))

    def test_formats_link_calls_results_images_and_replay_without_fabricated_summary(self):
        for client in ['opencode','kimi']:
            with self.subTest(client=client):
                result = parse_events(FIXTURES/f'{client}_format.jsonl', client)
                self.assertEqual(result.tools_called, ['observe','finish'])
                self.assertTrue(result.formal_finish_observed)
                self.assertFalse(any(x['kind'] == 'reasoning_summary' for x in result.canonical))
                calls = [x for x in result.canonical if x['kind'] == 'tool_call']
                results = [x for x in result.canonical if x['kind'] == 'tool_result']
                self.assertEqual([x['call_id'] for x in calls], [x['call_id'] for x in results])
                timeline = build_transcript(result.compat, [], [])
                self.assertEqual(sum(x['kind']=='tool_result' for x in timeline), 2)
                if client == 'kimi':
                    self.assertEqual(results[0]['result']['content'][1]['data'], 'RklYVFVSRQ==')
                    self.assertEqual(sum(x.get('image_count',0) for x in timeline), 1)
                with tempfile.TemporaryDirectory() as folder:
                    root = Path(folder); raw = root/'model_events.jsonl'
                    raw.write_bytes((FIXTURES/f'{client}_format.jsonl').read_bytes())
                    before = raw.read_bytes()
                    write_derived_events(root, result, client)
                    self.assertEqual(raw.read_bytes(), before)
                    self.assertEqual(event_path(root).name, 'model_events.compat.jsonl')

    def test_kimi_wire_failure_retains_call_id_and_error(self):
        events = [{'type':'context.append_loop_event','event':{
            'type':'tool.call','toolCallId':'c1','name':'mcp__manipulation__act','args':{'primitive':'wait'}}},
            {'type':'context.append_loop_event','event':{'type':'tool.result','toolCallId':'c1',
             'result':{'output':'{"ok":false}'},'error':{'message':'transport unavailable'}}}]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'raw.jsonl'; path.write_text(''.join(json.dumps(e)+'\n' for e in events))
            result=parse_events(path,'kimi')
        self.assertFalse(result.formal_finish_observed)
        self.assertEqual(result.canonical[-1]['call_id'],'c1')
        self.assertEqual(result.canonical[-1]['error']['message'],'transport unavailable')

    def test_unknown_records_do_not_become_success_and_partial_json_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'raw.jsonl'; raw='{"role":"assistant","content":"success"}\n{"unfinished":'
            path.write_text(raw)
            parsed=parse_events(path,'kimi')
            self.assertFalse(parsed.formal_finish_observed)
            self.assertEqual(parsed.invalid_lines,[2])
            self.assertEqual(path.read_text(),raw)

    def test_actual_subprocess_budget_stops_process_and_keeps_partial_raw(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)
            code='import sys,time; print(\'{"unfinished":\',flush=True); time.sleep(60)'
            project=PreparedProject([sys.executable,'-c',code],None,None)
            result=run_process(project,output,0.2)
            self.assertTrue(result.timed_out)
            self.assertEqual(result.returncode,-signal.SIGTERM)
            self.assertLess(result.duration_seconds,3)
            with self.assertRaises(ProcessLookupError):os.kill(result.pid,0)
            self.assertEqual(parse_events(output/'model_events.jsonl','codex').invalid_lines,[1])

    def test_actual_subprocess_nonzero_exit_remains_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            result=run_process(PreparedProject([sys.executable,'-c','raise SystemExit(7)'],None,None),Path(folder),3)
        self.assertEqual(result.returncode,7)
        self.assertFalse(result.timed_out)

    def test_successful_finish_receipt_reaps_lingering_client_without_task_timeout(self):
        event={'type':'item.completed','item':{'type':'mcp_tool_call','server':'manipulation',
            'tool':'finish','result':{'content':[{'type':'text','text':'{"ok":true,"closed":true}'}]}}}
        raw=json.dumps(event)+'\n'
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)
            code='import sys,time; sys.stdin.read(); print('+repr(raw.rstrip())+',flush=True); time.sleep(60)'
            result=run_process(PreparedProject([sys.executable,'-c',code],None,'task'),output,None,finish_exit_grace=.15)
            self.assertFalse(result.timed_out)
            self.assertTrue(result.finish_received)
            self.assertEqual(result.cleanup_reason,'client_linger_after_successful_finish')
            self.assertEqual(result.returncode,-signal.SIGTERM)
            self.assertLess(result.duration_seconds,3)
            self.assertEqual((output/'model_events.jsonl').read_text(),raw)
            with self.assertRaises(ProcessLookupError):os.kill(result.pid,0)

    def test_failed_requested_and_other_server_finish_do_not_trigger_cleanup(self):
        base={'type':'item.completed','item':{'type':'mcp_tool_call','server':'manipulation',
            'tool':'finish','result':{'content':[{'type':'text','text':'{"ok":true,"closed":true}'}]}}}
        cases=[{**base,'type':'item.started'},
            {**base,'item':{**base['item'],'server':'other'}},
            {**base,'item':{**base['item'],'error':{'message':'failed'}}},
            {**base,'item':{**base['item'],'result':{'content':[{'type':'text','text':'{"ok":false,"closed":true}'}]}}},
            {**base,'item':{**base['item'],'result':{**base['item']['result'],'isError':True}}}]
        for event in cases:
            with self.subTest(event=event),tempfile.TemporaryDirectory() as folder:
                code='import time; print('+repr(json.dumps(event))+',flush=True); time.sleep(.25)'
                result=run_process(PreparedProject([sys.executable,'-c',code],None,None),Path(folder),3,finish_exit_grace=.01)
                self.assertEqual(result.returncode,0)
                self.assertFalse(result.finish_received)
                self.assertIsNone(result.cleanup_reason)
                self.assertFalse(parse_events(Path(folder)/'model_events.jsonl','codex').formal_finish_observed)

    def test_unavailable_client_does_not_start_mcp_or_model(self):
        source=Path(__file__).resolve().parents[1]/'scripts'/'run_codex_controller.py'
        spec=importlib.util.spec_from_file_location('client_controller',source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'run'
            argv=['--client','kimi','--model','fixture','--instruction','task',
                  '--mcp-command','python','--mcp-args-json','[]','--output',str(output)]
            with patch('manipulation_agent.clients.base.shutil.which',return_value=None), \
                 patch.object(module,'check_server') as handshake:
                self.assertEqual(module.main(argv,allow_client=True),3)
            handshake.assert_not_called()
            metadata=json.loads((output/'controller.json').read_text())
            self.assertEqual(metadata['status'],'unavailable')
            self.assertIsNone(metadata['task_success'])

    def test_native_replay_reads_compat_without_external_controller(self):
        from manipulation_agent.replay import render_replay
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'run.json').write_text(json.dumps({'run_id':'native-fixture','status':'passed',
                'config':{'observation_mode':'rgb_only','model':'fixture-model'}}))
            (root/'events.jsonl').write_text('')
            (root/'captures.jsonl').write_text('')
            event={'type':'item.completed','source':'native_loop','item':{
                'type':'agent_message','id':'message-1','text':'Recorded public output'}}
            (root/'model_events.compat.jsonl').write_text(json.dumps(event)+'\n')
            data=render_replay(root)
            self.assertTrue(data['has_public_trace'])
            self.assertEqual(data['model'],'fixture-model')
            self.assertIsNone(data['audit'])
            self.assertEqual(data['model_transcript'][0]['source'],'native_loop')
            self.assertTrue((root/'model_public_events.jsonl').exists())
