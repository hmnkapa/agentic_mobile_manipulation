"""Synthetic progress fixtures exercise display only, never policy success."""
from io import StringIO
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from types import SimpleNamespace

from manipulation_agent.console_progress import ConsoleProgress, EventTail


def fixture_batch(tmp_path, *, status='planned'):
    controller = tmp_path / 'spool' / 'controllers' / 'fixture'
    controller.mkdir(parents=True)
    row = {'index': 93, 'task': 'fixture_task', 'run_id': 'fixture', 'instance': 301,
           'seed': 0, 'instruction': 'Fixture task instruction.', 'status': status,
           'stage': 'queued', 'controller_output': str(controller), 'worker_id': 'fixture-gpu0'}
    batch = SimpleNamespace(rows=[row], root=tmp_path, lock=threading.RLock(),
                            manifest={'model': 'fixture-model', 'episode_budget_policy': 'unbounded'},
                            storage=SimpleNamespace(spool=tmp_path / 'spool'), worker_states={})
    stream = StringIO()
    return batch, row, controller, stream, ConsoleProgress(batch, stream=stream)


def append_events(controller, *events):
    with (controller / 'model_events.jsonl').open('a') as stream:
        for event in events:
            stream.write(json.dumps(event) + '\n')


def call(kind, tool, *, arguments=None, body=None, at=None):
    item = {'type': 'mcp_tool_call', 'id': tool, 'tool': tool, 'arguments': arguments or {}}
    if body is not None:
        item['result'] = {'content': [{'type': 'text', 'text': json.dumps(body)},
                                     {'type': 'image', 'data': 'PRIVATE_RGB_BASE64'}]}
    return {'type': kind, 'item': item, 'at': at}


def test_live_calls_read_partial_lines_once_and_only_show_compact_public_details(tmp_path):
    batch, row, controller, stream, console = fixture_batch(tmp_path)
    console.poll()
    row.update(status='running', stage='policy_running')
    append_events(controller, {'type': 'turn.started'},
                  call('item.started', 'task_notes', arguments={'operation': 'write', 'content': 'PRIVATE_NOTE'}),
                  call('item.started', 'act', arguments={'primitive': 'grasp', 'revision': 0,
                       'target': {'image_ref': 'front-current', 'point': [0.25, 0.75]}},
                       at='2026-11-08T09:00:00+00:00'),
                  {'type': 'item.completed', 'item': {'type': 'reasoning', 'text': 'PRIVATE_THINKING'}})
    receipt = call('item.completed', 'act', body={'ok': False, 'error': {'code': 'out_of_reach'},
                   'observation': {'revision': 1}}, at='2026-11-08T09:00:12+00:00')
    path = controller / 'model_events.jsonl'
    with path.open('a') as output:
        output.write(json.dumps(receipt))
    console.poll()
    assert 'Return act' not in stream.getvalue()
    with path.open('a') as output:
        output.write('\n')
    append_events(controller, {'type': 'turn.completed', 'usage': {'input_tokens': 321, 'output_tokens': 42}},
                  call('item.started', 'finish', arguments={'outcome': 'success', 'reason': 'Model claim.'}),
                  call('item.completed', 'finish', body={'ok': True, 'closed': True}))
    original = path.read_bytes()
    console.poll()
    printed = stream.getvalue()
    console.poll()
    assert stream.getvalue() == printed
    assert path.read_bytes() == original
    for expected in ('1/1 fixture_task', 'instance=301 seed=0', 'Turn 1', 'grasp',
                     'front-current', 'act/look requests=1', 'out_of_reach', 'duration=12.0s',
                     'input_tokens=321 output_tokens=42', 'content_characters',
                     'finish acknowledged; task success is determined by independent evaluation'):
        assert expected in printed
    for hidden in ('PRIVATE_NOTE', 'PRIVATE_THINKING', 'PRIVATE_RGB_BASE64', 'Independent evaluation=success'):
        assert hidden not in printed


def test_waiting_heartbeat_and_nested_model_failure(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr('manipulation_agent.console_progress.time.monotonic', lambda: clock[0])
    batch, row, controller, stream, console = fixture_batch(tmp_path)
    batch.worker_states['fixture-gpu0'] = {
        'stage': 'waiting_for_resources', 'checks': {'gpu_capacity': False, 'disk_free': True}}
    console.poll()
    assert 'Waiting for simulator resources gpu_capacity' in stream.getvalue()
    row.update(status='running', stage='simulator_starting',
               startup_last_progress={'stage': 'scene_loading', 'status': 'running'})
    console.poll()
    assert 'scene_loading:running' in stream.getvalue()
    runtime = controller / 'qwen_runtime.json'
    runtime.write_text(json.dumps({'status': 'loading_model', 'gpu_uuid': 'GPU-fixture'}))
    console.poll()
    clock[0] += 31
    console.poll()
    assert 'Still waiting: Model: Loading model weights' in stream.getvalue()
    assert 'time in current state: 31s' in stream.getvalue()
    runtime.write_text(json.dumps({'status': 'failed', 'error': {'type': 'ExceptionGroup',
        'causes': [{'type': 'TrainingError', 'structured': {'code': 'inference_out_of_memory'}}]}}))
    console.poll()
    assert 'inference_out_of_memory' in stream.getvalue()


def test_official_success_is_separate_from_closure_and_late_score_is_reported(tmp_path):
    batch, row, controller, stream, console = fixture_batch(tmp_path)
    row.update(status='failed', stage='complete', task_success=None,
               controller_status='failed', video_validation='failed')
    console.poll()
    assert 'Independent evaluation=unknown (no valid score yet)' in stream.getvalue()
    row.update(task_success=True, q_score=1.0, agent_steps=3, tool_calls=9)
    console.poll()
    output = stream.getvalue()
    assert 'Independent evaluation=success Q=1.0; act/look=3 tool calls=9' in output
    assert 'controller=failed video=failed' in output
    assert 'independent evaluation successes 1' in output
    assert str(tmp_path / 'records' / 'fixture.json') in output


def test_tail_ignores_invalid_records_and_reopens_replaced_or_truncated_file(tmp_path):
    path = tmp_path / 'events.jsonl'
    tail = EventTail()
    assert list(tail.read(path)) == []
    path.write_bytes(b'not json\n[]\n{"type":"first"}\n{"type":')
    assert list(tail.read(path)) == [{'type': 'first'}]
    assert list(tail.read(path)) == []
    with path.open('ab') as stream:
        stream.write(b'"second"}\n')
    assert list(tail.read(path)) == [{'type': 'second'}]
    path.write_text('{}\n')
    assert list(tail.read(path)) == [{}]
    replacement = tmp_path / 'replacement'
    replacement.write_text('{"type":"replacement"}\n')
    replacement.replace(path)
    assert list(tail.read(path)) == [{'type': 'replacement'}]


def test_background_observer_flushes_fast_completion_and_handles_closed_output(tmp_path):
    batch, row, controller, stream, console = fixture_batch(tmp_path)
    console.start()
    try:
        row.update(status='failed', stage='complete', task_success=False, termination_reason='fixture_failure')
    finally:
        console.close()
    assert not console.thread.is_alive()
    assert 'Independent evaluation=failure' in stream.getvalue()
    assert 'fixture_failure' in stream.getvalue()
    stream.close()
    console.write('fixture', 'Closed output must not fail the batch.')
    assert not console.output_open


def test_completed_history_is_not_replayed_and_concurrent_tasks_are_identified(tmp_path):
    batch, row, controller, stream, console = fixture_batch(tmp_path, status='passed')
    for number in (1, 2):
        batch.rows.append({**row, 'run_id': f'active-{number}', 'task': f'active_task_{number}',
                           'status': 'running', 'stage': 'policy_running'})
    console.poll()
    assert 'fixture_task' not in stream.getvalue()
    assert '[2/3 active_task_1]' in stream.getvalue()
    assert '[3/3 active_task_2]' in stream.getvalue()


def test_display_error_retries_without_affecting_task_and_reports_plain_mcp_errors(tmp_path):
    batch, row, controller, stream, console = fixture_batch(tmp_path)
    row.update(status='running', stage='policy_running')
    runtime = controller / 'qwen_runtime.json'
    runtime.write_text('[]')
    console.poll()
    console.poll()
    assert stream.getvalue().count('Read failed; retrying later') == 1
    assert row['status'] == 'running'
    runtime.write_text('{"status":"model_ready"}')
    append_events(controller, {'type': 'item.completed', 'item': {
        'type': 'mcp_tool_call', 'id': 'invalid', 'tool': 'act', 'result': {
            'isError': True, 'content': [{'type': 'text', 'text': 'Invalid revision.'}]}}})
    console.poll()
    assert 'Return act ok=False' in stream.getvalue()
    assert 'Invalid revision.' in stream.getvalue()
    assert console.display_error is None


def test_shell_new_prepare_and_resume_enable_progress_without_changing_saved_settings(tmp_path):
    # A fake supervisor records argv only. No model, GPU, simulator or SSH is used.
    repo = tmp_path / 'checkout'
    scripts = repo / 'scripts'
    scripts.mkdir(parents=True)
    launcher = scripts / 'run_qwen_behavior.sh'
    shutil.copyfile(Path(__file__).resolve().parents[1] / 'scripts' / launcher.name, launcher)
    (scripts / 'run_behavior100.py').write_text(
        'import json, os, pathlib, sys\n'
        'pathlib.Path(os.environ["FIXTURE_SUPERVISOR_ARGV"]).write_text(json.dumps(sys.argv))\n')
    for args in (['init', '-q'], ['add', 'scripts'],
                 ['-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                  '-c', 'commit.gpgsign=false', 'commit', '-qm', 'Synthetic launcher fixture']):
        subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True)
    source = tmp_path / 'source-batch'
    source.mkdir()
    instruction = 'Fixture instruction.'
    manifest = {'tasks': [{'index': 93, 'task': 'fixture_task', 'name': 'Fixture task',
                 'scene': 'fixture_scene', 'instance': 301, 'seed': 0, 'instruction': instruction,
                 'instruction_sha256': hashlib.sha256(instruction.encode()).hexdigest(),
                 'instruction_source': 'synthetic_fixture', 'bddl_sha256': 'fixture'}]}
    (source / 'manifest.json').write_text(json.dumps(manifest))
    template = tmp_path / 'template.json'
    template.write_text(json.dumps({'batch': str(source), 'workers': [{
        'id': 'local-gpu0', 'expected_gpu_uuid': 'GPU-2222-bbbb'}]}))
    model = tmp_path / 'model'
    model.mkdir()
    capture = tmp_path / 'supervisor-argv.json'
    env = {**os.environ, 'FIXTURE_SUPERVISOR_ARGV': str(capture)}
    common = ['bash', str(launcher), '--controller-python', sys.executable,
              '--qwen-python', sys.executable, '--model', str(model), '--gpu-uuid', 'GPU-1111-aaaa',
              '--runner-template', str(template), '--task', 'fixture_task']
    prepared = tmp_path / 'prepared'
    result = subprocess.run([*common, '--output', str(prepared), '--prepare-only'],
                            env=env, check=True, capture_output=True, text=True)
    assert '--console-progress' in result.stdout
    assert not capture.exists()
    before = {path.name: path.read_bytes() for path in prepared.iterdir()}
    subprocess.run(['bash', str(launcher), '--controller-python', sys.executable,
                    '--resume', str(prepared)], env=env, check=True, capture_output=True, text=True)
    assert json.loads(capture.read_text())[-3:] == ['--config', str(prepared / 'runner.json'), '--console-progress']
    assert {path.name: path.read_bytes() for path in prepared.iterdir()} == before
    fresh = tmp_path / 'fresh'
    subprocess.run([*common, '--output', str(fresh)], env=env, check=True, capture_output=True, text=True)
    assert json.loads(capture.read_text())[-3:] == ['--config', str(fresh / 'runner.json'), '--console-progress']
    assert json.loads((fresh / 'manifest.json').read_text())['episode_budget_policy'] == 'unbounded'
