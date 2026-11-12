"""Read-only terminal progress from supervisor state and local model evidence."""
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from zoneinfo import ZoneInfo


FINAL = {'passed', 'failed', 'blocked'}
STAGES = {
    'asset_preflight': 'Checking task assets', 'simulator_starting': 'Initializing simulator',
    'controller_starting': 'Starting model controller', 'policy_running': 'Model running',
    'finishing': 'Finalizing and evaluating', 'execution_finished': 'Execution finished; awaiting archival',
    'archiving': 'Archiving evidence', 'cleanup_pending': 'Waiting for process cleanup', 'complete': 'Task complete',
    'starting': 'Starting', 'loading_model': 'Loading model weights', 'model_ready': 'Model ready',
    'closed': 'Model exited', 'admitting': 'Checking resource admission',
    'waiting_for_lease': 'Waiting for simulator resource lock', 'waiting_for_resources': 'Waiting for simulator resources',
    'waiting_for_inference_gpu': 'Waiting for inference GPU lock',
    'waiting_for_inference_memory': 'Waiting for inference GPU memory',
    'waiting_for_initialization': 'Waiting for initialization slot',
    'waiting_for_admission': 'Waiting for admission lock', 'waiting_for_capacity': 'Waiting for execution slot',
    'waiting_for_local_storage': 'Waiting for local storage recovery', 'drained': 'No longer accepting new tasks',
}


def compact(value, limit=600):
    """Keep model-controlled text on one terminal line, without control codes."""
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', str(value))
    text = ' '.join(re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text).split())
    return text if len(text) <= limit else text[:limit] + '…'


def error_summary(error):
    if not isinstance(error, dict):
        return compact(error)
    parts = [str(error[key]) for key in ('code', 'type', 'message') if error.get(key)]
    if error.get('structured'):
        parts.append(error_summary(error['structured']))
    parts.extend(error_summary(child) for child in error.get('causes', []))
    return compact('; '.join(parts))


def event_time(event):
    try:
        return datetime.fromisoformat(event['at']).timestamp()
    except (KeyError, TypeError, ValueError):
        return None


class EventTail:
    """Consume complete JSONL records once, leaving an in-flight line unread."""

    def __init__(self):
        self.offset = 0
        self.identity = None

    def read(self, path):
        try:
            with path.open('rb') as stream:
                stat = os.fstat(stream.fileno())
                identity = (stat.st_dev, stat.st_ino)
                if identity != self.identity or stat.st_size < self.offset:
                    self.offset = 0
                self.identity = identity
                stream.seek(self.offset)
                while True:
                    line = stream.readline()
                    if not line.endswith(b'\n'):
                        break
                    self.offset = stream.tell()
                    try:
                        event = json.loads(line)
                    except (ValueError, UnicodeError):
                        continue
                    if isinstance(event, dict):
                        yield event
        except OSError:
            return  # Display failures must never stop an episode.


@dataclass
class TaskProgress:
    label: str
    tail: EventTail = field(default_factory=EventTail)
    phase: tuple | None = None
    runtime: tuple | None = None
    failure: str | None = None
    turn: int = 0
    calls: int = 0
    steps: int = 0
    activity: str = 'Waiting for task progress'
    last_output: float = field(default_factory=time.monotonic)
    activity_since: float = field(default_factory=time.monotonic)
    pending: dict = field(default_factory=dict)
    final_result: tuple | None = None


class ConsoleProgress:
    """Observe local records without issuing simulator calls or changing policy."""

    def __init__(self, batch, *, stream=None, poll_seconds=1, heartbeat_seconds=30):
        self.batch = batch
        self.stream = sys.stdout if stream is None else stream
        self.poll_seconds = poll_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.tasks = {}
        self.workers = {}
        self.historical = {row['run_id'] for row in batch.rows if row['status'] in FINAL}
        self.stopped = threading.Event()
        self.thread = None
        self.output_open = True
        self.display_error = None

    def write(self, label, message):
        if not self.output_open:
            return
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%H:%M:%S')
        try:
            print(f'[{stamp}] [{compact(label)}] {compact(message, 1200)}',
                  file=self.stream, flush=True)
        except (OSError, ValueError):
            self.output_open = False

    def task_message(self, state, message, *, activity=False):
        self.write(state.label, message)
        state.last_output = time.monotonic()
        if activity:
            state.activity = message
            state.activity_since = state.last_output

    def summary(self):
        with self.batch.lock:
            rows = [dict(row) for row in self.batch.rows]
        self.write('Batch', f"Completed {sum(row['status'] in FINAL for row in rows)}/{len(rows)}; "
                   f"independent evaluation successes {sum(row.get('task_success') is True for row in rows)}; "
                   f"running {sum(row['status'] == 'running' for row in rows)}")

    def start(self):
        self.write('Batch', f"Model={self.batch.manifest.get('model')}; "
                   f"episode policy={self.batch.manifest.get('episode_budget_policy')}")
        self.write('Batch', f'Run directory={self.batch.root}')
        self.write('Batch', f'Live controller log directory={self.batch.storage.spool}/controllers')
        self.summary()
        self.poll()
        self.thread = threading.Thread(target=self._run, name='console-progress', daemon=True)
        self.thread.start()

    def close(self):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.thread is None or not self.thread.is_alive():
            self.poll()  # Include fast failures and the final tool receipt.
        self.summary()

    def _run(self):
        while not self.stopped.wait(self.poll_seconds):
            self.poll()

    def poll(self):
        try:
            self._poll()
        except Exception as exc:
            # Incomplete/unavailable display evidence cannot alter task execution or cleanup.
            error = (type(exc).__name__, str(exc))
            if error != self.display_error:
                self.write('Progress display', f'Read failed; retrying later: {error[0]} {compact(error[1])}')
            self.display_error = error
        else:
            self.display_error = None

    def _poll(self):
        with self.batch.lock:
            rows = [dict(row) for row in self.batch.rows]
            workers = {key: dict(value) for key, value in self.batch.worker_states.items()}
        for position, row in enumerate(rows, 1):
            if row['status'] == 'planned' or row['run_id'] in self.historical:
                continue
            state = self.tasks.get(row['run_id'])
            if state is None:
                state = self.tasks[row['run_id']] = TaskProgress(f"{position}/{len(rows)} {row['task']}")
                self.task_message(state, f"Task instance={row.get('instance')} seed={row.get('seed')} "
                                  f"attempt={row.get('attempt', 1)} worker={row.get('worker_id')}")
                self.task_message(state, f"Instruction: {compact(row.get('instruction', ''), 800)}")
            final_result = tuple(row.get(key) for key in (
                'status', 'stage', 'task_success', 'q_score', 'agent_steps', 'tool_calls',
                'controller_finish_acknowledged', 'controller_status', 'video_validation',
                'termination_reason', 'failure', 'finish_failure', 'archive_failure'))
            if row['status'] in FINAL and final_result == state.final_result:
                continue
            self.task_progress(row, state)
            if row['status'] in FINAL:
                state.final_result = final_result
        for worker_id, worker in workers.items():
            # Task lines already identify active workers; report admission and holds separately.
            stage = worker.get('stage', '')
            if stage in {'running', 'starting', 'controller_starting', 'available'}:
                continue
            checks = ','.join(key for key, value in worker.get('checks', {}).items() if value is False)
            details = compact(worker.get('reason') or checks or worker.get('previous_unit') or '')
            key = (stage, details)
            previous, last = self.workers.get(worker_id, (None, 0))
            current = time.monotonic()
            if key != previous or current - last >= self.heartbeat_seconds:
                self.write(f'worker {worker_id}', f'{STAGES.get(stage, stage)} {details}')
                self.workers[worker_id] = (key, current)

    def task_progress(self, row, state):
        milestone = row.get('startup_last_progress') or {}
        phase = (row.get('status'), row.get('stage'), milestone.get('stage'), milestone.get('status'))
        if phase != state.phase:
            state.phase = phase
            stage = row.get('stage', row['status'])
            message = f'Stage: {STAGES.get(stage, stage)} ({stage})'
            if stage in {'simulator_starting', 'controller_starting'} and milestone:
                message += f"; startup progress={milestone.get('stage')}:{milestone.get('status')}"
            self.task_message(state, message, activity=True)
        controller = row.get('controller_output')
        if controller:
            controller = Path(controller)
            try:
                runtime = json.loads((controller / 'qwen_runtime.json').read_text())
            except (OSError, ValueError):
                runtime = {}
            status = (runtime.get('status'), error_summary(runtime['error']) if runtime.get('error') else '')
            if status[0] and status != state.runtime:
                state.runtime = status
                self.task_message(state, f"Model: {STAGES.get(status[0], status[0])}; "
                                  f"GPU={runtime.get('gpu_uuid')} {status[1]}", activity=True)
            for event in state.tail.read(controller / 'model_events.jsonl'):
                self.model_event(state, event)
        failure = row.get('failure') or row.get('finish_failure') or row.get('archive_failure')
        if failure and failure != state.failure:
            state.failure = failure
            self.task_message(state, f'Error: {compact(failure)}')
        if row['status'] in FINAL:
            success = row.get('task_success')
            official = 'success' if success is True else 'failure' if success is False else 'unknown (no valid score yet)'
            self.task_message(state, f"Independent evaluation={official} Q={row.get('q_score')}; "
                              f"act/look={row.get('agent_steps', state.steps)} "
                              f"tool calls={row.get('tool_calls', state.calls)}; "
                              f"finish={row.get('controller_finish_acknowledged')} "
                              f"controller={row.get('controller_status')} video={row.get('video_validation')}; "
                              f"reason={row.get('termination_reason')}")
            self.task_message(state, f"Record={self.batch.root}/records/{row['run_id']}.json")
            self.summary()
        elif time.monotonic() - state.last_output >= self.heartbeat_seconds:
            elapsed = int(time.monotonic() - state.activity_since)
            self.task_message(state, f'Still waiting: {state.activity}; time in current state: {elapsed}s; '
                              f'act/look requests={state.steps} tool calls={state.calls}')

    def model_event(self, state, event):
        kind = event.get('type')
        if kind == 'turn.started':
            state.turn += 1
            self.task_message(state, f'Turn {state.turn}: generating model response', activity=True)
        elif kind == 'turn.completed':
            usage = event.get('usage') or {}
            self.task_message(state, f"Turn {state.turn} generation complete; input_tokens={usage.get('input_tokens')} "
                              f"output_tokens={usage.get('output_tokens')} status={event.get('status', 'completed')}",
                              activity=True)
        elif kind == 'generation.rejected':
            self.task_message(state, f"Invalid model response format; no tools executed; waiting for regeneration: {error_summary(event.get('error'))}")
        elif kind in {'turn.failed', 'error'}:
            self.task_message(state, f"Model error: {error_summary(event.get('error'))}", activity=True)
        item = event.get('item') or {}
        if kind == 'item.completed' and item.get('type') == 'agent_message':
            self.task_message(state, f"Model message: {compact(item.get('text', ''))}")
        if item.get('type') != 'mcp_tool_call':
            return
        tool, call_id = item.get('tool'), item.get('id')
        if kind == 'item.started':
            state.calls += 1
            state.steps += tool in {'act', 'look'}
            state.pending[call_id] = (time.monotonic(), event_time(event))
            # Omit note bodies and arbitrary fields; never dump RGB, prompts or private reasoning.
            args = item.get('arguments') or {}
            if not isinstance(args, dict):
                args = {}
            display = {key: args[key] for key in ('primitive', 'revision', 'target', 'yaw_degrees',
                       'pitch_degrees', 'placement_yaw_degrees', 'wait_seconds', 'outcome', 'reason',
                       'name', 'operation', 'expected_version') if key in args}
            if isinstance(args.get('content'), str):
                display['content_characters'] = len(args['content'])
            self.task_message(state, f'Call #{state.calls} {tool} '
                              f'{json.dumps(display, ensure_ascii=False)}; act/look requests={state.steps}', activity=True)
        elif kind == 'item.completed':
            result = item.get('result') or {}
            body = {}
            for block in result.get('content', []):
                if block.get('type') != 'text':
                    continue
                try:
                    value = json.loads(block.get('text', ''))
                except (ValueError, TypeError):
                    continue
                if isinstance(value, dict) and ('ok' in value or 'error' in value):
                    body = value
                    break
            error = item.get('error') or body.get('error')
            if not error and result.get('isError'):
                error = ' '.join(block.get('text', '') for block in result.get('content', [])
                                 if block.get('type') == 'text')
            ok = body.get('ok')
            if error or result.get('isError'):
                ok = False
            began = state.pending.pop(call_id, None)
            finished = event_time(event)
            if began is not None and began[1] is not None and finished is not None:
                duration = f'{max(0, finished - began[1]):.1f}s'
            else:
                duration = f'≈{time.monotonic() - began[0]:.1f}s' if began is not None else 'unknown'
            revision = (body.get('observation') or {}).get('revision')
            message = f'Return {tool} ok={ok} revision={revision} duration={duration}'
            if error:
                message += f'; {error_summary(error)}'
            if tool == 'finish' and ok is True and body.get('closed') is True:
                message += '; finish acknowledged; task success is determined by independent evaluation'
            self.task_message(state, message, activity=True)
