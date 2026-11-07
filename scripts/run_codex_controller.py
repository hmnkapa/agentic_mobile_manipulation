"""Run the installed model client against a project MCP endpoint using its existing login.

This is a policy under evaluation, not a coding worker. Shell, plugins, other agents,
web, browser and filesystem image tools are disabled. No authentication file is copied.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import shutil
import signal
import sys
import itertools
import re
from dataclasses import asdict

from manipulation_agent.clients import ClientConfig, get_adapter
from manipulation_agent.clients.events import write_derived_events

from manipulation_agent.tools import tool_specs
from manipulation_agent.vision_policy import system_prompt
from manipulation_agent.records import now, write_json, source_version
from manipulation_agent.model_trace import export_summaries
from manipulation_agent.mcp_preflight import check_server
from manipulation_agent.deadline import EpisodeDeadline, validate_execution_clock


def rate_limit_error(events):
    """Only a terminal transient provider failure permits same-session recovery."""
    terminal = next((event for event in reversed(events) if event.get('type') == 'turn.failed'), None)
    if terminal is None:
        return False
    message = str(terminal.get('message') or terminal.get('error') or '')
    return (bool(re.search(r'\b(?:429 Too Many Requests|(?:status|HTTP) (?:429|500|502|503|504)\b)', message))
            or 'API_KEY_CONCURRENCY_QUEUE_' in message
            or 'stream disconnected before completion' in message)


def codex_thread_id(events):
    return next((event.get('thread_id') for event in events
                 if event.get('type') == 'thread.started' and event.get('thread_id')), None)


def main(argv=None, *, allow_client=False):
    argv = list(sys.argv[1:] if argv is None else argv)
    probing = allow_client and '--probe' in argv
    p = argparse.ArgumentParser()
    if allow_client:
        p.add_argument('--client', choices=['codex', 'opencode', 'kimi', 'qwen'], default='codex')
        p.add_argument('--probe', action='store_true', help='Report installed CLI capabilities; does not call a model')
    p.add_argument("--model")
    p.add_argument('--qwen-config', type=Path, help='Local native-interface Qwen model locations and GPU')
    p.add_argument("--instruction", required=not probing)
    p.add_argument("--mcp-command", required=not probing)
    p.add_argument("--mcp-args-json", required=not probing)
    p.add_argument("--output", type=Path, required=not probing)
    p.add_argument("--timeout", type=int, default=None,help='Controller wall timeout; default 900, Qwen defaults to unbounded')
    timing=p.add_mutually_exclusive_group()
    timing.add_argument('--deadline-unix', type=float,
                        help='Legacy absolute deadline; preserved for existing run configurations')
    timing.add_argument('--execution-clock-command-json',
                        help='Private argv JSON: atomically arm remote execution clock after MCP handshake')
    p.add_argument('--isolate-client-storage', action='store_true',
                   help='Use per-run native sessions and logs without changing HOME, CODEX_HOME or authentication')
    p.add_argument('--agent-profile', choices=['minimal','skills','workflow'], default='skills')
    p.add_argument('--goal-skill',choices=['hierarchical','none'],default='hierarchical')
    p.add_argument('--reasoning-effort', choices=['low','medium','high','xhigh'])
    p.add_argument('--model-provider-profile', help='Use only this provider from the existing Codex config')
    p.add_argument('--model-provider-gateway', help='Optional loopback request queue for this provider')
    p.add_argument('--rate-limit-resumes', '--provider-error-resumes', type=int, default=None,
                   help='Same-session transient provider recovery count; -1 is unbounded (default for unbounded Codex episodes)')
    p.add_argument('--resume-backoff-seconds', type=int, default=90)
    args = p.parse_args(argv)
    client = getattr(args, 'client', 'codex')
    if args.model is None:args.model='Qwen/Qwen3.5-4B' if client=='qwen' else 'gpt-6.1-sol'
    if args.timeout is None:args.timeout=0 if client=='qwen' else 900
    if client == 'codex' and args.reasoning_effort is None:args.reasoning_effort='low'
    adapter = get_adapter(client)
    if probing:
        capability = adapter.probe()
        print(json.dumps(asdict(capability)))
        return 0 if capability.status == 'available' else 3
    deadline=(EpisodeDeadline() if args.execution_clock_command_json is not None else
              EpisodeDeadline(args.deadline_unix) if args.deadline_unix is not None else EpisodeDeadline.from_env())
    if args.timeout<0:p.error('--timeout must be nonnegative')
    if args.timeout==0:args.timeout=None
    if args.rate_limit_resumes is None:
        args.rate_limit_resumes = -1 if client == 'codex' and args.timeout is None else 0
    if args.rate_limit_resumes < -1 or args.resume_backoff_seconds < 1:
        p.error('Provider resume count must be -1 or nonnegative and backoff positive')
    if args.rate_limit_resumes and client != 'codex':
        p.error('Rate-limit resume is supported only for Codex')
    if client=='qwen':
        from manipulation_agent.clients.qwen.config import load_config
        if args.qwen_config is None:p.error('--client qwen requires --qwen-config')
        if args.timeout is not None or deadline.unix is not None:p.error('Qwen requires an unbounded episode')
        runtime=load_config(args.qwen_config)
        target=args.output.resolve()
        for protected in (runtime['model'],runtime['adapter']):
            if protected and (target==Path(protected) or Path(protected) in target.parents):
                p.error('Controller output must be outside model and adapter directories')
    elif args.qwen_config is not None:p.error('--qwen-config applies only to Qwen')
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(Path(__file__).resolve().parents[1] / "src", args.output / "source_snapshot",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    config = ClientConfig(client, args.model, args.instruction, args.mcp_command,
                          json.loads(args.mcp_args_json), args.output, args.timeout,
                          args.agent_profile, args.isolate_client_storage,
                          args.reasoning_effort, args.model_provider_profile, args.model_provider_gateway,
                          args.qwen_config)
    capability = adapter.probe()
    if capability.status != 'available':
        write_json(args.output / 'controller.json', {
            'started_at': now(), 'finished_at': now(), 'status': capability.status,
            'controller': client + '_cli', 'client_capability': asdict(capability),
            'model': args.model, 'model_reasoning_effort': args.reasoning_effort,
            'agent_profile': args.agent_profile, 'exit_code': 3,
            'formal_finish_observed': False, 'tools_called': [], 'task_success': None})
        print(json.dumps(asdict(capability)))
        return 3
    try:
        project = adapter.prepare_project(config, system_prompt(args.agent_profile,goal_skill=args.goal_skill),
                                          [t['name'] for t in tool_specs(args.agent_profile)])
    except (ValueError, RuntimeError) as exc:
        write_json(args.output / 'controller.json', {
            'started_at': now(), 'finished_at': now(), 'status': 'unsupported',
            'controller': client + '_cli', 'client_capability': asdict(capability),
            'model': args.model, 'model_reasoning_effort': args.reasoning_effort,
            'agent_profile': args.agent_profile, 'exit_code': 3,
            'failure_stage': 'prepare_project', 'failure_type': type(exc).__name__, 'reason': str(exc),
            'formal_finish_observed': False, 'tools_called': [], 'task_success': None})
        return 3
    command = project.argv
    native_sessions = project.native_sessions
    metadata = {"started_at": now(), "status": "running", "model": args.model,
                "model_reasoning_effort": args.reasoning_effort, "observation_mode": "rgb_only",
                "episode_deadline_unix":deadline.unix,"requested_timeout_seconds":args.timeout,
                "agent_profile": args.agent_profile, "goal_skill": args.goal_skill,
                "host": os.uname().nodename, "controller": client + "_cli", "command": command,
                "instruction": args.instruction, "source": source_version(),
                "client_version": capability.version, "client_capability": asdict(capability)}
    if client=='qwen':
        request=json.loads((args.output/'qwen_request.json').read_text())
        metadata.update(qwen_config=request['runtime'],episode_budget_policy='unbounded',
                        model_interface='native_mcp_point_0_1')
    if project.client_storage is not None:
        metadata['client_storage'] = project.client_storage
    metadata['provider_recovery'] = dict(max_resumes=args.rate_limit_resumes,
        backoff_seconds=args.resume_backoff_seconds, gateway=args.model_provider_gateway,
        scope='Same model thread and live MCP episode; terminal transient provider errors only')
    write_json(args.output / "controller.json", metadata)
    def expired_before_policy(stage):
        metadata.update(status='failed',timeout=True,termination_reason='episode_deadline_exceeded',
                        failure_stage=stage,finished_at=now(),formal_finish_observed=False,
                        tools_called=[],task_success=None,exit_code=124,effective_timeout_seconds=0)
        write_json(args.output/'controller.json',metadata)
        return 124
    if deadline.expired:return expired_before_policy('before_mcp_handshake')
    try:
        mcp_command=args.mcp_command; mcp_args=json.loads(args.mcp_args_json)
        if native_sessions is not None:
            # Validate the same namespace the model will use, including SSH's
            # access to /dev/null. A host-only handshake misses mount failures.
            prefix=project.mcp_prefix
            mcp_command=prefix[0];mcp_args=prefix[1:]+['--',args.mcp_command,*mcp_args]
        metadata['mcp_preflight']=check_server(mcp_command,mcp_args,tool_specs(args.agent_profile),args.output,
                                               timeout=deadline.remaining(90))
        metadata['mcp_preflight']['inside_client_storage_namespace']=native_sessions is not None
    except Exception as exc:
        if deadline.expired:return expired_before_policy('mcp_handshake')
        metadata.update(status='failed',failure_stage='mcp_handshake',failure_type=type(exc).__name__,
                        finished_at=now(),formal_finish_observed=False,tools_called=[],task_success=None)
        write_json(args.output/'controller.json',metadata)
        return 2
    if deadline.expired:return expired_before_policy('before_model_start')
    if args.execution_clock_command_json is not None:
        try:
            clock_command=json.loads(args.execution_clock_command_json)
            if not isinstance(clock_command,list) or not clock_command or any(not isinstance(v,str) or not v for v in clock_command):
                raise ValueError('Execution clock command must be a nonempty argv array')
            armed=subprocess.run(clock_command,capture_output=True,text=True,timeout=30,check=True)
            clock=validate_execution_clock(json.loads(armed.stdout))
            if client=='qwen' and clock['episode_deadline_unix'] is not None:
                raise ValueError('Qwen requires an unbounded execution clock')
            deadline=EpisodeDeadline(clock['episode_deadline_unix'])
            metadata.update(execution_clock=clock,execution_started_at_unix=clock['execution_started_at_unix'],
                episode_deadline_unix=clock['episode_deadline_unix'],execution_budget_seconds=clock['execution_budget_seconds'],
                timing_origin='model_execution_after_mcp_preflight',startup_excluded_from_execution_budget=True)
        except Exception as exc:
            metadata.update(status='failed',failure_stage='execution_clock',failure_type=type(exc).__name__,
                            finished_at=now(),formal_finish_observed=False,tools_called=[],task_success=None)
            write_json(args.output/'controller.json',metadata)
            return 2
        if deadline.expired:return expired_before_policy('execution_clock_already_expired')
    metadata['effective_timeout_seconds']=deadline.remaining(args.timeout)
    metadata['policy_started_at']=now();write_json(args.output/'controller.json',metadata)
    def on_start(pid):
        metadata['pid'] = pid
        write_json(args.output / 'controller.json', metadata)
    policy_started_monotonic=time.monotonic()
    process = adapter.run(project, args.output, deadline.remaining(args.timeout), on_start)
    raw_events=args.output/'model_events.jsonl'
    first_events=adapter.parse(raw_events).raw_events
    thread_id=codex_thread_id(first_events) if client == 'codex' else None
    segment_events=first_events
    continuations=[]
    for attempt in itertools.count(1):
        if args.rate_limit_resumes != -1 and attempt > args.rate_limit_resumes:
            break
        remaining=(None if metadata['effective_timeout_seconds'] is None else
                   metadata['effective_timeout_seconds']-(time.monotonic()-policy_started_monotonic))
        if (process.returncode == 0 or process.finish_received or process.timed_out or not thread_id or
                not rate_limit_error(segment_events) or
                (remaining is not None and remaining <= args.resume_backoff_seconds+30)):
            break
        metadata['provider_recovery'].update(state='waiting', next_attempt=attempt)
        write_json(args.output/'controller.json',metadata)
        time.sleep(args.resume_backoff_seconds)
        remaining=(None if metadata['effective_timeout_seconds'] is None else
                   metadata['effective_timeout_seconds']-(time.monotonic()-policy_started_monotonic))
        if remaining is not None and remaining <= 30:
            break
        segment_dir=args.output/'continuations'/str(attempt)
        segment_dir.mkdir(parents=True,exist_ok=False)
        resume_prompt=('Continue the same robot task in the same simulator episode. '
                       'A temporary model service error interrupted the previous turn. '
                       'Use the prior conversation and current RGB tool feedback; continue choosing '
                       'actions yourself, then call finish. Do not restart the task or claim success '
                       'without visual evidence.')
        resume_project=adapter.prepare_resume(project,thread_id,resume_prompt)
        resumed=adapter.run(resume_project,segment_dir,remaining,on_start)
        segment_path=segment_dir/'model_events.jsonl'
        segment_events=adapter.parse(segment_path).raw_events
        with raw_events.open('ab') as combined,segment_path.open('rb') as segment:
            combined.write(segment.read())
        continuations.append({'attempt':attempt,'thread_id':thread_id,
                              'returncode':resumed.returncode,'timed_out':resumed.timed_out,
                              'raw_events':str(segment_path.relative_to(args.output))})
        metadata['provider_recovery'].update(state='resumed', attempts=continuations)
        write_json(args.output/'controller.json',metadata)
        process.returncode=resumed.returncode
        process.timed_out=process.timed_out or resumed.timed_out
        process.policy_finished_at_unix=resumed.policy_finished_at_unix
        process.finish_received=resumed.finish_received
        process.cleanup_reason=resumed.cleanup_reason
        process.duration_seconds=time.monotonic()-policy_started_monotonic
    if continuations:
        metadata['rate_limit_continuations']=continuations
    policy_finished_at = process.policy_finished_at_unix
    if process.timed_out:
        metadata['timeout'] = True
        metadata['termination_reason'] = ('episode_deadline_exceeded' if deadline.unix is not None
            and policy_finished_at >= deadline.unix else 'controller_timeout')
    metadata['policy_finished_at_unix'] = policy_finished_at
    parsed = adapter.parse(args.output / 'model_events.jsonl')
    write_derived_events(args.output, parsed, client)
    events = parsed.raw_events
    if parsed.invalid_lines:
        metadata['event_decode_errors'] = {'line_numbers': parsed.invalid_lines, 'raw_preserved': True}
    closed = parsed.formal_finish_observed
    calls = parsed.tools_called
    metadata['event_records'] = {'raw': 'model_events.jsonl', 'canonical': 'canonical_events.jsonl',
        'replay': 'model_events.jsonl' if client == 'codex' else 'model_events.compat.jsonl'}
    if deadline.unix is not None and policy_finished_at>=deadline.unix:
        metadata.update(timeout=True,termination_reason='episode_deadline_exceeded')
    logical_complete = closed and not metadata.get('timeout') and (process.returncode == 0 or process.finish_received)
    metadata.update(status="passed" if logical_complete else "failed", exit_code=process.returncode,
                    client_cleanup_reason=process.cleanup_reason,
                    client_cleanup_status='forced_after_finish' if process.cleanup_reason else 'normal_exit',
                    duration_seconds=process.duration_seconds, finished_at=now(),
                    task_success=None, formal_finish_observed=closed, tools_called=calls,
                    validation_note="Client exit alone does not prove task success; join with simulator run.json")
    try:
        metadata['reasoning_trace']=(export_summaries(args.output,events,metadata['started_at'],metadata['finished_at'],sessions_root=native_sessions) if client == 'codex' else {'status': 'not_collected', 'reason': 'No run-owned provider summary export; raw events preserved'})
    except Exception as exc:
        metadata['reasoning_trace']={'status':'failed','error_type':type(exc).__name__,
                                     'reason':'Run-local summary export failed; raw controller events preserved'}
    write_json(args.output / "controller.json", metadata)
    print(json.dumps({k: metadata[k] for k in ("status", "exit_code", "model", "duration_seconds")}))
    return 0 if logical_complete else process.returncode or (124 if metadata.get('timeout') else 2)


if __name__ == "__main__":
    raise SystemExit(main())
