"""One local model and one real MCP session for an existing prepared episode."""
import argparse
import asyncio
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time
import uuid

from ...clients.events import successful_finish_event
from ...records import now, write_json, source_version
from ...training.artifacts import fingerprint, model_manifest
from ...sft.common import ConversionError, file_hash
from ...sft.protocol import build_student_context, from_student_arguments
from .protocol import (MCP_INTERFACE, MODEL_INTERFACE, parse_response,
                       protocol_error_message, receipt_message)
from .resources import process_start


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def emit(kind, **values):
    event = {'type': kind, 'at': now(), 'source': 'qwen_native_model_client', **values}
    print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)
    return event


def failure_record(exc):
    record={'type':type(exc).__name__,'message':str(exc)}
    if hasattr(exc,'record'):record['structured']=exc.record()
    if isinstance(exc,BaseExceptionGroup):
        record['causes']=[failure_record(child) for child in exc.exceptions]
    return record


async def run(request, output, backend):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from ...deadline import tool_wait_seconds

    system, tools = build_student_context(request['instructions'], request['tools'])
    model_catalog = [{'name': tool['function']['name'], 'inputSchema': tool['function']['parameters']}
                     for tool in tools]
    messages = [{'role': 'system', 'content': system},
                {'role': 'user', 'content': request['prompt']}]
    parameters = StdioServerParameters(command=request['mcp_command'], args=request['mcp_args'],
                                       env=dict(os.environ))
    emit('thread.started', thread_id=str(uuid.uuid4()))
    async with stdio_client(parameters, errlog=sys.stderr) as (reader, writer):
        async with ClientSession(reader, writer,
                                 read_timeout_seconds=timedelta(seconds=tool_wait_seconds(None))) as session:
            await session.initialize()
            found = await session.list_tools()
            catalog = [tool.model_dump(by_alias=True, exclude_none=True) for tool in found.tools]
            wanted = {t['name']: (t['description'], t['inputSchema']) for t in request['tools']}
            actual = {t['name']: (t.get('description', ''), t['inputSchema']) for t in catalog}
            if wanted != actual:
                raise RuntimeError('Live MCP catalog differs from the original controller catalog')
            write_json(output / 'qwen_inputs.json', {'instructions': system,
                       'prompt': request['prompt'], 'tools': tools,
                       'interface': MODEL_INTERFACE, 'mcp_interface': MCP_INTERFACE,
                       'native_inputs': {'instructions': request['instructions'], 'tools': request['tools']},
                       'model_input_sha256': fingerprint([system, request['prompt'], tools]),
                       'project_input_sha256': fingerprint([request['instructions'], request['prompt'], request['tools']])})
            turn = 0
            consecutive_failures = 0
            while True:
                turn += 1
                emit('turn.started')
                started = time.monotonic()
                generation = await asyncio.to_thread(backend.generate, messages, tools)
                append(output / 'generations.jsonl', {'turn': turn, 'at': now(),
                       'duration_seconds': time.monotonic() - started,
                       'messages_sha256': fingerprint(messages), **generation})
                if generation['finish_reason'] != 'eos':
                    raise RuntimeError('context_exhausted_during_generation')
                usage = {'input_tokens': generation['input_tokens'],
                         'output_tokens': generation['generated_tokens']}
                # Parse the entire response before publishing or dispatching any
                # call. Only pre-dispatch protocol errors may be regenerated.
                try:
                    public, calls = parse_response(generation['raw_response'], model_catalog,
                        thinking_prefix=generation.get('generation_prefix_has_think',False))
                    if not calls:
                        raise ConversionError('missing_tool_call',
                                              'Response ended without a tool call or formal finish')
                    # Convert every call before dispatch, so an invalid point in
                    # a later call cannot leave a partially executed response.
                    native_arguments = []
                    for call in calls:
                        try:
                            native_arguments.append(from_student_arguments(call['name'], call['arguments']))
                        except ConversionError as exc:
                            raise ConversionError(exc.code, str(exc), tool=call['name'], parameter='target') from exc
                except ConversionError as exc:
                    consecutive_failures += 1
                    feedback = protocol_error_message(exc)
                    messages.append(feedback)
                    emit('generation.rejected', turn=turn, consecutive_failures=consecutive_failures,
                         error=exc.record(), feedback=feedback, dispatched_tool_calls=0)
                    emit('turn.completed', status='retrying', usage=usage)
                    continue
                consecutive_failures = 0
                if public.strip():
                    emit('item.completed', item={'type': 'agent_message', 'id': str(uuid.uuid4()), 'text': public})
                assistant = {'role': 'assistant', 'content': public, 'tool_calls': []}
                for call in calls:
                    call['id'] = str(uuid.uuid4())
                    assistant['tool_calls'].append({'id': call['id'], 'type': 'function',
                        'function': {'name': call['name'], 'arguments': deepcopy(call['arguments'])}})
                messages.append(assistant)
                for call, arguments in zip(calls, native_arguments):
                    item = {'type': 'mcp_tool_call', 'id': call['id'], 'server': 'manipulation',
                            'tool': call['name'], 'arguments': arguments,
                            'model_arguments': deepcopy(call['arguments'])}
                    emit('item.started', item=item)
                    try:
                        receipt = await session.call_tool(call['name'], arguments)
                    except Exception as exc:
                        emit('item.completed', item={**item, 'error': {'code': 'execution_uncertain',
                             'error_type': type(exc).__name__}, 'result': None})
                        raise RuntimeError('execution_uncertain; no automatic action retry') from exc
                    result = receipt.model_dump(by_alias=True, exclude_none=True)
                    event = emit('item.completed', item={**item, 'result': result})
                    if successful_finish_event(event):
                        emit('turn.completed', usage=usage)
                        return
                    message, images = receipt_message(result, call['name'], call['id'], output)
                    messages.append(message)
                    append(output / 'received_images.jsonl', {'turn': turn, 'call_id': call['id'], 'images': images})
                emit('turn.completed', usage=usage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    output, runtime = args.request.parent, request['runtime']
    state = {'status': 'starting', 'host': os.uname().nodename, 'pid': os.getpid(),
             'pid_start': process_start(os.getpid()),
             'python': sys.executable, 'gpu_uuid': runtime['gpu_uuid'], 'started_at': now(),
             'validation_level': 'model_startup', 'training_contract_applied': False,
             'interface': MODEL_INTERFACE, 'mcp_interface': MCP_INTERFACE}
    state.update(source=source_version(),configuration=runtime,versions=request['versions'])
    backend = None
    write_json(output / 'qwen_runtime.json', state)
    try:
        base = model_manifest(runtime['model'], require_weights=True)
        adapter = None
        if runtime['adapter']:
            root = Path(runtime['adapter'])
            adapter_config = json.loads((root / 'adapter_config.json').read_text())
            if adapter_config.get('peft_type') != 'LORA':
                raise ValueError('Expected PEFT LoRA weights')
            adapter = {'root': str(root), 'files': {name: file_hash(root / name) for name in
                       ('adapter_config.json', 'adapter_model.safetensors')}}
            marker = root / 'manipulation_training.json'
            if marker.is_file():
                adapter['training_marker'] = json.loads(marker.read_text())
            adapter['training_interface_policy'] = 'record_only_shared_sft_coordinate_protocol'
        state.update(status='loading_model', base=base, adapter=adapter)
        write_json(output / 'qwen_runtime.json', state)
        from ...training.rollout_backend import NativeGenerationBackend
        backend = NativeGenerationBackend(runtime['model'], runtime['adapter'], runtime['device'],
                                          image_max_token_num=runtime.get('image_max_token_num'))
        state.update(status='model_ready', model_ready_at=now(), backend=backend.describe())
        write_json(output / 'qwen_runtime.json', state)
        asyncio.run(run(request, output, backend))
        state.update(status='closed', validation_level='real_model_mcp_episode_closed')
        return 0
    except Exception as exc:
        error = failure_record(exc)
        state.update(status='failed', error=error)
        emit('turn.failed', error=error)
        return 2
    finally:
        state['finished_at'] = now()
        write_json(output / 'qwen_runtime.json', state)
        if backend is not None:
            backend.close()


if __name__ == '__main__':
    raise SystemExit(main())
