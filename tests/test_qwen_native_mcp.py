"""Real SDK/bridge validation with scripted CPU fixtures, never model acceptance."""
import asyncio
import contextlib
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

from manipulation_agent.clients.events import parse_events
from manipulation_agent.clients.qwen.worker import run
from manipulation_agent.sft.protocol import build_student_context, rewrite_interface_text
from manipulation_agent.tools import tool_specs
from manipulation_agent.training.artifacts import TrainingError, fingerprint
from manipulation_agent.vision_policy import system_prompt
from test_qwen_native import xml


@pytest.fixture
def mock_episode(tmp_path):
    pytest.importorskip('mcp')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src')}
    for key in ('MAS_EXECUTION_CLOCK_PATH','MAS_EPISODE_DEADLINE_UNIX'):env.pop(key,None)
    output=tmp_path/'episode'
    code=('from manipulation_agent import bridge; bridge.CLOSED_REPLAY_SECONDS=.1; '
          'from manipulation_agent.vision_cli import main; main()')
    log=(tmp_path/'simulator.log').open('w')
    process=subprocess.Popen([sys.executable,'-c',code,'--backend','mock','--agent-profile','skills',
        '--unbounded-episode','--max-actions','1','--max-sim-steps','1',
        '--port',str(port),'--output',str(output)],env=env,stdout=log,stderr=log)
    url=f'http://127.0.0.1:{port}'
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        until=time.monotonic()+30
        while time.monotonic()<until:
            if process.poll() is not None:pytest.fail('CPU fixture exited: '+(tmp_path/'simulator.log').read_text())
            try:
                with opener.open(url+'/healthz',timeout=1) as response:
                    if json.load(response)['ready']:break
            except OSError:time.sleep(.03)
        else:pytest.fail('CPU fixture readiness timeout')
        yield url,output,process
    finally:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
        log.close()


class ScriptedFixture:
    """Predetermined transport exercise, explicitly not a visual model policy."""
    def __init__(self):
        self.turn=0;self.latest=None;self.first=None;self.messages=[]

    def generate(self,messages,tools):
        assert tools==build_student_context(system_prompt('skills'),tool_specs('skills'))[1]
        self.messages.append(deepcopy(messages))
        for message in messages:
            if message['role']=='tool':
                body=json.loads(message['content'][0]['text'])
                if body.get('observation'):
                    self.latest=body['observation']
                    if self.first is None:self.first=self.latest
        i=self.turn;self.turn+=1
        if i==0:name,args='initialize',{}
        elif i==1:name,args='list_skills',{}
        elif i==2:name,args='read_skill',{'name':'visual-manipulation','resource':'SKILL.md'}
        elif i==3:name,args='task_notes',{'operation':'read','content':None,'expected_version':None}
        elif i in (4,5):name,args='task_notes',{'operation':'write','content':'  unchanged notes\n','expected_version':0}
        elif i==6:name,args='look',{'revision':self.latest['revision'],'yaw_degrees':10}
        elif i in (7,8):
            name,args='act',{'primitive':'grasp','revision':self.latest['revision'],
                'target':{'image_ref':self.first['images'][0]['image_ref'] if i==8 else self.latest['images'][0]['image_ref'],
                          'point_2d':[500,500] if i==8 else [500,750]},
                'placement_yaw_degrees':None,'wait_seconds':None}
        elif i<60:
            name,args='act',{'primitive':'wait','revision':self.latest['revision'],'target':None,
                             'placement_yaw_degrees':None,'wait_seconds':21 if i==9 else .1}
        else:name,args='finish',{'outcome':'blocked','reason':'Scripted CPU transport fixture only.'}
        return {'raw_response':xml(name,args)+'<|im_end|>','input_tokens':100+i,
                'generated_tokens':20,'finish_reason':'eos','input_ids_sha256':'fixture-only'}


def test_worker_real_mcp_all_tools_errors_images_and_more_than_50_actions(mock_episode,tmp_path,monkeypatch):
    from mcp import ClientSession
    url,episode,process=mock_episode
    output=tmp_path/'controller';output.mkdir()
    request={'instructions':system_prompt('skills'),'prompt':'Synthetic CPU transport fixture only.',
             'tools':tool_specs('skills'),'mcp_command':sys.executable,
             'mcp_args':['-m','manipulation_agent.mcp_server','--bridge',url]}
    original_request=deepcopy(request)
    dispatched=[]
    original_call=ClientSession.call_tool
    async def capture(self,name,arguments,*args,**kwargs):
        dispatched.append((name,deepcopy(arguments)))
        return await original_call(self,name,arguments,*args,**kwargs)
    monkeypatch.setattr(ClientSession,'call_tool',capture)
    backend=ScriptedFixture();stream=io.StringIO()
    with contextlib.redirect_stdout(stream):asyncio.run(run(request,output,backend))
    raw=output/'model_events.jsonl';raw.write_text(stream.getvalue())
    parsed=parse_events(raw,'qwen')
    assert parsed.formal_finish_observed
    assert set(parsed.tools_called)=={spec['name'] for spec in tool_specs('skills')}
    results=[e['result'] for e in parsed.canonical if e['kind']=='tool_result']
    bodies=[json.loads(r['content'][0]['text']) for r in results]
    errors=[body['error']['code'] for body in bodies if body.get('error')]
    assert 'invalid_arguments' in errors and 'stale_image_ref' in errors
    assert any('version' in error for error in errors)
    assert any(body.get('observation') for body in bodies if not body.get('ok',True))
    calls=[e for e in parsed.canonical if e['kind']=='tool_call' and e['tool']=='act']
    assert calls[0]['arguments']['target']['point']==[.5,.75]
    assert calls[0]['arguments']==next(args for name,args in dispatched if name=='act')
    events=[json.loads(line) for line in stream.getvalue().splitlines()]
    action=next(e['item'] for e in events if e['type']=='item.started' and e['item']['tool']=='act')
    assert action['model_arguments']['target']['point_2d']==[500,750]
    assert action['arguments']==calls[0]['arguments']
    process.wait(timeout=15)
    record=json.loads((episode/'run.json').read_text())
    assert record['agent_steps']>50 and record['actions']>50
    assert record['execution_limit_policy']['episode_budget_policy']=='unbounded'
    assert record['evaluation']['protocol']=='mock_rgb_contract_only'
    latest=backend.messages[-1]
    system,tools=build_student_context(request['instructions'],request['tools'])
    assert latest[0]['content']==system
    assert latest[1]['content']==request['prompt']
    assert request==original_request
    model_calls=[call['function'] for message in latest if message['role']=='assistant'
                 for call in message['tool_calls'] if call['function']['name']=='act']
    assert model_calls[0]['arguments']['target']==action['model_arguments']['target']
    skill=next(m for m in latest if m['role']=='tool' and m['name']=='read_skill')
    native_skill=next(body for body in bodies if 'text' in body)
    assert json.loads(skill['content'][0]['text'])['text']==rewrite_interface_text(native_skill['text'])
    inputs=json.loads((output/'qwen_inputs.json').read_text())
    assert inputs['instructions']==system and inputs['tools']==tools
    assert inputs['native_inputs']=={'instructions':request['instructions'],'tools':request['tools']}
    assert inputs['model_input_sha256']==fingerprint([system,request['prompt'],tools])
    assert inputs['project_input_sha256']==fingerprint([request['instructions'],request['prompt'],request['tools']])
    # Every prior delivered image remains present, with original files intact.
    images=[part for message in latest if isinstance(message.get('content'),list)
            for part in message['content'] if part['type']=='image']
    assert len(images)>200 and all(Path(image['image']).is_file() for image in images)


def test_coordinate_validation_is_atomic_and_valid_points_are_converted_once(mock_episode,tmp_path,monkeypatch):
    from mcp import ClientSession
    url,episode,process=mock_episode
    output=tmp_path/'controller';output.mkdir()
    request={'instructions':system_prompt('minimal'),'prompt':'Scripted coordinate boundary fixture.',
             'tools':tool_specs('skills'),'mcp_command':sys.executable,
             'mcp_args':['-m','manipulation_agent.mcp_server','--bridge',url]}
    invalid_targets=[{'point_2d':point} for point in (
        [-1,0],[1001,0],[0.5,750],[0.0,1000.0],[True,0],[0],[0,1,2],['500',750])]
    invalid_targets += [{'point':[.5,.75]}, {'point':[.5,.75],'point_2d':[500,750]}, {}, []]
    points=[[0,1000],[1000,0],[125,875]]
    dispatched=[]
    original=ClientSession.call_tool
    async def capture(self,name,arguments,*args,**kwargs):
        dispatched.append((name,deepcopy(arguments)))
        return await original(self,name,arguments,*args,**kwargs)
    monkeypatch.setattr(ClientSession,'call_tool',capture)

    class CoordinateFixture:
        turn=0
        history=None
        def generate(self,messages,tools):
            i=self.turn;self.turn+=1
            assert self.turn<=len(invalid_targets)+len(points)+4, 'Unexpected coordinate retry'
            self.history=deepcopy(messages)
            if i==0:
                raw=xml('initialize',{})
            else:
                observed=[json.loads(m['content'][0]['text'])['observation'] for m in messages
                          if m['role']=='tool'][-1]
                args={'primitive':'grasp','revision':observed['revision'],
                      'target':{'image_ref':observed['images'][0]['image_ref']},
                      'placement_yaw_degrees':None,'wait_seconds':None}
                index=i-1
                if index<len(invalid_targets):
                    bad=invalid_targets[index]
                    args['target']=({**args['target'],**bad} if isinstance(bad,dict) else bad)
                    # A valid earlier call must not execute if any point is invalid.
                    wait={**args,'primitive':'wait','target':None,'wait_seconds':.1}
                    raw=xml('act',wait)+'\n'+xml('act',args)
                elif index<len(invalid_targets)+len(points):
                    args['target']['point_2d']=points[index-len(invalid_targets)]
                    raw=xml('act',args)
                elif index==len(invalid_targets)+len(points):
                    raw=xml('act',{**args,'primitive':'release','target':None})
                elif index==len(invalid_targets)+len(points)+1:
                    raw=xml('act',{**args,'primitive':'wait','target':None,'wait_seconds':.1})
                else:
                    raw=xml('finish',{'outcome':'blocked','reason':'CPU coordinate fixture only.'})
            return {'raw_response':raw,'input_tokens':100,'generated_tokens':20,'finish_reason':'eos'}

    backend=CoordinateFixture();stream=io.StringIO()
    with contextlib.redirect_stdout(stream):asyncio.run(run(request,output,backend))
    events=[json.loads(line) for line in stream.getvalue().splitlines()]
    rejected=[e for e in events if e['type']=='generation.rejected']
    assert len(rejected)==len(invalid_targets)
    assert all(e['error']['code']=='invalid_coordinates' and e['dispatched_tool_calls']==0 for e in rejected)
    assert all(e['error']['tool']=='act' and e['error']['parameter']=='target' for e in rejected)
    assert all('point_2d' in e['feedback']['content'] and '1000' in e['feedback']['content'] for e in rejected)
    assert [name for name,args in dispatched]==['initialize']+['act']*5+['finish']
    assert [args['target']['point'] for name,args in dispatched if name=='act' and args['target']]==[
        [0,1],[1,0],[.125,.875]]
    assert all(args['target'] is None for name,args in dispatched if name=='act' and args['primitive'] in ('wait','release'))
    model_targets=[c['function']['arguments']['target'] for m in backend.history if m['role']=='assistant'
                   for c in m['tool_calls'] if c['function']['name']=='act']
    assert [t['point_2d'] for t in model_targets if t]==points
    assert all('point' not in t for t in model_targets if t)
    process.wait(timeout=15)


def test_uncertain_dispatch_is_not_retried(mock_episode,tmp_path,monkeypatch):
    from mcp import ClientSession
    from manipulation_agent.bridge import rpc
    url,episode,process=mock_episode
    output=tmp_path/'controller';output.mkdir()
    request={'instructions':system_prompt('skills'),'prompt':'Scripted lost-receipt fixture.',
             'tools':tool_specs('skills'),'mcp_command':sys.executable,
             'mcp_args':['-m','manipulation_agent.mcp_server','--bridge',url]}
    original=ClientSession.call_tool
    dispatches=[]

    async def lost_receipt(self,name,arguments,*args,**kwargs):
        result=await original(self,name,arguments,*args,**kwargs)
        dispatches.append(name)
        if name=='act':raise ConnectionError('Synthetic lost reply after execution')
        return result

    class TwoCallFixture:
        turn=0
        def generate(self,messages,tools):
            self.turn+=1
            if self.turn==1:name,args='initialize',{}
            else:
                observed=json.loads(messages[-1]['content'][0]['text'])['observation']
                name,args='act',{'primitive':'wait','revision':observed['revision'],'target':None,
                                 'placement_yaw_degrees':None,'wait_seconds':.1}
            return {'raw_response':xml(name,args),'input_tokens':100,'generated_tokens':20,'finish_reason':'eos'}

    monkeypatch.setattr(ClientSession,'call_tool',lost_receipt)
    backend=TwoCallFixture();stream=io.StringIO()
    with contextlib.redirect_stdout(stream),pytest.raises(Exception):
        asyncio.run(run(request,output,backend))
    assert dispatches==['initialize','act'] and backend.turn==2
    events=[json.loads(line) for line in stream.getvalue().splitlines()]
    uncertain=[e for e in events if (e.get('item',{}).get('error') or {}).get('code')=='execution_uncertain']
    assert len(uncertain)==1
    assert not any(e.get('item',{}).get('tool')=='finish' for e in events)
    # Explicit fixture-side supervisor closure, never attributed to the model.
    rpc(url,'finish',{'outcome':'aborted','reason':'CPU fixture lost-reply cleanup'},
        'fixture-supervisor-finish',origin='supervisor')
    process.wait(timeout=15)
    record=json.loads((episode/'run.json').read_text())
    assert record['actions']==1


def test_format_errors_regenerate_without_partial_dispatch_or_episode_reset(mock_episode, tmp_path):
    url, episode, process = mock_episode
    output = tmp_path / 'controller'; output.mkdir()
    request = {'instructions': system_prompt('skills'), 'prompt': 'Scripted protocol recovery fixture.',
               'tools': tool_specs('skills'), 'mcp_command': sys.executable,
               'mcp_args': ['-m', 'manipulation_agent.mcp_server', '--bridge', url]}

    class RecoveryFixture:
        def __init__(self):
            self.messages = []
            self.responses = []

        def generate(self, messages, tools):
            i = len(self.messages)
            assert i < 10, 'Unexpected retry after fixture completion'
            self.messages.append(deepcopy(messages))
            receipts = [m for m in messages if m['role'] == 'tool']
            assert len(receipts) == (0 if i == 0 else 1 if i < 8 else 2)
            if i == 0:
                raw = xml('initialize', {})
            else:
                observed = json.loads(receipts[-1]['content'][0]['text'])['observation']
                act = xml('act', {'primitive': 'wait', 'revision': observed['revision'], 'target': None,
                                 'placement_yaw_degrees': None, 'wait_seconds': .1})
                # Match Qwen's native serialization, including on the successful retry.
                act = act.replace('\nnull\n', '\nNone\n')
                if i == 1:
                    raw = 'Unclosed synthetic thinking. ' + act
                elif i == 2:
                    # The first call is valid; the second must prevent BOTH dispatches.
                    raw = act + '\n<tool_call><function=look><parameter=revision>0</parameter>' \
                                '<parameter=revision>1</parameter></function></tool_call>'
                elif i == 3:
                    raw = xml('unavailable_tool', {})
                elif i in (4, 8):
                    raw = 'No tool call in this synthetic response.'
                elif i == 5:
                    raw = xml('act', {'target': '{"point": [None, 0]}'})
                elif i == 6:
                    raw = act.replace('</tool_call>', '')
                elif i == 7:
                    raw = act
                else:
                    raw = xml('finish', {'outcome': 'blocked', 'reason': 'CPU recovery fixture complete.'})
            if i != 1:
                raw = '</think>\n' + raw
            raw += '<|im_end|>'
            self.responses.append(raw)
            return {'raw_response': raw, 'input_tokens': 100 + i, 'generated_tokens': 20,
                    'finish_reason': 'eos', 'generation_prefix_has_think': True}

    backend = RecoveryFixture(); stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        asyncio.run(run(request, output, backend))
    path = output / 'model_events.jsonl'; path.write_text(stream.getvalue())
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    rejected = [event for event in events if event['type'] == 'generation.rejected']
    assert [e['consecutive_failures'] for e in rejected] == [1, 2, 3, 4, 5, 6, 1]
    assert [e['error']['code'] for e in rejected] == [
        'unfinished_thinking', 'malformed_parameter', 'unknown_or_malformed_tool',
        'missing_tool_call', 'invalid_parameter_json', 'malformed_tool_call', 'missing_tool_call',
    ]
    assert rejected[1]['error']['tool'] == 'look' and rejected[1]['error']['parameter'] == 'revision'
    assert rejected[2]['error']['tool'] == 'unavailable_tool'
    assert rejected[4]['error']['tool'] == 'act' and rejected[4]['error']['parameter'] == 'target'
    for event in rejected:
        feedback = event['feedback']
        assert event['dispatched_tool_calls'] == 0 and feedback['role'] == 'user'
        assert backend.messages[event['turn']][-1] == feedback
        body = json.loads(feedback['content'])
        assert body['source'] == 'qwen_native_client_protocol'
        assert body['dispatched_tool_calls'] == 0 and body['error'] == event['error']
    assert '</think>' in json.loads(rejected[0]['feedback']['content'])['retry_instruction']
    assert not any(e['type'] == 'turn.failed' for e in events)
    assert not any(e.get('item', {}).get('type') == 'agent_message' for e in events)
    completions = [e for e in events if e['type'] == 'turn.completed']
    assert len(completions) == 10
    assert [e['usage'] for e in completions] == [
        {'input_tokens': 100 + i, 'output_tokens': 20} for i in range(10)]
    assert sum(e.get('status') == 'retrying' for e in completions) == 7
    generations = [json.loads(line) for line in (output / 'generations.jsonl').read_text().splitlines()]
    assert [g['raw_response'] for g in generations] == backend.responses
    assert [g['messages_sha256'] for g in generations] == [fingerprint(m) for m in backend.messages]
    # Valid history, including the original images, survives every rejection.
    assert all(messages[:4] == backend.messages[1] for messages in backend.messages[1:])
    images = [part for message in backend.messages[-1] if message['role'] == 'tool'
              for part in message['content'] if part['type'] == 'image']
    assert len(images) == 8 and all(Path(part['image']).is_file() for part in images)
    parsed = parse_events(path, 'qwen')
    assert parsed.formal_finish_observed and parsed.tools_called == ['initialize', 'act', 'finish']
    assert not any(e['kind'] == 'client_error' for e in parsed.canonical)
    assert len([e for e in parsed.canonical if e['kind'] == 'usage']) == 10
    process.wait(timeout=15)
    record = json.loads((episode / 'run.json').read_text())
    assert record['actions'] == 1 and record['agent_steps'] == 1
    assert record['execution_limit_policy']['episode_budget_policy'] == 'unbounded'


@pytest.mark.parametrize('failure', ['context_limit', 'context_exhausted', 'inference_out_of_memory'])
def test_generation_fault_after_format_feedback_still_terminates(mock_episode, tmp_path, failure):
    from manipulation_agent.bridge import rpc
    from manipulation_agent.clients.qwen.worker import failure_record
    url, episode, process = mock_episode
    output = tmp_path / 'controller'; output.mkdir()
    request = {'instructions': system_prompt('skills'), 'prompt': 'Scripted terminal-fault fixture.',
               'tools': tool_specs('skills'), 'mcp_command': sys.executable,
               'mcp_args': ['-m', 'manipulation_agent.mcp_server', '--bridge', url]}

    class FailureFixture:
        turn = 0

        def generate(self, messages, tools):
            self.turn += 1
            assert self.turn <= 3, 'Terminal generation fault was retried'
            if self.turn == 3:
                assert json.loads(messages[-1]['content'])['error']['code'] == 'missing_tool_call'
                if failure != 'context_limit':
                    raise TrainingError(failure, 'Synthetic terminal inference fault')
            return {'raw_response': xml('initialize', {}) if self.turn == 1 else 'Incomplete fixture response',
                    'input_tokens': 100, 'generated_tokens': 20,
                    'finish_reason': 'context_limit' if self.turn == 3 else 'eos'}

    backend = FailureFixture(); stream = io.StringIO()
    with contextlib.redirect_stdout(stream), pytest.raises(Exception) as raised:
        asyncio.run(run(request, output, backend))
    assert backend.turn == 3
    expected = 'context_exhausted_during_generation' if failure == 'context_limit' else failure
    assert expected in json.dumps(failure_record(raised.value))
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert len([e for e in events if e['type'] == 'generation.rejected']) == 1
    assert [e['item']['tool'] for e in events if e['type'] == 'item.started'] == ['initialize']
    rpc(url, 'finish', {'outcome': 'aborted', 'reason': 'CPU terminal-fault fixture cleanup'},
        'fixture-supervisor-finish', origin='supervisor')
    process.wait(timeout=15)
    assert json.loads((episode / 'run.json').read_text())['actions'] == 0
