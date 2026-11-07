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
from manipulation_agent.clients.qwen.protocol import native_tools
from manipulation_agent.clients.qwen.worker import run
from manipulation_agent.tools import tool_specs
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
        assert tools==native_tools(tool_specs('skills'))
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
                          'point':[.5,.5] if i==8 else [500,750]},
                'placement_yaw_degrees':None,'wait_seconds':None}
        elif i<60:
            name,args='act',{'primitive':'wait','revision':self.latest['revision'],'target':None,
                             'placement_yaw_degrees':None,'wait_seconds':.1}
        else:name,args='finish',{'outcome':'blocked','reason':'Scripted CPU transport fixture only.'}
        return {'raw_response':xml(name,args)+'<|im_end|>','input_tokens':100+i,
                'generated_tokens':20,'finish_reason':'eos','input_ids_sha256':'fixture-only'}


def test_worker_real_mcp_all_tools_errors_images_and_more_than_50_actions(mock_episode,tmp_path):
    url,episode,process=mock_episode
    output=tmp_path/'controller';output.mkdir()
    request={'instructions':system_prompt('skills'),'prompt':'Synthetic CPU transport fixture only.',
             'tools':tool_specs('skills'),'mcp_command':sys.executable,
             'mcp_args':['-m','manipulation_agent.mcp_server','--bridge',url]}
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
    assert calls[0]['arguments']['target']['point']==[500,750]
    process.wait(timeout=15)
    record=json.loads((episode/'run.json').read_text())
    assert record['agent_steps']>50 and record['actions']>50
    assert record['execution_limit_policy']['episode_budget_policy']=='unbounded'
    assert record['evaluation']['protocol']=='mock_rgb_contract_only'
    latest=backend.messages[-1]
    assert latest[0]['content']==request['instructions']
    assert latest[1]['content']==request['prompt']
    # Every prior delivered image remains present, with original files intact.
    images=[part for message in latest if isinstance(message.get('content'),list)
            for part in message['content'] if part['type']=='image']
    assert len(images)>200 and all(Path(image['image']).is_file() for image in images)


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
