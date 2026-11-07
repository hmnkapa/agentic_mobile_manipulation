"""Native input/receipt contracts; synthetic fixtures are not policy evidence."""
import base64
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from manipulation_agent.clients import ClientConfig, get_adapter
from manipulation_agent.clients.base import run_process
from manipulation_agent.clients.types import PreparedProject
from manipulation_agent.clients.controller import controller_prefix, controller_options
from manipulation_agent.clients.events import parse_events
from manipulation_agent.clients.qwen.config import load_config
from manipulation_agent.clients.qwen.protocol import native_tools, parse_response, receipt_message
from manipulation_agent.observations.mock_rgb import png_fixture
from manipulation_agent.sft.common import ConversionError
from manipulation_agent.tools import tool_specs
from manipulation_agent.vision_policy import system_prompt


def xml(name,args):
    parameters=''.join('<parameter='+key+'>\n'+(value if isinstance(value,str) else json.dumps(value))+
                       '\n</parameter>\n' for key,value in args.items())
    return '<tool_call>\n<function='+name+'>\n'+parameters+'</function>\n</tool_call>'


def runtime_config(tmp_path, *, adapter=False):
    model=tmp_path/'model';model.mkdir(exist_ok=True)
    lora=tmp_path/'adapter';lora.mkdir(exist_ok=True)
    path=tmp_path/('lora.toml' if adapter else 'base.toml')
    path.write_text('python='+json.dumps(sys.executable)+'\nmodel="model"\ndevice="cpu"\n'+
                    ('adapter="adapter"\n' if adapter else ''))
    return path


def test_prompt_and_catalog_are_identical_to_codex(tmp_path):
    original=tool_specs('skills')
    prompt=system_prompt('skills')
    requests=[]
    for adapter in (False,True):
        output=tmp_path/('lora-out' if adapter else 'base-out');output.mkdir()
        config=ClientConfig('qwen','Qwen/Qwen3.5-4B','Unmodified task instruction.',sys.executable,
                            ['-m','manipulation_agent.mcp_server'],output,timeout=None,
                            qwen_config=runtime_config(tmp_path,adapter=adapter))
        with patch('subprocess.check_output',return_value='{"torch":"test","mcp":"1.28.1"}'):
            project=get_adapter('qwen').prepare_project(config,prompt,[s['name'] for s in original])
        request=json.loads((output/'qwen_request.json').read_text());requests.append(request)
        assert request['instructions']==prompt
        assert request['prompt']==config.prompt
        assert request['tools']==original
        assert 'qwen.worker' in project.argv[-3]
        assert not any('IMAGE' in name or 'PIXEL' in name for name in project.env_overlay)
    assert requests[0]['tools']==requests[1]['tools']
    assert requests[0]['instructions']==requests[1]['instructions']
    tools=native_tools(original)
    for spec,tool in zip(original,tools):
        assert tool['function']['parameters']==spec['inputSchema']
        assert tool['function']['description']==spec['description']


def test_native_parser_never_converts_or_repairs_arguments():
    args={'primitive':'grasp','revision':3,'target':{'image_ref':'front-current','point':[.25,.75]},
          'placement_yaw_degrees':None,'wait_seconds':None}
    raw='Internal thinking.</think>Public progress.\n'+xml('act',args)+'<|im_end|>'
    public,calls=parse_response(raw,tool_specs())
    assert public=='Public progress.\n'
    assert calls==[{'name':'act','arguments':args}]
    for target in ({'image_ref':'current','point':[500,750]},
                   {'image_ref':'current','point_2d':[500,750]},None):
        invalid={**args,'target':target,'revision':True}
        assert parse_response(xml('act',invalid),tool_specs())[1][0]['arguments']==invalid
    assert parse_response(xml('initialize',{'extra':'preserve'}),tool_specs())[1][0]['arguments']=={'extra':'preserve'}


def test_multiple_calls_are_serializable_without_loss():
    note='  line one\nline two\n '
    args={'operation':'write','content':note,'expected_version':0}
    _,calls=parse_response(xml('list_skills',{})+'\n'+xml('task_notes',args),tool_specs())
    assert calls==[{'name':'list_skills','arguments':{}},{'name':'task_notes','arguments':args}]
    args['content']='Keep literal </think> and <think> in my notes.'
    assert parse_response(xml('task_notes',args),tool_specs())[1][0]['arguments']==args
    with pytest.raises(ConversionError,match='Incomplete'):
        parse_response('No closing think. '+xml('initialize',{}),tool_specs(),thinking_prefix=True)


@pytest.mark.parametrize('raw',[
    '<think>unfinished', xml('initialize',{})+'trailing explanation',
    '<tool_call><function=initialize></function>', xml('hidden_shell',{}),
    '<tool_call><function=look><parameter=revision>0</parameter><parameter=revision>1</parameter></function></tool_call>',
])
def test_malformed_predictions_fail_without_dispatch(raw):
    with pytest.raises(ConversionError):parse_response(raw,tool_specs())


def test_original_rgb_and_public_errors_are_preserved(tmp_path):
    content=[];frames=[];raws=[]
    for i,view in enumerate(('front','back','left','right')):
        raw=png_fixture(bool(i%2));raws.append(raw)
        frames.append({'view':view,'image_ref':'current-'+view,'sha256':hashlib.sha256(raw).hexdigest()})
        content.extend([{'type':'text','text':'RGB view='+view},
                        {'type':'image','mimeType':'image/png','data':base64.b64encode(raw).decode()}])
    body={'ok':False,'error':{'code':'out_of_reach'},'observation':{'images':frames,'revision':1}}
    result={'content':[{'type':'text','text':json.dumps(body)},*content]}
    original=deepcopy(result)
    message,images=receipt_message(result,'act','test-call',tmp_path)
    assert result==original
    assert json.loads(message['content'][0]['text'])==body
    parts=[p for p in message['content'] if p['type']=='image']
    assert [Path(p['image']).read_bytes() for p in parts]==raws
    assert [p['view'] for p in images]==['front','back','left','right']
    result['content'][2]['data']=base64.b64encode(b'changed').decode()
    with pytest.raises(RuntimeError,match='SHA-256'):receipt_message(result,'act','next',tmp_path)


def test_config_cannot_smuggle_training_or_task_policy(tmp_path):
    path=runtime_config(tmp_path)
    assert load_config(path)['python']==os.path.abspath(sys.executable)
    for setting in ('image_max_token_num=64','max_length=17408','max_new_tokens=2048',
                    'history_compression="compact"','recipe="training.toml"'):
        original=path.read_text();path.write_text(original+setting+'\n')
        with pytest.raises(ValueError):load_config(path)
        path.write_text(original)
    with pytest.raises(ValueError):
        ClientConfig('qwen','model','task','python',[],tmp_path,qwen_config=path)


def test_gpu_environment_location_is_resolved_without_changing_visibility(tmp_path,monkeypatch):
    path=runtime_config(tmp_path)
    path.write_text(path.read_text().replace('device="cpu"','device="cuda:0"')+'gpu_uuid="${NATIVE_TEST_GPU}"\n')
    monkeypatch.setenv('NATIVE_TEST_GPU','GPU-1234-abcd')
    assert load_config(path)['gpu_uuid']=='GPU-1234-abcd'
    monkeypatch.delenv('NATIVE_TEST_GPU')
    with pytest.raises(ValueError):load_config(path)


def test_controller_selection_keeps_codex_command_and_low():
    source=Path('/frozen')
    original={'model_reasoning_effort':'low','episode_budget_policy':'unbounded'}
    assert controller_prefix(source,original,{})==[sys.executable,'/frozen/scripts/run_codex_controller.py']
    assert controller_options(original,{})==['--reasoning-effort','low']
    qwen={**original,'controller_client':'qwen_cli','qwen_config':'/runtime/base.toml'}
    assert controller_prefix(source,qwen,{})[1:]==['/frozen/scripts/run_model_controller.py','--client','qwen',
                                                  '--qwen-config','/runtime/base.toml']
    assert controller_options(qwen,{})==[]
    with pytest.raises(ValueError):controller_prefix(source,{**qwen,'episode_budget_policy':'agent_steps'},{})


def test_qwen_events_keep_source_and_formal_receipt(tmp_path):
    path=tmp_path/'model_events.jsonl'
    rows=[{'type':'item.started','source':'qwen_native_model_client','item':{
        'type':'mcp_tool_call','id':'1','server':'manipulation','tool':'finish',
        'arguments':{'outcome':'blocked','reason':'original reason'}}}]
    rows.append({'type':'item.completed','source':'qwen_native_model_client','item':{
        **rows[0]['item'],'result':{'content':[{'type':'text','text':'{"ok":true,"closed":true}'}]}}})
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    events=parse_events(path,'qwen')
    assert events.formal_finish_observed
    assert events.tools_called==['finish']
    assert [e['kind'] for e in events.canonical]==['tool_call','tool_result']
    assert all(e['client']=='qwen' for e in events.canonical)


def test_qwen_finish_cleanup_uses_the_existing_shared_receipt_rule(tmp_path):
    event={'type':'item.completed','item':{'type':'mcp_tool_call','id':'1','server':'manipulation',
        'tool':'finish','arguments':{},'result':{'content':[{'type':'text','text':'{"ok":true,"closed":true}'}]}}}
    script=('import json,time,pathlib; '
            'pathlib.Path("'+str(tmp_path/'qwen_runtime.json')+'").write_text("{\\"model_ready_at\\":1}"); '
            'print('+repr(json.dumps(event))+',flush=True); time.sleep(60)')
    result=run_process(PreparedProject([sys.executable,'-c',script],None,None),tmp_path,None,
                       event_client='qwen',finish_exit_grace=.1,model_startup_watchdog=10)
    assert result.finish_received and not result.timed_out
    assert result.cleanup_reason=='client_linger_after_successful_finish'
    assert result.returncode!=0  # actual forced process exit stays visible
    assert parse_events(tmp_path/'model_events.jsonl','qwen').formal_finish_observed


def test_model_startup_watchdog_is_separate_from_task_timeout(tmp_path):
    project=PreparedProject([sys.executable,'-c','import time; time.sleep(60)'],None,None)
    result=run_process(project,tmp_path,None,event_client='qwen',model_startup_watchdog=.1)
    assert result.cleanup_reason=='model_initialization_watchdog'
    assert not result.timed_out and not result.finish_received


def test_import_does_not_load_models_or_simulator():
    code=('import sys; from manipulation_agent.clients import get_adapter; get_adapter("qwen"); '
          'from manipulation_agent.training.rollout_backend import NativeGenerationBackend; '
          'assert not any(k in sys.modules for k in ("torch","transformers","swift","omnigibson"))')
    subprocess.run([sys.executable,'-c',code],check=True)


def test_native_turn_end_is_not_confused_with_document_end():
    from manipulation_agent.training.rollout_backend import native_generation_config
    original=SimpleNamespace(eos_token_id=248044)
    tokenizer=SimpleNamespace(eos_token_id=248046)
    config=native_generation_config(original,tokenizer)
    assert original.eos_token_id==248044
    assert config.eos_token_id==[248044,248046]
    assert native_generation_config(config,tokenizer).eos_token_id==[248044,248046]


def test_sdk_exception_groups_preserve_the_original_structured_fault():
    from manipulation_agent.clients.qwen.worker import failure_record
    error=ExceptionGroup('MCP transport failed',[ConversionError('execution_uncertain','No action retry')])
    record=failure_record(error)
    assert record['causes'][0]['structured']['code']=='execution_uncertain'
