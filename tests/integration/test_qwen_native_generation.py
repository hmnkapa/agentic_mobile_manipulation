"""Explicit real-weight component checks, not autonomous simulator episodes."""
import json
import os
from pathlib import Path
import sys

import pytest

from manipulation_agent.clients.qwen.protocol import native_tools, parse_response
from manipulation_agent.sft.common import ConversionError
from manipulation_agent.tools import tool_specs
from manipulation_agent.training.artifacts import model_manifest
from manipulation_agent.training.rollout_backend import NativeGenerationBackend
from manipulation_agent.vision_policy import system_prompt
from manipulation_agent.records import source_version, now
from manipulation_agent.sft.common import file_hash


@pytest.mark.gpu
def test_real_native_weights_zero_and_four_rgb(request,tmp_path):
    model=os.environ.get('QWEN_NATIVE_MODEL')
    adapter=os.environ.get('QWEN_NATIVE_ADAPTER')
    if not model or not adapter:
        if 'gpu' in (request.config.option.markexpr or ''):
            pytest.fail('Explicit real weights require QWEN_NATIVE_MODEL and LoRA requires QWEN_NATIVE_ADAPTER')
        pytest.skip('Set explicit native model and adapter paths')
    from PIL import Image
    device=os.environ.get('QWEN_NATIVE_DEVICE','cuda:0')
    report={'status':'running','validation_level':'offline_native_generation_component',
            'task_success_evaluated':False,'device':device,
            'started_at':now(),'host':os.uname().nodename,'pid':os.getpid(),'python':sys.executable,
            'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES') if device!='cpu' else None,
            'source':source_version(),'task_instance':None,'assets':'synthetic_four_rgb_component_fixture',
            'base':model_manifest(model,require_weights=True),'adapter':adapter,'samples':[]}
    report['adapter_sha256']=file_hash(Path(adapter)/'adapter_model.safetensors')
    path=tmp_path/'native_generation.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    backend=None
    try:
        backend=NativeGenerationBackend(model,None,device)
        report['backend']=backend.describe()
        initial=[{'role':'system','content':system_prompt('skills')},
                  {'role':'user','content':'Use only the manipulation MCP tools to complete this simulation task.\n'
                   "Turn on the radio receiver that's on the table in the living room."}]
        catalog=tool_specs('skills');tools=native_tools(catalog)
        content=[{'type':'text','text':'Synthetic component observation. revision=0'}]
        for i,view in enumerate(('front','back','left','right')):
            image=tmp_path/(view+'.png');Image.new('RGB',(512,512),(i*50,100,150)).save(image)
            content.extend([{'type':'text','text':'RGB view='+view+', image_ref=current-'+view},
                            {'type':'image','image':str(image)}])
        report['rgb_sha256']={view:file_hash(tmp_path/(view+'.png'))
                              for view in ('front','back','left','right')}
        four=initial+[{'role':'assistant','content':'','tool_calls':[{'id':'init','type':'function',
                    'function':{'name':'initialize','arguments':{}}}]},
              {'role':'tool','name':'initialize','tool_call_id':'init','content':content}]
        encoded={}
        for kind in ('base','lora'):
            if kind=='lora':
                from peft import PeftModel
                backend.model=PeftModel.from_pretrained(backend.model,adapter,is_trainable=False,local_files_only=True)
                backend.model.eval();backend.model.requires_grad_(False)
                assert backend.describe()==report['backend']
            for count,messages in ((0,initial),(4,four)):
                result=backend.generate(messages,tools)
                assert result['generated_tokens']>0 and result['raw_response']
                assert result['finish_reason']=='eos'
                assert len(result['image_grid_thw'])==count
                if kind=='base':encoded[count]=result['input_ids_sha256']
                else:assert result['input_ids_sha256']==encoded[count]
                try:
                    _,calls=parse_response(result['raw_response'],catalog,
                        thinking_prefix=result['generation_prefix_has_think'])
                    result['parsed_calls']=calls
                except ConversionError as exc:
                    result['protocol_error']=exc.record()
                report['samples'].append({'kind':kind,'images':count,**result})
                path.write_text(json.dumps(report,indent=2)+'\n')
                assert 'protocol_error' not in result and result['parsed_calls'], 'Native model protocol check failed; see preserved report'
        report['status']='passed'
    finally:
        if report['status']=='running':report['status']='failed'
        report['finished_at']=now()
        path.write_text(json.dumps(report,indent=2)+'\n')
        if backend is not None:backend.close()
