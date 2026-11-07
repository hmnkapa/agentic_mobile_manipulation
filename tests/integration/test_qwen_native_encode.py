"""Real native processor checks, independent of training recipes and GPU models."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest

from manipulation_agent.clients.qwen.protocol import native_tools
from manipulation_agent.tools import tool_specs
from manipulation_agent.training.rollout_backend import NativeGenerationBackend
from manipulation_agent.vision_policy import system_prompt


@pytest.mark.qwen_encode
def test_native_processor_full_images_history_and_generation_prefix(request,tmp_path):
    model=os.environ.get('QWEN_NATIVE_MODEL')
    if not model:
        if 'qwen_encode' in (request.config.option.markexpr or ''):
            pytest.fail('Explicit native encoding validation requires QWEN_NATIVE_MODEL')
        pytest.skip('Set QWEN_NATIVE_MODEL for real native processor validation')
    from transformers import AutoProcessor
    from PIL import Image
    backend=NativeGenerationBackend.__new__(NativeGenerationBackend)
    backend.processor=AutoProcessor.from_pretrained(model,local_files_only=True)
    backend.model_max_length=json.loads((Path(model)/'config.json').read_text())['text_config']['max_position_embeddings']
    tools=native_tools(tool_specs('skills'))
    messages=[{'role':'system','content':system_prompt('skills')},
              {'role':'user','content':'Turn on the radio receiver that is on the table in the living room.'}]
    original=deepcopy(messages)
    zero=backend.encode(messages,tools)
    assert messages==original and 'labels' not in zero
    assert 'image_grid_thw' not in zero or len(zero['image_grid_thw'])==0
    content=[{'type':'text','text':'Original public observation text. revision=0'}]
    hashes=[]
    for index,view in enumerate(('front','back','left','right')):
        path=tmp_path/(view+'.png')
        Image.new('RGB',(512,512),(index*50,100,150)).save(path)
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        content.extend([{'type':'text','text':'RGB view='+view+', image_ref=current-'+view},
                        {'type':'image','image':str(path)}])
    messages.extend([{'role':'assistant','content':'','tool_calls':[{'id':'init','type':'function',
        'function':{'name':'initialize','arguments':{}}}]},
        {'role':'tool','name':'initialize','tool_call_id':'init','content':content}])
    original=deepcopy(messages)
    four=backend.encode(messages,tools)
    assert messages==original and 'labels' not in four
    assert four['image_grid_thw'].tolist()==[[1,32,32]]*4
    decoded=backend.processor.tokenizer.decode(four['input_ids'][0].tolist(),skip_special_tokens=False)
    assert 'point_2d' not in decoded
    assert 'normalized' in decoded and all('current-'+view in decoded for view in ('front','back','left','right'))
    assert decoded.endswith('<|im_start|>assistant\n<think>\n')
    messages.extend([{'role':'assistant','content':'Native action.','tool_calls':[{'id':'act','type':'function',
        'function':{'name':'act','arguments':{'primitive':'grasp','revision':0,
                    'target':{'image_ref':'current-front','point':[.25,.75]},
                    'placement_yaw_degrees':None,'wait_seconds':None}}}]},
        {'role':'tool','name':'act','tool_call_id':'act','content':'Unmodified error feedback.'},
        {'role':'assistant','content':' Keep the complete public history. '*10000}])
    original=deepcopy(messages)
    long=backend.encode(messages,tools)
    assert messages==original
    assert long['input_ids'].shape[1]>17408
    assert long['image_grid_thw'].tolist()==[[1,32,32]]*4
    assert [hashlib.sha256((tmp_path/(view+'.png')).read_bytes()).hexdigest()
            for view in ('front','back','left','right')]==hashes
    report={'status':'passed','validation_level':'real_processor_native_prefix_only_no_model_weights',
            'task_success_evaluated':False,'base_model':model,'rgb_sha256':hashes,
            'zero_tokens':int(zero['input_ids'].shape[1]),'four_tokens':int(four['input_ids'].shape[1]),
            'long_tokens':int(long['input_ids'].shape[1]),'image_grid_thw':long['image_grid_thw'].tolist()}
    (tmp_path/'native_encoding.json').write_text(json.dumps(report,indent=2)+'\n')
