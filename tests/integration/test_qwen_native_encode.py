"""Real native processor checks, independent of training recipes and GPU models."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest

from manipulation_agent.clients.qwen.protocol import parse_response, protocol_error_message
from manipulation_agent.sft.common import ConversionError
from manipulation_agent.sft.protocol import build_student_context, from_student_arguments
from manipulation_agent.tools import tool_specs
from manipulation_agent.training.rollout_backend import NativeGenerationBackend
from manipulation_agent.vision_policy import system_prompt


def native_model_path(request):
    model=os.environ.get('QWEN_NATIVE_MODEL')
    if not model:
        if 'qwen_encode' in (request.config.option.markexpr or ''):
            pytest.fail('Explicit native encoding validation requires QWEN_NATIVE_MODEL')
        pytest.skip('Set QWEN_NATIVE_MODEL for real native processor validation')
    return model


@pytest.mark.qwen_encode
@pytest.mark.parametrize('budget', [256, 64])
def test_online_image_token_budget_with_real_processor(request, tmp_path, budget):
    model = native_model_path(request)
    from transformers import AutoProcessor
    from PIL import Image
    backend = NativeGenerationBackend.__new__(NativeGenerationBackend)
    backend.processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    backend.image_max_token_num = budget
    backend.model_max_length = json.loads((Path(model) / 'config.json').read_text())['text_config']['max_position_embeddings']
    settings = deepcopy(backend.processor.image_processor.to_dict())
    content, hashes = [], []
    for index, view in enumerate(('front', 'back', 'left', 'right')):
        path = tmp_path / (view + '.png')
        Image.new('RGB', (512, 512), (index * 50, 100, 150)).save(path)
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        content.extend([{'type': 'text', 'text': 'RGB view=' + view + ', image_ref=current-' + view},
                        {'type': 'image', 'image': str(path)}])
    messages = [{'role': 'user', 'content': content}]
    original = deepcopy(messages)
    batch = backend.encode(messages, [])
    grids = batch['image_grid_thw'].tolist()
    merge = backend.processor.image_processor.merge_size
    counts = [t * h * w // merge ** 2 for t, h, w in grids]
    assert counts == [budget] * 4
    image_token = backend.processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    assert int((batch['input_ids'] == image_token).sum()) == 4 * budget
    assert messages == original and backend.processor.image_processor.to_dict() == settings
    assert [hashlib.sha256((tmp_path / (view + '.png')).read_bytes()).hexdigest()
            for view in ('front', 'back', 'left', 'right')] == hashes
    decoded = backend.processor.tokenizer.decode(batch['input_ids'][0].tolist(), skip_special_tokens=False)
    assert all('current-' + view in decoded for view in ('front', 'back', 'left', 'right'))
    (tmp_path / 'image_token_encoding.json').write_text(json.dumps({
        'status': 'passed', 'validation_level': 'real_processor_only_no_model_weights',
        'task_success_evaluated': False, 'image_max_token_num': budget,
        'image_grid_thw': grids, 'visual_tokens_per_image': counts, 'rgb_sha256': hashes,
    }, indent=2) + '\n')


@pytest.mark.qwen_encode
def test_native_processor_full_images_history_and_generation_prefix(request,tmp_path):
    model=native_model_path(request)
    from transformers import AutoProcessor
    from PIL import Image
    backend=NativeGenerationBackend.__new__(NativeGenerationBackend)
    backend.processor=AutoProcessor.from_pretrained(model,local_files_only=True)
    backend.model_max_length=json.loads((Path(model)/'config.json').read_text())['text_config']['max_position_embeddings']
    system,tools=build_student_context(system_prompt('skills'),tool_specs('skills'))
    messages=[{'role':'system','content':system},
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
    assert 'point_2d' in decoded and 'integers from 0 to 1000' in decoded
    assert 'point: [x,y]' not in decoded and 'right=1,' not in decoded
    assert all('current-'+view in decoded for view in ('front','back','left','right'))
    assert decoded.endswith('<|im_start|>assistant\n<think>\n')
    messages.extend([{'role':'assistant','content':'Native action.','tool_calls':[{'id':'act','type':'function',
        'function':{'name':'act','arguments':{'primitive':'grasp','revision':0,
                    'target':{'image_ref':'current-front','point_2d':[250,750]},
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


@pytest.mark.qwen_encode
def test_native_template_scalar_roundtrip_and_feedback_with_four_images(request, tmp_path):
    model = native_model_path(request)
    from transformers import AutoProcessor
    from PIL import Image
    backend = NativeGenerationBackend.__new__(NativeGenerationBackend)
    backend.processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    backend.model_max_length = json.loads((Path(model) / 'config.json').read_text())['text_config']['max_position_embeddings']
    catalog = tool_specs('skills')
    # The extra boolean tool is a serialization fixture, never dispatched.
    fixture_catalog = catalog + [{'name': 'boolean_fixture', 'inputSchema': {
        'type': 'object', 'properties': {'enabled': {'type': 'boolean'}, 'disabled': {'type': 'boolean'}}}}]
    calls = [
        {'name': 'act', 'arguments': {'primitive': 'grasp', 'revision': 0,
                                    'target': {'image_ref': 'current-front', 'point_2d': [250, 750]},
                                    'placement_yaw_degrees': None, 'wait_seconds': None}},
        {'name': 'act', 'arguments': {'primitive': 'release', 'revision': 0, 'target': None,
                                    'placement_yaw_degrees': None, 'wait_seconds': None}},
        {'name': 'task_notes', 'arguments': {'operation': 'read', 'content': None, 'expected_version': None}},
        {'name': 'boolean_fixture', 'arguments': {'enabled': True, 'disabled': False}},
        {'name': 'finish', 'arguments': {'outcome': 'blocked', 'reason': 'None'}},
    ]
    system,fixture_tools = build_student_context(system_prompt('skills'), fixture_catalog)
    model_catalog = [{'name': t['function']['name'], 'inputSchema': t['function']['parameters']}
                     for t in fixture_tools]
    messages = [{'role': 'system', 'content': system},
                {'role': 'user', 'content': 'Synthetic native-template validation only.'}]
    rendered = backend.processor.apply_chat_template(messages + [{
        'role': 'assistant', 'content': '', 'tool_calls': [
            {'type': 'function', 'function': deepcopy(call)} for call in calls]}],
        tools=fixture_tools, add_generation_prompt=False, tokenize=False)
    response = rendered.rsplit('<|im_start|>assistant\n', 1)[1].split('<|im_end|>', 1)[0]
    assert all('\n' + value + '\n' in response for value in ('None', 'True', 'False'))
    assert parse_response(response, model_catalog)[1] == calls
    assert from_student_arguments('act', calls[0]['arguments'])['target'] == {
        'image_ref': 'current-front', 'point': [.25, .75]}

    content = [{'type': 'text', 'text': 'Synthetic observation, revision=0.'}]
    hashes = []
    for index, view in enumerate(('front', 'back', 'left', 'right')):
        path = tmp_path / (view + '.png')
        Image.new('RGB', (512, 512), (index * 50, 100, 150)).save(path)
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        content.extend([{'type': 'text', 'text': 'RGB view=' + view + ', image_ref=current-' + view},
                        {'type': 'image', 'image': str(path)}])
    messages.extend([{'role': 'assistant', 'content': '', 'tool_calls': [
        {'type': 'function', 'function': {'name': 'initialize', 'arguments': {}}}]},
        {'role': 'tool', 'name': 'initialize', 'content': content}])
    tools = build_student_context(system_prompt('skills'), catalog)[1]
    before = backend.encode(messages, tools)
    feedback = [protocol_error_message(ConversionError('unfinished_thinking', 'Incomplete native thinking block')),
                protocol_error_message(ConversionError('invalid_parameter_json', 'Invalid nested JSON',
                                                       tool='act', parameter='target'))]
    messages.extend(feedback)
    original = deepcopy(messages)
    after = backend.encode(messages, tools)
    assert messages == original
    assert after['image_grid_thw'].tolist() == before['image_grid_thw'].tolist()
    assert len(after['image_grid_thw']) == 4
    assert after['pixel_values'].equal(before['pixel_values'])
    decoded = backend.processor.tokenizer.decode(after['input_ids'][0].tolist(), skip_special_tokens=False)
    assert decoded.endswith('<|im_start|>assistant\n<think>\n')
    assert all(message['content'] in decoded for message in feedback)
    assert all('current-' + view in decoded for view in ('front', 'back', 'left', 'right'))
    assert [hashlib.sha256((tmp_path / (view + '.png')).read_bytes()).hexdigest()
            for view in ('front', 'back', 'left', 'right')] == hashes
    (tmp_path / 'native_recovery_encoding.json').write_text(json.dumps({
        'status': 'passed', 'validation_level': 'real_native_template_and_processor_only_no_model_weights',
        'task_success_evaluated': False, 'base_model': model, 'rgb_sha256': hashes,
        'scalar_roundtrip': ['None', 'True', 'False'], 'feedback_messages': len(feedback),
        'input_tokens_before_feedback': int(before['input_ids'].shape[1]),
        'input_tokens_after_feedback': int(after['input_ids'].shape[1]),
        'image_grid_thw': after['image_grid_thw'].tolist(),
    }, indent=2) + '\n')
