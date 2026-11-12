"""Qwen serialization and SFT-compatible model feedback at the native MCP boundary."""
import base64
from copy import deepcopy
import hashlib
from pathlib import Path
import re

from ...sft.common import ConversionError, dumps, loads
from ...sft.protocol import COORDINATE_GUIDE, rewrite_interface_text
from ...training.rollout_metrics import _parameter_value

MODEL_INTERFACE = 'sft_point_2d_integer_0_1000'
MCP_INTERFACE = 'native_mcp_point_0_1'


def native_tools(catalog):
    return [{'type': 'function', 'function': {'name': spec['name'],
            'description': spec.get('description', ''), 'parameters': deepcopy(spec['inputSchema'])}}
            for spec in catalog]


def _native_parameter_value(text, schema):
    # Qwen's template renders scalar values with Jinja's string filter, but
    # renders mappings/sequences with tojson. Keep the offline parser strict.
    kind = schema.get('type')
    kinds = kind if isinstance(kind, list) else [kind]
    literal = text.strip()
    if literal == 'None' and ('null' in kinds or 'string' not in kinds):
        return None
    if 'string' not in kinds and literal in ('True', 'False'):
        return literal == 'True'
    return _parameter_value(text, schema)


def protocol_error_message(error):
    """Client feedback for an unexecuted response, not a fabricated MCP receipt."""
    instruction = ('Regenerate a complete tool-call response using the provided model-facing catalog. '
                   'Close all tool_call, function and parameter tags. '
                   'No tools from the rejected response were executed. '
                   'Continue from the current history and observations. ' + COORDINATE_GUIDE)
    if error.code == 'unfinished_thinking':
        instruction = 'Close the thinking block with </think> before any <tool_call>. ' + instruction
    return {'role': 'user', 'content': dumps({
        'source': 'qwen_native_client_protocol', 'error': error.record(),
        'dispatched_tool_calls': 0, 'retry_instruction': instruction,
    })}


def parse_response(raw, catalog, *, thinking_prefix=False):
    text = raw
    for token in ('<|im_end|>', '<|endoftext|>'):
        if text.endswith(token):
            text = text[:-len(token)]
    # The base template already supplies <think> in the generation prefix.
    first_call=text.find('<tool_call>')
    boundary=first_call if first_call>=0 else len(text)
    closing=text.find('</think>',0,boundary)
    if closing>=0:
        text=text[closing+len('</think>'):]
    elif thinking_prefix or '<think>' in text[:boundary]:
        raise ConversionError('unfinished_thinking', 'Incomplete native thinking block')
    start = text.find('<tool_call>')
    if start < 0:
        if '</tool_call>' in text:
            raise ConversionError('malformed_tool_call', 'Unpaired native tool delimiters')
        return text, []
    public = text[:start]
    rest = text[start:]
    matches = list(re.finditer(r'<tool_call>(.*?)</tool_call>', rest, re.DOTALL))
    if not matches or re.sub(r'<tool_call>.*?</tool_call>', '', rest, flags=re.DOTALL).strip():
        raise ConversionError('malformed_tool_call', 'Incomplete or trailing native call content')
    specs = {spec['name']: spec['inputSchema'] for spec in catalog}
    calls = []
    for match in matches:
        function = re.fullmatch(r'\s*<function=([^<>\s=]+)>(.*?)</function>\s*', match[1], re.DOTALL)
        if function is None:
            raise ConversionError('unknown_or_malformed_tool', 'Tool is not in the original MCP catalog')
        name, body = function[1], function[2]
        if name not in specs:
            raise ConversionError('unknown_or_malformed_tool', 'Tool is not in the original MCP catalog', tool=name)
        arguments, end = {}, 0
        properties = specs[name].get('properties', {})
        for parameter in re.finditer(r'<parameter=([^<>\s=]+)>(.*?)</parameter>', body, re.DOTALL):
            key = parameter[1]
            if body[end:parameter.start()].strip() or key in arguments:
                raise ConversionError('malformed_parameter', 'Ambiguous native tool parameter',
                                      tool=name, parameter=key)
            # Unknown parameters are preserved for the harness to reject.
            try:
                arguments[key] = _native_parameter_value(parameter[2], properties.get(key, {'type': 'string'}))
            except ConversionError as exc:
                raise ConversionError(exc.code, str(exc), tool=name, parameter=key) from exc
            end = parameter.end()
        if body[end:].strip():
            raise ConversionError('malformed_parameter', 'Incomplete native tool parameter', tool=name)
        calls.append({'name': name, 'arguments': arguments})
    return public, calls


def receipt_message(result, name, call_id, output):
    """Preserve receipts/RGB; adapt only skill interface text, as in SFT."""
    content, images, expected = [], [], None
    for block in result.get('content', []):
        if block['type'] == 'text':
            content.append({'type': 'text', 'text': block['text']})
            try:
                data = loads(block['text'])
                if name == 'read_skill' and isinstance(data, dict) and isinstance(data.get('text'), str):
                    data['text'] = rewrite_interface_text(data['text'])
                    content[-1]['text'] = dumps(data)
                frames = data.get('observation', {}).get('images') if isinstance(data, dict) else None
                if frames is not None:
                    expected = frames
            except ValueError:
                pass
        elif block['type'] == 'image':
            raw = base64.b64decode(block['data'], validate=True)
            sha = hashlib.sha256(raw).hexdigest()
            index = len(images)
            frame = expected[index] if expected is not None and index < len(expected) else None
            if expected is not None and (frame is None or frame.get('sha256') != sha):
                raise RuntimeError('MCP RGB order or SHA-256 differs from the public observation')
            directory = Path(output) / 'images'
            directory.mkdir(exist_ok=True)
            path = directory / (sha + ('.png' if block['mimeType'] == 'image/png' else '.jpg'))
            if path.exists():
                if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                    raise RuntimeError('Archived RGB bytes changed')
            else:
                path.write_bytes(raw)
            content.append({'type': 'image', 'image': str(path.resolve())})
            images.append({'sha256': sha, 'mime_type': block['mimeType'],
                           'image_ref': frame.get('image_ref') if frame else None,
                           'view': frame.get('view') if frame else None})
        else:
            raise RuntimeError('Unsupported MCP content type: ' + str(block['type']))
    if expected is not None and len(images) != len(expected):
        raise RuntimeError('MCP receipt omitted observation image bytes')
    return {'role': 'tool', 'name': name, 'tool_call_id': call_id, 'content': content}, images
