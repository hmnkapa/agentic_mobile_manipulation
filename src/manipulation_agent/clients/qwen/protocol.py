"""Lossless native-MCP serialization; no student coordinates or prompt rewrites."""
import base64
from copy import deepcopy
import hashlib
from pathlib import Path
import re

from ...sft.common import ConversionError, loads
from ...training.rollout_metrics import _parameter_value


def native_tools(catalog):
    return [{'type': 'function', 'function': {'name': spec['name'],
            'description': spec.get('description', ''), 'parameters': deepcopy(spec['inputSchema'])}}
            for spec in catalog]


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
        if function is None or function[1] not in specs:
            raise ConversionError('unknown_or_malformed_tool', 'Tool is not in the original MCP catalog')
        name, body = function[1], function[2]
        arguments, end = {}, 0
        properties = specs[name].get('properties', {})
        for parameter in re.finditer(r'<parameter=([^<>\s=]+)>(.*?)</parameter>', body, re.DOTALL):
            key = parameter[1]
            if body[end:parameter.start()].strip() or key in arguments:
                raise ConversionError('malformed_parameter', 'Ambiguous native tool parameter')
            # Unknown parameters are preserved for the harness to reject.
            arguments[key] = _parameter_value(parameter[2], properties.get(key, {'type': 'string'}))
            end = parameter.end()
        if body[end:].strip():
            raise ConversionError('malformed_parameter', 'Incomplete native tool parameter')
        calls.append({'name': name, 'arguments': arguments})
    return public, calls


def receipt_message(result, name, call_id, output):
    """Return the exact public receipt and image bytes in their original order."""
    content, images, expected = [], [], None
    for block in result.get('content', []):
        if block['type'] == 'text':
            content.append({'type': 'text', 'text': block['text']})
            try:
                data = loads(block['text'])
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
