"""Local model and image encoding settings; task policy belongs to the harness."""
from pathlib import Path
import os
import re
import tomllib


def load_config(path):
    path = Path(path).resolve()
    data = tomllib.loads(path.read_text())
    allowed = {'python', 'model', 'adapter', 'device', 'gpu_uuid', 'image_max_token_num'}
    if set(data) - allowed or not {'python', 'model'} <= data.keys():
        raise ValueError('Qwen config accepts python, model, optional adapter, device, gpu_uuid and image_max_token_num only')
    image_max_token_num = data.get('image_max_token_num')
    if 'image_max_token_num' in data and (type(image_max_token_num) is not int or image_max_token_num not in (256, 64)):
        raise ValueError('image_max_token_num must be 256 or 64')
    result = {}
    for key in ('python', 'model', 'adapter'):
        value = data.get(key)
        if value is None and key == 'adapter':
            result[key] = None
            continue
        if not isinstance(value, str) or not value:
            raise ValueError('Missing local Qwen path: ' + key)
        value = os.path.expandvars(os.path.expanduser(value))
        if '$' in value:
            raise ValueError('Unresolved Qwen path environment variable: ' + key)
        local = Path(value)
        if not local.is_absolute():
            local = path.parent / local
        # Resolving a venv's Python symlink loses its site-packages.
        local = Path(os.path.abspath(local)) if key == 'python' else local.resolve()
        if not (local.is_file() if key == 'python' else local.is_dir()):
            raise ValueError('Qwen path is unavailable: ' + key)
        result[key] = str(local)
    device = data.get('device', 'cuda:0')
    uuid = data.get('gpu_uuid')
    if isinstance(uuid,str):uuid=os.path.expandvars(uuid)
    if device not in {'cpu', 'cuda:0'}:
        raise ValueError('Use cpu or cuda:0 with an explicit physical GPU UUID')
    if device != 'cpu' and (not isinstance(uuid, str) or not re.fullmatch(r'GPU-[0-9a-fA-F-]+', uuid)):
        raise ValueError('CUDA inference requires gpu_uuid')
    if device == 'cpu' and uuid is not None:
        raise ValueError('CPU inference must not claim a GPU')
    # Omitted settings retain native encoding for existing run configurations.
    result.update(device=device, gpu_uuid=uuid, image_max_token_num=image_max_token_num)
    return result
