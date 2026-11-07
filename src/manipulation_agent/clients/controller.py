"""Client selection only; episode behavior is owned by the existing harness."""
from pathlib import Path
import sys


def controller_client(manifest):
    name=manifest.get('controller_client','codex_cli').removesuffix('_cli')
    if name not in {'codex','opencode','kimi','qwen'}:
        raise ValueError('Unsupported controller client')
    return name


def qwen_config_path(manifest, worker):
    return worker.get('qwen_config') or manifest.get('qwen_config')


def controller_prefix(source, manifest, worker):
    client=controller_client(manifest)
    if client=='codex':return [sys.executable,str(Path(source)/'scripts/run_codex_controller.py')]
    command=[sys.executable,str(Path(source)/'scripts/run_model_controller.py'),'--client',client]
    if client=='qwen':
        path=qwen_config_path(manifest,worker)
        if not path or manifest.get('episode_budget_policy')!='unbounded':
            raise ValueError('Qwen requires qwen_config and an unbounded episode policy')
        command+=['--qwen-config',str(path)]
    return command


def controller_options(manifest, worker):
    if controller_client(manifest)!='codex':return []
    command=[]
    if manifest.get('model_reasoning_effort'):
        command+=['--reasoning-effort',manifest['model_reasoning_effort']]
    for field in ('model_provider_profile','model_provider_gateway'):
        if worker.get(field):command+=['--'+field.replace('_','-'),worker[field]]
    if worker.get('isolate_client_storage'):command+=['--isolate-client-storage']
    return command
