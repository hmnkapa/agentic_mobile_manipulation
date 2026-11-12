"""An SFT-coordinate model client, with no simulator or ML imports at import time."""
import json
import subprocess

from ..base import ClientAdapter, run_process
from ..types import ClientCapability, PreparedProject
from ...records import write_json
from .config import load_config
from .protocol import MCP_INTERFACE, MODEL_INTERFACE


class QwenAdapter(ClientAdapter):
    name = 'qwen'

    def probe(self):
        # Configured interpreter/dependencies are checked in prepare_project.
        return ClientCapability(self.name, 'available', validation_level='adapter_available_not_model_validated')

    def run(self, project, output, timeout, on_start=None):
        return run_process(project, output, timeout, on_start, event_client=self.name,
                           model_startup_watchdog=1800)

    def prepare_project(self, config, instructions, tool_names):
        from ...tools import tool_specs
        runtime = load_config(config.qwen_config)
        if config.isolate_client_storage:
            raise ValueError('Codex session storage isolation does not apply to a local model process')
        script = ('import importlib.metadata as m,json; '
                  'print(json.dumps({p:m.version(p) for p in ("torch","transformers","peft","mcp")}))')
        try:
            versions = json.loads(subprocess.check_output([runtime['python'], '-c', script],
                                 text=True, stderr=subprocess.PIPE, timeout=30))
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            raise RuntimeError('Qwen interpreter requires torch, transformers, peft and mcp') from exc
        catalog = tool_specs(config.agent_profile)
        if [t['name'] for t in catalog] != tool_names:
            raise ValueError('Qwen must receive the original tool catalog')
        request = {'runtime': runtime, 'model': config.model, 'instructions': instructions,
                   'prompt': config.prompt, 'tools': catalog, 'mcp_command': config.mcp_command,
                   'mcp_args': config.mcp_args, 'versions': versions,
                   'interface': MODEL_INTERFACE, 'mcp_interface': MCP_INTERFACE,
                   'episode_budget_policy': 'unbounded'}
        path = config.output / 'qwen_request.json'
        write_json(path, request)
        env = {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
        if runtime['gpu_uuid']:
            env['CUDA_VISIBLE_DEVICES'] = runtime['gpu_uuid']
        return PreparedProject([runtime['python'], '-u', '-m',
                                'manipulation_agent.clients.qwen.worker', '--request', str(path.resolve())],
                               None, None, env_overlay=env)
