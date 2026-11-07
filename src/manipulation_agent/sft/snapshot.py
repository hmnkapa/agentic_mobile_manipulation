"""Restore the trusted run-owned CPU tool catalog and recorded prompt."""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
from .common import ConversionError, dumps, loads


def _snapshot_context(controller_dir, controller, config):
    snapshot = controller_dir / "source_snapshot"
    expected = controller.get("source", {}).get("source_sha256")
    digest = hashlib.sha256()
    paths = sorted(snapshot.rglob("*.py"))
    if not paths or not expected:
        raise ConversionError("missing_source_snapshot", "A hashed controller source snapshot is required")
    for path in paths:
        if not path.resolve().is_relative_to(snapshot.resolve()):
            raise ConversionError("snapshot_path_escape", "Snapshot source leaves its directory")
        digest.update(str(Path("src") / path.relative_to(snapshot)).encode())
        digest.update(path.read_bytes())
    if digest.hexdigest() != expected:
        raise ConversionError("snapshot_hash_mismatch", "Controller source snapshot differs from its recorded hash")
    prompts = [value.split("=", 1)[1] for value in controller.get("command", [])
               if isinstance(value, str) and value.startswith("developer_instructions=")]
    if len(prompts) != 1:
        raise ConversionError("missing_recorded_prompt", "Expected one recorded developer_instructions value")
    try:
        system = loads(prompts[0])
    except ValueError as exc:
        raise ConversionError("invalid_recorded_prompt", "Cannot decode the recorded prompt") from exc
    if not isinstance(system, str) or not system.strip():
        raise ConversionError("invalid_recorded_prompt", "Recorded prompt is empty")
    profile = controller.get("agent_profile")
    if profile not in {"minimal", "skills", "workflow"}:
        raise ConversionError("unknown_profile", "No supported archived agent profile")
    if ("executor_profile" in config and "executor" in config
            and config["executor_profile"] != config["executor"]):
        raise ConversionError("conflicting_executor_config", "Recorded executor and executor_profile disagree")
    options, sources = {}, {"profile": "controller.agent_profile"}
    for parameter, keys in (("executor", ("executor_profile", "executor")),
                            ("nonofficial_actions", ("nonofficial_actions",))):
        key = next((key for key in keys if key in config), None)
        if key is not None:
            value = config[key]
            if not isinstance(value, str) or not value.strip():
                raise ConversionError("invalid_catalog_config", "Recorded catalog option must be a nonempty string", option=key)
            options[parameter] = value
            sources[parameter] = "run.config." + key
    # Import only the run-owned CPU tool catalog in a separate interpreter.
    # -B prevents snapshot changes; -I prevents ambient PYTHONPATH injection.
    # Snapshot loading is local Python execution: use only trusted run archives.
    code = '''import importlib.abc, inspect, json, sys
class NoSimulator(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'omnigibson', 'omni', 'habitat', 'habitat_sim', 'isaacsim'}:
            raise ImportError('Simulator import forbidden in SFT catalog loading')
sys.meta_path.insert(0, NoSimulator())
sys.path.insert(0, sys.argv[1])
from manipulation_agent.tools import tool_specs
options = json.loads(sys.argv[3])
signature = inspect.signature(tool_specs)
supported = {k: v for k, v in options.items() if k in signature.parameters}
bound = signature.bind(sys.argv[2], **supported)
bound.apply_defaults()
print(json.dumps({'catalog': tool_specs(*bound.args, **bound.kwargs),
                  'arguments': bound.arguments,
                  'unsupported_recorded_options': {k: v for k, v in options.items() if k not in supported}},
                 allow_nan=False))
'''
    try:
        with tempfile.TemporaryDirectory(prefix="manip-sft-catalog-") as cwd:
            process = subprocess.run([sys.executable, "-I", "-B", "-c", code,
                                      str(snapshot.resolve()), profile, dumps(options)], cwd=cwd,
                                     capture_output=True, text=True, timeout=30)
        if process.returncode:
            raise ValueError("Snapshot import failed")
        restored = loads(process.stdout)
        catalog = restored["catalog"]
        schemas = {t["name"]: t["inputSchema"] for t in catalog}
        # check_server uses stdlib JSON's default ASCII escaping and separators.
        import json
        schema_hash = hashlib.sha256(json.dumps(schemas, sort_keys=True).encode()).hexdigest()
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        # Never echo arbitrary snapshot stdout/stderr or controller argv.
        raise ConversionError("snapshot_catalog_error", "Cannot load the archived CPU tool catalog") from exc
    preflight = controller.get("mcp_preflight", {})
    if schema_hash != preflight.get("schema_sha256") or sorted(schemas) != sorted(preflight.get("tool_names", [])):
        raise ConversionError("schema_hash_mismatch", "Archived tool catalog differs from recorded MCP discovery")
    if len(schemas) != len(catalog):
        raise ConversionError("duplicate_tool", "Archived tool names are not unique")
    return system, catalog, {"source_snapshot_sha256": expected, "schema_sha256": schema_hash,
                              "recorded_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
                              "catalog_restore": {
                                  "arguments": restored["arguments"],
                                  "argument_sources": {key: sources.get(key, "snapshot.tool_specs.default")
                                                       for key in restored["arguments"]},
                                  "unsupported_recorded_options": restored["unsupported_recorded_options"]}}


