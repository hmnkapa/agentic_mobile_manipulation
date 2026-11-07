"""Read immutable Codex evidence without importing a simulator or replay code."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
from pathlib import Path

from .snapshot import _snapshot_context
from .assets import ImageAsset, _captures, public_content
from .common import ConversionError, dumps, file_hash, inside, jsonl, loads, read_json
from .protocol import (build_student_context, rewrite_interface_text,
                       to_student_arguments, validate_arguments)



@dataclass(frozen=True)
class EpisodeSource:
    run_dir: Path
    controller_dir: Path

    def paths(self):
        return {"run_dir": str(self.run_dir.resolve()), "controller_dir": str(self.controller_dir.resolve())}




@dataclass
class PublicText:
    text: str
    line: int


@dataclass
class Decision:
    call_id: str
    tool: str
    arguments: object
    text: str
    call_line: int
    result_line: int
    result: dict
    content: list[dict]
    target: bool
    evidence_id: str | None
    transport_error: bool = False
    interleaved_text: list[PublicText] = field(default_factory=list)

    def assistant(self, *, target=False):
        args = self.arguments if isinstance(self.arguments, str) else dumps(self.arguments)
        message = {"role": "assistant", "content": self.text,
                   "tool_calls": [{"id": self.call_id, "type": "function", "function": {
                       "name": self.tool, "arguments": args}}]}
        if not target:
            message["loss"] = False
        return message

    def reply(self):
        return {"role": "tool", "name": self.tool, "tool_call_id": self.call_id,
                "content": deepcopy(self.content)}

    def history(self):
        # These messages arrived after dispatch. They belong only in later
        # decisions' prefixes, between this call and its actual receipt.
        return [self.assistant(),
                *({"role": "assistant", "content": part.text, "loss": False}
                  for part in self.interleaved_text), self.reply()]


@dataclass
class Episode:
    source: EpisodeSource
    provenance: dict
    system: str = ""
    user: str = ""
    tools: list[dict] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    assets: dict[str, ImageAsset] = field(default_factory=dict)
    warnings: list[dict] = field(default_factory=list)
    skipped: str | None = None


def independent_success(evaluation):
    # The official evaluator is authoritative even if recorder/client status
    # disagrees. Do not infer success from Q, a finish claim or exit status.
    if not isinstance(evaluation, dict):
        return None
    key = "official_task_success" if "official_task_success" in evaluation else "task_success"
    value = evaluation.get(key)
    return value if type(value) is bool else None




def _simulator_index(path):
    calls, results = {}, {}
    for line, event in jsonl(path):
        kind = event.get("kind")
        if kind == "tool_call":
            key = event.get("request_id")
            if not key or key in calls:
                raise ConversionError("ambiguous_simulator_call", "Simulator call ID is absent or duplicated", line=line)
            calls[key] = event
        elif kind == "tool_result":
            key = event.get("id")
            if not key or key in results:
                raise ConversionError("ambiguous_evidence_id", "Simulator evidence ID is absent or duplicated", line=line)
            results[key] = event
    return calls, results


def _align(item, result, calls, results, line):
    event = results.get(result.get("evidence_id"))
    if event is None:
        raise ConversionError("missing_evidence", "Returned evidence_id has no simulator result", line=line)
    call = calls.get(event.get("request_id"))
    if (call is None or call.get("origin", "controller") != "controller"
            or event.get("origin", "controller") != "controller"
            or call.get("name") != item["tool"] or call.get("arguments") != item["arguments"]
            or event.get("name") != item["tool"]
            or result != {**event["result"], "evidence_id": event["id"]}):
        raise ConversionError("evidence_mismatch", "Controller call/result does not match its simulator evidence", line=line)






def _check_call_ledger(dispatches, pending, simulator_calls, simulator_results):
    """Account for every simulator call, including timed-out model requests.

    A transport error may have reached the simulator, or failed before dispatch.
    Never take the private result as a model observation. A known later receipt
    reserves its simulator call, preventing an earlier timeout from stealing it.
    """
    calls = [call for call in simulator_calls.values()
             if not (call.get("origin") == "supervisor" and call.get("name") == "finish")]
    reserved = {simulator_results[d["evidence_id"]]["request_id"]
                for d in dispatches if d["evidence_id"] is not None}
    index = 0
    for dispatch in dispatches:
        evidence = dispatch["evidence_id"]
        next_call = calls[index] if index < len(calls) else None
        if evidence is not None:
            request = simulator_results[evidence]["request_id"]
            if next_call is None or next_call["request_id"] != request:
                raise ConversionError("call_ledger_mismatch", "Simulator calls are missing from or reordered in the controller trace")
            index += 1
        elif (next_call and next_call["request_id"] not in reserved
              and next_call.get("origin", "controller") == "controller"
              and next_call.get("name") == dispatch["name"]
              and next_call.get("arguments") == dispatch["arguments"]):
            index += 1
    if pending and index < len(calls):
        item = pending[0]
        if (calls[index].get("origin", "controller") == "controller"
                and calls[index].get("name") == item["tool"] and calls[index].get("arguments") == item["arguments"]):
            index += 1
    if index != len(calls):
        raise ConversionError("unattributed_simulator_call", "Simulator has calls not attributable to the recorded model")


def load_episode(source: EpisodeSource) -> Episode:
    try:
        return _load_episode(source)
    except ConversionError:
        raise
    except (OSError, KeyError, IndexError, AttributeError, TypeError, ValueError) as exc:
        raise ConversionError("invalid_source_structure", "Source records have an unsupported or incomplete structure",
                              **source.paths()) from exc


def _load_episode(source: EpisodeSource) -> Episode:
    run_dir, controller_dir = source.run_dir.resolve(), source.controller_dir.resolve()
    source = EpisodeSource(run_dir, controller_dir)
    initial_hashes = {path: file_hash(path) for path in
                      (run_dir / "run.json", controller_dir / "controller.json")}
    run, ctrl = read_json(run_dir / "run.json"), read_json(controller_dir / "controller.json")
    evaluation = run.get("evaluation") or {}
    success = independent_success(evaluation)
    config = run.get("config") or {}
    provenance = {**source.paths(), "run_id": run.get("run_id"), "task_success": success,
                  "recorder_task_success": run.get("task_success"),
                  "task_instance": evaluation.get("task_instance", {}),
                  "scene": evaluation.get("task_instance", {}).get("scene"),
                  "model": ctrl.get("model"), "model_reasoning_effort": ctrl.get("model_reasoning_effort"),
                  "agent_profile": ctrl.get("agent_profile"), "goal_skill": ctrl.get("goal_skill"),
                  "controller_exit_code": ctrl.get("exit_code"), "controller_status": ctrl.get("status"),
                  "recorded_formal_finish": ctrl.get("formal_finish_observed"),
                  "run_source": run.get("source"), "controller_source": ctrl.get("source"),
                  "run_environment": {key: run.get(key) for key in ("host", "interpreter", "gpu_uuid", "pid", "unit")},
                  "controller_host": ctrl.get("host"), "controller_pid": ctrl.get("pid"),
                  "run_config": config, "evaluation_protocol": evaluation.get("protocol"),
                  "validation_level": "cpu_record_and_image_validation", "official_submission_eligible": False}
    episode = Episode(source, provenance)
    if success is not True:
        episode.skipped = "independent_task_not_successful" if success is False else "independent_success_unknown"
        return episode
    if (config.get("backend") != "omnigibson" or config.get("observation_mode") != "rgb_only"
            or config.get("policy") != "serve" or ctrl.get("controller") != "codex_cli"
            or evaluation.get("demo_motion") is True):
        raise ConversionError("ineligible_source", "Expected a real Codex RGB controller and simulator episode")
    if ctrl.get("rate_limit_continuations"):
        raise ConversionError("unsupported_resume", "Resumed runs need their additional user prompts; raw stdout alone is insufficient")
    system, catalog, context_provenance = _snapshot_context(controller_dir, ctrl, config)
    episode.system, episode.tools = build_student_context(system, catalog)
    episode.user = ctrl.get("instruction")
    if not isinstance(episode.user, str) or not episode.user.strip() or episode.user != config.get("instruction"):
        raise ConversionError("instruction_mismatch", "Controller and simulator instructions must agree")
    provenance.update(context_provenance)
    schemas = {spec["name"]: spec["inputSchema"] for spec in catalog}
    artifact_paths = {"run": run_dir / "run.json", "controller": controller_dir / "controller.json",
                      "simulator_events": run_dir / "events.jsonl", "model_events": controller_dir / "model_events.jsonl"}
    for key, path in (("captures", run_dir / "captures.jsonl"),
                      ("skill_manifest", run_dir / "skill_manifest.json"),
                      ("dependencies", run_dir / "dependency_versions.json"),
                      ("environment_config", run_dir / "environment_config.json")):
        if path.exists():
            artifact_paths[key] = path
    provenance["input_files"] = {key: {"path": str(path), "sha256": file_hash(path)} for key, path in artifact_paths.items()}
    for path, expected_hash in initial_hashes.items():
        if file_hash(path) != expected_hash:
            raise ConversionError("source_changed", "Source metadata changed during conversion", path=str(path))
    calls, results = _simulator_index(run_dir / "events.jsonl")
    captures = _captures(run_dir)
    pending = None
    text = []
    completed = {}
    used_evidence, dispatches = set(), []
    latest = None
    thread_count = 0
    turn_count = 0
    for line, event in jsonl(controller_dir / "model_events.jsonl"):
        typ, item = event.get("type"), event.get("item") or {}
        if typ == "thread.started":
            thread_count += 1
            if thread_count > 1:
                raise ConversionError("unsupported_resume", "Multiple client invocations need recorded continuation prompts", line=line)
        if typ == "turn.started":
            turn_count += 1
            if turn_count > 1:
                raise ConversionError("unsupported_resume", "Additional client turns need their recorded user prompts", line=line)
        if typ not in {"item.started", "item.completed", "item.updated"}:
            if typ in {"thread.started", "turn.started", "turn.completed", "turn.failed", "error"}:
                continue
            raise ConversionError("unsupported_model_event", "Unknown event may contain unrecorded input or model context", line=line)
        kind = item.get("type")
        if kind == "reasoning":
            continue  # Summaries/encrypted reasoning are not teacher CoT.
        if typ == "item.completed" and kind == "agent_message":
            if not isinstance(item.get("text"), str):
                raise ConversionError("invalid_assistant_text", "Expected recorded public text", line=line)
            if pending:
                pending[3].append(PublicText(item["text"], line))
            else:
                text.append(item["text"])
            continue
        if kind != "mcp_tool_call":
            if kind == "agent_message":
                continue
            raise ConversionError("unsupported_model_item", "Executed non-MCP or unknown item cannot be omitted", line=line)
        if item.get("server") != "manipulation" or item.get("tool") not in schemas:
            raise ConversionError("unknown_model_tool", "Call is outside the archived manipulation catalog", line=line)
        call_id = item.get("id")
        if not isinstance(call_id, str) or not call_id:
            raise ConversionError("missing_call_id", "MCP call has no stable ID", line=line)
        if typ == "item.updated":
            continue
        if typ == "item.started":
            if pending or call_id in completed:
                raise ConversionError("overlapping_calls", "Calls overlap or reuse an ID", line=line)
            pending = (deepcopy(item), line, "\n\n".join(text), [])
            text.clear()
            continue
        if call_id in completed:
            if completed[call_id] != hashlib.sha256(dumps(item, sort_keys=True).encode()).hexdigest():
                raise ConversionError("conflicting_completion", "A call ID has conflicting completions", line=line)
            episode.warnings.append({"code": "duplicate_completion_ignored", "line": line})
            continue
        if pending is None:
            raise ConversionError("missing_call_start", "Cannot establish the input boundary without item.started", line=line)
        start, call_line, public_text, interleaved_text = pending
        if any(start.get(key) != item.get(key) for key in ("id", "tool", "arguments", "server")):
            raise ConversionError("call_pair_mismatch", "Started and completed MCP call differ", line=line)
        pending = None
        completed[call_id] = hashlib.sha256(dumps(item, sort_keys=True).encode()).hexdigest()
        response = item.get("result") or {}
        blocks = response.get("content") or []
        result = {}
        if blocks and blocks[0].get("type") == "text":
            try:
                decoded = loads(blocks[0]["text"])
                if isinstance(decoded, dict):
                    result = decoded
            except ValueError:
                pass
        transport_error = bool(item.get("error"))
        # Native Codex MCP failures can have error=null and omit isError from
        # the serialized result. The completed item's status is authoritative.
        native_failed = item.get("status") == "failed"
        mcp_error = native_failed or bool(response.get("isError"))
        target = result.get("ok") is True and not transport_error and not mcp_error
        if item["tool"] == "finish":
            target = target and result.get("closed") is True
        if "evidence_id" in result:
            _align(item, result, calls, results, line)
            if result["evidence_id"] in used_evidence:
                raise ConversionError("duplicate_evidence_binding", "Different model calls reuse one simulator receipt", line=line)
            used_evidence.add(result["evidence_id"])
        elif not transport_error and not mcp_error:
            raise ConversionError("missing_evidence", "A public tool result has no evidence_id", line=line)
        dispatches.append({"name": item["tool"], "arguments": item["arguments"],
                           "evidence_id": result.get("evidence_id")})
        content, assets = public_content(item, result, run_dir, captures, line)
        episode.assets.update(assets)
        if transport_error:
            content.append({"type": "text", "text": "Recorded MCP transport error: " + dumps(item["error"])})
        if response.get("isError"):
            content.append({"type": "text", "text": "Recorded MCP isError=true"})
        if native_failed:
            content.append({"type": "text", "text": "Recorded MCP status=failed"})
        if "structuredContent" in response and response["structuredContent"] != result:
            raise ConversionError("unsupported_structured_result", "MCP structured content differs from its text", line=line)
        args = item.get("arguments")
        if target:
            validate_arguments(args, schemas[item["tool"]])
            if item["tool"] in {"act", "look"}:
                if not latest or args.get("revision") != latest.get("revision"):
                    raise ConversionError("stale_target_revision", "Successful target does not use the latest delivered revision", line=line)
                if args.get("target") and args["target"]["image_ref"] not in {m["image_ref"] for m in latest["images"]}:
                    raise ConversionError("stale_target_image", "Successful target uses an image absent from the latest reply", line=line)
        try:
            adapted = to_student_arguments(item["tool"], args, strict=target)
        except ConversionError:
            if target:
                raise
            adapted = deepcopy(args)
            episode.warnings.append({"code": "invalid_failed_arguments_preserved", "line": line})
        if item["tool"] == "read_skill" and result:
            student_result = deepcopy(result)
            if isinstance(student_result.get("text"), str):
                student_result["text"] = rewrite_interface_text(student_result["text"])
                content[0] = {"type": "text", "text": dumps(student_result)}
        episode.decisions.append(Decision(call_id, item["tool"], adapted, public_text,
                                          call_line, line, result, content, target,
                                          result.get("evidence_id"), transport_error, interleaved_text))
        if result.get("observation", {}).get("images"):
            latest = result["observation"]
    if pending:
        episode.warnings.append({"code": "unfinished_call_not_a_target", "call_id": pending[0]["id"], "line": pending[1]})
    _check_call_ledger(dispatches, pending, calls, results)
    provenance["supervisor_finish_calls"] = sum(call.get("origin") == "supervisor" and call.get("name") == "finish"
                                                 for call in calls.values())
    provenance["model_finish_observed"] = any(d.tool == "finish" and d.target for d in episode.decisions)
    provenance["evidence_alignment"] = ("verified_public_results" if all(
        d.evidence_id is not None and not d.transport_error for d in episode.decisions)
        else "partial_transport_errors_recorded")
    if not any(d.target for d in episode.decisions):
        episode.skipped = "no_successful_tool_targets"
    for artifact in provenance["input_files"].values():
        if file_hash(Path(artifact["path"])) != artifact["sha256"]:
            raise ConversionError("source_changed", "An input changed during conversion", path=artifact["path"])
    return episode
