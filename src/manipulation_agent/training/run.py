"""Launch and resume native ms-swift training with immutable contracts."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from ..sft.common import ConversionError, file_hash, jsonl, loads, read_json
from .artifacts import (ProcessFailure, TrainingError, canonical, create_output, encoder_source, environment,
                        fingerprint, journal, model_manifest, now, versions, write_json)
from .config import config_fingerprint, encoding_contract, validate
from .data import index_package, materialize, select_smoke
from .swift_backend import require_gpu, training_command, vision_environment
from .tracking import require_tracking, tracking_environment, tracking_plan, tracking_result


def trainer_source():
    root = Path(__file__).resolve().parent
    return {name: file_hash(root / name) for name in ("run.py", "swift_plugin.py", "tracking.py", "worker.py")}


def _gpu_probe_main():
    try:
        require_gpu(loads(sys.stdin.read()))
        print(canonical({"status": "passed", "environment": environment()}))
    except ConversionError as exc:
        print(canonical({"status": "failed", "error": exc.record()}))
        raise SystemExit(2)


def gpu_resource_probe(config):
    # A resource query creates a CUDA context. Run it separately so the CPU
    # controller does not retain hundreds of MiB during native training.
    command = [sys.executable, "-c", "from manipulation_agent.training.run import _gpu_probe_main; _gpu_probe_main()"]
    try:
        result = subprocess.run(command, input=canonical(config), text=True, capture_output=True,
                                timeout=60, start_new_session=True)
        report = loads(result.stdout)
    except (subprocess.TimeoutExpired, ValueError, OSError) as exc:
        raise TrainingError("gpu_probe_failed", "GPU resource subprocess did not return a valid report") from exc
    if result.returncode or report.get("status") != "passed":
        error = report.get("error", {})
        raise TrainingError(error.get("code", "gpu_probe_failed"), error.get("message", "GPU resource check failed"),
                            probe_return_code=result.returncode)
    return report["environment"]


def launch_command(config, model, prepared, output, *, resume=None):
    effective = deepcopy(config)
    if effective["launch"]["nproc_per_node"] == 1:
        # Accelerate treats an explicitly supplied backend as distributed even
        # for a direct single-process launch. Let it select NO distribution.
        effective["launch"].pop("ddp_backend", None)
    return training_command(effective, model, prepared, output, resume=resume)


def run_process(command, root, env, *, label="training"):
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    with (logs / (label + ".stdout.log")).open("x") as out, (logs / (label + ".stderr.log")).open("x") as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, env=env, cwd=root, start_new_session=True)
        journal(root, "process_started", label=label, pid=process.pid)
        try:
            code = process.wait()
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=20)
            journal(root, "process_interrupted", pid=process.pid, child_return_code=process.returncode)
            raise
    journal(root, "process_finished", label=label, pid=process.pid, child_return_code=code)
    if code:
        raise ProcessFailure("child_process_failed", "Training/export subprocess failed", child_return_code=code, label=label)
    return code


def checkpoint_manifest(path):
    path = Path(path).resolve()
    required = ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json")
    if not all((path / name).is_file() for name in required) or not list(path.glob("rng_state*.pth")):
        raise TrainingError("incomplete_checkpoint", "Checkpoint lacks adapter, optimizer, scheduler or RNG state", path=str(path))
    state = read_json(path / "trainer_state.json")
    if type(state.get("global_step")) is not int or state["global_step"] < 1:
        raise TrainingError("invalid_checkpoint_step", "Checkpoint global step is invalid")
    files = {}
    for item in sorted(path.rglob("*")):
        if item.is_file():
            if not item.resolve().is_relative_to(path):
                raise TrainingError("checkpoint_path_escape", "Checkpoint contains an external file link")
            files[item.relative_to(path).as_posix()] = file_hash(item)
    return {"path": str(path), "global_step": state["global_step"], "files": files,
            "fingerprint": fingerprint(files)}


def verify_checkpoint_binding(manifest, marker, contract, origin):
    if marker.get("contract_fingerprint") != fingerprint(contract):
        raise TrainingError("checkpoint_contract_mismatch", "Checkpoint does not match its immutable contract")
    entries = read_json(Path(origin) / "checkpoints.json").get("checkpoints", [])
    if not any(e.get("global_step") == manifest["global_step"] and e.get("files") == manifest["files"] for e in entries):
        raise TrainingError("checkpoint_changed", "Checkpoint differs from its recorded save-time hashes")
    declared = read_json(Path(manifest["path"]) / "adapter_config.json").get("base_model_name_or_path")
    if declared and (not isinstance(declared, str) or Path(declared).resolve() != Path(contract["model"]["root"]).resolve()):
        raise TrainingError("checkpoint_base_mismatch", "Adapter declares a different base model than its contract")


def validate_training_output(root, *, start_step, target_step):
    try:
        audit = read_json(root / "audit.json")
        if (audit.get("status") != "passed" or audit.get("start_step") != start_step
                or (target_step > 0 and audit.get("end_step") != target_step) or not audit.get("adapter_updated")):
            raise TrainingError("training_audit_failed", "Training did not pass its evidence checks")
        saved = read_json(root / "checkpoints.json")["checkpoints"]
        matches = [x for x in saved if x["global_step"] == audit["end_step"]]
        if len(matches) != 1:
            raise TrainingError("missing_checkpoint", "Expected one complete final checkpoint")
        verified = checkpoint_manifest(matches[0]["path"])
        if verified != matches[0]:
            raise TrainingError("checkpoint_changed", "Checkpoint changed after callback validation")
        return audit, verified
    except (ConversionError, OSError, KeyError, TypeError) as exc:
        detail = exc.record() if isinstance(exc, ConversionError) else {"error_type": type(exc).__name__}
        raise ProcessFailure("training_artifact_validation_failed", "Training exited but its artifacts did not validate",
                             validation_error=detail) from exc


def verify_preflight(index, config, model, path):
    path = Path(path).resolve()
    summary = read_json(path / "summary.json")
    if (summary.get("status") != "complete" or summary.get("scan_complete") is not True
            or summary.get("encoding_passed") is not True or summary.get("structure_only") is not False
            or summary.get("validation_level") != "qwen_processor_encoding"):
        raise TrainingError("preflight_not_verified", "Supplied preflight must contain complete real encoding checks")
    reports = summary.get("report_files", {})
    required = {"samples.jsonl", "errors.jsonl", "package_manifest.json", "encoding_contract.json"}
    if not required.issubset(reports) or any(file_hash(path / name) != sha for name, sha in reports.items()):
        raise TrainingError("preflight_changed", "Preflight reports changed after validation")
    saved = read_json(path / "encoding_contract.json")
    expected = {"package": index.fingerprint, "model": model["fingerprint"], "encoding": encoding_contract(config),
                "packages": versions(), "encoder_source": encoder_source()}
    if saved.get("fingerprint") != fingerprint(expected) or saved.get("contract") != expected:
        raise TrainingError("preflight_contract_mismatch", "Data, model, encoder, environment or template differs from preflight")
    rows = [r for _, r in jsonl(path / "samples.jsonl")]
    if (len(rows) != len(index.samples) or len({r.get("sample_id") for r in rows}) != len(rows)
            or {r.get("sample_id") for r in rows} != {s.sample_id for s in index.samples}
            or any(r.get("status") != "passed" or r.get("complete_target_verified") is not True for r in rows)):
        raise TrainingError("preflight_sample_mismatch", "Preflight lacks a verified result for every input")
    return rows


def select_smoke_without_preflight(index, config):
    """Select structural coverage without measuring or inventing token lengths."""
    selected, reasons = [], {}

    def choose(pool, reason, *, most_images=False):
        if not pool:
            raise TrainingError("smoke_coverage_gap", "No sample for required smoke coverage", requirement=reason)
        sample = min(pool, key=lambda s: ((-1 if most_images else 1) * s.image_count, s.sample_id))
        if sample.sample_id not in reasons:
            selected.append(sample)
            reasons[sample.sample_id] = []
        reasons[sample.sample_id].append(reason)

    for split in ("train", "val"):
        pool = [s for s in index.samples if s.split == split]
        tools = ("initialize", "list_skills", "read_skill", "task_notes", "act", "look", "finish") if split == "train" else ("initialize", "act", "finish")
        for tool in tools:
            choose([s for s in pool if s.target_tool == tool and (split != "val" or tool != "act" or s.image_count >= 8)],
                   split + ":" + tool)
        if split == "train":
            choose([s for s in pool if not s.image_count], "zero_images")
            choose([s for s in pool if s.image_count >= 8], "multiple_captures")
            choose([s for s in pool if s.has_failed_history], "failed_history")
        choose(pool, split + ":most_images", most_images=True)
        count = config["smoke"][split + "_samples"]
        if sum(s.split == split for s in selected) > count or len(pool) < count:
            raise TrainingError("smoke_coverage_gap", "Not enough distinct smoke samples", split=split)
        for sample in sorted(pool, key=lambda s: fingerprint([42, s.sample_id])):
            if sum(s.split == split for s in selected) >= count:
                break
            if sample.sample_id not in reasons:
                selected.append(sample)
                reasons[sample.sample_id] = ["seeded_fill"]
    return selected, {"algorithm": "fewest_images_per_requirement_then_most_images_then_seeded_fill", "seed": 42,
                      "lengths_verified": False, "input_samples": len(index.samples),
                      "samples": [{**asdict(s), "reasons": reasons[s.sample_id]} for s in selected]}


def train(data, config, model, preflight, output, *, dry_run=False):
    index = index_package(data, expected_k=config["data"]["k"],
                          expected_history_compression=config["data"].get("history_compression", "legacy"))
    manifest = model_manifest(model, require_weights=not dry_run)
    if dry_run:
        root = Path(output).resolve()
        if root.exists() or any(root.is_relative_to(p) or p.is_relative_to(root) for p in (index.root, Path(manifest["root"]))):
            raise TrainingError("invalid_output", "Output must be new and separate from data/model inputs")
        return {"status": "dry_run", "output": str(root), "selection_verified": False,
                "environment_verified": False, "validation_level": "cpu_plan_validation",
                "resolved_config": config, "config_fingerprint": config_fingerprint(config),
                "argv": launch_command(config, manifest["root"], root / "prepared", root / "swift")}
    reports = verify_preflight(index, config, manifest, preflight) if preflight is not None else None
    if config["smoke"]["enabled"]:
        selected, selection = (select_smoke(index, reports, config) if reports is not None
                               else select_smoke_without_preflight(index, config))
        selection["lengths_verified"] = reports is not None
    else:
        selected, selection = index.samples, {"scope": "full_dataset", "samples": [asdict(s) for s in index.samples],
                                              "lengths_verified": reports is not None}
        if reports is not None and any(r["tokens"] > min(config["data"]["max_length"], r["model_max_length"]) for r in reports):
            raise TrainingError("overlength_samples", "Full training is blocked by overlength samples; inspect preflight")
    return _execute(index, config, manifest, selected, selection, output,
                    preflight=str(Path(preflight).resolve()) if preflight is not None else None, dry_run=dry_run)


def resume(checkpoint, output, *, dry_run=False):
    checkpoint = Path(checkpoint).resolve()
    checkpoint_info = checkpoint_manifest(checkpoint)
    info = read_json(checkpoint / "manipulation_training.json")
    parent = Path(info["run_root"]).resolve()
    contract = read_json(parent / "training_contract.json")
    if info.get("contract_fingerprint") != fingerprint(contract):
        raise TrainingError("resume_contract_mismatch", "Checkpoint does not match its immutable training contract")
    verify_checkpoint_binding(checkpoint_info, info, contract, parent)
    config = deepcopy(contract["config"])
    config["smoke"]["stop_after_step"] = 0
    validate(config)
    if checkpoint_info["global_step"] >= config["training"]["max_steps"] > 0:
        raise TrainingError("resume_already_finished", "Checkpoint already reached the training step budget")
    index = index_package(contract["package"]["root"], expected_k=config["data"]["k"],
                          expected_history_compression=config["data"].get("history_compression", "legacy"))
    if index.fingerprint != contract["package"]["fingerprint"]:
        raise TrainingError("resume_data_changed", "Data package changed since original training")
    model = model_manifest(contract["model"]["root"], require_weights=True)
    if (model != contract["model"] or (not dry_run and versions() != contract["packages"])
            or encoder_source() != contract["encoder_source"] or trainer_source() != contract.get("trainer_source")):
        raise TrainingError("resume_environment_changed", "Model, dependencies, encoder or trainer changed since original training")
    allocator = {k: os.environ[k] for k in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF") if k in os.environ}
    if not dry_run and allocator != contract.get("allocator_environment", {}):
        raise TrainingError("resume_environment_changed", "Allocator configuration changed since original training")
    ids = contract["sample_ids"]
    by_id = {s.sample_id: s for s in index.samples}
    if len(set(ids)) != len(ids) or not set(ids).issubset(by_id):
        raise TrainingError("resume_selection_changed", "Saved sample selection is invalid")
    selected = [by_id[i] for i in ids]
    preflight = contract.get("preflight")
    if preflight is not None:
        verify_preflight(index, config, model, preflight)
    return _execute(index, config, model, selected, read_json(parent / "selection.json"), output,
                    preflight=preflight, checkpoint=checkpoint_info, parent=str(parent), dry_run=dry_run)


def _execute(index, config, model, selected, selection, output, *, preflight, checkpoint=None, parent=None, dry_run=False):
    destination = Path(output).resolve()
    protected = (index.root, Path(model["root"]))
    if destination.exists() or any(destination.is_relative_to(p) or p.is_relative_to(destination) for p in protected):
        raise TrainingError("invalid_output", "Output must be new and separate from data/model inputs")
    command = launch_command(config, model["root"], destination / "prepared", destination / "swift",
                             resume=checkpoint["path"] if checkpoint else None)
    if dry_run:
        return {"status": "dry_run", "output": str(destination), "argv": command,
                "resolved_config": config, "config_fingerprint": config_fingerprint(config),
                "samples": len(selected), "validation_level": "cpu_plan_validation"}
    require_tracking(config)
    tracking = tracking_plan(config, destination, parent=parent)
    probe = gpu_resource_probe(config)
    root = create_output(destination, protected=protected)
    run = {"status": "preparing", "started_at": now(), "validation_scope": "smoke_subset" if config["smoke"]["enabled"] else "full_dataset",
           "config_fingerprint": config_fingerprint(config),
           "validation_level": "gpu_training_pending", "parent_run": parent, "resume_checkpoint": checkpoint,
           "expected_start_step": checkpoint["global_step"] if checkpoint else 0,
           "startup_validation": {"encoding_preflight": "verified_report" if preflight is not None else "not_run",
                                  "loader_encoding": "not_run"}}
    write_json(root / "run.json", run)
    write_json(root / "wandb.json", {**tracking, "status": "planned" if tracking["enabled"] else "disabled"})
    write_json(root / "resolved_config.json", config)
    selection = {**selection, "selected_samples": len(selected), "unselected_samples": len(index.samples) - len(selected)}
    write_json(root / "selection.json", selection)
    recorded_environment = environment()
    recorded_environment["selected_gpus"] = probe.get("selected_gpus", [])
    recorded_environment["gpu_probe_pid"] = probe["pid"]
    write_json(root / "environment.json", recorded_environment)
    write_json(root / "gpu_probe.json", probe)
    write_json(root / "command.json", {"argv": command})
    journal(root, "run_preparing")
    try:
        prepared = materialize(index, root / "prepared", selected)
        write_json(root / "loader_validation.json", {"status": "not_run", "samples": 0,
                                                    "reason": "startup_full_encoding_removed"})
        torch = sys.modules.get("torch")
        recorded_environment["controller_cuda_initialized"] = bool(torch and torch.cuda.is_initialized())
        write_json(root / "environment.json", recorded_environment)
        contract = {"schema_version": 1, "config": config, "config_fingerprint": config_fingerprint(config),
                    "package": index.manifest(), "model": model,
                    "sample_ids": [s.sample_id for s in selected], "prepared_files": prepared["files"],
                    "preflight": preflight, "startup_validation": run["startup_validation"],
                    "packages": versions(), "encoder_source": encoder_source(),
                    "trainer_source": trainer_source(), "wandb": tracking,
                    "allocator_environment": {k: os.environ[k] for k in
                        ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF") if k in os.environ},
                    "expected_splits": prepared["splits"], "expected_start_step": run["expected_start_step"]}
        write_json(root / "training_contract.json", contract)
        env = tracking_environment(config, root, vision_environment(config))
        env["MANIP_SFT_RUN_ROOT"] = str(root)
        run.update(status="running")
        write_json(root / "run.json", run)
        run["child_return_code"] = run_process(command, root, env)
        target = config["smoke"]["stop_after_step"] or config["training"]["max_steps"]
        audit, verified = validate_training_output(root, start_step=run["expected_start_step"], target_step=target)
        index.assert_unchanged(images=True)
        if model_manifest(model["root"], require_weights=True) != model:
            raise ProcessFailure("model_changed", "Base model changed during training")
        run.update(status="awaiting_resume" if config["smoke"]["stop_after_step"] else "completed",
                   validation_level="real_gpu_lora_training", final_step=audit["end_step"], checkpoint=verified,
                   completed_at=now(), wandb=tracking_result(root))
        write_json(root / "selected_checkpoint.json", {"last": verified, "best": audit.get("best_model_checkpoint")})
        write_json(root / "run.json", run)
        journal(root, "run_finished", status=run["status"], global_step=audit["end_step"])
        return {**run, "output": str(root)}
    except BaseException as exc:
        run.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", completed_at=now(),
                   failure=exc.record() if isinstance(exc, ConversionError) else {"error_type": type(exc).__name__})
        if isinstance(exc, ProcessFailure) and "child_return_code" in exc.location:
            run["child_return_code"] = exc.location["child_return_code"]
        run["wandb"] = tracking_result(root)
        write_json(root / "run.json", run)
        journal(root, "run_failed", failure=run["failure"])
        raise
