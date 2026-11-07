"""Offline next-tool-call evaluation of immutable SFT packages."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math
import os
from pathlib import Path
import re
import sys
import time

from ..sft.common import ConversionError, file_hash, read_json
from .artifacts import (ProcessFailure, TrainingError, canonical, create_output, encoder_source,
                        environment, fingerprint, journal, model_manifest, now, write_json)
from .config import encoding_contract, load_config, validate
from .data import index_package
from .rollout_metrics import parse_tool_call, score_call, summarize, target_call, validate_call


def _adapter_manifest(root):
    required = ("adapter_config.json", "adapter_model.safetensors")
    if not all((root / name).is_file() for name in required):
        raise TrainingError("incomplete_adapter", "Inference requires adapter_config.json and adapter_model.safetensors")
    names = (*required, "manipulation_training.json", "tokenizer.json", "tokenizer_config.json",
             "preprocessor_config.json", "processor_config.json", "chat_template.jinja", "special_tokens_map.json")
    files = {name: file_hash(root / name) for name in names if (root / name).is_file()}
    return {"root": str(root), "files": files, "fingerprint": fingerprint(files)}


def _origin_contract(record, field, *, artifact_root=None):
    origin = record.get(field)
    if not isinstance(origin, str) or not origin:
        raise TrainingError("invalid_checkpoint_metadata", "Checkpoint metadata lacks its training run path")
    path = Path(origin) / "training_contract.json"
    candidates = [path]
    if artifact_root is not None:
        # A deployment may copy the sealed run without reproducing its source
        # machine's absolute directory. Preserve the marker bytes/hash and look
        # only in the documented artifact bundle, never scan storage roots.
        root = Path(artifact_root)
        candidates.extend([root / 'training_contract.json', root.parent / 'training_contract.json'])
        if root.parent.name == 'swift':
            candidates.append(root.parent.parent / 'training_contract.json')
    for candidate in candidates:
        try:
            present = candidate.is_file()
        except PermissionError:
            # Copied checkpoints retain paths such as /root/... from their
            # source host. Treat an inaccessible historical path as unavailable;
            # local contracts and explicit inference settings remain usable.
            continue
        if present:
            return read_json(candidate), candidate
    return None, path


def resolve_inputs(checkpoint, *, model=None, recipe=None):
    """Resolve inference files/configuration without loading optimizer or RNG state."""
    root = Path(checkpoint).resolve()
    if not root.is_dir():
        raise TrainingError("missing_checkpoint", "Provide an existing checkpoint directory", path=str(root))
    is_adapter = (root / "adapter_config.json").is_file()
    identity = _adapter_manifest(root) if is_adapter else model_manifest(root, require_weights=True)
    contract, metadata = None, {"configuration_source": "explicit_recipe", "checkpoint_binding": "not_available"}
    marker = root / "manipulation_training.json"
    exported = root.parent / "export_manifest.json"
    if is_adapter and marker.is_file():
        saved = read_json(marker)
        contract, path = _origin_contract(saved, "run_root", artifact_root=root)
        metadata["training_contract"] = str(path)
        if contract is not None:
            if saved.get("contract_fingerprint") != fingerprint(contract):
                raise TrainingError("checkpoint_contract_mismatch", "Checkpoint training contract hash differs")
            entries = read_json(path.parent / "checkpoints.json").get("checkpoints", [])
            if not any(all(entry.get("files", {}).get(k) == v for k, v in identity["files"].items()) for entry in entries):
                raise TrainingError("checkpoint_changed", "Inference files differ from the sealed checkpoint")
            metadata["checkpoint_binding"] = "sealed_inference_files"
    elif not is_adapter and exported.is_file():
        saved = read_json(exported)
        if saved.get("status") != "complete" or saved.get("merged_model", {}).get("fingerprint") != identity["fingerprint"]:
            raise TrainingError("export_model_mismatch", "Export is incomplete or merged model files differ")
        contract, path = _origin_contract(saved, "source_run", artifact_root=root)
        metadata.update(export_manifest=str(exported), training_contract=str(path),
                        checkpoint_binding="exported_model_fingerprint")
    if contract is not None:
        try:
            config = validate(deepcopy(contract["config"]))
        except (KeyError, TypeError) as exc:
            raise TrainingError("invalid_checkpoint_metadata", "Training contract lacks a supported configuration") from exc
        metadata["configuration_source"] = "training_contract"
        if recipe is not None:
            explicit = load_config(recipe)
            if (encoding_contract(explicit) != encoding_contract(config)
                    or explicit["data"]["max_length"] != config["data"]["max_length"]):
                raise TrainingError("rollout_config_mismatch", "Explicit recipe conflicts with checkpoint encoding settings")
    else:
        if recipe is None:
            raise TrainingError("missing_inference_recipe", "Without training metadata provide --recipe explicitly")
        config = load_config(recipe)
    if is_adapter:
        if model is None and contract is None:
            raise TrainingError("missing_base_model", "An adapter without training metadata requires --model")
        if model is None:
            try:
                model = contract["model"]["root"]
            except (KeyError, TypeError) as exc:
                raise TrainingError("invalid_checkpoint_metadata", "Training contract lacks its base model") from exc
        base = model_manifest(model, require_weights=True)
        if contract is not None:
            expected = contract.get("model", {})
            if any(base[k] != expected.get(k) for k in ("files", "weights")):
                raise TrainingError("checkpoint_base_mismatch", "Base model bytes differ from the training contract")
        adapter_config = read_json(root / "adapter_config.json")
        if adapter_config.get("peft_type") != "LORA":
            raise TrainingError("unsupported_adapter", "Offline rollout supports PEFT LoRA adapters")
    else:
        if model is not None and Path(model).resolve() != root:
            raise TrainingError("unexpected_base_model", "A full checkpoint is already the model; omit --model")
        base = identity
    return {"kind": "lora" if is_adapter else "full", "checkpoint": identity,
            "model": base, "config": config, "metadata": metadata}


def _output_path(output, protected):
    path = Path(output).resolve()
    if path.exists():
        raise TrainingError("output_exists", "Choose a new output directory", path=str(path))
    # Some sandboxes mount empty .git protection directories in /tmp. A real
    # repository has HEAD, or a gitdir file for a linked worktree.
    if any((parent / ".git").is_file() or (parent / ".git/HEAD").is_file() for parent in (path, *path.parents)):
        raise TrainingError("output_inside_git", "Generated evaluations must be outside Git repositories", path=str(path))
    for source in protected:
        source = Path(source).resolve()
        if path.is_relative_to(source) or source.is_relative_to(path):
            raise TrainingError("output_overlaps_input", "Evaluation output must be separate from its inputs")
    return path


def _source_manifest():
    root = Path(__file__).parent
    return encoder_source() | {"training/" + name: file_hash(root / name) for name in
                              ("rollout.py", "rollout_backend.py", "rollout_metrics.py", "cli.py", "artifacts.py")}


def build_prompt(record):
    prompt = deepcopy(record)
    prompt["messages"].pop()
    return prompt


def _summary(rows, planned, status, validation_level, output, point_tolerance):
    return {"schema_version": 1, "status": status, "evaluation": "offline_next_tool_call",
            "task_success_evaluated": False, "validation_level": validation_level,
            "complete": status == "complete", "metrics_scope": "selected_samples" if status == "complete" else "processed_only",
            "planned_samples": planned, "processed_samples": len(rows), "coverage": len(rows) / planned,
            "point_tolerance": point_tolerance, "output": str(output), **summarize(rows)}


def rollout(data, checkpoint, output, *, model=None, recipe=None, split="val", limit=None,
            device="cuda:0", max_new_tokens=2048, point_tolerance=50.0, dry_run=False):
    if (split not in {"train", "val", "all"} or (limit is not None and (type(limit) is not int or limit <= 0))
            or type(max_new_tokens) is not int or max_new_tokens <= 0
            or type(point_tolerance) not in (int, float) or not math.isfinite(point_tolerance) or point_tolerance < 0
            or not re.fullmatch(r"cpu|cuda:[0-9]+", device)):
        raise TrainingError("invalid_rollout_options", "Use a valid split, positive limits, nonnegative finite tolerance, and cpu/cuda:N")
    inputs = resolve_inputs(checkpoint, model=model, recipe=recipe)
    config = inputs["config"]
    index = index_package(data, expected_k=config["data"]["k"],
                          expected_history_compression=config["data"].get("history_compression", "legacy"))
    processor_contract = index.config.get("processor_encoding_contract")
    if processor_contract is not None and processor_contract != encoding_contract(config):
        raise TrainingError("package_encoding_mismatch", "Package real-token trimming uses different encoding settings")
    selected = sorted((s for s in index.samples if split == "all" or s.split == split),
                      key=lambda s: (s.split != "train", s.line))
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise TrainingError("empty_evaluation", "Selected split contains no evaluation samples")
    protected = [index.root, inputs["checkpoint"]["root"], inputs["model"]["root"]]
    if "training_contract" in inputs["metadata"]:
        protected.append(Path(inputs["metadata"]["training_contract"]).parent)
    root = _output_path(output, protected)
    selection = {"split": split, "limit": limit, "samples": len(selected), "sample_ids": [s.sample_id for s in selected]}
    settings = {"device": device, "max_new_tokens": max_new_tokens, "point_tolerance": point_tolerance,
                "do_sample": False, "batch_size": 1, "seed": 42}
    manifest = {"inputs": inputs, "package": index.manifest(), "selection": selection,
                "generation": settings, "source": _source_manifest()}
    if dry_run:
        return {"status": "dry_run", "validation_level": "cpu_plan_validation", "task_success_evaluated": False,
                "weights_loaded": False, "encoding_validated": False, "output": str(root), **manifest}

    from .rollout_backend import GenerationBackend
    create_output(root, protected=protected)
    rows, backend = [], None
    level = "offline_generation_pending"
    run = {"schema_version": 1, "status": "running", "started_at": now(), "evaluation": "offline_next_tool_call",
           "task_success_evaluated": False, "input_fingerprint": fingerprint(manifest)}
    try:
        write_json(root / "run.json", run)
        write_json(root / "config.json", {"encoding": config, "generation": settings})
        write_json(root / "input_manifest.json", manifest)
        write_json(root / "environment.json", environment())
        (root / "predictions.jsonl").touch(exist_ok=False)
        journal(root, "rollout_started", planned_samples=len(selected))
        with (root / "selection.jsonl").open("x", encoding="utf-8") as stream:
            for sample in selected:
                stream.write(canonical(asdict(sample)) + "\n")
        write_json(root / "summary.json", _summary(rows, len(selected), "running", level, root, point_tolerance))
        backend = GenerationBackend(config, inputs["model"]["root"],
                                    inputs["checkpoint"]["root"] if inputs["kind"] == "lora" else None,
                                    device, max_new_tokens)
        level = "model_loaded_no_generation"
        run["backend"] = backend.describe()
        write_json(root / "environment.json", environment())
        write_json(root / "run.json", run)
        with (root / "predictions.jsonl").open("a", encoding="utf-8") as stream:
            for sample in selected:
                record = index.record(sample, absolute_images=True)
                gold = target_call(record)
                prediction = {"sample": asdict(sample), "gold": gold, "prediction": None,
                              "raw_response": None, "input_tokens": None, "generated_tokens": 0,
                              "finish_reason": None, "error": None}
                started, fatal, valid = time.monotonic(), None, False
                journal(root, "sample_started", sample_id=sample.sample_id)
                try:
                    generated = backend.generate(build_prompt(record))
                    prediction.update(generated)
                    if generated["generated_tokens"]:
                        level = backend.validation_level
                    if not generated["raw_response"].strip() or not generated["generated_tokens"]:
                        raise TrainingError("empty_generation", "Model produced no continuation")
                    if generated["finish_reason"] == "length":
                        try:
                            prediction["prediction"] = parse_tool_call(generated["raw_response"], record["tools"])
                        except ConversionError:
                            pass
                        raise TrainingError("generation_truncated", "Continuation exhausted max_new_tokens")
                    prediction["prediction"] = parse_tool_call(generated["raw_response"], record["tools"])
                    validate_call(prediction["prediction"], record["tools"])
                    valid = True
                except ConversionError as exc:
                    prediction["error"] = exc.record()
                    if prediction["input_tokens"] is None:
                        prediction["input_tokens"] = exc.location.get("input_tokens")
                    if isinstance(exc, ProcessFailure):
                        fatal = exc
                prediction["elapsed_seconds"] = time.monotonic() - started
                prediction["metrics"] = score_call(gold, prediction["prediction"], valid=valid,
                                                    point_tolerance=point_tolerance)
                stream.write(canonical(prediction) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                # Keep only scoring data in memory; raw generations stay on disk.
                rows.append({key: prediction[key] for key in ("gold", "metrics", "error")})
                journal(root, "sample_finished", sample_id=sample.sample_id,
                        error=prediction["error"], elapsed_seconds=prediction["elapsed_seconds"])
                write_json(root / "summary.json", _summary(rows, len(selected), "running", level, root, point_tolerance))
                if fatal is not None:
                    raise fatal
                if prediction["error"]:
                    backend.recover()
                print(f"rollout {len(rows)}/{len(selected)} strict={sum(r['metrics']['strict_match'] for r in rows)}/{len(rows)}",
                      file=sys.stderr, flush=True)
        index.assert_unchanged(images=True)
        current = (_adapter_manifest(Path(inputs["checkpoint"]["root"])) if inputs["kind"] == "lora"
                   else model_manifest(inputs["checkpoint"]["root"], require_weights=True))
        if (current != inputs["checkpoint"]
                or (inputs["kind"] == "lora" and model_manifest(inputs["model"]["root"], require_weights=True) != inputs["model"])
                or _source_manifest() != manifest["source"]):
            raise ProcessFailure("evaluation_inputs_changed", "Checkpoint, base model or evaluator source changed during evaluation")
        run.update(status="complete", completed_at=now(), validation_level=level)
        journal(root, "rollout_finished", processed_samples=len(rows), status="complete")
        summary = _summary(rows, len(selected), "complete", level, root, point_tolerance)
        summary["report_files"] = {name: file_hash(root / name) for name in
                                   ("predictions.jsonl", "selection.jsonl", "config.json", "environment.json", "input_manifest.json")}
        write_json(root / "run.json", run)
        write_json(root / "summary.json", summary)
        for key in ("strict_accuracy", "tolerant_accuracy"):
            metric = summary["metrics"][key]
            print(f"offline {key}: {metric['correct']}/{metric['total']} ({metric['accuracy']:.2%}); task success not evaluated",
                  file=sys.stderr)
        return summary
    except BaseException as exc:
        error = exc.record() if isinstance(exc, ConversionError) else {"code": "interrupted" if isinstance(exc, KeyboardInterrupt)
                                                                     else "rollout_failed", "error_type": type(exc).__name__}
        run.update(status="incomplete", completed_at=now(), validation_level=level, failure=error)
        journal(root, "rollout_incomplete", error=error, processed_samples=len(rows))
        summary = _summary(rows, len(selected), "incomplete", level, root, point_tolerance)
        summary["failure"] = error
        write_json(root / "run.json", run)
        write_json(root / "summary.json", summary)
        if isinstance(exc, (ConversionError, KeyboardInterrupt)):
            raise
        raise ProcessFailure("rollout_failed", "Offline evaluation did not finish", error_type=type(exc).__name__) from exc
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:
                run["cleanup_error"] = {"error_type": type(exc).__name__}
                journal(root, "cleanup_failed", **run["cleanup_error"])
                write_json(root / "run.json", run)
