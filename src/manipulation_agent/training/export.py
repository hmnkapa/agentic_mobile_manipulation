"""Snapshot and merge a managed adapter, then validate actual model loading."""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

from ..sft.common import ConversionError, file_hash, jsonl, read_json
from .artifacts import (ProcessFailure, TrainingError, create_output, environment, fingerprint,
                        journal, model_manifest, now, versions, write_json)
from .run import checkpoint_manifest, run_process, verify_checkpoint_binding
from .swift_backend import export_command, require_framework, vision_environment


def processor_identity(processor):
    tokenizer = processor.tokenizer
    settings = processor.image_processor.to_dict()
    return {"tokenizer_vocab": fingerprint(tokenizer.get_vocab()),
            "special_tokens": fingerprint(tokenizer.special_tokens_map),
            "chat_template": fingerprint(getattr(processor, "chat_template", None) or tokenizer.chat_template),
            "image_processor_class": type(processor.image_processor).__name__,
            "image_settings": fingerprint({k: v for k, v in settings.items() if not k.startswith("_")})}


def resolve_checkpoint(run, checkpoint):
    root = Path(run).resolve()
    record = read_json(root / "run.json")
    if record.get("status") not in {"completed", "awaiting_resume"} or record.get("validation_level") != "real_gpu_lora_training":
        raise TrainingError("unverified_training_run", "Export requires a verified real training run")
    if str(checkpoint) == "best":
        last = record.get("checkpoint", {}).get("path")
        best = read_json(Path(last) / "trainer_state.json").get("best_model_checkpoint") if last else None
        if not best:
            raise TrainingError("missing_best_checkpoint", "Trainer did not record a best checkpoint")
        checkpoint = Path(best)
        if not checkpoint.is_absolute():
            checkpoint = root / "swift" / checkpoint
    else:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = root / "swift" / checkpoint
    manifest = checkpoint_manifest(checkpoint)
    marker = read_json(Path(manifest["path"]) / "manipulation_training.json")
    origin = Path(marker["run_root"])
    contract = read_json(origin / "training_contract.json")
    requested = read_json(root / "training_contract.json")
    if (marker.get("contract_fingerprint") != fingerprint(contract)
            or contract["model"] != requested["model"] or contract["package"] != requested["package"]
            or contract["sample_ids"] != requested["sample_ids"]):
        raise TrainingError("export_checkpoint_mismatch", "Checkpoint is not bound to the requested run/base/data")
    verify_checkpoint_binding(manifest, marker, contract, origin)
    return manifest, requested


def export_model(run, checkpoint, output, *, dry_run=False):
    root = Path(output).resolve()
    source, contract = resolve_checkpoint(run, checkpoint)
    base = model_manifest(contract["model"]["root"], require_weights=True)
    if base != contract["model"] or (not dry_run and versions() != contract["packages"]):
        raise TrainingError("export_environment_changed", "Base model or dependencies differ from training")
    protected = (Path(base["root"]), Path(run).resolve(), Path(source["path"]), Path(contract["package"]["root"]))
    if root.exists() or any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise TrainingError("invalid_output", "Export directory must be new and separate from its inputs")
    command = export_command(base["root"], root / "checkpoint_snapshot", root / "model")
    if dry_run:
        return {"status": "dry_run", "argv": command, "output": str(root), "validation_level": "cpu_plan_validation"}
    require_framework()
    create_output(root, protected=protected)
    record = {"status": "running", "started_at": now(), "source_run": str(Path(run).resolve()),
              "checkpoint": source, "base_model": base, "validation_level": "merge_pending"}
    write_json(root / "export_manifest.json", record)
    write_json(root / "environment.json", environment())
    journal(root, "export_started")
    try:
        shutil.copytree(source["path"], root / "checkpoint_snapshot")
        snapshot = checkpoint_manifest(root / "checkpoint_snapshot")
        if snapshot["files"] != source["files"]:
            raise TrainingError("checkpoint_changed", "Checkpoint changed while snapshotting")
        env = vision_environment(contract["config"])
        env["CUDA_VISIBLE_DEVICES"] = ""
        write_json(root / "command.json", {"argv": command, "merge_device": "cpu"})
        record["merge_return_code"] = run_process(command, root, env, label="merge")
        merged = root / "model"
        if (not all((merged / p).is_file() for p in ("config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja"))
                or not any((merged / p).is_file() for p in ("preprocessor_config.json", "processor_config.json"))):
            raise ProcessFailure("incomplete_export", "Merged model lacks model/tokenizer/processor files")
        if not list(merged.glob("*.safetensors")):
            raise ProcessFailure("incomplete_export", "Merged model lacks safetensors weights")
        # Generation runs in a fresh process: the CPU merge must not hide GPUs
        # from its successor, or retain a second loaded base model in memory.
        validate = [sys.executable, "-m", "manipulation_agent.training.export", "--validate-model", str(merged),
                    "--run", str(Path(run).resolve()), "--result", str(root / "generation_validation.json")]
        record["generation_return_code"] = run_process(validate, root, vision_environment(contract["config"]), label="generation")
        try:
            result = read_json(root / "generation_validation.json")
        except ConversionError as exc:
            raise ProcessFailure("missing_generation_report", "Generation subprocess exited without its result") from exc
        if result.get("status") != "passed":
            raise ProcessFailure("export_generation_failed", "Merged-model loading/forward/generation check failed")
        record.update(status="complete", completed_at=now(), validation_level="merged_model_loading_forward_generation",
                      merged_model=model_manifest(merged), model_path=str(merged))
        write_json(root / "export_manifest.json", record)
        journal(root, "export_finished", status="complete")
        return {"status": "complete", "output": str(root), "model_path": str(merged), "validation_level": record["validation_level"]}
    except BaseException as exc:
        record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      completed_at=now(), failure=exc.record() if isinstance(exc, ConversionError) else {"error_type": type(exc).__name__})
        if isinstance(exc, ProcessFailure) and "child_return_code" in exc.location:
            record[exc.location.get("label", "child") + "_return_code"] = exc.location["child_return_code"]
        write_json(root / "export_manifest.json", record)
        journal(root, "export_failed", failure=record["failure"])
        raise


def validate_generation(model_path, run, result_path):
    import torch
    from .swift_backend import build_encoding_backend
    from .encoding import normalize
    root = Path(run)
    config = read_json(root / "resolved_config.json")
    result = {"status": "failed", "task_success_evaluated": False, "validation_level": "offline_model_generation", "records": []}
    try:
        backend = build_encoding_backend(config, model_path, unbounded=False)
        contract_path = root / "training_contract.json"
        reference_path = read_json(contract_path)["model"]["root"] if contract_path.is_file() else model_path
        reference = build_encoding_backend(config, reference_path, unbounded=False)
        actual_identity, reference_identity = processor_identity(backend.processor), processor_identity(reference.processor)
        if actual_identity != reference_identity:
            raise TrainingError("export_processor_mismatch", "Exported tokenizer/processor differs from the training base")
        result.update(processor_matched=True, processor_identity=actual_identity)
        del reference
        device = "cuda:0" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] >= 12 * 1024 ** 3 else "cpu"
        model, processor = backend.arguments.get_model_processor(load_model=True, device_map=device)
        model.eval()
        template = backend.arguments.get_template(processor)
        # The collator needs the actual model's multimodal position builder.
        # Leaving this unset asks swift to infer a new dummy model's type.
        template.model = model
        template.set_mode("transformers")
        rows = [row for _, row in jsonl(root / "prepared/val.jsonl")]
        chosen = [next((r for r in rows if not r["images"]), None),
                  min((r for r in rows if len(r["images"]) >= 2), key=lambda r: len(r["images"]), default=None)]
        if any(row is None for row in chosen):
            raise TrainingError("generation_coverage_gap", "Generation requires no-image and multi-image validation inputs")
        for original in chosen:
            row = deepcopy(original)
            row["messages"] = row["messages"][:-1]
            encoded = template.encode(normalize(row, backend.preprocessor))
            batch = template.data_collator([encoded])
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items() if k not in {"labels", "loss_scale"}}
            with torch.inference_mode():
                forward = model(**batch, logits_to_keep=1)
                finite = bool(torch.isfinite(forward.logits).all().item())
                del forward
                generated = model.generate(**batch, max_new_tokens=64, do_sample=False)
            suffix = generated[0, batch["input_ids"].shape[1]:].tolist()
            text = template.tokenizer.decode(suffix, skip_special_tokens=False)
            if not finite or not suffix or not text.strip():
                raise TrainingError("generation_failure", "Nonfinite forward or empty generated output")
            calls = template.agent_template.get_toolcall(text)
            result["records"].append({"images": len(row["images"]), "finite_forward": finite,
                                      "input_tokens": int(batch["input_ids"].shape[1]),
                                      "generated_tokens": len(suffix), "text": text,
                                      "parsed_tool_calls": [getattr(c, "__dict__", str(c)) for c in calls]})
            write_json(Path(result_path), result)
        result.update(status="passed", device=device)
    except BaseException as exc:
        result["failure"] = exc.record() if isinstance(exc, ConversionError) else {"error_type": type(exc).__name__}
        raise
    finally:
        write_json(Path(result_path), result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-model", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    validate_generation(args.validate_model, args.run, args.result)
