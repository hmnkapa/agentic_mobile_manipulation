"""Re-trim an exported train/val package using exact Qwen processor lengths."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import hashlib
from itertools import chain
import multiprocessing
import os
from pathlib import Path
import shutil

from .common import ConversionError, dumps, file_hash, jsonl, read_json
from .format import estimate_length
from .processor_budget import PROCESSOR_CONTEXT_POLICY, trim_record

_BACKEND = _MEASURE = _IMAGE_ROOT = None
_TRIM_FIELDS = ("processor_tokens_before", "processor_tokens", "processor_trim_triggered", "processor_slimmed_images",
                "processor_dropped_history_pairs", "processor_images_removed", "processor_over_budget",
                "processor_over_budget_reason", "processor_length_measurements")


def _initialize(config, model, image_root):
    global _BACKEND, _MEASURE, _IMAGE_ROOT
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from ..training.swift_backend import build_encoding_backend
    from ..training.token_length import QwenTokenLength
    _BACKEND = build_encoding_backend(config, model, unbounded=True)
    _IMAGE_ROOT = Path(image_root)
    _MEASURE = QwenTokenLength(_BACKEND, _IMAGE_ROOT)


def _process(job):
    from ..training.encoding import check_encoding
    sample, row, prior = job
    # Before any edits, require exact input-token identity with the sealed full
    # processor scan. Grid expansion is not a character/image-token estimate.
    before = _MEASURE(row)
    if before != prior["tokens"] or _MEASURE.input_fingerprint(row) != prior["input_ids_sha256"]:
        raise ConversionError("length_backend_mismatch", "Native length expansion differs from the prior processor input",
                              sample_id=sample["sample_id"])
    trimmed, stats = trim_record(row, _MEASURE, max_length=sample["max_length"], k=sample["k"])
    actual = deepcopy(trimmed)
    actual["images"] = [str(_IMAGE_ROOT / relative) for relative in actual["images"]]
    result = check_encoding(actual, _BACKEND.template, _BACKEND.processor,
                            preprocessor=_BACKEND.preprocessor, render_target=_BACKEND.render_target)
    if (result["tokens"] != stats["processor_tokens"]
            or result["input_ids_sha256"] != _MEASURE.input_fingerprint(trimmed)
            or result["loss_tokens"] != prior["loss_tokens"]):
        raise ConversionError("final_processor_mismatch", "Full processor disagrees with trimming or target supervision",
                              sample_id=sample["sample_id"])
    return trimmed, stats, {**result, "sample_id": sample["sample_id"],
                           "source_id": sample["source_id"], "split": sample["split"], "call_id": sample["call_id"],
                           "before_tokens": before, "before_images": len(row["images"]),
                           "before_image_tokens": prior["image_tokens"],
                           "before_input_ids_sha256": prior["input_ids_sha256"],
                           "before_exact_token_ids_verified": True, **stats}


def _ordered_jobs(pool, jobs, *, limit):
    pending, jobs = deque(), iter(jobs)
    for _ in range(limit):
        job = next(jobs, None)
        if job is None:
            break
        pending.append((job[0], pool.submit(_process, job)))
    while pending:
        sample, future = pending.popleft()
        yield sample, future.result()
        job = next(jobs, None)
        if job is not None:
            pending.append((job[0], pool.submit(_process, job)))


def retrim(data, model, baseline_preflight, output, *, max_length=17408, workers=1, provenance_root=None,
           resume_prefix=None, recipe=None):
    from ..records import source_version
    from ..training.artifacts import (encoder_source, environment, fingerprint, journal,
                                     model_manifest, write_json)
    from ..training.config import encoding_contract, load_config
    from ..training.data import index_package
    from ..training.preflight import distribution

    if type(max_length) is not int or max_length < 1 or type(workers) is not int or not 1 <= workers <= 2:
        raise ConversionError("invalid_processor_budget", "Use a positive budget and one or two CPU workers")
    index = index_package(data)
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(index.root) or index.root.is_relative_to(output):
        raise ConversionError("invalid_output", "Choose a new output separate from the input package")
    baseline = Path(baseline_preflight).resolve()
    summary, saved = read_json(baseline / "summary.json"), read_json(baseline / "encoding_contract.json")
    manifest = read_json(baseline / "package_manifest.json")
    if (summary.get("status") != "complete" or summary.get("encoding_passed") is not True
            or summary.get("scan_complete") is not True or summary.get("samples") != len(index.samples)
            or saved["contract"]["package"] != index.fingerprint or manifest["files"] != index.files
            or saved["fingerprint"] != fingerprint(saved["contract"])):
        raise ConversionError("baseline_mismatch", "Need a sealed full real-encoding report for this exact package")
    for name, digest in summary["report_files"].items():
        if file_hash(baseline / name) != digest:
            raise ConversionError("baseline_changed", "A baseline report file changed", file=name)
    config = load_config(recipe)
    if index.config.get("history_compression", "legacy") != config["data"].get("history_compression", "legacy"):
        raise ConversionError("history_compression_mismatch", "Recipe and package history compression policies differ")
    config["data"]["max_length"] = max_length
    env, model_info = environment(), model_manifest(model)
    contract = saved["contract"]
    if (contract["encoding"] != encoding_contract(config) or contract["packages"] != env["packages"]
            or contract["model"] != model_info["fingerprint"]):
        raise ConversionError("encoding_contract_changed", "Model, template, vision settings or dependencies differ from baseline")
    priors = {r["sample_id"]: r for _, r in jsonl(baseline / "samples.jsonl")}
    if len(priors) != len(index.samples):
        raise ConversionError("invalid_baseline", "The real-encoding report contains duplicate or missing samples")
    metadata = [m for _, m in jsonl(index.root / "sample_sources.jsonl")]
    cached_count, cached_hashes = 0, {}
    if resume_prefix:
        resume_prefix = Path(resume_prefix).resolve()
        previous = read_json(resume_prefix / "length_contract.json")
        if (read_json(resume_prefix / "build.json")["status"] != "failed"
                or previous["max_length"] != max_length or previous["k"] != index.config["k"]
                or previous["encoding"] != encoding_contract(config)
                or previous["model"]["fingerprint"] != model_info["fingerprint"]
                or previous["environment"]["packages"] != env["packages"]
                or previous["encoder_source"] != encoder_source()
                or previous["parent_package_fingerprint"] != index.fingerprint):
            raise ConversionError("resume_prefix_mismatch", "Cached prefix data/model/template/dependencies/encoder differ")
        for name in ("train.jsonl", "val.jsonl", "sample_sources.jsonl", "token_lengths.jsonl"):
            path = resume_prefix / (name + ".incomplete")
            cached_hashes[name] = file_hash(path)
        cached_count = sum(1 for _ in jsonl(resume_prefix / "token_lengths.jsonl.incomplete"))
        if not 0 < cached_count <= len(index.samples):
            raise ConversionError("invalid_resume_prefix", "Cached prefix count is invalid")
    output.mkdir(parents=True)
    write_json(output / "build.json", {"status": "running", "operation": "real_processor_token_trim"})
    journal(output, "processor_trim_started", samples=len(index.samples), max_length=max_length, workers=workers)
    write_json(output / "length_contract.json", {
        "schema_version": 1, "method": "native_template_tokens_and_real_processor_grid_expansion",
        "full_processor_validation": "every_output_row_including_counterfactual_loss_mask",
        "max_length": max_length, "k": index.config["k"], "encoding": encoding_contract(config),
        "recipe": str(Path(recipe).resolve()) if recipe else None,
        "recipe_sha256": file_hash(recipe) if recipe else None,
        "model": model_info, "environment": env, "encoder_source": encoder_source(),
        "trimming_source": source_version(), "baseline_report": str(baseline),
        "baseline_report_files": summary["report_files"], "parent_package_fingerprint": index.fingerprint,
        "parent_package_files": index.files, "weights_loaded": False, "gpu_used": False,
        "resume_prefix": str(resume_prefix) if resume_prefix else None,
        "resume_prefix_file_hashes": cached_hashes, "reused_verified_samples": cached_count})
    staged = ["train.jsonl", "val.jsonl", "sample_sources.jsonl", "token_lengths.jsonl"]
    lengths_before, lengths_after, approx_before, approx_after = [], [], [], []
    totals, used_images, exceptions, grouped = Counter(), set(), [], {}

    def jobs():
        for i, ((sample, row), meta) in enumerate(zip(index.records(), metadata, strict=True)):
            if i < cached_count:
                continue
            prior = priors[sample.sample_id]
            if (prior.get("status") != "passed" or prior["sha256"] != sample.sha256
                    or prior["all_history_masked"] is not True or prior["complete_target_verified"] is not True
                    or prior["weights_loaded"] is not False):
                raise ConversionError("invalid_baseline_sample", "Baseline row lacks genuine matching processor evidence")
            yield {**asdict(sample), "max_length": max_length, "k": index.config["k"]}, row, prior

    def cached_results():
        if not resume_prefix:
            return
        with ExitStack() as stack:
            streams = {s: stack.enter_context((resume_prefix / (s + ".jsonl.incomplete")).open("rb"))
                       for s in ("train", "val")}
            cached_meta = jsonl(resume_prefix / "sample_sources.jsonl.incomplete")
            cached_tokens = jsonl(resume_prefix / "token_lengths.jsonl.incomplete")
            for i, ((_, meta), (_, encoded)) in enumerate(zip(cached_meta, cached_tokens, strict=True)):
                sample = index.samples[i]
                prior = priors[sample.sample_id]
                raw = streams[sample.split].readline()
                from .common import loads
                row = loads(raw)
                original = index.record(sample)
                if (encoded["status"] != "passed" or encoded["sample_id"] != sample.sample_id
                        or meta["source_id"] != sample.source_id or meta["call_id"] != sample.call_id
                        or meta["split"] != sample.split or hashlib.sha256(raw).hexdigest() != encoded["row_sha256"]
                        or encoded["before_input_ids_sha256"] != prior["input_ids_sha256"]
                        or encoded["before_tokens"] != prior["tokens"] or encoded["loss_tokens"] != prior["loss_tokens"]
                        or encoded["all_history_masked"] is not True or encoded["complete_target_verified"] is not True
                        or encoded["before_exact_token_ids_verified"] is not True or encoded["weights_loaded"] is not False
                        or row["messages"][:2] != original["messages"][:2]
                        or row["messages"][-1] != original["messages"][-1] or row["tools"] != original["tools"]
                        or len(row["images"]) != encoded["images"]):
                    raise ConversionError("invalid_resume_prefix", "Cached row or genuine encoding proof differs", sample_id=sample.sample_id)
                trim = {key: encoded[key] for key in _TRIM_FIELDS}
                yield {**asdict(sample), "max_length": max_length, "k": index.config["k"]}, (row, trim, encoded)
            if any(stream.read(1) for stream in streams.values()):
                raise ConversionError("invalid_resume_prefix", "Cached dataset has rows without encoding proof")
        for name, digest in cached_hashes.items():
            if file_hash(resume_prefix / (name + ".incomplete")) != digest:
                raise ConversionError("resume_prefix_changed", "A sealed cached prefix file changed")

    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                               initializer=_initialize, initargs=(config, str(model), str(index.root)),
                               max_tasks_per_child=200)
    try:
        with ExitStack() as stack:
            handles = {name: stack.enter_context((output / (name + ".incomplete")).open("x", encoding="utf-8"))
                       for name in staged}
            results = chain(cached_results(), _ordered_jobs(pool, jobs(), limit=workers * 2))
            for i, (sample, (row, trim, encoded)) in enumerate(results):
                meta = metadata[i]
                before_approx = {k: meta[k] for k in ("approx_tokens", "image_count")}
                after_approx = estimate_length(row)
                extra = {**meta, **after_approx, **trim,
                         "parent_length_control": {k: meta[k] for k in (
                             "approx_tokens_before", "image_count_before", "approx_tokens", "image_count",
                             "slim_triggered", "slimmed_images", "dropped_history_pairs", "images_removed", "over_budget")},
                         "approx_tokens_before": before_approx["approx_tokens"],
                         "image_count_before": before_approx["image_count"],
                         "slim_triggered": trim["processor_trim_triggered"],
                         "slimmed_images": trim["processor_slimmed_images"],
                         "dropped_history_pairs": trim["processor_dropped_history_pairs"],
                         "images_removed": trim["processor_images_removed"], "over_budget": trim["processor_over_budget"]}
                line = dumps(row) + "\n"
                handles[sample["split"] + ".jsonl"].write(line)
                handles["sample_sources.jsonl"].write(dumps(extra) + "\n")
                encoded["row_sha256"] = hashlib.sha256(line.encode()).hexdigest()
                handles["token_lengths.jsonl"].write(dumps(encoded) + "\n")
                lengths_before.append(trim["processor_tokens_before"])
                lengths_after.append(trim["processor_tokens"])
                approx_before.append(before_approx["approx_tokens"])
                approx_after.append(after_approx["approx_tokens"])
                used_images.update(row["images"])
                totals["trimmed_samples"] += trim["processor_trim_triggered"]
                totals["slimmed_samples"] += trim["processor_slimmed_images"] > 0
                totals["truncated_samples"] += trim["processor_dropped_history_pairs"] > 0
                totals["over_budget_samples"] += trim["processor_over_budget"]
                totals["image_occurrences_before"] += before_approx["image_count"]
                totals["image_occurrences_after"] += after_approx["image_count"]
                totals["image_tokens_before"] += encoded["before_image_tokens"]
                totals["image_tokens_after"] += encoded["image_tokens"]
                for key in ("processor_slimmed_images", "processor_dropped_history_pairs", "processor_images_removed"):
                    totals[key] += trim[key]
                for key in ("split", "target_tool", "scene"):
                    group = key + ":" + str(sample[key])
                    stats = grouped.setdefault(group, {"before": [], "after": [], "trimmed": 0, "over_budget": 0})
                    stats["before"].append(trim["processor_tokens_before"])
                    stats["after"].append(trim["processor_tokens"])
                    stats["trimmed"] += trim["processor_trim_triggered"]
                    stats["over_budget"] += trim["processor_over_budget"]
                if trim["processor_over_budget"]:
                    exceptions.append({k: sample[k] for k in ("sample_id", "source_id", "split", "task", "scene", "target_tool", "primitive", "call_id")} | trim)
                if (i + 1) % 25 == 0:
                    for handle in handles.values():
                        handle.flush()
                    journal(output, "processor_trim_progress", samples=i + 1, trimmed=totals["trimmed_samples"],
                            over_budget=totals["over_budget_samples"])
                    print(dumps({"samples": i + 1, "total": len(index.samples), "trimmed": totals["trimmed_samples"],
                                 "over_budget": totals["over_budget_samples"]}), flush=True)
        pool.shutdown(wait=True)
        index.assert_unchanged(images=True)
        output.joinpath("images").mkdir()
        with ThreadPoolExecutor(max_workers=8) as copies:
            def copy_image(relative):
                target = output / relative
                shutil.copyfile(index.root / relative, target)
                if file_hash(target) != index.images[relative]["sha256"]:
                    raise ConversionError("copied_image_changed", "Copied image differs from its verified source")
            list(copies.map(copy_image, sorted(used_images)))
        (output / "images.jsonl").write_text("".join(dumps(index.images[p]) + "\n" for p in sorted(used_images)))
        for name in ("sources.jsonl", "errors.jsonl", "skipped.jsonl"):
            shutil.copyfile(index.root / name, output / name)
        for name in ("compression_contract.json", "compression_report.json"):
            if (index.root / name).is_file():
                shutil.copyfile(index.root / name, output / name)
        for name in staged:
            (output / (name + ".incomplete")).rename(output / name)
        report = {"status": "complete", "samples": len(index.samples), "splits": index.stats["splits"],
                  "trajectories": len(index.sources), "max_length": max_length, "k": index.config["k"],
                  "reused_verified_samples": cached_count,
                  "tokens_before": distribution(lengths_before), "tokens_after": distribution(lengths_after),
                  "total_tokens_before": sum(lengths_before), "total_tokens_after": sum(lengths_after),
                  **dict(totals), "unique_images_before": len(index.images), "unique_images_after": len(used_images),
                  "untrimmable_samples": exceptions,
                  "groups": {key: {"count": len(g["before"]), "before": distribution(g["before"]),
                                   "after": distribution(g["after"]), "trimmed": g["trimmed"], "over_budget": g["over_budget"]}
                             for key, g in grouped.items()},
                  "validation_level": "qwen_processor_encoding_and_counterfactual_loss_validation",
                  "all_original_input_token_ids_verified": True, "all_output_rows_full_processor_verified": True,
                  "full_dataset_within_budget": totals["over_budget_samples"] == 0, "weights_loaded": False, "gpu_used": False}
        package_config = {**index.config, "context_policy": PROCESSOR_CONTEXT_POLICY, "max_len": max_length,
                          "keep_all_images": False, "length_measurement": "qwen_processor_tokens",
                          "processor_trim_version": "manipulation_sft_processor_trim_v1",
                          "processor_encoding_contract": encoding_contract(config),
                          "processor_model_fingerprint": model_info["fingerprint"],
                          "length_contract_sha256": file_hash(output / "length_contract.json"),
                          "token_lengths_sha256": file_hash(output / "token_lengths.jsonl"),
                          "parent_package": {"root": str(provenance_root or index.root), "fingerprint": index.fingerprint,
                                             "context_policy": index.config["context_policy"], "files": index.files}}
        stats = {**index.stats, "unique_images": len(used_images), "qwen_encoding_verified": True,
                 "validation_level": report["validation_level"],
                 "approx_tokens": distribution(approx_after), "approx_tokens_before": distribution(approx_before),
                 "processor_tokens": report["tokens_after"], "processor_tokens_before": report["tokens_before"],
                 "length_control": {"context_policy": PROCESSOR_CONTEXT_POLICY, "max_len": max_length,
                                    "length_measurement": "qwen_processor_tokens", **dict(totals)}}
        write_json(output / "config.json", package_config)
        write_json(output / "stats.json", stats)
        write_json(output / "trimming_report.json", report)
        write_json(output / "build.json", {"status": "complete", "converter_version": index.config["converter_version"],
                                          "processor_trim_version": "manipulation_sft_processor_trim_v1"})
        journal(output, "processor_trim_complete", samples=len(index.samples), over_budget=totals["over_budget_samples"])
        return report
    except BaseException as exc:
        worker_exits = {str(p.pid): p.exitcode for p in pool._processes.values()} if pool._processes else {}
        for process in pool._processes.values() if pool._processes is not None else ():
            if process.is_alive():
                process.terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        error = exc.record() if isinstance(exc, ConversionError) else {"error_type": type(exc).__name__}
        error["worker_exit_codes"] = worker_exits
        write_json(output / "build.json", {"status": "failed", "error": error})
        journal(output, "processor_trim_failed", error=error)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline-preflight", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, help="Use the same model/template/vision recipe as the baseline preflight")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=17408)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--provenance-root", type=Path)
    parser.add_argument("--resume-prefix", type=Path, help="Closed failed run with a fully verified output prefix")
    args = parser.parse_args(argv)
    result = retrim(args.data, args.model, args.baseline_preflight, args.out_dir,
                    max_length=args.max_length, workers=args.workers, provenance_root=args.provenance_root,
                    resume_prefix=args.resume_prefix, recipe=args.recipe)
    print(dumps({k: v for k, v in result.items() if k not in {"groups", "untrimmable_samples"}}), flush=True)


if __name__ == "__main__":
    main()
