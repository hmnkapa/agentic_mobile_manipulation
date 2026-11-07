"""Stream a complete package through real training preprocessing and encoding."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict
import multiprocessing
from pathlib import Path

from ..sft.common import ConversionError, file_hash, jsonl, read_json
from .artifacts import (TrainingError, canonical, create_output, environment, fingerprint,
                        journal, model_manifest, write_json, encoder_source)
from .config import encoding_contract
from .data import index_package, materialize
from .encoding import check_encoding
from .swift_backend import build_encoding_backend, load_prepared_datasets


_WORKER_BACKEND = None
_ENCODE_FIELDS = ("status", "tokens", "loss_tokens", "masked_tokens", "images", "image_tokens",
                  "input_ids_sha256", "labels_sha256", "all_history_masked", "complete_target_verified", "weights_loaded")


def seal_encoding_cache(root):
    """Seal closed per-sample results, including a prefix of an interrupted scan."""
    root = Path(root).resolve()
    if not (root / "samples.jsonl").is_file() or not (root / "encoding_contract.json").is_file():
        return
    summary = read_json(root / "summary.json")
    if summary.get("status") == "running" or summary.get("structure_only") is not False:
        raise TrainingError("unsealable_encoding_cache", "Only closed real-encoding reports can be cached")
    saved = read_json(root / "encoding_contract.json")
    if fingerprint(saved["contract"]) != saved["fingerprint"]:
        raise TrainingError("invalid_encoding_cache", "Encoding contract fingerprint is invalid")
    count = sum(r.get("status") == "passed" for _, r in jsonl(root / "samples.jsonl"))
    write_json(root / "encoding_cache.json", {"schema_version": 1, "contract_fingerprint": saved["fingerprint"],
               "samples_sha256": file_hash(root / "samples.jsonl"), "passed_samples": count,
               "source_scan_complete": summary.get("scan_complete") is True})


def load_encoding_cache(path, contract, index):
    if path is None:
        return {}
    root = Path(path).resolve()
    source_summary = read_json(root / "summary.json")
    if (source_summary.get("validation_level") != "qwen_processor_encoding"
            or source_summary.get("structure_only") is not False
            or source_summary.get("status") not in {"complete", "interrupted", "failed"}):
        raise TrainingError("unverified_encoding_cache", "Cache must come from a closed real-processor scan")
    manifest = read_json(root / "encoding_cache.json")
    saved = read_json(root / "encoding_contract.json")
    if (manifest.get("schema_version") != 1 or saved.get("contract") != contract
            or saved.get("fingerprint") != fingerprint(contract)
            or manifest.get("contract_fingerprint") != fingerprint(contract)
            or manifest.get("samples_sha256") != file_hash(root / "samples.jsonl")):
        raise TrainingError("encoding_cache_mismatch", "Cache data/model/template/dependencies/code or result bytes changed")
    samples = {s.sample_id: asdict(s) for s in index.samples}
    cached = {}
    for _, row in jsonl(root / "samples.jsonl"):
        if row.get("status") != "passed":
            continue  # Failed rows must be encoded again, never silently excluded.
        sample_id = row.get("sample_id")
        if (sample_id not in samples or sample_id in cached
                or any(row.get(k) != v for k, v in samples[sample_id].items())
                or any(k not in row for k in _ENCODE_FIELDS)
                or row["all_history_masked"] is not True or row["complete_target_verified"] is not True
                or row["weights_loaded"] is not False or row["loss_tokens"] <= 0):
            raise TrainingError("invalid_encoding_cache", "Cache sample identity or real supervision evidence is invalid")
        cached[sample_id] = {k: row[k] for k in _ENCODE_FIELDS}
    if len(cached) != manifest.get("passed_samples"):
        raise TrainingError("invalid_encoding_cache", "Cache sample count differs from its seal")
    if file_hash(root / "samples.jsonl") != manifest["samples_sha256"]:
        raise TrainingError("encoding_cache_changed", "Cache changed while reading")
    return cached


def _initialize_worker(config, model):
    global _WORKER_BACKEND
    _WORKER_BACKEND = build_encoding_backend(config, model, unbounded=True)


def _encode(row, normalized, backend):
    try:
        return {"result": check_encoding(row, backend.template, backend.processor,
                                         preprocessor=backend.preprocessor, render_target=backend.render_target,
                                         normalized=normalized)}
    except (ConversionError, ValueError, TypeError, KeyError, RuntimeError, OSError) as exc:
        return {"error": exc.record() if isinstance(exc, ConversionError) else {
            "code": "encoding_failed", "error_type": type(exc).__name__}}


def _encode_worker(row, normalized):
    return _encode(row, normalized, _WORKER_BACKEND)


def bounded_results(executor, jobs, *, limit):
    """Keep at most limit records in flight, preserving the package order."""
    jobs, pending = iter(jobs), deque()
    for _ in range(limit):
        item = next(jobs, None)
        if item is None:
            break
        sample, row, normalized = item
        pending.append((sample, executor.submit(_encode_worker, row, normalized)))
    while pending:
        sample, future = pending.popleft()
        yield sample, future.result()
        item = next(jobs, None)
        if item is not None:
            sample, row, normalized = item
            pending.append((sample, executor.submit(_encode_worker, row, normalized)))


def encoding_results(index, datasets, backend, config, model, workers, cached):
    cursors = Counter()

    def jobs():
        for sample, row in index.records(absolute_images=True):
            normalized = datasets[sample.split][cursors[sample.split]]
            cursors[sample.split] += 1
            if sample.sample_id in cached:
                continue
            yield sample, row, normalized

    if workers == 1:
        # Retain source order while avoiding image processing for sealed rows.
        for sample, row in index.records(absolute_images=True):
            position = cursors[sample.split]
            cursors[sample.split] += 1
            if sample.sample_id in cached:
                yield sample, {"result": cached[sample.sample_id], "cached": True}
            else:
                yield sample, _encode(row, datasets[sample.split][position], backend)
    else:
        # Each worker owns one processor and one sample's image tensors. The
        # native dataset loader itself remains dataset_num_proc=1.
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                       initializer=_initialize_worker, initargs=(config, str(model)))
        try:
            results = iter(bounded_results(executor, jobs(), limit=workers * 2))
            for sample in index.samples:
                if sample.sample_id in cached:
                    yield sample, {"result": cached[sample.sample_id], "cached": True}
                else:
                    yield next(results)
        except BaseException as exc:
            exits = {str(p.pid): p.exitcode for p in executor._processes.values()}
            # Python 3.11 has no public terminate_workers API. Terminate only
            # this executor's known children rather than waiting for long rows.
            for process in tuple(executor._processes.values()):
                if process.is_alive():
                    process.terminate()
            if isinstance(exc, BrokenProcessPool):
                raise TrainingError("encoding_worker_failed", "An encoder process exited unexpectedly; preserve this run and inspect resources",
                                    worker_exit_codes=exits) from exc
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)


def distribution(values):
    values = sorted(values)
    if not values:
        return {}
    return {"min": values[0], "max": values[-1], "mean": sum(values) / len(values),
            **{name: values[int((len(values) - 1) * q)] for name, q in (("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99))}}


def preflight(data, config, output, *, model=None, structure_only=False, encoding_workers=1, encoding_cache=None):
    if type(encoding_workers) is not int or not 1 <= encoding_workers <= 8 or (structure_only and encoding_workers != 1):
        raise TrainingError("invalid_encoding_workers", "Use 1..8 encoding workers; structure-only requires 1")
    if structure_only and encoding_cache:
        raise TrainingError("invalid_encoding_cache", "Structure-only checks cannot use encoding caches")
    index = index_package(data, expected_k=config["data"]["k"],
                          expected_history_compression=config["data"].get("history_compression", "legacy"))
    protected = [index.root] + ([Path(model)] if model else []) + ([Path(encoding_cache)] if encoding_cache else [])
    if not structure_only and model is None:
        raise TrainingError("missing_model", "Real encoding preflight requires --model")
    root = create_output(output, protected=protected)
    summary = {"status": "running", "scan_complete": False, "encoding_passed": False,
               "full_dataset_trainable": False, "structure_only": structure_only,
               "validation_level": "cpu_package_validation" if structure_only else "qwen_processor_encoding",
               "package_fingerprint": index.fingerprint, "expected_samples": len(index.samples),
               "encoding_workers": encoding_workers}
    write_json(root / "summary.json", summary)
    write_json(root / "package_manifest.json", index.manifest())
    write_json(root / "resolved_config.json", config)
    journal(root, "preflight_started", structure_only=structure_only)
    try:
        env = environment()
        write_json(root / "environment.json", env)
        prepared = materialize(index, root / "prepared")
        write_json(root / "prepared_manifest.json", prepared)
        if structure_only:
            write_json(root / "encoding_contract.json", {"contract": encoding_contract(config), "verified": False})
            (root / "errors.jsonl").touch(exist_ok=False)
            with (root / "samples.jsonl").open("x") as stream:
                for sample in index.samples:
                    stream.write(canonical({**asdict(sample), "status": "structure_passed", "encoding_verified": False}) + "\n")
            summary.update(status="complete", scan_complete=True, samples=len(index.samples),
                           structure_passed=True, eligible_samples=None)
        else:
            model_info = model_manifest(model)
            write_json(root / "model_manifest.json", model_info)
            contract = {"package": index.fingerprint, "model": model_info["fingerprint"],
                        "encoding": encoding_contract(config), "packages": env["packages"], "encoder_source": encoder_source()}
            write_json(root / "encoding_contract.json", {"contract": contract, "fingerprint": fingerprint(contract)})
            cached = load_encoding_cache(encoding_cache, contract, index)
            summary.update(cache_source=str(Path(encoding_cache).resolve()) if encoding_cache else None, cache_hits=len(cached))
            backend = build_encoding_backend(config, model, unbounded=True)
            datasets = load_prepared_datasets(backend, root / "prepared")
            if any(len(datasets[s]) != prepared["splits"][s] for s in ("train", "val")):
                raise TrainingError("framework_changed_dataset", "Framework changed the number of rows")
            eligible, lengths, groups = Counter(), [], defaultdict(list)
            errors = processed = 0
            with (root / "samples.jsonl").open("x") as rows, (root / "errors.jsonl").open("x") as failures:
                for sample, outcome in encoding_results(index, datasets, backend, config, model, encoding_workers, cached):
                    report = asdict(sample)
                    if "result" in outcome:
                        encoded = outcome["result"]
                        within = encoded["tokens"] <= config["data"]["max_length"]
                        model_within = encoded["tokens"] <= backend.model_max_length
                        report.update(encoded, model_max_length=backend.model_max_length,
                                      within_budget=within, within_model_capacity=model_within)
                        if outcome.get("cached"):
                            report["encoding_cache_source"] = summary["cache_source"]
                        if within and model_within:
                            eligible[sample.split] += 1
                        lengths.append(encoded["tokens"])
                        for kind in ("split", "scene", "task", "target_tool", "primitive"):
                            groups[kind + ":" + str(getattr(sample, kind))].append(encoded["tokens"])
                    else:
                        errors += 1
                        error = outcome["error"]
                        report.update(status="failed", error=error)
                        failures.write(canonical({"sample_id": sample.sample_id, **error}) + "\n")
                    rows.write(canonical(report) + "\n")
                    processed += 1
                    if processed % 25 == 0:
                        rows.flush()
                        journal(root, "encoding_progress", samples=processed, errors=errors)
            index.assert_unchanged(images=True)
            if encoder_source() != contract["encoder_source"]:
                raise TrainingError("encoder_changed", "Encoding code changed during preflight")
            if model_manifest(model)["fingerprint"] != model_info["fingerprint"]:
                raise TrainingError("model_changed", "Model changed during preflight")
            summary.update(status="complete" if not errors else "failed", scan_complete=True,
                           encoding_passed=errors == 0, samples=processed, errors=errors,
                           eligible_samples={s: eligible[s] for s in ("train", "val")},
                           full_dataset_trainable=errors == 0 and sum(eligible.values()) == len(index.samples),
                           tokens=distribution(lengths), grouped_tokens={k: {"count": len(v), **distribution(v)} for k, v in groups.items()},
                           model_max_length=backend.model_max_length)
        summary["report_files"] = {name: file_hash(root / name) for name in
                                   ("samples.jsonl", "errors.jsonl", "package_manifest.json", "encoding_contract.json")}
        write_json(root / "summary.json", summary)
        if not structure_only:
            seal_encoding_cache(root)
        journal(root, "preflight_finished", status=summary["status"])
        if summary["status"] != "complete":
            raise TrainingError("encoding_preflight_failed", "Some samples failed real encoding", output=str(root))
        return {**summary, "output": str(root)}
    except BaseException as exc:
        summary["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        summary["failure"] = exc.record() if isinstance(exc, ConversionError) else {"error_type": type(exc).__name__}
        write_json(root / "summary.json", summary)
        journal(root, "preflight_failed", status=summary["status"], failure=summary["failure"])
        if not structure_only:
            try:
                seal_encoding_cache(root)
            except (ConversionError, OSError, KeyError):
                journal(root, "cache_seal_unavailable")
        raise
