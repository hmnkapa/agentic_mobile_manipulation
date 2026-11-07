"""Create an image-preserving package with versioned compact history replies."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import shutil

from .common import ConversionError, dumps, file_hash, jsonl
from .format import estimate_length
from .history import COMPACT_CONTEXT_POLICY
from .history_compaction import COMPACT_HISTORY, LEGACY_HISTORY, compact_record


def compact_package(data, output, *, provenance_root=None):
    from ..records import source_version
    from ..training.artifacts import create_output, encoder_source, journal, write_json
    from ..training.data import index_package

    index = index_package(data)
    if index.config.get("history_compression", LEGACY_HISTORY) != LEGACY_HISTORY:
        raise ConversionError("already_compacted", "Use a legacy package as the input to history compaction")
    root = create_output(output, protected=(index.root,))
    write_json(root / "build.json", {"status": "running", "operation": "compact_history"})
    journal(root, "history_compaction_started", policy=COMPACT_HISTORY, samples=len(index.samples))
    parent = index.manifest()
    if provenance_root:
        parent["read_from"] = parent["root"]
        parent["root"] = str(Path(provenance_root).resolve())
    contract = {"schema_version": 1, "history_compression": COMPACT_HISTORY, "k": index.config["k"],
                "parent_package": parent, "compression_source": encoder_source(),
                "converter_source": source_version(), "images_preserved": True, "targets_preserved": True,
                "weights_loaded": False, "gpu_used": False}
    write_json(root / "compression_contract.json", contract)
    counts = Counter()
    before_lengths, after_lengths = [], []
    staged = ("train.jsonl", "val.jsonl", "sample_sources.jsonl")
    try:
        metadata = (meta for _, meta in jsonl(index.root / "sample_sources.jsonl"))
        with ExitStack() as stack:
            handles = {name: stack.enter_context((root / (name + ".incomplete")).open("x")) for name in staged}
            for (sample, row), meta in zip(index.records(), metadata, strict=True):
                result, compression = compact_record(row, k=index.config["k"])
                before, after = estimate_length(row), estimate_length(result)
                before_lengths.append(before["approx_tokens"])
                after_lengths.append(after["approx_tokens"])
                counts["samples"] += 1
                counts["changed_samples"] += bool(compression["changed_replies"])
                counts["changed_reply_occurrences"] += compression["changed_replies"]
                rewritten = {key: value for key, value in meta.items() if not key.startswith("processor_")}
                rewritten.update(after, approx_tokens_before=before["approx_tokens"],
                                 image_count_before=before["image_count"], slim_triggered=False,
                                 slimmed_images=0, dropped_history_pairs=0, images_removed=0, over_budget=False,
                                 history_compaction=compression, parent_sample_metadata=meta)
                handles[sample.split + ".jsonl"].write(dumps(result) + "\n")
                handles["sample_sources.jsonl"].write(dumps(rewritten) + "\n")
                if counts["samples"] % 250 == 0:
                    journal(root, "history_compaction_progress", **dict(counts))
                    print(dumps(dict(counts)), flush=True)
        (root / "images").mkdir()

        def copy_image(relative):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(index.root / relative, target)
            if file_hash(target) != index.images[relative]["sha256"]:
                raise ConversionError("copied_image_changed", "Copied image differs from its source", file=relative)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(copy_image, sorted(index.images)))
        for name in ("sources.jsonl", "images.jsonl", "errors.jsonl", "skipped.jsonl"):
            if (index.root / name).is_file():
                shutil.copyfile(index.root / name, root / name)
        index.assert_unchanged(images=True)
        for name in staged:
            (root / (name + ".incomplete")).rename(root / name)
        report = {"status": "complete", "history_compression": COMPACT_HISTORY, "k": index.config["k"],
                  **dict(counts), "splits": index.stats["splits"], "trajectories": len(index.sources),
                  "unique_images": len(index.images), "parent_package": parent,
                  "approx_tokens_before": {"min": min(before_lengths), "max": max(before_lengths),
                                           "mean": sum(before_lengths)/len(before_lengths)},
                  "approx_tokens": {"min": min(after_lengths), "max": max(after_lengths),
                                    "mean": sum(after_lengths)/len(after_lengths)},
                  "images_removed": 0, "messages_removed": 0, "targets_unchanged": True,
                  "validation_level": "cpu_record_and_image_validation", "qwen_encoding_verified": False,
                  "gpu_used": False, "training_started": False}
        write_json(root / "compression_report.json", report)
        # The parent retains its original trimming evidence. Changed text has
        # no valid inherited processor lengths; new preflight measures it.
        config = {key: deepcopy(value) for key, value in index.config.items()
                  if not key.startswith("processor_") and key not in (
                      "length_contract_sha256", "token_lengths_sha256", "all_token_lengths_sha256",
                      "measured_input_samples", "separated_over_budget_samples", "over_budget_policy")}
        config.update(context_policy=COMPACT_CONTEXT_POLICY, history_compression=COMPACT_HISTORY,
                      max_len=0, keep_all_images=True, length_measurement="not_yet_processor_verified",
                      parent_package=parent, converter_source=source_version(),
                      compression_contract_sha256=file_hash(root / "compression_contract.json"),
                      compression_report_sha256=file_hash(root / "compression_report.json"))
        stats = {key: deepcopy(value) for key, value in index.stats.items() if not key.startswith("processor_")}
        stats.update(approx_tokens=report["approx_tokens"], approx_tokens_before=report["approx_tokens_before"],
                     qwen_encoding_verified=False, validation_level=report["validation_level"],
                     length_control={"context_policy": COMPACT_CONTEXT_POLICY, "max_len": 0,
                                     "history_compression": COMPACT_HISTORY, **dict(counts),
                                     "slimmed_images": 0, "dropped_history_pairs": 0, "images_removed": 0})
        write_json(root / "config.json", config)
        write_json(root / "stats.json", stats)
        write_json(root / "build.json", {"status": "complete", "converter_version": config["converter_version"],
                                         "history_compression": COMPACT_HISTORY})
        final = index_package(root, expected_k=index.config["k"], expected_history_compression=COMPACT_HISTORY)
        if [s.sample_id for s in final.samples] != [s.sample_id for s in index.samples]:
            raise ConversionError("compaction_identity_changed", "Compaction changed sample identities or splits")
        journal(root, "history_compaction_complete", **dict(counts))
        return report
    except BaseException as exc:
        error = exc.record() if isinstance(exc, ConversionError) else {"error_type": type(exc).__name__}
        write_json(root / "build.json", {"status": "failed", "error": error})
        journal(root, "history_compaction_failed", error=error)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--provenance-root", type=Path)
    args = parser.parse_args(argv)
    try:
        print(dumps(compact_package(args.data, args.out_dir, provenance_root=args.provenance_root)))
        return 0
    except (ConversionError, OSError) as exc:
        error = exc.record() if isinstance(exc, ConversionError) else {"error_type": type(exc).__name__}
        print(dumps({"status": "failed", "error": error}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
