"""Build, validate and package per-decision SFT records from explicit sources."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
from pathlib import Path
import platform
import shutil
import sys

from . import VERSION
from .common import ConversionError, dumps, file_hash, jsonl, loads, read_json
from .history import DEFAULT_MAX_LEN, build_sample_messages, context_policy, validate_max_len
from .history_compaction import LEGACY_HISTORY, validate_history_compression
from .format import to_msswift, validate_record
from .protocol import validate_arguments
from .source import Episode, EpisodeSource, independent_success, load_episode


def manifest_sources(path: Path) -> list[EpisodeSource]:
    sources = []
    for line, row in jsonl(path):
        if set(row) != {"run_dir", "controller_dir"} or not all(isinstance(v, str) and v for v in row.values()):
            raise ConversionError("invalid_manifest_row", "Use exactly run_dir and controller_dir strings", line=line)
        sources.append(EpisodeSource((path.parent / row["run_dir"]).resolve(),
                                     (path.parent / row["controller_dir"]).resolve()))
    if not sources:
        raise ConversionError("empty_manifest", "The source manifest is empty")
    return sources


def source_id(episode):
    files = episode.provenance["input_files"]
    return hashlib.sha256((files["run"]["sha256"] + files["model_events"]["sha256"]).encode()).hexdigest()


def episode_samples(episode: Episode, k=2, *, max_len=DEFAULT_MAX_LEN, history_compression=LEGACY_HISTORY):
    validate_max_len(max_len)
    history = [{"role": "system", "content": episode.system}, {"role": "user", "content": episode.user}]
    schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in episode.tools}
    for index, decision in enumerate(episode.decisions):
        if decision.target:
            messages, length_stats = build_sample_messages(
                history, decision.assistant(target=True), episode.tools, k=k, max_len=max_len,
                history_compression=history_compression)
            record = to_msswift(messages, episode.tools)
            validate_record(record)
            function = record["messages"][-1]["tool_calls"][0]["function"]
            validate_arguments(loads(function["arguments"]), schemas[function["name"]])
            metadata = {"source_id": source_id(episode), "decision_index": index,
                        "call_id": decision.call_id, "call_line": decision.call_line,
                        "result_line": decision.result_line, "evidence_id": decision.evidence_id,
                        "target_tool": decision.tool,
                        "primitive": decision.arguments.get("primitive") if isinstance(decision.arguments, dict) else None,
                        "scene": episode.provenance.get("scene"), **length_stats}
            yield record, metadata
        history.extend(decision.history())


def choose_validation_scenes(counts: dict[str, int], *, seed=42, ratio=0.1):
    if len(counts) < 2:
        raise ConversionError("insufficient_scenes", "Batch splitting requires at least two eligible scenes; use single mode for one scene")
    ordered = sorted(counts, key=lambda scene: (hashlib.sha256(f"{seed}:{scene}".encode()).hexdigest(), scene))
    total = sum(counts.values())
    # Consider all reachable sample counts, not just prefixes of the seeded
    # scene order. Store one deterministic scene group per count; positive
    # counts let us exclude the empty group and the all-validation split.
    groups = {0: ()}
    for index, scene in enumerate(ordered):
        for count, group in list(groups.items()):
            combined, candidate = count + counts[scene], (*group, index)
            if combined not in groups or candidate < groups[combined]:
                groups[combined] = candidate
    target = Fraction(str(ratio)) * total
    count = min((count for count in groups if 0 < count < total),
                key=lambda count: (abs(count - target), groups[count]))
    return sorted(ordered[index] for index in groups[count])


@dataclass
class Dataset:
    episodes: list[Episode]
    k: int
    batch: bool
    errors: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    validation_scenes: list[str] = field(default_factory=list)
    max_len: int = DEFAULT_MAX_LEN
    history_compression: str = LEGACY_HISTORY

    def records(self):
        val = set(self.validation_scenes)
        for episode in self.episodes:
            split = "val" if episode.provenance["scene"] in val else "train"
            if not self.batch:
                split = "dataset"
            for record, metadata in episode_samples(episode, self.k, max_len=self.max_len,
                                                     history_compression=self.history_compression):
                yield record, {**metadata, "split": split}


def _error_outcome(source):
    try:
        run = read_json(source.run_dir / "run.json")
        return {"task_success": independent_success(run.get("evaluation")), "run_id": run.get("run_id")}
    except ConversionError:
        return {"task_success": None}


def _length_distribution(lengths):
    return {"min": min(lengths, default=0), "max": max(lengths, default=0),
            "mean": sum(lengths) / len(lengths) if lengths else 0,
            "method": "chars/4+64/image; not processor tokens; no sample length filtering"}


def prepare_dataset(sources: list[EpisodeSource], *, batch=False, k=2, max_len=DEFAULT_MAX_LEN,
                    history_compression=LEGACY_HISTORY):
    if type(k) is not int or k < 1:
        raise ConversionError("invalid_window", "k must be a positive integer")
    validate_max_len(max_len)
    validate_history_compression(history_compression)
    if not batch and len(sources) != 1:
        raise ConversionError("invalid_sources", "Single mode requires exactly one episode/controller pair")
    data = Dataset([], k, batch, max_len=max_len, history_compression=history_compression)
    seen = set()
    for source in sources:
        try:
            episode = load_episode(source)
            if episode.skipped:
                data.skipped.append({**source.paths(), "reason": episode.skipped,
                                     "task_success": episode.provenance["task_success"]})
                continue
            identity = source_id(episode)
            if identity in seen:
                raise ConversionError("duplicate_source", "The manifest repeats the same immutable trajectory")
            scene = episode.provenance.get("scene")
            if batch and (not isinstance(scene, str) or not scene.strip()):
                raise ConversionError("missing_scene", "Batch sources need an evaluator-recorded scene")
            # Fully validate every record before accepting this episode. No
            # partially converted trajectory is mixed into the batch.
            for record, metadata in episode_samples(episode, k, max_len=max_len,
                                                     history_compression=history_compression):
                if any(path not in episode.assets for path in record["images"]):
                    raise ConversionError("unverified_image", "Sample references an unverified image")
            seen.add(identity)
            data.episodes.append(episode)
        except (ConversionError, OSError, KeyError, TypeError, ValueError) as exc:
            error = exc if isinstance(exc, ConversionError) else ConversionError(
                "invalid_source_structure", "Source records have an unsupported or incomplete structure")
            if not batch:
                raise error from (exc if error is not exc else None)
            data.errors.append({**source.paths(), **_error_outcome(source), **error.record()})
    data.episodes.sort(key=source_id)
    scenes = Counter()
    for episode in data.episodes:
        scenes[episode.provenance.get("scene")] += sum(d.target for d in episode.decisions)
    if batch:
        try:
            data.validation_scenes = choose_validation_scenes(dict(scenes))
        except ConversionError as exc:
            exc.location.update(source_errors=data.errors, skipped=data.skipped)
            raise
    tools, splits, image_paths, lengths = Counter(), Counter(), set(), []
    lengths_before, trimming = [], Counter()
    for record, meta in data.records():
        tools[meta["target_tool"]] += 1
        splits[meta["split"]] += 1
        lengths.append(meta["approx_tokens"])
        lengths_before.append(meta["approx_tokens_before"])
        image_paths.update(record["images"])
        trimming["slim_triggered_samples"] += meta["slim_triggered"]
        trimming["slimmed_samples"] += meta["slimmed_images"] > 0
        trimming["truncated_samples"] += meta["dropped_history_pairs"] > 0
        trimming["over_budget_samples"] += meta["over_budget"]
        for key in ("slimmed_images", "dropped_history_pairs", "images_removed"):
            trimming[key] += meta[key]
    data.stats = {"schema_version": 1, "converter_version": VERSION, "input_episodes": len(sources),
                  "eligible_episodes": len(data.episodes), "skipped_episodes": len(data.skipped),
                  "error_episodes": len(data.errors), "samples": sum(splits.values()),
                  "splits": dict(splits), "target_tools": dict(tools), "unique_images": len(image_paths),
                  "scene_samples": dict(scenes), "validation_scenes": data.validation_scenes,
                  "validation_ratio": splits["val"] / sum(splits.values()) if batch else None,
                  "approx_tokens": _length_distribution(lengths),
                  "approx_tokens_before": _length_distribution(lengths_before),
                  "length_control": {"context_policy": context_policy(history_compression), "max_len": max_len,
                                     **{key: trimming[key] for key in (
                                         "slim_triggered_samples", "slimmed_samples", "truncated_samples",
                                         "over_budget_samples", "slimmed_images", "dropped_history_pairs", "images_removed")}},
                  "warnings": [{"source_id": source_id(e), **w} for e in data.episodes for w in e.warnings],
                  "validation_level": "cpu_record_and_image_validation", "qwen_encoding_verified": False}
    return data


def assert_inputs_unchanged(data):
    for episode in data.episodes:
        for artifact in episode.provenance["input_files"].values():
            if file_hash(Path(artifact["path"])) != artifact["sha256"]:
                raise ConversionError("source_changed", "An input changed during conversion", path=artifact["path"])


def write_package(data: Dataset, output: Path, *, source_manifest: Path | None = None):
    output = output.resolve()
    for episode in data.episodes:
        for source in (episode.source.run_dir, episode.source.controller_dir):
            if output.is_relative_to(source) or source.is_relative_to(output):
                raise ConversionError("output_overlaps_source", "Output and source directories must be separate")
    if output.exists():
        raise ConversionError("output_exists", "Choose a new output directory", path=str(output))
    assert_inputs_unchanged(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # Exclusive creation; never overwrite an existing package.
    files = ["train.jsonl", "val.jsonl"] if data.batch else ["dataset.jsonl"]
    staged = files + ["sample_sources.jsonl", "sources.jsonl", "images.jsonl", "errors.jsonl", "skipped.jsonl"]
    handles = {}
    try:
        for name in staged:
            handles[name] = (output / (name + ".incomplete")).open("x", encoding="utf-8")
        assets = {key: asset for episode in data.episodes for key, asset in episode.assets.items()}
        used = set()
        for record, metadata in data.records():
            name = metadata["split"] + ".jsonl"
            handles[name].write(dumps(record) + "\n")
            handles["sample_sources.jsonl"].write(dumps(metadata) + "\n")
            used.update(record["images"])
        for episode in data.episodes:
            handles["sources.jsonl"].write(dumps({"source_id": source_id(episode), **episode.provenance}) + "\n")
        for name, rows in (("errors.jsonl", data.errors), ("skipped.jsonl", data.skipped)):
            for row in rows:
                handles[name].write(dumps(row) + "\n")
        (output / "images").mkdir()
        for relative in sorted(used):
            asset = assets[relative]
            target = output / relative
            shutil.copyfile(asset.source, target)
            if file_hash(target) != asset.sha256:
                raise ConversionError("source_image_changed", "An image changed after verification", path=str(asset.source))
            handles["images.jsonl"].write(dumps({"file": relative, "sha256": asset.sha256,
                                               "mime_type": asset.mime_type, "source": str(asset.source)}) + "\n")
        assert_inputs_unchanged(data)
        from ..records import source_version
        config = {"converter_version": VERSION, "converter_source": source_version(),
                  "python": sys.executable, "host": platform.node(), "k": data.k,
                  "window_unit": "RGB-bearing tool replies", "keep_all_images": data.max_len == 0,
                  "context_policy": context_policy(data.history_compression), "max_len": data.max_len,
                  "coordinate_protocol": "target.point_2d_integer_0_1000", "thinking": "empty_think",
                  "length_limit": None, "trajectory_turn_limit": None,
                  "split_seed": 42 if data.batch else None, "requested_validation_ratio": 0.1 if data.batch else None,
                  "split_method": "closest_sample_count_scene_subset; seeded_scene_order_tiebreak" if data.batch else None,
                  "source_manifest_sha256": file_hash(source_manifest) if source_manifest else None,
                  "image_path_base": "package root", "training_started": False}
        if data.history_compression != LEGACY_HISTORY:
            config["history_compression"] = data.history_compression
        for name, value in (("config.json", config), ("stats.json", data.stats)):
            (output / name).write_text(dumps(value, indent=2) + "\n", encoding="utf-8")
        for handle in handles.values():
            handle.close()
        for name in staged:
            (output / (name + ".incomplete")).rename(output / name)
        (output / "build.json").write_text(dumps({"status": "complete", "converter_version": VERSION}) + "\n")
    except Exception as exc:
        for handle in handles.values():
            handle.close()
        error = exc.record() if isinstance(exc, ConversionError) else {"code": "write_failed", "error_type": type(exc).__name__}
        (output / "build.json").write_text(dumps({"status": "failed", "error": error}) + "\n")
        raise
    return output
