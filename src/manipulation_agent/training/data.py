"""CPU-only package indexing and explicit, provenance-preserving selections."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from ..sft import VERSION
from ..sft.common import ConversionError, dumps, file_hash, inside, jsonl, loads, read_json
from ..sft.format import validate_record
from ..sft.history import CONTEXT_POLICY, COMPACT_CONTEXT_POLICY
from ..sft.history_compaction import COMPACT_HISTORY, LEGACY_HISTORY, HISTORY_COMPRESSION_POLICIES
from ..sft.processor_budget import PROCESSOR_CONTEXT_POLICY
from ..sft.protocol import validate_arguments
from .artifacts import TrainingError, canonical, fingerprint, write_json


@dataclass(frozen=True)
class SampleRef:
    sample_id: str
    source_id: str
    decision_index: int
    call_id: str
    split: str
    line: int
    offset: int
    sha256: str
    target_tool: str
    primitive: str | None
    scene: str
    task: str | None
    image_count: int
    has_failed_history: bool


@dataclass
class PackageIndex:
    root: Path
    samples: list[SampleRef]
    sources: dict
    images: dict
    image_paths: dict[str, str]
    files: dict
    config: dict
    stats: dict

    @property
    def fingerprint(self):
        return fingerprint(self.files)

    def manifest(self):
        return {"root": str(self.root), "fingerprint": self.fingerprint, "files": self.files,
                "samples": len(self.samples), "splits": dict(Counter(s.split for s in self.samples)),
                "sources": len(self.sources), "images": len(self.images)}

    def record(self, sample, *, absolute_images=False, _stream=None):
        if _stream is None:
            with (self.root / (sample.split + ".jsonl")).open("rb") as stream:
                return self.record(sample, absolute_images=absolute_images, _stream=stream)
        _stream.seek(sample.offset)
        raw = _stream.readline()
        if hashlib.sha256(raw).hexdigest() != sample.sha256:
            raise TrainingError("source_changed", "A package row changed after indexing", sample_id=sample.sample_id)
        row = loads(raw)
        if absolute_images:
            # Paths were resolved and bytes verified when indexing. Repeating
            # resolve() for every occurrence of a historical image causes
            # hundreds of thousands of redundant NFS metadata operations.
            row["images"] = [self.image_paths[p] for p in row["images"]]
        return row

    def records(self, samples=None, *, absolute_images=False):
        with ExitStack() as stack:
            streams = {s: stack.enter_context((self.root / (s + ".jsonl")).open("rb")) for s in ("train", "val")}
            for sample in self.samples if samples is None else samples:
                yield sample, self.record(sample, absolute_images=absolute_images, _stream=streams[sample.split])

    def assert_unchanged(self, *, images=False):
        for name, expected in self.files.items():
            if file_hash(self.root / name) != expected:
                raise TrainingError("source_changed", "Package changed after validation", file=name)
        if images:
            if verify_images(self.root, self.images) != self.image_paths:
                raise TrainingError("source_image_changed", "An image path changed after indexing")


def verify_images(root, images):
    # Independent read-only files: bound NFS I/O concurrency and hash each
    # unique asset once per integrity pass. No simulator access occurs here.
    def verify(item):
        name, metadata = item
        path = inside(root, name)
        if not path.is_relative_to(root / "images") or file_hash(path) != metadata["sha256"]:
            raise TrainingError("image_hash_mismatch", "Image does not match the package manifest", file=name)
        return name, str(path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(pool.map(verify, images.items()))


def has_failed_history(record):
    decoder = json.JSONDecoder()
    for message in record["messages"][:-1]:
        if message.get("role") != "tool":
            continue
        text = message.get("content", "")
        if "Recorded MCP transport error:" in text or "Recorded MCP isError=true" in text:
            return True
        try:
            result, _ = decoder.raw_decode(text.removeprefix("Result: ").lstrip())
            if isinstance(result, dict) and result.get("ok") is False:
                return True
        except (ValueError, TypeError):
            pass
    return False


def _next_record(stream, line):
    while True:
        offset = stream.tell()
        raw = stream.readline()
        if not raw:
            raise TrainingError("missing_package_row", "Source metadata has no corresponding dataset row")
        line += 1
        if raw.strip():
            try:
                row = loads(raw)
                if not isinstance(row, dict):
                    raise ValueError()
                return line, offset, raw, row
            except (ValueError, UnicodeError) as exc:
                raise TrainingError("invalid_package_row", "Invalid dataset JSONL", line=line) from exc


def _supported_context(config):
    history = config.get("history_compression", LEGACY_HISTORY)
    if history not in HISTORY_COMPRESSION_POLICIES:
        return False
    version = config.get("converter_version")
    if version == "manipulation_sft_v1.2.1":
        return (history == LEGACY_HISTORY and config.get("context_policy") is None and config.get("max_len") is None
                and config.get("keep_all_images") is True)
    max_len = config.get("max_len")
    if config.get("context_policy") == PROCESSOR_CONTEXT_POLICY:
        contract = config.get("processor_encoding_contract")
        return (version == VERSION and type(max_len) is int and max_len > 0
                and config.get("keep_all_images") is False
                and config.get("window_unit") == "RGB-bearing tool replies"
                and config.get("length_measurement") == "qwen_processor_tokens"
                and isinstance(contract, dict) and contract.get("k") == config.get("k")
                and contract.get("history_compression", LEGACY_HISTORY) == history
                and all(isinstance(contract.get(k), dict) for k in ("model", "template", "vision"))
                and isinstance(config.get("processor_model_fingerprint"), str))
    policy = COMPACT_CONTEXT_POLICY if history == COMPACT_HISTORY else CONTEXT_POLICY
    return (version == VERSION and config.get("context_policy") == policy
            and type(max_len) is int and max_len >= 0
            and config.get("keep_all_images") is (max_len == 0)
            and config.get("window_unit") == "RGB-bearing tool replies")


def index_package(path, *, expected_k=2, expected_history_compression=None):
    root = Path(path).resolve()
    required = ("build.json", "config.json", "stats.json", "train.jsonl", "val.jsonl",
                "sources.jsonl", "sample_sources.jsonl", "images.jsonl")
    if not all((root / p).is_file() for p in required) or not (root / "images").is_dir():
        raise TrainingError("incomplete_package", "Training requires a complete train/val data package")
    if any(not (root / p).resolve().is_relative_to(root) for p in required):
        raise TrainingError("package_path_escape", "Package metadata leaves the package directory")
    if list(root.glob("*.incomplete")) or read_json(root / "build.json").get("status") != "complete":
        raise TrainingError("incomplete_package", "Package build is not complete")
    files = {p: file_hash(root / p) for p in required}
    config, stats = read_json(root / "config.json"), read_json(root / "stats.json")
    if (expected_history_compression is not None
            and config.get("history_compression", LEGACY_HISTORY) != expected_history_compression):
        raise TrainingError("history_compression_mismatch", "Recipe and package history compression policies differ")
    for name, field in (("compression_contract.json", "compression_contract_sha256"),
                        ("compression_report.json", "compression_report_sha256")):
        if field in config:
            evidence = root / name
            if (not evidence.is_file() or not evidence.resolve().is_relative_to(root)
                    or file_hash(evidence) != config[field]):
                raise TrainingError("compression_evidence_mismatch", "History compression evidence is missing or changed", file=name)
            files[name] = config[field]
    if config.get("context_policy") == PROCESSOR_CONTEXT_POLICY:
        for name, field in (("length_contract.json", "length_contract_sha256"),
                            ("token_lengths.jsonl", "token_lengths_sha256")):
            path = root / name
            if (not path.is_file() or not path.resolve().is_relative_to(root)
                    or file_hash(path) != config.get(field)):
                raise TrainingError("processor_evidence_mismatch", "Real-token trimming evidence is missing or changed", file=name)
            files[name] = config[field]
    if (not _supported_context(config) or stats.get("schema_version") != 1
            or stats.get("converter_version") != config.get("converter_version")
            or config.get("k") != expected_k
            or config.get("coordinate_protocol") != "target.point_2d_integer_0_1000"
            or config.get("thinking") != "empty_think"):
        raise TrainingError("unsupported_package_contract", "Package version or student context contract is unsupported")
    sources, images = {}, {}
    for _, source in jsonl(root / "sources.jsonl"):
        key = source.get("source_id")
        if not isinstance(key, str) or key in sources:
            raise TrainingError("duplicate_source", "Source identity is missing or duplicated")
        try:
            expected = hashlib.sha256((source["input_files"]["run"]["sha256"]
                                       + source["input_files"]["model_events"]["sha256"]).encode()).hexdigest()
        except (KeyError, TypeError) as exc:
            raise TrainingError("invalid_source_metadata", "Source fingerprint fields are missing") from exc
        if key != expected or source.get("task_success") is not True:
            raise TrainingError("invalid_source_metadata", "Source identity or independent success is invalid")
        if (not isinstance(source.get("run_config"), dict)
                or not isinstance(source["run_config"].get("instruction"), str)
                or not isinstance(source.get("task_instance"), dict)
                or source.get("scene") != source["task_instance"].get("scene")):
            raise TrainingError("invalid_source_metadata", "Source task/configuration metadata is invalid")
        sources[key] = source
    for _, image in jsonl(root / "images.jsonl"):
        name = image.get("file")
        if (not isinstance(name, str) or Path(name).is_absolute() or name in images
                or image.get("mime_type") not in {"image/png", "image/jpeg"}):
            raise TrainingError("invalid_image_manifest", "Image entry is invalid or duplicated")
        if not isinstance(image.get("sha256"), str):
            raise TrainingError("invalid_image_manifest", "Image hash is missing")
        images[name] = image
    image_paths = verify_images(root, images)
    samples, seen, source_splits, scene_splits, used_images = [], set(), {}, {}, set()
    counts, scene_counts = Counter(), Counter()
    streams = {s: (root / (s + ".jsonl")).open("rb") for s in ("train", "val")}
    lines = {s: 0 for s in streams}
    try:
        for meta_line, meta in jsonl(root / "sample_sources.jsonl"):
            split, source_id = meta.get("split"), meta.get("source_id")
            if split not in streams or source_id not in sources:
                raise TrainingError("invalid_sample_source", "Unknown split or source", line=meta_line)
            line, offset, raw, row = _next_record(streams[split], lines[split])
            lines[split] = line
            try:
                validate_record(row)
                if set(row) != {"messages", "images", "tools"}:
                    raise ValueError()
                call = row["messages"][-1]["tool_calls"][0]
                if not isinstance(call.get("id"), str) or not call["id"]:
                    raise ValueError()
                function = call["function"]
                arguments = loads(function["arguments"])
                specs = {t["function"]["name"]: t["function"]["parameters"] for t in row["tools"]}
                validate_arguments(arguments, specs[function["name"]])
                if (call["id"] != meta["call_id"] or function["name"] != meta["target_tool"]
                        or arguments.get("primitive") != meta.get("primitive")
                        or len(row["images"]) != meta["image_count"]
                        or not row["messages"][-1]["content"].startswith("<think>\n\n</think>")):
                    raise ValueError()
                if type(meta["decision_index"]) is not int or meta["decision_index"] < 0:
                    raise ValueError()
                source = sources[source_id]
                scene = source["scene"]
                if not isinstance(scene, str) or not scene or scene != meta["scene"]:
                    raise ValueError()
                if row["messages"][1]["content"] != source["run_config"]["instruction"]:
                    raise ValueError()
                if any(not isinstance(p, str) or p not in images for p in row["images"]):
                    raise ValueError()
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                raise TrainingError("package_provenance_mismatch", "Row, schema and source metadata disagree",
                                    split=split, line=line, metadata_line=meta_line) from exc
            identity = fingerprint([source_id, meta["decision_index"], meta["call_id"]])
            if identity in seen:
                raise TrainingError("duplicate_sample", "Sample identity is duplicated")
            for mapping, key in ((source_splits, source_id), (scene_splits, scene)):
                if key in mapping and mapping[key] != split:
                    raise TrainingError("split_leakage", "A scene or trajectory occurs in both splits")
                mapping[key] = split
            seen.add(identity)
            used_images.update(row["images"])
            counts[split] += 1
            scene_counts[scene] += 1
            samples.append(SampleRef(identity, source_id, meta["decision_index"], meta["call_id"], split,
                                     line, offset, hashlib.sha256(raw).hexdigest(), function["name"],
                                     meta.get("primitive"), scene, source.get("task_instance", {}).get("task"),
                                     len(row["images"]), has_failed_history(row)))
        for split, stream in streams.items():
            if any(line.strip() for line in stream):
                raise TrainingError("extra_package_rows", "Dataset has rows without source metadata", split=split)
    finally:
        for stream in streams.values():
            stream.close()
    if (not all(counts[s] for s in streams) or dict(counts) != stats.get("splits")
            or len(samples) != stats.get("samples") or len(sources) != stats.get("eligible_episodes")
            or len(source_splits) != len(sources) or len(images) != stats.get("unique_images")
            or used_images != set(images) or dict(scene_counts) != stats.get("scene_samples")):
        raise TrainingError("package_count_mismatch", "Package counts disagree with its manifests")
    index = PackageIndex(root, samples, sources, images, image_paths, files, config, stats)
    index.assert_unchanged()
    return index


def materialize(index, output, samples=None):
    chosen = list(index.samples if samples is None else samples)
    output = Path(output)
    output.mkdir(exist_ok=False)
    index.assert_unchanged(images=True)
    handles = {split: (output / (split + ".jsonl")).open("x", encoding="utf-8") for split in ("train", "val")}
    try:
        for sample, row in index.records(chosen, absolute_images=True):
            # Fingerprints use sorted JSON; schema rendering must preserve the
            # original dictionary insertion order to preserve prompt tokens.
            handles[sample.split].write(dumps(row) + "\n")
    finally:
        for stream in handles.values():
            stream.close()
    index.assert_unchanged()
    manifest = {"package": index.manifest(), "samples": [asdict(s) for s in chosen],
                "splits": dict(Counter(s.split for s in chosen)),
                "files": {s: file_hash(output / (s + ".jsonl")) for s in ("train", "val")}}
    write_json(output / "manifest.json", manifest)
    return manifest


def select_smoke(index, reports, config):
    reports = {r["sample_id"]: r for r in reports}
    selected, reasons = [], {}
    eligible = [s for s in index.samples if s.sample_id in reports
                and reports[s.sample_id].get("status") == "passed"
                and reports[s.sample_id]["tokens"] <= config["data"]["max_length"]
                and reports[s.sample_id]["tokens"] <= reports[s.sample_id]["model_max_length"]]

    def choose(candidates, reason, *, longest=False):
        if not candidates:
            raise TrainingError("smoke_coverage_gap", "No eligible sample for required coverage", requirement=reason)
        sample = sorted(candidates, key=lambda s: ((-1 if longest else 1) * reports[s.sample_id]["tokens"], s.sample_id))[0]
        if sample.sample_id not in reasons:
            selected.append(sample)
            reasons[sample.sample_id] = []
        reasons[sample.sample_id].append(reason)

    for split in ("train", "val"):
        pool = [s for s in eligible if s.split == split]
        tools = ("initialize", "list_skills", "read_skill", "task_notes", "act", "look", "finish") if split == "train" else ("initialize", "act", "finish")
        for tool in tools:
            choose([s for s in pool if s.target_tool == tool and (split != "val" or tool != "act" or s.image_count >= 8)], split + ":" + tool)
        if split == "train":
            choose([s for s in pool if not s.image_count], "zero_images")
            choose([s for s in pool if s.image_count >= 8], "multiple_captures")
            choose([s for s in pool if s.has_failed_history], "failed_history")
        choose(pool, split + ":longest", longest=True)
        count = config["smoke"][split + "_samples"]
        if sum(s.split == split for s in selected) > count or len(pool) < count:
            raise TrainingError("smoke_coverage_gap", "Not enough distinct eligible smoke samples", split=split)
        for sample in sorted(pool, key=lambda s: hashlib.sha256(("42:" + s.sample_id).encode()).hexdigest()):
            if sum(s.split == split for s in selected) >= count:
                break
            if sample.sample_id not in reasons:
                selected.append(sample)
                reasons[sample.sample_id] = ["seeded_fill"]
    return selected, {"algorithm": "shortest_per_requirement_then_longest_then_sha256_42", "seed": 42,
                      "samples": [{**asdict(s), "tokens": reports[s.sample_id]["tokens"], "reasons": reasons[s.sample_id]} for s in selected],
                      "input_samples": len(index.samples), "eligible_samples": len(eligible),
                      "excluded_overlength": sum(r.get("status") == "passed" and not r.get("within_budget", False) for r in reports.values())}
