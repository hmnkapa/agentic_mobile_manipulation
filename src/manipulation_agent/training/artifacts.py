"""Standard-library run records and immutable input fingerprints."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import uuid

from ..sft.common import ConversionError, file_hash, read_json


class TrainingError(ConversionError):
    """A safe diagnostic; never include arbitrary environment or model payloads."""


class ProcessFailure(TrainingError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":"))


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def encoder_source():
    root = Path(__file__).resolve().parent
    files = [root / name for name in ("encoding.py", "swift_backend.py", "swift_loss.py", "data.py", "config.py")]
    files += [root.parent / "sft" / name for name in ("format.py", "protocol.py", "history.py", "history_compaction.py", "common.py")]
    return {str(p.relative_to(root.parent)): file_hash(p) for p in files}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def journal(root, event, **fields):
    with (Path(root) / "journal.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(canonical({"at": now(), "event": event, **fields}) + "\n")


def create_output(path, *, protected=()):
    output = Path(path).resolve()
    for source in protected:
        source = Path(source).resolve()
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise TrainingError("output_overlaps_input", "Output must be separate from inputs", path=str(output))
    if output.exists():
        raise TrainingError("output_exists", "Choose a new output directory", path=str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    return output


PACKAGES = ("ms-swift", "torch", "transformers", "peft", "accelerate", "datasets",
            "qwen-vl-utils", "pillow", "torchvision", "decord")


def versions():
    result = {re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower(): d.version
              for d in metadata.distributions() if d.metadata["Name"]}
    for package in PACKAGES:
        result.setdefault(package, None)
    return result


def environment():
    from ..records import source_version
    result = {"python": sys.executable, "python_version": platform.python_version(),
              "host": platform.node(), "pid": os.getpid(), "packages": versions(),
              "source": source_version(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "unit": os.environ.get("MANIP_SFT_UNIT")}
    result["allocator_environment"] = {k: os.environ[k] for k in
        ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF") if k in os.environ}
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        result["selected_gpus"] = [{"logical_index": i, "name": torch.cuda.get_device_name(i),
                                    "uuid": str(getattr(torch.cuda.get_device_properties(i), "uuid", "unknown"))}
                                   for i in range(torch.cuda.device_count())]
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,driver_version",
                                "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        result["gpu_inventory"] = query.stdout.strip().splitlines() if query.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired):
        result["gpu_inventory"] = []
    return result


def model_manifest(model, *, require_weights=False):
    root = Path(model).resolve()
    if not root.is_dir():
        raise TrainingError("missing_model", "Provide an existing local model directory", path=str(root))
    names = ("config.json", "tokenizer_config.json", "tokenizer.json", "preprocessor_config.json",
             "processor_config.json", "video_preprocessor_config.json", "chat_template.jinja",
             "special_tokens_map.json", "generation_config.json", "model.safetensors.index.json")
    hashes = {name: file_hash(root / name) for name in names if (root / name).is_file()}
    if not {"config.json", "tokenizer_config.json"}.issubset(hashes):
        raise TrainingError("incomplete_model", "Model config and tokenizer configuration are required")
    revision = root.name if root.parent.name == "snapshots" else None
    declared = {}
    if (root / "model_manifest.json").is_file():
        declaration = read_json(root / "model_manifest.json")
        revision = declaration.get("revision", revision)
        declared = {k: v for k, v in declaration.get("files", declaration.get("expected_weights", {})).items()
                    if k.endswith(".safetensors")}
    # Hash every weight shard. A local directory without a Hub revision must
    # still have an identity that cannot silently change between encode/train.
    weights = {p.name: file_hash(p) for p in sorted(root.glob("*.safetensors"))}
    if declared and any(name not in declared or declared[name] != sha for name, sha in weights.items()):
        raise TrainingError("model_hash_mismatch", "Downloaded weights differ from the pinned snapshot")
    if require_weights and (not weights or (declared and weights != declared)):
        raise TrainingError("incomplete_model_weights", "Training/export requires all verified model weight shards")
    identity_weights = declared or weights
    return {"root": str(root), "revision": revision, "files": hashes, "weights": identity_weights,
            "weights_verified": bool(weights) and (not declared or weights == declared),
            "fingerprint": fingerprint({"revision": revision, "files": hashes, "weights": identity_weights})}
