"""Validate public four-camera RGB content against archived image bytes."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
from pathlib import Path
from .common import ConversionError, inside, jsonl
from ..observations.boundary import public_observation

VIEWS = {"front", "back", "left", "right"}
MIME_EXTENSIONS = {"image/jpeg": ".jpg", "image/png": ".png"}


@dataclass(frozen=True)
class ImageAsset:
    source: Path
    relative: str
    sha256: str
    mime_type: str


def _captures(run_dir):
    path = run_dir / "captures.jsonl"
    if not path.exists():
        return {}
    result = {}
    for line, capture in jsonl(path):
        for image in capture.get("images", []):
            ref = image.get("image_ref")
            if not ref or (ref in result and result[ref] != image):
                raise ConversionError("ambiguous_image_ref", "Capture has a missing or conflicting image_ref", line=line)
            result[ref] = image
    return result


def public_content(item, result, run_dir, captures, line):
    blocks = (item.get("result") or {}).get("content", [])
    if not isinstance(blocks, list):
        raise ConversionError("invalid_content", "MCP content must be a list", line=line)
    observation = result.get("observation") or {}
    metadata = observation.get("images", [])
    image_blocks = [block for block in blocks if block.get("type") == "image"]
    if metadata:
        try:
            public_observation({key: value for key, value in observation.items() if key != "revision"},
                               observation["revision"])
        except (RuntimeError, KeyError, TypeError) as exc:
            raise ConversionError("observation_boundary_violation", "Observation contains unsupported or private fields", line=line) from exc
        if (observation.get("observation_mode") != "rgb_only" or len(metadata) != 4
                or {m.get("view") for m in metadata} != VIEWS
                or len({m.get("image_ref") for m in metadata}) != 4):
            raise ConversionError("not_four_camera_rgb", "Expected the four recorded robot RGB views", line=line)
    if len(metadata) != len(image_blocks):
        raise ConversionError("image_count_mismatch", "MCP images do not match observation metadata", line=line)
    content, index, assets = [], 0, {}
    for block in blocks:
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            content.append({"type": "text", "text": block["text"]})
            continue
        if block.get("type") != "image":
            raise ConversionError("unknown_content_block", "Unsupported model-facing MCP block", line=line)
        meta = metadata[index]
        index += 1
        ref, mime = meta.get("image_ref"), meta.get("mime_type")
        archived = captures.get(ref)
        if archived is None or mime not in MIME_EXTENSIONS:
            raise ConversionError("missing_image_archive", "Image has no supported capture record", line=line)
        path = inside(run_dir, archived["file"])
        try:
            payload = base64.b64decode(block["data"], validate=True)
            data = path.read_bytes()
        except (OSError, ValueError, KeyError) as exc:
            raise ConversionError("unreadable_image", "Cannot read recorded image bytes", line=line, image_ref=ref) from exc
        sha = hashlib.sha256(payload).hexdigest()
        if (payload != data or sha != meta.get("sha256") or sha != archived.get("sha256")
                or block.get("mimeType") != mime):
            raise ConversionError("image_hash_mismatch", "Delivered and archived image bytes disagree", line=line, image_ref=ref)
        label = f"RGB view={meta['view']}, image_ref={ref}"
        if not content or content[-1] != {"type": "text", "text": label}:
            raise ConversionError("image_label_mismatch", "Image view/ref label does not match metadata", line=line)
        relative = "images/" + sha + MIME_EXTENSIONS[mime]
        assets[relative] = ImageAsset(path, relative, sha, mime)
        content.append({"type": "image", "image": relative})
    return content, assets


