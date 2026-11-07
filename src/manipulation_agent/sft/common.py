"""Strict JSON and small, structured diagnostics for offline conversion."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


class ConversionError(ValueError):
    def __init__(self, code: str, message: str, **location):
        super().__init__(message)
        self.code, self.location = code, location

    def record(self):
        return {"code": self.code, "message": str(self), **self.location}


def dumps(value, **kwargs):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, **kwargs)


def loads(text):
    def invalid_constant(value):
        raise ValueError("Non-finite JSON number")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result

    def finite_float(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("Non-finite JSON number")
        return value

    return json.loads(text, parse_constant=invalid_constant, parse_float=finite_float,
                      object_pairs_hook=unique_keys)


def read_json(path: Path):
    try:
        value = loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Expected an object")
        return value
    except (OSError, UnicodeError, ValueError) as exc:
        raise ConversionError("invalid_json", "Cannot read a JSON object", path=str(path)) from exc


def jsonl(path: Path):
    try:
        with path.open(encoding="utf-8") as stream:
            for line, text in enumerate(stream, 1):
                if not text.strip():
                    continue
                try:
                    value = loads(text)
                    if not isinstance(value, dict):
                        raise ValueError("Expected an object")
                except ValueError as exc:
                    raise ConversionError("invalid_jsonl", "Invalid JSONL object",
                                          path=str(path), line=line) from exc
                yield line, value
    except (OSError, UnicodeError) as exc:
        raise ConversionError("unreadable_file", "Cannot read source file", path=str(path)) from exc


def file_hash(path: Path):
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ConversionError("unreadable_file", "Cannot hash source file", path=str(path)) from exc
    return digest.hexdigest()


def inside(root: Path, relative: str):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ConversionError("path_outside_root", "Artifact path leaves its source directory")
    return path
