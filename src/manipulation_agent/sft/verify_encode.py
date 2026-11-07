"""Optional offline Qwen3.5/ms-swift encoding check; never load model weights."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .common import ConversionError, dumps, inside, jsonl
from .format import validate_record
from ..training.encoding import check_encoding




def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True, help="Exported dataset/train/val JSONL")
    parser.add_argument("--model", type=Path, required=True, help="Local Qwen3.5-4B model/processor directory")
    parser.add_argument("--index", type=int, default=0, help="Zero-based record index")
    args = parser.parse_args(argv)
    if args.index < 0:
        parser.error("--index must be nonnegative")
    try:
        if not args.model.is_dir():
            raise ConversionError("missing_processor", "Provide an existing local Qwen3.5-4B processor directory")
        row = next((row for i, (_, row) in enumerate(jsonl(args.record)) if i == args.index), None)
        if row is None:
            raise ConversionError("missing_record", "Record index exceeds the JSONL length")
        validate_record(row)
        root = args.record.resolve().parent
        for i, relative in enumerate(row["images"]):
            path = inside(root, relative)
            if not path.is_file():
                raise ConversionError("missing_image", "Package image does not exist", path=str(path))
            row["images"][i] = str(path)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from ..training.config import load_config
        from ..training.swift_backend import build_encoding_backend
        backend = build_encoding_backend(load_config(), args.model, unbounded=True)
        print(dumps(check_encoding(row, backend.template, backend.processor,
                                   preprocessor=backend.preprocessor, render_target=backend.render_target)))
        return 0
    except ConversionError as exc:
        print(dumps({"status": "failed", "error": exc.record()}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
