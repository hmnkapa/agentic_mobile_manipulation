"""Convert recorded Codex RGB episodes to Qwen3.5 ms-swift SFT data."""
import argparse
from pathlib import Path

from .common import ConversionError, dumps
from .dataset import manifest_sources, prepare_dataset, write_package
from .history import DEFAULT_MAX_LEN
from .history_compaction import HISTORY_COMPRESSION_POLICIES, LEGACY_HISTORY
from .source import EpisodeSource


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path, help="Single episode directory containing run.json")
    source.add_argument("--manifest", type=Path, help="JSONL rows with run_dir and controller_dir; paths relative to manifest")
    parser.add_argument("--controller-dir", type=Path, help="Controller directory paired with --run-dir")
    parser.add_argument("--out-dir", type=Path, help="New data package directory; omit for read-only validation")
    parser.add_argument("--k", type=int, default=2, help="Full-history window in RGB-bearing replies (default: 2)")
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN,
                        help="Approximate token budget for trimming old context (default: 24576; 0 disables)")
    parser.add_argument("--history-compression", choices=HISTORY_COMPRESSION_POLICIES, default=LEGACY_HISTORY)
    args = parser.parse_args(argv)
    if bool(args.run_dir) != bool(args.controller_dir):
        parser.error("--run-dir and --controller-dir must be supplied together")
    try:
        sources = manifest_sources(args.manifest) if args.manifest else [EpisodeSource(args.run_dir, args.controller_dir)]
        data = prepare_dataset(sources, batch=bool(args.manifest), k=args.k, max_len=args.max_len,
                               history_compression=args.history_compression)
        output = write_package(data, args.out_dir, source_manifest=args.manifest) if args.out_dir else None
        print(dumps({"status": "complete" if output else "checked", "dry_run": output is None,
                     "output": str(output) if output else None, "stats": data.stats,
                     "errors": data.errors, "skipped": data.skipped}))
        return 0
    except (ConversionError, OSError) as exc:
        error = exc.record() if isinstance(exc, ConversionError) else {"code": "io_error", "error_type": type(exc).__name__}
        print(dumps({"status": "failed", "error": error}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
