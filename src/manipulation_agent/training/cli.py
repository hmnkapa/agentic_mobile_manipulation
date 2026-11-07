"""Validate, train, resume and export Qwen3.5-4B SFT artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..sft.common import ConversionError
from .artifacts import ProcessFailure, TrainingError, canonical
from .config import load_config, validate


def command_config(args, *, smoke=False):
    config = load_config(args.recipe, launch=getattr(args, "launch", None), smoke=smoke)
    if args.max_length is not None:
        config["data"]["max_length"] = args.max_length
    for field in ("mode", "project", "entity", "name", "group"):
        value = getattr(args, "wandb_" + field, None)
        if value is not None:
            config["wandb"][field] = value
    return validate(config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight")
    check.add_argument("--data", type=Path, required=True)
    check.add_argument("--recipe", type=Path)
    check.add_argument("--model", type=Path)
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--structure-only", action="store_true")
    check.add_argument("--encoding-workers", type=int, default=1)
    check.add_argument("--encoding-cache", type=Path)
    check.add_argument("--max-length", type=int, help="Explicit budget; complete preflight measurements remain untruncated")
    train = commands.add_parser("train")
    for flag in ("data", "recipe", "launch", "model", "resume-from"):
        train.add_argument("--" + flag, type=Path)
    train.add_argument("--preflight", type=Path, help="Optional completed encoding report; never launches a new preflight")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--max-length", type=int, help="Explicit context budget override, recorded in the immutable contract")
    train.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    for field in ("project", "entity", "name", "group"):
        train.add_argument("--wandb-" + field)
    export = commands.add_parser("export")
    export.add_argument("--run", type=Path, required=True)
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            from .preflight import preflight
            result = preflight(args.data, command_config(args), args.output,
                               model=args.model, structure_only=args.structure_only, encoding_workers=args.encoding_workers,
                               encoding_cache=args.encoding_cache)
        elif args.command == "train":
            from .run import resume, train
            if args.resume_from:
                if (any((args.data, args.recipe, args.launch, args.model, args.preflight, args.smoke))
                        or args.max_length is not None
                        or any(getattr(args, "wandb_" + k) is not None for k in ("mode", "project", "entity", "name", "group"))):
                    raise TrainingError("resume_override_forbidden", "Resume takes its parameters only from the saved contract")
                result = resume(args.resume_from, args.output, dry_run=args.dry_run)
            else:
                if not args.data or not args.model:
                    raise TrainingError("missing_training_input", "Training requires --data and --model")
                result = train(args.data, command_config(args, smoke=args.smoke),
                               args.model, args.preflight, args.output, dry_run=args.dry_run)
        else:
            from .export import export_model
            result = export_model(args.run, args.checkpoint, args.output, dry_run=args.dry_run)
        print(canonical(result))
        return 0
    except KeyboardInterrupt:
        print(canonical({"status": "interrupted"}))
        return 130
    except (ConversionError, OSError) as exc:
        error = exc.record() if isinstance(exc, ConversionError) else {"code": "io_error", "error_type": type(exc).__name__}
        print(canonical({"status": "failed", "error": error}))
        return 1 if isinstance(exc, ProcessFailure) else 2


if __name__ == "__main__":
    raise SystemExit(main())
