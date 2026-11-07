"""Native ms-swift CLI with an owned W&B run and explicit failure cleanup."""
from pathlib import Path
import os
import runpy
import signal

from .artifacts import TrainingError, canonical, journal
from .tracking import finish_tracking, start_tracking


def _terminate(signum, frame):
    # The controller terminates only its owned process group. Let this worker
    # run its bounded SDK cleanup when it receives that termination request.
    raise KeyboardInterrupt


def main():
    root = Path(os.environ["MANIP_SFT_RUN_ROOT"])
    tracker = None
    exit_code = 0
    previous_term = signal.signal(signal.SIGTERM, _terminate)
    try:
        tracker = start_tracking(root)
        # Execute the original CLI so its single-device/Unsloth/Ray setup and
        # argument handling stay identical to the pinned ms-swift entry point.
        runpy.run_module("swift.cli.sft", run_name="__main__")
    except BaseException as exc:
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        if isinstance(exc, SystemExit):
            exit_code = exc.code if isinstance(exc.code, int) else 1 if exc.code else 0
        raise
    finally:
        try:
            finish_tracking(root, tracker, exit_code=exit_code)
        except Exception as exc:
            journal(root, "wandb_finish_failed", error_type=type(exc).__name__)
            if not exit_code:
                raise TrainingError("wandb_finish_failed", "W&B did not finish cleanly", error_type=type(exc).__name__) from None
        finally:
            signal.signal(signal.SIGTERM, previous_term)


if __name__ == "__main__":
    try:
        main()
    except TrainingError as exc:
        print(canonical({"status": "failed", "error": exc.record()}))
        raise SystemExit(2)
