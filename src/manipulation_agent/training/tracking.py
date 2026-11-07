"""W&B identity and lifecycle; importing this module needs only the standard library."""
from __future__ import annotations

from importlib.util import find_spec
import os
from pathlib import Path
import sys
import uuid

from ..sft.common import read_json
from .artifacts import TrainingError, journal, now, write_json
from .config import DEFAULTS


def enabled(config):
    return (config["training"]["report_to"] == "wandb"
            and config.get("wandb", DEFAULTS["wandb"])["mode"] != "disabled")


def require_tracking(config):
    if enabled(config) and find_spec("wandb") is None:
        raise TrainingError("missing_wandb", "Install requirements/training-wandb.in in the SFT environment")


def tracking_plan(config, root, *, parent=None):
    if not enabled(config):
        return {"enabled": False}
    settings = config["wandb"]
    run_id = uuid.uuid4().hex[:12]
    plan = {"enabled": True, "mode": settings["mode"], "entity": settings["entity"],
            "project": settings["project"], "id": run_id,
            "name": settings["name"] or Path(root).name,
            "group": settings["group"] or run_id, "tags": settings["tags"],
            "resume": "never" if settings["mode"] == "online" else None, "parent_id": None}
    if parent:
        previous = read_json(Path(parent) / "training_contract.json").get("wandb", {})
        if not previous.get("enabled") or any(previous[k] != plan[k] for k in ("mode", "entity", "project")):
            raise TrainingError("resume_tracking_mismatch", "W&B identity is missing or differs from the saved contract")
        plan.update(parent_id=previous["id"], group=previous["group"])
        if settings["mode"] == "online":
            # Keep offline segments independently syncable with distinct IDs.
            # Online runs must exist; never silently replace a missing run.
            plan.update(id=previous["id"], name=previous["name"], resume="must")
    return plan


def tracking_environment(config, root, base):
    env = dict(base)
    if not enabled(config):
        return env
    # The immutable plan owns identity. Inherited interactive-shell settings
    # must not redirect a resume, upload model weights or record console secrets.
    for key in ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_SWEEP_ID", "WANDB_DISABLED",
                "WANDB_RUN_GROUP", "WANDB_NAME", "WANDB_TAGS", "WANDB_JOB_TYPE"):
        env.pop(key, None)
    settings = config["wandb"]
    env.update(WANDB_MODE=settings["mode"], WANDB_PROJECT=settings["project"],
               WANDB_ENTITY=settings["entity"], WANDB_DIR=str(root),
               WANDB_CACHE_DIR=str(Path(root) / "wandb-cache"), WANDB_DATA_DIR=str(Path(root) / "wandb-data"),
               # Pinned W&B 0.31 uses Forge for new logins and also reads legacy
               # api.wandb.ai credentials. The reverse lookup is not supported.
               WANDB_BASE_URL="https://api.forge.coreweave.com", WANDB_CONSOLE="off",
               WANDB_LOG_MODEL="false", WANDB_WATCH="false", WANDB_DISABLE_CODE="true",
               WANDB_DISABLE_GIT="true", WANDB_SILENT="true")
    return env


def start_tracking(root):
    """Called before model loading, on global rank zero only."""
    root = Path(root)
    contract = read_json(root / "training_contract.json")
    plan = contract.get("wandb", {"enabled": False})
    if not plan["enabled"] or int(os.environ.get("RANK", "0")) != 0:
        return None
    require_tracking(contract["config"])
    import wandb

    env = read_json(root / "environment.json")
    run_record = read_json(root / "run.json")
    metadata = {"resolved_config": contract["config"], "config_fingerprint": contract["config_fingerprint"],
                "dataset_fingerprint": contract["package"]["fingerprint"],
                "model_fingerprint": contract["model"]["fingerprint"], "model_revision": contract["model"]["revision"],
                "sample_count": len(contract["sample_ids"]), "splits": contract["expected_splits"],
                "encoder_source": contract["encoder_source"], "trainer_source": contract["trainer_source"],
                "packages": contract["packages"], "source": env["source"], "host": env["host"],
                "python": env["python"], "gpu_inventory": env["gpu_inventory"],
                "selected_gpus": env.get("selected_gpus", []), "unit": env.get("unit"),
                "worker_pid": os.getpid(), "run_root": str(root), "parent_run": run_record["parent_run"],
                "parent_wandb_id": plan["parent_id"], "expected_start_step": contract["expected_start_step"],
                "validation_scope": run_record["validation_scope"], "task_evaluation_performed": False}
    record = {**plan, "status": "initializing", "started_at": now()}
    write_json(root / "wandb.json", record)
    try:
        run = wandb.init(project=plan["project"], entity=plan["entity"] or None,
                         id=plan["id"], name=plan["name"], group=plan["group"],
                         tags=[*plan["tags"], run_record["validation_scope"]], job_type="sft",
                         mode=plan["mode"], resume=plan["resume"], dir=str(root),
                         config={"manipulation": metadata}, allow_val_change=True, save_code=False,
                         settings=wandb.Settings(init_timeout=60, login_timeout=30, finish_timeout=15,
                                                 finish_timeout_raises=True, console="off",
                                                 disable_code=True, disable_git=True))
    except Exception as exc:
        record.update(status="initialization_failed", error_type=type(exc).__name__)
        write_json(root / "wandb.json", record)
        raise TrainingError("wandb_initialization_failed",
                            "W&B initialization failed; check SDK/login/network or explicitly select offline mode",
                            error_type=type(exc).__name__) from None
    record.update(status="running", entity=run.entity, project=run.project,
                  url=run.url if plan["mode"] == "online" else None,
                  sync_dir=str(Path(run.dir).parent))
    write_json(root / "wandb.json", record)
    run.summary.update({"lifecycle/status": "running", "validation/level": "gpu_training_pending",
                        "validation/task_evaluation_performed": False})
    journal(root, "wandb_started", id=run.id, mode=plan["mode"], url=record["url"])
    return run


def log_checkpoint(item, best=None):
    """Only the auditing rank calls this after checkpoint hashes are verified."""
    sdk = sys.modules.get("wandb")
    run = getattr(sdk, "run", None)
    if run is not None:
        run.summary.update({"checkpoint/path": item["path"], "checkpoint/global_step": item["global_step"],
                            "checkpoint/fingerprint": item["fingerprint"], "checkpoint/best": best})


def finish_tracking(root, run, *, exit_code):
    if run is None:
        return
    root = Path(root)
    record = read_json(root / "wandb.json")
    status = "interrupted" if exit_code == 130 else "failed" if exit_code else "trainer_completed"
    # The controller validates immutable inputs after the worker returns.
    # A successful SDK/trainer exit is neither that validation nor task success.
    summary = {"lifecycle/status": status, "lifecycle/child_exit_code": exit_code,
               "validation/level": "gpu_training_pending_controller_validation"}
    try:
        audit_path = root / "audit.json"
        if audit_path.is_file():
            audit = read_json(audit_path)
            summary.update({"validation/trainer_audit": audit["status"],
                            "train/end_step": audit.get("end_step"),
                            "train/adapter_updated": audit.get("adapter_updated", False)})
        run.summary.update(summary)
    except BaseException:
        run.finish(exit_code=exit_code or 1)
        raise
    else:
        run.finish(exit_code=exit_code)
    record.update(status=status, child_exit_code=exit_code, finished_at=now())
    write_json(root / "wandb.json", record)
    journal(root, "wandb_finished", id=record["id"], status=status)


def tracking_result(root):
    path = Path(root) / "wandb.json"
    return read_json(path) if path.is_file() else {"enabled": False}
