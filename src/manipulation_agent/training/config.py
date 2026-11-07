"""Typed TOML recipes, composed without importing optional ML libraries."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import math
import os
import re
import sys
import tomllib

from .artifacts import TrainingError, fingerprint

DEFAULTS = {
    "model": {"model_type": "qwen3_5", "torch_dtype": "bfloat16"},
    "template": {"template": "qwen3_5", "agent_template": "qwen3_5", "template_backend": "swift",
                 "enable_thinking": True, "preserve_thinking": True, "add_non_thinking_prefix": False,
                 "loss_scale": "default", "disable_ignore_empty_think": True},
    "vision": {"image_min_token_num": 4, "image_max_token_num": 1024},
    # Preserve every image in the package; source-history trimming is recorded
    # by the converter's versioned context policy, never performed by training.
    "data": {"max_length": 16384, "overlength_policy": "error", "k": 2, "keep_all_images": True,
             "history_compression": "legacy"},
    "training": {"tuner_type": "lora", "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.05,
                 "freeze_vit": True, "freeze_aligner": True, "target_modules": "all-linear",
                 "learning_rate": 1e-4, "lr_scheduler_type": "cosine", "warmup_ratio": 0.05,
                 "per_device_train_batch_size": 1, "per_device_eval_batch_size": 1,
                 "gradient_accumulation_steps": 8, "num_train_epochs": 3, "max_steps": -1,
                 "gradient_checkpointing": True, "packing": False, "padding_free": False,
                 "seed": 42, "data_seed": 42, "dataset_shuffle": True,
                 "train_dataloader_shuffle": True, "group_by_length": False,
                 "dataloader_drop_last": False, "eval_strategy": "steps", "save_strategy": "steps",
                 "eval_steps": 40, "save_steps": 40, "logging_steps": 1, "save_total_limit": 3,
                 "load_best_model_at_end": True, "metric_for_best_model": "loss",
                 "greater_is_better": False, "report_to": "none", "strict": True,
                 "split_dataset_ratio": 0.0, "lazy_tokenize": True, "ignore_data_skip": False},
    "launch": {"nproc_per_node": 1, "attn_impl": "sdpa", "ddp_backend": "nccl",
               "use_liger_kernel": False, "dataset_num_proc": 1, "dataloader_num_workers": 0},
    "smoke": {"enabled": False, "train_samples": 16, "val_samples": 8, "stop_after_step": 0},
    "wandb": {"project": "agentic_distillation", "entity": "qili0502-zhejiang-university",
              "mode": "online", "name": "", "group": "", "tags": ["qwen3.5-4b", "sft", "lora"]},
}


def config_root():
    local = Path(__file__).resolve().parents[3] / "configs/training"
    return local if local.is_dir() else Path(sys.prefix) / "share/manipulation-agentic-system/training"


def _read(path):
    try:
        with Path(path).open("rb") as stream:
            return tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise TrainingError("invalid_recipe", "Cannot read TOML configuration", path=str(path)) from exc


def _merge(config, overlay, *, launch_only=False):
    for section, fields in overlay.items():
        if section not in DEFAULTS or not isinstance(fields, dict) or (launch_only and section != "launch"):
            raise TrainingError("unknown_config_section", "Configuration section is not allowed", section=section)
        for key, value in fields.items():
            expected = DEFAULTS[section].get(key)
            if key not in DEFAULTS[section]:
                raise TrainingError("unknown_config_field", "Unknown configuration field", field=section + "." + key)
            if type(value) is not type(expected):
                # TOML allows integral numbers for a floating-point setting.
                if type(expected) is float and type(value) is int:
                    value = float(value)
                else:
                    raise TrainingError("invalid_config_type", "Configuration value has wrong type", field=section + "." + key)
            config[section][key] = value


def validate(config):
    from ..sft.history_compaction import validate_history_compression
    validate_history_compression(config["data"].get("history_compression", "legacy"))
    _merge(deepcopy(DEFAULTS), config)
    if any(type(v) is float and not math.isfinite(v) for fields in config.values() for v in fields.values()):
        raise TrainingError("invalid_config_value", "Configuration numbers must be finite")
    for section, keys in (("data", ("max_length", "k")), ("vision", ("image_min_token_num", "image_max_token_num")),
                          ("launch", ("nproc_per_node", "dataset_num_proc")),
                          ("training", ("lora_rank", "lora_alpha", "per_device_train_batch_size",
                                        "per_device_eval_batch_size", "gradient_accumulation_steps", "save_steps", "eval_steps"))):
        for key in keys:
            if config[section][key] <= 0:
                raise TrainingError("invalid_config_value", "Expected positive configuration value", field=section + "." + key)
    required = {"model": {"model_type": "qwen3_5", "torch_dtype": "bfloat16"},
                "template": DEFAULTS["template"],
                "data": {"overlength_policy": "error", "keep_all_images": True},
                "training": {"tuner_type": "lora", "freeze_vit": True, "freeze_aligner": True,
                             "strict": True, "packing": False, "padding_free": False,
                             "split_dataset_ratio": 0.0, "dataloader_drop_last": False}}
    for section, fields in required.items():
        for key, value in fields.items():
            if config[section][key] != value:
                raise TrainingError("unsupported_training_contract", "Setting conflicts with the supported data contract",
                                    field=section + "." + key)
    if (config["vision"]["image_min_token_num"] > config["vision"]["image_max_token_num"]
            or config["launch"]["nproc_per_node"] not in (1, 2)
            or config["launch"]["dataloader_num_workers"] < 0
            or not 0 <= config["training"]["lora_dropout"] < 1
            or config["training"]["learning_rate"] <= 0
            or not 0 <= config["training"]["warmup_ratio"] <= 1
            or config["training"]["max_steps"] == 0 or config["training"]["max_steps"] < -1
            or config["training"]["num_train_epochs"] <= 0):
        raise TrainingError("invalid_config_value", "Invalid vision, launch or optimizer configuration")
    if config["smoke"]["enabled"] and config["launch"]["nproc_per_node"] != 1:
        raise TrainingError("unsupported_smoke_launch", "Smoke validation requires one GPU")
    if config["smoke"]["enabled"]:
        expected = {"max_steps": 4, "gradient_accumulation_steps": 4, "per_device_train_batch_size": 1,
                    "dataset_shuffle": False, "train_dataloader_shuffle": False, "group_by_length": False,
                    "ignore_data_skip": False, "load_best_model_at_end": False}
        if (any(config["training"][k] != v for k, v in expected.items())
                or config["smoke"]["train_samples"] != 16 or config["smoke"]["val_samples"] != 8
                or config["smoke"]["stop_after_step"] not in (0, 2)):
            raise TrainingError("invalid_smoke_schedule", "Smoke requires the fixed 16/8 sample, four-step schedule")
    tracking = config.get("wandb", DEFAULTS["wandb"])
    if config["training"]["report_to"] not in ("none", "wandb") or tracking["mode"] not in ("online", "offline", "disabled"):
        raise TrainingError("invalid_tracking_config", "Use report_to=none/wandb and wandb.mode=online/offline/disabled")
    if (not re.fullmatch(r"[A-Za-z0-9_-]+", tracking["project"])
            or not re.fullmatch(r"[A-Za-z0-9_-]*", tracking["entity"])
            or any(not isinstance(tag, str) or not tag.strip() for tag in tracking["tags"])
            or any(any(ord(c) < 32 for c in tracking[k]) for k in ("name", "group"))):
        raise TrainingError("invalid_tracking_config", "Invalid W&B project, entity, name, group or tags")
    return config


def load_config(recipe=None, *, launch=None, smoke=False):
    config = deepcopy(DEFAULTS)
    _merge(config, _read(recipe or config_root() / "qwen35_4b_lora.toml"))
    if smoke:
        _merge(config, _read(config_root() / "smoke.toml"))
    if launch:
        _merge(config, _read(launch), launch_only=True)
    # Only named, non-secret settings enter the resolved configuration. Resume
    # reads the saved contract directly and never reapplies ambient overrides.
    for field, variable in WANDB_ENVIRONMENT.items():
        if variable in os.environ:
            config["wandb"][field] = os.environ[variable]
    return validate(config)


WANDB_ENVIRONMENT = {"project": "WANDB_PROJECT", "entity": "WANDB_ENTITY", "mode": "WANDB_MODE",
                     "name": "WANDB_NAME", "group": "WANDB_RUN_GROUP"}


def encoding_contract(config):
    from .swift_loss import NATIVE_LOSS_SCALE
    # Optimizer/launch/length-budget changes do not change untruncated tokens.
    result = {key: config[key] for key in ("model", "template", "vision")} | {
        "k": config["data"]["k"], "keep_all_images": config["data"]["keep_all_images"],
        "native_loss_scale": NATIVE_LOSS_SCALE}
    if config["data"].get("history_compression", "legacy") != "legacy":
        result["history_compression"] = config["data"]["history_compression"]
    return result


def config_fingerprint(config):
    return fingerprint(validate(config))
