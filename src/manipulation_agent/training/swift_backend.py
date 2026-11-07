"""The version-pinned ms-swift adapter. Heavy imports are local to execution."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys

from .artifacts import TrainingError, versions
from .swift_loss import NATIVE_LOSS_SCALE, register_loss

SWIFT_VERSION = "4.6.0"
_INITIALIZED_VISION = None


def require_framework():
    installed = versions()
    if installed["ms-swift"] != SWIFT_VERSION:
        raise TrainingError("unsupported_ms_swift", "Use the isolated environment with ms-swift==4.6.0",
                            installed=installed["ms-swift"])
    for name in ("torch", "transformers", "qwen-vl-utils"):
        if installed[name] is None:
            raise TrainingError("missing_training_dependency", "Required training dependency is absent", package=name)


def vision_environment(config, base=None):
    env = dict(os.environ if base is None else base)
    if any(env.get(k) for k in ("MAX_PIXELS", "MIN_PIXELS")):
        raise TrainingError("conflicting_image_environment", "Unset legacy MAX_PIXELS/MIN_PIXELS before this command")
    env.update(IMAGE_MIN_TOKEN_NUM=str(config["vision"]["image_min_token_num"]),
               IMAGE_MAX_TOKEN_NUM=str(config["vision"]["image_max_token_num"]),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    return env


@dataclass
class EncodingBackend:
    arguments: object
    processor: object
    template: object
    preprocessor: object
    model_max_length: int

    def render_target(self, function):
        from ..sft.common import loads
        call = {"name": function["name"], "arguments": loads(function["arguments"])}
        # This is the same pinned formatter called by Template._preprocess_tool_call.
        return self.template.agent_template._format_tool_calls([{"role": "tool_call", "content": call}])


def build_encoding_backend(config, model, *, unbounded=True):
    global _INITIALIZED_VISION
    require_framework()
    vision = tuple(sorted(config["vision"].items()))
    if _INITIALIZED_VISION is not None and vision != _INITIALIZED_VISION:
        raise TrainingError("vision_environment_already_initialized", "Changed image settings require a fresh interpreter")
    env = vision_environment(config)
    for key in ("IMAGE_MIN_TOKEN_NUM", "IMAGE_MAX_TOKEN_NUM", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM"):
        os.environ[key] = env[key]
    try:
        from swift import BaseArguments
        from swift.dataset import MessagesPreprocessor
        register_loss()
        _INITIALIZED_VISION = vision
        template_settings = {**config["template"], "loss_scale": NATIVE_LOSS_SCALE}
        arguments = BaseArguments(model=str(Path(model).resolve()), use_hf=True, device_map="cpu",
                                  attn_impl="sdpa", max_length=config["data"]["max_length"],
                                  **config["model"], **template_settings)
        _, processor = arguments.get_model_processor(load_model=False)
        template = arguments.get_template(processor)
        template.set_mode("train")
        if arguments.loss_scale != NATIVE_LOSS_SCALE:
            raise TrainingError("framework_changed_loss_scale", "Framework changed the requested supervision policy")
        maximum = int(arguments.model_info.max_model_len)
        if unbounded:
            # init_processor fills a None limit from model metadata. Clear it
            # only after initialization, on this measurement-only instance.
            template.max_length = None
        return EncodingBackend(arguments, processor, template, MessagesPreprocessor(), maximum)
    except TrainingError:
        raise
    except (ImportError, TypeError, ValueError, AttributeError, OSError) as exc:
        raise TrainingError("encoding_backend_unavailable", "Cannot initialize the pinned Qwen processor/template",
                            error_type=type(exc).__name__) from exc


def load_prepared_datasets(backend, prepared):
    args = backend.arguments
    args.dataset = [str(Path(prepared) / "train.jsonl")]
    args.val_dataset = [str(Path(prepared) / "val.jsonl")]
    args.dataset_shuffle = False
    args.val_dataset_shuffle = False
    args.split_dataset_ratio = 0.0
    args.strict = True
    args.dataset_num_proc = 1
    args.load_from_cache_file = False
    train, val = args.load_dataset()
    return {"train": train, "val": val}


def _arguments(fields):
    argv = []
    for name, value in fields.items():
        value = str(value).lower() if type(value) is bool else str(value)
        argv.extend(["--" + name, value])
    return argv


def training_command(config, model, prepared, output, *, resume=None):
    from .tracking import enabled
    tracked = enabled(config)
    fields = {**config["model"], **config["template"], **config["training"],
              **{k: v for k, v in config["launch"].items() if k != "nproc_per_node"},
              "model": str(Path(model).resolve()), "use_hf": True,
              "dataset": str(Path(prepared) / "train.jsonl"), "val_dataset": str(Path(prepared) / "val.jsonl"),
              "max_length": config["data"]["max_length"], "truncation_strategy": "delete",
              "loss_scale": NATIVE_LOSS_SCALE,
              "load_from_cache_file": False, "add_version": False, "output_dir": str(output),
              "external_plugins": str(Path(__file__).with_name("swift_plugin.py")),
              "callbacks": "manipulation_sft_audit"}
    fields["report_to"] = "wandb" if tracked else "none"
    if resume:
        fields["resume_from_checkpoint"] = str(resume)
    nproc = config["launch"]["nproc_per_node"]
    module = "manipulation_agent.training.worker" if tracked else "swift.cli.sft"
    command = [sys.executable, "-m", module] if nproc == 1 else [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(nproc),
        "--module", module]
    return command + _arguments(fields)


def export_command(base, adapter, output):
    return [sys.executable, "-m", "swift.cli.export"] + _arguments({
        "model": str(base), "model_type": "qwen3_5", "use_hf": True, "adapters": str(adapter),
        "merge_lora": True, "attn_impl": "sdpa", "torch_dtype": "bfloat16",
        "output_dir": str(output), "exist_ok": False, "safe_serialization": True})


def require_gpu(config):
    require_framework()
    import torch
    count = config["launch"]["nproc_per_node"]
    if not torch.cuda.is_available() or torch.cuda.device_count() < count:
        raise TrainingError("gpu_unavailable", "Required CUDA GPUs are unavailable", required=count)
    for device in range(count):
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise TrainingError("bf16_unavailable", "Selected GPU does not support BF16", device=device)
            free, total = torch.cuda.mem_get_info()
            if free < 12 * 1024 ** 3:
                raise TrainingError("insufficient_gpu_memory", "Selected GPU has insufficient free memory; preserve existing jobs",
                                    device=device, free_bytes=free, total_bytes=total)
