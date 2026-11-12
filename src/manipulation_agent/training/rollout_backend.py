"""Lazy, single-device Transformers generation with the training template."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from copy import deepcopy
import inspect
from pathlib import Path

from ..sft.common import ConversionError
from .artifacts import ProcessFailure, TrainingError, fingerprint


def native_generation_config(config, tokenizer):
    """Honor the base tokenizer's assistant-turn boundary as well as its EOS.

    Some local snapshots lack generation_config.json and the model config only
    supplies endoftext. Continuing past tokenizer.eos_token would manufacture
    another chat role rather than return the model's completed assistant turn.
    """
    result=deepcopy(config)
    original=result.eos_token_id
    eos=list(original) if isinstance(original,(list,tuple)) else ([] if original is None else [original])
    if tokenizer.eos_token_id is None:
        raise TrainingError('missing_native_eos','The base tokenizer must declare its native end-of-turn token')
    if tokenizer.eos_token_id not in eos:eos.append(tokenizer.eos_token_id)
    result.eos_token_id=eos
    return result


@contextmanager
def cpu_reference_kernels(module):
    """Use Transformers' own Torch references when CUDA-only extras are installed.

    The pinned Qwen decorators prefer installed causal-conv1d/FLA regardless of
    the input device. Unwrap only these four functions for this serial CPU call,
    then restore them, without editing packages or affecting other processes.
    """
    originals = {}
    try:
        for name in ("causal_conv1d_fn", "causal_conv1d_update", "torch_chunk_gated_delta_rule",
                     "torch_recurrent_gated_delta_rule"):
            function = getattr(module, name, None)
            if function is not None:
                originals[name] = function
                setattr(module, name, inspect.unwrap(function))
        yield
    finally:
        for name, function in originals.items():
            setattr(module, name, function)


class GenerationBackend:
    validation_level = "real_offline_model_generation"

    def __init__(self, config, model_path, adapter_path, device, max_new_tokens):
        from .swift_backend import build_encoding_backend, require_framework
        require_framework()
        import torch
        from swift.template import update_generation_config_eos_token

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available() or self.device.index >= torch.cuda.device_count():
                raise TrainingError("gpu_unavailable", "Requested CUDA device is unavailable", device=device)
            torch.cuda.set_device(self.device)
            if not torch.cuda.is_bf16_supported():
                raise TrainingError("bf16_unavailable", "Requested GPU does not support BF16", device=device)
            free, _ = torch.cuda.mem_get_info(self.device)
            minimum = sum(p.stat().st_size for p in Path(model_path).glob("*.safetensors")) + 2 * 1024 ** 3
            if free < minimum:
                raise TrainingError("insufficient_gpu_memory", "Selected GPU has insufficient free memory; preserve existing jobs",
                                    device=device, free_bytes=free, required_free_bytes=minimum)
        self.config, self.max_new_tokens = config, max_new_tokens
        self.dtype = torch.float32 if self.device.type == "cpu" else torch.bfloat16
        torch.manual_seed(42)
        try:
            backend = build_encoding_backend(config, model_path, unbounded=True)
            model, processor = backend.arguments.get_model_processor(
                load_model=True, device_map=str(self.device), torch_dtype=self.dtype)
            if adapter_path is not None:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False,
                                                 local_files_only=True)
            self.model = model.eval()
            self.model.requires_grad_(False)
            self.template = backend.arguments.get_template(processor)
            self.template.model = model
            self.template.set_mode("transformers")
            # Never let the template silently delete/crop an overlong prompt.
            self.template.max_length = None
            self.preprocessor = backend.preprocessor
            self.model_max_length = backend.model_max_length
            self.generation_config = deepcopy(model.generation_config)
            update_generation_config_eos_token(self.generation_config, self.template)
            self.generation_config.do_sample = False
            self.generation_config.num_beams = 1
            self.generation_config.num_return_sequences = 1
            self.generation_config.repetition_penalty = 1.0
            self.generation_config.temperature = 1.0
            self.generation_config.top_p = 1.0
            self.generation_config.top_k = 50
            self.generation_config.max_new_tokens = max_new_tokens
            self.generation_config.return_dict_in_generate = False
            self.generation_config.output_scores = False
            self.generation_config.output_logits = False
            self.cpu_module = None
            if self.device.type == "cpu":
                from transformers.models.qwen3_5 import modeling_qwen3_5
                self.cpu_module = modeling_qwen3_5
        except (ImportError, OSError, RuntimeError, ValueError, TypeError, AttributeError) as exc:
            raise ProcessFailure("model_loading_failed", "Cannot load checkpoint for offline generation",
                                 error_type=type(exc).__name__) from exc

    def describe(self):
        return {"device": str(self.device), "dtype": str(self.dtype), "seed": 42,
                "model_max_length": self.model_max_length, "do_sample": False, "batch_size": 1,
                "cpu_reference_kernels": self.cpu_module is not None,
                "eos_token_id": self.generation_config.eos_token_id}

    def _to_device(self, value):
        if self.torch.is_tensor(value):
            return value.to(self.device)
        if isinstance(value, dict):
            return {k: self._to_device(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self._to_device(v) for v in value)
        return value

    def generate(self, prompt):
        from .encoding import normalize
        torch = self.torch
        input_tokens = None
        try:
            encoded = self.template.encode(normalize(prompt, self.preprocessor))
            input_tokens = len(encoded["input_ids"])
            if (input_tokens > self.config["data"]["max_length"]
                    or input_tokens + self.max_new_tokens > self.model_max_length):
                raise TrainingError("input_overlength", "Prompt or requested continuation exceeds the context budget",
                                    input_tokens=input_tokens, prompt_limit=self.config["data"]["max_length"],
                                    model_max_length=self.model_max_length, max_new_tokens=self.max_new_tokens)
            batch = self.template.data_collator([encoded])
            del encoded
            batch = self._to_device({k: v for k, v in batch.items() if k not in {"labels", "loss_scale"}})
            prompt_length = int(batch["input_ids"].shape[1])
            input_hash = fingerprint(batch["input_ids"][0].tolist())
            kernels = cpu_reference_kernels(self.cpu_module) if self.cpu_module is not None else nullcontext()
            with kernels, torch.inference_mode():
                generated = self.model.generate(**batch, generation_config=self.generation_config, logits_to_keep=1)
            suffix = generated[0, prompt_length:].tolist()
            text = self.template.tokenizer.decode(suffix, skip_special_tokens=False)
            eos = self.generation_config.eos_token_id
            eos = eos if isinstance(eos, (list, tuple)) else [eos]
            reason = "eos" if suffix and suffix[-1] in eos else "length" if len(suffix) >= self.max_new_tokens else "stop"
            return {"raw_response": text, "input_tokens": input_tokens, "generated_tokens": len(suffix),
                    "input_ids_sha256": input_hash, "generated_ids_sha256": fingerprint(suffix),
                    "finish_reason": reason}
        except ConversionError:
            raise
        except torch.cuda.OutOfMemoryError as exc:
            raise TrainingError("inference_out_of_memory", "Sample exhausted CUDA memory",
                                input_tokens=input_tokens) from exc
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, AssertionError) as exc:
            # A broken CUDA context cannot safely serve the following samples.
            fatal = isinstance(exc, RuntimeError) and any(
                word in str(exc).lower() for word in ("cuda error", "device-side assert", "illegal memory access"))
            error = ProcessFailure if fatal else TrainingError
            raise error("cuda_context_failure" if fatal else "sample_inference_failed",
                        "Cannot generate this sample", error_type=type(exc).__name__, input_tokens=input_tokens) from exc

    def recover(self):
        # Called after the exception frame (and its tensors) has been released.
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()

    def close(self):
        self.template.model = None
        del self.model
        self.recover()


def native_image_resize_options(processor, image_max_token_num):
    """Map an explicit per-image token cap to the base processor's pixel units."""
    if image_max_token_num is None:
        return {}
    if type(image_max_token_num) is not int or image_max_token_num not in (256, 64):
        raise ValueError('image_max_token_num must be 256 or 64')
    image_processor = processor.image_processor
    pixels = image_max_token_num * (image_processor.patch_size * image_processor.merge_size) ** 2
    size = image_processor.size
    minimum = size['shortest_edge'] if isinstance(size, dict) else size.shortest_edge
    return {'size': {'shortest_edge': min(minimum, pixels), 'longest_edge': pixels}}


class NativeGenerationBackend(GenerationBackend):
    """Online model-native encoding; none of the offline training policy applies.

    Device transfer, CPU reference kernels and memory cleanup are shared with
    the offline backend. Processor and generation settings always come from the
    base model, including when PEFT weights are attached. An explicit online
    image token cap overrides only the processor's resize bounds.
    """
    validation_level = 'real_native_model_generation'

    def __init__(self, model_path, adapter_path, device, *, image_max_token_num=None):
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        self.torch, self.device = torch, torch.device(device)
        self.dtype = torch.float32 if self.device.type == 'cpu' else torch.bfloat16
        if self.device.type == 'cuda':
            if not torch.cuda.is_available() or self.device.index >= torch.cuda.device_count():
                raise TrainingError('gpu_unavailable', 'Requested CUDA device is unavailable')
            torch.cuda.set_device(self.device)
            if not torch.cuda.is_bf16_supported():
                raise TrainingError('bf16_unavailable', 'Requested GPU does not support BF16')
            free, _ = torch.cuda.mem_get_info(self.device)
            weights = sum(p.stat().st_size for p in Path(model_path).glob('*.safetensors'))
            if free < weights:
                raise TrainingError('insufficient_gpu_memory', 'Free GPU memory is smaller than the base weights',
                                    free_bytes=free, required_free_bytes=weights)
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.image_max_token_num = image_max_token_num
        native_image_resize_options(self.processor, image_max_token_num)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, dtype=self.dtype,
            device_map=str(self.device), attn_implementation='sdpa')
        self.generation_config = native_generation_config(model.generation_config,self.processor.tokenizer)
        self.model_max_length = int(model.config.text_config.max_position_embeddings)
        if adapter_path is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False,
                                             local_files_only=True)
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.cpu_module = None
        if self.device.type == 'cpu':
            from transformers.models.qwen3_5 import modeling_qwen3_5
            self.cpu_module = modeling_qwen3_5

    def encode(self, messages, tools):
        # No answer placeholder, supervised think prefix, truncation, historical
        # image window or training-recipe vision settings enter this path.
        batch = self.processor.apply_chat_template(
            deepcopy(messages), tools=deepcopy(tools), add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors='pt',
            processor_kwargs={'truncation': False, 'images_kwargs': native_image_resize_options(
                self.processor, getattr(self, 'image_max_token_num', None))})
        length = int(batch['input_ids'].shape[1])
        if length >= self.model_max_length:
            raise TrainingError('context_exhausted', 'Complete public history exceeds the native model context',
                                input_tokens=length, model_max_length=self.model_max_length)
        return batch

    def describe(self):
        return {'device': str(self.device), 'dtype': str(self.dtype), 'batch_size': 1,
                'model_max_length': self.model_max_length, 'thinking': 'native_chat_template_default',
                'generation_config': self.generation_config.to_dict(),
                'processor_image_settings': self.processor.image_processor.to_dict(),
                'image_max_token_num': self.image_max_token_num,
                'processor_image_overrides': native_image_resize_options(self.processor, self.image_max_token_num),
                'input_policy': 'all_public_history_original_rgb_native_mcp',
                'training_contract_applied': False, 'episode_limits': None}

    def generate(self, messages, tools):
        torch = self.torch
        try:
            batch = self.encode(messages, tools)
            length = int(batch['input_ids'].shape[1])
            input_hash = fingerprint(batch['input_ids'][0].tolist())
            grids = batch.get('image_grid_thw')
            image_grids = grids.tolist() if grids is not None else []
            prefix_has_think=self.processor.tokenizer.decode(
                batch['input_ids'][0,-16:].tolist(),skip_special_tokens=False).endswith('<think>\n')
            config = deepcopy(self.generation_config)
            # Override Transformers' fallback max_length=20 with the actual
            # native capacity. This is a model limit, not an episode budget.
            config.max_new_tokens = self.model_max_length - length
            batch = self._to_device(batch)
            kernels = cpu_reference_kernels(self.cpu_module) if self.cpu_module is not None else nullcontext()
            with kernels, torch.inference_mode():
                output = self.model.generate(**batch, generation_config=config, logits_to_keep=1)
            sequences = output.sequences if hasattr(output, 'sequences') else output
            suffix = sequences[0, length:].tolist()
            eos = config.eos_token_id
            eos = eos if isinstance(eos, (list, tuple)) else [eos]
            return {'raw_response': self.processor.tokenizer.decode(suffix, skip_special_tokens=False),
                    'input_tokens': length, 'generated_tokens': len(suffix),
                    'input_ids_sha256': input_hash, 'generated_ids_sha256': fingerprint(suffix),
                    'generation_prefix_has_think': prefix_has_think,
                    'image_grid_thw': image_grids,
                    'finish_reason': 'eos' if suffix and suffix[-1] in eos else 'context_limit'}
        except torch.cuda.OutOfMemoryError as exc:
            raise TrainingError('inference_out_of_memory', 'Native model exhausted CUDA memory') from exc

    def close(self):
        del self.model
        self.recover()
