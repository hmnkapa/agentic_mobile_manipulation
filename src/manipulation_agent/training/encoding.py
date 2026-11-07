"""Actual token/label/image checks, independent of model weight loading."""
from __future__ import annotations

from copy import deepcopy
import math
import re

from ..sft.common import ConversionError
from .artifacts import fingerprint


def values(value):
    return value.tolist() if hasattr(value, "tolist") else list(value)


def normalize(record, preprocessor):
    try:
        normalized = preprocessor.preprocess(deepcopy(record))
    except (TypeError, ValueError, KeyError, AssertionError) as exc:
        raise ConversionError("encoding_preprocess_failure", "ms-swift cannot preprocess this OpenAI record") from exc
    calls = sum(len(m.get("tool_calls") or []) for m in record["messages"])
    if (not isinstance(normalized, dict) or not normalized.get("messages")
            or any(m.get("tool_calls") for m in normalized["messages"])
            or sum(m.get("role") == "tool_call" for m in normalized["messages"]) != calls):
        raise ConversionError("unsupported_ms_swift_preprocessor",
                              "ms-swift must normalize all OpenAI tool_calls before encoding")
    return normalized


def check_encoding(record, template, processor, *, preprocessor=None, render_target=None, normalized=None):
    """Check unshifted labels and the final-assistant-only counterfactual.

    A formatter is required by production preflight. The optional fallback
    preserves the historical injectable helper API for existing CPU callers.
    """
    if preprocessor is None:
        try:
            from swift.dataset import MessagesPreprocessor
        except ImportError as exc:
            raise ConversionError("missing_ms_swift", "Use an ms-swift environment with OpenAI message preprocessing") from exc
        preprocessor = MessagesPreprocessor()
    encoded = template.encode(deepcopy(normalized) if normalized is not None else normalize(record, preprocessor), return_length=True)
    ids, labels = values(encoded["input_ids"]), values(encoded["labels"])
    grid = encoded.get("image_grid_thw")
    grids = values(grid) if grid is not None else []
    del grid
    del encoded  # Release pixel_values before decoding a second image batch.
    masked = deepcopy(record)
    masked["messages"][-1]["loss"] = False
    encoded = template.encode(normalize(masked, preprocessor), return_length=True)
    masked_ids, masked_labels = values(encoded["input_ids"]), values(encoded["labels"])
    del encoded
    if (len(ids) != len(labels) or ids != masked_ids or len(masked_labels) != len(ids)
            or any(label != -100 for label in masked_labels)):
        raise ConversionError("encoding_mask_failure", "Template does not respect the final-assistant-only loss contract")
    loss_ids = [token for token, label in zip(ids, labels) if label != -100]
    if not loss_ids or any(label not in (-100, token) for token, label in zip(ids, labels)):
        raise ConversionError("encoding_label_failure", "Expected nonempty, unshifted target labels")
    text = template.tokenizer.decode(loss_ids, skip_special_tokens=False)
    function = record["messages"][-1]["tool_calls"][0]["function"]
    if not all(part in text for part in ("<think>", "</think>", "<tool_call>", function["name"])):
        raise ConversionError("encoding_target_failure", "Target think/tool-call structure is absent from the loss region")
    if render_target is not None:
        expected = render_target(function)
        # Check within an uninterrupted supervised span, not a concatenation
        # that could conceal masked tokens inside the target call.
        spans, span = [], []
        for token, label in zip(ids, labels):
            if label == -100:
                if span:
                    spans.append(template.tokenizer.decode(span, skip_special_tokens=False))
                    span = []
            else:
                span.append(token)
        if span:
            spans.append(template.tokenizer.decode(span, skip_special_tokens=False))
        if not expected or not any(expected in value for value in spans):
            raise ConversionError("encoding_target_failure", "Complete rendered tool name/arguments are absent from a loss span")
        if not any(re.search(r"<think>\s*</think>", value) for value in spans):
            raise ConversionError("encoding_target_failure", "Empty think markers must remain supervised")
    image_token = getattr(processor, "image_token", None) or "<|image_pad|>"
    image_id = template.tokenizer.convert_tokens_to_ids(image_token)
    if image_id is None or image_id == getattr(template.tokenizer, "unk_token_id", None):
        raise ConversionError("unknown_image_token", "Cannot identify the processor image token")
    image_tokens = ids.count(image_id)
    merge = getattr(getattr(processor, "image_processor", None), "merge_size", 1)
    valid_grids = type(merge) is int and merge > 0 and all(
        len(g) == 3 and all(type(d) is int and d > 0 for d in g) and math.prod(g) % (merge ** 2) == 0 for g in grids)
    expected_image_tokens = sum(math.prod(g) // (merge ** 2) for g in grids) if valid_grids else -1
    if (bool(image_tokens) != bool(record["images"]) or len(grids) != len(record["images"])
            or image_tokens != expected_image_tokens):
        raise ConversionError("encoding_image_failure", "Encoded image tokens/grid disagree with the record")
    return {"status": "passed", "tokens": len(ids), "loss_tokens": len(loss_ids),
            "masked_tokens": len(ids) - len(loss_ids), "images": len(record["images"]),
            "image_tokens": image_tokens, "input_ids_sha256": fingerprint(ids), "labels_sha256": fingerprint(labels),
            "all_history_masked": True, "complete_target_verified": render_target is not None, "weights_loaded": False}
