"""Lossless package-row trimming with an injected real-token length function.

This module is CPU-only and has no model or simulator imports. It operates on
an already exported row; it never restores history removed by an earlier export.
"""
from __future__ import annotations

from copy import deepcopy

from .common import ConversionError, loads
from .format import validate_record
from .history import _slim_images

PROCESSOR_CONTEXT_POLICY = "rgb_window_processor_budget_v1"


def unpack_record(record):
    """Restore image blocks without changing any text, whitespace or think tags."""
    validate_record(record)
    messages = deepcopy(record["messages"])
    images = iter(record["images"])
    for message in messages:
        text = message["content"]
        if "<image>" not in text:
            continue
        content = []
        for line in text.split("\n"):
            if line == "<image>":
                content.append({"type": "image", "image": next(images)})
            else:
                if "<image>" in line:
                    raise ConversionError("invalid_image_marker", "Expected a standalone image marker")
                content.append({"type": "text", "text": line})
        message["content"] = content
    assert next(images, None) is None
    return messages


def pack_record(messages, tools):
    """Flatten working blocks; existing final think markers stay untouched."""
    result, images = deepcopy(messages), []
    for message in result:
        if not isinstance(message["content"], list):
            continue
        text = []
        for part in message["content"]:
            if part["type"] == "image":
                images.append(part["image"])
                text.append("<image>")
            else:
                text.append(part["text"])
        message["content"] = "\n".join(text)
    return {"messages": result, "images": images, "tools": deepcopy(tools)}


def _pairs(messages):
    """Pair native calls and the converter's older textual Action/Result groups."""
    pending, result = None, {}
    for i, message in enumerate(messages):
        calls = message.get("tool_calls") or []
        if calls:
            if pending is not None or message["role"] != "assistant" or len(calls) != 1:
                raise ConversionError("broken_history", "Expected serial historical calls")
            pending = (i, calls[0].get("id"))
        elif (pending is None and message["role"] == "assistant"
              and isinstance(message["content"], str) and message["content"].startswith("Action: ")):
            action = loads(message["content"][len("Action: "):])
            if not isinstance(action, dict) or set(action) != {"name", "arguments"}:
                raise ConversionError("broken_history", "Invalid compressed historical call")
            pending = (i, None)
        if message["role"] != "tool":
            continue
        if pending is None or message.get("tool_call_id") != pending[1]:
            raise ConversionError("broken_history", "Historical reply has no matching call")
        start, _ = pending
        if any(m["role"] != "assistant" or m.get("tool_calls") for m in messages[start + 1:i]):
            raise ConversionError("broken_history", "Cannot trim overlapping call groups")
        result[i] = start
        pending = None
    if pending is not None:
        raise ConversionError("broken_history", "Historical call has no reply")
    return result


def trim_record(record, measure, *, max_length=17408, k=2):
    """Slim older RGB replies, then delete complete groups oldest first.

    `measure` must return the actual untruncated processor/template token length,
    including tools, vision tokens and the target. The protected prefix, target,
    last K RGB groups and later history never change. Untrimmable rows remain
    present and are explicitly reported rather than silently losing supervision.
    """
    if type(max_length) is not int or max_length < 1 or type(k) is not int or k < 1:
        raise ConversionError("invalid_processor_budget", "Use positive integer max_length and k")
    messages = unpack_record(record)
    assert pack_record(messages, record["tools"]) == record
    history = messages[:-1]
    pairs = _pairs(history)
    rgb = [i for i, m in enumerate(history) if m["role"] == "tool"
           and isinstance(m["content"], list) and any(p["type"] == "image" for p in m["content"])]
    if any(i not in pairs for i in rgb):
        raise ConversionError("broken_history", "RGB reply is outside a complete call group")
    cutoff = rgb[-k] if len(rgb) >= k else 0
    measurements = 0

    def length(row):
        nonlocal measurements
        value = measure(row)
        if type(value) is not int or value < 1:
            raise ConversionError("invalid_token_measurement", "Expected a positive real-token count")
        measurements += 1
        return value

    before = after = length(record)
    slimmed = dropped = removed_messages = 0
    output = deepcopy(record)
    triggered = before > max_length
    if triggered:
        for i in rgb:
            if i >= cutoff:
                break
            messages[i]["content"], count = _slim_images(messages[i]["content"])
            slimmed += count
        output = pack_record(messages, record["tools"])
        after = length(output)
        for end, start in pairs.items():
            if after <= max_length or end >= cutoff:
                break
            del messages[start - removed_messages:end + 1 - removed_messages]
            removed_messages += end + 1 - start
            dropped += 1
            output = pack_record(messages, record["tools"])
            after = length(output)
    validate_record(output)
    assert output["messages"][:2] == record["messages"][:2]
    assert output["messages"][-1] == record["messages"][-1]
    return output, {"processor_tokens_before": before, "processor_tokens": after,
                    "processor_trim_triggered": triggered, "processor_slimmed_images": slimmed,
                    "processor_dropped_history_pairs": dropped,
                    "processor_images_removed": len(record["images"]) - len(output["images"]),
                    "processor_over_budget": after > max_length,
                    "processor_over_budget_reason": "protected_context_exceeds_budget" if after > max_length else None,
                    "processor_length_measurements": measurements}
