"""Deterministic RGB-window compression; shared with future student inference."""
from __future__ import annotations

from copy import deepcopy

from .common import ConversionError, dumps, loads
from .format import to_msswift, validate_record, estimate_length
from .history_compaction import COMPACT_HISTORY, LEGACY_HISTORY, compact_history, validate_history_compression


DEFAULT_MAX_LEN = 24576
CONTEXT_POLICY = "rgb_window_budget_v1"
COMPACT_CONTEXT_POLICY = "rgb_window_budget_compact_success_v1"


def context_policy(history_compression=LEGACY_HISTORY):
    validate_history_compression(history_compression)
    return COMPACT_CONTEXT_POLICY if history_compression == COMPACT_HISTORY else CONTEXT_POLICY


def validate_max_len(max_len):
    if type(max_len) is not int or max_len < 0:
        raise ConversionError("invalid_max_len", "max_len must be a nonnegative integer; 0 disables budget trimming")


def result_object(message):
    content = message.get("content", [])
    if isinstance(content, list) and content and content[0].get("type") == "text":
        try:
            value = loads(content[0]["text"])
            return value if isinstance(value, dict) else {}
        except ValueError:
            pass
    return {}


def _history_pairs(messages):
    """Map reply indices to dispatch indices before compression removes IDs."""
    # Public assistant text may occur between dispatch and receipt. Resolve
    # calls by identity and order, never by adjacency to the tool response.
    calls_by_id, pairs, replied = {}, {}, set()
    for i, message in enumerate(messages):
        calls = message.get("tool_calls") or []
        if calls:
            if message.get("role") != "assistant" or len(calls) != 1:
                raise ConversionError("broken_history", "Expected one call per historical assistant")
            call_id = calls[0].get("id")
            if not isinstance(call_id, str) or not call_id or call_id in calls_by_id:
                raise ConversionError("broken_history", "Tool history has an ambiguous call ID")
            calls_by_id[call_id] = i
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in calls_by_id or call_id in replied:
                raise ConversionError("broken_history", "Tool history has an ambiguous call pairing")
            pairs[i] = calls_by_id[call_id]
            replied.add(call_id)
    return pairs


def _rgb_replies(messages):
    return [i for i, m in enumerate(messages) if m.get("role") == "tool"
            and isinstance(m.get("content"), list)
            and any(p.get("type") == "image" for p in m["content"])]


def compress_history(messages: list[dict], k: int = 2, *, history_compression=LEGACY_HISTORY) -> list[dict]:
    """Keep K recent RGB replies, all pixels, read skills and latest returned notes.

    Input is the *prefix before a decision*, in OAI messages with text/image
    content blocks. No final episode state is accepted by this function.
    """
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    validate_history_compression(history_compression)
    out = deepcopy(messages)
    pairs = _history_pairs(out)
    replies = list(pairs)
    rgb = _rgb_replies(out)
    cutoff = rgb[-k] if len(rgb) >= k else 0
    latest_notes = next((i for i in reversed(replies)
                         if isinstance(result_object(out[i]).get("notes"), dict)), None)
    for i in replies:
        if i >= cutoff or i == latest_notes or out[i].get("name") in {"read_skill", "list_skills"}:
            continue
        result = result_object(out[i])
        if not result:
            # Preserve unstructured MCP/transport errors in full. Their first
            # text block must not disappear into an empty JSON summary.
            continue
        call_index = pairs[i]
        assistant, reply = out[call_index], out[i]
        function = assistant["tool_calls"][0]["function"]
        try:
            args = loads(function["arguments"])
        except ValueError:
            args = {"invalid_arguments_text": function["arguments"]}
        if reply.get("name") == "task_notes" and isinstance(args, dict) and "content" in args:
            args = {key: value for key, value in args.items() if key != "content"}
            args["historical_note_content_omitted"] = True
        out[call_index] = {"role": "assistant", "content": "Action: " + dumps(
            {"name": function["name"], "arguments": args}, sort_keys=True), "loss": False}
        if "notes" in result:
            result["notes"] = {"version": result["notes"].get("version"), "historical_content_omitted": True}
        observation = result.get("observation")
        if isinstance(observation, dict):
            result["observation"] = {key: observation[key] for key in ("revision", "capture") if key in observation}
            result["observation"]["images"] = [
                {key: image[key] for key in ("view", "image_ref")} for image in observation.get("images", [])]
        # The first block is the result JSON; all following blocks (including
        # the original per-image view/ref labels) retain their original order.
        reply["content"] = [{"type": "text", "text": "Result: " + dumps(result, sort_keys=True)},
                            *reply["content"][1:]]
        reply.pop("tool_call_id", None)  # Old calls are now textual history.
    return compact_history(out, k=k)[0] if history_compression == COMPACT_HISTORY else out


def _slim_images(content):
    """Keep the first delivered image; remove only later images and their labels."""
    images = [i for i, part in enumerate(content) if part.get("type") == "image"]
    removed = set(images[1:])
    for i in images[1:]:
        if (i > 0 and content[i - 1].get("type") == "text"
                and content[i - 1].get("text", "").startswith("RGB view=")):
            removed.add(i - 1)
    return [part for i, part in enumerate(content) if i not in removed], max(0, len(images) - 1)


def build_sample_messages(history, target, tools, *, k=2, max_len=DEFAULT_MAX_LEN, history_compression=LEGACY_HISTORY):
    """Compress a raw prefix, then trim old images/call groups to a soft budget.

    Returns (messages, length_stats). The target and recent RGB window cannot
    be trimmed, so over_budget can remain true. No input is mutated and no
    sample is rejected for length. Estimation uses the final ms-swift format,
    including tools and target think markers, rather than processor tokens.
    """
    validate_max_len(max_len)
    # The raw indices remain valid through text compression and image slimming.
    pairs = _history_pairs(history)
    messages = compress_history(history, k, history_compression=history_compression) + [deepcopy(target)]
    before = after = estimate_length(to_msswift(messages, tools))
    slim_triggered = bool(max_len and before["approx_tokens"] > max_len)
    slimmed_images = dropped_pairs = 0
    if slim_triggered:
        rgb = _rgb_replies(history)
        cutoff = rgb[-k] if len(rgb) >= k else 0
        for i in rgb:
            if i >= cutoff:
                break
            messages[i]["content"], count = _slim_images(messages[i]["content"])
            slimmed_images += count
        after = estimate_length(to_msswift(messages, tools))
        removed_messages = 0
        for end, start in pairs.items():
            if after["approx_tokens"] <= max_len or end >= cutoff:
                break
            # A serial call may contain public progress assistants. Never
            # delete another call, a user/system message or an unrelated reply.
            if any(m.get("role") != "assistant" or m.get("tool_calls")
                   for m in history[start + 1:end]):
                raise ConversionError("broken_history", "Cannot trim overlapping or non-serial tool history")
            del messages[start - removed_messages:end + 1 - removed_messages]
            removed_messages += end + 1 - start
            dropped_pairs += 1
            after = estimate_length(to_msswift(messages, tools))
    return messages, {**after, "approx_tokens_before": before["approx_tokens"],
                      "image_count_before": before["image_count"], "slim_triggered": slim_triggered,
                      "slimmed_images": slimmed_images, "dropped_history_pairs": dropped_pairs,
                      "images_removed": before["image_count"] - after["image_count"],
                      "over_budget": bool(max_len and after["approx_tokens"] > max_len)}
