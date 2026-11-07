"""CPU-only, versioned summaries of older successful observation replies."""
from __future__ import annotations

from copy import deepcopy
import json

from .common import ConversionError, dumps, loads
from .format import validate_record

LEGACY_HISTORY = "legacy"
COMPACT_HISTORY = "old_success_observation_summary_v1"
HISTORY_COMPRESSION_POLICIES = (LEGACY_HISTORY, COMPACT_HISTORY)
ADMINISTRATIVE_FIELDS = frozenset({"observation", "evidence_id", "execution_budget", "actions_used"})
LOCAL_TELEMETRY = frozenset({"repositions", "primitive_attempts", "departure", "message"})
KNOWN_LOCAL_FIELDS = LOCAL_TELEMETRY | {"operation_committed", "recovery_required"}
GENERIC_MESSAGES = frozenset({
    "Operation completed on the same selected target after bounded local preparation.",
    "Ground-point travel completed. Select the intended object in the updated RGB for local manipulation.",
})


def validate_history_compression(policy):
    if policy not in HISTORY_COMPRESSION_POLICIES:
        raise ConversionError("unsupported_history_compression", "Unknown history compression policy")
    return policy


def _has_rgb(message):
    content = message.get("content")
    return message.get("role") == "tool" and (
        isinstance(content, str) and "<image>" in content
        or isinstance(content, list) and any(part.get("type") == "image" for part in content))


def _first_text(message):
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content and content[0].get("type") == "text":
        return content[0].get("text")
    return None


def _result(text):
    if not isinstance(text, str):
        return None, ""
    start = len("Result: ") if text.startswith("Result: ") else 0
    try:
        _, end = json.JSONDecoder().raw_decode(text, start)
        result = loads(text[start:end])
    except (TypeError, ValueError):
        return None, text
    return (result if isinstance(result, dict) else None), text[end:]


def compact_history(messages, *, k=2):
    """Return copied history and stats; preserve K RGB replies and everything later.

    Accepts raw text/image blocks or exported ms-swift strings. The caller
    supplies only observations already returned before the current decision.
    All assistant messages, skills, notes, initialization and errors survive.
    """
    if type(k) is not int or k < 1:
        raise ConversionError("invalid_window", "k must be a positive integer")
    output = deepcopy(messages)
    rgb = [i for i, message in enumerate(messages) if _has_rgb(message)]
    cutoff = rgb[-k] if len(rgb) >= k else 0
    changed = []
    for i, message in enumerate(output[:cutoff]):
        if message.get("role") != "tool" or message.get("name") not in ("act", "look"):
            continue
        result, suffix = _result(_first_text(message))
        if not result or result.get("ok") is not True or result.get("error") or result.get("isError"):
            continue
        summary = {key: value for key, value in result.items() if key not in ADMINISTRATIVE_FIELDS}
        effect = summary.get("effect")
        if isinstance(effect, dict):
            local = effect.get("local_adjustment")
            if (isinstance(local, dict) and set(local) <= KNOWN_LOCAL_FIELDS
                    and local.get("operation_committed") is True and local.get("recovery_required") is False):
                effect["local_adjustment"] = {key: value for key, value in local.items()
                                              if key not in LOCAL_TELEMETRY}
            message_text = effect.get("message")
            if isinstance(message_text, str) and message_text in GENERIC_MESSAGES:
                effect.pop("message")
        text = "Result: " + dumps(summary, sort_keys=True) + suffix
        if isinstance(message["content"], str):
            message["content"] = text
        else:
            message["content"][0]["text"] = text
        if message != messages[i]:
            changed.append(i)
    return output, {"history_compression": COMPACT_HISTORY, "changed_replies": len(changed),
                    "changed_indices": changed, "protected_cutoff": cutoff, "k": k}


def compact_record(record, *, k=2):
    """Apply the same shared rule to a packaged sample without changing targets."""
    validate_record(record)
    output = deepcopy(record)
    history, stats = compact_history(record["messages"][:-1], k=k)
    output["messages"] = history + [deepcopy(record["messages"][-1])]
    validate_record(output)
    assert output["images"] == record["images"] and output["tools"] == record["tools"]
    assert output["messages"][:2] == record["messages"][:2]
    assert output["messages"][-1] == record["messages"][-1]
    return output, stats
