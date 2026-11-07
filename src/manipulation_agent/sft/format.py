"""CPU-only messages/images/tools row format and validation."""
from __future__ import annotations

from copy import deepcopy
from .common import ConversionError, dumps


def to_msswift(messages: list[dict], tools: list[dict]) -> dict:
    converted, images = [], []
    for message in messages:
        row = {key: deepcopy(message[key]) for key in
               ("role", "name", "tool_calls", "tool_call_id", "loss") if key in message}
        content = message.get("content", "")
        if isinstance(content, list):
            chunks = []
            for part in content:
                if part.get("type") == "image":
                    images.append(part["image"])
                    chunks.append("<image>")
                elif part.get("type") == "text":
                    chunks.append(part["text"])
                else:
                    raise ConversionError("unknown_content_block", "Unsupported history content block")
            content = "\n".join(chunks)
        row["content"] = content
        if row["role"] == "assistant" and row.get("loss") is not False:
            row["content"] = "<think>\n\n</think>" + ("\n\n" + content if content else "")
        converted.append(row)
    return {"messages": converted, "images": images, "tools": deepcopy(tools)}


def validate_record(record):
    messages, images = record["messages"], record["images"]
    if len(messages) < 3 or [m["role"] for m in messages[:2]] != ["system", "user"]:
        raise ConversionError("invalid_sample", "Sample must start with system and user")
    assistants = [m for m in messages if m["role"] == "assistant"]
    if (messages[-1]["role"] != "assistant" or assistants[-1].get("loss") is False
            or sum(m.get("loss") is not False for m in assistants) != 1):
        raise ConversionError("invalid_loss_mask", "Only the final assistant may carry loss")
    if sum(m["content"].count("<image>") for m in messages) != len(images):
        raise ConversionError("image_placeholder_mismatch", "Image markers and image paths disagree")
    calls = messages[-1].get("tool_calls", [])
    if len(calls) != 1:
        raise ConversionError("invalid_sample", "Target must contain exactly one tool call")
    return True


def estimate_length(record):
    # Soft context-trimming budget, NOT a Qwen processor length or sample filter.
    chars = sum(len(m["content"]) + len(dumps(m.get("tool_calls", []))) for m in record["messages"])
    chars += len(dumps(record["tools"]))
    return {"approx_tokens": chars // 4 + 64 * len(record["images"]), "text_characters": chars,
            "image_count": len(record["images"]), "method": "chars/4+64/image; not processor tokens"}
