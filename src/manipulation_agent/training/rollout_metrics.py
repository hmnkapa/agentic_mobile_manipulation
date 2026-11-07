"""Strict Qwen3.5 tool-call parsing and offline fidelity metrics (stdlib only)."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import math
import re

from ..sft.common import ConversionError, loads
from ..sft.protocol import validate_arguments


def _error(code, message, **fields):
    return ConversionError(code, message, **fields)


def _parameter_value(text, schema):
    # The native formatter adds one framing newline on each side. Do not use
    # strip() for strings: notes/reasons can contain intentional whitespace.
    if text.startswith("\r\n"):
        text = text[2:]
    elif text.startswith("\n"):
        text = text[1:]
    if text.endswith("\r\n"):
        text = text[:-2]
    elif text.endswith("\n"):
        text = text[:-1]
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind]
    if "null" in kinds and text.strip() == "null":
        return None
    if "string" in kinds:
        return text
    try:
        return loads(text.strip())
    except ValueError as exc:
        raise _error("invalid_parameter_json", "Non-string XML parameter must contain strict JSON") from exc


def parse_tool_call(response, tools):
    """Parse exactly one complete XML call, without lossy native coercions.

    Argument schema validation is separate so a structurally parsed, but
    invalid, prediction can still be saved and its tool name inspected.
    """
    text = response
    if text.lstrip().startswith("<think>"):
        closing = text.find("</think>")
        if closing < 0:
            raise _error("unfinished_thinking", "The generated thinking block is incomplete")
        text = text[closing + len("</think>"):]
    else:
        # The generation prompt may already contain the opening think tag.
        closing, opening = text.find("</think>"), text.find("<tool_call>")
        if closing >= 0 and (opening < 0 or closing < opening):
            text = text[closing + len("</think>"):]
    if text.count("<tool_call>") != 1 or text.count("</tool_call>") != 1:
        raise _error("invalid_call_count", "Expected exactly one complete tool_call block")
    start, end = text.index("<tool_call>"), text.index("</tool_call>")
    if end < start:
        raise _error("malformed_tool_call", "Tool-call delimiters are out of order")
    body = text[start + len("<tool_call>"):end]
    match = re.fullmatch(r"\s*<function=([^<>\s=]+)>(.*?)</function>\s*", body, re.DOTALL)
    if not match:
        raise _error("malformed_tool_call", "Expected a complete Qwen function block")
    name, parameters = match.groups()
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    if name not in schemas:
        raise _error("unknown_tool", "Generated tool is absent from the sample catalog", tool=name)
    properties = schemas[name].get("properties", {})
    arguments, position = {}, 0
    for param in re.finditer(r"<parameter=([^<>\s=]+)>(.*?)</parameter>", parameters, re.DOTALL):
        if parameters[position:param.start()].strip():
            raise _error("malformed_tool_call", "Unexpected content between XML parameters")
        key, value = param.groups()
        if key in arguments:
            raise _error("duplicate_parameter", "A parameter occurs more than once", parameter=key)
        if key not in properties:
            raise _error("unknown_parameter", "Parameter is absent from the sample schema", parameter=key)
        arguments[key] = _parameter_value(value, properties[key])
        position = param.end()
    if parameters[position:].strip():
        raise _error("malformed_tool_call", "Incomplete or unexpected XML parameter content")
    return {"name": name, "arguments": arguments}


def validate_call(call, tools):
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    if call["name"] not in schemas:
        raise _error("unknown_tool", "Generated tool is absent from the sample catalog")
    validate_arguments(call["arguments"], schemas[call["name"]])


def target_call(record):
    function = record["messages"][-1]["tool_calls"][0]["function"]
    return {"name": function["name"], "arguments": loads(function["arguments"])}


def json_equal(left, right):
    """JSON structural equality; booleans never compare equal to numbers."""
    numeric = (int, float)
    if type(left) in numeric and type(right) in numeric:
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(json_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(json_equal(a, b) for a, b in zip(left, right))
    return left == right


def _point(call):
    if not call or call["name"] != "act":
        return None
    target = call["arguments"].get("target")
    if not isinstance(target, dict):
        return None
    point = target.get("point_2d")
    if (not isinstance(point, list) or len(point) != 2
            or any(type(v) is not int or not 0 <= v <= 1000 for v in point)):
        return None
    return point


def score_call(gold, prediction, *, valid, point_tolerance):
    tool_match = bool(prediction and prediction["name"] == gold["name"])
    primitive_match = bool(tool_match and gold["name"] == "act"
                           and prediction["arguments"].get("primitive") == gold["arguments"]["primitive"])
    strict = bool(valid and prediction and json_equal(gold, prediction))
    distance, tolerant = None, strict
    a, b = _point(gold), _point(prediction)
    if (valid and a is not None and b is not None
            and gold["arguments"]["target"]["image_ref"] == prediction["arguments"]["target"].get("image_ref")):
        distance = math.dist(a, b)
        aligned = deepcopy(prediction)
        aligned["arguments"]["target"]["point_2d"] = a
        tolerant = distance <= point_tolerance and json_equal(gold, aligned)
    return {"strict_match": strict, "tolerant_match": tolerant, "tool_match": tool_match,
            "primitive_match": primitive_match, "valid_call": valid, "point_distance": distance,
            "gold_has_point": a is not None}


def _summarize(rows):
    total = len(rows)
    def rate(field, subset=rows):
        correct = sum(bool(r["metrics"][field]) for r in subset)
        return {"correct": correct, "total": len(subset),
                "accuracy": correct / len(subset) if subset else None}
    acts = [r for r in rows if r["gold"]["name"] == "act"]
    distances = [r["metrics"]["point_distance"] for r in rows if r["metrics"]["point_distance"] is not None]
    return {"samples": total, "strict_accuracy": rate("strict_match"),
            "tolerant_accuracy": rate("tolerant_match"), "tool_accuracy": rate("tool_match"),
            "primitive_accuracy": rate("primitive_match", acts), "valid_call_rate": rate("valid_call"),
            "point_distance": {"count": len(distances),
                               "gold_point_samples": sum(r["metrics"]["gold_has_point"] for r in rows),
                               "mean": sum(distances) / len(distances) if distances else None,
                               "max": max(distances) if distances else None},
            "errors_by_code": dict(Counter(r["error"]["code"] for r in rows if r.get("error")))}


def summarize(rows):
    by_tool, by_primitive = defaultdict(list), defaultdict(list)
    for row in rows:
        by_tool[row["gold"]["name"]].append(row)
        if row["gold"]["name"] == "act":
            by_primitive[row["gold"]["arguments"]["primitive"]].append(row)
    return {"metrics": _summarize(rows),
            "by_tool": {key: _summarize(value) for key, value in sorted(by_tool.items())},
            "by_primitive": {key: _summarize(value) for key, value in sorted(by_primitive.items())}}
