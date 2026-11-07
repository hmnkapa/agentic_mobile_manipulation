"""Pure protocol transforms shared by SFT and a future Qwen inference client.

The native tool interface remains unchanged. Do not clamp invalid points or
invent defaults: a failed teacher call must not become a valid training target.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re

from .common import ConversionError, dumps

COORDINATE_GUIDE = (
    "Student coordinate protocol: act.target uses image_ref and point_2d=[x,y]. "
    "Both coordinates are integers from 0 to 1000: x runs left to right, y top to bottom. "
    "Keep the latest image_ref and revision. release/wait still use target=null."
)


def _convert_arguments(name, arguments, *, inverse=False, strict=True):
    args = deepcopy(arguments)
    if name != "act" or not isinstance(args, dict) or args.get("target") is None:
        return args
    target = args["target"]
    source, dest = ("point_2d", "point") if inverse else ("point", "point_2d")
    if not isinstance(target, dict) or source not in target or dest in target:
        raise ConversionError("invalid_coordinates", "Expected exactly one native/student point field")
    point = target[source]
    numeric = (isinstance(point, list) and len(point) == 2
               and all(type(v) is int or (type(v) is float and math.isfinite(v)) for v in point))
    if not numeric or (inverse and any(type(v) is not int for v in point)):
        raise ConversionError("invalid_coordinates", "Expected two finite coordinate values")
    limit = 1000 if inverse else 1
    if strict and any(v < 0 or v > limit for v in point):
        raise ConversionError("invalid_coordinates", "Coordinates outside the declared range")
    try:
        target[dest] = [v / 1000 for v in point] if inverse else [round(1000 * v) for v in point]
    except OverflowError as exc:
        raise ConversionError("invalid_coordinates", "Coordinate conversion overflows") from exc
    del target[source]
    return args


def to_student_arguments(name, arguments, *, strict=True):
    """Native point -> integer point_2d. strict=False retains out-of-range failures."""
    return _convert_arguments(name, arguments, strict=strict)


def from_student_arguments(name, arguments):
    """Validate the student's coordinate representation, then divide by 1000."""
    return _convert_arguments(name, arguments, inverse=True)


def rewrite_interface_text(text: str) -> str:
    """Rewrite explicit API examples/coordinate descriptions, never free-form notes."""
    def example(match):
        body = match.group(2)
        pieces = body.split(",")
        try:
            point = [float(v.strip()) for v in pieces]
            if len(point) != 2 or not all(math.isfinite(v) and 0 <= v <= 1 for v in point):
                raise ValueError
        except ValueError:
            return "point_2d" + match.group(1) + "[" + body + "]"
        return "point_2d" + match.group(1) + dumps([round(v * 1000) for v in point])

    text = re.sub(r"\bpoint(\s*[:=]\s*)\[([^\]\n]*)\]", example, text)
    text = re.sub(r"\btarget\.point\b", "target.point_2d", text)
    text = text.replace("with normalized x left-to-right and y top-to-bottom",
                        "with integer x/y in 0–1000, x left-to-right and y top-to-bottom")
    text = text.replace("select a normalized pixel", "select a 0–1000 relative pixel using target.point_2d")
    text = text.replace("x: left=0/right=1, y: top=0/bottom=1",
                        "point_2d integers: x left=0/right=1000, y top=0/bottom=1000")
    text = text.replace(
        "Coordinates are normalized image coordinates: x/y 0.0 is left/top, 1.0 is right/bottom.",
        "Coordinates are integer relative image coordinates: x/y 0 is left/top, 1000 is right/bottom.")
    return text


def build_student_context(system: str, catalog: list[dict]) -> tuple[str, list[dict]]:
    """Adapt a verified, run-owned system string and MCP catalog to OAI tools."""
    tools = []
    for spec in catalog:
        parameters = deepcopy(spec["inputSchema"])
        description = rewrite_interface_text(spec.get("description", ""))
        if spec["name"] == "act":
            try:
                target = parameters["properties"]["target"]
                point = target["properties"].pop("point")
                if point.get("items", {}).get("minimum") != 0 or point["items"].get("maximum") != 1:
                    raise KeyError("Unknown source coordinate scale")
                point["items"] = {"type": "integer", "minimum": 0, "maximum": 1000}
                target["properties"]["point_2d"] = point
                target["required"] = ["point_2d" if k == "point" else k for k in target["required"]]
            except (KeyError, TypeError) as exc:
                raise ConversionError("unsupported_schema", "Cannot adapt the archived act target schema") from exc
            description += "\n" + COORDINATE_GUIDE
        tools.append({"type": "function", "function": {
            "name": spec["name"], "description": description, "parameters": parameters}})
    return rewrite_interface_text(system) + "\n" + COORDINATE_GUIDE, tools


def validate_arguments(value, schema, path="arguments"):
    """Validate the JSON Schema subset used by the recorded MCP tool catalogs."""
    kind = schema.get("type")
    kinds = kind if isinstance(kind, list) else [kind]
    types = {"null": value is None, "boolean": type(value) is bool,
             "object": isinstance(value, dict), "array": isinstance(value, list),
             "string": isinstance(value, str), "integer": type(value) is int,
             "number": type(value) is int or (type(value) is float and math.isfinite(value))}
    if not any(types.get(k, False) for k in kinds):
        raise ConversionError("invalid_arguments", "Value does not match its archived schema", field=path)
    if "enum" in schema and value not in schema["enum"]:
        raise ConversionError("invalid_arguments", "Value is not in the archived enum", field=path)
    if isinstance(value, dict):
        props = schema.get("properties", {})
        if not set(schema.get("required", [])).issubset(value):
            raise ConversionError("invalid_arguments", "Missing required argument", field=path)
        if schema.get("additionalProperties") is False and set(value) - set(props):
            raise ConversionError("invalid_arguments", "Unexpected argument", field=path)
        for key in value.keys() & props.keys():
            validate_arguments(value[key], props[key], path + "." + key)
    elif isinstance(value, list):
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", math.inf):
            raise ConversionError("invalid_arguments", "Invalid array length", field=path)
        for i, item in enumerate(value):
            validate_arguments(item, schema["items"], f"{path}[{i}]")
    elif isinstance(value, str):
        if not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", math.inf):
            raise ConversionError("invalid_arguments", "Invalid string length", field=path)
    elif type(value) in (int, float):
        if not schema.get("minimum", -math.inf) <= value <= schema.get("maximum", math.inf):
            raise ConversionError("invalid_arguments", "Number outside schema range", field=path)
