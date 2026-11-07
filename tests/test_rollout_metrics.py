"""Offline scoring contracts, with synthetic calls and no ML dependencies."""
from copy import deepcopy
import json
import unittest

from manipulation_agent.sft.common import ConversionError
from manipulation_agent.sft.protocol import build_student_context
from manipulation_agent.tools import tool_specs
from manipulation_agent.training.rollout_metrics import (
    json_equal, parse_tool_call, score_call, summarize, validate_call,
)


def xml_call(name, arguments, *, thinking=True):
    parts = ["<tool_call>\n<function=" + name + ">\n"]
    for key, value in arguments.items():
        encoded = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        parts.append(f"<parameter={key}>\n{encoded}\n</parameter>\n")
    parts.append("</function>\n</tool_call>")
    return ("<think>\n\n</think>\n" if thinking else "") + "".join(parts) + "<|im_end|>"


ACT = {"name": "act", "arguments": {"primitive": "toggle_on", "revision": 1,
       "target": {"image_ref": "rgb-00001-left", "point_2d": [500, 500]},
       "placement_yaw_degrees": None, "wait_seconds": None}}


class RolloutMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.tools = build_student_context("Synthetic test system", tool_specs("skills"))

    def test_native_xml_restores_integer_object_and_null_types(self):
        parsed = parse_tool_call(xml_call(**ACT), self.tools)
        self.assertEqual(parsed, ACT)
        self.assertIs(type(parsed["arguments"]["revision"]), int)
        validate_call(parsed, self.tools)

    def test_string_whitespace_and_multiline_content_are_preserved(self):
        arguments = {"outcome": "achieved", "reason": "  first line\n\nsecond line  \n"}
        parsed = parse_tool_call(xml_call("finish", arguments), self.tools)
        self.assertEqual(parsed["arguments"], arguments)
        validate_call(parsed, self.tools)

    def test_empty_call_and_prose_outside_call(self):
        text = "I will initialize.\n" + xml_call("initialize", {}, thinking=False) + "\nDone."
        self.assertEqual(parse_tool_call(text, self.tools), {"name": "initialize", "arguments": {}})

    def test_thinking_calls_are_ignored_and_prompt_opening_tag_may_be_absent(self):
        thought = "<think>example " + xml_call("initialize", {}, thinking=False) + "</think>"
        self.assertEqual(parse_tool_call(thought + xml_call(**ACT, thinking=False), self.tools), ACT)
        text = "some thought\n</think>\n" + xml_call(**ACT, thinking=False)
        self.assertEqual(parse_tool_call(text, self.tools), ACT)

    def test_malformed_and_ambiguous_calls_are_rejected(self):
        valid = xml_call(**ACT)
        cases = ["plain prose", valid + valid, valid.replace("</tool_call>", ""),
                 valid.replace("</function>", ""), "<think>" + valid.replace("</think>", ""),
                 valid.replace("<function=act>", "<function=missing>"),
                 valid.replace("</function>", "<parameter=revision>1</parameter></function>"),
                 valid.replace("<parameter=revision>", "<parameter=unexpected>"),
                 valid.replace("<parameter=revision>\n1", "<parameter=revision>\nNaN"),
                 valid.replace("</function>", "unparsed debris</function>")]
        for text in cases:
            with self.subTest(text=text), self.assertRaises(ConversionError):
                parse_tool_call(text, self.tools)

    def test_duplicate_nested_json_keys_are_not_silently_overwritten(self):
        text = xml_call(**ACT).replace('"point_2d": [500, 500]', '"point_2d": [1, 2], "point_2d": [500, 500]')
        with self.assertRaisesRegex(ConversionError, "strict JSON"):
            parse_tool_call(text, self.tools)

    def test_schema_rejects_missing_fields_out_of_range_and_bool_coordinates(self):
        cases = [{}, {**ACT["arguments"], "revision": True}]
        for point in ([1001, 0], [True, 0], [0.0, 0], [-1, 0]):
            args = deepcopy(ACT["arguments"])
            args["target"]["point_2d"] = point
            cases.append(args)
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ConversionError):
                validate_call(parse_tool_call(xml_call("act", args), self.tools), self.tools)

    def score(self, predicted, **kwargs):
        return score_call(ACT, predicted, valid=True, point_tolerance=50, **kwargs)

    def test_tolerance_boundary_and_other_fields_remain_exact(self):
        shifted = deepcopy(ACT)
        shifted["arguments"]["target"]["point_2d"] = [530, 540]
        metrics = self.score(shifted)
        self.assertFalse(metrics["strict_match"])
        self.assertTrue(metrics["tolerant_match"])
        self.assertEqual(metrics["point_distance"], 50)
        shifted["arguments"]["target"]["point_2d"] = [531, 540]
        self.assertFalse(self.score(shifted)["tolerant_match"])
        for field, value in (("revision", 2), ("primitive", "toggle_off"), ("wait_seconds", 1)):
            changed = deepcopy(ACT)
            changed["arguments"][field] = value
            self.assertFalse(self.score(changed)["tolerant_match"])
        changed = deepcopy(ACT)
        changed["arguments"]["target"]["image_ref"] = "rgb-00001-front"
        self.assertFalse(self.score(changed)["tolerant_match"])
        self.assertIsNone(self.score(changed)["point_distance"])

    def test_null_targets_and_non_act_calls_use_strict_matching(self):
        for gold in ({"name": "initialize", "arguments": {}},
                     {"name": "act", "arguments": {**ACT["arguments"], "primitive": "wait", "target": None}}):
            metrics = score_call(gold, deepcopy(gold), valid=True, point_tolerance=50)
            self.assertTrue(metrics["strict_match"] and metrics["tolerant_match"])
            self.assertIsNone(metrics["point_distance"])

    def test_json_equality_ignores_key_order_and_numeric_spelling_but_not_booleans_or_strings(self):
        self.assertTrue(json_equal({"a": 1.0, "b": [None]}, {"b": [None], "a": 1}))
        self.assertFalse(json_equal({"a": 1}, {"a": True}))
        self.assertFalse(json_equal({"a": " a "}, {"a": "a"}))

    def test_invalid_calls_count_in_denominators_and_groups(self):
        rows = [{"gold": ACT, "metrics": self.score(ACT), "error": None},
                {"gold": ACT, "metrics": score_call(ACT, None, valid=False, point_tolerance=50),
                 "error": {"code": "inference_out_of_memory"}}]
        result = summarize(rows)
        self.assertEqual(result["metrics"]["strict_accuracy"], {"correct": 1, "total": 2, "accuracy": 0.5})
        self.assertEqual(result["by_primitive"]["toggle_on"]["samples"], 2)
        self.assertEqual(result["by_tool"]["act"]["errors_by_code"], {"inference_out_of_memory": 1})
        self.assertIsNone(summarize([])["metrics"]["strict_accuracy"]["accuracy"])
