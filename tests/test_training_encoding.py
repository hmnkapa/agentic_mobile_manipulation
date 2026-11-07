"""Synthetic CPU fixtures only; these are not autonomous simulator experiments."""
from copy import deepcopy
import base64
import hashlib
import io
from itertools import combinations
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
import zlib

from manipulation_agent.sft.cli import main
from manipulation_agent.sft.common import ConversionError
from manipulation_agent.sft.dataset import (choose_validation_scenes, episode_samples,
    manifest_sources, prepare_dataset, write_package)
from manipulation_agent.sft.history import compress_history, validate_record
from manipulation_agent.sft.protocol import (build_student_context, from_student_arguments,
    rewrite_interface_text, to_student_arguments)
from manipulation_agent.sft.source import EpisodeSource, load_episode
from manipulation_agent.sft.verify_encode import check_encoding
from manipulation_agent.tools import tool_specs
from manipulation_agent.vision_policy import system_prompt

from sft_fixtures import ROOT, Fixture, save, save_lines, png, SftFixtureTestCase

class EncodingContractTests(unittest.TestCase):
    def test_optional_encoder_checks_mask_contract_without_claiming_real_qwen(self):
        seen = []
        class Preprocessor:
            def preprocess(self, row):
                seen.append(deepcopy(row))
                messages = []
                for message in row["messages"]:
                    calls = message.pop("tool_calls", [])
                    messages.append(message)
                    for call in calls:
                        messages.append({"role": "tool_call", "content": call["function"],
                                         "loss": message.get("loss", True)})
                row["messages"] = messages
                return row
        class Tokenizer:
            unk_token_id = -1
            def decode(self, tokens, **kwargs):
                return "<think></think><tool_call>initialize</tool_call>"
            def convert_tokens_to_ids(self, text):
                return 99
        class Template:
            tokenizer = Tokenizer()
            def encode(self, record, **kwargs):
                assert record["messages"][-1]["role"] == "tool_call"
                assert not any(m.get("tool_calls") for m in record["messages"])
                return {"input_ids": [1, 2], "labels": [-100, -100] if record["messages"][-1].get("loss") is False else [-100, 2]}
        row = {"messages": [{"role": "assistant", "content": "progress", "loss": False},
                            {"role": "assistant", "content": "<think>\n\n</think>",
                             "tool_calls": [{"function": {"name": "initialize", "arguments": "{}"}}]}], "images": []}
        original = deepcopy(row)
        self.assertTrue(check_encoding(row, Template(), object(), preprocessor=Preprocessor())["all_history_masked"])
        self.assertEqual(row, original)
        self.assertEqual(len(seen), 2)
        self.assertNotIn("loss", seen[0]["messages"][-1])
        self.assertFalse(seen[1]["messages"][-1]["loss"])
        self.assertEqual(seen[0]["messages"][:-1], seen[1]["messages"][:-1])
        class BrokenTemplate(Template):
            def encode(self, record, **kwargs):
                return {"input_ids": [1, 2], "labels": [1, 2]}
        with self.assertRaisesRegex(ConversionError, "loss contract"):
            check_encoding(row, BrokenTemplate(), object(), preprocessor=Preprocessor())
        class OldPreprocessor:
            def preprocess(self, row):
                return row
        with self.assertRaisesRegex(ConversionError, "normalize all OpenAI"):
            check_encoding(row, Template(), object(), preprocessor=OldPreprocessor())


class ExactEncodingTests(unittest.TestCase):
    """Character tokens expose dropped arguments without a real model."""
    class Tokenizer:
        unk_token_id = -1
        def decode(self, tokens, **kwargs):
            return "".join(chr(t) for t in tokens if t != 500000)
        def convert_tokens_to_ids(self, text):
            return 500000

    class Preprocessor:
        def preprocess(self, row):
            messages = []
            for message in row["messages"]:
                calls = message.pop("tool_calls", [])
                messages.append(message)
                for call in calls:
                    messages.append({"role": "tool_call", "content": call["function"], "loss": message.get("loss", True)})
            row["messages"] = messages
            return row

    @staticmethod
    def render(function):
        return "<tool_call>" + json.dumps({"name": function["name"], "arguments": json.loads(function["arguments"])}, sort_keys=True) + "</tool_call>"

    def template(self, *, drop_arguments=False):
        parent = self
        class Template:
            tokenizer = parent.Tokenizer()
            def encode(self, row, **kwargs):
                ids = [500000] * len(row["images"])
                labels = [-100] * len(ids)
                for m in row["messages"]:
                    content = parent.render(m["content"]) if m["role"] == "tool_call" else m["content"]
                    if drop_arguments:
                        content = content.replace('"operation": "read"', '"operation": "write"')
                    tokens = list(map(ord, content))
                    ids.extend(tokens)
                    labels.extend(tokens if m["role"] in {"assistant", "tool_call"} and m.get("loss", True) else [-100] * len(tokens))
                return {"input_ids": ids, "labels": labels, "image_grid_thw": [[1, 1, 1]] * len(row["images"])}
        return Template()

    def row(self, images=0):
        return {"messages": [{"role": "assistant", "content": "old history", "loss": False},
                             {"role": "assistant", "content": "<think>\n\n</think>", "tool_calls": [
                                 {"function": {"name": "task_notes", "arguments": '{"operation":"read"}'}}]}],
                "images": ["image.png"] * images, "tools": []}

    def test_complete_arguments_counterfactual_and_images(self):
        from manipulation_agent.training.encoding import check_encoding
        for images in (0, 8):
            row = self.row(images)
            original = deepcopy(row)
            result = check_encoding(row, self.template(), object(), preprocessor=self.Preprocessor(), render_target=self.render)
            self.assertTrue(result["complete_target_verified"])
            self.assertEqual(result["image_tokens"], images)
            self.assertEqual(row, original)

    def test_present_name_but_changed_parameter_does_not_pass(self):
        from manipulation_agent.training.encoding import check_encoding
        with self.assertRaisesRegex(ConversionError, "Complete rendered"):
            check_encoding(self.row(), self.template(drop_arguments=True), object(),
                           preprocessor=self.Preprocessor(), render_target=self.render)

    def test_matching_image_count_with_wrong_grid_token_total_fails(self):
        from manipulation_agent.training.encoding import check_encoding
        template = self.template()
        encode = template.encode

        def wrong_grid(row, **kwargs):
            result = encode(row, **kwargs)
            result["image_grid_thw"] = [[1, 2, 2]]
            return result

        template.encode = wrong_grid
        with self.assertRaisesRegex(ConversionError, "tokens/grid"):
            check_encoding(self.row(1), template, object(), preprocessor=self.Preprocessor(), render_target=self.render)

    def test_eos_follows_masked_message_including_merged_message_lists(self):
        from manipulation_agent.training.swift_loss import suffix_context_types
        original = ["prompt", "response", "suffix"]
        for loss in (False, [False, False], [False, [False]]):
            self.assertEqual(suffix_context_types(original, [{"loss": loss}], "suffix", "other"),
                             ["prompt", "response", "other"])
        for loss in (None, True, [False, True], [False, None]):
            self.assertEqual(suffix_context_types(original, [{"loss": loss}], "suffix", "other"), original)
        self.assertEqual(original, ["prompt", "response", "suffix"])

