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

class HistoryTests(SftFixtureTestCase):
    def test_interleaved_text_stays_after_dispatch_and_before_receipt(self):
        for parts in (["progress"], ["first progress", "second progress"]):
            with self.subTest(parts=parts):
                fixture = self.fixture(str(len(parts)))
                baseline = [r for r, _ in episode_samples(load_episode(fixture.simple()))]
                fixture.interleave("item_1", parts)
                episode = load_episode(fixture.flush())
                rows = [r for r, _ in episode_samples(episode)]
                self.assertEqual(rows[:2], baseline[:2])
                decision = episode.decisions[1]
                self.assertEqual([p.text for p in decision.interleaved_text], parts)
                self.assertEqual([p.line for p in decision.interleaved_text],
                                 list(range(decision.call_line + 1, decision.result_line)))
                messages = rows[-1]["messages"]
                start = next(i for i, m in enumerate(messages) if m.get("tool_calls", [{}])[0].get("id") == "item_1")
                self.assertEqual(messages[start]["content"], "Public action explanation")
                self.assertEqual(messages[start + 1:start + 1 + len(parts)],
                                 [{"role": "assistant", "content": p, "loss": False} for p in parts])
                self.assertEqual(messages[start + 1 + len(parts)]["tool_call_id"], "item_1")
                self.assertEqual(messages[-1]["content"], "<think>\n\n</think>")
                for row in rows:
                    validate_record(row)


    def test_interleaved_text_target_images_and_future_notes_do_not_leak(self):
        variants = []
        for variant in range(2):
            fixture = self.fixture(str(variant))
            fixture.initialize()
            fixture.add("act", fixture.args(), {"ok": True, "observation": fixture.capture(color_offset=variant)}, text="before")
            fixture.interleave("item_1", [f"AFTER_DISPATCH_{variant}"])
            fixture.add("task_notes", {"operation": "read", "content": None, "expected_version": None},
                        {"ok": True, "notes": {"version": variant, "content": f"FUTURE_NOTE_{variant}"}})
            fixture.finish()
            rows = [r for r, _ in episode_samples(load_episode(fixture.flush()))]
            variants.append(rows)
            self.assertNotIn("AFTER_DISPATCH", json.dumps(rows[1]))
            self.assertNotIn("FUTURE_NOTE", json.dumps(rows[2]))
            self.assertIn(f"AFTER_DISPATCH_{variant}", json.dumps(rows[2]))
            self.assertIn(f"FUTURE_NOTE_{variant}", json.dumps(rows[-1]))
        self.assertEqual(variants[0][:2], variants[1][:2])
        self.assertNotEqual(variants[0][2]["images"], variants[1][2]["images"])


    def test_failed_interleaved_call_is_history_including_after_compression(self):
        fixture = self.fixture()
        fixture.initialize()
        fixture.add("act", fixture.args(), {"ok": False, "error": {"code": "ACTION_FAILED"},
                                            "observation": fixture.observation})
        fixture.interleave("item_1", ["FIRST_PROGRESS", "SECOND_PROGRESS"])
        for _ in range(3):
            fixture.act()
        fixture.finish()
        episode = load_episode(fixture.flush())
        records = list(episode_samples(episode))
        self.assertNotIn("item_1", [m["call_id"] for _, m in records])
        for record, _ in (records[1], records[-1]):
            messages = record["messages"]
            first = next(i for i, m in enumerate(messages) if m["content"] == "FIRST_PROGRESS")
            self.assertEqual(messages[first + 1], {"role": "assistant", "content": "SECOND_PROGRESS", "loss": False})
            self.assertIn("ACTION_FAILED", messages[first + 2]["content"])
            self.assertFalse(messages[first]["loss"])
            self.assertTrue(messages[first - 1].get("tool_calls") or messages[first - 1]["content"].startswith("Action: "))
            validate_record(record)
        self.assertEqual(len(records[-1][0]["images"]), 20)


    def test_compression_matches_ids_preserving_interleaved_text_and_k2(self):
        fixture = self.fixture()
        fixture.initialize()
        for _ in range(3):
            fixture.act()
        fixture.finish()
        fixture.interleave("item_1", ["OLD_PROGRESS"])
        episode = load_episode(fixture.flush())
        history = [m for d in episode.decisions[:-1] for m in d.history()]
        original = deepcopy(history)
        result = compress_history(history)
        self.assertEqual(history, original)
        index = next(i for i, m in enumerate(result) if m.get("content") == "OLD_PROGRESS")
        self.assertTrue(result[index - 1]["content"].startswith("Action: "))
        self.assertEqual(json.loads(result[index - 1]["content"].removeprefix("Action: "))["arguments"]["target"]["point_2d"], [123, 988])
        self.assertEqual(result[index], {"role": "assistant", "content": "OLD_PROGRESS", "loss": False})
        self.assertTrue(result[index + 1]["content"][0]["text"].startswith("Result: "))
        for call_id in ("item_2", "item_3"):
            before = [m for m in original if m.get("tool_call_id") == call_id or m.get("tool_calls", [{}])[0].get("id") == call_id]
            self.assertEqual([m for m in result if m.get("tool_call_id") == call_id or m.get("tool_calls", [{}])[0].get("id") == call_id], before)
        images = lambda ms: [p for m in ms if isinstance(m["content"], list) for p in m["content"][1:]]
        self.assertEqual(images(result), images(original))
        for bad in ("missing", "duplicate", "out_of_order"):
            with self.subTest(bad=bad):
                broken = deepcopy(history)
                if bad == "missing":
                    broken[1]["tool_call_id"] = "absent"
                elif bad == "duplicate":
                    broken.append(deepcopy(broken[1]))
                else:
                    broken[0], broken[1] = broken[1], broken[0]
                with self.assertRaisesRegex(ConversionError, "pairing"):
                    compress_history(broken)


    def test_transport_error_survives_old_history_compression(self):
        fixture = self.fixture()
        fixture.initialize()
        fixture.add("task_notes", {"operation": "read", "content": None, "expected_version": None},
                    {"ok": True, "notes": {"version": 0, "content": ""}})
        fixture.raw[-1]["item"].update(result=None, error={"message": "ACTUAL_TRANSPORT_TIMEOUT"})
        for _ in range(3):
            fixture.act()
        fixture.finish()
        episode = load_episode(fixture.flush())
        records = [r for r, _ in episode_samples(episode)]
        self.assertEqual(len(records), 5)
        self.assertIn("ACTUAL_TRANSPORT_TIMEOUT", json.dumps(records[-1]))
        self.assertEqual(episode.provenance["evidence_alignment"], "partial_transport_errors_recorded")


    def test_compression_keeps_skills_latest_notes_all_images_and_metadata(self):
        fixture = self.fixture()
        fixture.initialize()
        fixture.add("read_skill", {"name": "visual-exploration", "resource": "SKILL.md"},
                    {"ok": True, "text": "KEEP_SKILL target={image_ref, point:[x,y]}"})
        fixture.add("task_notes", {"operation": "write", "expected_version": 0, "content": "OLD_NOTE"},
                    {"ok": True, "notes": {"version": 1, "content": "OLD_NOTE"}})
        fixture.act()
        fixture.act()
        fixture.add("task_notes", {"operation": "write", "expected_version": 1, "content": "LATEST_NOTE"},
                    {"ok": True, "notes": {"version": 2, "content": "LATEST_NOTE"}})
        fixture.act()
        fixture.finish()
        fixture.interleave("item_1", ["SKILL_IN_FLIGHT"])
        fixture.interleave("item_2", ["EARLIER_WRITE_IN_FLIGHT"])
        fixture.interleave("item_5", ["LATEST_NOTE_IN_FLIGHT"])
        episode = load_episode(fixture.flush())
        rows = [r for r, _ in episode_samples(episode)]
        last = json.dumps(rows[-1])
        self.assertIn("KEEP_SKILL", last)
        self.assertIn("LATEST_NOTE", last)
        self.assertNotIn("OLD_NOTE", last)
        for text in ("SKILL_IN_FLIGHT", "EARLIER_WRITE_IN_FLIGHT", "LATEST_NOTE_IN_FLIGHT"):
            self.assertIn({"role": "assistant", "content": text, "loss": False}, rows[-1]["messages"])
        self.assertIn("target_locked_approach_v2", last)
        self.assertIn("Action:", last)
        self.assertIn("point_2d", last)
        self.assertEqual(len(rows[-1]["images"]), 16)
        self.assertNotIn("LATEST_NOTE", json.dumps(rows[3]))
        source_history = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        for step in episode.decisions:
            source_history.extend(step.history())
        original = deepcopy(source_history)
        self.assertEqual(compress_history(source_history), compress_history(source_history))
        self.assertEqual(source_history, original)


