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
from manipulation_agent.tools import tool_specs
from manipulation_agent.vision_policy import system_prompt

from sft_fixtures import ROOT, Fixture, save, save_lines, png, SftFixtureTestCase

class SourceTests(SftFixtureTestCase):
    def test_real_shape_pipeline_masks_history_and_excludes_future_results(self):
        fixture = self.fixture()
        source = fixture.simple()
        episode = load_episode(source)
        rows = [r for r, _ in episode_samples(episode)]
        self.assertEqual([len(r["images"]) for r in rows], [0, 4, 8])
        for i, row in enumerate(rows):
            assistants = [m for m in row["messages"] if m["role"] == "assistant"]
            self.assertEqual(sum(a.get("loss") is not False for a in assistants), 1)
            self.assertIn("<think>\n\n</think>", assistants[-1]["content"])
            body = json.dumps(row)
            self.assertNotIn("NEVER_INCLUDE_PRIVATE", body)
            self.assertNotIn("NEVER_READ_FINAL_NOTES", body)
            self.assertNotIn(f"result-{i}", body)
        act = json.loads(rows[1]["messages"][-1]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(act["target"]["point_2d"], [123, 988])


    def test_evaluator_success_survives_missing_finish_and_failed_exit(self):
        fixture = self.fixture()
        fixture.simple(finish=False)
        fixture.controller.update(status="failed", exit_code=-15, formal_finish_observed=False)
        fixture.run_record["task_success"] = False
        episode = load_episode(fixture.flush())
        self.assertEqual(sum(d.target for d in episode.decisions), 2)
        self.assertTrue(episode.provenance["task_success"])
        self.assertFalse(episode.provenance["model_finish_observed"])
        self.assertEqual(episode.provenance["controller_exit_code"], -15)


    def test_success_is_not_inferred_from_finish_or_score(self):
        fixture = self.fixture()
        fixture.simple()
        fixture.run_record["evaluation"].update(official_task_success=False, goal_satisfaction_fraction=1.0)
        data = prepare_dataset([fixture.flush()])
        self.assertEqual(data.stats["samples"], 0)
        self.assertEqual(data.skipped[0]["reason"], "independent_task_not_successful")
        fixture.run_record["evaluation"] = {"goal_satisfaction_fraction": 1.0}
        self.assertEqual(load_episode(fixture.flush()).skipped, "independent_success_unknown")


    def test_mock_and_scripted_sources_are_rejected(self):
        fixture = self.fixture()
        fixture.simple()
        fixture.run_record["config"]["policy"] = "scripted"
        with self.assertRaisesRegex(ConversionError, "real Codex"):
            load_episode(fixture.flush())


    def test_failed_call_is_history_and_not_target(self):
        fixture = self.fixture()
        fixture.initialize()
        args = fixture.args()
        args["target"]["point"] = [1.5, 0.4]
        fixture.add("act", args, {"ok": False, "error": {"code": "invalid_arguments"}, "observation": fixture.observation})
        fixture.act()
        fixture.finish()
        episode = load_episode(fixture.flush())
        rows = [r for r, _ in episode_samples(episode)]
        self.assertEqual(len(rows), 3)
        self.assertIn("invalid_arguments", json.dumps(rows[1]))
        historical = [m for m in rows[1]["messages"] if m["role"] == "assistant"][-2]
        self.assertFalse(historical["loss"])
        self.assertIn("1500", historical["tool_calls"][0]["function"]["arguments"])


    def test_native_failed_receipt_preserves_error_and_later_samples(self):
        for reached_simulator in (False, True):
            with self.subTest(reached_simulator=reached_simulator):
                fixture = self.fixture(str(reached_simulator))
                fixture.initialize()
                fixture.add("act", fixture.args(), {"ok": False, "observation": fixture.observation,
                            "error": {"code": "UNDELIVERED_SERVER_ERROR"}})
                # Codex CLI 0.165.0 omits isError from the MCP result envelope;
                # a server error can have status=failed with error=null.
                error_text = "Error executing tool act: synthetic RGB download failure"
                fixture.raw[-2]["item"]["status"] = "in_progress"
                fixture.raw[-1]["item"].update(status="failed", error=None, result={
                    "content": [{"type": "text", "text": error_text}], "structured_content": None})
                for _ in range(3):
                    fixture.act()
                fixture.finish()
                if not reached_simulator:
                    fixture.events = [e for e in fixture.events if e.get("request_id") != "req-1"]
                source = fixture.flush()
                original = (fixture.ctrl / "model_events.jsonl").read_bytes()
                data = prepare_dataset([source])
                episode = data.episodes[0]
                records = list(data.records())
                self.assertEqual([meta["call_id"] for _, meta in records],
                                 ["item_0", "item_2", "item_3", "item_4", "item_5"])
                self.assertEqual([len(row["images"]) for row, _ in records], [0, 4, 8, 12, 16])
                self.assertFalse(episode.decisions[1].target)
                self.assertIsNone(episode.decisions[1].evidence_id)
                self.assertTrue(episode.provenance["task_success"])
                self.assertTrue(episode.provenance["model_finish_observed"])
                # Check the next decision and an older, compressed prefix.
                for record, _ in (records[1], records[-1]):
                    error_reply = next(m for m in record["messages"] if m.get("tool_call_id") == "item_1")
                    self.assertIn(error_text, error_reply["content"])
                    self.assertIn("Recorded MCP status=failed", error_reply["content"])
                    historical_call = next(m for m in record["messages"]
                                           if m.get("tool_calls", [{}])[0].get("id") == "item_1")
                    self.assertFalse(historical_call["loss"])
                    self.assertNotIn("UNDELIVERED_SERVER_ERROR", json.dumps(record))
                    self.assertNotIn("result-1", json.dumps(record))
                    validate_record(record)
                self.assertEqual((fixture.ctrl / "model_events.jsonl").read_bytes(), original)


    def test_native_failed_status_overrides_success_payload(self):
        for tool, call_id in (("act", "item_1"), ("finish", "item_2")):
            with self.subTest(tool=tool):
                fixture = self.fixture(tool)
                fixture.simple()
                item = next(row["item"] for row in fixture.raw
                            if row["type"] == "item.completed" and row["item"].get("id") == call_id)
                item.update(status="failed", error=None)
                item["result"]["structured_content"] = None
                episode = load_episode(fixture.flush())
                records = list(episode_samples(episode))
                self.assertEqual(len(records), 2)
                self.assertNotIn(call_id, [meta["call_id"] for _, meta in records])
                failed = next(d for d in episode.decisions if d.call_id == call_id)
                self.assertFalse(failed.target)
                self.assertIn({"type": "text", "text": "Recorded MCP status=failed"}, failed.content)
                self.assertEqual(episode.provenance["model_finish_observed"], tool != "finish")
                self.assertTrue(episode.provenance["task_success"])


    def test_native_failed_receipts_still_validate_present_evidence(self):
        for mismatch in ("missing_evidence", "evidence_mismatch"):
            with self.subTest(mismatch=mismatch):
                fixture = self.fixture(mismatch)
                fixture.simple()
                item = next(row["item"] for row in fixture.raw if row["type"] == "item.completed"
                            and row["item"].get("tool") == "act")
                item["status"] = "failed"
                result = json.loads(item["result"]["content"][0]["text"])
                if mismatch == "missing_evidence":
                    result["evidence_id"] = "unknown-evidence"
                else:
                    result["effect"]["status"] = "changed"
                item["result"]["content"][0]["text"] = json.dumps(result)
                with self.assertRaises(ConversionError) as caught:
                    load_episode(fixture.flush())
                self.assertEqual(caught.exception.code, mismatch)


    def test_nonfailed_native_receipt_still_requires_evidence(self):
        for status in (None, "completed"):
            with self.subTest(status=status):
                fixture = self.fixture(str(status))
                fixture.simple()
                item = next(row["item"] for row in fixture.raw if row["type"] == "item.completed"
                            and row["item"].get("tool") == "act")
                if status is not None:
                    item["status"] = status
                item["result"] = {"content": [{"type": "text", "text": "Error without a failure signal"}],
                                  "structured_content": None}
                with self.assertRaises(ConversionError) as caught:
                    load_episode(fixture.flush())
                self.assertEqual(caught.exception.code, "missing_evidence")


    def test_actual_overlapping_calls_still_rejected(self):
        fixture = self.fixture()
        fixture.simple()
        index = next(i for i, r in enumerate(fixture.raw) if r["type"] == "item.started")
        extra = deepcopy(fixture.raw[index])
        extra["item"]["id"] = "overlap"
        fixture.raw.insert(index + 1, extra)
        with self.assertRaises(ConversionError) as caught:
            load_episode(fixture.flush())
        self.assertEqual(caught.exception.code, "overlapping_calls")


    def test_finish_without_closed_receipt_is_not_a_target(self):
        fixture = self.fixture()
        fixture.simple(finish=False)
        fixture.add("finish", {"outcome": "achieved", "reason": "claim"}, {"ok": True, "closed": False})
        episode = load_episode(fixture.flush())
        self.assertEqual(len(list(episode_samples(episode))), 2)
        self.assertFalse(episode.provenance["model_finish_observed"])


    def test_successful_call_with_stale_image_or_revision_is_rejected(self):
        fixture = self.fixture()
        fixture.initialize()
        stale = fixture.args()
        fixture.act()
        fixture.add("act", stale, {"ok": True, "observation": fixture.capture()})
        with self.assertRaisesRegex(ConversionError, "latest delivered revision"):
            load_episode(fixture.flush())
        for row in fixture.raw:
            item = row.get("item", {})
            if item.get("id") == "item_2":
                item["arguments"]["revision"] = 1
        next(e for e in fixture.events if e.get("request_id") == "req-2" and e["kind"] == "tool_call")["arguments"]["revision"] = 1
        with self.assertRaisesRegex(ConversionError, "latest reply"):
            load_episode(fixture.flush())


    def test_long_trajectory_and_long_text_are_not_filtered(self):
        fixture = self.fixture()
        fixture.initialize()
        for _ in range(41):
            fixture.act()
        fixture.finish()
        fixture.raw[1]["item"]["text"] += " public text" * 10000
        episode = load_episode(fixture.flush())
        rows = list(episode_samples(episode))
        self.assertEqual(len(rows), 43)
        self.assertGreater(rows[0][1]["approx_tokens"], 24576)


    def test_successful_invalid_coordinates_are_not_repaired(self):
        fixture = self.fixture()
        fixture.initialize()
        arguments = fixture.args()
        arguments["target"]["point"] = [-0.1, 0.5]
        fixture.add("act", arguments, {"ok": True, "observation": fixture.capture()})
        with self.assertRaisesRegex(ConversionError, "schema range"):
            load_episode(fixture.flush())


    def test_schema_snapshot_and_recorded_prompt_are_historical(self):
        fixture = self.fixture()
        fixture.simple()
        fixture.controller["command"][-1] = "developer_instructions=" + json.dumps("ARCHIVED_PROMPT target.point=[0.5,0.5]")
        episode = load_episode(fixture.flush())
        self.assertIn("ARCHIVED_PROMPT", episode.system)
        fixture.controller["mcp_preflight"]["schema_sha256"] = "bad"
        with self.assertRaisesRegex(ConversionError, "catalog differs"):
            load_episode(fixture.flush())


    def test_historical_executor_options_and_provenance_are_restored(self):
        for keys in (("executor_profile",), ("executor",), ("executor_profile", "executor")):
            with self.subTest(keys=keys):
                fixture = self.fixture("-".join(keys))
                fixture.legacy_catalog(recorded_actions="prune")
                fixture.run_record["config"].update({key: "official_symbolic" for key in keys}, nonofficial_actions="prune")
                episode = load_episode(fixture.simple())
                restore = episode.provenance["catalog_restore"]
                self.assertEqual(restore["arguments"], {"profile": "skills", "executor": "official_symbolic", "nonofficial_actions": "prune"})
                self.assertEqual(restore["argument_sources"]["executor"], "run.config." + keys[0])
                self.assertEqual(restore["argument_sources"]["nonofficial_actions"], "run.config.nonofficial_actions")
                self.assertEqual(episode.provenance["schema_sha256"], fixture.controller["mcp_preflight"]["schema_sha256"])
                self.assertEqual(len(list(episode_samples(episode))), 3)


    def test_missing_catalog_options_use_frozen_defaults_only(self):
        fixture = self.fixture()
        fixture.legacy_catalog(recorded_executor="snapshot-custom", recorded_actions="snapshot-retain")
        episode = load_episode(fixture.simple())
        restore = episode.provenance["catalog_restore"]
        self.assertEqual(restore["arguments"]["executor"], "snapshot-custom")
        self.assertEqual(restore["arguments"]["nonofficial_actions"], "snapshot-retain")
        for key in ("executor", "nonofficial_actions"):
            self.assertEqual(restore["argument_sources"][key], "snapshot.tool_specs.default")
        fixture.run_record["config"]["executor"] = "official_symbolic"
        with self.assertRaisesRegex(ConversionError, "catalog differs"):
            load_episode(fixture.flush())


    def test_profile_only_catalog_ignores_unsupported_options_but_checks_hash(self):
        fixture = self.fixture()
        fixture.run_record["config"].update(executor_profile="official_symbolic", nonofficial_actions="retain")
        episode = load_episode(fixture.simple())
        restore = episode.provenance["catalog_restore"]
        self.assertEqual(restore["arguments"], {"profile": "skills"})
        self.assertEqual(restore["argument_sources"], {"profile": "controller.agent_profile"})
        self.assertEqual(restore["unsupported_recorded_options"], {"executor": "official_symbolic", "nonofficial_actions": "retain"})
        fixture.controller["mcp_preflight"]["tool_names"].pop()
        with self.assertRaisesRegex(ConversionError, "catalog differs"):
            load_episode(fixture.flush())


    def test_conflicting_and_invalid_recorded_catalog_options_are_rejected(self):
        fixture = self.fixture()
        fixture.simple()
        fixture.run_record["config"].update(executor="custom", executor_profile="official_symbolic")
        with self.assertRaises(ConversionError) as caught:
            load_episode(fixture.flush())
        self.assertEqual(caught.exception.code, "conflicting_executor_config")
        del fixture.run_record["config"]["executor_profile"]
        for value in (None, "", 42):
            with self.subTest(value=value):
                fixture.run_record["config"]["executor"] = value
                with self.assertRaises(ConversionError) as caught:
                    load_episode(fixture.flush())
                self.assertEqual(caught.exception.code, "invalid_catalog_config")


    def test_changed_snapshot_is_rejected(self):
        fixture = self.fixture()
        source = fixture.simple()
        path = fixture.ctrl / "source_snapshot/manipulation_agent/tools/action.py"
        path.write_text(path.read_text() + "\n# modified\n")
        with self.assertRaisesRegex(ConversionError, "snapshot differs"):
            load_episode(source)


    def test_corrupt_and_missing_images_are_rejected(self):
        fixture = self.fixture()
        source = fixture.simple()
        path = fixture.run / fixture.captures[0]["images"][0]["file"]
        path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ConversionError, "image bytes disagree"):
            load_episode(source)
        path.unlink()
        with self.assertRaisesRegex(ConversionError, "recorded image bytes"):
            load_episode(source)


    def test_call_pair_and_simulator_evidence_mismatches_are_rejected(self):
        fixture = self.fixture()
        fixture.simple()
        started = next(r for r in fixture.raw if r["type"] == "item.started" and r["item"]["tool"] == "act")
        started["item"]["arguments"]["revision"] = 5
        with self.assertRaisesRegex(ConversionError, "Started and completed"):
            load_episode(fixture.flush())
        started["item"]["arguments"]["revision"] = 0
        next(e for e in fixture.events if e.get("kind") == "tool_call" and e["name"] == "act")["arguments"]["revision"] = 5
        with self.assertRaisesRegex(ConversionError, "simulator evidence"):
            load_episode(fixture.flush())


    def test_unlogged_simulator_actions_are_not_accepted_as_model_behavior(self):
        fixture = self.fixture()
        fixture.simple()
        fixture.events.insert(1, {"id": "extra", "kind": "tool_call", "request_id": "unlogged",
                                  "origin": "controller", "name": "act", "arguments": {"primitive": "toggle_on"}})
        with self.assertRaisesRegex(ConversionError, "controller trace"):
            load_episode(fixture.flush())


    def test_supervisor_finish_is_not_a_training_target(self):
        fixture = self.fixture()
        fixture.simple(finish=False)
        fixture.events.append({"id": "closure", "kind": "tool_call", "request_id": "supervisor",
                               "origin": "supervisor", "name": "finish", "arguments": {"outcome": "aborted"}})
        episode = load_episode(fixture.flush())
        self.assertEqual(len(list(episode_samples(episode))), 2)
        self.assertFalse(episode.provenance["model_finish_observed"])


    def test_oracle_fields_in_observations_violate_rgb_boundary(self):
        fixture = self.fixture()
        fixture.simple()
        first = next(e for e in fixture.events if e.get("kind") == "tool_result")
        first["result"]["observation"]["oracle_objects"] = ["private object"]
        model = next(r["item"] for r in fixture.raw if r["type"] == "item.completed" and r.get("item", {}).get("type") == "mcp_tool_call")
        result = json.loads(model["result"]["content"][0]["text"])
        result["observation"]["oracle_objects"] = ["private object"]
        model["result"]["content"][0]["text"] = json.dumps(result)
        with self.assertRaisesRegex(ConversionError, "private fields"):
            load_episode(fixture.flush())


    def test_duplicate_completion_does_not_duplicate_samples(self):
        fixture = self.fixture()
        fixture.simple()
        fixture.raw.append(deepcopy(fixture.raw[-1]))
        episode = load_episode(fixture.flush())
        self.assertEqual(len(list(episode_samples(episode))), 3)
        self.assertEqual(episode.warnings[0]["code"], "duplicate_completion_ignored")


    def test_unfinished_tail_call_does_not_invalidate_successful_prefix(self):
        fixture = self.fixture()
        fixture.simple(finish=False)
        before = [r for r, _ in episode_samples(load_episode(fixture.flush()))]
        fixture.raw.append({"type": "item.started", "item": {"id": "unfinished", "type": "mcp_tool_call", "server": "manipulation",
                            "tool": "finish", "arguments": {"outcome": "achieved", "reason": "closing"}}})
        fixture.interleave("unfinished", ["UNFINISHED_PROGRESS", "STILL_PENDING"])
        episode = load_episode(fixture.flush())
        self.assertEqual(len(list(episode_samples(episode))), 2)
        self.assertEqual([r for r, _ in episode_samples(episode)], before)
        self.assertEqual(episode.warnings[0]["code"], "unfinished_call_not_a_target")


    def test_missing_dispatch_boundary_and_resumed_context_are_rejected(self):
        fixture = self.fixture()
        fixture.simple()
        fixture.raw = [r for r in fixture.raw if r["type"] != "item.started"]
        with self.assertRaisesRegex(ConversionError, "item.started"):
            load_episode(fixture.flush())
        fixture.controller["rate_limit_continuations"] = [{"attempt": 1}]
        with self.assertRaisesRegex(ConversionError, "additional user prompts"):
            load_episode(fixture.flush())


    def test_unknown_input_events_are_not_silently_discarded(self):
        fixture = self.fixture()
        fixture.simple()
        fixture.raw.insert(1, {"type": "user_message", "text": "Unaccounted follow-up"})
        with self.assertRaisesRegex(ConversionError, "Unknown event"):
            load_episode(fixture.flush())


