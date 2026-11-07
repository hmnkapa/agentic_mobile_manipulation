"""CPU-only context-budget behavior; synthetic messages are not task evidence."""
from copy import deepcopy
import io
import json
import unittest
from contextlib import redirect_stdout

from manipulation_agent.sft import build_sample_messages
from manipulation_agent.sft.cli import main
from manipulation_agent.sft.common import ConversionError
from manipulation_agent.sft.dataset import episode_samples, prepare_dataset, write_package
from manipulation_agent.sft.format import estimate_length, to_msswift, validate_record
from manipulation_agent.sft.history import compress_history
from manipulation_agent.sft.source import load_episode
from sft_fixtures import SftFixtureTestCase


def call(identity, tool="act", *, target=False):
    message = {"role": "assistant", "content": "PUBLIC_" + identity,
               "tool_calls": [{"id": identity, "type": "function", "function": {
                   "name": tool, "arguments": json.dumps({"image_ref": identity + "-left"})}}]}
    if not target:
        message["loss"] = False
    return message


def group(identity, *, rgb=True, tool="act", result=None, progress=None):
    views = ("left", "front", "right", "back")  # First delivered image need not be front.
    if result is None:
        result = {"ok": True}
    if rgb:
        result = {**result, "observation": {"images": [
            {"view": view, "image_ref": identity + "-" + view} for view in views]}}
    blocks = [{"type": "text", "text": json.dumps(result)}]
    if rgb:
        for view in views:
            blocks.extend([{"type": "text", "text": f"RGB view={view}, image_ref={identity}-{view}"},
                           {"type": "image", "image": f"{identity}-{view}.png"}])
    blocks.append({"type": "text", "text": "EXTRA_" + identity})
    return [call(identity, tool),
            {"role": "assistant", "content": progress or "PROGRESS_" + identity, "loss": False},
            {"role": "tool", "tool_call_id": identity, "name": tool, "content": blocks}]


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.prefix = [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "USER"}]
        self.target = call("target", target=True)
        self.tools = [{"type": "function", "function": {"name": "act", "parameters": {"type": "object"}}}]
        self.groups = [group(str(i)) for i in range(4)]
        self.history = self.prefix + [m for g in self.groups for m in g]

    def tokens(self, messages):
        return estimate_length(to_msswift(messages, self.tools))["approx_tokens"]

    def check(self, messages, stats):
        record = to_msswift(messages, self.tools)
        validate_record(record)
        self.assertEqual({key: stats[key] for key in estimate_length(record)}, estimate_length(record))
        self.assertEqual(stats["image_count_before"] - stats["image_count"], stats["images_removed"])
        self.assertEqual(messages[:2], self.prefix)
        self.assertEqual(messages[-1], self.target)
        return record

    def test_equal_under_budget_and_disabled_preserve_previous_messages(self):
        expected = compress_history(self.history) + [self.target]
        size = self.tokens(expected)
        for limit in (0, size, size + 1):
            with self.subTest(limit=limit):
                messages, stats = build_sample_messages(self.history, self.target, self.tools, max_len=limit)
                self.assertEqual(messages, expected)
                self.assertFalse(stats["slim_triggered"])
                self.assertFalse(stats["over_budget"])
                self.assertEqual(stats["dropped_history_pairs"], 0)
                self.assertEqual(stats["images_removed"], 0)
                self.check(messages, stats)

    def test_slim_first_images_and_labels_before_dropping_any_history(self):
        baseline = compress_history(self.history) + [self.target]
        limit = self.tokens(baseline) - 1
        messages, stats = build_sample_messages(self.history, self.target, self.tools, max_len=limit)
        record = self.check(messages, stats)
        expected_images = ["0-left.png", "1-left.png"] + [
            f"{i}-{view}.png" for i in (2, 3) for view in ("left", "front", "right", "back")]
        self.assertEqual(record["images"], expected_images)
        self.assertEqual(stats["slimmed_images"], 6)
        self.assertEqual(stats["dropped_history_pairs"], 0)
        self.assertTrue(stats["slim_triggered"])
        self.assertFalse(stats["over_budget"])
        self.assertLessEqual(stats["approx_tokens"], limit)
        self.assertEqual(messages[-7:], self.groups[2] + self.groups[3] + [self.target])
        for index in (4, 7):
            original, slimmed = baseline[index]["content"], messages[index]["content"]
            self.assertEqual(slimmed, [original[0], original[1], original[2], original[-1]])
        for i in (0, 1):
            self.assertIn("PROGRESS_" + str(i), json.dumps(messages))
            self.assertNotIn(f"RGB view=back, image_ref={i}-back", json.dumps(messages))

    def test_delete_multiple_complete_groups_then_stop_at_protected_window(self):
        expected = self.prefix + self.groups[2] + self.groups[3] + [self.target]
        limit = self.tokens(expected)
        messages, stats = build_sample_messages(self.history, self.target, self.tools, max_len=limit)
        self.assertEqual(messages, expected)
        self.assertEqual(stats["dropped_history_pairs"], 2)
        self.assertEqual(stats["slimmed_images"], 6)
        self.assertEqual(stats["images_removed"], 8)
        self.assertEqual(stats["approx_tokens"], limit)
        self.assertFalse(stats["over_budget"])
        self.check(messages, stats)

    def test_stop_deleting_as_soon_as_the_budget_is_met(self):
        self.history[3]["content"] = "OLD_PROGRESS " * 1000
        # Enough room for the second old call even before its images are slimmed.
        limit = self.tokens(compress_history(self.prefix + self.history[5:]) + [self.target])
        messages, stats = build_sample_messages(self.history, self.target, self.tools, max_len=limit)
        self.assertEqual(stats["dropped_history_pairs"], 1)
        self.assertEqual(stats["image_count"], 9)
        self.assertNotIn("OLD_PROGRESS", json.dumps(messages))
        self.assertIn("PROGRESS_1", json.dumps(messages))
        self.assertFalse(stats["over_budget"])
        self.check(messages, stats)

    def test_default_24576_budget_actually_trims_long_history(self):
        self.history[3]["content"] = "OLD_PROGRESS " * 10000
        messages, stats = build_sample_messages(self.history, self.target, self.tools)
        self.assertGreater(stats["approx_tokens_before"], 24576)
        self.assertLessEqual(stats["approx_tokens"], 24576)
        self.assertEqual(stats["slimmed_images"], 6)
        self.assertEqual(stats["dropped_history_pairs"], 1)
        self.assertNotIn("OLD_PROGRESS", json.dumps(messages))
        self.assertIn("PROGRESS_1", json.dumps(messages))
        self.check(messages, stats)

    def test_old_skills_latest_notes_and_transport_error_can_be_deleted(self):
        specials = [group("skills", rgb=False, tool="list_skills"),
                    group("skill", rgb=False, tool="read_skill", result={"ok": True, "text": "SKILL_BODY"}),
                    group("notes", rgb=False, tool="task_notes", result={"ok": True, "notes": {
                        "version": 7, "content": "LATEST_NOTE"}}),
                    group("error", rgb=False)]
        specials[-1][-1]["content"][0]["text"] = "Recorded MCP transport error: TIMEOUT"
        tail = group("tail", rgb=False, tool="read_skill")
        history = self.prefix + [m for g in specials + self.groups for m in g] + tail
        messages, stats = build_sample_messages(history, self.target, self.tools, max_len=1)
        self.assertEqual(messages, self.prefix + self.groups[2] + self.groups[3] + tail + [self.target])
        self.assertEqual(stats["dropped_history_pairs"], 6)
        self.assertTrue(stats["over_budget"])
        self.check(messages, stats)

    def test_k_counts_rgb_replies_and_protects_the_whole_suffix(self):
        text_group = group("between", rgb=False, tool="task_notes")
        history = self.prefix + self.groups[0] + self.groups[1] + text_group + self.groups[2] + self.groups[3]
        messages, stats = build_sample_messages(history, self.target, self.tools, max_len=1)
        self.assertEqual(messages, self.prefix + self.groups[2] + self.groups[3] + [self.target])
        self.assertEqual(stats["dropped_history_pairs"], 3)
        for k in (1, 3, 4, 5):
            with self.subTest(k=k):
                messages, stats = build_sample_messages(self.history, self.target, self.tools, k=k, max_len=1)
                self.assertEqual(messages, self.prefix + [m for g in self.groups[-k:] for m in g] + [self.target])
                self.assertEqual(stats["image_count"], 4 * min(k, 4))
                self.assertTrue(stats["over_budget"])
                self.check(messages, stats)

    def test_fewer_than_k_observations_keeps_all_history(self):
        for rgb_count in (0, 1):
            history = self.prefix + group("setup", rgb=False, tool="read_skill")
            history += [m for g in self.groups[:rgb_count] for m in g]
            messages, stats = build_sample_messages(history, self.target, self.tools, max_len=1)
            self.assertEqual(messages, history + [self.target])
            self.assertEqual(stats["dropped_history_pairs"], 0)
            self.assertEqual(stats["images_removed"], 0)
            self.assertTrue(stats["over_budget"])

    def test_repeated_capture_is_still_two_protected_observations(self):
        repeated = group("repeat")
        repeated[-1]["content"] = deepcopy(self.groups[-1][-1]["content"])
        history = self.history + repeated
        messages, stats = build_sample_messages(history, self.target, self.tools, max_len=1)
        self.assertEqual(messages, self.prefix + self.groups[-1] + repeated + [self.target])
        record = self.check(messages, stats)
        self.assertEqual(record["images"][:4], record["images"][4:])
        self.assertEqual(stats["image_count"], 8)

    def test_default_budget_keeps_oversized_protected_text_and_tools(self):
        for field in ("system", "user", "target", "tools"):
            with self.subTest(field=field):
                prefix, target, tools = deepcopy((self.prefix, self.target, self.tools))
                if field in ("system", "user"):
                    prefix[0 if field == "system" else 1]["content"] = "x" * 100000
                elif field == "target":
                    target["content"] = "x" * 100000
                else:
                    tools[0]["function"]["description"] = "x" * 100000
                messages, stats = build_sample_messages(prefix, target, tools)
                self.assertEqual(messages, prefix + [target])
                self.assertTrue(stats["over_budget"])
                self.assertGreater(stats["approx_tokens"], 24576)
                self.assertEqual(stats["dropped_history_pairs"], 0)

    def test_inputs_stay_immutable_and_each_call_starts_from_raw_history(self):
        before = deepcopy((self.history, self.target, self.tools))
        first = build_sample_messages(self.history, self.target, self.tools, max_len=1)
        self.assertEqual(first, build_sample_messages(self.history, self.target, self.tools, max_len=1))
        restored, _ = build_sample_messages(self.history, self.target, self.tools, max_len=0)
        self.assertEqual(restored, compress_history(self.history) + [self.target])
        first[0][0]["content"] = "mutated output"
        self.assertEqual((self.history, self.target, self.tools), before)

    def test_invalid_limits_and_unsafe_call_groups_fail_explicitly(self):
        for limit in (-1, True, None, 24.5, "24576"):
            with self.subTest(limit=limit), self.assertRaisesRegex(ConversionError, "nonnegative integer"):
                build_sample_messages(self.history, self.target, self.tools, max_len=limit)
        for middle in (call("overlap"), {"role": "user", "content": "PROTECTED_USER"}):
            history = deepcopy(self.history)
            history.insert(3, middle)
            with self.subTest(middle=middle), self.assertRaisesRegex(ConversionError, "non-serial"):
                build_sample_messages(history, self.target, self.tools, max_len=1)


class BudgetDatasetTests(SftFixtureTestCase):
    def long_fixture(self, name="one", **kwargs):
        fixture = self.fixture(name, **kwargs)
        fixture.initialize()
        for _ in range(4):
            fixture.act()
        fixture.finish()
        fixture.interleave("item_1", ["OLD_PROGRESS"])
        return fixture

    def test_samples_preserve_targets_sources_and_no_future_observations(self):
        fixture = self.long_fixture()
        episode = load_episode(fixture.flush())
        before = deepcopy(episode)
        full = list(episode_samples(episode, max_len=0))
        trimmed = list(episode_samples(episode, max_len=1))
        self.assertEqual(len(trimmed), len(full))
        for index, ((record, meta), (original, original_meta)) in enumerate(zip(trimmed, full)):
            self.assertEqual(record["messages"][-1], original["messages"][-1])
            self.assertEqual(meta["call_id"], original_meta["call_id"])
            self.assertEqual(record["images"], original["images"][-8:])
            self.assertEqual(meta["dropped_history_pairs"], max(0, index - 2))
            self.assertEqual(meta["approx_tokens_before"], original_meta["approx_tokens"])
            validate_record(record)
        self.assertEqual(episode, before)
        self.assertEqual(trimmed, list(episode_samples(episode, max_len=1)))
        self.assertEqual(full, list(episode_samples(episode, max_len=0)))
        # Alter only a future result and post-dispatch text: the target prefix is unchanged.
        changed = deepcopy(episode)
        changed.decisions[3].content[-1]["image"] = "future.png"
        changed.decisions[3].interleaved_text = []
        self.assertEqual(trimmed[:4], list(episode_samples(changed, max_len=1))[:4])


    def test_cli_defaults_override_disable_and_negative_limit(self):
        source = self.long_fixture().flush()
        base = ["--run-dir", str(source.run_dir), "--controller-dir", str(source.controller_dir)]
        for extra, limit in (([], 24576), (["--max-len", "1"], 1), (["--max-len", "0"], 0)):
            with self.subTest(extra=extra), io.StringIO() as stream, redirect_stdout(stream):
                self.assertEqual(main(base + extra), 0)
                payload = json.loads(stream.getvalue())
                self.assertTrue(payload["dry_run"])
                self.assertEqual(payload["stats"]["length_control"]["max_len"], limit)
                self.assertEqual(payload["stats"]["samples"], 6)
        with io.StringIO() as stream, redirect_stdout(stream):
            self.assertEqual(main(base + ["--max-len", "-1"]), 2)
            self.assertEqual(json.loads(stream.getvalue())["error"]["code"], "invalid_max_len")
        for batch in (False, True):
            with self.subTest(batch=batch), self.assertRaisesRegex(ConversionError, "nonnegative integer"):
                prepare_dataset([source], batch=batch, max_len=-1)

    def test_images_seen_only_in_dropped_failed_history_are_not_copied(self):
        fixture = self.fixture()
        fixture.initialize()
        for _ in range(4):
            fixture.add("act", fixture.args(), {"ok": False, "error": {"code": "ACTION_FAILED"},
                                                "observation": fixture.capture()})
        fixture.finish()
        data = prepare_dataset([fixture.flush()], max_len=1)
        self.assertEqual(len(data.episodes[0].assets), 20)
        records = list(data.records())
        self.assertEqual(len(records), 2)
        self.assertEqual(len(records[-1][0]["images"]), 8)
        package = write_package(data, self.root / "package")
        copied = {str(path.relative_to(package)) for path in (package / "images").iterdir()}
        used = {path for record, _ in records for path in record["images"]}
        self.assertEqual(copied, used)
        self.assertEqual(len(copied), 8)
        self.assertEqual(len((package / "images.jsonl").read_text().splitlines()), 8)
