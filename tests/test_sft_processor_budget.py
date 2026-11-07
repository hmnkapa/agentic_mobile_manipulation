"""Synthetic length callbacks test trimming semantics, not Qwen validation."""
from copy import deepcopy
import unittest

from manipulation_agent.sft.common import ConversionError
from manipulation_agent.sft.format import to_msswift, validate_record
from manipulation_agent.sft.history import compress_history
from manipulation_agent.sft.processor_budget import pack_record, trim_record, unpack_record
from test_sft_budget import call, group


class ProcessorBudgetTests(unittest.TestCase):
    def setUp(self):
        self.prefix = [{"role": "system", "content": "SYSTEM\n\n"}, {"role": "user", "content": "USER"}]
        self.groups = [group(str(i), progress=f"PUBLIC_PROGRESS_{i}") for i in range(4)]
        self.target = call("target", target=True)
        self.tools = [{"type": "function", "function": {"name": "act", "parameters": {"type": "object"}}}]
        self.raw = self.prefix + [m for g in self.groups for m in g]
        self.record = to_msswift(compress_history(self.raw) + [self.target], self.tools)

    @staticmethod
    def measure(row):
        # Deliberately much higher image cost than the old rough estimator.
        return sum(len(m["content"]) for m in row["messages"]) + 3000 * len(row["images"])

    def test_unchanged_rows_round_trip_whitespace_and_only_one_think(self):
        original = deepcopy(self.record)
        self.assertEqual(pack_record(unpack_record(original), original["tools"]), original)
        output, stats = trim_record(original, self.measure, max_length=self.measure(original))
        self.assertEqual(output, original)
        self.assertFalse(stats["processor_trim_triggered"])
        self.assertEqual(stats["processor_length_measurements"], 1)
        self.assertEqual(output["messages"][-1]["content"].count("<think>"), 1)

    def test_older_images_slim_before_groups_and_keep_first_delivered_view(self):
        output, stats = trim_record(self.record, self.measure, max_length=self.measure(self.record) - 1)
        self.assertEqual(stats["processor_slimmed_images"], 6)
        self.assertEqual(stats["processor_dropped_history_pairs"], 0)
        self.assertEqual(output["images"][:2], ["0-left.png", "1-left.png"])
        self.assertIn("RGB view=left, image_ref=0-left", str(output))
        self.assertNotIn("RGB view=front, image_ref=0-front", str(output))
        self.assertIn("PUBLIC_PROGRESS_0", str(output))
        self.assertLessEqual(self.measure(output), self.measure(self.record) - 1)

    def test_drop_complete_old_groups_and_preserve_recent_window_target_and_loss(self):
        protected = to_msswift(self.prefix + self.groups[2] + self.groups[3] + [self.target], self.tools)
        before = deepcopy(self.record)
        output, stats = trim_record(self.record, self.measure, max_length=self.measure(protected))
        self.assertEqual(output, protected)
        self.assertEqual(stats["processor_dropped_history_pairs"], 2)
        self.assertEqual(stats["processor_images_removed"], 8)
        self.assertNotIn("PUBLIC_PROGRESS_0", str(output))
        self.assertNotIn("PUBLIC_PROGRESS_1", str(output))
        self.assertTrue(validate_record(output))
        self.assertEqual(self.record, before)

    def test_protected_context_over_budget_stays_present(self):
        protected = to_msswift(self.prefix + self.groups[2] + self.groups[3] + [self.target], self.tools)
        output, stats = trim_record(self.record, self.measure, max_length=1)
        self.assertEqual(output, protected)
        self.assertTrue(stats["processor_over_budget"])
        self.assertEqual(stats["processor_over_budget_reason"], "protected_context_exceeds_budget")
        self.assertEqual(output["messages"][-1], self.record["messages"][-1])

    def test_fewer_than_k_rgb_replies_protect_all_history_including_failures(self):
        history = self.prefix + group("failure", rgb=False, result={"ok": False, "error": "real error"}) + self.groups[-1]
        record = to_msswift(compress_history(history) + [self.target], self.tools)
        output, stats = trim_record(record, self.measure, max_length=1)
        self.assertEqual(output, record)
        self.assertTrue(stats["processor_over_budget"])
        self.assertEqual(stats["processor_dropped_history_pairs"], 0)

    def test_mismatched_or_overlapping_historical_calls_are_rejected(self):
        for overlapping in (False, True):
            record = deepcopy(self.record)
            if overlapping:
                record["messages"].insert(-3, call("overlap"))
            else:
                next(m for m in record["messages"] if m.get("tool_call_id"))["tool_call_id"] = "unmatched"
            with self.subTest(overlapping=overlapping), self.assertRaises(ConversionError):
                trim_record(record, self.measure, max_length=1)


if __name__ == "__main__":
    unittest.main()
