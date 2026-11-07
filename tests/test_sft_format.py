"""CPU format boundary regressions."""
from copy import deepcopy
import unittest

from manipulation_agent.sft.common import ConversionError
from manipulation_agent.sft.format import estimate_length, to_msswift, validate_record
from manipulation_agent.sft import history


class FormatTests(unittest.TestCase):
    def row(self):
        return {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                             {"role": "assistant", "content": "<think>\n\n</think>", "tool_calls": [{"function": {"name": "initialize", "arguments": "{}"}}]}],
                "images": [], "tools": []}

    def test_old_imports_refer_to_same_functions(self):
        self.assertIs(history.validate_record, validate_record)
        self.assertIs(history.to_msswift, to_msswift)

    def test_history_loss_or_image_marker_corruption_is_rejected(self):
        for change in ("loss", "images"):
            row = self.row()
            if change == "loss":
                row["messages"].insert(2, {"role": "assistant", "content": "old"})
            else:
                row["images"] = ["missing.png"]
            with self.assertRaises(ConversionError):
                validate_record(row)

    def test_image_order_and_input_immutability(self):
        messages = [{"role": "tool", "content": [{"type": "text", "text": "left"},
                     {"type": "image", "image": "left.png"}, {"type": "image", "image": "front.png"}]}]
        before = deepcopy(messages)
        result = to_msswift(messages, [])
        self.assertEqual(result["images"], ["left.png", "front.png"])
        self.assertEqual(messages, before)
        self.assertIn("not processor tokens", estimate_length(result)["method"])
