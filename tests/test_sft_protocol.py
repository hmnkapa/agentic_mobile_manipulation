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

class ProtocolTests(unittest.TestCase):
    def test_point_scale_roundtrip_preserves_other_fields_and_inputs(self):
        for point in ([0, 1], [0.1234, 0.9876], [0.5, 0.5], [0.0005, 0.9995]):
            args = {"primitive": "grasp", "revision": 7, "target": {"image_ref": "left", "point": point}, "wait_seconds": None}
            original = deepcopy(args)
            student = to_student_arguments("act", args)
            native = from_student_arguments("act", student)
            self.assertEqual(args, original)
            self.assertNotIn("point", student["target"])
            self.assertEqual(student["revision"], 7)
            for a, b in zip(point, native["target"]["point"]):
                self.assertLessEqual(abs(a - b), 0.00050000001)
        self.assertEqual(to_student_arguments("act", {"target": None}), {"target": None})

    def test_invalid_points_never_become_valid_targets(self):
        for point in ([True, 0], [-0.1, 0.2], [0, 2], [float("nan"), 0], [0]):
            with self.assertRaises(ConversionError):
                to_student_arguments("act", {"target": {"point": point}})
        failed = to_student_arguments("act", {"target": {"point": [1.5, -0.2]}}, strict=False)
        self.assertEqual(failed["target"]["point_2d"], [1500, -200])
        with self.assertRaises(ConversionError):
            from_student_arguments("act", {"target": {"point_2d": [0.5, 20]}})

    def test_context_and_explicit_skill_examples_use_student_units(self):
        system, tools = build_student_context(system_prompt("skills"), tool_specs("skills"))
        act = next(t["function"] for t in tools if t["function"]["name"] == "act")
        target = act["parameters"]["properties"]["target"]
        self.assertNotIn("point", target["properties"])
        self.assertIn("point_2d", target["required"])
        self.assertEqual(target["properties"]["point_2d"]["items"]["maximum"], 1000)
        self.assertNotIn("with normalized x", system)
        self.assertNotIn("right=1,", act["description"])
        self.assertEqual(rewrite_interface_text("target={image_ref, point:[0.5,0.8]}"),
                         "target={image_ref, point_2d:[500, 800]}")

