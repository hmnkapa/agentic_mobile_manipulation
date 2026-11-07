"""Configuration/dry-run tests require no ML framework."""
from copy import deepcopy
import io
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from manipulation_agent.training.artifacts import TrainingError
from manipulation_agent.training.cli import main
from manipulation_agent.training.config import DEFAULTS, load_config, validate
from manipulation_agent.training.swift_backend import training_command, vision_environment


class ConfigTests(unittest.TestCase):
    def test_smoke_uses_four_step_schedule_and_pause_at_two(self):
        c = load_config(smoke=True)
        self.assertEqual(c["training"]["max_steps"], 4)
        self.assertEqual(c["smoke"]["stop_after_step"], 2)
        self.assertFalse(c["training"]["dataset_shuffle"])
        self.assertTrue(c["template"]["disable_ignore_empty_think"])

    def test_unknown_types_sections_and_launch_semantic_override_rejected(self):
        for text, launch in (("[training]\nunknown=1", False), ("[training]\nstrict=1", False),
                             ("[alien]\nx=1", False), ("[template]\nenable_thinking=false", True)):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "config.toml"
                p.write_text(text)
                with self.assertRaises(TrainingError):
                    load_config(launch=p) if launch else load_config(p)

    def test_invalid_values_and_supervision_changes_rejected(self):
        for section, key, value in (("data", "max_length", 0), ("training", "strict", False),
                                    ("training", "packing", True), ("template", "disable_ignore_empty_think", False),
                                    ("launch", "nproc_per_node", 3)):
            c = deepcopy(DEFAULTS)
            c[section][key] = value
            with self.subTest(key=key), self.assertRaises(TrainingError):
                validate(c)

    def test_native_command_preserves_paths_and_does_not_pass_project_policy(self):
        argv = training_command(load_config(smoke=True), "/models/a b", "/prepared/a b", "/out/a b")
        self.assertEqual(argv[argv.index("--model") + 1], "/models/a b")
        self.assertNotIn("--overlength_policy", argv)
        self.assertEqual(argv[argv.index("--disable_ignore_empty_think") + 1], "true")
        self.assertEqual(argv[:3], [sys.executable, "-m", "manipulation_agent.training.worker"])
        self.assertEqual(argv[argv.index("--report_to") + 1], "wandb")

    def test_legacy_vision_environment_conflict_rejected(self):
        with self.assertRaises(TrainingError):
            vision_environment(load_config(), {"MAX_PIXELS": "1000"})

    def test_single_process_does_not_request_a_distributed_backend(self):
        from manipulation_agent.training.run import launch_command
        config = load_config(smoke=True)
        before = deepcopy(config)
        argv = launch_command(config, "/model", "/prepared", "/output")
        self.assertNotIn("--ddp_backend", argv)
        self.assertEqual(config, before)
        config = load_config()
        config["launch"]["nproc_per_node"] = 2
        self.assertIn("--ddp_backend", launch_command(config, "/model", "/prepared", "/output"))

    def test_core_import_and_help_do_not_load_training_framework(self):
        code = ("import sys; from manipulation_agent.training.cli import main; "
                "from manipulation_agent.training import data, config, run, export, swift_plugin, tracking, worker; "
                "assert not any(x.split('.')[0] in {'torch','swift','transformers','omnigibson','habitat','wandb'} for x in sys.modules)")
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_errors_are_structured_and_resume_overrides_rejected(self):
        with io.StringIO() as stream, redirect_stdout(stream):
            code = main(["train", "--resume-from", "/missing", "--model", "/other", "--output", "/unused"])
            self.assertEqual(code, 2)
            self.assertIn("resume_override_forbidden", stream.getvalue())
        with io.StringIO() as stream, redirect_stdout(stream):
            code = main(["train", "--resume-from", "/missing", "--max-length", "17408", "--output", "/unused"])
            self.assertEqual(code, 2)
            self.assertIn("resume_override_forbidden", stream.getvalue())
        with io.StringIO() as stream, redirect_stdout(stream):
            code = main(["preflight", "--data", "/missing", "--max-length", "0", "--output", "/unused"])
            self.assertEqual(code, 2)
            self.assertIn("invalid_config_value", stream.getvalue())
