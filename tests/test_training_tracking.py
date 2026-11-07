"""CPU tracking contracts and failure isolation; no training or remote writes."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from manipulation_agent.training.artifacts import TrainingError, write_json
from manipulation_agent.training.cli import main
from manipulation_agent.training.config import DEFAULTS, load_config, validate
from manipulation_agent.training.swift_backend import training_command
from manipulation_agent.training.tracking import (enabled, finish_tracking, require_tracking,
                                                   start_tracking, tracking_environment, tracking_plan)
from manipulation_agent.training import worker


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = deepcopy(DEFAULTS)
        self.config["training"]["report_to"] = "wandb"

    def test_disabled_uses_native_cli_without_sdk_or_tracking_directory(self):
        self.config["wandb"]["mode"] = "disabled"
        self.assertFalse(enabled(self.config))
        with patch("manipulation_agent.training.tracking.find_spec", side_effect=AssertionError):
            require_tracking(self.config)
        argv = training_command(self.config, "/model", "/data", "/out")
        self.assertEqual(argv[:3], [sys.executable, "-m", "swift.cli.sft"])
        self.assertEqual(argv[argv.index("--report_to") + 1], "none")
        self.assertEqual(tracking_plan(self.config, self.root), {"enabled": False})
        self.assertFalse((self.root / "wandb").exists())

    def test_online_resume_keeps_identity_and_offline_segments_keep_group(self):
        for mode in ("online", "offline"):
            with self.subTest(mode=mode):
                self.config["wandb"]["mode"] = mode
                original = tracking_plan(self.config, self.root)
                write_json(self.root / "training_contract.json", {"wandb": original})
                resumed = tracking_plan(self.config, self.root / "resumed", parent=self.root)
                self.assertEqual(resumed["group"], original["group"])
                self.assertEqual(resumed["parent_id"], original["id"])
                if mode == "online":
                    self.assertEqual(resumed["id"], original["id"])
                    self.assertEqual(resumed["resume"], "must")
                else:
                    self.assertNotEqual(resumed["id"], original["id"])
                    self.assertIsNone(resumed["resume"])
                self.config["wandb"]["project"] = "other"
                with self.assertRaisesRegex(TrainingError, "identity"):
                    tracking_plan(self.config, self.root / "bad", parent=self.root)
                self.config["wandb"]["project"] = original["project"]

    def test_shell_identity_cannot_redirect_saved_tracking_contract(self):
        base = {"WANDB_API_KEY": "test-secret", "WANDB_MODE": "disabled", "WANDB_PROJECT": "wrong",
                "WANDB_RUN_ID": "wrong", "WANDB_RESUME": "auto", "WANDB_LOG_MODEL": "checkpoint",
                "WANDB_WATCH": "all", "WANDB_DISABLED": "true", "CUDA_VISIBLE_DEVICES": "0",
                "WANDB_BASE_URL": "https://api.wandb.ai"}
        env = tracking_environment(self.config, self.root, base)
        self.assertEqual(env["WANDB_PROJECT"], "agentic_distillation")
        self.assertEqual(env["WANDB_ENTITY"], "qili0502-zhejiang-university")
        self.assertEqual(env["WANDB_BASE_URL"], "https://api.forge.coreweave.com")
        self.assertEqual(env["WANDB_MODE"], "online")
        self.assertNotIn("WANDB_RUN_ID", env)
        self.assertNotIn("WANDB_DISABLED", env)
        self.assertEqual(env["WANDB_LOG_MODEL"], "false")
        self.assertEqual(env["WANDB_CONSOLE"], "off")
        self.assertEqual(env["WANDB_API_KEY"], "test-secret")
        self.assertNotIn("test-secret", json.dumps(tracking_plan(self.config, self.root)))
        self.assertEqual(base["WANDB_PROJECT"], "wrong")

    def test_ddp_uses_global_rank_and_only_zero_initializes_sdk(self):
        self.config["launch"]["nproc_per_node"] = 2
        argv = training_command(self.config, "/model", "/data", "/out")
        self.assertEqual(argv[argv.index("--module") + 1], "manipulation_agent.training.worker")
        write_json(self.root / "training_contract.json", {"wandb": {"enabled": True}})
        with patch.dict(os.environ, {"RANK": "1", "LOCAL_RANK": "0"}), patch(
                "manipulation_agent.training.tracking.require_tracking", side_effect=AssertionError):
            self.assertIsNone(start_tracking(self.root))

    def test_missing_dependency_and_invalid_config_fail_clearly(self):
        with patch("manipulation_agent.training.tracking.find_spec", return_value=None):
            with self.assertRaisesRegex(TrainingError, "training-wandb.in"):
                require_tracking(self.config)
        for field, value in (("mode", "auto"), ("project", "team/project"), ("tags", [42])):
            config = deepcopy(self.config)
            config["wandb"][field] = value
            with self.subTest(field=field), self.assertRaises(TrainingError):
                validate(config)

    def test_environment_selection_and_resume_overrides(self):
        with patch.dict(os.environ, {"WANDB_MODE": "offline", "WANDB_PROJECT": "selected"}):
            config = load_config()
        self.assertEqual(config["wandb"]["mode"], "offline")
        self.assertEqual(config["wandb"]["project"], "selected")
        with io.StringIO() as stream, redirect_stdout(stream):
            code = main(["train", "--resume-from", "/missing", "--output", "/unused", "--wandb-mode", "offline"])
            self.assertEqual(code, 2)
            self.assertIn("resume_override_forbidden", stream.getvalue())

    def test_worker_failure_and_interrupt_finish_owned_run_with_exit_code(self):
        previous = signal.getsignal(signal.SIGTERM)
        for error, expected in ((RuntimeError("synthetic failure"), 1), (KeyboardInterrupt(), 130)):
            with self.subTest(expected=expected), patch.dict(os.environ, {"MANIP_SFT_RUN_ROOT": str(self.root)}), patch.object(
                    worker, "start_tracking", return_value="owned") as start, patch.object(
                    worker.runpy, "run_module", side_effect=error), patch.object(worker, "finish_tracking") as finish:
                with self.assertRaises(type(error)):
                    worker.main()
                start.assert_called_once_with(self.root)
                finish.assert_called_once_with(self.root, "owned", exit_code=expected)
                self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_invalid_audit_still_finishes_sdk_as_failed(self):
        tracker = SimpleNamespace(summary={}, finish=Mock())
        write_json(self.root / "wandb.json", {"id": "fixture"})
        (self.root / "audit.json").write_text("not-json")
        with self.assertRaises(Exception):
            finish_tracking(self.root, tracker, exit_code=0)
        tracker.finish.assert_called_once_with(exit_code=1)

    def test_final_summary_keeps_task_success_separate_and_records_failure(self):
        tracker = SimpleNamespace(summary={}, finish=Mock())
        write_json(self.root / "wandb.json", {"id": "fixture"})
        write_json(self.root / "audit.json", {"status": "running"})
        finish_tracking(self.root, tracker, exit_code=1)
        tracker.finish.assert_called_once_with(exit_code=1)
        self.assertEqual(tracker.summary["lifecycle/status"], "failed")
        self.assertNotIn("task_success", tracker.summary)
        self.assertEqual(json.loads((self.root / "wandb.json").read_text())["status"], "failed")
