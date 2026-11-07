"""CPU launch regressions; no processor, model training, or external run is started."""
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from sft_fixtures import ROOT, SftFixtureTestCase, training_package
from manipulation_agent.sft.common import read_json
from manipulation_agent.training.artifacts import ProcessFailure, TrainingError, fingerprint, write_json
from manipulation_agent.training.cli import main
from manipulation_agent.training.config import load_config
from manipulation_agent.training.data import index_package
from manipulation_agent.training.run import (
    checkpoint_manifest, resume, select_smoke_without_preflight, train,
)


class TrainingStartupTests(SftFixtureTestCase):
    def inputs(self):
        package = training_package(self.root)
        model = self.root / "model"
        model.mkdir()
        write_json(model / "config.json", {"synthetic": True})
        write_json(model / "tokenizer_config.json", {"synthetic": True})
        (model / "model.safetensors").write_bytes(b"Synthetic fixture, never loaded")
        config = load_config()
        config["wandb"]["mode"] = "disabled"
        return package, config, model

    def stopped_child(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch("manipulation_agent.training.run.require_tracking"))
        stack.enter_context(patch("manipulation_agent.training.run.tracking_plan", return_value={"enabled": False}))
        stack.enter_context(patch("manipulation_agent.training.run.gpu_resource_probe", return_value={"pid": 123, "selected_gpus": []}))
        stack.enter_context(patch("manipulation_agent.training.run.environment", return_value={"validation": "synthetic_cpu_fixture"}))
        stack.enter_context(patch("manipulation_agent.training.swift_backend.build_encoding_backend",
                                  side_effect=AssertionError("Startup must not construct a processor")))
        stack.enter_context(patch("manipulation_agent.training.encoding.check_encoding",
                                  side_effect=AssertionError("Startup must not encode samples")))
        child = stack.enter_context(patch("manipulation_agent.training.run.run_process", side_effect=ProcessFailure(
            "synthetic_child_stop", "Stop before any real training", child_return_code=7)))
        return child

    def test_cli_accepts_training_without_a_preflight_report(self):
        with patch("manipulation_agent.training.run.train", return_value={"status": "synthetic"}) as launch:
            with io.StringIO() as stream, redirect_stdout(stream):
                self.assertEqual(main(["train", "--data", "/data", "--model", "/model", "--output", "/output"]), 0)
            self.assertIsNone(launch.call_args.args[3])

    def test_full_training_reaches_child_without_cpu_encoding_or_report(self):
        package, config, model = self.inputs()
        child = self.stopped_child()
        output = self.root / "training"
        with patch("manipulation_agent.training.run.verify_preflight", side_effect=AssertionError("No report supplied")):
            with self.assertRaises(ProcessFailure):
                train(package, config, model, None, output)
        child.assert_called_once()
        contract = read_json(output / "training_contract.json")
        self.assertIsNone(contract["preflight"])
        self.assertEqual(contract["startup_validation"], {"encoding_preflight": "not_run", "loader_encoding": "not_run"})
        self.assertEqual(contract["expected_splits"], {"train": 3, "val": 3})
        self.assertEqual(read_json(output / "loader_validation.json")["status"], "not_run")
        self.assertFalse(read_json(output / "selection.json")["lengths_verified"])
        self.assertEqual(read_json(output / "run.json")["child_return_code"], 7)
        for split in ("train", "val"):
            self.assertEqual(len((output / "prepared" / (split + ".jsonl")).read_text().splitlines()), 3)

    def test_explicit_preflight_keeps_report_and_length_validation(self):
        package, config, model = self.inputs()
        child = self.stopped_child()
        output = self.root / "training"
        report = self.root / "explicit_report"
        with patch("manipulation_agent.training.run.verify_preflight", return_value=[{
            "tokens": 100, "model_max_length": 1000}]) as verify:
            with self.assertRaises(ProcessFailure):
                train(package, config, model, report, output)
            verify.assert_called_once()
        self.assertTrue(read_json(output / "selection.json")["lengths_verified"])
        self.assertEqual(read_json(output / "training_contract.json")["preflight"], str(report))
        self.assertEqual(read_json(output / "run.json")["startup_validation"]["encoding_preflight"], "verified_report")
        child.reset_mock()
        with patch("manipulation_agent.training.run.verify_preflight", return_value=[{
            "tokens": config["data"]["max_length"] + 1, "model_max_length": 999999}]):
            with self.assertRaisesRegex(TrainingError, "overlength"):
                train(package, config, model, report, self.root / "overlength")
        child.assert_not_called()

    def test_resume_without_preflight_reaches_child_and_keeps_contract(self):
        package, config, model = self.inputs()
        child = self.stopped_child()
        parent = self.root / "parent"
        with self.assertRaises(ProcessFailure):
            train(package, config, model, None, parent)
        contract = read_json(parent / "training_contract.json")
        checkpoint = parent / "swift/checkpoint-1"
        checkpoint.mkdir(parents=True)
        for name in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
            (checkpoint / name).write_bytes(b"Synthetic checkpoint, never loaded")
        write_json(checkpoint / "adapter_config.json", {"base_model_name_or_path": str(model)})
        write_json(checkpoint / "trainer_state.json", {"global_step": 1})
        write_json(checkpoint / "manipulation_training.json", {
            "run_root": str(parent), "contract_fingerprint": fingerprint(contract)})
        write_json(parent / "checkpoints.json", {"checkpoints": [checkpoint_manifest(checkpoint)]})
        output = self.root / "resumed"
        with patch("manipulation_agent.training.run.verify_preflight", side_effect=AssertionError("No report saved")):
            with self.assertRaises(ProcessFailure):
                resume(checkpoint, output)
        self.assertEqual(child.call_count, 2)
        resumed = read_json(output / "training_contract.json")
        self.assertIsNone(resumed["preflight"])
        self.assertEqual(resumed["sample_ids"], contract["sample_ids"])
        self.assertEqual(resumed["expected_start_step"], 1)
        command = child.call_args.args[0]
        self.assertEqual(command[command.index("--resume_from_checkpoint") + 1], str(checkpoint))

    def test_smoke_without_report_preserves_deterministic_structural_coverage(self):
        index = index_package(training_package(self.root, rich=True))
        config = load_config(smoke=True)
        selected, selection = select_smoke_without_preflight(index, config)
        self.assertEqual((selected, selection), select_smoke_without_preflight(index, config))
        self.assertEqual(len({s.sample_id for s in selected}), 24)
        self.assertEqual(sum(s.split == "train" for s in selected), 16)
        self.assertEqual(sum(s.split == "val" for s in selected), 8)
        self.assertEqual({s.target_tool for s in selected if s.split == "train"},
                         {"initialize", "list_skills", "read_skill", "task_notes", "act", "look", "finish"})
        self.assertTrue(any(s.has_failed_history for s in selected if s.split == "train"))
        self.assertTrue(any(s.image_count == 0 for s in selected))
        self.assertTrue(any(s.image_count >= 8 for s in selected if s.split == "val" and s.target_tool == "act"))
        self.assertFalse(selection["lengths_verified"])
        self.assertTrue(all("tokens" not in s for s in selection["samples"]))
        index.samples = [s for s in index.samples if s.target_tool != "finish"]
        with self.assertRaisesRegex(TrainingError, "coverage"):
            select_smoke_without_preflight(index, config)


class LauncherTests(SftFixtureTestCase):
    def invoke(self, *args, child_code=0):
        model, data = self.root / "model", self.root / "data"
        model.mkdir(exist_ok=True)
        data.mkdir(exist_ok=True)
        log = self.root / "calls.jsonl"
        log.write_text("")
        python = self.root / "record python"
        python.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
with open(os.environ["SFT_LAUNCH_TEST_LOG"], "a") as stream:
    stream.write(json.dumps({"argv": sys.argv[1:], "cuda": os.environ.get("CUDA_VISIBLE_DEVICES")}) + "\\n")
raise SystemExit(int(os.environ["SFT_LAUNCH_TEST_EXIT"]))
''')
        python.chmod(0o755)
        env = {**os.environ, "SFT_LAUNCH_TEST_LOG": str(log), "SFT_LAUNCH_TEST_EXIT": str(child_code)}
        result = subprocess.run(["bash", str(ROOT / "scripts/train_qwen35_4b.sh"),
            "--python", str(python), "--data", str(data), "--model", str(model), "--gpu", "0",
            "--work-root", str(self.root / "work"), "--output", str(self.root / "output"), *args],
            env=env, cwd=self.root, text=True, capture_output=True, timeout=10)
        return result, [json.loads(line) for line in log.read_text().splitlines()]

    def test_normal_smoke_and_dry_run_never_schedule_a_preflight(self):
        for args in ((), ("--smoke",), ("--dry-run",)):
            with self.subTest(args=args):
                result, calls = self.invoke(*args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["argv"][3], "train")
                self.assertNotIn("--preflight", calls[0]["argv"])
                self.assertEqual(calls[0]["cuda"], "0")
                self.assertNotIn("CPU preflight", result.stdout)
                self.assertFalse((self.root / "work/preflights").exists())

    def test_preflight_only_remains_explicit_cpu_operation(self):
        result, calls = self.invoke("--preflight-only", "--encoding-workers", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["argv"][3], "preflight")
        self.assertEqual(calls[0]["cuda"], "")
        self.assertIn("--encoding-workers", calls[0]["argv"])
        result, calls = self.invoke("--preflight-only", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [])

    def test_existing_report_is_forwarded_without_encoding_and_failures_propagate(self):
        report = self.root / "old report"
        report.mkdir()
        write_json(report / "summary.json", {})
        result, calls = self.invoke("--preflight", str(report))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 1)
        argv = calls[0]["argv"]
        self.assertEqual(argv[argv.index("--preflight") + 1], str(report))
        self.assertNotIn("CPU preflight", result.stdout)
        result, calls = self.invoke(child_code=7)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(len(calls), 1)
