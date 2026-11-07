"""Synthetic CPU lifecycle checks, not real training evidence."""
from copy import deepcopy
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

from sft_fixtures import SftFixtureTestCase, training_package
from manipulation_agent.sft.common import ConversionError, file_hash
from manipulation_agent.training.artifacts import (ProcessFailure, create_output, encoder_source,
                                                    fingerprint, model_manifest, versions, write_json)
from manipulation_agent.training.cli import main
from manipulation_agent.training.config import encoding_contract, load_config
from manipulation_agent.training.data import index_package
from manipulation_agent.training.preflight import bounded_results, load_encoding_cache, preflight, seal_encoding_cache
from manipulation_agent.training.run import checkpoint_manifest, gpu_resource_probe, resume, run_process, verify_preflight, validate_training_output
from manipulation_agent.training.export import resolve_checkpoint


class BackendTests(SftFixtureTestCase):
    def model(self):
        p = self.root / "unit model"
        p.mkdir()
        write_json(p / "config.json", {"fixture": True})
        write_json(p / "tokenizer_config.json", {"fixture": True})
        return p

    def test_dry_run_needs_no_framework_and_writes_no_output(self):
        package, model = training_package(self.root), self.model()
        output = self.root / "out space"
        with io.StringIO() as stream, redirect_stdout(stream):
            code = main(["train", "--data", str(package), "--model", str(model), "--output", str(output), "--dry-run", "--max-length", "17408"])
            result = json.loads(stream.getvalue())
        self.assertEqual(code, 0)
        self.assertFalse(output.exists())
        self.assertFalse(result["selection_verified"])
        self.assertEqual(result["resolved_config"]["data"]["max_length"], 17408)
        self.assertEqual(result["argv"][result["argv"].index("--max_length") + 1], "17408")
        self.assertEqual(result["argv"][result["argv"].index("--model") + 1], str(model))

    def test_structure_report_is_not_encoding_evidence(self):
        package = training_package(self.root)
        out = self.root / "preflight"
        result = preflight(package, load_config(), out, structure_only=True)
        self.assertTrue(result["scan_complete"])
        self.assertFalse(result["encoding_passed"])
        with self.assertRaisesRegex(ConversionError, "real encoding"):
            verify_preflight(index_package(package), load_config(), model_manifest(self.model()), out)

    def test_changed_settings_and_tampered_reports_reject_reuse(self):
        package, model = training_package(self.root), self.model()
        index, config = index_package(package), load_config()
        root = self.root / "report"
        root.mkdir()
        contract = {"package": index.fingerprint, "model": model_manifest(model)["fingerprint"],
                    "encoding": encoding_contract(config), "packages": versions(), "encoder_source": encoder_source()}
        write_json(root / "encoding_contract.json", {"contract": contract, "fingerprint": fingerprint(contract)})
        write_json(root / "package_manifest.json", index.manifest())
        rows = [{"sample_id": s.sample_id, "status": "passed", "complete_target_verified": True} for s in index.samples]
        (root / "samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in rows))
        (root / "errors.jsonl").touch()
        write_json(root / "summary.json", {"status": "complete", "scan_complete": True, "encoding_passed": True,
                   "structure_only": False, "validation_level": "qwen_processor_encoding", "report_files": {n: file_hash(root / n) for n in
                   ("samples.jsonl", "errors.jsonl", "package_manifest.json", "encoding_contract.json")}})
        self.assertEqual(len(verify_preflight(index, config, model_manifest(model), root)), 6)
        changed = deepcopy(config)
        changed["vision"]["image_max_token_num"] = 512
        with self.assertRaisesRegex(ConversionError, "differs from preflight"):
            verify_preflight(index, changed, model_manifest(model), root)
        (root / "samples.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ConversionError, "reports changed"):
            verify_preflight(index, config, model_manifest(model), root)

    def test_nonzero_child_code_and_logs_preserved(self):
        root = self.root / "failed_process"
        root.mkdir()
        with self.assertRaises(ProcessFailure) as error:
            run_process([sys.executable, "-c", "print('failure evidence'); raise SystemExit(7)"], root, {})
        self.assertEqual(error.exception.location["child_return_code"], 7)
        self.assertIn("failure evidence", (root / "logs/training.stdout.log").read_text())
        self.assertIn('"child_return_code":7', (root / "journal.jsonl").read_text())

    def test_gpu_probe_uses_current_interpreter_and_preserves_failure(self):
        from types import SimpleNamespace
        with patch("manipulation_agent.training.run.subprocess.run") as call:
            call.return_value = SimpleNamespace(returncode=0, stdout=json.dumps({"status": "passed", "environment": {"pid": 12}}))
            self.assertEqual(gpu_resource_probe(load_config()), {"pid": 12})
            self.assertEqual(call.call_args.args[0][0], sys.executable)
            self.assertEqual(json.loads(call.call_args.kwargs["input"])["model"]["model_type"], "qwen3_5")
            call.return_value = SimpleNamespace(returncode=2, stdout=json.dumps({"status": "failed", "error": {"code": "gpu_unavailable", "message": "No GPU"}}))
            with self.assertRaisesRegex(ConversionError, "No GPU") as error:
                gpu_resource_probe(load_config())
            self.assertEqual(error.exception.code, "gpu_unavailable")

    def test_interrupt_targets_only_the_owned_process_group(self):
        root = self.root / "interrupted"
        root.mkdir()
        with patch("manipulation_agent.training.run.subprocess.Popen") as popen, patch("os.killpg") as killpg:
            process = popen.return_value
            process.pid = 123456
            process.wait.side_effect = [KeyboardInterrupt, -15]
            process.returncode = -15
            with self.assertRaises(KeyboardInterrupt):
                run_process([sys.executable, "-c", "pass"], root, {})
            self.assertEqual(killpg.call_count, 1)
            self.assertEqual(killpg.call_args.args[0], 123456)
        self.assertIn('"child_return_code":-15', (root / "journal.jsonl").read_text())

    def test_parallel_preflight_bounds_pending_rows_and_keeps_order(self):
        class Executor:
            pending = maximum = 0

            def submit(self, function, row, normalized):
                self.pending += 1
                self.maximum = max(self.maximum, self.pending)
                executor = self

                class Future:
                    def result(self):
                        executor.pending -= 1
                        return {"result": row}

                return Future()

        executor = Executor()
        rows = list(bounded_results(executor, ((i, {"i": i}, {}) for i in range(30)), limit=4))
        self.assertEqual(rows, [(i, {"result": {"i": i}}) for i in range(30)])
        self.assertEqual(executor.maximum, 4)
        self.assertEqual(executor.pending, 0)

    def test_structure_only_rejects_parallel_encoding_without_output(self):
        output = self.root / "invalid"
        with self.assertRaisesRegex(ConversionError, "structure-only"):
            preflight("unused", load_config(), output, structure_only=True, encoding_workers=2)
        self.assertFalse(output.exists())

    def test_sealed_partial_cache_binds_source_contract_and_result_bytes(self):
        index = index_package(training_package(self.root))
        cache = self.root / "partial_cache"
        cache.mkdir()
        contract = {"fixture": True, "vision": {"max_tokens": 1024}}
        summary = {"status": "interrupted", "structure_only": False, "scan_complete": False,
                   "validation_level": "qwen_processor_encoding"}
        write_json(cache / "summary.json", summary)
        write_json(cache / "encoding_contract.json", {"contract": contract, "fingerprint": fingerprint(contract)})
        rows = [{**asdict(s), "status": "passed", "tokens": 3, "loss_tokens": 1, "masked_tokens": 2,
                 "images": s.image_count, "image_tokens": s.image_count,
                 "input_ids_sha256": "a" * 64, "labels_sha256": "b" * 64,
                 "all_history_masked": True, "complete_target_verified": True, "weights_loaded": False}
                for s in index.samples[:2]]
        rows.append({**asdict(index.samples[2]), "status": "failed", "error": {"code": "synthetic"}})
        payload = "".join(json.dumps(row) + "\n" for row in rows)
        (cache / "samples.jsonl").write_text(payload)
        seal_encoding_cache(cache)
        self.assertEqual(set(load_encoding_cache(cache, contract, index)), {s.sample_id for s in index.samples[:2]})
        write_json(cache / "summary.json", {**summary, "validation_level": "synthetic_cpu_fixture"})
        with self.assertRaisesRegex(ConversionError, "real-processor scan"):
            load_encoding_cache(cache, contract, index)
        write_json(cache / "summary.json", summary)
        with self.assertRaisesRegex(ConversionError, "Cache data/model/template"):
            load_encoding_cache(cache, {"fixture": "changed"}, index)
        (cache / "samples.jsonl").write_text(payload + "{}\n")
        with self.assertRaisesRegex(ConversionError, "result bytes changed"):
            load_encoding_cache(cache, contract, index)

    def test_adapter_without_optimizer_is_not_resume(self):
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        for name in ("adapter_config.json", "adapter_model.safetensors", "trainer_state.json"):
            (checkpoint / name).write_text('{"global_step":2}')
        with self.assertRaisesRegex(ConversionError, "optimizer"):
            resume(checkpoint, self.root / "next")

    def test_zero_exit_without_training_artifacts_is_failure(self):
        root = self.root / "empty_output"
        root.mkdir()
        with self.assertRaises(ProcessFailure) as error:
            validate_training_output(root, start_step=0, target_step=2)
        self.assertEqual(error.exception.code, "training_artifact_validation_failed")

    def test_best_is_not_guessed_from_largest_directory(self):
        run = self.root / "run"
        run.mkdir()
        write_json(run / "run.json", {"status": "completed", "validation_level": "real_gpu_lora_training"})
        write_json(run / "audit.json", {"best_model_checkpoint": None})
        (run / "checkpoint-999").mkdir()
        with self.assertRaisesRegex(ConversionError, "best checkpoint"):
            resolve_checkpoint(run, "best")

    def test_best_uses_trainer_state_and_rejects_other_base(self):
        run = self.root / "managed_fixture"
        run.mkdir()
        contract = {"model": {"fingerprint": "base-a"}, "package": {"fingerprint": "data-a"},
                    "sample_ids": ["synthetic"]}
        write_json(run / "training_contract.json", contract)
        for step in (2, 4):
            checkpoint = run / "swift" / ("checkpoint-" + str(step))
            checkpoint.mkdir(parents=True)
            for name in ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
                (checkpoint / name).write_bytes(b"synthetic CPU fixture")
            write_json(checkpoint / "adapter_config.json", {})
            write_json(checkpoint / "trainer_state.json", {"global_step": step,
                       "best_model_checkpoint": str(run / "swift/checkpoint-2")})
            write_json(checkpoint / "manipulation_training.json", {
                "run_root": str(run), "contract_fingerprint": fingerprint(contract)})
        write_json(run / "checkpoints.json", {"checkpoints": [checkpoint_manifest(run / "swift" / ("checkpoint-" + str(s))) for s in (2, 4)]})
        write_json(run / "run.json", {"status": "completed", "validation_level": "real_gpu_lora_training",
                   "checkpoint": {"path": str(run / "swift/checkpoint-4")}})
        selected, _ = resolve_checkpoint(run, "best")
        self.assertEqual(selected["global_step"], 2)
        nested = run / "swift/checkpoint-2/metadata/state.json"
        nested.parent.mkdir()
        nested.write_text('{"evidence":"original"}')
        before = checkpoint_manifest(nested.parent.parent)
        nested.write_text('{"evidence":"changed"}')
        after = checkpoint_manifest(nested.parent.parent)
        self.assertIn("metadata/state.json", before["files"])
        self.assertNotEqual(before["fingerprint"], after["fingerprint"])
        with self.assertRaisesRegex(ConversionError, "save-time hashes"):
            resolve_checkpoint(run, "best")
        nested.unlink()
        nested.parent.rmdir()
        other = self.root / "other_fixture"
        other.mkdir()
        write_json(other / "run.json", {"status": "completed", "validation_level": "real_gpu_lora_training"})
        changed = deepcopy(contract)
        changed["model"]["fingerprint"] = "base-b"
        write_json(other / "training_contract.json", changed)
        with self.assertRaisesRegex(ConversionError, "not bound"):
            resolve_checkpoint(other, run / "swift/checkpoint-2")

    def test_overlapping_outputs_rejected(self):
        package = training_package(self.root)
        for output in (package, package / "output", package.parent):
            with self.assertRaises(ConversionError):
                create_output(output, protected=[package])
