"""Synthetic offline rollout data-flow, provenance and failure tests."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch
from functools import wraps
from types import SimpleNamespace

from sft_fixtures import ROOT, SftFixtureTestCase, save, training_package
from test_rollout_metrics import xml_call
from manipulation_agent.sft.common import ConversionError, file_hash, jsonl, read_json
from manipulation_agent.training.artifacts import ProcessFailure, fingerprint, model_manifest
from manipulation_agent.training.cli import main
from manipulation_agent.training.config import load_config
from manipulation_agent.training.data import index_package
from manipulation_agent.training.rollout import _adapter_manifest, resolve_inputs, rollout
from manipulation_agent.training.rollout_metrics import target_call


class RolloutTests(SftFixtureTestCase):
    def setUp(self):
        super().setUp()
        self.package = training_package(self.root)
        self.recipe = ROOT / "configs/training/qwen35_4b_lora.toml"
        self.base = self.root / "base"
        self.base.mkdir()
        for name in ("config.json", "tokenizer_config.json"):
            save(self.base / name, {"synthetic_cpu_fixture": True})
        (self.base / "model.safetensors").write_bytes(b"Synthetic weights, never loaded")
        self.adapter = self.root / "checkpoint-40"
        self.adapter.mkdir()
        save(self.adapter / "adapter_config.json", {"peft_type": "LORA", "base_model_name_or_path": str(self.base)})
        (self.adapter / "adapter_model.safetensors").write_bytes(b"Synthetic adapter, never loaded")
        self.output = self.root / "evaluation"
        self.index = index_package(self.package)
        self.val = sorted((s for s in self.index.samples if s.split == "val"), key=lambda s: s.line)
        self.answers = [target_call(self.index.record(s)) for s in self.val]
        self.prompts, self.closed, self.recovered = [], [], []

    def managed_adapter(self):
        run = self.root / "training"
        run.mkdir()
        contract = {"config": load_config(), "model": model_manifest(self.base, require_weights=True),
                    "package": {"root": "/some/other/training/package"}}
        save(run / "training_contract.json", contract)
        save(self.adapter / "manipulation_training.json", {"run_root": str(run), "contract_fingerprint": fingerprint(contract)})
        save(run / "checkpoints.json", {"checkpoints": [_adapter_manifest(self.adapter)]})
        return run

    def fake_backend(self, responses=None):
        test = self
        values = iter(responses if responses is not None else [xml_call(**answer) for answer in self.answers])

        class SyntheticBackend:
            validation_level = "synthetic_cpu_fixture"

            def __init__(self, *args):
                self.arguments = args

            def describe(self):
                return {"synthetic_cpu_fixture": True}

            def generate(self, prompt):
                test.prompts.append(deepcopy(prompt))
                value = next(values)
                if callable(value):
                    value = value()
                if isinstance(value, BaseException):
                    raise value
                if isinstance(value, dict):
                    return value
                return {"raw_response": value, "generated_tokens": 10, "input_tokens": 100, "finish_reason": "eos"}

            def close(self):
                test.closed.append(True)

            def recover(self):
                test.recovered.append(True)

        return SyntheticBackend

    def evaluate(self, responses=None, **kwargs):
        with patch("manipulation_agent.training.rollout_backend.GenerationBackend", self.fake_backend(responses)), \
             patch("manipulation_agent.training.rollout.environment", return_value={"synthetic_cpu_fixture": True}), \
             redirect_stderr(io.StringIO()):
            return rollout(self.package, self.adapter, self.output, model=self.base, recipe=self.recipe, **kwargs)

    def test_default_val_answers_removed_history_and_image_order_preserved(self):
        result = self.evaluate()
        self.assertEqual(result["metrics"]["strict_accuracy"], {"correct": 3, "total": 3, "accuracy": 1})
        self.assertEqual(result["validation_level"], "synthetic_cpu_fixture")
        self.assertFalse(result["task_success_evaluated"])
        for sample, prompt in zip(self.val, self.prompts):
            original = self.index.record(sample, absolute_images=True)
            self.assertEqual(prompt["messages"], original["messages"][:-1])
            self.assertEqual(prompt["images"], original["images"])
            self.assertEqual(prompt["tools"], original["tools"])
            self.assertTrue(all(Path(image).is_absolute() for image in prompt["images"]))
        self.assertEqual(self.prompts[0]["images"], [])
        self.assertGreater(len(self.prompts[1]["images"]), 1)
        rows = [r for _, r in jsonl(self.output / "predictions.jsonl")]
        self.assertTrue(all(r["sample"]["split"] == "val" for r in rows))
        self.assertEqual(len(self.closed), 1)
        for name, digest in result["report_files"].items():
            self.assertEqual(file_hash(self.output / name), digest)

    def test_sample_failures_and_truncations_stay_in_denominator(self):
        truncated = {"raw_response": xml_call(**self.answers[1]), "generated_tokens": 2048,
                     "input_tokens": 100, "finish_reason": "length"}
        result = self.evaluate([ConversionError("input_overlength", "Synthetic long input", input_tokens=20000),
                                truncated, "invalid XML"])
        self.assertEqual(result["metrics"]["strict_accuracy"], {"correct": 0, "total": 3, "accuracy": 0})
        self.assertEqual(result["metrics"]["valid_call_rate"]["total"], 3)
        self.assertEqual(result["metrics"]["errors_by_code"],
                         {"input_overlength": 1, "generation_truncated": 1, "invalid_call_count": 1})
        records = [r for _, r in jsonl(self.output / "predictions.jsonl")]
        self.assertEqual(records[0]["input_tokens"], 20000)
        self.assertEqual(records[1]["prediction"], self.answers[1])
        self.assertFalse(records[1]["metrics"]["strict_match"])

    def test_invalid_schema_keeps_parsed_call_but_cannot_match(self):
        bad = deepcopy(self.answers[1])
        bad["arguments"]["target"]["point_2d"] = [1001, 0]
        result = self.evaluate([xml_call(**self.answers[0]), xml_call(**bad), xml_call(**self.answers[2])])
        self.assertEqual(result["metrics"]["strict_accuracy"]["correct"], 2)
        self.assertEqual(result["metrics"]["tool_accuracy"]["correct"], 3)
        self.assertEqual(result["metrics"]["valid_call_rate"]["correct"], 2)

    def test_interruption_preserves_completed_rows_and_partial_metrics(self):
        with self.assertRaises(KeyboardInterrupt):
            self.evaluate([xml_call(**self.answers[0]), KeyboardInterrupt()])
        summary = read_json(self.output / "summary.json")
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["metrics_scope"], "processed_only")
        self.assertEqual(summary["processed_samples"], 1)
        self.assertEqual(summary["planned_samples"], 3)
        self.assertAlmostEqual(summary["coverage"], 1 / 3)
        self.assertEqual(len(list(jsonl(self.output / "predictions.jsonl"))), 1)
        self.assertEqual(len(self.closed), 1)

    def test_unrecoverable_cuda_error_is_recorded_then_stops(self):
        with self.assertRaises(ProcessFailure):
            self.evaluate([xml_call(**self.answers[0]), ProcessFailure("cuda_context_failure", "Synthetic failure")])
        summary = read_json(self.output / "summary.json")
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["processed_samples"], 2)
        self.assertEqual(summary["metrics"]["strict_accuracy"]["total"], 2)
        self.assertEqual(summary["failure"]["code"], "cuda_context_failure")

    def test_loading_failure_has_incomplete_summary_and_zero_coverage(self):
        with patch("manipulation_agent.training.rollout_backend.GenerationBackend",
                   side_effect=ProcessFailure("model_loading_failed", "Synthetic load failure")), \
             patch("manipulation_agent.training.rollout.environment", return_value={}):
            with self.assertRaises(ProcessFailure):
                rollout(self.package, self.adapter, self.output, model=self.base, recipe=self.recipe)
        summary = read_json(self.output / "summary.json")
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["coverage"], 0)
        self.assertIsNone(summary["metrics"]["strict_accuracy"]["accuracy"])
        self.assertEqual(list(jsonl(self.output / "predictions.jsonl")), [])

    def test_changed_model_invalidates_complete_result_and_preserves_predictions(self):
        def change_model():
            (self.base / "model.safetensors").write_bytes(b"Mutated during evaluation")
            return xml_call(**self.answers[0])
        with self.assertRaises(ProcessFailure):
            self.evaluate([change_model, xml_call(**self.answers[1]), xml_call(**self.answers[2])])
        summary = read_json(self.output / "summary.json")
        self.assertEqual(summary["failure"]["code"], "evaluation_inputs_changed")
        self.assertFalse(summary["complete"])
        self.assertEqual(len(list(jsonl(self.output / "predictions.jsonl"))), 3)

    def test_full_model_runs_without_adapter_and_incompatible_history_is_rejected(self):
        with patch("manipulation_agent.training.rollout_backend.GenerationBackend", self.fake_backend()), \
             patch("manipulation_agent.training.rollout.environment", return_value={}), redirect_stderr(io.StringIO()):
            summary = rollout(self.package, self.base, self.output, recipe=self.recipe)
        self.assertEqual(summary["processed_samples"], 3)
        manifest = read_json(self.output / "input_manifest.json")
        self.assertEqual(manifest["inputs"]["kind"], "full")
        with self.assertRaisesRegex(ConversionError, "history compression"):
            rollout(self.package, self.adapter, self.root / "different", model=self.base,
                    recipe=ROOT / "configs/training/qwen35_4b_lora_img64_compact.toml", dry_run=True)

    def test_managed_adapter_recovers_config_without_optimizer_and_other_data_allowed(self):
        self.managed_adapter()
        result = resolve_inputs(self.adapter)
        self.assertEqual(result["kind"], "lora")
        self.assertEqual(result["metadata"]["configuration_source"], "training_contract")
        self.assertFalse((self.adapter / "optimizer.pt").exists())
        plan = rollout(self.package, self.adapter, self.output, dry_run=True)
        self.assertEqual(plan["selection"]["samples"], 3)
        self.assertFalse(self.output.exists())
        self.assertFalse(plan["weights_loaded"])

    def test_changed_adapter_or_contract_is_rejected(self):
        run = self.managed_adapter()
        (self.adapter / "adapter_model.safetensors").write_bytes(b"Changed")
        with self.assertRaisesRegex(ConversionError, "sealed checkpoint"):
            resolve_inputs(self.adapter)
        contract = read_json(run / "training_contract.json")
        contract["config"]["vision"]["image_max_token_num"] = 64
        save(run / "training_contract.json", contract)
        with self.assertRaisesRegex(ConversionError, "contract hash"):
            resolve_inputs(self.adapter)

    def test_explicit_inputs_required_without_metadata_and_recipe_conflicts_rejected(self):
        with self.assertRaisesRegex(ConversionError, "recipe"):
            resolve_inputs(self.adapter, model=self.base)
        with self.assertRaisesRegex(ConversionError, "--model"):
            resolve_inputs(self.adapter, recipe=self.recipe)
        self.managed_adapter()
        with self.assertRaisesRegex(ConversionError, "conflicts"):
            resolve_inputs(self.adapter, recipe=ROOT / "configs/training/qwen35_4b_lora_img64.toml")

    def test_inaccessible_remote_origin_allows_explicit_inputs_or_local_contract(self):
        from manipulation_agent.training.rollout import _origin_contract
        origin = "/inaccessible-source-host/run"
        remote = Path(origin) / "training_contract.json"
        marker = {"run_root": origin, "contract_fingerprint": "unavailable"}
        save(self.adapter / "manipulation_training.json", marker)
        original = Path.is_file

        def is_file(path):
            if path == remote:
                raise PermissionError("Synthetic inaccessible source-host path")
            return original(path)

        with patch.object(Path, "is_file", is_file):
            resolved = resolve_inputs(self.adapter, model=self.base, recipe=self.recipe)
            self.assertEqual(resolved["metadata"]["configuration_source"], "explicit_recipe")
            self.assertEqual(resolved["kind"], "lora")
            with self.assertRaisesRegex(ConversionError, "--recipe"):
                resolve_inputs(self.adapter, model=self.base)
            contract = {"synthetic_local_contract": True}
            save(self.adapter / "training_contract.json", contract)
            actual, path = _origin_contract(marker, "run_root", artifact_root=self.adapter)
            self.assertEqual(actual, contract)
            self.assertEqual(path, self.adapter / "training_contract.json")

    def test_full_model_and_export_metadata(self):
        full = resolve_inputs(self.base, recipe=self.recipe)
        self.assertEqual(full["kind"], "full")
        run = self.managed_adapter()
        export = self.root / "export"
        merged = export / "model"
        shutil.copytree(self.base, merged)
        save(export / "export_manifest.json", {"status": "complete", "source_run": str(run),
                                              "merged_model": model_manifest(merged, require_weights=True)})
        recovered = resolve_inputs(merged)
        self.assertEqual(recovered["config"], load_config())
        self.assertEqual(recovered["metadata"]["configuration_source"], "training_contract")
        (merged / "model.safetensors").write_bytes(b"Changed merged model")
        with self.assertRaisesRegex(ConversionError, "merged model"):
            resolve_inputs(merged)

    def test_dry_run_limit_and_split_and_output_protection(self):
        with patch("manipulation_agent.training.rollout_backend.GenerationBackend", side_effect=AssertionError("No loading")):
            for split, expected in (("val", 3), ("train", 3), ("all", 6)):
                plan = rollout(self.package, self.adapter, self.output, model=self.base, recipe=self.recipe,
                               split=split, dry_run=True)
                self.assertEqual(plan["selection"]["samples"], expected)
            limited = rollout(self.package, self.adapter, self.output, model=self.base, recipe=self.recipe, limit=1, dry_run=True)
            self.assertEqual(limited["selection"]["samples"], 1)
        self.assertFalse(self.output.exists())
        outputs = [self.package / "evaluation", ROOT / "evaluation_test_never_created"]
        existing = self.root / "existing"
        existing.mkdir()
        outputs.append(existing)
        for output in outputs:
            with self.subTest(output=output), self.assertRaises(ConversionError):
                rollout(self.package, self.adapter, output, model=self.base, recipe=self.recipe, dry_run=True)

    def test_option_validation_and_both_cli_entrypoints(self):
        for options in ({"limit": 0}, {"max_new_tokens": 0}, {"point_tolerance": float("nan")},
                        {"point_tolerance": -1}, {"device": "auto"}):
            with self.subTest(options=options), self.assertRaises(ConversionError):
                rollout(self.package, self.adapter, self.output, model=self.base, recipe=self.recipe, **options)
        argv = ["--data", str(self.package), "--checkpoint", str(self.adapter), "--model", str(self.base),
                "--recipe", str(self.recipe), "--output", str(self.output), "--dry-run"]
        with redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(main(["rollout", *argv]), 0)
        self.assertEqual(json.loads(stream.getvalue())["status"], "dry_run")
        result = subprocess.run([sys.executable, str(ROOT / "scripts/rollout_sft.py"), *argv],
                                cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "dry_run")

    def test_imports_and_help_need_no_ml_or_simulator(self):
        code = ("import sys; from manipulation_agent.training import rollout, rollout_metrics, rollout_backend; "
                "assert not any(n.split('.')[0] in {'torch','swift','peft','transformers','omnigibson'} for n in sys.modules)")
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cpu_reference_kernels_restore_gpu_wrappers_even_on_failure(self):
        from manipulation_agent.training.rollout_backend import cpu_reference_kernels
        def reference(value):
            return value * 2
        @wraps(reference)
        def gpu_only(value):
            raise RuntimeError("CUDA input required")
        module = SimpleNamespace(causal_conv1d_fn=gpu_only, causal_conv1d_update=gpu_only,
                                 torch_chunk_gated_delta_rule=gpu_only, torch_recurrent_gated_delta_rule=gpu_only)
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            with cpu_reference_kernels(module):
                self.assertEqual(module.causal_conv1d_fn(3), 6)
                self.assertEqual(module.torch_chunk_gated_delta_rule(4), 8)
                raise RuntimeError("synthetic failure")
        self.assertIs(module.causal_conv1d_fn, gpu_only)
        self.assertIs(module.torch_chunk_gated_delta_rule, gpu_only)
