"""Opt-in native template and checkpoint checks; never task-success evidence."""
from dataclasses import asdict
import os

import pytest

from manipulation_agent.sft.common import ConversionError
from manipulation_agent.training.artifacts import environment, now, write_json
from manipulation_agent.training.data import index_package
from manipulation_agent.training.rollout import build_prompt, resolve_inputs
from manipulation_agent.training.rollout_metrics import json_equal, parse_tool_call, target_call, validate_call


def required(request, marker, names):
    values = {name: os.environ.get(name) for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        expression = request.config.option.markexpr or ""
        if marker in expression and "not " + marker not in expression:
            pytest.fail("Explicit validation requires: " + ", ".join(missing))
        pytest.skip("Set " + ", ".join(missing) + " for real rollout validation")
    return values


def coverage_samples(index):
    selected = []
    for tool in ("initialize", "act", "finish"):
        candidates = [s for s in index.samples if s.split == "val" and s.target_tool == tool
                      and (s.image_count == 0 if tool == "initialize" else s.image_count >= 2)]
        assert candidates, f"No {tool} sample with the required image coverage"
        selected.append(min(candidates, key=lambda s: (s.image_count, s.line)))
    return selected


@pytest.mark.qwen_encode
def test_native_generation_prompt_and_xml_roundtrip(request):
    values = required(request, "qwen_encode", ("MANIP_TEST_ROLLOUT_DATA", "MANIP_TEST_ROLLOUT_FULL", "MANIP_TEST_ROLLOUT_RECIPE"))
    from manipulation_agent.training.encoding import normalize
    from manipulation_agent.training.swift_backend import build_encoding_backend
    inputs = resolve_inputs(values["MANIP_TEST_ROLLOUT_FULL"], recipe=values["MANIP_TEST_ROLLOUT_RECIPE"])
    config = inputs["config"]
    index = index_package(values["MANIP_TEST_ROLLOUT_DATA"], expected_k=config["data"]["k"],
                          expected_history_compression=config["data"]["history_compression"])
    backend = build_encoding_backend(config, inputs["model"]["root"], unbounded=True)
    for sample, row in index.records():
        gold = target_call(row)
        rendered = backend.render_target(row["messages"][-1]["tool_calls"][0]["function"])
        parsed = parse_tool_call(rendered, row["tools"])
        validate_call(parsed, row["tools"])
        assert json_equal(parsed, gold), sample.sample_id
    backend.template.set_mode("transformers")
    backend.template.max_length = None
    for sample in coverage_samples(index):
        original = index.record(sample, absolute_images=True)
        prompt = build_prompt(original)
        assert prompt["messages"] == original["messages"][:-1]
        assert prompt["images"] == original["images"]
        encoded = backend.template.encode(normalize(prompt, backend.preprocessor))
        assert 0 < len(encoded["input_ids"]) <= config["data"]["max_length"]
        assert len(encoded.get("image_grid_thw", [])) == len(prompt["images"])


@pytest.mark.gpu
@pytest.mark.parametrize("format_name,variable", [("lora", "MANIP_TEST_ROLLOUT_LORA"), ("full", "MANIP_TEST_ROLLOUT_FULL")])
def test_real_checkpoint_generation(request, tmp_path, format_name, variable):
    values = required(request, "gpu", ("MANIP_TEST_ROLLOUT_DATA", variable))
    from manipulation_agent.training.rollout_backend import GenerationBackend
    inputs = resolve_inputs(values[variable], model=os.environ.get("MANIP_TEST_ROLLOUT_MODEL") if format_name == "lora" else None,
                            recipe=os.environ.get("MANIP_TEST_ROLLOUT_RECIPE"))
    assert inputs["kind"] == format_name
    config = inputs["config"]
    index = index_package(values["MANIP_TEST_ROLLOUT_DATA"], expected_k=config["data"]["k"],
                          expected_history_compression=config["data"]["history_compression"])
    report = {"status": "running", "started_at": now(), "task_success_evaluated": False,
              "validation_level": "offline_generation_pending", "inputs": inputs,
              "package": index.manifest(), "samples": []}
    result_path = tmp_path / (format_name + "_generation.json")
    backend = None
    try:
        backend = GenerationBackend(config, inputs["model"]["root"],
                                    inputs["checkpoint"]["root"] if format_name == "lora" else None,
                                    os.environ.get("MANIP_TEST_ROLLOUT_DEVICE", "cuda:0"),
                                    int(os.environ.get("MANIP_TEST_ROLLOUT_MAX_NEW_TOKENS", "128")))
        report["environment"] = environment()
        report["backend"] = backend.describe()
        for sample in coverage_samples(index):
            row = index.record(sample, absolute_images=True)
            result = backend.generate(build_prompt(row))
            assert result["generated_tokens"] > 0 and result["raw_response"].strip()
            try:
                parsed = parse_tool_call(result["raw_response"], row["tools"])
                validate_call(parsed, row["tools"])
                result["parsed_call"] = parsed
            except ConversionError as exc:
                result["parse_error"] = exc.record()
            report["samples"].append({"sample": asdict(sample), **result})
            write_json(result_path, report)
        report.update(status="passed", validation_level=backend.validation_level)
    finally:
        if report["status"] == "running":
            report["status"] = "failed"
        report["completed_at"] = now()
        write_json(result_path, report)
        if backend is not None:
            backend.close()
