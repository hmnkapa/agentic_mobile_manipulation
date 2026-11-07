"""Evidence from actual GPU training/resume/export, requested explicitly."""
import os
from pathlib import Path
import pytest
from manipulation_agent.sft.common import read_json


@pytest.mark.gpu
def test_real_gpu_resume_and_export(request):
    names = ("MANIP_TEST_SMOKE_FIRST", "MANIP_TEST_SMOKE_RESUMED", "MANIP_TEST_EXPORT")
    if not all(os.environ.get(name) for name in names):
        expression = request.config.option.markexpr or ""
        if "gpu" in expression and "not gpu" not in expression:
            pytest.fail("Explicit GPU validation requires first/resumed/export evidence paths")
        pytest.skip("Real GPU evidence paths are not configured")
    first, second, exported = [Path(os.environ[name]) for name in names]
    a, b = read_json(first / "run.json"), read_json(second / "run.json")
    assert a["status"] == "awaiting_resume" and a["final_step"] == 2
    assert b["status"] == "completed" and b["expected_start_step"] == 2 and b["final_step"] == 4
    audit = read_json(second / "audit.json")
    assert audit["optimizer_restored"] and audit["rng_restored"] and audit["adapter_updated"] and audit["epoch"] == 1.0
    assert audit["finite_gradient_steps"] == [3, 4]
    assert audit["train_samples"] == 16 and all(e["samples"] == 8 for e in audit["eval_counts"])
    result = read_json(exported / "export_manifest.json")
    assert result["status"] == "complete" and result["checkpoint"]["global_step"] == 4
    generation = read_json(exported / "generation_validation.json")
    assert generation["status"] == "passed" and not generation["task_success_evaluated"]
    assert all(row["finite_forward"] and row["generated_tokens"] for row in generation["records"])
