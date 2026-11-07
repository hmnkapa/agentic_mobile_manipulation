"""Explicit real-processor evidence checks; skips are not validation."""
import os
from pathlib import Path
import pytest
from manipulation_agent.sft.common import read_json, jsonl, file_hash


@pytest.mark.qwen_encode
def test_full_real_encoding_report(request):
    value = os.environ.get("MANIP_TEST_PREFLIGHT")
    if not value:
        expression = request.config.option.markexpr or ""
        if "qwen_encode" in expression and "not qwen_encode" not in expression:
            pytest.fail("MANIP_TEST_PREFLIGHT is required for explicit real encoding validation")
        pytest.skip("Set MANIP_TEST_PREFLIGHT to inspect real encoding evidence")
    root = Path(value)
    summary = read_json(root / "summary.json")
    assert summary["validation_level"] == "qwen_processor_encoding"
    assert summary["scan_complete"] and summary["encoding_passed"]
    assert summary["samples"] == summary["expected_samples"] == 3323
    for name, digest in summary["report_files"].items():
        assert file_hash(root / name) == digest
    rows = [r for _, r in jsonl(root / "samples.jsonl")]
    assert len(rows) == 3323
    assert all(r["complete_target_verified"] and r["all_history_masked"] and not r["weights_loaded"] for r in rows)
