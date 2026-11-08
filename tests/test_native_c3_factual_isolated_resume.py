"""C3 DEV-only validation of native factual learning with isolated candidate restore."""
import random

import pytest
import torch

from scripts import native_p3_continuity as factual
from scripts.native_c3_factual_isolated_resume import verify_one, run


def test_factual_native_40_to_80_isolated_resume_vs_reference(tmp_path):
    torch.set_num_threads(1)
    python_before = random.getstate()
    result = verify_one(factual.MODEL_INITIALIZATIONS[0], tmp_path)
    assert result["status"] == "PASS_FACTUAL_CANDIDATE_ISOLATED_RESUME_DEV"
    assert all(result["checks"].values())
    assert result["checkpoint_at_steps"] == 40
    assert result["final_steps"] == 80
    assert result["final_consumed_examples"] == 80
    assert result["protected_reference_unchanged"]
    assert result["science_first_use"] == "NOT_RUN"
    assert result["training_activated"] is False
    assert result["model_promoted"] is False
    assert random.getstate() == python_before


def test_c3_original_science_admission_explicitly_blocked(monkeypatch):
    monkeypatch.setenv("NATIVE_C3_ENGINEERING_ONLY", "0")
    with pytest.raises(RuntimeError, match="SCIENCE_FIRST_USE_FORBIDDEN"):
        run()
    monkeypatch.delenv("NATIVE_C3_ENGINEERING_ONLY")
    monkeypatch.setenv("NATIVE_P3_ENGINEERING_ONLY", "0")
    with pytest.raises(RuntimeError, match="SCIENCE_FIRST_USE_FORBIDDEN"):
        run()
