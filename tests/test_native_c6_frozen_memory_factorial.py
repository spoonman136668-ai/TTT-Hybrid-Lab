"""C6: saved-model/native-memory factorial has NO outcome-driven write."""
from copy import deepcopy
import hashlib
import json

import pytest
import torch

from scripts import native_c6_frozen_memory_factorial as c6


def test_graft_fast_copy_and_same_capacity_scrambled_negative():
    torch.manual_seed(11)
    old = {
        "recurrent": torch.ones(1, 12),
        "fast_weights": torch.arange(12.0).reshape(1, 1, 12),
        "history": torch.ones(1, 2, 12),
    }
    recent = {
        "recurrent": torch.zeros(1, 12),
        "fast_weights": -torch.ones(1, 1, 12),
        "history": torch.zeros(1, 4, 12),
    }
    old_saved, recent_saved = deepcopy(old), deepcopy(recent)
    graft = c6.graft_fast(old, recent)
    sham = c6.graft_fast(old, recent, rolled=True)
    assert torch.equal(graft["fast_weights"], old_saved["fast_weights"])
    assert torch.equal(graft["recurrent"], recent_saved["recurrent"])
    assert torch.equal(graft["history"], recent_saved["history"])
    assert torch.equal(sham["fast_weights"], torch.roll(old["fast_weights"], 1, -1))
    assert torch.equal(torch.sort(sham["fast_weights"].flatten()).values,
                       torch.sort(old["fast_weights"].flatten()).values)
    graft["fast_weights"].add_(13)
    assert torch.equal(old, old_saved) if False else torch.equal(
        old["fast_weights"], old_saved["fast_weights"])
    assert torch.equal(recent["fast_weights"], recent_saved["fast_weights"])


def test_reject_wrong_or_incomplete_native_fast_shape():
    old = {"recurrent": torch.ones(1, 12),
           "fast_weights": torch.ones(1, 1, 12),
           "history": torch.ones(1, 3, 12)}
    bad = deepcopy(old)
    bad["fast_weights"] = torch.zeros(1, 1, 6)
    with pytest.raises(AssertionError, match="FAST_SHAPE"):
        c6.graft_fast(old, bad)
    del bad["history"]
    with pytest.raises(AssertionError, match="NATIVE_FIELDS"):
        c6.graft_fast(old, bad)


def test_genuine_saved_checkpoints_no_feedback_evaluation(tmp_path):
    torch.set_num_threads(1)
    trial = tmp_path / "isolated"
    trial.mkdir()
    result = c6.measure_one(c6._spec()["seeds"][0], trial)
    assert result["status"] == "PASS_NO_FEEDBACK_FACTORIAL_EVIDENCE_INTEGRITY_DEV"
    assert result["root_sha256"] != result["step40_sha256"]
    assert result["step40_sha256"] != result["step80_sha256"]
    assert len(result["variants"]) == len(c6.ARMS) == 7
    assert len(result["diagnostic_hypotheses"]) == 5
    assert all(type(v) is bool for v in result["diagnostic_hypotheses"].values())
    for arm, entry in result["variants"].items():
        assert entry["samples_per_regime"] == 16
        assert entry["read_only"] is True
        assert len(entry["raw_evidence"]["vectors"]["old_regime_mse"]) == 16
        assert len(entry["raw_evidence"]["vectors"]["postreversal_mse"]) == 16
        assert entry["evidence_sha256"] == hashlib.sha256(
            c6._canonical(entry["raw_evidence"])).hexdigest()
        assert entry["old_mse"] == pytest.approx(
            sum(entry["raw_evidence"]["vectors"]["old_regime_mse"]) / 16)
    assert result["fast_memory_writes_during_evaluation"] == 0
    assert result["optimizer_updates_during_evaluation"] == 0
    assert result["all_models_optimizers_and_saved_memory_unchanged"] is True
    assert result["actual_context_router"].startswith("NOT_PRESENT")
    assert result["scientific_first_use"] == "NOT_RUN"
    assert result["scientifically_accepted"] is False
    assert result["model_promoted"] is False
    assert not (trial / "accepted").exists()
    assert not (trial / "latest").exists()


def test_scientific_first_use_denied(monkeypatch):
    monkeypatch.setenv("NATIVE_C6_ENGINEERING_ONLY", "0")
    with pytest.raises(RuntimeError, match="SCIENCE_FIRST_USE_FORBIDDEN"):
        c6.run()
    monkeypatch.delenv("NATIVE_C6_ENGINEERING_ONLY")
    monkeypatch.setenv("NATIVE_P3_ENGINEERING_ONLY", "0")
    with pytest.raises(RuntimeError, match="SCIENCE_FIRST_USE_FORBIDDEN"):
        c6.run()
