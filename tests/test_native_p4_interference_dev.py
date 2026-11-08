"""P4 developmental, not scientific-source or seeded world testing."""
import ast
import json
from pathlib import Path

import pytest
import torch

from scripts.native_p4_interference_dev import (
    ARMS, BLOCK_SIZE, CONTEXTS, EVENTS, MODEL_INITIALIZATIONS,
    context_at, controls_at, factual_outcome, run_initialization, run)


def test_two_known_contexts_and_algebraic_action_reversal():
    assert EVENTS == 96
    assert CONTEXTS == ("A", "B", "A", "B")
    assert [context_at(i) for i in (0, 23, 24, 47, 48, 71, 72, 95)] == (
        ["A", "A", "B", "B", "A", "A", "B", "B"])
    for t in (0, 24, 48, 72):
        s, action, dt = controls_at(t)
        assert action.item() == -1
        assert dt.item() > 0
        neg = factual_outcome(context_at(t), s, action)
        pos = factual_outcome(context_at(t), s, -action)
        assert ((pos - neg).item() > .8) == (context_at(t) == "A")
    with pytest.raises(ValueError, match="EVENT_OUT_OF_RANGE"):
        context_at(EVENTS)


def test_realized_observation_rejects_malformed_shapes():
    with pytest.raises(ValueError, match="INVALID_ANALYTIC_CONTEXT"):
        factual_outcome("Z", torch.zeros(1, 1), torch.ones(1, 1))
    with pytest.raises(ValueError, match="NONFINITE_INPUT"):
        factual_outcome("A", torch.tensor([[float("nan")]]), torch.ones(1, 1))


def test_one_model_factual_fast_vs_bank_and_no_feedback():
    torch.set_num_threads(1)
    receipt = run_initialization(MODEL_INITIALIZATIONS[0])
    assert receipt["status"] == "PASS_ENGINEERING_INTEGRITY"
    assert all(receipt["checks"].values())
    assert receipt["fast_bank_capacity_confounded_oracle"] is True
    assert all(set(phase) == set(ARMS) for phase in receipt["phase_mse"].values())
    assert all(all(x >= 0 for x in phase.values())
               for phase in receipt["phase_mse"].values())


def test_all_initializations_preserve_null_hypothesis_outcomes():
    result = run()
    assert result["status"] == "PASS_DEV_INTEGRITY_NOT_SCIENCE"
    assert len(result["records"]) == 8
    assert result["science_first_use"] == "NOT_RUN"
    assert result["world_seeds_consumed"] == []
    assert result["worlds_constructed"] == 0
    assert result["model_optimizer_updates"] == 0
    assert all(0 <= x <= 8 for x in result["hypothesis_summary_count_of_eight"].values())


def test_frozen_manifest_does_not_grant_scientific_firstuse():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/"experiments/native_p4_interference_dev_frozen.json").read_text())
    assert manifest["schema"] == "ttt-hybrid.native-p4.interference-dev-frozen.v1"
    assert manifest["source"]["parent_p3_sha"] == "82d68d947242d50d4993942ba2e9d5acf6343055"
    assert manifest["science_first_use"]["authorized"] is False
    assert manifest["science_first_use"]["seed_assignments"] == {}
    assert manifest["science_first_use"]["original_worlds_consumed"] == 0


def test_science_fence_and_source_isolation(monkeypatch):
    monkeypatch.setenv("NATIVE_P4_DEV_ONLY","0")
    with pytest.raises(RuntimeError, match="SCIENTIFIC_FIRST_USE_FORBIDDEN"):
        run()
    root=Path(__file__).resolve().parents[1]
    script=(root/"scripts/native_p4_interference_dev.py").read_text()
    imports=[node.module or "" for node in ast.walk(ast.parse(script))
             if isinstance(node,ast.ImportFrom)]
    assert not any(marker in name for name in imports for marker in
                   ("g7_s20","g7_s25","g7_s26","g7_s27"))
