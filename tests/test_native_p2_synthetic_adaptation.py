"""P2 synthetic-only API/provenance negatives: no scientific-world construction."""
import ast
import json
from pathlib import Path
import pytest
import torch

from scripts.native_p2_synthetic_adaptation import (
    MODEL_INITIALIZATIONS, TRIALS, REVERSE_AT, analytic_consequence, run_one, run)


def test_action_consequence_reversal_is_synthetic_algebra():
    neg = torch.tensor([[-1.0]])
    pos = -neg
    assert (analytic_consequence(pos, 0)-analytic_consequence(neg, 0)).item() > .8
    assert (analytic_consequence(pos, REVERSE_AT)-analytic_consequence(neg, REVERSE_AT)).item() < -.8
    assert torch.equal(analytic_consequence(pos, 0), analytic_consequence(pos, REVERSE_AT-1))
    with pytest.raises(ValueError, match="TRIAL_OUT_OF_RANGE"):
        analytic_consequence(pos, TRIALS)
    with pytest.raises(ValueError, match="INVALID_ACTION"):
        analytic_consequence(torch.ones(2, 1), 0)


def test_registered_development_initializations_only():
    assert MODEL_INITIALIZATIONS == tuple(range(8))
    with pytest.raises(ValueError, match="UNREGISTERED_ENGINEERING_INITIALIZATION"):
        run_one(7207)


def test_fixed_weight_factual_feedback_recovers_with_sign_and_reversal_cost():
    # A synthetic engineering check only, not scientific model qualification.
    torch.set_num_threads(1)
    r=run_one(0)
    assert r["status"] == "PASS_SYNTHETIC_ADAPTATION_API_ONLY"
    assert r["controls_same_model_weights"] is True
    assert r["prequential_prediction_before_factual_observation"] is True
    assert r["alternate_consequence_not_used_for_learning"] is True
    assert all(r["checks"].values())


def test_science_execution_fence_fails_before_any_probe(monkeypatch):
    monkeypatch.setenv("NATIVE_P2_SYNTHETIC_ONLY", "0")
    with pytest.raises(RuntimeError, match="SCIENCE_FIRSTUSE_FORBIDDEN"):
        run()


def test_no_s20_or_s26_s27_data_sources_imported():
    src=(Path(__file__).resolve().parents[1]/"scripts/native_p2_synthetic_adaptation.py").read_text()
    tree=ast.parse(src)
    modules=[n.module or "" for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert not any(k in m for m in modules for k in ("g7_s20", "g7_s25", "g7_s26", "g7_s27"))


def test_p2_protocol_only_engineering_no_science_roles():
    p=json.loads((Path(__file__).resolve().parents[1]/
       "experiments/native_p2_synthetic_reversal_engineering.json").read_text())
    assert p["schema"] == "ttt-hybrid.native-p2.synthetic-reversal-engineering.v1"
    assert p["source"]["parent_p1_sha"] == "706658b5038c11642829868b0f292ac2fe4452e0"
    assert p["science"]["authorized"] is False
    assert p["science"]["seed_allocations"] == {}
    assert p["science"]["data_consumed"] is False
    assert p["scope"]["frozen_weights"] is True
