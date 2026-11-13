"""Formal F8-R3 confounded-policy impossibility witness; no source effects."""
import inspect
import math
from pathlib import Path

import pytest

from experiments import p85mr_f8r3_counterpolicy_identifiability as f8


def test_freeze_was_committed_before_source_and_no_firstuse():
    data = f8.locked()
    assert data["original_f8_pr"] == 236
    assert data["scope"]["synthetic_mathematical_construction_only"] is True
    assert data["evidence"]["original_f7_actions_consumed"] == 12288
    assert data["evidence"]["source_interventions_new"] == 0
    assert not data["scope"]["no_independent_scientific_PASS"] is False


def test_two_different_worlds_are_exactly_indistinguishable_on_policy():
    grid = f8.original_synthetic_grid()
    assert len(grid) == 101
    assert len(set(grid)) == 101
    for s in grid:
        a = f8.behavior(s)
        assert f8.synthetic_effect(s, a, +1) == f8.synthetic_effect(s, a, -1)
        assert f8.synthetic_effect(s, a, +1) == 0.5 * s


def test_unseen_action_has_independent_worlds_and_irreducible_bound():
    r = f8.receipt()
    assert r["max_logged_policy_outcome_difference"] == 0
    assert r["unobserved_alternative_outcome_gap"] == pytest.approx(0.6)
    assert r["unavoidable_common_predictor_alternative_mse_lower_bound"] == pytest.approx(0.09)
    for s in f8.original_synthetic_grid():
        y_plus = f8.synthetic_effect(s, 1-f8.behavior(s), 1)
        y_minus = f8.synthetic_effect(s, 1-f8.behavior(s), -1)
        midpoint = (y_plus + y_minus) / 2
        for prediction in (midpoint, midpoint-0.4, midpoint+0.2, 0.0):
            mean_mse = ((prediction-y_plus)**2+(prediction-y_minus)**2)/2
            assert mean_mse >= 0.09-1e-12


def test_same_goal_planner_must_choose_different_actions():
    assert f8.chooser(1, 0.2, -0.2) == 0
    assert f8.chooser(-1, 0.2, -0.2) == 1
    assert f8.receipt()["planner_optimal_actions_depend_on_hidden_world"] == {
        "plus": 0, "minus": 1}


def test_input_denials_and_zero_authority():
    for s in (float("nan"), float("inf"), -1.01, 1.01, True, "0"):
        with pytest.raises(ValueError, match="INVALID_SYNTHETIC_STATE"):
            f8.behavior(s)
    for a in (2, -1, 1.0, True, None):
        with pytest.raises(ValueError, match="UNREGISTERED_ACTION"):
            f8.synthetic_effect(0.0, a, 1)
    for w in (0, 2, False, "plus"):
        with pytest.raises(ValueError, match="UNREGISTERED_WORLD"):
            f8.synthetic_effect(0, 1, w)
    for t in (float("nan"), float("inf"), None):
        with pytest.raises(ValueError, match="INVALID_SYNTHETIC_GOAL"):
            f8.chooser(1, 0.2, t)
    with pytest.raises(PermissionError, match="EXTERNAL_SOURCE_FIRSTUSE_NOT_AUTHORIZED"):
        f8.refuse_real_source()
    result = f8.receipt()
    assert result["new_effects_executed"] == 0
    assert result["source_worlds"] == 0
    assert result["scientific_pass"] is False


def test_no_real_simulator_or_external_action_path():
    src = inspect.getsource(f8)
    for invalid in ("import causal_gym", "from causal_gym", "env.do(",
                    "env.reset(", "workflow_dispatch(", "subprocess.run(",
                    "torch.load(", "requests.get(", "urlopen("):
        assert invalid not in src
