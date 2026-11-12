"""F8-R3: exact analytical counterpolicy non-identifiability negative control.

All 'worlds' here are purely mathematical functions. They are NOT original
Causal-Gymnasium worlds, F7 holdout data, new causal effects, or models
trained on evidence. The point is to falsify the inference that near-zero
logged-policy prediction error implies learned unchosen-action consequences.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/p85mr_f8r3_counterpolicy_identifiability_freeze.json"
FREEZE_BLOB = "1727345ae8cd953699ab1a30b67f35b391642298"
SCHEMA = "ttt-hybrid.f8r3.counterpolicy-identifiability-dev.v1"
STATE_COUNT = 101
C = 0.3


def locked():
    raw = FREEZE.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if blob != FREEZE_BLOB:
        raise ValueError("F8R3_FROZEN_PREIMPLEMENTATION_DRIFT")
    data = json.loads(raw)
    p, scope = data["preset"], data["scope"]
    if (data["schema"] != SCHEMA or data["classification"] !=
            "SOURCE_NEUTRAL_LOGGED_POLICY_NONIDENTIFIABILITY_NEGATIVE_CONTROL"
            or data["predecessor_head"] !=
            "311a0db437a5e003aedac410b4e211d575db0056"
            or p["n_states"] != STATE_COUNT
            or p["minimum_unavoidable_mse_per_counterpolicy_state"] != 0.09
            or p["target_counterpolicy_gap"] != 0.6
            or p["synthetic_analytical_world_only_no_empirical_measurement"] is not True
            or scope["no_new_source_firstuse"] is not True
            or scope["no_F7_data_reevaluation"] is not True
            or scope["no_independent_scientific_PASS"] is not True):
        raise ValueError("F8R3_PROTOCOL_OR_AUTHORITY_DRIFT")
    return data


def _state(s: float) -> float:
    if (type(s) not in (int, float) or
            not math.isfinite(s) or not -1.0 <= s <= 1.0):
        raise ValueError("F8R3_INVALID_SYNTHETIC_STATE")
    return float(s)


def behavior(s: float) -> int:
    return int(_state(s) >= 0.0)


def synthetic_effect(s: float, action: int, world_sign: int) -> float:
    """Two different *invented* physical rules, observationally identical.

    y(+/- | s, a) = s/2 +/- 0.3 * (a - deterministic_behavior(s)).
    """
    s = _state(s)
    if type(action) is not int or action not in (0, 1):
        raise ValueError("F8R3_UNREGISTERED_ACTION")
    if type(world_sign) is not int or world_sign not in (-1, 1):
        raise ValueError("F8R3_UNREGISTERED_WORLD")
    return 0.5 * s + world_sign * C * (action - behavior(s))


def original_synthetic_grid():
    locked()
    return tuple((i - 50) / 50 for i in range(STATE_COUNT))


def chooser(world_sign: int, s: float, target: float) -> int:
    """Same goal and action budget; oracle diagnostic only, NOT a learned policy."""
    if type(target) not in (int, float) or not math.isfinite(target):
        raise ValueError("F8R3_INVALID_SYNTHETIC_GOAL")
    return min((0, 1), key=lambda a: (
        (synthetic_effect(s, a, world_sign) - float(target))**2, a))


def receipt():
    """Constructive proof: same observations, different opposite-action facts."""
    locked()
    grid = original_synthetic_grid()
    logged = tuple((s, behavior(s), synthetic_effect(s, behavior(s), 1))
                   for s in grid)
    logged_negative = tuple((s, behavior(s), synthetic_effect(s, behavior(s), -1))
                            for s in grid)
    if logged != logged_negative:
        raise AssertionError("F8R3_LOGGED_WORLDS_NOT_IDENTICAL")
    counter = tuple((
        synthetic_effect(s, 1-behavior(s), 1),
        synthetic_effect(s, 1-behavior(s), -1))
        for s in grid)
    gap = max(abs(a-b) for a, b in counter)
    if any(abs(abs(a-b)-0.6) > 1e-12 for a, b in counter):
        raise AssertionError("F8R3_OPPOSITE_ACTION_GAP_DRIFT")
    # For ANY single prediction z, the mean error to the two possible
    # unobserved outcomes is (z - midpoint)**2 + (gap/2)**2.
    # Its unavoidable per-state lower bound is (0.6/2)**2 = 0.09.
    minimax_bound = min(((a-b)/2)**2 for a, b in counter)
    signed_policy = {
        "plus": chooser(1, 0.2, -0.2),
        "minus": chooser(-1, 0.2, -0.2),
    }
    if signed_policy != {"plus": 0, "minus": 1}:
        raise AssertionError("F8R3_DIFFERENT_GOAL_ACTIONS_NOT_PROVEN")
    return {
        "schema": SCHEMA,
        "classification": "SYNTHETIC_MATHEMATICAL_NEGATIVE_NOT_EMPIRICAL",
        "frozen_blob": FREEZE_BLOB,
        "source_worlds": 0,
        "original_F7_reuse": False,
        "new_effects_executed": 0,
        "common_observed_worlds": len(logged),
        "max_logged_policy_outcome_difference": 0.0,
        "unobserved_alternative_outcome_gap": gap,
        "unavoidable_common_predictor_alternative_mse_lower_bound": minimax_bound,
        "shared_planner_goal": -0.2,
        "planner_optimal_actions_depend_on_hidden_world": signed_policy,
        "independently_identifies_unchosen_consequences": False,
        "trained_model_beats_linear": False,
        "source_firstuse_authorized": False,
        "scientific_pass": False,
        "deployment": False,
    }


def refuse_real_source(*args, **kwargs):
    raise PermissionError("F8R3_EXTERNAL_SOURCE_FIRSTUSE_NOT_AUTHORIZED")
