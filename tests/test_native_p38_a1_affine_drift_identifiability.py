"""P38-A1 engineering-only causal-identifiability controls (no source first use).

A single opposite-action pair is not sufficient to identify the sign of an
unknown action effect when an unobserved nuisance intercept drifts in time.
A predeclared three-observation ABA contrast cancels AFFINE nuisance drift,
but not unrestricted nonlinear drift. No learner, trained checkpoint, seeded
science cohort, external-source action, privileged context, or controller.
"""
from __future__ import annotations

import math

import pytest


BETA = 0.30


def observed(action: int, mode: int, nuisance: float) -> float:
    """Evaluator-only explicit analytic fixture; 'mode' never given to rules."""
    if action not in (-1, 1) or mode not in (-1, 1) or not math.isfinite(nuisance):
        raise ValueError("P38_A1_INVALID_AUTHORED_EVENT")
    return nuisance + BETA * action * mode


def two_action_contrast(completed: tuple[tuple[int, float], ...]) -> int | None:
    """P6-style naive last-opposite-action contrast, source facts only."""
    seen: dict[int, float] = {}
    for action, value in completed:
        if action not in (-1, 1) or not math.isfinite(value):
            raise ValueError("P38_A1_INVALID_FACTUAL_HISTORY")
        seen[action] = value
    if len(seen) != 2:
        return None
    slope = (seen[1] - seen[-1]) / 2
    return 1 if slope > 0 else (-1 if slope < 0 else None)


def aba_affine_drift_contrast(
    completed: tuple[tuple[float, int, float], ...],
) -> int | None:
    """No-training baseline, three *completed* equally spaced factual actions.

    Requires A,-A,A in chronological order. Unknown linear drift b(t)=b0+vt
    cancels in ((y0+y2)/2-y1)*A = 2*BETA*mode. No future labels or outcomes.
    """
    if len(completed) != 3:
        return None
    (t0, a0, y0), (t1, a1, y1), (t2, a2, y2) = completed
    if (any(not math.isfinite(x) for x in (t0, t1, t2, y0, y1, y2))
        or a0 not in (-1, 1) or a1 != -a0 or a2 != a0
        or not t0 < t1 < t2 or not math.isclose(t1-t0, t2-t1, abs_tol=1e-12, rel_tol=1e-12)):
        raise ValueError("P38_A1_UNIDENTIFIABLE_THREE_STEP_HISTORY")
    contrast = a0 * ((y0 + y2) / 2 - y1)
    return 1 if contrast > 0 else (-1 if contrast < 0 else None)


def test_opposite_modes_can_have_bit_identical_two_step_observations():
    """Without nuisance-drift assumptions, causal mode is unidentifiable."""
    # The same physical action schedule and exact observed sequence arise from
    # different actual modes under different (unobserved) time-varying offsets.
    actions = (1, -1)
    world_plus = ((1, observed(1, 1, 0.0)), (-1, observed(-1, 1, 0.0)))
    world_minus = ((1, observed(1, -1, 0.6)), (-1, observed(-1, -1, -0.6)))
    assert world_plus == world_minus == ((1, 0.3), (-1, -0.3))
    assert two_action_contrast(world_plus) == two_action_contrast(world_minus) == 1
    assert actions == tuple(action for action, _ in world_plus)
    # Each world needs an opposite action for positive goal displacement.
    assert 1 * 1 != 1 * -1


def test_three_factual_aba_events_cancel_affine_background_drift():
    """Full 2 modes x 2 starting actions x 3 offsets x 5 drifts = 60 cases."""
    count = 0
    naive_errors = 0
    for mode in (-1, 1):
        for first_action in (-1, 1):
            for base in (-0.9, 0.0, 0.7):
                for drift in (-0.8, -0.15, 0.0, 0.15, 0.8):
                    actions = (first_action, -first_action, first_action)
                    facts = tuple((float(i), action, observed(action, mode, base+drift*i))
                                  for i, action in enumerate(actions))
                    inferred = aba_affine_drift_contrast(facts)
                    assert inferred == mode
                    assert aba_affine_drift_contrast(tuple(facts)) == mode  # cold-value replay
                    naive = two_action_contrast(tuple((a,y) for _,a,y in facts[:2]))
                    naive_errors += naive != mode
                    count += 1
    assert count == 60
    assert naive_errors > 0  # incomplete P6-style two-action control fails under drift


def test_contrast_not_guaranteed_under_quadratic_or_sudden_mode_shift():
    """Explicit failure domain; an authored ABA rule is not full causal reasoning."""
    mode = 1
    actions = (1, -1, 1)
    # Nuisance time curvature q*t*t contributes q to the ABA contrast.
    q = -1.0
    facts = tuple((float(i), actions[i], observed(actions[i], mode, q*i*i)) for i in range(3))
    assert aba_affine_drift_contrast(facts) == -1  # wrong; must preserve this negative
    assert aba_affine_drift_contrast(facts[:2]) is None


def test_malformed_chronology_or_counterfactual_action_pattern_rejected():
    valid = ((0., 1, .3), (1., -1, -.3), (2., 1, .3))
    assert aba_affine_drift_contrast(valid) == 1
    for malformed in (
        ((0., 1, .3), (2., -1, -.3), (3., 1, .3)),
        ((0., 1, .3), (1., 1, -.3), (2., 1, .3)),
        ((0., 1, .3), (1., -1, float('nan')), (2., 1, .3)),
        ((0., 1, .3), (0., -1, -.3), (2., 1, .3)),
    ):
        with pytest.raises(ValueError, match="P38_A1_UNIDENTIFIABLE_THREE_STEP_HISTORY"):
            aba_affine_drift_contrast(malformed)
    with pytest.raises(ValueError, match="P38_A1_INVALID_FACTUAL_HISTORY"):
        two_action_contrast(((1, float('nan')),))
