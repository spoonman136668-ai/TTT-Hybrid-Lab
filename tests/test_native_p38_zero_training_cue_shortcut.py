"""P38 independent no-training shortcut audit (DEV, NO protected source first use).

The authored world exposes cue and explicit goal. With |cue| >= .45,
its physical action coefficient never changes sign except with cue.
This zero-parameter decision rule needs no learned model or training data.
Use a mathematical bound plus a fixed hand-authored grid: no P38/P38-R1
random seeds, heldout cohorts, checkpoints, or owner/runner execution.
"""
import hashlib
import math
from pathlib import Path

SOURCE_BLOB = "16d4b1c691659365251d97d6a5635021b41d080e"
SOURCE = Path(__file__).resolve().parents[1] / "native_p38_conditional_action_choice_dev.py"
CUE_MIN = 0.45
GOAL_DISPLACEMENT = 0.32


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def zero_training_action(goal_direction: int, visible_cue: float) -> int:
    """No gradient, fitted coefficient, environment roll-out, or model state."""
    if goal_direction not in (-1, 1) or not math.isfinite(visible_cue) or abs(visible_cue) < CUE_MIN:
        raise ValueError("P38_SHORTCUT_OUT_OF_SOURCE_DOMAIN")
    return goal_direction * (1 if visible_cue > 0 else -1)


def authored_next_x(x: float, cue: float, action: int) -> float:
    """Restate the SHA-pinned authored P38 physics, independently of source RNG."""
    return x + 0.3 * action * cue + 0.04 * action * math.sin(3 * x) + 0.02 * math.sin(5 * x)


def test_source_is_exact_original_frozen_p38_implementation():
    assert git_blob(SOURCE.read_bytes()) == SOURCE_BLOB


def test_analytical_bound_proves_goal_cue_sign_rule_without_learning():
    # Let k=0.3*cue+0.04*sin(3*x), drift=0.02*sin(5*x).
    # |k|>=0.3*0.45-0.04=0.095 > 0, hence sign(k)=sign(cue).
    # |drift|<=0.02 < goal displacement 0.32.
    # Costs at actions +/-1 differ by 4*k*(drift-goal_displacement*direction).
    # This proves the sign rule is optimal for every real x in P38's domain.
    assert 0.3 * CUE_MIN - 0.04 > 0
    assert 0.02 < GOAL_DISPLACEMENT


def test_no_training_policy_is_perfect_and_fixed_controls_only_half_on_balanced_cues():
    # Independent hand-authored grid, not a seeded/source-generated cohort.
    xs = (-1.0, -0.5, 0.0, 0.5, 1.0)
    cues = (-0.99, -0.75, -0.45, 0.45, 0.75, 0.99)
    for direction in (-1, 1):
        selected = []
        for x in xs:
            for cue in cues:
                target = x + direction * GOAL_DISPLACEMENT
                errors = {a: (authored_next_x(x, cue, a) - target) ** 2 for a in (-1, 1)}
                optimal = min((-1, 1), key=lambda a: errors[a])
                chosen = zero_training_action(direction, cue)
                assert chosen == optimal
                selected.append(optimal)
        assert len(selected) == len(xs) * len(cues)
        assert selected.count(-1) == selected.count(1) == len(selected) // 2


def test_rule_rejects_out_of_domain_inputs():
    for goal, cue in ((0, 0.8), (1, 0), (-1, float("nan")), (1, 0.449999), (1, float("inf"))):
        try:
            zero_training_action(goal, cue)
        except ValueError as exc:
            assert str(exc) == "P38_SHORTCUT_OUT_OF_SOURCE_DOMAIN"
        else:
            raise AssertionError("Source-domain guard unexpectedly accepted input")
