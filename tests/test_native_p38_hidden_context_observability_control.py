"""P38-A1 no-training hidden-context control: history is necessary, not a novel learner.

Independent unseeded analytic source. Neither P38/P38-R1 original science nor
P6/P39 external sources are run. Original P6 hand-coded factual-pair inference
exists already; this is a comparator specification, not a replacement agent.
"""
import math
import pytest

SLOPE = 0.3
TARGET = 0.32


def public_observation():
    # Both modes expose the identical current sensor and action interface.
    return (0.0,)


def factual_outcome(mode: int, nuisance: float, action: int) -> float:
    if mode not in (-1, 1) or action not in (-1, 1) or not math.isfinite(nuisance):
        raise ValueError("P38_A1_INVALID_FACTUAL_SOURCE_INPUT")
    return nuisance + SLOPE * mode * action


def infer_mode_from_completed_facts(facts):
    """Conventional no-training history comparator; never sees hidden mode.

    Each tuple must be an independently completed (executed_action, outcome).
    One action polarity is insufficient when the nuisance intercept is unknown.
    """
    observations = {}
    for action, consequence in facts:
        if action not in (-1, 1) or not math.isfinite(consequence):
            raise ValueError("P38_A1_INVALID_FACTUAL_EVIDENCE")
        observations[action] = consequence
    if len(observations) < 2:
        return None
    contrast = (observations[1] - observations[-1]) / 2
    return (1 if contrast > 0 else -1) if abs(contrast) >= 0.1 else None


def action_with_factual_history(goal: int, facts):
    if goal not in (-1, 1):
        raise ValueError("P38_A1_INVALID_GOAL")
    learned_mode = infer_mode_from_completed_facts(facts)
    return None if learned_mode is None else goal * learned_mode


def test_identical_current_sensor_proves_snapshot_decision_no_better_than_half():
    # Both hidden modes are equiprobable and reward opposite actions.
    assert public_observation() == public_observation()
    for goal in (-1, 1):
        best_actions = tuple(goal * mode for mode in (-1, 1))
        for history_blind_choice in (-1, 1):
            assert sum(history_blind_choice == best for best in best_actions) == 1


def test_first_factual_observation_is_also_ambiguous_with_unknown_intercept():
    # A single +1 experiment yields identical observation in two opposite modes.
    # True intercept is unobserved and may differ between worlds.
    cases = ((-1, SLOPE), (1, -SLOPE))
    first_facts = []
    for mode, nuisance in cases:
        y = factual_outcome(mode, nuisance, 1)
        first_facts.append(((1, y),))
        assert action_with_factual_history(1, [(1, y)]) is None
    assert first_facts[0] == first_facts[1] == ((1, 0.0),)


def test_two_real_opposite_action_results_resolve_mode_and_both_goals():
    for mode, nuisance in ((-1, SLOPE), (1, -SLOPE)):
        facts = [(1, factual_outcome(mode, nuisance, 1))]
        assert infer_mode_from_completed_facts(facts) is None
        facts.append((-1, factual_outcome(mode, nuisance, -1)))
        assert infer_mode_from_completed_facts(facts) == mode
        for goal in (-1, 1):
            action = action_with_factual_history(goal, facts)
            target = nuisance + goal * TARGET
            losses = {a: (factual_outcome(mode, nuisance, a) - target)**2 for a in (-1, 1)}
            assert action == min((-1, 1), key=lambda a: losses[a])


def test_wrong_action_credit_inverts_conclusion_and_cold_history_is_required():
    for mode, nuisance in ((-1, SLOPE), (1, -SLOPE)):
        factual = [(1, factual_outcome(mode, nuisance, 1)),
                   (-1, factual_outcome(mode, nuisance, -1))]
        sham = [(-a, y) for a, y in factual]
        assert infer_mode_from_completed_facts(sham) == -mode
        cold_restored_facts = tuple((int(a), float(y)) for a, y in factual)
        assert infer_mode_from_completed_facts(cold_restored_facts) == mode
        assert action_with_factual_history(1, ()) is None


def test_nonfactual_or_invalid_inputs_fail_closed():
    for invalid in ((0, 0.2), (1, float('nan')), (1, float('inf'))):
        with pytest.raises(ValueError, match='P38_A1_INVALID_FACTUAL_EVIDENCE'):
            infer_mode_from_completed_facts((invalid,))
    with pytest.raises(ValueError, match='P38_A1_INVALID_GOAL'):
        action_with_factual_history(0, ())
