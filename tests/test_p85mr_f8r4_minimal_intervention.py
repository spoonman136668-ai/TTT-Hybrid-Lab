"""F8-R4 math: same evidence budget, one informative intervention, cold bit."""
import inspect

import pytest

from experiments import p85mr_f8r4_minimal_intervention as f8
from experiments.p85mr_f8r3_counterpolicy_identifiability import (
    behavior, synthetic_effect,
)


def test_frozen_source_neutral_protocol_precedes_experiment():
    f = f8.locked()
    assert f["conditions"]["query_budget_per_arm"] == 1
    assert f["boundaries"]["no_original_F7_firstuse_or_any_replay"]
    assert f["boundaries"]["no_new_external_actions"]
    assert f["conditions"]["thresholds"]["active_transfer_goal_action_accuracy"] == "2/2"


def test_equal_one_action_budget_discriminates_only_off_policy():
    for hidden in (-1, 1):
        active, passive = f8.OneBitBelief(), f8.OneBitBelief()
        assert behavior(0.2) == 1
        active.observe(.2, 0, synthetic_effect(.2, 0, hidden))
        passive.observe(.2, 1, synthetic_effect(.2, 1, hidden))
        assert active.observations == passive.observations == 1
        assert active.sign == hidden
        assert passive.sign is None
        with pytest.raises(ValueError, match="ACTION_EVIDENCE_BUDGET_EXHAUSTED"):
            active.observe(.2, 1, synthetic_effect(.2, 1, hidden))
        with pytest.raises(ValueError, match="NO_IDENTIFIED_WORLD_TO_PERSIST"):
            passive.cold_export()


def test_cold_one_bit_selection_transfers_to_different_state():
    receipt = f8.evaluate()
    assert receipt["active_transfer_action_accuracy"] == "2/2"
    assert receipt["passive_transfer_action_accuracy"] == "1/2"
    assert receipt["original_F7_or_external_source_actions"] == 0
    assert receipt["scientific_pass"] is False
    assert [x["active_cold_recovered_action"] for x in receipt["evaluation_rows"]] == [0, 1]
    assert [x["passive_action"] for x in receipt["evaluation_rows"]] == [0, 0]


def test_unknown_mechanism_and_cold_receipt_denials():
    b = f8.OneBitBelief()
    assert b.choose(-.2, .2) == 0
    with pytest.raises(ValueError, match="OUTCOME_INCOMPATIBLE_WITH_BOTH_WORLDS"):
        b.observe(.2, 0, 25.)
    assert b.observations == 0
    with pytest.raises(ValueError, match="INVALID_EFFECT"):
        b.observe(.2, 0, float("nan"))
    with pytest.raises(ValueError, match="INVALID_ACTION"):
        b.observe(.2, True, 0.)
    with pytest.raises(ValueError, match="INVALID_STATE"):
        b.observe(float("inf"), 0, 0.)
    for forged in ({}, {"schema": f8.BELIEF_SCHEMA, "world_bit": 2},
                   {"schema": f8.BELIEF_SCHEMA, "world_bit": True},
                   {"schema": f8.BELIEF_SCHEMA, "world_bit": 0, "secret": 1}):
        with pytest.raises(ValueError, match="INVALID_COLD_BIT_RECEIPT"):
            f8.OneBitBelief.cold_restore(forged)


def test_factual_query_is_not_hidden_world_leak():
    for hidden in (-1, 1):
        b = f8.OneBitBelief()
        value = synthetic_effect(.2, 0, hidden)
        b.observe(.2, 0, value)
        receipt = b.cold_export()
        restored = f8.OneBitBelief.cold_restore(receipt)
        assert restored.sign == hidden
        assert restored.choose(-.2, .2) == (0 if hidden == -1 else 1)
    with pytest.raises(PermissionError, match="EXTERNAL_SCIENTIFIC_FIRST_USE_NOT_AUTHORIZED"):
        f8.refuse_external_first_use()


def test_no_real_simulator_owner_runner_or_training_side_effects():
    code = inspect.getsource(f8)
    for forbidden in ("import causal_gym", "from causal_gym", "env.do(",
                      "env.reset(", "workflow_dispatch(", "torch.load(",
                      "subprocess.run(", "requests.get(", "urlopen(",
                      "optimizer.step("):
        assert forbidden not in code
