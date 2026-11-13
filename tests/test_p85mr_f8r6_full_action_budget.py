"""F8-R6 exact proposed original-source effect ceiling; no first-use."""
import inspect

import pytest
from experiments import p85mr_f8r6_full_action_budget as budget


def test_frozen_original_custody_and_total_budget():
    f = budget.locked()
    assert f["status"] == "FROZEN_PROPOSAL_NOT_APPROVED_FOR_ANY_ORIGINAL_SOURCE_FIRSTUSE"
    assert f["phase_first_use"]["calibration"]["approved"] is False
    assert f["phase_first_use"]["sealed_model_selected_test"]["approved"] is False
    receipt = budget.proposed_receipt()
    assert receipt["sessions_proposed"] == 480
    assert receipt["physical_factual_actions_worst_case"] == 46080
    assert receipt["session_counts"] == {
        "calibration": 48, "fit": 64, "selector": 16, "test": 352}
    assert receipt["original_factual_actions_executed_now"] == 0
    assert receipt["future_CAL_OOD_effects_authorized"] is False
    assert receipt["scientific_pass"] is False


def test_paired_seed_repetitions_are_fresh_distinct_proposed_sessions():
    sessions = budget.planned_sessions()
    identity = [x["identity"] for x in sessions]
    assert len(set(identity)) == 480
    test = [x for x in sessions if x["phase"] == "test"]
    assert len(test) == 352
    assert len(set(x["source_seed"] for x in test)) == 16
    for world in range(16):
        paired = [s for s in test if s["world_index"] == world]
        assert len(paired) == 22
        assert len(set(s["identity"] for s in paired)) == 22
    assert len(set(s["source_seed"] for s in sessions)) == 144
    assert not any(1821000000 <= x["source_seed"] < 1821000128
                   for x in sessions)


def test_fabricated_maximum_counter_bounds_and_scoped_denials():
    s = budget.planned_sessions()
    all_at_cap = [{"classification": "FABRICATED_DEV_COUNTER_ONLY",
                   "identity": row["identity"],
                   "attempted_factual_actions": 96} for row in s]
    result = budget.check_fabricated_counter_panel(all_at_cap)
    assert result["sessions_counted"] == 480
    assert result["factual_actions_counted"] == 46080
    assert not result["genuine_owner_custody_verified"]
    assert not result["source_first_use_authorized"]
    with pytest.raises(PermissionError, match="OWNER_FIRSTUSE_NOT_AUTHORIZED"):
        budget.refuse_original_source_first_use()


def test_fail_closed_for_oob_duplicated_nonfactual_and_unknown_replay():
    ident = budget.planned_sessions()[0]["identity"]
    good = {"classification": "FABRICATED_DEV_COUNTER_ONLY",
            "identity": ident, "attempted_factual_actions": 1}
    for different in (
            {"classification": "ORIGINAL_SOURCE_RECEIPT"},
            {"identity": "unknown-world-session"},
            {"attempted_factual_actions": -1},
            {"attempted_factual_actions": 97},
            {"attempted_factual_actions": True},
            {"attempted_factual_actions": 1.0}):
        corrupt = dict(good)
        corrupt.update(different)
        with pytest.raises(ValueError, match="DUPLICATE_OR_EXCESS"):
            budget.check_fabricated_counter_panel([corrupt])
    with pytest.raises(ValueError, match="DUPLICATE_OR_EXCESS"):
        budget.check_fabricated_counter_panel([good, good])
    with pytest.raises(ValueError, match="MALFORMED_TEST_COUNTER_PANEL"):
        budget.check_fabricated_counter_panel(None)


def test_no_original_external_source_or_implicit_owner_execution():
    code = inspect.getsource(budget)
    for forbidden in ("import causal_gym", "from causal_gym", "env.do(",
                      "env.reset(", "workflow_dispatch(", "subprocess.run(",
                      "torch.load(", "requests.get(", "urlopen(",
                      "optimizer.step(", "os.system("):
        assert forbidden not in code
