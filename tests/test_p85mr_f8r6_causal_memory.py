"""F8-R6 frozen, learned causal memory from real *toy* selected action effects."""
from pathlib import Path
import math

import pytest
import numpy as np

from experiments import p85mr_f8r6_causal_memory as r6


def test_f8r6_frozen_scope_and_sign_blind_training():
    freeze=r6.locked()
    assert freeze["parent"]["head_sha"]=="90570bd0f7fa13af3898001ad612ed13f255b64e"
    assert freeze["authority"]["external_source_effects"]==0
    assert freeze["authority"]["new_F8_CausalGymnasium_firstuse"] is False
    logs=r6.training_episodes()
    assert len(logs)==24 and all(len(ep)==12 for ep in logs)
    assert all(len(event)==3 and event[1] in (0,1) for ep in logs for event in ep)
    fitted=r6.fit_controls(logs)
    assert fitted["experts"].shape==(2,2)
    assert fitted["pooled"].shape==(2,)
    assert fitted["experts"][0,1]<fitted["experts"][1,1]
    assert len(r6.fixed_pairs())==48
    # No numerical-regression bias may masquerade as a causal observation.
    p=r6.infer(fitted["experts"],.2,1,.1)
    assert p==(0.5,0.5)
    a=r6.infer(fitted["experts"],.2,0,r6._effect(.2,0,1))
    assert a!=(0.5,0.5)
    assert r6.training_episodes()==logs


def test_f8r6_explicit_unselected_action_and_bad_data_blocked():
    with pytest.raises(ValueError,match="INVALID_EXECUTED_ACTION"):
        r6.feature(.2,2)
    with pytest.raises(ValueError,match="INVALID_EXECUTED_ACTION"):
        r6.feature(.2,True)
    with pytest.raises(ValueError,match="INVALID_STATE"):
        r6.policy(float("nan"))
    with pytest.raises(ValueError,match="INVALID_FIT_INFORMATION"):
        r6._fit(np.zeros((2,3)),np.zeros(2))
    with pytest.raises(ValueError,match="UNEXPECTED_FIT_COHORT"):
        r6.fit_controls([])
    with pytest.raises(ValueError,match="INVALID_COLD_PLANNER_BELIEF"):
        r6.plan(np.zeros((2,2)),(2.,-1.),0.,0.)
    with pytest.raises(PermissionError,match="REAL_FIRSTUSE_AND_F7_REPLAY_NOT_AUTHORIZED"):
        r6.refuse_external("CausalGym")
    source=Path(r6.__file__).read_text()
    for prohibited in ("import causal_gym","from causal_gym","env.do(",
       "env.reset(","torch.load(","requests.get(","workflow_dispatch(",
       "subprocess.run("):
        assert prohibited not in source


def test_f8r6_full_trained_control_comparison_and_cold_retention():
    result=r6.run()
    assert result["status"]=="F8R6_SOURCE_NEUTRAL_DEV_COMPLETE"
    assert result["fresh_toy_fit_effects"]==288
    assert result["score_probes_and_selected_effects_per_arm"]==192
    assert result["original_external_effects"]==0
    assert result["action_budget_matched"]
    assert not result["independent_scientific_pass"]
    assert not result["trained_Hybrid_scientific_gain"]
    assert result["cold_restoration"]["A_to_B_to_A_unchanged"]
    assert result["candidate_dev_pass"] in (True,False)
    assert math.isfinite(result["negative_wrong_action_credit"]["wrong_to_correct_ratio"])
    assert len(result["controls"])==4
    assert all(c["factual_selected_actions"]==192 and
               c["real_external_actions"]==0
               for c in result["controls"].values())
    r6.store_ci_receipt(result)
