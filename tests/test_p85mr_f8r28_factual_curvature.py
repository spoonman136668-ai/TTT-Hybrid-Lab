"""R28: actual nonlinear factual intervention learning vs equal-budget probes."""
import inspect
import math

import pytest
import torch

from experiments import p85mr_f8r28_factual_curvature as x
torch.set_num_threads(1)


def test_prespecified_exact_freeze_source_owner_and_seeds():
    f=x.locked()
    assert f["stage"]=="FROZEN_PRE_IMPLEMENTATION_AND_NUMERIC_OUTCOME"
    assert f["world"]["FIT_expected_effects"]==57344
    assert f["training"]["total_optimizer_updates"]==1920
    assert f["score"]["independent_scientific_pass"] is False
    assert all(z is False for z in f["authority"].values())
    with pytest.raises(PermissionError,match="FIRSTUSE_NOT_AUTHORIZED"):
        x.refuse_original_external_firstuse()


def test_full_factorial_fit_has_real_fifth_and_return_selected_effects_only():
    rows=x.fit_cohort()
    assert x.fit_audit(rows)=={
        "full_factored_factual_sessions":4096,
        "actual_selected_factual_effects":57344,
        "each_episode_has_only_one_fifth_action":True,
        "each_episode_has_only_one_return_action":True}
    assert len(rows)==4096
    assert {z["selected_fifth"] for z in rows}=={0,1}
    assert {z["selected_return"] for z in rows}=={0,1}
    assert {z["return_state"] for z in rows}=={-.7,.7}
    assert {tuple(z["q_evaluator_only"]) for z in rows}=={
        (-.55,-.55),(-.55,.55),(.55,-.55),(.55,.55)}
    for z in rows[:8]:
        assert z["action"].shape==(14,16)
        assert z["effect"].shape==(14,16)
        assert all(math.isfinite(float(v)) for v in z["effect"].flatten())


def test_heldout_worlds_independent_and_matched_A_twin_B_only():
    for panel in x.PANELS:
        rows=x.worlds(panel)
        audit=x.audit_worlds(rows,panel)
        assert audit["paired_new_groups"]==12
        assert audit["B_history_identical_within_swapped_A_twins"]
        assert audit["first_return_effect_unobserved"]
        assert all(len(z["B"])==x.B_STEPS[panel] for z in rows)
        assert all(len(x.r27._plans(rows,policy))==96 for policy in x.POLICIES)
    with pytest.raises(ValueError,match="UNREGISTERED_NEW_WORLD"):
        x.new_world("id",12,0)


def test_factual_fifth_has_identifying_information_that_early_four_lack():
    for panel in ("id","ood"):
        a,b=x.worlds(panel)[:2]
        for z in (a,b):
            early=[]
            for state,action in x.PROBES:
                early.append(x.r27.after_A(state,action,z["b"],z["k"],z["q"]))
                assert early[-1]==pytest.approx(
                    x.r27.after_A(state,action,z["b"],z["k"],(0.,0.)))
            for act in (0,1):
                assert x.r27.after_A(.7,act,z["b"],z["k"],z["q"])!=pytest.approx(
                    x.r27.after_A(.7,act,z["b"],z["k"],(0.,0.)))


def test_conventional_and_native_early_fifth_heldout_prediction_causal_barrier():
    rows=x.worlds("id")
    for arm in x.ARMS:
        model=x.r25.r22.model_for(arm,985201).cpu().eval()
        plans=x.r27._plans(rows,"fixed_action0")
        choices=x._choices(arm,model,plans[:4],False)
        erased=x._choices(arm,model,plans[:4],True)
        assert len(choices)==len(erased)==4
        assert all(z in (0,1) for z in choices+erased)


def test_5_real_trained_models_frozen_updates_and_exact_cold_replay():
    result=x.run()
    assert result["actual_training_models"]==5
    assert result["total_optimizer_updates"]==1920
    assert result["factual_fifth_and_return_selected_fit"]["full_factored_factual_sessions"]==4096
    assert result["factual_fifth_and_return_selected_fit"]["actual_selected_factual_effects"]==57344
    assert result["decision"]["native_DEV_pass"] in (True,False)
    assert len(result["decision"]["native_seed_decisions"])==3
    assert result["original_external_effects"]==0
    assert result["spent_F7_replay"] is False
    assert result["scientific_pass"] is False
    assert result["deployment"]=="NONE"
    for row in result["trained_models"]:
        assert row["updates"]==384
        assert row["parameters"]>0
        assert row["initial_hash"]!=row["trained_hash"]
        assert row["cold_checkpoint_actions_equal"]
        for panel in x.PANELS:
            for policy in x.POLICIES:
                z=row["panels"][panel][policy]
                assert z["actual_selected_actions"]==96
                assert z["factual_fifth_probe_effects"]==96
                assert len(z["pergroup_regret"])==12
                assert math.isfinite(z["mean_regret"])
    x.retain_ci_receipt(result)


def test_no_external_source_new_scheduler_or_model_promotion():
    src=inspect.getsource(x)
    for forbidden in ("import causal_gym","from causal_gym","env.reset(",
                      "env.do(","ctf_do(","workflow_dispatch(",
                      "subprocess.run(","requests.get(","urlopen("):
        assert forbidden not in src
