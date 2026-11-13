"""R34 frozen neural factual-memory objective, leakage and science fences."""
import inspect
import math

import pytest
import torch

from experiments import p85mr_f8r34_learned_factual_memory as x

torch.set_num_threads(1)


def test_frozen_source_and_rejected_original_external_firstuse():
    protocol=x.locked()
    assert protocol["stage"]=="FROZEN_BEFORE_IMPLEMENTATION_TRAINING_OR_RESULTS"
    assert protocol["parent"]["head"]=="0f5e7381037b8dfdd365be222a5159f2e42fe39f"
    assert protocol["score"]["native_seed_both_panels_min"]==2
    assert not any(protocol["authority"].values())
    with pytest.raises(PermissionError,match="EXTERNAL_FIRSTUSE"):
        x.refuse_external_firstuse()


def test_genuine_paired_selected_factual_return_without_oracle_input():
    fit=x.r33.fit_cohort()
    assert len(fit)==3072
    assert set(fit[0])=={"action","effect","physical_target","direct_target"}
    a,e,y=x.fit_batch(fit,[0,1,2,3])
    assert a.shape==(8,14,16) and e.shape==a.shape and y.shape==(8,)
    assert x.FIT_PAIRS==1152
    assert all(torch.equal(a[i][:-1],a[i+1][:-1])
               for i in (0,2,4,6))
    assert all(torch.equal(e[i][:-1],e[i+1][:-1])
               for i in (0,2,4,6))
    assert torch.isfinite(y).all()
    with pytest.raises(ValueError,match="INVALID_TRAIN_PAIR"):
        x.fit_batch(fit,[x.FIT_PAIRS])


def test_last_return_effect_mask_never_influences_preaction_predictions():
    fit=x.r33.fit_cohort()
    a,e,target=x.fit_batch(fit,[0,1])
    model=x.FactualMemory("native_factual_memory",x.SEEDS[0])
    model.eval()
    with torch.no_grad():
        first=model(a,e)
        changed=e.clone()
        changed[:,-1,:]=torch.randn_like(changed[:,-1,:])*3.
        second=model(a,changed)
    assert first.shape==(4,)
    assert torch.allclose(first,second,atol=0.,rtol=0.)


def test_frozen_new_model_training_and_negative_scientific_claims():
    result=x.run()
    assert result["optimizer_updates"]==1536
    assert result["neural_model_instances"]==6
    assert result["factual_fit_episodes"]==3072
    assert result["train_factual_episodes"]==2304
    assert result["validation_factual_episodes"]==768
    assert result["new_external_effects"]==0
    assert not result["spent_F7_replay"]
    assert not result["capacity_FLOPs_matched"]
    assert not result["independent_scientific_pass"]
    assert result["deployment"]=="NONE"
    assert isinstance(result["judgment"]["source_neutral_native_development_pass"],bool)
    for row in result["models"]:
        assert row["optimizer_updates"]==256
        assert row["trained_weight_hash"]!=row["initial_weight_hash"]
        assert row["cold_restore_actions_equal"]
        assert [x["updates"] for x in row["validation_trace"]]==[128,256]
        for p in x.PANELS:
            score=row["panels"][p]
            assert math.isfinite(score["erasure_regret_penalty"])
            assert math.isfinite(score["backbone_zeroed_regret_penalty"])
            assert len(score["policies"])==4
            for k in x.POLICIES:
                policy=score["policies"][k]
                assert len(policy["choices"])==48
                assert len(policy["pergroup_regret"])==12
                assert 0<=policy["sign_pair_flips"]<=24
    x.retain_ci_receipt(result)


def test_factual_only_objective_no_evaluator_leakage_or_new_external_dispatch():
    for fn in (x.fit_batch,x.factual_pair_loss,x.validation_mse,
               x.FactualMemory.forward,x.train_arm):
        code=inspect.getsource(fn)
        for word in ("q_evaluator_only","evaluate_regret_only(",
                     "oracle_label","hidden_sign"):
            assert word not in code
    source=inspect.getsource(x)
    for banned in ("workflow_dispatch(","subprocess.run(","requests.get(",
                   "urlopen(","import causal_gym","from causal_gym"):
        assert banned not in source


def test_ood_long_B_history_is_admitted_without_current_effect_leak():
    w=x.r33.worlds("ood")[0]
    a,e=x.r33.candidate_inputs(w,"active_information")
    assert a.shape==(2,22,16) and e.shape==a.shape
    model=x.FactualMemory("history_only",x.SEEDS[0])
    model.eval()
    with torch.no_grad():
        before=model(a,e)
        modified=e.clone()
        modified[:,-1,:]=7.
        after=model(a,modified)
    assert before.shape==(2,)
    assert torch.equal(before,after)
