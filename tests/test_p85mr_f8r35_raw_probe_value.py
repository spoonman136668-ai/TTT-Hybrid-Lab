"""F8-R35 genuine raw neural factual value learning controls."""
import inspect
import math

import pytest
import torch

from experiments import p85mr_f8r35_raw_probe_value as x

torch.set_num_threads(1)


def test_frozen_protocol_and_authority():
    f=x.locked()
    assert f["parent"]["head"]=="cdec26f67397d8bd96f419b649321fd0da68f8b6"
    assert f["arms"]["model_count"]==7
    assert f["optimization"]["updates_each"]==512
    assert not any(f["authority"].values())
    with pytest.raises(PermissionError,match="EXTERNAL_FIRSTUSE"):
        x.refuse_external_firstuse()


def test_factual_pairs_are_separately_selected_and_fit_only():
    fit=x.r33.fit_cohort()
    assert x._factual_pair_integrity(fit)==1536
    a,e,y=x._batch(fit,list(range(16)))
    assert a.shape==(32,14,16)
    assert e.shape==a.shape and y.shape==(32,)
    for i in range(0,len(y),2):
        assert torch.equal(a[i,:-1],a[i+1,:-1])
        assert torch.equal(e[i,:-1],e[i+1,:-1])
        assert not torch.equal(a[i,-1],a[i+1,-1])
        assert math.isfinite(float(y[i])) and math.isfinite(float(y[i+1]))
    with pytest.raises(ValueError,match="UNREGISTERED_TRAIN_PAIR"):
        x._batch(fit,[1152]*16)


def test_model_masks_current_return_and_erased_probe_effect():
    fit=x.r33.fit_cohort()
    a,e,_=x._batch(fit,list(range(16)))
    model=x.ProbeValueMLP(x.SEEDS[0])
    model.eval()
    with torch.no_grad():
        baseline=model(a,e)
        poison=e.clone()
        poison[:,-1,0]=.8
        assert torch.equal(baseline,model(a,poison))
        erased=model(a,e,erase_fifth=True)
        neg=e.clone()
        neg[:,4,:].zero_()
        assert torch.equal(erased,model(a,neg))
    assert baseline.shape==(32,)
    with pytest.raises(ValueError,match="INVALID_FACTUAL_INPUT"):
        model(a[:,:2],e[:,:2])
    w=x.r33.worlds("ood")[0]
    aa,ee=x.r33.candidate_inputs(w,"active_information")
    assert aa.shape==(2,22,16)
    assert model(aa,ee).shape==(2,)


def test_objective_uses_only_selected_factual_cost_labels():
    fit=x.r33.fit_cohort()
    a,e,y=x._batch(fit,list(range(16)))
    model=x.ProbeValueMLP(x.SEEDS[0])
    out=model(a,e)
    strong=x.objective(out,y,"compact_strong")
    weak=x.objective(out,y,"compact_weak")
    assert bool(torch.isfinite(strong)) and bool(torch.isfinite(weak))
    assert strong!=weak
    with pytest.raises(ValueError,match="UNREGISTERED_OBJECTIVE"):
        x.objective(out,y,"unseen")


def test_frozen_train_all_arms_and_negative_scientific_claim():
    result=x.run()
    assert result["actual_optimizer_updates"]==3584
    assert result["neural_model_instances"]==7
    assert result["factual_training_pairs"]==1152
    assert result["factual_holdout_pairs"]==384
    assert not result["learned_native_hybrid"]
    assert not result["independent_mechanism_transfer"]
    assert not result["independent_scientific_pass"]
    assert result["original_external_effects"]==0
    assert not result["spent_F7_replay"] and result["deployment"]=="NONE"
    assert isinstance(result["judgment"]["source_neutral_development_pass"],bool)
    for model in result["models"]:
        assert model["optimizer_updates"]==512
        assert model["initial_weight_sha256"]!=model["trained_weight_sha256"]
        assert model["cold_restore_actions_equal"]
        assert [t["updates"] for t in model["checkpoints"]]==[128,256,512]
        assert all(math.isfinite(c["heldout_fit_mse_on_objective_scale"])
                   for c in model["checkpoints"])
        for panel in x.PANELS:
            z=model["score"][panel]
            for policy in x.POLICIES:
                j=z["policies"][policy]
                assert len(j["choices"])==48
                assert len(j["pergroup_regret"])==12
                assert math.isfinite(j["mean_regret"])
            assert len(z["erased_active"]["choices"])==48
    x.retain_ci_receipt(result)


def test_learning_does_not_use_evaluator_hidden_q_or_oracle():
    for fn in (x.ProbeValueMLP.forward,x._batch,x.objective,x._train,
               x._validation):
        src=inspect.getsource(fn)
        for forbidden in ("q_evaluator_only","evaluate_regret_only(",
                          "r32.after(", "hidden_sign", "oracle_label"):
            assert forbidden not in src
    src=inspect.getsource(x)
    for forbidden in ("workflow_dispatch(","subprocess.run(","urlopen(",
                      "requests.get(","from causal_gym","import causal_gym"):
        assert forbidden not in src
