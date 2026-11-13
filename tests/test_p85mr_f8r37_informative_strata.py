"""F8-R37 prospective factual identifiability train/sampling and science fences."""
import inspect
import math

import pytest
import torch

from experiments import p85mr_f8r37_informative_strata as x

torch.set_num_threads(1)


def test_frozen_preregistration_pretraining_amendment_and_authority_fences():
    f,a=x.locked()
    assert f["stage"]=="FROZEN_BEFORE_IMPLEMENTATION_TRAINING_AND_RESULTS"
    assert f["parent"]["head"]=="c3fbd1c1ca5bbe474ad2d7f2e9c8d3cbd6c262f9"
    assert a["score_override"]["original_value"]==.14
    assert a["score_override"]["dev_min"]==.14
    assert a["score_override"]["ood_min"]==.12
    assert not any(f["authority"].values())
    assert not any(a["authority"].values())
    with pytest.raises(PermissionError,match="EXTERNAL_FIRSTUSE"):
        x.refuse_external_firstuse()


def test_public_identifiability_selection_is_balanced_and_same_fit_only():
    rows=x.r33.fit_cohort()
    label=x.r36.fit_pairs(rows)
    pool=x.selection_pools(rows,label)
    assert len(rows)==3072 and len(label)==1536
    assert len(pool["all"])==1152
    assert len(pool["informative"])==576
    assert len(pool["noninformative"])==576
    assert len(pool["heldout_informative"])==192
    assert len(pool["heldout_noninformative"])==192
    assert [sum(label[i]==j for i in pool["informative"]) for j in (0,1)]==[288,288]
    assert all(x.informative_id(rows,i) for i in pool["informative"])
    assert all(not x.informative_id(rows,i) for i in pool["noninformative"])
    assert not set(pool["all"]) & set(pool["heldout_informative"])
    with pytest.raises(ValueError,match="UNREGISTERED_PAIR"):
        x.informative_id(rows,1536)


def test_fit_only_baseline_donor_negative_decorrelation_and_selected_return_labels():
    rows=x.r33.fit_cohort()
    labels=x.r36.fit_pairs(rows)
    pool=x.selection_pools(rows,labels)
    baseline=x.fit_only_offset(rows,pool)
    assert math.isfinite(baseline) and -.6<baseline<-.1
    donor,phi=x._donor_map(rows,pool)
    assert len(donor)==576
    assert abs(phi)<=.1
    assert all(x.informative_id(rows,i) for i in donor)
    assert len(set(donor.values()))==576
    # A shuffled negative-control donor must be a DIFFERENT factual trajectory.
    assert all(i!=j for i,j in donor.items())
    sample=pool["informative"][:16]
    a,e,y=x._batch(rows,labels,sample,donor_map=donor)
    raw_a,raw_e,raw_y=x._batch(rows,labels,sample)
    assert a.shape==e.shape==(32,14,16)
    assert y.shape==(16,)
    assert torch.equal(a,raw_a) and torch.equal(y,raw_y)
    assert all(torch.equal(e[2*i,4,:],e[2*i+1,4,:])
               for i in range(16))
    assert any(not torch.equal(e[2*i,4,:],raw_e[2*i,4,:])
               for i in range(16))
    with pytest.raises(ValueError,match="SHUFFLE_REQUIRES_INFORMATIVE"):
        x._batch(rows,labels,pool["noninformative"][:16],
                 donor_map=donor)


def test_preaction_return_effect_masks_and_fifth_erasure():
    w=x.worlds("dev")[0]
    a,e=x.candidate_inputs(w,"active_information")
    assert a.shape==e.shape==(2,14,16)
    assert torch.count_nonzero(e[:,-1,:])==0
    rows=x.r33.fit_cohort()
    labels=x.r36.fit_pairs(rows)
    pool=x.selection_pools(rows,labels)
    baseline=x.fit_only_offset(rows,pool)
    for arm in ("raw_informative","gated_informative"):
        model=x.model_for(arm,985201,baseline).eval()
        with torch.no_grad():
            original=x._pred(model,arm,a,e)
            poison=e.clone()
            poison[:,-1,0]=.8
            assert torch.allclose(original,x._pred(model,arm,a,poison),atol=0,rtol=0)
            erased=x._pred(model,arm,a,e,erase=True)
            removed=e.clone()
            removed[:,4,:].zero_()
            assert torch.allclose(erased,x._pred(model,arm,a,removed),atol=0,rtol=0)
    assert original.shape==(2,)
    ood_a,ood_e=x.candidate_inputs(x.worlds("ood")[0],"active_information")
    assert ood_a.shape==ood_e.shape==(2,22,16)
    assert x._pred(model,arm,ood_a,ood_e).shape==(2,)


def test_new_score_worlds_have_hidden_sign_decision_opposites_and_evaluator_only():
    for panel in x.PANELS:
        ww=x.worlds(panel)
        assert len(ww)==64
        assert len({w["group"] for w in ww})==16
        sign_pairs=0
        for i in range(0,64,2):
            a=ww[i]
            b=ww[i+1]
            assert a["b_history"]==b["b_history"]
            assert a["q_evaluator_only"]==-b["q_evaluator_only"]
            # The mathematical simulator may be used only for evaluation,
            # never as a labeled model input.
            ca=[(x.r33.r32.after(a["return_state"],v,
                 a["public_context"],a["q_evaluator_only"])-1)**2
                 for v in (0,1)]
            cb=[(x.r33.r32.after(b["return_state"],v,
                 b["public_context"],b["q_evaluator_only"])-1)**2
                 for v in (0,1)]
            sign_pairs += int(ca.index(min(ca))!=cb.index(min(cb)))
        assert sign_pairs==32


def test_full_original_frozen_neural_comparisons_preserve_negative_results():
    result=x.run()
    assert result["schema"]==x.SCHEMA
    assert result["fit_factual_episodes"]==3072
    assert result["train_factual_pairs"]==1152
    assert result["train_informative_pairs"]==576
    assert result["validation_informative_pairs"]==192
    assert result["model_instances"]==13
    assert result["optimizer_updates"]==6656
    assert result["original_external_effects"]==0
    assert not result["spent_F7_replay"]
    assert not result["independent_scientific_pass"]
    assert not result["new_mechanism_transfer"]
    assert result["deployment"]=="NONE"
    assert isinstance(result["judgment"]["gated_informative_development_pass"],bool)
    for row in result["models"]:
        assert row["updates"]==512
        assert row["initial_sha256"]!=row["trained_sha256"]
        assert row["cold_restore"]
        assert [r["updates"] for r in row["checkpoints"]]==[128,256,512]
        assert all(math.isfinite(step["heldout"][c]["choice_CE"])
                   for step in row["checkpoints"]
                   for c in ("informative","noninformative"))
        for panel in x.PANELS:
            results=row["panels"][panel]
            assert set(results["policies"])==set(x.POLICIES)
            assert len(results["erased_active"]["choices"])==64
            for policy in x.POLICIES:
                score=results["policies"][policy]
                assert len(score["choices"])==64
                assert len(score["pergroup_regret"])==16
                assert 0<=score["sign_pair_flips"]<=32
                assert math.isfinite(score["mean_regret"])
    x.retain_ci_receipt(result)


def test_no_evaluator_hidden_sign_or_unselected_return_label_in_training():
    for fn in (x.informative_id,x.selection_pools,x.fit_only_offset,
               x._batch,x._pred,x._validation,x.train_arm):
        s=inspect.getsource(fn)
        for forbidden in ("q_evaluator_only","regret_evaluator_only(",
                          "r33.r32.after(","oracle_label","hidden_sign"):
            assert forbidden not in s
    source=inspect.getsource(x)
    for bad in ("workflow_dispatch(","subprocess.run(","requests.get(",
                "urlopen(","from causal_gym","import causal_gym"):
        assert bad not in source
