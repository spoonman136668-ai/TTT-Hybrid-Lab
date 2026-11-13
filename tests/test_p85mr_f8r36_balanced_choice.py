"""R36: freeze, no evaluator-choice leakage, factual-label balance, neural test."""
import inspect
import math
import pytest
import torch

from experiments import p85mr_f8r36_balanced_choice as x

torch.set_num_threads(1)


def test_frozen_limits_and_authority_no_external_firstuse():
    f=x.locked()
    assert f["parent"]["sha"]=="99dcb05cc0e16d9da33facd5482daaa9f5e2221d"
    assert f["stage"]=="FROZEN_BEFORE_IMPLEMENTATION_TRAINING_OR_RESULTS"
    assert f["development"]["raw_seed_pass_min"]==2
    assert f["development"]["native_seed_pass_min"]==2
    assert not any(f["authority"].values())
    with pytest.raises(PermissionError,match="EXTERNAL_FIRSTUSE"):
        x.refuse_external_firstuse()


def test_balanced_independently_selected_factual_training_choice_pairs():
    fit=x.r33.fit_cohort()
    winners=x.fit_pairs(fit)
    assert len(winners)==1536
    assert winners[:1152].count(0)==576
    assert winners[:1152].count(1)==576
    a,e,y=x.batch(fit,winners,list(range(16)))
    assert a.shape==(32,14,16) and e.shape==a.shape
    assert y.shape==(16,) and y.dtype==torch.long
    for i in range(16):
        assert torch.equal(a[2*i,:-1],a[2*i+1,:-1])
        assert torch.equal(e[2*i,:-1],e[2*i+1,:-1])
        assert not torch.equal(a[2*i,-1],a[2*i+1,-1])
    with pytest.raises(ValueError,match="UNREGISTERED_TRAIN_BATCH"):
        x.batch(fit,winners,[1152]*16)


def test_new_score_worlds_independent_identifiable_and_length():
    for panel in x.PANELS:
        ww=x.worlds(panel)
        assert len(ww)==64
        for g in range(16):
            for ctx in (0,1):
                minus=ww[g*4+2*ctx]
                plus=ww[g*4+2*ctx+1]
                assert minus["b_history"]==plus["b_history"]
                assert minus["q_evaluator_only"]==-plus["q_evaluator_only"]
                # Opposite signs demand opposite optimal return actions.
                a=[x.regret_evaluator_only([w]*64,[act]*64)[0]
                   for w in (minus,plus) for act in (0,1)]
                assert (a[0]<a[1]) != (a[2]<a[3])
        actions,effects=x.candidate_inputs(ww[0],"active_information")
        assert actions.shape==(2,14 if panel=="dev" else 22,16)
        assert torch.count_nonzero(effects[:,-1,:])==0
        for p in x.POLICIES:
            assert x._probe(ww[0],p) in (0,1)
    with pytest.raises(ValueError,match="INVALID_WORLD"):
        x.world("dev",16,0,+1)


def test_preaction_effect_mask_and_erased_input_for_three_models():
    a,e=x.candidate_inputs(x.worlds("dev")[0],"active_information")
    for arm in ("raw","explicit","native","history","erased"):
        m=x.model_for(arm,985201)
        m.eval()
        with torch.no_grad():
            initial=x.preaction(m,arm,a,e)
            poison=e.clone()
            poison[:,-1,0]=.8
            assert torch.allclose(initial,x.preaction(m,arm,a,poison))
            erased=x.preaction(m,arm,a,e,erase=True)
            input_erased=e.clone()
            input_erased[:,4,:].zero_()
            assert torch.allclose(erased,
                        x.preaction(m,arm,a,input_erased))
        assert initial.shape==(2,)


def test_full_original_frozen_training_and_negative_science_claim():
    result=x.run()
    assert result["model_instances"]==11
    assert result["optimizer_updates"]==5632
    assert result["balanced_train_labels"]=={"action0":576,"action1":576}
    assert result["validation_pairs"]==384
    assert result["new_external_effects"]==0
    assert not result["spent_F7_replay"]
    assert not result["new_mechanism_transfer"]
    assert not result["capacity_FLOPs_matched_across_model_families"]
    assert not result["independent_scientific_pass"]
    assert result["deployment"]=="NONE"
    assert isinstance(result["judgment"]["raw_development_pass"],bool)
    assert isinstance(result["judgment"]["explicit_development_pass"],bool)
    assert isinstance(result["judgment"]["native_development_pass"],bool)
    for rec in result["models"]:
        assert rec["initial_sha256"]!=rec["trained_sha256"]
        assert rec["optimizer_updates"]==512
        assert rec["cold_restore_equal"]
        assert [v["updates"] for v in rec["checkpoints"]]==[128,256,512]
        assert all(math.isfinite(v["heldout_factual_choice_CE"])
                   for v in rec["checkpoints"])
        for panel in x.PANELS:
            assert len(rec["panels"][panel]["policies"])==4
            assert len(rec["panels"][panel]["erased_active"]["choices"])==64
            for pol in x.POLICIES:
                score=rec["panels"][panel]["policies"][pol]
                assert len(score["choices"])==64
                assert len(score["pergroup_regret"])==16
                assert 0<=score["sign_pair_flips"]<=32
                assert math.isfinite(score["mean_regret"])
    x.retain_ci_receipt(result)


def test_supervised_losses_never_receive_evaluator_hidden_q_or_unselected_return():
    for fn in (x.fit_pairs,x.batch,x.ExplicitGateMLP.forward,
               x.preaction,x._validation,x.train_arm):
        source=inspect.getsource(fn)
        for forbidden in ("q_evaluator_only","regret_evaluator_only(",
                          "r33.r32.after(", "hidden_sign", "oracle_label"):
            assert forbidden not in source
    whole=inspect.getsource(x)
    for forbidden in ("workflow_dispatch(", "subprocess.run(", "requests.get(",
                      "urlopen(","from causal_gym","import causal_gym"):
        assert forbidden not in whole
