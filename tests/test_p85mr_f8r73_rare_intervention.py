"""F8-R73 prospective rare-intervention factual-learning qualification."""
import hashlib
import json

import numpy as np
import pytest
import torch

from experiments import p85mr_f8r73_rare_intervention as r73


def test_frozen_protocol_exact_original_blob_and_denied_first_use():
    spec=r73.frozen()
    assert r73.git_blob(r73.FREEZE.read_bytes())==r73.FREEZE_BLOB
    assert spec["parent"]["R72_SHA"]==r73.PARENT
    assert spec["parent"]["all_R64_to_R72_scored_cohorts_spent"]
    assert spec["source"]["counts"]==r73.COUNTS
    assert spec["scoring"]["no_post_result_tuning"]
    assert all(v is False for v in spec["authority"].values())
    assert r73.CLASS_WEIGHTS==(37./3.,1.,1.)
    for panel in ("DEV","CAL","R72_HOLDOUT","R72_FIT","OOD","NEGATIVE"):
        with pytest.raises(PermissionError,match="F8R73_SPENT_OR_PROTECTED_SOURCE_DENIED"):
            r73.worlds(panel)
    with pytest.raises(PermissionError,match="F8R73_PROTECTED_EXTERNAL_GPU_FIRSTUSE_DENIED"):
        r73.deny_protected_firstuse()


def test_fresh_balanced_source_and_proxy_negative_privacy():
    fit=r73.worlds("FIT")
    hold=r73.worlds("HOLDOUT")
    def digest(w):
        return hashlib.sha256(
            w["potential_innovations_evaluator_only"].tobytes()).hexdigest()
    assert set(map(digest,fit)).isdisjoint(map(digest,hold))
    for panel,source in (("FIT",fit),("HOLDOUT",hold)):
        assert len(source)==r73.COUNTS[panel]
        assert sum(int(w["switched_evaluator_only"]) for w in source)==len(source)//2
        assert sum(int(w["initial_sign_evaluator_only"]==1)
                   for w in source)==len(source)//2
        for w in source:
            assert w["R73_source_panel"]==panel
            assert w["panel"]==("FIT" if panel=="FIT" else "DEV")
            assert 9<=w["B_start_evaluator_only"]<=18
            assert 22<=w["A_return_start_evaluator_only"]<=35
            assert w["potential_innovations_evaluator_only"].shape==(40,4,4)
            assert np.isfinite(w["potential_innovations_evaluator_only"]).all()
    noise=r73.negative_proxy(hold)
    for w,q in zip(hold,noise):
        assert np.array_equal(w["latent_law_evaluator_only"],
                              q["latent_law_evaluator_only"])
        assert np.array_equal(w["potential_innovations_evaluator_only"],
                              q["potential_innovations_evaluator_only"])
        assert not np.array_equal(w["unobserved_nuisance_proxy"],
                                  q["unobserved_nuisance_proxy"])
    with pytest.raises(PermissionError,match="F8R73_NEGATIVE_REQUIRES_FRESH_HOLDOUT"):
        r73.negative_proxy(fit)


def test_scripted_selected_fact_source_and_rare_labels():
    fit=r73.worlds("FIT")
    for w in fit[:5]:
        actions,obs=r73.teacher(w)
        assert len(actions)==len(obs)==40
        assert tuple(i for i,a in enumerate(actions) if a==0)==(0,16,32)
        assert set(actions).issubset({0,2,3})
        assert all(v.shape==(4,) and np.isfinite(v).all() for v in obs)
        _,h=r73.r70.transition(w,0,0,0)
        _,sh=r73.r70.transition(w,0,1,0)
        true_sign=int(w["latent_law_evaluator_only"][1])
        chosen=2 if true_sign>0 else 3
        calibrated,_=r73.r70.transition(w,1,chosen,h)
        sham,_=r73.r70.transition(w,1,chosen,sh)
        assert calibrated[0]-sham[0]==pytest.approx(.8)
    invalid={**fit[0],"R73_source_panel":"R72_SPENT"}
    with pytest.raises(PermissionError,match="F8R73_FACTUAL_TEACHER_WRONG_SOURCE"):
        r73.teacher(invalid)


def test_factual_supervised_gradients_and_all_model_shapes():
    w=r73.worlds("FIT")[0]
    for arm in r73.ARMS:
        a=r73.Model(arm,r73.SEEDS[0])
        assert a.arm==arm
        for method in r73.METHODS[:3]:
            loss=r73.factual_loss(a,w,method)
            assert torch.isfinite(loss)
            a.zero_grad(set_to_none=True)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in a.parameters())
        out=r73.r71.rollout(a,w,greedy=True)
        assert len(out["actions"])==40
        assert all(action in (0,2,3) for action in out["actions"])
    with pytest.raises(ValueError,match="F8R73_INVALID_TRAINED_MODEL"):
        r73.Model("ORACLE",r73.SEEDS[0])
    with pytest.raises(ValueError,match="F8R73_ILLEGAL_BC_METHOD"):
        r73.factual_loss(r73.Model(r73.ARMS[0],r73.SEEDS[0]),w,r73.METHODS[3])
    with pytest.raises(PermissionError,match="F8R73_UNREGISTERED_OR_NON_FIT_TRAINING"):
        r73.train(r73.ARMS[0],r73.METHODS[0],r73.SEEDS[0],r73.worlds("HOLDOUT"))


def test_holdout_future_selected_consequence_does_not_affect_past():
    w=r73.worlds("FIT")[0]
    changed={**w,"potential_innovations_evaluator_only":
             w["potential_innovations_evaluator_only"].copy()}
    changed["potential_innovations_evaluator_only"][39,:,0]+=60
    for arm in r73.ARMS:
        policy=r73.Model(arm,r73.SEEDS[0])
        before=r73.r71.rollout(policy,w,greedy=True)
        after=r73.r71.rollout(policy,changed,greedy=True)
        assert before["actions"]==after["actions"]
    a,_=r73.teacher(w)
    b,_=r73.teacher(changed)
    assert a==b


def test_two_seed_frozen_full_R73_training_and_independent_holdout():
    result=r73.run()
    assert result["schema"]==r73.SCHEMA
    assert result["freeze_blob"]==r73.FREEZE_BLOB
    assert result["parent"]==r73.PARENT
    assert result["source"]=={
       "FIT":128,"HOLDOUT":64,"switched_HOLDOUT":32,
       "protected_external_OOD":0}
    assert len(result["verdicts"])==2
    for seed in r73.SEEDS:
        key=str(seed)
        trained=result["training"][key]
        scored=result["scores"][key]
        factual=result["factual_teacher_agreement"][key]
        expected={a+"_"+method for a in r73.ARMS for method in r73.NAMES}
        assert set(trained)==expected
        assert set(factual)==expected
        assert set(scored)==expected|set(r73.FIXED)
        for meta in trained.values():
            assert meta["steps"]==64
            assert meta["FIT_worlds_per_step"]==8
            assert meta["FIT_selected_event_exposures"]==64*8*40
            assert len(meta["weights_sha256"])==64
            assert meta["trainable_params"]>0
            assert np.isfinite(meta["initial_loss"])
            assert np.isfinite(meta["final_loss"])
        for score in scored.values():
            assert score["selected_action_events"]==64*40
            assert np.isfinite(score["episode_reward"])
        for values in factual.values():
            assert values["observed_factual_actions"]==64*40
            assert 0<=values["calibration_recall"]<=1
            assert 0<=values["calibration_precision"]<=1
            assert 0<=values["calibration_f1"]<=1
            assert 0<=values["calibration_false_positive_rate"]<=1
    for v in result["verdicts"]:
        assert len(v["checks"])==14
        assert v["checks"]["source_balance_and_FIT_HOLDOUT_nonoverlap"]
        assert v["checks"]["same_FIT_steps_draws_factual_exposures"]
        assert v["checks"]["negative_unobserved_proxy_invariance"]
        assert v["checks"]["exact_cold_replay"]
        assert v["checks"]["original_causal_effect"]
        assert v["all_frozen_checks"]==all(v["checks"].values())
    assert result["DEV_admission"] in (
      "DEV_RARE_SCRIPTED_ACTION_LEARNABILITY_SIGNAL_ONLY","DEV_DIAGNOSTIC_NO_GO")
    assert (result["DEV_admission"]=="DEV_RARE_SCRIPTED_ACTION_LEARNABILITY_SIGNAL_ONLY")==all(
        v["all_frozen_checks"] for v in result["verdicts"])
    assert result["independent_scientific_pass"] is False
    assert result["native_causal_law_discovery"] is False
    assert result["full_hybrid_substrate_qualification"] is False
    assert result["external_science_firstuse"] is False
    assert result["GPU_firstuse"] is False
    assert result["deployment"]=="NONE"
    json.dumps(result,allow_nan=False)
    r73.retain_ci_receipt(result)
