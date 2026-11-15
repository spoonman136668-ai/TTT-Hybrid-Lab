"""R72 genuine selected-consequence and public-teacher optimization diagnostics."""
import hashlib
import json

import numpy as np
import pytest
import torch

from experiments import p85mr_f8r72_gradient_diagnostic as r72


def test_frozen_source_custody_and_no_spent_or_protected_firstuse():
    freeze=r72.locked()
    assert r72.blob_sha(r72.FREEZE.read_bytes())==r72.FREEZE_BLOB
    assert freeze["source_custody"]["parent_R71_SHA"]==r72.PARENT
    assert freeze["source_custody"]["R71_original_result"]=="ENGINEERING_PASS_DEV_NO_GO"
    assert freeze["source_custody"]["all_R64_to_R71_scored_worlds_spent"]
    assert freeze["scoring"]["no_post_score_changes"]
    assert all(v is False for v in freeze["authority"].values())
    for p in ("DEV","CAL","OOD","R71_DEV","R70_DEV","PROXY_NEGATIVE"):
        with pytest.raises(PermissionError,match="F8R72_SPENT_OR_PROTECTED_SOURCE_DENIED"):
            r72.worlds(p)
    with pytest.raises(PermissionError,match="F8R72_PROTECTED_GPU_OR_EXTERNAL_FIRSTUSE_DENIED"):
        r72.deny_protected_firstuse()


def test_new_source_causal_sign_changes_and_negative_proxy_privacy():
    source={p:r72.worlds(p) for p in r72.COUNTS}
    ids=lambda worlds:{hashlib.sha256(
        w["potential_innovations_evaluator_only"].tobytes()).hexdigest()
        for w in worlds}
    assert ids(source["FIT"]).isdisjoint(ids(source["HOLDOUT"]))
    for panel,worlds in source.items():
        assert len(worlds)==r72.COUNTS[panel]
        assert sum(w["switched_evaluator_only"] for w in worlds)==len(worlds)//2
        assert sum(w["initial_sign_evaluator_only"]==1 for w in worlds)==len(worlds)//2
        for world in worlds:
            assert world["R72_source_panel"]==panel
            assert world["panel"]==world["source_panel"]==(
                "FIT" if panel=="FIT" else "DEV")
            assert 9<=world["B_start_evaluator_only"]<=18
            assert 22<=world["A_return_start_evaluator_only"]<=35
            assert world["potential_innovations_evaluator_only"].shape==(40,4,4)
            assert world["unobserved_nuisance_proxy"].shape==(40,4)
            assert np.isfinite(world["potential_innovations_evaluator_only"]).all()
    proxy=r72.negative_proxy(source["HOLDOUT"])
    for w,q in zip(source["HOLDOUT"],proxy):
        assert np.array_equal(w["latent_law_evaluator_only"],
                              q["latent_law_evaluator_only"])
        assert np.array_equal(w["potential_innovations_evaluator_only"],
                              q["potential_innovations_evaluator_only"])
        assert not np.array_equal(w["unobserved_nuisance_proxy"],
                                  q["unobserved_nuisance_proxy"])
    with pytest.raises(PermissionError,match="F8R72_NEGATIVE_REQUIRES_FRESH_HOLDOUT"):
        r72.negative_proxy(source["FIT"])


def test_teacher_factual_selected_observations_are_only_supervision():
    fit=r72.worlds("FIT")
    for world in fit[:4]:
        acts,observations=r72.teacher_history(world)
        assert len(acts)==len(observations)==40
        assert tuple(t for t,a in enumerate(acts) if a==0)==(0,16,32)
        assert all(a in (0,2,3) for a in acts)
        for arm in r72.ARMS:
            model=r72.r71.TrainablePolicy(arm,r72.SEEDS[0])
            loss=r72.teacher_loss(model,world)
            assert torch.isfinite(loss)
            model.zero_grad(set_to_none=True)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.parameters())
            wrong=r72.teacher_loss(model,world,shift_labels=True)
            assert torch.isfinite(wrong)
            assert wrong.detach().item()!=pytest.approx(loss.detach().item(),abs=1e-12)


def test_same_frozen_FIT_batches_and_reward_gradient_diagnostics():
    fit=r72.worlds("FIT")
    for seed in r72.SEEDS:
        a=r72._fit_batches(seed,fit)
        b=r72._fit_batches(seed,fit)
        assert np.array_equal(a,b)
        assert a.shape==(32,8)
        for arm in r72.ARMS:
            stats=r72.gradient_quality(arm,seed,fit)
            assert stats["FIT_worlds_per_gradient_batch"]==8
            assert -1.0000001<=stats["initial_gradient_cosine_disjoint_FIT_batches"]<=1.0000001
            assert stats["initial_gradient_norm_batchA"]>0
            assert stats["initial_gradient_norm_batchB"]>0
    with pytest.raises(PermissionError,match="F8R72_TRAIN_ONLY_FRESH_FIT_FROZEN_ARMS"):
        r72.train_one(r72.ARMS[0],r72.SEEDS[0],r72.METHODS[0],
                      r72.worlds("HOLDOUT"))


def test_future_selected_outcome_never_changes_prior_action():
    w=r72.worlds("FIT")[0]
    shifted={**w,"potential_innovations_evaluator_only":
             w["potential_innovations_evaluator_only"].copy()}
    shifted["potential_innovations_evaluator_only"][39,:,0]+=90
    for arm in r72.ARMS:
        m=r72.r71.TrainablePolicy(arm,r72.SEEDS[0])
        original=r72.r71.rollout(m,w,greedy=True)
        altered=r72.r71.rollout(m,shifted,greedy=True)
        assert original["actions"]==altered["actions"]
    a,_=r72.teacher_history(w)
    b,_=r72.teacher_history(shifted)
    assert a==b


def test_full_two_seed_frozen_FIT_gradient_and_public_behavior_diagnostic():
    report=r72.run()
    assert report["schema"]==r72.SCHEMA
    assert report["freeze_blob"]==r72.FREEZE_BLOB
    assert report["parent"]==r72.PARENT
    assert report["source"]=={
        "FIT":128,"HOLDOUT":48,"switched_HOLDOUT":24,
        "protected_external_OOD":0}
    assert len(report["verdicts"])==2
    for seed in r72.SEEDS:
        label=str(seed)
        training=report["training"][label]
        scored=report["scores"][label]
        agree=report["teacher_forced_factual_action_agreement"][label]
        gradient=report["initial_selected_reward_gradient_diagnostics"][label]
        expected={arm+"_"+suffix for arm in r72.ARMS for suffix in ("BC","PG")}
        expected.add(r72.NEGATIVE)
        assert set(training)==expected
        assert set(agree)==expected
        assert set(scored)==expected|set(r72.FIXED)
        assert set(gradient)==set(r72.ARMS)
        for arm,s in training.items():
            assert s["steps"]==32
            assert s["FIT_worlds_per_step"]==8
            assert s["FIT_actual_selected_event_exposures"]==32*8*40
            assert s["trainable_parameters"]>0
            assert len(s["trained_weight_sha256"])==64
            assert np.isfinite(s["loss_first"])
            assert np.isfinite(s["loss_last"])
        for arm,s in scored.items():
            assert s["selected_action_events"]==48*40
            assert np.isfinite(s["episode_reward"])
        for arm,a in agree.items():
            assert a["observed_factual_selected_events"]==48*40
            assert 0<=a["teacher_factual_action_agreement"]<=1
            assert 0<=a["public_calibration_recall"]<=1
            assert 0<=a["outside_calibration_false_positive_rate"]<=1
    for v in report["verdicts"]:
        assert len(v["checks"])==8
        assert v["source_nonoverlap"]
        assert v["balanced_laws"]
        assert v["cold_replay_and_unobserved_proxy_negative"]
        assert v["all_frozen_checks"]==all(v["checks"].values())
    assert report["DEV_admission"] in (
        "DEV_DIAGNOSTIC_NO_GO",
        "DEV_SCRIPTED_BEHAVIOR_LEARNABILITY_SIGNAL_ONLY")
    assert (report["DEV_admission"]==
            "DEV_SCRIPTED_BEHAVIOR_LEARNABILITY_SIGNAL_ONLY") == all(
                v["all_frozen_checks"] for v in report["verdicts"])
    assert report["independent_scientific_pass"] is False
    assert report["novel_causal_law_learning_proven"] is False
    assert report["full_hybrid_substrate_qualification"] is False
    assert report["GPU_firstuse"] is False
    assert report["external_scientific_firstuse"] is False
    assert report["deployment"]=="NONE"
    json.dumps(report,allow_nan=False)
    r72.retain_ci_receipt(report)
