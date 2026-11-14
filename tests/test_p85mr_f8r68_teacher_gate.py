"""F8-R68 frozen causal memory teacher imitation and negative-control proof."""
import hashlib

import numpy as np
import pytest
import torch

from experiments import p85mr_f8r68_teacher_gate as r68


def test_prospective_freeze_ancestry_and_source_admission():
    f=r68.locked()
    assert r68.git_blob(r68.FREEZE.read_bytes())==r68.FREEZE_BLOB
    assert f["parent"]["exact_R66_SHA"]==r68.PARENT
    assert f["parent"]["R67_result"]=="DEV_NO_GO"
    assert f["parent"]["no_spent_cohorts_or_test_tuning"]
    assert f["metrics"]["post_result_tuning_forbidden"]
    assert all(v is False for v in f["authority"].values())
    assert f["training"]["optimizer_steps_per_seed_per_trainable_arm"]==128
    assert f["training"]["FIT_world_batch_per_step"]==16
    with pytest.raises(PermissionError,match="F8R68_EXTERNAL_FIRST_USE_DENIED"):
        r68.deny_external_firstuse()
    for panel in ("R66_DEV","R67_DEV","R65_DEV","OOD","PROXY_NEGATIVE"):
        with pytest.raises(PermissionError,match="F8R68_PROTECTED_OR_SPENT_SOURCE_DENIED"):
            r68.worlds(panel)


def test_independent_source_task_withheld_mechanism_and_proxy_negative():
    sources={p:r68.worlds(p) for p in r68.COUNTS}
    for panel,worlds in sources.items():
        assert len(worlds)==r68.COUNTS[panel]
        assert sum(w["evaluator_switched_only"] for w in worlds)==len(worlds)//2
        assert all(w["panel"]==panel for w in worlds)
        families={w["evaluator_family_only"] for w in worlds}
        assert families==(
            {"stable","rotation","swap"} if panel=="DEV"
            else {"stable","rotation","gain"})
        for w in (worlds[0],worlds[-1]):
            assert w["roles"].shape==(48,)
            assert np.array_equal(w["roles"],np.arange(48)%4)
            assert w["effects"].shape==(48,4)
            assert w["proxy"].shape==(48,4)
            assert np.isfinite(w["effects"]).all()
            assert np.isfinite(w["proxy"]).all()
    digest=lambda w:hashlib.sha256(w["effects"].tobytes()).hexdigest()
    assert set(map(digest,sources["FIT"])).isdisjoint(map(digest,sources["DEV"]))
    assert sum(w["evaluator_family_only"]=="swap" for w in sources["DEV"])==24
    assert all(w["evaluator_family_only"]!="swap" for w in sources["FIT"])
    neg=r68.negative_proxy(sources["DEV"])
    for x,y in zip(sources["DEV"],neg):
        assert np.array_equal(x["roles"],y["roles"])
        assert np.array_equal(x["effects"],y["effects"])
        assert not np.array_equal(x["proxy"],y["proxy"])
    with pytest.raises(PermissionError,match="F8R68_PROXY_SOURCE_MUST_BE_DEV"):
        r68.negative_proxy(sources["FIT"])


def test_teacher_exact_original_r66_and_public_selected_effects():
    fit=r68.worlds("FIT")
    traces=[r68.teacher_trace(w) for w in fit[:8]]
    assert r68.teacher_identity(fit[:8],traces)
    assert all(x["labels"].shape==(48,) and x["ordered"].shape==(48,10)
               and x["invariant"].shape==(48,10) for x in traces)
    assert all(np.isfinite(x["ordered"]).all() for x in traces)
    assert all(x["errors"][0]==pytest.approx(
        float(np.mean(w["effects"][0]**2)),abs=1e-12)
        for x,w in zip(traces,fit))
    src=fit[0]
    modified={**src,"effects":src["effects"].copy()}
    modified["effects"][47]+=8
    base=r68.teacher_trace(src)
    alt=r68.teacher_trace(modified)
    # An effect in the FUTURE never changes earlier decisions or scoring.
    assert np.array_equal(base["errors"][:47],alt["errors"][:47])
    assert np.array_equal(base["labels"][:47],alt["labels"][:47])
    with pytest.raises(ValueError,match="F8R68_INVALID_FEATURE_MODE"):
        r68.observable_features(.2,.3,.4,.5,"private_phase")
    with pytest.raises(ValueError,match="F8R68_INVALID_OBSERVED_FEATURE"):
        r68.observable_features(.2,.3,float("nan"),.5,"ordered")


def test_strong_supervised_arms_have_frozen_capacity_and_fit_labels():
    fit=r68.worlds("FIT")
    traces=[r68.teacher_trace(w) for w in fit]
    positives=sum(int(x["labels"].sum()) for x in traces)
    assert 0<positives<128*48
    for seed in r68.SEEDS:
        for arm in r68.TRAINED:
            model=r68.BinaryGate(arm,seed)
            n=sum(p.numel() for p in model.parameters())
            assert n==(11 if arm=="FULL_ORDERED_LINEAR" else 193)
            features,labels=r68.training_arrays(traces,arm,seed)
            assert features.shape==(128,48,10)
            assert labels.shape==(128,48)
            assert float(labels.sum())==positives
            logits=model(features[:2].reshape(-1,10))
            loss=torch.nn.functional.binary_cross_entropy_with_logits(
                logits,labels[:2].reshape(-1))
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.parameters())
    with pytest.raises(ValueError,match="F8R68_TRAINABLE_ARM_OR_SEED_DENIED"):
        r68.BinaryGate("FORGE_ORACLE",r68.SEEDS[0])


def test_full_frozen_teacher_imitation_dev_receipts_not_assumed_success():
    result=r68.run()
    assert result["schema"]==r68.SCHEMA
    assert result["freeze_blob"]==r68.FREEZE_BLOB
    assert result["parent"]==r68.PARENT
    assert result["source"]=={"FIT":128,"CAL":32,"DEV":96,
                              "switched_DEV":48,"held_out_swap_DEV":24,
                              "protected_external_OOD":0}
    assert result["DEV_admission"] in (
        "DEV_TEACHER_IMITATION_MECHANISM_SIGNAL_ONLY","DEV_NO_GO")
    assert result["independent_scientific_pass"] is False
    assert result["CUDA_used"] is False
    assert result["external_firstuse"] is False
    assert result["deployment"]=="NONE"
    assert 0<result["FIT_teacher_positive_count"]<128*48
    assert len(result["verdicts"])==2
    for seed in r68.SEEDS:
        train=result["training"][str(seed)]
        assert set(train)==set(r68.TRAINED)
        for arm in r68.TRAINED:
            t=train[arm]
            assert t["optimizer_steps"]==128
            assert t["FIT_world_batch"]==16
            assert t["FIT_selected_factual_event_exposures"]==128*16*48
            assert t["persistent_fast_floats"]==36
            assert len(t["trained_weight_SHA256"])==64
            assert np.isfinite(t["BCE_last_batch"])
        for panel in ("CAL","DEV"):
            scores=result["scores"][panel][str(seed)]
            accuracy=result["accuracy"][panel][str(seed)]
            assert set(scores)==set(r68.ARMS)
            assert set(accuracy)==set(r68.TRAINED)
            for arm,score in scores.items():
                assert score["allocated_fast_floats"]==36
                assert score["selected_event_count"]==48*r68.COUNTS[panel]
                assert all(np.isfinite(score[k]) for k in
                           ("B_MSE","A_return_MSE","stable_MSE","joint_MSE"))
            for arm,acc in accuracy.items():
                assert 0<=acc["positive_recall"]<=1
                assert 0<=acc["negative_specificity"]<=1
                assert 0<=acc["balanced_accuracy"]<=1
    for v in result["verdicts"]:
        assert v["best_trained_control"] in r68.TRAINED[1:]
        assert v["checks"]["teacher_trace_exact_R66"]
        assert v["checks"]["FIT_optimizer_and_event_parity"]
        assert v["checks"]["proxy_negative_invariance"]
        assert v["checks"]["cold_start_replay_exact"]
        assert v["all_frozen_checks"]==all(v["checks"].values())
    r68.retain_ci_receipt(result)


def test_binary_gate_closed_loop_receipt_json_is_builtin_int():
    """Regression for both original OS failures at SHA 69862f35: NumPy int64."""
    import json

    source = r68.worlds("FIT")
    stable = next(x for x in source if not x["evaluator_switched_only"])
    switched = next(x for x in source if x["evaluator_switched_only"])
    examples = [stable, switched]
    model = r68.BinaryGate("FULL_ORDERED_MLP", r68.SEEDS[0])
    receipt = r68.score("FULL_ORDERED_MLP", model, examples, [])
    assert type(receipt["total_bank_switches"]) is int
    assert receipt["selected_event_count"] == 2*r68.N
    assert receipt["allocated_fast_floats"] == r68.FAST
    json.dumps(receipt, allow_nan=False)
