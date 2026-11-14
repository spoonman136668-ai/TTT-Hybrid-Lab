"""R67 prospective causal bank FIT-training, admission, and negative tests."""
import hashlib

import numpy as np
import pytest
import torch

from experiments import p85mr_f8r67_trained_protected_memory as r67


def test_frozen_ancestor_and_no_unauthorized_science():
    f=r67.locked()
    assert r67.git_blob(r67.FREEZE.read_bytes())==r67.FREEZE_BLOB
    assert f["parent"]["R66_SHA"]==r67.PARENT
    assert f["parent"]["R66_result"]=="ENGINEERING_PASS_SYNTHETIC_DEV_SIGNAL_ONLY"
    assert f["parent"]["no_R66_or_earlier_scored_world_reuse"]
    assert f["training"]["optimizer_steps_each_trainable_arm_seed"]==96
    assert f["metrics"]["eligible_controls"]==list(r67.CONTROL)
    assert all(v is False for v in f["authority"].values())
    for panel in ("OOD","R66_DEV","R65_DEV","R64_DEV","PROXY_NEGATIVE"):
        with pytest.raises(PermissionError,match="F8R67_PROTECTED_OR_OLD_SOURCE_DENIED"):
            r67.worlds(panel)
    with pytest.raises(PermissionError,match="F8R67_EXTERNAL_SCIENTIFIC_FIRSTUSE_DENIED"):
        r67.deny_external_firstuse()


def test_fresh_observed_causal_laws_and_proxy_negative():
    for panel,count in r67.COUNTS.items():
        worlds=r67.worlds(panel)
        assert len(worlds)==count
        assert sum(x["switched_evaluator_only"] for x in worlds)==count//2
        families={x["mechanism_evaluator_only"] for x in worlds}
        assert families==(
            {"stable","rotation","crossrole"} if panel=="DEV"
            else {"stable","rotation","gain"})
        for w in (worlds[0],worlds[-1]):
            assert w["panel"]==panel
            assert np.array_equal(w["roles"],np.arange(48)%4)
            assert w["effects"].shape==(48,4)
            assert w["proxy"].shape==(48,4)
            assert np.isfinite(w["effects"]).all()
            assert np.isfinite(w["proxy"]).all()
    fit=r67.worlds("FIT")
    dev=r67.worlds("DEV")
    dig=lambda w:hashlib.sha256(w["effects"].tobytes()).hexdigest()
    assert set(map(dig,fit)).isdisjoint(map(dig,dev))
    assert sum(w["mechanism_evaluator_only"]=="crossrole" for w in dev)==24
    negative=r67.proxy_negative(dev)
    for w,n in zip(dev,negative):
        assert np.array_equal(w["roles"],n["roles"])
        assert np.array_equal(w["effects"],n["effects"])
        assert not np.array_equal(w["proxy"],n["proxy"])
    with pytest.raises(PermissionError,match="F8R67_PROXY_NEEDS_ORIGINAL_DEV"):
        r67.proxy_negative(fit)


def test_selected_action_online_read_before_write_and_matched_fast_state():
    fit=r67.worlds("FIT")
    roles,effects=r67.factual_batch(fit,[0,1])
    assert roles.dtype==torch.int64 and effects.dtype==torch.float64
    assert roles.shape==(2,48) and effects.shape==(2,48,4)
    assert (r67.D,r67.FAST,r67.N)==(4,36,48)
    for mode in r67.TRAINED:
        model=r67.init(mode,r67.SEEDS[0])
        with torch.no_grad():
            data=model(roles,effects,details=True)
        assert data["errors"].shape==(2,48)
        assert data["state"].shape==(2,36)
        assert np.isfinite(data["errors"]).all()
        assert np.isfinite(data["state"]).all()
        if mode in r67.TRAINED[:2]:
            # Both memory banks begin at zero, so first event cannot be
            # predicted using its own future physical result.
            assert data["errors"][0,0]==pytest.approx(
                float(np.mean(fit[0]["effects"][0]**2)),abs=1e-12)
        assert sum(p.numel() for p in model.parameters())>=5
    bad=effects.clone()
    bad[0,0,0]=float("nan")
    with pytest.raises(ValueError,match="F8R67_BAD_FACTUAL_BATCH"):
        r67.init(r67.TRAINED[0],r67.SEEDS[0])(roles,bad)
    with pytest.raises(ValueError,match="F8R67_UNREGISTERED_LEARNED_GATE"):
        r67.SparseBankGate(0,r67.TRAINED[0])
    with pytest.raises(ValueError,match="F8R67_NOT_TRAINABLE_ARM"):
        r67.init(r67.ARMS[-1],r67.SEEDS[0])


def test_all_trainable_arms_propagate_finite_factual_gradients():
    roles,effects=r67.factual_batch(r67.worlds("FIT"),[0,1])
    for arm in r67.TRAINED:
        model=r67.init(arm,r67.SEEDS[0])
        loss=model(roles,effects)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters())


def test_original_full_trained_dev_qualification_records_null_honestly():
    result=r67.run()
    assert result["schema"]==r67.SCHEMA
    assert result["freeze_blob"]==r67.FREEZE_BLOB
    assert result["parent"]==r67.PARENT
    assert result["source"]=={
        "FIT":128,"CAL":32,"DEV":96,"switched_DEV":48,
        "unfamiliar_crossrole_DEV":24,"protected_OOD_generated":0}
    assert result["DEV_admission"] in (
        "DEV_NO_GO","DEV_TRAINED_MECHANISM_SIGNAL_ONLY")
    assert result["independent_scientific_pass"] is False
    assert result["CUDA_used"] is False
    assert result["original_external_firstuse"] is False
    assert result["deployment"]=="NONE"
    assert len(result["verdicts"])==2
    for seed in r67.SEEDS:
        receipts=result["training"][str(seed)]
        assert set(receipts)==set(r67.TRAINED)
        for arm in r67.TRAINED:
            z=receipts[arm]
            assert z["optimizer_steps"]==96
            assert z["FIT_factual_selected_events"]==96*8*48
            assert z["fast_state_floats"]==36
            assert len(z["weights_sha256"])==64
            assert np.isfinite(z["train_final_MSE"])
        for panel in ("CAL","DEV"):
            scores=result["scores"][panel][str(seed)]
            assert set(r67.ARMS).issubset(scores)
            for arm in r67.ARMS:
                z=scores[arm]
                assert z["fast_state_floats"]==36
                assert z["factual_events"]==48*r67.COUNTS[panel]
                assert all(np.isfinite(z[k]) for k in
                           ("B_MSE","A_return_MSE","stable_MSE","joint_MSE"))
    for verdict in result["verdicts"]:
        assert verdict["seed"] in r67.SEEDS
        assert verdict["strongest_eligible_control"] in r67.CONTROL
        assert verdict["checks"]["train_step_and_factual_event_parity"]
        assert verdict["checks"]["proxy_invariance"]
        assert verdict["checks"]["cold_restore_exact"]
        assert verdict["all_DEV_frozen_checks"]==all(verdict["checks"].values())
    r67.retain_ci_receipt(result)
