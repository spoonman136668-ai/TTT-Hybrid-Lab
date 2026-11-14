"""F8-R65 frozen FIT-only two-bank learned causal-memory negative controls."""
import hashlib
import numpy as np
import pytest
import torch
from experiments import p85mr_f8r65_trained_two_bank_retention as r65


def test_original_freeze_identity_and_science_fences():
    f=r65.locked()
    assert r65.git_blob(r65.FREEZE.read_bytes())==r65.FREEZE_BLOB
    assert f["parent"]["base_R63_sha"]==r65.PARENT
    assert f["parent"]["R64_source_and_DEV_spent"]
    assert f["parent"]["never_reuse_R64_scored_source"]
    assert f["source"]["salts"]==r65.SALT
    assert f["train"]["optimizer_steps_per_seed_per_trainable_arm"]==r65.STEPS
    assert all(not value for value in f["authority"].values())
    assert not f["metrics"]["post_score_tuning_allowed"]
    with pytest.raises(PermissionError,match="F8R65_PROTECTED_SCIENCE_FIRSTUSE_DENIED"):
        r65.deny_external_firstuse()
    with pytest.raises(PermissionError,match="F8R65_PROTECTED_OR_UNREGISTERED_SOURCE_DENIED"):
        r65.worlds("OOD")
    with pytest.raises(PermissionError,match="F8R65_PROTECTED_OR_UNREGISTERED_SOURCE_DENIED"):
        r65.worlds("R64_DEV")


def test_source_real_selected_regime_shift_and_disjoint_panels():
    fit=r65.worlds("FIT")
    cal=r65.worlds("CAL")
    dev=r65.worlds("DEV")
    assert (len(fit),len(cal),len(dev))==(96,32,64)
    for panel,worlds in (("FIT",fit),("CAL",cal),("DEV",dev)):
        assert sum(int(w["switched_evaluator_only"]) for w in worlds)==len(worlds)//2
        assert all(w["panel"]==panel for w in worlds)
        for w in (worlds[0],worlds[1],worlds[-1]):
            assert np.array_equal(w["roles"],np.arange(48)%4)
            assert w["effects"].shape==(48,4)
            assert w["proxy"].shape==(48,4)
            assert np.isfinite(w["effects"]).all()
            assert np.isfinite(w["proxy"]).all()
    neg=r65.proxy_negative(dev)
    for a,b in zip(dev,neg):
        assert np.array_equal(a["roles"],b["roles"])
        assert np.array_equal(a["effects"],b["effects"])
        assert not np.array_equal(a["proxy"],b["proxy"])
    with pytest.raises(PermissionError,match="F8R65_PROXY_REQUIRES_ORIGINAL_DEV"):
        r65.proxy_negative(fit)
    def effect_digest(w):
        return hashlib.sha256(w["effects"].tobytes()).hexdigest()
    assert set(map(effect_digest,fit)).isdisjoint(map(effect_digest,dev))
    assert set(map(effect_digest,cal)).isdisjoint(map(effect_digest,dev))


def test_factual_read_before_write_and_both_bank_budget():
    dev=r65.worlds("DEV")
    roles,effects=r65.tensor_batch(dev[:2],range(2))
    assert roles.shape==(2,48) and effects.shape==(2,48,4)
    assert effects.dtype==torch.float64
    fixed=r65.rollout(r65.FIXED,None,roles,effects,"DEV",[0,1],True)
    assert fixed["writes"]==48 and fixed["fast_state_floats"]==33
    assert fixed["errors"].shape==(2,48)
    assert fixed["state"].shape==(2,33)
    # On first action both state banks start empty, so no hidden answer can leak.
    assert fixed["errors"][0,0]==pytest.approx(float(np.mean(dev[0]["effects"][0]**2)),abs=1e-12)
    theta=torch.zeros(5,dtype=torch.float64)
    for mode in r65.TRAINED:
        out=r65.rollout(mode,theta,roles,effects,"DEV",[0,1],True)
        assert out["writes"]==48 and out["fast_state_floats"]==33
        assert np.isfinite(out["errors"]).all()
    with pytest.raises(ValueError,match="F8R65_UNREGISTERED_ARM"):
        r65.rollout("leaky_future",theta,roles,effects,"DEV",[0,1],True)
    bad=effects.clone()
    bad[0,0,0]=float("nan")
    with pytest.raises(ValueError,match="F8R65_MALFORMED_SELECTED_FACTS"):
        r65.rollout(r65.FIXED,None,roles,bad,"DEV",[0,1],True)


def test_full_frozen_train_and_dev_original_negative_without_tuning():
    result=r65.run()
    assert result["schema"]==r65.SCHEMA
    assert result["frozen_blob"]==r65.FREEZE_BLOB
    assert result["parent"]==r65.PARENT
    assert result["source"]=={"FIT":96,"CAL":32,"DEV":64,"switched_DEV":32}
    assert result["DEV_frozen_admission"] in ("DEV_NO_GO",
                                             "DEV_TRAINED_MECHANISM_SIGNAL_ONLY")
    assert result["no_R64_DEV_reuse"]
    assert result["no_protected_OOD_source"]
    assert result["external_firstuse"] is False
    assert result["GPU_firstuse"] is False
    assert result["independent_scientific_pass"] is False
    assert result["deployment"]=="NONE"
    for seed in r65.SEEDS:
        trained=result["train_receipts"][str(seed)]
        assert set(trained)==set(r65.TRAINED)
        for arm in r65.TRAINED:
            assert trained[arm]["optimizer_steps"]==96
            assert trained[arm]["selected_factual_events"]==96*12*48
            assert np.isfinite(trained[arm]["FIT_last_batch_MSE"])
            assert len(trained[arm]["weights_sha256"])==64
        for split in ("CAL","DEV"):
            arms=result["score"][split][str(seed)]
            assert set(arms)==set(r65.ARMS)
            for arm in r65.ARMS:
                row=arms[arm]
                assert row["writes_per_world"]==48
                assert row["fast_state_floats"]==33
                assert all(np.isfinite(row[key]) for key in
                           ("B_changed_MSE","A_return_first_MSE",
                            "stable_world_MSE","joint"))
    assert len(result["judgments"])==len(r65.SEEDS)
    for verdict in result["judgments"]:
        assert verdict["strongest_control"] in (
            "nondirectional_surprise","random_negative",r65.FIXED)
        assert verdict["checks"]["cold_restore_exact"]
        assert verdict["checks"]["proxy_invariance"]
        assert verdict["all_scientific_DEV_conditions_met"]==all(verdict["checks"].values())
    r65.retain_ci_receipt(result)
