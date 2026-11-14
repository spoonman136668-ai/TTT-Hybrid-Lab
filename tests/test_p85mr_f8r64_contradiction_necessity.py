"""R64 DEV-only contradiction directionality: frozen source and honest null."""
import pytest
import numpy as np
from experiments import p85mr_f8r64_contradiction_necessity as r64


def test_original_frozen_protocol_and_original_firstuse_fence():
    f=r64.locked()
    assert f["schema"]==r64.SCHEMA
    assert f["parent"]["r63_head_sha"]==r64.PARENT
    assert r64.git_blob(r64.FREEZE.read_bytes())==r64.FREEZE_BLOB
    assert f["source"]["salts"]==r64.SALT
    assert f["scoring"]["independent_scientific_pass"] is False
    assert all(not v for v in f["authority"].values())
    with pytest.raises(PermissionError,match="F8R64_EXTERNAL_ORIGINAL_FIRSTUSE_DENIED"):
        r64.deny_external_original_first_use()
    for forbidden in ("OOD","PROXY_NEGATIVE","unregistered"):
        with pytest.raises(ValueError,match="F8R64_NONDEV_SOURCE_DENIED"):
            r64.worlds(forbidden)


def test_fresh_selected_physical_laws_and_proxy_not_an_input():
    for panel in ("FIT","DEV"):
        ws=r64.worlds(panel)
        assert len(ws)==r64.COUNTS[panel]
        assert sum(int(w["switch_evaluator_only"]) for w in ws)==len(ws)//2
        for w in (ws[0],ws[1],ws[-1]):
            assert np.array_equal(w["roles"],np.arange(48)%4)
            assert w["effects"].shape==(48,4)
            assert w["proxy"].shape==(48,4)
            assert np.isfinite(w["effects"]).all()
            assert np.isfinite(w["proxy"]).all()
            assert np.isfinite(w["effects"]).all()
    d=r64.worlds("DEV")
    p=r64.proxy_negative(d)
    assert len(p)==len(d)
    for before,after in zip(d,p):
        assert np.array_equal(before["roles"],after["roles"])
        assert np.array_equal(before["effects"],after["effects"])
        assert not np.array_equal(before["proxy"],after["proxy"])
    with pytest.raises(ValueError,match="F8R64_PROXY_REQUIRES_ORIGINAL_DEV"):
        r64.proxy_negative(r64.worlds("FIT"))


def test_single_event_read_before_write_and_no_hidden_phase_inputs():
    cal=r64.fit_calibration(r64.worlds("FIT"))
    assert cal["FIT_events"]==128*48
    assert 0<=cal["fit_contradiction_boost_fraction"]<=1
    w=r64.worlds("DEV")[0]
    a=r64.rollout(w,"all_write_eta_0.5",cal)
    # All start with zero state, so first event error is squared actual effect.
    assert a["errors"][0]==pytest.approx(np.mean(w["effects"][0]**2),abs=1e-12)
    assert a["writes"]==48 and a["fast_floats"]==16
    assert a["boosts"]==0
    with pytest.raises(ValueError,match="F8R64_UNREGISTERED_MODE"):
        r64.rollout(w,"leak-regime",cal)
    bad={**w,"effects":w["effects"].copy()}
    bad["effects"][0,0]=np.nan
    with pytest.raises(ValueError,match="F8R64_INVALID_FACTUAL_EPISODE"):
        r64.rollout(bad,"contradiction",cal)


def test_complete_dev_falsifier_preserves_negative_and_cost_receipts():
    result=r64.run()
    assert result["schema"]==r64.SCHEMA
    assert result["freeze_blob"]==r64.FREEZE_BLOB
    assert result["parent"]==r64.PARENT
    assert result["source"]["FIT"]=={"worlds":128,"switched":64}
    assert result["source"]["DEV"]=={"worlds":64,"switched":32}
    assert set(result["method_scores"])==set(r64.MODES)
    assert result["total_r64_protected_OOD_calls"]==0
    assert result["external_original_first_use"] is False
    assert result["novel_model_training"] is False
    assert result["independent_scientific_pass"] is False
    assert result["deployment"]=="NONE"
    for mode,x in result["method_scores"].items():
        assert len(x["per_world"])==64
        assert x["writes_total"]==64*48
        assert x["fast_floats"]==16
        assert np.isfinite(x["B_changed_MSE"])
        assert np.isfinite(x["stable_world_MSE"])
        assert np.isfinite(x["A_return_first_MSE"])
    j=result["judgment"]
    assert isinstance(j["DEV_mechanism_signal"],bool)
    assert j["checks"]["proxy_goal_invariance"]
    assert j["checks"]["cold_replay_exact"]
    assert j["independent_scientific_pass"] is False
    r64.retain_ci_receipt(result)
