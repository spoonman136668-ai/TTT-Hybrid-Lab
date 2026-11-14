"""R66 source-custody, factual-action negative and bank identity DEV checks."""
import hashlib

import numpy as np
import pytest

from experiments import p85mr_f8r66_bank_identity as r66


def test_original_freeze_source_admission_and_first_use_rejection():
    f=r66.locked()
    assert r66.git_blob(r66.FREEZE.read_bytes())==r66.FREEZE_BLOB
    assert f["parent"]["base_R63_SHA"]==r66.PARENT
    assert f["parent"]["do_not_reuse_spent_scored_cohorts"]
    assert f["parent"]["R64_frozen_science"]=="DEV_NO_GO"
    assert f["parent"]["R65_frozen_science"]=="DEV_NO_GO"
    assert f["metrics"]["post_DEV_adjustments_forbidden"]
    assert all(v is False for v in f["authority"].values())
    for panel in ("OOD","R65_DEV","R64_DEV","PROXY_NEGATIVE"):
        with pytest.raises(PermissionError,match="F8R66_PROTECTED_OR_UNREGISTERED_SOURCE_DENIED"):
            r66.worlds(panel)
    with pytest.raises(PermissionError,match="F8R66_EXTERNAL_ORIGINAL_FIRST_USE_DENIED"):
        r66.deny_original_firstuse()


def test_fresh_physical_changed_and_stable_worlds_without_hidden_inputs():
    fit=r66.worlds("FIT")
    dev=r66.worlds("DEV")
    assert len(fit)==48 and len(dev)==96
    assert sum(bool(w["switched_evaluator_only"]) for w in fit)==24
    assert sum(bool(w["switched_evaluator_only"]) for w in dev)==48
    assert all(w["panel"]=="FIT" for w in fit)
    assert all(w["panel"]=="DEV" for w in dev)
    for w in (fit[0],fit[-1],dev[0],dev[-1]):
        assert np.array_equal(w["roles"],np.arange(48)%4)
        assert w["selected_effects"].shape==(48,4)
        assert w["proxy"].shape==(48,4)
        assert np.isfinite(w["selected_effects"]).all()
        if not w["switched_evaluator_only"]:
            assert np.array_equal(w["private_law_A"],w["private_law_B"])
        else:
            assert not np.array_equal(w["private_law_A"],w["private_law_B"])
            assert np.array_equal(w["private_law_A"][:,2:],w["private_law_B"][:,2:])
    digest=lambda w:hashlib.sha256(w["selected_effects"].tobytes()).hexdigest()
    assert set(map(digest,fit)).isdisjoint(map(digest,dev))
    proxy=r66.proxy_negative(dev)
    assert len(proxy)==len(dev)
    for a,b in zip(dev,proxy):
        assert np.array_equal(a["roles"],b["roles"])
        assert np.array_equal(a["selected_effects"],b["selected_effects"])
        assert not np.array_equal(a["proxy"],b["proxy"])
    with pytest.raises(PermissionError,match="F8R66_PROXY_SOURCE_MUST_BE_DEV"):
        r66.proxy_negative(fit)


def test_read_before_observed_consequence_and_memory_budget():
    world=r66.worlds("FIT")[0]
    for mode in r66.MODES:
        o=r66.update_role_bank(mode,world["roles"],world["selected_effects"])
        assert o["factual_selected_events"]==48
        assert o["allocated_fast_floats"]==36
        assert o["matrix_write_operations"] in (48,96)
        assert len(o["state_hash"])==64
        assert o["errors"][0]==pytest.approx(
            float(np.mean(world["selected_effects"][0]**2)),abs=1e-12)
        assert np.isfinite(o["errors"]).all()
        assert len(o["snapshot"])==2
        assert all(x.shape==(4,4) for x in o["snapshot"])
    with pytest.raises(ValueError,match="F8R66_INVALID_MECHANISM"):
        r66.update_role_bank("leaky_future",world["roles"],world["selected_effects"])
    altered=world["selected_effects"].copy()
    altered[0,0]=float("nan")
    with pytest.raises(ValueError,match="F8R66_BAD_SELECTED_FACTUAL_EVENTS"):
        r66.update_role_bank("protected_role_bank",world["roles"],altered)


def test_full_frozen_DEV_falsifier_reports_result_not_assumed_success():
    x=r66.run()
    assert x["schema"]==r66.SCHEMA
    assert x["freeze_blob"]==r66.FREEZE_BLOB
    assert x["parent"]==r66.PARENT
    assert x["source"]=={"FIT":48,"DEV":96,"switched_DEV":48,
                         "protected_OOD_generated":0}
    assert set(x["modes"])==set(r66.MODES)
    for mode,score in x["modes"].items():
        assert score["fast_floats_per_world"]==36
        assert score["factual_selected_events"]==96*48
        assert score["matrix_writes_total"] in (96*48,96*48*2)
        assert all(np.isfinite(score[k]) for k in (
           "return_MSE","B_MSE","stable_MSE","A_truth_MSE",
           "B_truth_MSE","separation"))
    j=x["judgment"]
    assert j["strongest_eligible_control"] in r66.CONTROLS
    assert j["checks"]["cold_replay_exact"]
    assert j["checks"]["proxy_invariance"]
    assert j["DEV_mechanism_signal"]==all(j["checks"].values())
    assert x["verdict"] in ("DEV_NO_GO","DEV_MECHANISM_SIGNAL_ONLY")
    assert x["independent_scientific_pass"] is False
    assert x["no_parametric_training"] is True
    assert x["external_science_firstuse"] is False
    assert x["GPU_firstuse"] is False
    r66.retain_ci_receipt(x)
