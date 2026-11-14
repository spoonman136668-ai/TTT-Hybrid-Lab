"""Prospective R60 FP64 portability qualification on new law cohorts."""
import pytest
import torch
from experiments import p85mr_f8r60_crossplatform_precision as r60


def test_new_freeze_preserved_and_original_source_firstuse_denied():
    f=r60.locked()
    assert f["schema"]==r60.SCHEMA
    assert f["stage"]=="FROZEN_BEFORE_NEW_R60_SOURCE_OR_SCORE"
    assert r60.git_blob(r60.FREEZE.read_bytes())==r60.FREEZE_BLOB
    assert f["parent"]["R59_SHA"]==r60.PARENT
    assert f["source"]["salts"]==r60.SALT
    assert f["source"]["cases"]==r60.CASES
    assert f["source"]["no_old_R49_R59_OOD_data"]
    assert f["execution"]["trainable_total_attempts"]==6144
    assert f["execution"]["torch_dtype"]=="float64"
    assert f["execution"]["torch_num_threads"]==1
    assert f["execution"]["mkldnn_enabled"] is False
    assert f["frozen_admission"]["every_decision_bitwise_equal_Windows_Ubuntu"]
    assert f["frozen_admission"]["independent_scientific_pass"] is False
    assert not any(f["authority"].values())
    with pytest.raises(PermissionError,match="EXTERNAL_ORIGINAL_FIRSTUSE_DENIED"):
        r60.deny_external_original_firstuse()
    with pytest.raises(ValueError,match="UNREGISTERED_SELECTED_CASE"):
        r60.selected_case("OOD",1024)


def test_four_physical_law_families_public_selected_outcomes_and_float64():
    f=r60.locked()
    summ=r60.source_sanity()
    assert set(summ)==set(r60.PANELS)
    for p in r60.PANELS:
        assert summ[p]["family_counts"]==[r60.CASES[p]//4]*4
        assert .4<=summ[p]["label0_fraction"]<=.6
        x,y=r60.cohort(p)
        assert x.shape==(r60.CASES[p],10)
        assert x.dtype is torch.float64
        assert y.dtype is torch.int64
        for idx in (0,1,r60.CASES[p]-1):
            q=r60.selected_case(p,idx)
            assert len(q["selected_public_records"])==12
            assert len(q["model_feature_vector"])==10
            assert q["model_feature_vector"]==pytest.approx(x[idx].tolist(),abs=1e-12)
            for bridge in range(3):
                obs=q["selected_public_records"][bridge*4:(bridge+1)*4]
                assert [z["actually_selected_action"] for z in obs]==[0,0,1,1]
                assert [z["sensor"] for z in obs]==[
                    "terminal_goal","noncausal_proxy",
                    "terminal_goal","noncausal_proxy"]
                assert all(z["public_probe"]==bridge for z in obs)
                assert (obs[0]["observed_effect"]-obs[2]["observed_effect"])*.5==pytest.approx(
                    x[idx,bridge+3].item(),abs=1e-12)
                assert (obs[1]["observed_effect"]-obs[3]["observed_effect"])*.5==pytest.approx(
                    x[idx,bridge+6].item(),abs=1e-12)
            if p.startswith("FIT_"):
                trails=q["FIT_selected_future_return_trials_evaluator_only"]
                assert len(trails)==2
                assert [z["actually_selected_action"] for z in trails]==[0,1]
            else:
                assert "FIT_selected_future_return_trials_evaluator_only" not in q


def test_original_policy_shapes_with_explicit_double_precision():
    for arm,seed in r60.METHODS:
        m=r60.Policy(arm,seed)
        assert all(p.dtype==torch.float64 for p in m.parameters())
        with torch.no_grad():
            out=m(torch.zeros((3,10),dtype=torch.float64))
        assert out.shape==(3,2)
        assert torch.isfinite(out).all()
        with pytest.raises(ValueError,match="INVALID_OBSERVED_INPUT"):
            m(torch.zeros((3,10),dtype=torch.float32))
    assert r60.TIE_EPS==.003


def test_frozen_new_source_6144_real_FP64_steps_negative_controls_and_cold():
    z=r60.run()
    assert z["schema"]==r60.SCHEMA
    assert z["freeze_blob"]==r60.FREEZE_BLOB
    assert z["parent"]==r60.PARENT
    assert z["cohort_sizes"]==r60.CASES
    assert z["actual_optimizer_attempts"]==6144
    assert z["actual_FIT_only_label_shuffle_updates"]==256
    assert len(z["models"])==4
    assert z["independent_scientific_pass"] is False
    assert z["novel_model_trained"] is False
    assert z["external_original_firstuse"] is False
    assert z["deployment"]=="NONE"
    assert z["judgment"]["crossplatform_exact_decision_parity"]==(
        "UNVERIFIED_UNTIL_ORIGINAL_BOTH_ARTIFACTS")
    assert z["judgment"]["frozen_final_admission"]==(
        "UNVERIFIED_UNTIL_BOTH_PLATFORM_PARITY")
    assert len(z["judgment"]["seeds"])==2
    for m in z["models"]:
        assert m["cold_restore"]
        assert m["actual_FIT_optimizer_updates"]==1536
        assert m["productive_gradient_steps"]>0
        assert [v["step"] for v in m["trace"]]==list(r60.CHECKPOINTS)
        for panel in r60.PANELS:
            assert len(m["scores"][panel]["active"]["choices"])==r60.CASES[panel]
        assert len(m["scores"]["OOD"]["erase_selected_terminal"]["choices"])==1024
        assert len(m["scores"]["OOD"]["erase_noncausal_proxy"]["choices"])==1024
    r60.retain_ci_receipt(z)
