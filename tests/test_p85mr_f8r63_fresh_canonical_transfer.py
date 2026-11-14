"""F8-R63 prospective fresh-source selected intervention task admission."""
import pytest
import torch
from experiments import p85mr_f8r63_fresh_canonical_transfer as r63


def test_prospective_source_freeze_and_external_firstuse_denial():
    f=r63.locked()
    assert f["schema"]==r63.SCHEMA
    assert r63.git_blob(r63.FREEZE.read_bytes())==r63.FREEZE_BLOB
    assert f["stage"]=="FROZEN_BEFORE_R63_SOURCE_OR_SCORED_RESULTS"
    assert f["parent"]["r62_SHA"]==r63.PARENT
    assert f["source"]["new_salts"]==r63.SALT
    assert f["source"]["cases"]==r63.CASES
    assert f["parent"]["no_old_R49_R62_scored_cohorts_reused"]
    assert f["models"]["total_optimizer_attempts"]==6144
    assert f["models"]["expected_all_crossplatform_actions"]==61440
    assert f["frozen_acceptance"]["exact_all_61440_decisions_OS_parity"]
    assert not f["frozen_acceptance"]["independent_scientific_pass"]
    assert not any(f["authority"].values())
    with pytest.raises(PermissionError,match="EXTERNAL_ORIGINAL_FIRSTUSE_DENIED"):
        r63.reject_external_original_firstuse()


def test_fresh_four_law_selected_actions_nuisance_and_target_custody():
    summary=r63.source_sanity()
    assert set(summary)==set(r63.PANELS)
    assert r63.SALT["FIT_train"]!=r63.r60.SALT["FIT_train"]
    assert r63.SALT["OOD"]!=r63.r60.SALT["OOD"]
    for panel in r63.PANELS:
        assert summary[panel]["family_counts"]==[r63.CASES[panel]//4]*4
        assert .4<=summary[panel]["label0_fraction"]<=.6
        x,y=r63.cohort(panel)
        assert x.shape==(r63.CASES[panel],10)
        assert x.dtype==torch.float64 and y.dtype==torch.long
        for idx in (0,1,2,r63.CASES[panel]-1):
            w=r63.selected_case(panel,idx)
            assert len(w["selected_factual_public_records"])==12
            assert w["model_public_feature_vector"]==pytest.approx(
                x[idx].tolist(),abs=1e-12)
            for bridge in range(3):
                rows=w["selected_factual_public_records"][bridge*4:(bridge+1)*4]
                assert [o["actually_selected_action"] for o in rows]==[0,0,1,1]
                assert [o["sensor"] for o in rows]==[
                    "terminal_goal","noncausal_proxy",
                    "terminal_goal","noncausal_proxy"]
                assert all(o["public_probe"]==bridge for o in rows)
                assert (rows[0]["observed_effect"]-rows[2]["observed_effect"])*.5==pytest.approx(
                    x[idx,3+bridge].item(),abs=1e-12)
                assert (rows[1]["observed_effect"]-rows[3]["observed_effect"])*.5==pytest.approx(
                    x[idx,6+bridge].item(),abs=1e-12)
            if panel.startswith("FIT_"):
                ret=w["FIT_only_actual_selected_target_action_trials_evaluator_only"]
                assert len(ret)==2
                assert ret[0]["selected_return_effect"]==-ret[1]["selected_return_effect"]
            else:
                assert "FIT_only_actual_selected_target_action_trials_evaluator_only" not in w
        assert int((y==0).sum())>0 and int((y==1).sum())>0
    with pytest.raises(ValueError,match="INVALID_SELECTED_CASE"):
        r63.selected_case("OOD",1024)


def test_frozen_canonical_MLP_initial_bytes_and_matched_baseline_features():
    r63.locked()
    for arm,seed in r63.METHODS:
        m,h=r63.initialize(arm,seed)
        assert all(p.dtype==torch.float64 for p in m.parameters())
        assert sum(p.numel() for p in m.parameters())>0
        with torch.no_grad():
            logits=m(torch.zeros((3,10),dtype=torch.float64))
        assert logits.shape==(3,2)
        if arm==r63.ARMS[0]:
            assert h==r63.r62.EXPECTED[seed]
        else:
            assert h is None


def test_strong_and_weak_public_baseline_on_exact_source():
    fixed=r63.fixed_controls()
    assert set(fixed)==set(r63.PANELS)
    for panel in r63.PANELS:
        for name in ("fixed_quadratic","single_center","majority","constant"):
            x=fixed[panel][name]
            assert len(x["choices"])==r63.CASES[panel]
            assert 0<=x["accuracy"]<=1


def test_6144_prescribed_FIT_steps_all_original_actions_negative_controls():
    result=r63.run()
    assert result["schema"]==r63.SCHEMA
    assert result["freeze_blob"]==r63.FREEZE_BLOB
    assert result["parent"]==r63.PARENT
    assert result["source_salts"]==r63.SALT
    assert result["cohort_sizes"]==r63.CASES
    assert result["selected_factual_probe_outcomes_per_world"]==12
    assert result["optimizer_attempts"]==6144
    assert result["all_categorical_decision_count"]==61440
    assert len(result["models"])==4
    assert len(result["judgment"]["original_seeds"])==2
    assert result["judgment"]["cross_OS_parity"]==(
        "UNVERIFIED_UNTIL_ORIGINAL_BOTH_OS_ARTIFACTS")
    assert result["judgment"]["final_frozen_admission"]==(
        "UNVERIFIED_UNTIL_CROSS_OS_COMPARISON")
    assert not result["independent_scientific_pass"]
    assert not result["Native_specific_gain"]
    assert not result["external_original_firstuse"]
    assert result["deployment"]=="NONE"
    assert result["FIT_only_label_shuffle_negative"]["FIT_only"]
    assert result["FIT_only_label_shuffle_negative"]["steps"]==256
    for instance in result["models"]:
        assert instance["cold_restore"]
        assert instance["productive_FIT_steps"]==1536
        assert [x["step"] for x in instance["FIT_trace"]]==list(r63.CHECKPOINTS)
        assert len(instance["per_law_OOD_accuracy"])==4
        if instance["arm"]==r63.ARMS[0]:
            assert instance["canonical_MLP_init_sha256"]==(
                r63.r62.EXPECTED[instance["seed"]])
        for panel in r63.PANELS:
            assert len(instance["scores"][panel]["active"]["choices"])==(
                r63.CASES[panel])
        for ab in ("erase_selected_terminal","erase_noncausal_proxy"):
            assert len(instance["scores"]["OOD"][ab]["choices"])==1024
    r63.retain_ci_receipt(result)
