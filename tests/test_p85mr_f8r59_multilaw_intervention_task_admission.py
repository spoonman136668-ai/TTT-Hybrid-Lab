"""F8-R59 prospective task-admission: genuine selected intervention outcomes.

Tests preserve the originally frozen protocol and report negative outcomes
without changing the cohort or prespecified success thresholds.
"""
import pytest
import torch

from experiments import p85mr_f8r59_multilaw_intervention_task_admission as r59


def test_frozen_protocol_and_original_external_first_use_boundary():
    freeze=r59.locked()
    assert freeze["schema"]==r59.SCHEMA
    assert freeze["stage"]=="FROZEN_BEFORE_R59_FORMAL_SOURCE_OR_TRAINING"
    assert r59.git_blob(r59.FREEZE.read_bytes())==r59.FREEZE_BLOB
    assert freeze["parent"]["r58_sha"]==r59.PARENT
    assert freeze["source"]["salts"]==r59.SALT
    assert freeze["source"]["cases"]==r59.CASES
    assert freeze["source"]["no_old_heldout_reuse"]
    assert freeze["trained_controls"]["optimizer_attempts_total"]==6144
    assert freeze["pre_freeze_pilot"]["status"]=="DEV_ONLY_NOT_FORMAL_SCIENTIFIC_EVIDENCE"
    assert freeze["admission"]["independent_scientific_pass"] is False
    assert not any(freeze["authority"].values())
    with pytest.raises(PermissionError,match="EXTERNAL_ORIGINAL_FIRSTUSE_DENIED"):
        r59.deny_external_original_firstuse("external")
    with pytest.raises(ValueError,match="UNREGISTERED_SELECTED_CASE"):
        r59.selected_case("OOD",1024)
    with pytest.raises(ValueError,match="UNREGISTERED_PANEL"):
        r59.cohort("unregistered")


def test_four_genuinely_distinct_laws_and_selected_action_consequences():
    summary=r59.source_sanity()
    assert set(summary)==set(r59.PANELS)
    for panel in r59.PANELS:
        assert summary[panel]["family_counts"]==[r59.CASES[panel]//4]*4
        assert .4<=summary[panel]["label0_fraction"]<=.6
        x,y=r59.cohort(panel)
        assert x.shape==(r59.CASES[panel],10)
        assert y.shape==(r59.CASES[panel],)
        for idx in (0,1,2,3,100,r59.CASES[panel]-1):
            case=r59.selected_case(panel,idx)
            assert case["panel_evaluator_only"]==panel
            events=case["selected_public_records"]
            assert len(events)==12
            assert len(case["model_feature_vector"])==10
            assert case["model_feature_vector"]==pytest.approx(x[idx].tolist(),
                                                                abs=1e-7)
            for bridge in range(3):
                records=events[4*bridge:4*(bridge+1)]
                assert [r["actually_selected_action"] for r in records]==[
                    0,0,1,1]
                assert [r["sensor"] for r in records]==[
                    "terminal_goal","noncausal_proxy",
                    "terminal_goal","noncausal_proxy"]
                assert all(r["public_probe"]==bridge for r in records)
                assert records[0]["public_x"]==records[2]["public_x"]
                selected_terminal=(
                    records[0]["observed_effect"]-
                    records[2]["observed_effect"])*.5
                selected_proxy=(
                    records[1]["observed_effect"]-
                    records[3]["observed_effect"])*.5
                assert selected_terminal==pytest.approx(
                    x[idx,3+bridge].item(),abs=1e-6)
                assert selected_proxy==pytest.approx(
                    x[idx,6+bridge].item(),abs=1e-6)
            if panel.startswith("FIT_"):
                ret=case["FIT_selected_future_return_trials_evaluator_only"]
                assert [v["actually_selected_action"] for v in ret]==[0,1]
                assert len(ret)==2
                assert ret[0]["observed_goal"]==-ret[1]["observed_goal"]
            else:
                assert "FIT_selected_future_return_trials_evaluator_only" not in case
            assert set(case)-{
                "panel_evaluator_only","selected_public_records",
                "public_target_x","model_feature_vector",
                "future_goal_label_evaluator_only",
                "FIT_selected_future_return_trials_evaluator_only"}==set()


def test_strong_fixed_interpolator_and_weak_baseline_metrics():
    r59.locked()
    fixed=r59.weak_controls()
    assert set(fixed)==set(r59.PANELS)
    for panel in r59.PANELS:
        v=fixed[panel]
        assert len(v["fixed_quadratic_choices"])==r59.CASES[panel]
        assert len(v["single_center_choices"])==r59.CASES[panel]
        assert len(v["majority_choices"])==r59.CASES[panel]
        assert len(v["constant_choices"])==r59.CASES[panel]
        for k in ("fixed_quadratic_interpolation_accuracy",
                  "single_center_probe_accuracy",
                  "majority_three_selected_probes_accuracy",
                  "constant_action0_accuracy"):
            assert 0.<=v[k]<=1.


def test_both_conventional_models_share_selected_goal_context_information():
    r59.locked()
    for arm,seed in r59.METHODS:
        m=r59.Policy(arm,seed)
        x=torch.zeros((3,10),dtype=torch.float32)
        with torch.no_grad():
            logits=m(x)
        assert logits.shape==(3,2)
        assert torch.isfinite(logits).all()
        assert sum(p.numel() for p in m.parameters())>0
        with pytest.raises(ValueError,match="INVALID_OBSERVED_INPUT"):
            m(torch.zeros((3,8)))


def test_full_frozen_baseline_training_and_negative_task_result_preservation():
    result=r59.run()
    assert result["schema"]==r59.SCHEMA
    assert result["freeze_blob"]==r59.FREEZE_BLOB
    assert result["parent"]==r59.PARENT
    assert result["source_neutral_task_admission_only"]
    assert result["cohort_sizes"]==r59.CASES
    assert result["actual_optimizer_attempts"]==6144
    assert result["actual_FIT_only_label_shuffle_updates"]==256
    assert result["FIT_label_shuffle_negative"]["no_OOD_training"]
    assert len(result["models"])==4
    assert len(result["judgment"]["seed_scores"])==2
    assert result["independent_scientific_pass"] is False
    assert result["judgment"]["independent_scientific_pass"] is False
    assert result["novel_model_trained"] is False
    assert result["external_original_firstuse"] is False
    assert result["deployment"]=="NONE"
    for m in result["models"]:
        assert m["cold_restore"]
        assert m["actual_FIT_optimizer_updates"]==1536
        assert 0<=m["productive_gradient_steps"]<=1536
        assert m["trainable_parameters"]>0
        assert [x["step"] for x in m["trace"]]==list(r59.CHECKPOINTS)
        for panel in r59.PANELS:
            assert len(m["scores"][panel]["active"]["choices"])==r59.CASES[panel]
            assert 0<=m["scores"][panel]["active"]["accuracy"]<=1
        for k in ("erase_selected_terminal","erase_noncausal_proxy"):
            assert len(m["scores"]["OOD"][k]["choices"])==r59.CASES["OOD"]
        assert set(m["OOD_accuracy_per_law_family"])=={"0","1","2","3"}
    for seed in result["judgment"]["seed_scores"]:
        assert seed["seed"] in r59.SEEDS
        assert isinstance(seed["seed_task_admission_pass"],bool)
    assert isinstance(result["judgment"]["task_admitted"],bool)
    r59.retain_ci_receipt(result)
