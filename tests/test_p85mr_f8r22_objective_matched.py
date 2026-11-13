"""R22 native vs GRU/SSM matched factual objective, full source-neutral regression."""
import inspect
import math
import torch
import pytest

from experiments import p85mr_f8r22_objective_matched as x

torch.set_num_threads(1)


def test_frozen_science_and_no_original_firstuse():
    f=x.locked()
    assert f["status"]=="FROZEN_BEFORE_IMPLEMENTATION_TRAINING_AND_RESULTS"
    assert f["models"]["total_optimizer_updates"]==6912
    assert f["models"]["same_focused_objective_native_GRU_SSM_MLP"]
    assert f["authority"]["external_scientific_firstuse"] is False
    with pytest.raises(PermissionError,match="FIRSTUSE_NOT_AUTHORIZED"):
        x.refuse_original_firstuse()


def test_matched_latent_B_and_complete_factual_FIT_support():
    for name in x.COHORTS:
        es=x.cohort(name)
        q=x.factorial_audit(es,name)
        if name=="fit":
            assert q["eight_factorial_cells"]==8
            assert q["per_cell_factual_FIT"]==32
            assert len(es)==256
        else:
            assert q["matched_B_pairs"]==2*x.COHORTS[name][1]
            assert q["group_clusters"]==x.COHORTS[name][1]
            assert all(e["selected_return"] is None for e in es)
    assert x.COHORTS["ood"][2]==16
    with pytest.raises(ValueError,match="INVALID_SPLIT_OR_GROUP"):
        x.episode("id",12,-1,-1)
    with pytest.raises(ValueError,match="INVALID_FACTUAL_RETURN"):
        x.episode("ood",0,1,-1,0)
    with pytest.raises(ValueError,match="INVALID_REGIME"):
        x.episode("fit",0,0,-1,0)


def test_pre_action_mask_and_no_recent_B_identifiability():
    eps=x.cohort("ood")[:2]
    for arm in x.ARMS:
        model=x.model_for(arm,985201).eval()
        pred=x._hypothetical(arm,model,eps,erase_A=True)
        assert pred.shape==(2,2)
        assert torch.allclose(torch.tensor(pred[0]),torch.tensor(pred[1]),
                              atol=1e-6,rtol=0)
        assert all(math.isfinite(float(v)) for v in pred.reshape(-1))


def test_FIT_only_causal_memory_vs_short_history_control():
    fit=x.cohort("fit")
    score=x.cohort("ood")
    early=x.fit_ridge(fit,use_early=True)
    short=x.fit_ridge(fit,use_early=False)
    a=x.ridge_score(early,score,"ood")
    b=x.ridge_score(short,score,"ood")
    assert a["mean_regret"]<.001
    assert b["mean_regret"]>=.015-1e-10


def test_all_actual_18_trained_model_runs_and_checkpoint_restore():
    out=x.run()
    assert out["model_count"]==18
    assert out["optimizer_updates"]==6912
    assert out["synthetic_selected_FIT_effects"]==2560
    assert out["original_external_source_effects"]==0
    assert out["spent_F7_replay"] is False
    assert out["scientific_pass"] is False
    assert out["conventional_convergence_independently_qualified"] is False
    assert out["decision"]["native_frozen_DEV_PASS"] in (True,False)
    assert len(out["decision"]["seed_decisions"])==3
    for arm in x.ARMS:
        assert len([r for r in out["models"] if r["arm"]==arm])==3
    for row in out["models"]:
        assert row["updates"]==384
        assert row["original_weight_hash"]!=row["trained_weight_hash"]
        assert row["cold_restored_exact_action_parity"]
        assert len(row["dev_convergence_curve"])==4
        assert all(math.isfinite(z["DEV_regret"]) for z in row["dev_convergence_curve"])
        assert math.isfinite(row["train_seconds"])
        for split in ("id","ood"):
            assert len(row[split]["pergroup_regret"])==12
            assert row[split]["contrast_opportunities"]==48
            assert 0<=row[split]["A_sign_choice_contrasts"]<=48
    x.retain_ci_receipt(out)


def test_no_external_environment_source_or_controller():
    src=inspect.getsource(x)
    for forbidden in ("import causal_gym","from causal_gym","env.reset(",
                      "env.do(","workflow_dispatch(", "subprocess.run(",
                      "requests.get(","urlopen("):
        assert forbidden not in src
