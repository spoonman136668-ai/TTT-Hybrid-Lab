"""F8-R23 truly different world dynamics + independent B interference controls."""
import inspect
import math

import pytest
import torch

from experiments import p85mr_f8r23_novel_causal as x

torch.set_num_threads(1)


def test_prospective_frozen_source_and_owner_deny():
    f=x.locked()
    assert f["status"]=="FROZEN_BEFORE_NEW_IMPLEMENTATION_AND_NUMERIC_RESULTS"
    assert f["models"]["total_optimizer_updates"]==3456
    assert f["authority"]["original_source_firstuse"] is False
    assert f["thresholds"]["scientific_pass"] is False
    with pytest.raises(PermissionError,match="FIRSTUSE_NOT_AUTHORIZED"):
        x.refuse_original_external_source()
    with pytest.raises(ValueError,match="UNREGISTERED_PANEL_GROUP"):
        x.episode("nonlinear_opposing_actions",12,-1,1)
    with pytest.raises(ValueError,match="INVALID_LAW_INPUT"):
        x.law("nonlinear_opposing_actions",0.,True,1)


def test_new_law_nonzero_return_action1_and_nonlinear_curvature():
    assert x.law("nonlinear_opposing_actions",0.,0,1)==pytest.approx(-.24)
    assert x.law("nonlinear_opposing_actions",0.,1,1)==pytest.approx(+.24)
    assert x.law("nonlinear_opposing_actions",.5,0,1)!=pytest.approx(
        .5*x.law("nonlinear_opposing_actions",1.,0,1))
    assert x.law("two_B_regime_interference",0.,1,1)==0


def test_matched_panel_B_and_analytical_full_information_barrier():
    for panel in x.PANELS:
        episodes=x.cohort(panel)
        assert len(episodes)==48
        assert x.identity(episodes,panel)=={
            "matched_A_pairs":24,"B_actions_effects_identical":True,
            "first_return_effect_observed":False,"no_original_source_effects":True}
        ctl=x.analytic(episodes,panel)
        expected=.096 if panel=="nonlinear_opposing_actions" else .015
        assert math.isclose(ctl["no_history_mean_regret"],expected,abs_tol=1e-12)
        assert ctl["oracle_regret"]==0.
        for i in range(0,len(episodes),2):
            a,b=episodes[i:i+2]
            assert torch.equal(a["action"],b["action"])
            assert torch.equal(a["effect"][1:],b["effect"][1:])
            assert a["action"].shape==(17,16)
            assert not torch.equal(a["effect"][0],b["effect"][0])


def test_fit_only_original_memory_controls_no_test_training():
    fit=x.r22.cohort("fit")
    for panel in x.PANELS:
        episodes=x.cohort(panel)
        early=x.ridge_score(x.r22.fit_ridge(fit,use_early=True),episodes,panel)
        last=x.ridge_score(x.r22.fit_ridge(fit,use_early=False),episodes,panel)
        assert math.isfinite(early["mean_regret"])
        assert math.isfinite(last["mean_regret"])
        assert early["uses_early_factual_A_memory"] is True
        assert last["uses_early_factual_A_memory"] is False


def test_original_native_and_strong_conventional_preaction_causality():
    eps=x.cohort("nonlinear_opposing_actions")[:2]
    for arm in x.ARMS:
        model=x.r22.model_for(arm,985201).cpu().eval()
        live=x.hypothetical(arm,model,eps)
        erased=x.hypothetical(arm,model,eps,erase_early=True)
        assert live.shape==erased.shape==(2,2)
        assert torch.allclose(torch.tensor(erased[0]),
                              torch.tensor(erased[1]),atol=1e-6,rtol=0)
        assert all(math.isfinite(v) for v in live.reshape(-1))


def test_nine_actual_models_frozen_384_steps_cold_retention_and_receipt():
    r=x.run()
    assert r["model_count"]==9
    assert r["total_optimizer_updates"]==3456
    assert r["trained_original_R22_FIT_sessions"]==256
    assert r["original_CausalGym_external_effects"]==0
    assert not r["F7_original_replay"]
    assert not r["independent_scientific_pass"]
    assert not r["conventional_independently_convergence_qualified"]
    assert r["deployment"]=="NONE"
    assert len(r["models"])==9
    assert len(r["decisions"]["native_seed_decisions"])==3
    assert r["decisions"]["retention_dev_gate_pass"] in (True,False)
    for model in r["models"]:
        assert model["steps"]==384
        assert model["parameter_count"]>0
        assert model["initial_sha"]!=model["trained_sha"]
        assert model["cold_restored_actions_equal"] is True
        for name in x.PANELS:
            score=model["panels"][name]
            assert score["selected_decisions"]==96
            assert score["two_action_predictions"]==96
            assert score["contrast_opportunities"]==48
            assert len(score["paired_group_regret"])==12
            assert math.isfinite(score["mean_regret"])
            assert math.isfinite(score["probe_erased_mean_regret"])
    x.retain_ci_receipt(r)


def test_no_original_world_source_runner_or_promotion():
    src=inspect.getsource(x)
    for forbidden in ("import causal_gym","from causal_gym","env.reset(",
                      "env.do(","ctf_do(","workflow_dispatch(",
                      "subprocess.run(","requests.get(","urlopen("):
        assert forbidden not in src
