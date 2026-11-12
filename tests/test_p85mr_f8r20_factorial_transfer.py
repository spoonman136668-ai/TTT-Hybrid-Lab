"""R19 registered factorial causal-memory controls, source-neutral only."""
import inspect
import math

import pytest
import torch

from experiments import p85mr_f8r20_factorial_transfer as x

torch.set_num_threads(1)


def test_freeze_pin_and_unqualified_source_denial():
    f=x.locked()
    assert f["stage"]=="FROZEN_BEFORE_IMPLEMENTATION_AND_NUMERIC_RESULTS"
    assert f["model"]["total_optimizer_updates"]==6912
    assert f["authority"]["original_external_source_firstuse"] is False
    assert f["authority"]["new_workflow"] is False
    with pytest.raises(PermissionError,match="FIRSTUSE_NOT_AUTHORIZED"):
        x.refuse_original_source_firstuse()


def test_original_factual_fit_factorial_coverage_and_strict_negative_bounds():
    eps=x.cohort("fit")
    receipt=x.verify_matched(eps,"fit")
    assert receipt=={"coverage_per_A_B_return_cell":24,"factorial_cells":8,"episodes":192}
    assert sum(len(ep["effect"]) for ep in eps)==1920
    assert {ep["z_b"] for ep in eps}=={-1,1}
    assert {ep["z_a"] for ep in eps}=={-1,1}
    assert {ep["factual_return"] for ep in eps}=={0,1}
    assert all(set(x.factual(ep))=={"action","effect"} for ep in eps)
    with pytest.raises(ValueError,match="UNREGISTERED_GROUP"):
        x.episode("id",12,1,1)
    with pytest.raises(ValueError,match="UNREGISTERED_LATENT_SIGN"):
        x.episode("fit",0,0,1,0)
    with pytest.raises(ValueError,match="FIT_REQUIRES_FACTUAL"):
        x.episode("fit",0,1,1)
    with pytest.raises(ValueError,match="SCORE_MUST_NOT_OBSERVE"):
        x.episode("ood",0,1,1,0)


def test_matched_B_identifiability_both_horizons_and_analytical_oracle():
    for split, groups, length in (("dev",8,9),("id",12,9),("ood",12,13)):
        eps=x.cohort(split)
        v=x.verify_matched(eps,split)
        assert v["identical_B_paired"]
        assert v["groups"]==groups
        for i in range(0,len(eps),2):
            a,b=eps[i:i+2]
            assert a["z_a"]==-1 and b["z_a"]==1 and a["z_b"]==b["z_b"]
            assert torch.equal(a["action"],b["action"])
            assert torch.equal(a["effect"][1:],b["effect"][1:])
            assert not torch.equal(a["effect"][0],b["effect"][0])
            assert a["action"].shape==(length,16)
            assert a["effect"].shape==(length,16)
        if split in ("id","ood"):
            controls=x.analytic_controls(eps,split)
            assert math.isclose(controls["means"]["always1"],.015,abs_tol=1e-12)
            assert math.isclose(controls["means"]["always0"],.105,abs_tol=1e-12)
            assert controls["oracle_early_A"]==0.
            assert controls["best_history_free"]=="always1"


def test_native_and_controls_mask_no_current_effect_and_true_probe_negative():
    pair=x.cohort("id")[:2]
    keys=[torch.tensor(x.r18.r5._action_key((0.,0.,0.,0.),a),
                       dtype=torch.float32) for a in (0,1)]
    for arm in x.ARMS:
        model=x.r19._model(x.MAPPING[arm],985201).cpu().eval()
        original=x._predict_actions(arm,model,pair)
        masked=x._predict_actions(arm,model,pair,erase_A=True)
        assert original.shape==masked.shape==(2,2)
        assert torch.allclose(torch.tensor(masked[0]),
                              torch.tensor(masked[1]),atol=1e-6,rtol=0)
        ep=pair[0]
        aa=torch.cat((ep["action"],keys[0].unsqueeze(0)))
        base=torch.cat((ep["effect"],torch.zeros((1,16))))
        altered=base.clone(); altered[-1,:7]=.5
        with torch.no_grad():
            first=x.r19._pred(x.MAPPING[arm],model,{"action":aa[None],"effect":base[None]})
            second=x.r19._pred(x.MAPPING[arm],model,{"action":aa[None],"effect":altered[None]})
        assert torch.allclose(first[:,-1],second[:,-1],atol=1e-6,rtol=0)


def test_eighteen_trained_real_gradient_development_and_cold_restore():
    r=x.run()
    assert r["model_count"]==18 and r["training_optimizer_updates"]==6912
    assert r["synthetic_model_selected_decisions"]==3456
    assert r["factual_FIT_sessions"]==192 and r["factual_FIT_effects"]==1920
    assert r["original_external_source_effects"]==0
    assert not r["F7_replayed"] and not r["independent_scientific_pass"]
    assert r["conventional_independent_convergence"] is False
    assert r["unfamiliar_B_length_tested"]==12 and r["trained_B_length"]==8
    assert len(r["decision"]["candidate_seeds"])==3
    assert r["decision"]["candidate_dev_gate_pass"] in (True,False)
    assert {z["arm"] for z in r["models"]}==set(x.ARMS)
    assert {z["seed"] for z in r["models"]}==set(x.SEEDS)
    for model in r["models"]:
        assert model["updates"]==384
        assert model["parameters"]>0
        assert model["original_weights_sha256"]!=model["trained_weights_sha256"]
        assert model["cold_restored_exact_action_parity"] is True
        assert [d["updates"] for d in model["diagnostic_dev_learning_curve"]]==[128,256,384]
        assert math.isfinite(model["train_seconds"])
        assert math.isfinite(model["first32_fit_loss"])
        assert math.isfinite(model["last32_fit_loss"])
        for part in ("id","ood"):
            sc=model[part]
            assert len(sc["pergroup_regret"])==12
            assert len(sc["pergroup_probe_erased"])==12
            assert sc["model_selected_actions"]==96
            assert sc["two_action_queries"]==96
            assert len(sc["chosen_actions"])==96
            assert math.isfinite(sc["mean_regret"])
            assert math.isfinite(sc["mean_probe_erased_regret"])
            assert 0<=sc["A_sign_action_contrast"]<=sc["A_sign_action_contrast_opportunities"]
    x.retain_ci_receipt(r)


def test_no_external_simulator_action_or_new_runner():
    source=inspect.getsource(x)
    for forbidden in ("import causal_gym","from causal_gym",
                      "env.reset(","env.do(","ctf_do(",
                      "workflow_dispatch(","subprocess.run(",
                      "requests.get(","urlopen("):
        assert forbidden not in source
