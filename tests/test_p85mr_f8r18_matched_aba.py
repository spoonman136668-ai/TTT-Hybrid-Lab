"""F8-R18: crossed hidden A, identical B, source-neutral real neural training."""
from __future__ import annotations

import inspect
import math

import pytest
import torch

from experiments import p85mr_f8r18_matched_aba as x

torch.set_num_threads(1)


def test_frozen_protocol_and_owner_denial():
    p=x.locked()
    assert p["stage"]=="FROZEN_BEFORE_CODE_TRAINING_RESULTS"
    assert p["training"]["total_optimizer_updates"]==1920
    assert p["authority"]["external_source_firstuse"] is False
    assert p["score"]["independent_scientific_pass"] is False
    with pytest.raises(PermissionError,match="FIRSTUSE_NOT_AUTHORIZED"):
        x.refuse_original_firstuse()


def test_identical_B_history_A_probe_only_and_no_early_test_return_effect():
    for split,expected in (("fit",64),("dev",24),("score",32)):
        eps=x.pair_cohort(split)
        assert len(eps)==expected
        for k in range(0,len(eps),2):
            a,b=eps[k:k+2]
            assert a["sign_a"]==-1 and b["sign_a"]==1
            assert a["sign_b"]==b["sign_b"]
            assert a["group"]==b["group"]
            assert torch.equal(a["action"],b["action"])
            assert not torch.equal(a["effect"][0],b["effect"][0])
            assert torch.equal(a["effect"][1:9],b["effect"][1:9])
            assert torch.equal(a["action"][1:9],b["action"][1:9])
            assert a["return_state"]==b["return_state"]==0.
            assert set(x.factual(a))=={"action","effect"}
            assert a["action"].shape==(10 if split=="fit" else 9,16)
            assert a["effect"].shape==(10 if split=="fit" else 9,16)
        assert {p["sign_b"] for p in eps}=={-1,1}
    with pytest.raises(ValueError,match="UNREGISTERED_PAIR_WORLD"):
        x.episode("score",16,-1)
    with pytest.raises(ValueError,match="UNREGISTERED_PAIR_WORLD"):
        x.episode("fit",True,1)
    with pytest.raises(ValueError,match="UNREGISTERED_PAIR_WORLD"):
        x.episode("fit",0,0)


def test_matched_B_fixed_policy_identifiability_bound_and_real_oracle():
    pairs=x.pair_cohort("score")
    controls=x._fixed_controls(pairs)
    assert controls["means"]["fixed1"]==pytest.approx(.015,abs=1e-12)
    assert controls["means"]["fixed0"]==pytest.approx(.105,abs=1e-12)
    assert controls["analytic_A_memory_oracle_mean_regret"]==0.
    assert all(v>=.015-1e-12 for v in controls["means"].values())
    for sign in (-1,1):
        for goal in x.GOALS:
            a=min((0,1),key=lambda act:(
                (x.r3.synthetic_effect(0.,act,sign)-goal)**2,act))
            assert x._regret(sign,goal,a)==pytest.approx(0.,abs=1e-12)


def test_future_effect_mask_and_zero_A_probe_countercontrol():
    pair=x.pair_cohort("score")[:2]
    for arm in x.ARMS:
        model=x._model(arm,985201).cpu().eval()
        clean=x._state_predictions(arm,model,pair)
        # The two copies have identical public current states and B history;
        # only the early observed A effect can distinguish them.
        neutral=x._state_predictions(arm,model,pair,erase_first_A=True)
        assert neutral.shape==(2,2)
        assert torch.allclose(torch.tensor(neutral[0]),torch.tensor(neutral[1]),atol=1e-6,rtol=0)
        assert math.isfinite(float(clean[0,0]))
        with pytest.raises(ValueError,match="INVALID_EVALUATOR_INPUT"):
            x._regret(3,.2,0)


def test_full_actual_trained_native_and_controls_aba_falsification():
    r=x.run()
    assert r["classification"]=="SOURCE_NEUTRAL_TRAINED_NATIVE_DEV_ONLY"
    assert r["model_count"]==15 and r["optimizer_updates"]==1920
    assert r["synthetic_fit_factual_effects"]==640
    assert r["synthetic_score_observed_early_A_and_B_effects"]==288
    assert r["score_model_selected_synthetic_actions"]==960
    assert r["new_external_effects"]==0 and r["original_F7_replay"] is False
    assert r["score_identical_B_paired_groups"]==16
    assert not r["independent_scientific_pass"]
    assert not r["conventional_training_convergence_qualified"]
    assert r["candidate_dev_gate_pass"] in (True,False)
    assert len(r["candidate_seed_decisions"])==3
    assert {z["arm"] for z in r["models"]}==set(x.ARMS)
    assert {z["seed"] for z in r["models"]}==set(x.SEEDS)
    for row in r["models"]:
        assert row["updates"]==128
        assert row["model_parameters"]>0
        assert row["original_weight_hash"]!=row["trained_weight_hash"]
        assert row["cold_model_restore_exact_action_parity"]
        assert math.isfinite(row["train_seconds"])
        for phase in ("dev","test"):
            s=row[phase]
            assert len(s["per_group_regret"])==(12 if phase=="dev" else 16)
            assert s["model_selected_synthetic_actions"]==(48 if phase=="dev" else 64)
            assert s["hypothetical_queries"]==(48 if phase=="dev" else 64)*2
            assert s["A_sign_action_contrast_pairs"]==(24 if phase=="dev" else 32)
            assert 0<=s["A_sign_action_contrast_count"]<=s["A_sign_action_contrast_pairs"]
            assert math.isfinite(s["mean_regret"])
            assert math.isfinite(s["erased_initial_A_mean_regret"])
    x.retain_ci_receipt(r)


def test_source_contains_no_external_environment_or_owner_dispatch():
    src=inspect.getsource(x)
    for forbidden in ("import causal_gym","from causal_gym",
                      "env.reset(", "env.do(", "ctf_do(", "workflow_dispatch(",
                      "subprocess.run(", "requests.get(", "urlopen("):
        assert forbidden not in src
