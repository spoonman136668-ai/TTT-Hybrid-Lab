"""F8-R19 actual trained native vs conventional balanced causal memory."""
from __future__ import annotations

import inspect
import math

import pytest
import torch

from experiments import p85mr_f8r19_balanced_return as x

torch.set_num_threads(1)


def test_frozen_protocol_and_source_authority():
    f=x.locked()
    assert f["stage"]=="PRECOMMITTED_BEFORE_CODE_TRAINING_AND_RESULTS"
    assert f["training"]["total_optimizer_updates"]==2400
    assert f["source"]["return_A"]["balanced_four_way_FIT"]
    assert f["authority"]["original_F8_external_firstuse"] is False
    assert f["evaluation"]["scientific_pass"] is False
    with pytest.raises(PermissionError,match="FIRSTUSE_NOT_AUTHORIZED"):
        x.refuse_original_source_firstuse()


def test_complete_factorial_fit_realized_actions_and_B_pair_parity():
    fit=x.cohort("fit")
    audit=x.audit_fit_support(fit)
    assert audit["factorial_cells"]==8
    assert audit["count_per_B_A_return_cell"]==16
    assert audit["count_per_B_return_cell"]==32
    assert len(fit)==128
    for idx in range(0,128,4):
        cases=fit[idx:idx+4]
        assert [(v["sign_a"],v["return_action"]) for v in cases]==[
            (-1,0),(-1,1),(1,0),(1,1)]
        for v in cases:
            assert v["effect"].shape==(10,16)
            assert v["action"].shape==(10,16)
            assert set(x.factual(v))=={"action","effect"}
            assert bool(torch.isfinite(v["effect"]).all())
    for part,expected in (("dev",24),("score",32)):
        pairs=x.cohort(part)
        assert len(pairs)==expected
        for i in range(0,len(pairs),2):
            a,b=pairs[i:i+2]
            assert a["sign_a"]==-1 and b["sign_a"]==1
            assert a["sign_b"]==b["sign_b"]
            assert a["return_state"]==b["return_state"]==0.
            assert torch.equal(a["action"],b["action"])
            assert torch.equal(a["effect"][1:],b["effect"][1:])
            assert not torch.equal(a["effect"][0],b["effect"][0])
            assert a["effect"].shape==(9,16)
    for part,grp,sign,ret in (
        ("score",16,-1,None),("fit",True,-1,0),
        ("fit",0,0,0),("fit",0,-1,None),("dev",0,-1,0)):
        with pytest.raises(ValueError,match="F8R19_UNREGISTERED"):
            x.episode(part,grp,sign,ret)


def test_observed_FIT_only_learned_memory_solves_unfamiliar_pairs():
    fit=x.cohort("fit")
    score=x.cohort("score")
    learned=x.learn_factual_memory(fit)
    assert learned["action0_FIT_examples"]==64
    assert learned["action1_FIT_examples"]==64
    assert learned["uses_hidden_regime_labels"] is False
    assert learned["uses_counterfactual_effects"] is False
    result=x.score_factual_memory(learned,score)
    assert result["mean_regret"]<.001
    assert result["A_sign_action_contrast_count"]==32
    assert result["model_selected_synthetic_actions"]==64
    fixed=x.r18._fixed_controls(score)
    assert fixed["means"]["fixed1"]==pytest.approx(.015,abs=1e-12)
    assert fixed["analytic_A_memory_oracle_mean_regret"]==0.


def test_native_and_conventional_early_A_erasure_observability():
    pairs=x.cohort("score")[:2]
    for arm in x.ARMS:
        model=x._model(arm,985201).eval()
        clean=x.r18._state_predictions(x.UNDERLYING[arm],model,pairs)
        erased=x.r18._state_predictions(x.UNDERLYING[arm],model,pairs,
                                         erase_first_A=True)
        assert clean.shape==erased.shape==(2,2)
        assert torch.allclose(
            torch.tensor(erased[0]),torch.tensor(erased[1]),
            atol=1e-6,rtol=0)
        assert all(math.isfinite(float(v)) for v in clean.reshape(-1))


def test_actual_15_model_2400_update_source_neutral_falsifier():
    result=x.run()
    assert result["model_count"]==15
    assert result["total_optimizer_updates"]==2400
    assert result["source_neutral_factual_FIT_effects"]==1280
    assert result["source_neutral_SCORE_factual_effects"]==288
    assert result["score_model_selected_toy_decisions"]==960
    assert result["external_source_effects"]==0
    assert not result["F7_original_replay"]
    assert not result["controls_independently_convergence_qualified"]
    assert not result["scientific_pass"]
    assert result["native_focus_dev_gate_pass"] in (True,False)
    assert len(result["native_focus_candidate_decisions"])==3
    assert result["learned_memory_score"]["mean_regret"]<.001
    for arm in x.ARMS:
        assert sum(v["arm"]==arm for v in result["models"])==3
    for row in result["models"]:
        assert row["updates"]==160 and row["parameters"]>0
        assert row["initial_hash"]!=row["trained_hash"]
        assert row["cold_restore_action_parity"]
        assert math.isfinite(row["first16_loss"])
        assert math.isfinite(row["last16_loss"])
        assert row["dev_regret_at_update80"]>=0
        for split,n in (("dev",12),("test",16)):
            score=row[split]
            assert len(score["per_group_regret"])==n
            assert 0<=score["A_sign_action_contrast_count"]<=2*n
            assert math.isfinite(score["mean_regret"])
            assert math.isfinite(score["erased_initial_A_mean_regret"])
    x.save_ci_receipt(result)


def test_no_original_environment_or_owner_execution():
    code=inspect.getsource(x)
    for pattern in ("import causal_gym","from causal_gym","env.reset(",
                    "env.do(","ctf_do(","workflow_dispatch(",
                    "subprocess.run(","requests.get(","urlopen("):
        assert pattern not in code
