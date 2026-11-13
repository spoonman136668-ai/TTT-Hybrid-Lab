"""F8-R26 frozen two-factor B-gradient interference vs matched A protection."""
import inspect
import math

import pytest
import torch

from experiments import p85mr_f8r26_gradient_conflict as x

torch.set_num_threads(1)


def test_frozen_protocol_and_source_firstuse_denial():
    f=x.locked()
    assert f["stage"]=="FROZEN_BEFORE_IMPLEMENTATION_AND_R26_NUMERICAL_RESULTS"
    assert f["training"]["B_updates_total"]==240
    assert f["training"]["pretrained_updates_total"]==1920
    assert f["authority"]["original_F8_external_firstuse"] is False
    assert f["evaluation"]["independent_scientific_pass"] is False
    with pytest.raises(PermissionError,match="FIRSTUSE_NOT_AUTHORIZED"):
        x.refuse_original_external_firstuse()


def test_two_independent_B_hidden_factors_and_matched_A_swaps():
    tr,hold,score=(x.cohort(s) for s in x.SPLITS)
    receipt=x.audit(tr,hold,score)
    assert receipt["B_train_FIT_episodes"]==64
    assert receipt["B_holdout_episodes"]==32
    assert receipt["paired_twins"]==16
    assert {q["b"] for q in tr}=={-1,1}
    assert {q["c"] for q in tr}=={-1,1}
    assert len(score)==32
    for i in range(0,len(score),2):
        first,swapped=score[i:i+2]
        assert torch.equal(first["action"],swapped["action"])
        assert torch.equal(first["effect"][4:],swapped["effect"][4:])
        assert not torch.equal(first["effect"][:4],swapped["effect"][:4])
    assert x.b_after(0.,1,1,1)!=x.b_after(0.,1,1,-1)
    assert x.b_after(0.,1,-1,1)!=x.b_after(0.,1,1,1)
    with pytest.raises(ValueError,match="UNKNOWN_REGISTERED_WORLD"):
        x.episode("A_SCORE",16,1,1,twin=0)
    with pytest.raises(ValueError,match="BAD_INTERACTING"):
        x.b_after(0.,True,1,1)


def test_factual_four_probe_memory_identifies_offgrid_return():
    eps=x.cohort("A_SCORE")
    ctrl=x.controls(eps)
    assert ctrl["best_fixed"]>0
    assert ctrl["observed_four_probe_linear_memory"]<1e-8
    for arm in ("native_focus20","gru_focus20","ssm_focus20"):
        model=x.r22.model_for(arm,985201).eval()
        erased=x._predict_returns(arm,model,eps[:2],erase_probes=True)
        assert erased.shape==(2,2,2)
        assert torch.allclose(torch.tensor(erased[0]),torch.tensor(erased[1]),
                              atol=1e-6,rtol=0)


def test_actual_five_pretrained_models_240_B_gradient_updates_and_restore():
    out=x.run()
    assert out["model_count"]==5
    assert out["actual_A_pretraining_optimizer_updates"]==1920
    assert out["actual_B_adaptation_optimizer_updates"]==240
    assert out["original_external_effects"]==0
    assert not out["F7_original_effect_replay"]
    assert not out["independent_scientific_pass"]
    assert not out["conventional_independently_convergence_qualified"]
    assert out["decision"]["frozen_development_pass"] in (True,False)
    assert len(out["decision"]["native_seed_outcomes"])==3
    assert len(out["models"])==5
    for row in out["models"]:
        assert row["parameters"]>0
        assert row["pretrain_updates"]==384
        assert set(row["conditions"])==set(x.CONDITIONS)
        a=row["conditions"]["no_update"]
        assert a["updates"]==0
        for cond in ("plain_B_update","orthogonal_B_update"):
            e=row["conditions"][cond]
            assert e["updates"]==24
            assert e["pretrained_hash"]==row["pretrained_hash"]
            assert e["adapted_hash"]!=e["pretrained_hash"]
            assert 0<=e["negative_gradient_conflicts"]<=24
            assert 0<=e["projected_steps"]<=24
            assert e["cold_replay_action_parity"]
            assert math.isfinite(e["B_holdout_MSE"])
            assert len(e["A_first_return"]["paired_group_regret"])==16
            assert e["A_first_return"]["action_count"]==128
        assert row["conditions"]["plain_B_update"]["projected_steps"]==0
        assert row["conditions"]["orthogonal_B_update"]["projected_steps"]==(
            row["conditions"]["orthogonal_B_update"]["negative_gradient_conflicts"])
    x.retain_ci_receipt(out)


def test_no_original_scientific_environment_or_new_runner():
    text=inspect.getsource(x)
    for forbidden in ("import causal_gym","from causal_gym","env.reset(",
                      "env.do(","ctf_do(","workflow_dispatch(",
                      "subprocess.run(","requests.get(","urlopen("):
        assert forbidden not in text
