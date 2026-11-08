"""P11 focused chronological causal-credit and trained-control regression."""
import json
import math
from pathlib import Path
import subprocess
import sys
import pytest
import torch

from scripts.native_p11_temporal_credit_dev import (
    LearnedTemporalCredit,ConventionalCausalGRU,INIT_SEEDS,ARMS,
    DEV_WORLD_SEEDS,WARMUP,DECISIONS,World,plan,data,
    train_arm,evaluate_episode,history_tensor,require_dev
)


def test_temporal_slot_and_gru_use_only_complete_factual_history():
    torch.manual_seed(353)
    hist=torch.zeros(2,8,2)
    action=torch.ones(2,1)
    a=LearnedTemporalCredit()
    b=ConventionalCausalGRU()
    assert a(hist,action).shape==b(hist,action).shape==(2,1)
    assert sum(x.numel() for x in a.parameters())==859
    assert sum(x.numel() for x in b.parameters())==833
    with pytest.raises(ValueError,match="P11_CONTEXT_ACTION_SHAPE"):
        a(torch.zeros(2,7,2),action)
    with pytest.raises(ValueError,match="P11_COMPLETED_OBSERVATIONS_REQUIRED"):
        history_tensor([(1,.2)])


def test_random_lag_training_source_is_factual_and_disjoint():
    x,a,y=data(54131,2)
    v,va,vy=data(54179,2)
    assert x.shape==(80,8,2) and a.shape==(80,1)
    assert y.shape==(80,1)
    assert v.shape==x.shape
    assert not torch.equal(x,v)
    with pytest.raises(ValueError,match="P11_UNREGISTERED_DATA_SOURCE"):
        data(7207,2)


def test_true_action_delay_and_confound_from_unchanged_P10_world():
    t=12
    s=DEV_WORLD_SEEDS[0]
    a=World("lagged",s)
    b=World("lagged",s)
    assert a.realize(t,1,-1)==b.realize(t,-1,-1)
    c=World("instant",s)
    d=World("instant",s)
    assert abs(c.realize(t,1,-1)-d.realize(t,-1,-1))>.7


def test_real_gradients_for_both_registered_models_and_equal_sampler_digest():
    torch.set_num_threads(1)
    fit=data(54131,2)
    valid=data(54179,2)
    a,ra=train_arm(LearnedTemporalCredit,INIT_SEEDS[0],fit,valid,updates=4)
    b,rb=train_arm(ConventionalCausalGRU,INIT_SEEDS[0],fit,valid,updates=4)
    assert ra["gradient_changed_weights"] and rb["gradient_changed_weights"]
    assert ra["sample_indices_sha256"]==rb["sample_indices_sha256"]
    assert ra["actual_optimizer_updates"]==rb["actual_optimizer_updates"]==4


def test_model_planning_uses_two_stage_forecast_not_actual_future():
    torch.manual_seed(353)
    model=LearnedTemporalCredit().eval()
    h=[(1 if i%2 else -1,(-.2 if i%2 else .2))
       for i in range(8)]
    a,p=plan(model,h,lookahead=True)
    assert a in (-1,1) and math.isfinite(p)
    assert len(h)==8  # hypothetical future was NOT appended to real trace
    b,q=plan(model,h,lookahead=False)
    assert b in (-1,1) and math.isfinite(q)


def test_one_real_closed_loop_episode_all_controls_equal_budgets():
    torch.manual_seed(353)
    models={"structured":LearnedTemporalCredit().eval(),
            "gru":ConventionalCausalGRU().eval()}
    ep=evaluate_episode("lagged",DEV_WORLD_SEEDS[0],models)
    assert ep["prefix_exactly_equal_for_all_arms"]
    assert ep["all_64_factual_action_budgets_equal"]
    assert ep["all_factual_outcomes_finite"]
    assert set(ep["data"])==set(ARMS)
    for a in ep["data"].values():
        assert a["decisions"]==DECISIONS
        assert len(a["factual_actions"])==DECISIONS
        assert len(a["factual_consequences"])==DECISIONS
        assert len(a["receipt_sha256"])==64


def test_isolated_cli_and_science_first_use_denial(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    r=subprocess.run([sys.executable,"-I","-B",
                      str(root/"scripts/native_p11_temporal_credit_dev.py"),"--help"],
                     cwd=root,capture_output=True,text=True,timeout=20)
    assert r.returncode==0,r.stderr[-800:]
    manifest=json.loads((root/"experiments/native_p11_temporal_credit_frozen.json").read_text())
    assert manifest["schema"]=="ttt-hybrid.native-p11.temporal-credit-frozen.v1"
    assert manifest["source"]["parent_P10_sha"]=="cf79b157582717cba263437c5ccaabd0dda0a413"
    assert manifest["science"]["authorized"] is False
    assert manifest["science"]["original_world_seeds"]=={}
    assert manifest["science"]["first_use_consumed"] is False
    monkeypatch.setenv("NATIVE_P11_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="P11_SCIENTIFIC_SOURCE_FIRST_USE_DENIED"):
        require_dev()
