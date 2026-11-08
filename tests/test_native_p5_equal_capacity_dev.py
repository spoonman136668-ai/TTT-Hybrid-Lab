"""Synthetic engineering controls only; no scientific source first-use."""
import ast
import json
from pathlib import Path

import pytest
import torch

from native_continuous_state import NativeContinuousHybrid
from scripts.native_p5_equal_capacity_dev import (
    ARMS, WIDTH, HALF, MODEL_INITIALIZATIONS,
    RoutedEpisode, routed_mask, route_key, run, run_initialization)


def test_mask_uses_current_observed_sensor_and_preserves_total_capacity():
    a=torch.tensor([[.35]])
    b=torch.tensor([[-.35]])
    pos=torch.ones((1,1))
    neg=-pos
    sensor_a=routed_mask(a, pos, route="sensor_sign")
    sensor_b=routed_mask(b, pos, route="sensor_sign")
    sign_a=routed_mask(a, pos, route="action_sign")
    sign_b=routed_mask(a, neg, route="action_sign")
    assert torch.equal(sensor_a,sign_a)
    assert torch.equal(sensor_b,sign_b)
    assert torch.equal(sensor_a+sensor_b, torch.ones((1,WIDTH)))
    assert int(sensor_a.sum()) == int(sensor_b.sum()) == HALF
    assert int((sensor_a*sensor_b).sum()) == 0
    model=NativeContinuousHybrid(1,1,width=WIDTH)
    for s,act in ((a,pos),(a,neg),(b,pos),(b,neg)):
        k=route_key(model,s,act,"sensor_sign")
        assert torch.isclose(torch.linalg.vector_norm(k),torch.tensor(1.),atol=1e-6)
        assert torch.count_nonzero(k*(1-routed_mask(s,act,route="sensor_sign")))==0


def test_invalid_route_and_zero_ambiguous_observation_rejected():
    s=torch.tensor([[.35]])
    act=torch.ones((1,1))
    with pytest.raises(ValueError, match="ROUTE_REQUIRED"):
        routed_mask(s,act,route="secret_context")
    with pytest.raises(ValueError,match="AMBIGUOUS_ROUTE"):
        routed_mask(torch.zeros_like(s),act,route="sensor_sign")
    with pytest.raises(ValueError,match="INVALID_OBSERVED_INPUT"):
        routed_mask(torch.tensor([[float("nan")]]),act,route="sensor_sign")


def test_routed_factual_only_and_prediction_before_observation():
    torch.manual_seed(361)
    model=NativeContinuousHybrid(1,1,width=WIDTH,history_limit=4)
    episode=RoutedEpisode(model,"sensor_sign")
    sensor=torch.tensor([[.35]])
    act=torch.tensor([[-1.]])
    dt=torch.tensor([[.2]])
    with pytest.raises(RuntimeError,match="NO_PENDING_PREDICTION"):
        episode.observe(torch.ones((1,1)), factual=True)
    pred, alt=episode.predict_pair(sensor,act,dt)
    assert pred.shape==alt.shape==(1,1)
    with pytest.raises(RuntimeError,match="PENDING_FUTURE_OUTCOME"):
        episode.predict_pair(sensor,act,dt)
    with pytest.raises(ValueError,match="COUNTERFACTUAL_FEEDBACK_FORBIDDEN"):
        episode.observe(torch.zeros((1,1)),factual=False)
    with pytest.raises(ValueError,match="INVALID_REALIZED_OUTCOME"):
        episode.observe(torch.tensor([[float("nan")]]),factual=True)
    episode.observe(torch.tensor([[.35]]),factual=True)
    assert episode.pending is None and episode.events==1
    assert episode.state.fast_weights.shape==(1,1,WIDTH)
    assert torch.count_nonzero(episode.state.fast_weights[...,HALF:])==0


def test_eight_dev_initializations_keep_hypotheses_independent_of_integrity():
    report=run()
    assert report["status"]=="PASS_DEV_INTEGRITY_NOT_SCIENCE"
    assert len(report["records"])==8
    assert report["science_first_use"]=="NOT_RUN"
    assert report["science_worlds_constructed"]==0
    assert report["scientific_seeds_consumed"]==[]
    assert report["memory_budget_scalars_all_arms"]==WIDTH
    assert all(r["status"]=="PASS_DEV_INTEGRITY" for r in report["records"])
    assert all(all(r["checks"].values()) for r in report["records"])
    assert all(set(phase)==set(ARMS)
               for r in report["records"] for phase in r["phase_mse"].values())
    assert all(0<=n<=8 for n in report["hypothesis_support_counts_of_eight"].values())


def test_unregistered_seed_and_science_gate(monkeypatch):
    with pytest.raises(ValueError,match="UNREGISTERED_ENGINEERING_INITIALIZATION"):
        run_initialization(7207)
    monkeypatch.setenv("NATIVE_P5_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="P5_SCIENCE_FIRST_USE_FORBIDDEN"):
        run()


def test_protocol_and_independent_source_fence():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/"experiments/native_p5_equal_capacity_dev_frozen.json").read_text())
    assert manifest["schema"]=="ttt-hybrid.native-p5.equal-capacity-dev-frozen.v1"
    assert manifest["source"]["parent_interference_pr56_sha"]=="b84b910833d80db7296fe07dda5607b61c9040f1"
    assert manifest["science"]["authorized"] is False
    assert manifest["science"]["world_seeds"] == {}
    assert manifest["science"]["first_use_consumed"] is False
    assert manifest["arms"]["gated_total_fast_slots"]==WIDTH
    assert manifest["arms"]["per_context_fast_slots"]==HALF
    script=(root/"scripts/native_p5_equal_capacity_dev.py").read_text()
    mods=[n.module or "" for n in ast.walk(ast.parse(script))
          if isinstance(n,ast.ImportFrom)]
    assert not any(m in mod for mod in mods for m in
                   ("g7_s20","g7_s25","g7_s26","g7_s27"))
