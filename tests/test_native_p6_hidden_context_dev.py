"""Negative controls of hidden-context inference; no science worlds or seeds."""
import ast
import inspect
import json
from pathlib import Path

import pytest
import torch

from native_continuous_state import NativeContinuousHybrid
from scripts.native_p6_hidden_context_dev import (
    SCHEMA, ARMS, ORDERS, INITIALIZATIONS, BLOCK, EVENTS, WIDTH,
    ContextEpisode, FactualSlopeEvidence, fixture_event, observed_factual,
    run_initialization,run)


def test_fixture_labels_are_not_in_current_observations_or_actions():
    for order in ORDERS:
        sensors=set()
        action_pairs={}
        for t in range(EVENTS):
            label,s,a,dt=fixture_event(t,order)
            sensors.add(float(s.item()))
            action_pairs.setdefault(label,set()).add(float(a.item()))
            assert dt.item() > 0
        assert len(sensors)==1
        assert action_pairs == {"A":{-1.,1.},"B":{-1.,1.}}
    assert fixture_event(0,ORDERS[0])[0] != fixture_event(0,ORDERS[1])[0]


def test_slope_router_updates_after_two_opposite_factual_actions_only():
    r=FactualSlopeEvidence()
    assert r.choice==1 and r.observations==0
    a_neg=torch.tensor([[-1.]])
    a_pos=-a_neg
    r.observe(a_neg,torch.tensor([[-.33]]))
    assert r.choice==1 and r.observations==1
    r.observe(a_pos,torch.tensor([[.57]]))
    assert r.choice==1 and r.observations==2
    r.observe(a_neg,torch.tensor([[.57]])) # switch, one stale positive
    assert r.choice==1
    r.observe(a_pos,torch.tensor([[-.33]]))
    assert r.choice==-1 and r.observations==4
    with pytest.raises(ValueError,match="REALIZED_PAIR_INVALID"):
        r.observe(torch.zeros((1,1)),torch.ones((1,1)))


def test_hidden_context_is_not_an_argument_to_inferred_episode():
    model=NativeContinuousHybrid(1,1,width=WIDTH,history_limit=4)
    e=ContextEpisode(model,oracle=False)
    sensor=torch.tensor([[.07]])
    action=torch.tensor([[-1.]])
    dt=torch.tensor([[.2]])
    with pytest.raises(ValueError,match="HIDDEN_LABEL_TO_INFERRED_FORBIDDEN"):
        e.predict_pair(sensor,action,dt,oracle_label="A")
    pred,alternate=e.predict_pair(sensor,action,dt)
    assert pred.shape==alternate.shape==(1,1)
    with pytest.raises(RuntimeError,match="PENDING_OBSERVATION"):
        e.predict_pair(sensor,action,dt)
    with pytest.raises(ValueError,match="COUNTERFACTUAL_FEEDBACK_FORBIDDEN"):
        e.observe(torch.ones((1,1)),factual=False)
    with pytest.raises(ValueError,match="REALIZED_OUTCOME_INVALID"):
        e.observe(torch.tensor([[float("nan")]]),factual=True)
    assert e.evidence.observations==0
    e.observe(torch.tensor([[.4]]),factual=True)
    assert e.evidence.observations==1
    assert e.pending is None
    assert e.state.fast_weights.shape==(1,1,WIDTH)
    assert "index" not in inspect.signature(e.predict_pair).parameters
    assert "context" not in inspect.signature(e.predict_pair).parameters


def test_oracle_label_required_and_only_upper_bound():
    model=NativeContinuousHybrid(1,1,width=WIDTH,history_limit=4)
    e=ContextEpisode(model,oracle=True)
    sensor=torch.tensor([[.07]])
    action=torch.tensor([[1.]])
    dt=torch.tensor([[.2]])
    with pytest.raises(ValueError,match="ORACLE_REQUIRES_EXPLICIT_LABEL"):
        e.predict_pair(sensor,action,dt)
    a,b=e.predict_pair(sensor,action,dt,oracle_label="B")
    assert a.shape==b.shape==(1,1)
    e.observe(torch.zeros((1,1)),factual=True)
    assert e.evidence is None


def test_full_dev_records_preserve_hypothesis_failures_if_any():
    result=run()
    assert result["status"]=="PASS_DEV_INTEGRITY_NOT_SCIENCE"
    assert len(result["records"])==len(INITIALIZATIONS)*len(ORDERS)==16
    assert result["science_first_use"]=="NOT_RUN"
    assert result["world_seeds_consumed"]==[]
    assert result["worlds_constructed"]==0
    assert result["optimizer_updates"]==0
    assert result["memory_scalar_budget_each_arm"]==WIDTH
    for r in result["records"]:
        assert all(r["checks"].values())
        assert all(set(phase)==set(ARMS) for phase in r["phase_mse"].values())
        assert r["inference"]["factual_observations"]==EVENTS
    assert all(0<=n<=16 for n in result["hypothesis_support_count_of_16"].values())


def test_original_zero_science_protocol_and_isolated_modules(monkeypatch):
    p=json.loads((Path(__file__).resolve().parents[1]/
        "experiments/native_p6_hidden_context_dev_frozen.json").read_text())
    assert p["schema"]=="ttt-hybrid.native-p6.hidden-context-factual-evidence.v1"
    assert p["source"]["parent_p5_sha"]=="c7f0253cd536666b6aea2e24e8d0313f68d5f20b"
    assert p["science"]["authorized"] is False
    assert p["science"]["world_seeds"]=={}
    assert p["science"]["consumed"] is False
    monkeypatch.setenv("NATIVE_P6_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="SCIENCE_FIRST_USE_FORBIDDEN"):
        run()
    root=Path(__file__).resolve().parents[1]
    source=(root/"scripts/native_p6_hidden_context_dev.py").read_text()
    mods=[n.module or "" for n in ast.walk(ast.parse(source))
          if isinstance(n,ast.ImportFrom)]
    assert not any(tag in mod for mod in mods for tag in
                   ("g7_s20","g7_s25","g7_s26","g7_s27"))
