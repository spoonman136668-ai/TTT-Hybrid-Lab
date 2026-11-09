"""P27: original-source factual paired interventions and contextual negative tests."""
import json
import math
import numpy as np
import pytest
import torch
from scripts import native_p20_reward_only_ppo_dev as p20
from scripts import native_p27_state_dependent_effect_dev as p27

def test_prospective_source_protocol_and_sealed_science_fence(monkeypatch):
    contract=p27.frozen()
    assert contract["parent"]["sha"]=="c6e26fd43c783403f6037cc1e960b80f66320c42"
    assert contract["source"]["scientific_first_use"] is False
    assert contract["physical_protocol"]["heldout_lengths"]==[.42,.58]
    assert set(p27.TRAIN_SEEDS).isdisjoint(p27.EVAL_SEEDS)
    assert set(p27.VAL_SEEDS).isdisjoint(p27.TRAIN_SEEDS)
    assert set(p27.MODEL_SEEDS).isdisjoint((3671,3691,3701))
    monkeypatch.setenv("NATIVE_P27_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="PROTECTED_SCIENCE_FIRST_USE_DENIED"):
        p27.require_dev()

class RealActionFakeEnv:
    def __init__(self):
        self.unwrapped=self
        self.gravity=9.8
        self.length=.5
        self.masspole=.1
        self.polemass_length=.05
        self.x=self.xdot=self.theta=self.theta_dot=0.
        self.n=0
        self.real_actions=[]
    def reset(self,seed=None):
        self.n=0
        self.x=(seed%17)*.0001
        self.theta=.01
        self.xdot=.0
        self.theta_dot=.0
        return np.array([self.x,self.xdot,self.theta,self.theta_dot],
                        dtype=np.float32),{}
    def step(self,a):
        assert a in (0,1)
        self.real_actions.append(a)
        self.n+=1
        delta=2*a-1
        self.xdot+=.006*delta
        self.theta_dot+=.02*delta/self.length
        self.x+=.02*self.xdot
        self.theta+=.02*self.theta_dot
        return np.array([self.x,self.xdot,self.theta,self.theta_dot],
                        dtype=np.float32),1.,False,False,{}
    def close(self): pass

class FakeGym:
    def __init__(self): self.envs=[]
    def make(self,name,render_mode=None):
        assert name=="CartPole-v1" and render_mode is None
        e=RealActionFakeEnv()
        self.envs.append(e)
        return e

def test_source_world_rejects_unauthorized_length_and_drift():
    g=FakeGym()
    e=g.make("CartPole-v1")
    with pytest.raises(RuntimeError,match="UNREGISTERED"):
        p27.configure_physics(e,.8)
    e.masspole=.33
    with pytest.raises(RuntimeError,match="ORIGINAL_UPSTREAM_PHYSICS_DRIFT"):
        p27.configure_physics(e,.42)
    e.masspole=.1
    p27.configure_physics(e,.42)
    assert e.length==.42
    assert math.isclose(e.polemass_length,.042,abs_tol=1e-9)

def test_real_source_sister_action_branches_observed_both_and_no_hidden_length(monkeypatch):
    monkeypatch.setattr(p27,"TRAIN_SEEDS",range(120000,120003))
    monkeypatch.setattr(p27,"TRAIN_LENGTHS",(.35,.5,.65))
    monkeypatch.setattr(p27,"PREFIXES",(4,8))
    g=FakeGym()
    cases,receipt=p27.source_pairs(g,p27.TRAIN_SEEDS,p27.TRAIN_LENGTHS,"train")
    assert len(cases)==18
    assert receipt["valid_pairs_total"]==18
    assert receipt["unexecuted_counterfactual_targets"]==0
    assert len(receipt["all_real_sister_world_events_sha256"])==64
    assert receipt["original_real_environment_actions"]==2*(6+10)*9
    assert all(p[0].shape==(8,6) for p in cases)
    assert all(torch.count_nonzero(p[0][:,1])==0
               and torch.count_nonzero(p[0][:,3])==0 for p in cases)
    assert all(not torch.equal(p[1],p[2]) for p in cases)
    assert all(event["valid"] and event["first_interventions"]==[0,1]
               for event in receipt["raw_factual_real_sister_world_events"])
    assert len(g.envs)==36
    assert json.loads(json.dumps(receipt))["valid_pairs_total"]==18
    with pytest.raises(RuntimeError,match="UNREGISTERED"):
        p27.source_pairs(FakeGym(),range(999,1000),p27.TRAIN_LENGTHS,"train")

def test_hidden_context_has_nonconstant_physical_action_effect(monkeypatch):
    monkeypatch.setattr(p27,"TRAIN_SEEDS",range(120000,120003))
    monkeypatch.setattr(p27,"PREFIXES",(4,8))
    cases,_=p27.source_pairs(FakeGym(),p27.TRAIN_SEEDS,p27.TRAIN_LENGTHS,"train")
    effects={}
    for L in p27.TRAIN_LENGTHS:
        rows=p27.group_by_length(cases,L)
        mean=torch.stack([r[2]-r[1] for r in rows]).mean(dim=0)
        effects[L]=mean
    assert not torch.allclose(effects[.35],effects[.65],atol=1e-4)

def test_action_head_interaction_can_be_history_conditioned_unlike_additive():
    torch.manual_seed(3851)
    model=p27.ContextGRU()
    x=torch.randn((8,p27.HISTORY,6))
    a1=torch.ones((8,1))
    a0=-a1
    effects=model(x,a1)-model(x,a0)
    additive=model(x,a1,"additive")-model(x,a0,"additive")
    assert float(torch.var(effects[:,0]))>0
    assert torch.allclose(additive,additive[0].expand_as(additive),atol=1e-7)
    with pytest.raises(ValueError,match="INTERACTION"):
        model(x,a1,"oracle_context")

def test_source_fitted_strong_ridge_and_no_effect_oracle(monkeypatch):
    monkeypatch.setattr(p27,"TRAIN_SEEDS",range(120000,120003))
    monkeypatch.setattr(p27,"PREFIXES",(4,8))
    cases,_=p27.source_pairs(FakeGym(),p27.TRAIN_SEEDS,p27.TRAIN_LENGTHS,"train")
    effects=torch.stack([r[2]-r[1] for r in cases])
    avg=effects.mean(dim=0)
    ridge=p27.ridge_fit(cases)
    a=p27.effect_metrics(ridge,cases,"trained_interaction_ridge",avg)
    b=p27.effect_metrics(None,cases,"train_fitted_constant_effect",avg)
    assert a["action_effect_MSE"]>=0
    assert b["factual_sister_pairs"]==len(cases)
    assert a["no_eval_optimizer_steps"] is True
    torch.manual_seed(3851)
    model=p27.ContextGRU()
    z=p27.effect_metrics(model,cases,"additive_constant_effect_GRU",avg)
    assert z["factual_sister_pairs"]==len(cases)

def test_equal_training_update_budget_and_source_scoped_sham(monkeypatch):
    monkeypatch.setattr(p27,"TRAIN_SEEDS",range(120000,120003))
    monkeypatch.setattr(p27,"PREFIXES",(4,8))
    cases,_=p27.source_pairs(FakeGym(),p27.TRAIN_SEEDS,p27.TRAIN_LENGTHS,"train")
    models={}
    for arm in p27.ARMS:
        m,rec=p27.train_arm(cases,cases,p27.MODEL_SEEDS[0],arm)
        assert rec["training_factual_branch_examples"]==2*len(cases)
        assert rec["optimizer_minibatches"]==p27.EPOCHS
        assert rec["learning_changed_weights"]
        models[arm]=rec
    assert models["contextual_factual_GRU"]["initial_model_sha256"]==models[
        "shuffled_history_context_GRU"]["initial_model_sha256"]==models[
        "additive_constant_effect_GRU"]["initial_model_sha256"]
    assert models["shuffled_history_context_GRU"][
        "sham_history_context_per_epoch_permuted"]
    assert models["additive_constant_effect_GRU"][
        "constant_effect_head_cannot_depend_on_history"]
