"""P25 public DEV: source-matched factual branch intervention and failure controls."""
import json
from pathlib import Path
import pytest
import torch
from scripts import native_p20_reward_only_ppo_dev as p20
from scripts import native_p25_paired_action_consequence_dev as p25

def test_protocol_frozen_before_script_and_no_official_science(monkeypatch):
    j=p25.frozen()
    assert j["parent"]["exact_sha"]=="45b7f925c28c3c58b54a7c5b61f11398474a388c"
    assert j["source"]["protected_science_first_use"] is False
    assert set(p25.TRAIN_SEEDS).isdisjoint(p25.VAL_SEEDS)
    assert set(p25.VAL_SEEDS).isdisjoint(p25.EVAL_SEEDS)
    assert set(p25.EVAL_SEEDS).isdisjoint(p25.TRAIN_SEEDS)
    assert j["metric_contract"]["action_effect_MSE"].startswith("Mean squared")
    monkeypatch.setenv("NATIVE_P25_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="PROTECTED_SCIENCE_FIRST_USE_DENIED"):
        p25.require_dev()

def test_public_velocity_never_reaches_models_and_history_source_immutable():
    rows=[p20.factual_row([.01,50.,.03,100.],0,0.),
          p20.factual_row([.04,-500.,.06,-100.],1,1.)]
    before=[list(x) for x in rows]
    tensor=p25.visible_history(rows)
    assert tensor.shape==(p20.HISTORY,6)
    assert torch.count_nonzero(tensor[:,1])==0
    assert torch.count_nonzero(tensor[:,3])==0
    assert rows==before

def test_registered_effect_models_and_same_gru_capacity():
    x=torch.zeros((5,p20.HISTORY,6),dtype=torch.float32)
    a=torch.ones((5,1),dtype=torch.float32)
    for m in (p25.ActionGRU(),p25.ActionMLP()):
        assert m(x,a).shape==(5,2)
    with pytest.raises(ValueError,match="SHAPE"):
        p25.ActionGRU()(x,a[:1])
    assert len(p25.MODEL_ARMS)==4
    assert p25.action_input(a,"action_blind_same_capacity_GRU").abs().sum()==0
    assert torch.equal(p25.action_input(a,"sham_independent_action_GRU",
                                        -torch.ones_like(a)),-a)

class FakeEnv:
    def __init__(self):
        self.unwrapped=self
        self.gravity=9.8
        self.x=self.xdot=self.theta=self.theta_dot=0
        self.n=0
    def reset(self,seed=None):
        self.n=0
        self.x=float(seed%13)*0.0001
        self.theta=.01
        self.xdot=.02
        self.theta_dot=.02
        return [self.x,self.xdot,self.theta,self.theta_dot],{}
    def step(self,action):
        self.x+=.02*self.xdot
        self.theta+=.02*self.theta_dot
        acceleration=.2*(2*action-1)
        self.xdot+=.02*acceleration
        self.theta_dot+=.02*(1.5*acceleration)
        self.n+=1
        return [self.x,self.xdot,self.theta,self.theta_dot],1.,False,False,{}
    def close(self): pass

class FakeGym:
    def make(self,name,render_mode=None):
        assert name=="CartPole-v1" and render_mode is None
        return FakeEnv()

def test_factual_sister_interventions_both_really_execute_and_effect_measurable(monkeypatch):
    monkeypatch.setattr(p25,"TRAIN_SEEDS",range(104000,104002))
    monkeypatch.setattr(p25,"PREFIX_LENGTHS",(0,3))
    samples,receipt=p25.source_pairs(FakeGym(),p25.TRAIN_SEEDS,"original","train")
    assert len(samples)==4
    assert receipt["valid_sister_pairs"]==4
    assert receipt["real_original_engine_steps"]==2*(2*2+2*5)
    assert len(receipt["factual_action_consequence_sha256"])==64
    for x,y0,y1,seed,prefix in samples:
        assert x.shape==(8,6)
        assert y0.shape==y1.shape==(2,)
        assert not torch.equal(y0,y1)
    assert all(r["executed_intervention_actions"]==[0,1]
               and r["identical_followup_actions"]==[0,0]
               for r in receipt["all_sister_action_consequence_events"])
    with pytest.raises(RuntimeError,match="UNREGISTERED"):
        p25.source_pairs(FakeGym(),range(999,1000),"original","train")

def test_action_contrast_negative_and_true_sign_with_factual_data(monkeypatch):
    monkeypatch.setattr(p25,"TRAIN_SEEDS",range(104000,104002))
    monkeypatch.setattr(p25,"PREFIX_LENGTHS",(0,3))
    samples,_=p25.source_pairs(FakeGym(),p25.TRAIN_SEEDS,"original","train")
    x,a,y=p25.arrays(samples)
    assert x.shape[0]==a.shape[0]==y.shape[0]==8
    blind=p25.ActionGRU().eval()
    r=p25.effect_metrics(blind,samples,"action_blind_same_capacity_GRU")
    assert r["real_paired_worlds"]==4
    assert abs(r["effect_R2_against_zero"])<=1e-7
    assert r["zero_action_effect_reference_mse"]>0

def test_ridge_uses_factual_observations_and_trained_mlp_receipts(monkeypatch):
    monkeypatch.setattr(p25,"TRAIN_SEEDS",range(104000,104002))
    monkeypatch.setattr(p25,"PREFIX_LENGTHS",(0,3))
    samples,_=p25.source_pairs(FakeGym(),p25.TRAIN_SEEDS,"original","train")
    ridge=p25.ridge_fit(samples)
    r=p25.effect_metrics(ridge,samples,"factual_finite_difference_ridge")
    assert ridge.shape[1]==2
    assert r["action_effect_mse"]>=0
    m,receipt=p25.train_model(samples,samples,p25.MODEL_SEEDS[0],
                               "action_conditioned_factual_GRU")
    assert receipt["no_teacher_action_labels"]
    assert receipt["optimizer_minibatches"]==64
    assert receipt["weights_updated"]
    assert p25.effect_metrics(m,samples,"action_conditioned_factual_GRU")[
               "real_paired_worlds"]==4

def test_reject_unregistered_conditions_and_source_drift():
    with pytest.raises(ValueError,match="UNREGISTERED"):
        p25.condition_env(FakeEnv(),"secret_gravity")
    e=FakeEnv()
    e.gravity=900
    with pytest.raises(RuntimeError,match="GRAVITY_DRIFT"):
        p25.condition_env(e,"original")

def test_native_numpy_float32_observation_serialization_repaired(monkeypatch):
    import numpy as np
    samples=np.asarray([np.float32(.018),np.float32(-.001)],dtype=np.float32)
    assert p25.real_floats(samples)==[float(samples[0]),float(samples[1])]
    import json
    assert isinstance(json.dumps(p25.real_floats(samples)),str)
    monkeypatch.setattr(p25,"TRAIN_SEEDS",range(104000,104002))
    monkeypatch.setattr(p25,"PREFIX_LENGTHS",(0,3))
    class NumPyGym:
        def make(self,name,render_mode=None):
            class NumPyEnv(FakeEnv):
                def reset(self,seed=None):
                    original,meta=super().reset(seed=seed)
                    return np.array(original,dtype=np.float32),meta
                def step(self,action):
                    observation,reward,term,trunc,meta=super().step(action)
                    return np.array(observation,dtype=np.float32),reward,term,trunc,meta
            return NumPyEnv()
    factual,receipt=p25.source_pairs(NumPyGym(),p25.TRAIN_SEEDS,"original","train")
    assert len(factual)==4
    assert receipt["valid_sister_pairs"]==4
    assert len(receipt["factual_action_consequence_sha256"])==64
    assert json.loads(json.dumps(receipt))["valid_sister_pairs"]==4

def test_fresh_R1_dev_seed_family_and_original_p25_no_replay():
    original=p25.frozen()
    assert original["paired_intervention"]["train_seed_range"]==[101000,101255]
    assert p25.TRAIN_SEEDS.start==104000
    assert p25.VAL_SEEDS.start==105000
    assert p25.EVAL_SEEDS.start==106000
    assert set(p25.TRAIN_SEEDS).isdisjoint(range(101000,101256))
    assert set(p25.EVAL_SEEDS).isdisjoint(range(103000,103096))
