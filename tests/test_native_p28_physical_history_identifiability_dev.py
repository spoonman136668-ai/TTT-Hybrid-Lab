"""P28: physical context-identifiability proof, negatives, and science fences."""
import json
import numpy as np
import pytest
import torch

from scripts import native_p27_state_dependent_effect_dev as p27
from scripts import native_p28_physical_history_identifiability_dev as p28


def test_protocol_frozen_and_disjoint_external_public_dev_worlds(monkeypatch):
    f=p28.frozen()
    assert f["state"]=="FROZEN_BEFORE_NEW_SOURCE_USE_OR_IMPLEMENTATION"
    assert f["parent"]["sha"]=="5a1eb9569aa7e44278a57d196f6c2a8c2c6f7c0d"
    assert f["source"]["no_protected_scientific_first_use"]
    assert set(p28.TRAIN_SEEDS).isdisjoint(p27.TRAIN_SEEDS)
    assert set(p28.TEST_SEEDS).isdisjoint(p27.EVAL_SEEDS)
    assert p28.TRAIN_LENGTHS==(.30,.42,.55,.70)
    assert p28.HELDOUT_LENGTHS==(.36,.62)
    monkeypatch.setenv("NATIVE_P28_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="PROTECTED_FIRST_USE_DENIED"):
        p28.require_dev()


def test_fresh_source_scope_exactly_restores_independent_P27():
    names=("TRAIN_SEEDS","VAL_SEEDS","EVAL_SEEDS","TRAIN_LENGTHS","HELDOUT_LENGTHS")
    before=[getattr(p27,n) for n in names]
    with p28.scoped_fresh_source():
        assert p27.TRAIN_SEEDS==p28.TRAIN_SEEDS
        assert p27.VAL_SEEDS==p28.VAL_SEEDS
        assert p27.EVAL_SEEDS==p28.TEST_SEEDS
        assert p27.TRAIN_LENGTHS==p28.TRAIN_LENGTHS
        assert p27.HELDOUT_LENGTHS==p28.HELDOUT_LENGTHS
        assert p28.frozen()
    assert [getattr(p27,n) for n in names]==before
    with pytest.raises(RuntimeError,match="deliberate_test_rollback"):
        with p28.scoped_fresh_source():
            raise RuntimeError("deliberate_test_rollback")
    assert [getattr(p27,n) for n in names]==before


class PublicFakePhysicalEngine:
    def __init__(self):
        self.unwrapped=self
        self.gravity=9.8
        self.length=.5
        self.masspole=.1
        self.polemass_length=.05
        self.cart=self.theta=self.xdot=self.omega=0.
        self.steps=0
    def reset(self,seed=None):
        self.steps=0
        self.cart=(seed%19)*.0002
        self.theta=.01
        self.xdot=.025
        self.omega=.012
        return self._obs(),{}
    def _obs(self):
        return np.asarray([self.cart,self.xdot,self.theta,self.omega],dtype=np.float32)
    def step(self,action):
        assert action in (0,1)
        force=2*action-1
        self.xdot+=.02*.25*force
        self.omega+=.02*.9*force/self.length
        self.cart+=.02*self.xdot
        self.theta+=.02*self.omega
        self.steps+=1
        return self._obs(),1.,False,False,{}
    def close(self): pass


class PublicFakeSource:
    def make(self,name,render_mode=None):
        assert name=="CartPole-v1" and render_mode is None
        return PublicFakePhysicalEngine()


def source_fixtures(monkeypatch):
    monkeypatch.setattr(p28,"TRAIN_SEEDS",range(122000,122002))
    monkeypatch.setattr(p28,"VAL_SEEDS",range(122500,122502))
    monkeypatch.setattr(p28,"TEST_SEEDS",range(123000,123002))
    with p28.scoped_fresh_source():
        tr,train_receipt=p27.source_pairs(PublicFakeSource(),p28.TRAIN_SEEDS,
                                         p28.TRAIN_LENGTHS,"train")
        ev,eval_receipt=p27.source_pairs(PublicFakeSource(),p28.TEST_SEEDS,
                                        p28.HELDOUT_LENGTHS,"evaluation")
    return tr,train_receipt,ev,eval_receipt


def test_same_actual_engine_sister_worlds_and_hidden_length_never_input(monkeypatch):
    train,receipt,heldout,evaluation=source_fixtures(monkeypatch)
    assert len(train)==2*4*3
    assert len(heldout)==2*2*3
    assert receipt["unexecuted_counterfactual_targets"]==0
    assert receipt["source_world_length_given_to_models"] is False
    assert receipt["true_velocities_given_to_models"] is False
    assert len(receipt["all_real_sister_world_events_sha256"])==64
    assert json.loads(json.dumps(receipt))["valid_pairs_total"]==24
    h,e,l=p28.true_arrays(train)
    assert h.shape==(24,8,6)
    assert e.shape==(24,2)
    assert torch.count_nonzero(h[:,:,1])==0
    assert torch.count_nonzero(h[:,:,3])==0
    assert len(torch.unique(l))==4
    assert set(torch.unique(l).tolist()) != {0.36,0.62}
    assert all(a["first_interventions"]==[0,1] and a["valid"]
               for a in receipt["raw_factual_real_sister_world_events"])
    assert evaluation["valid_pairs_total"]==12


def test_model_visible_features_cannot_read_hidden_length():
    h=torch.randn((6,8,6))
    h[:,:,1]=0.; h[:,:,3]=0.
    visible=p28.visible_features(h)
    source_only=torch.full((6,1),.30)
    changed_source_length=torch.full((6,1),.70)
    assert visible.shape[0]==6
    assert torch.equal(visible,p28.visible_features(h.clone()))
    assert not torch.equal(p28.oracle_features(source_only),
                           p28.oracle_features(changed_source_length))
    with pytest.raises(ValueError,match="VISIBLE_HISTORY_SHAPE"):
        p28.visible_features(torch.zeros(8,6))


def test_training_and_sham_equal_optimization_budget(monkeypatch):
    training,_,_,_=source_fixtures(monkeypatch)
    fx,fe,fl=p28.true_arrays(training)
    assert fx.shape[0]==24
    oracle=p28.ridge_fit(p28.oracle_features(fl),fe)
    visible=p28.ridge_fit(p28.visible_features(fx),fe)
    assert p28.predict_ridge(p28.oracle_features(fl),oracle).shape==(24,2)
    assert p28.predict_ridge(p28.visible_features(fx),visible).shape==(24,2)
    a,ar=p28.train_model(training,training,p28.SEEDS[0],"history_effect_MLP")
    b,br=p28.train_model(training,training,p28.SEEDS[0],"shuffled_history_effect_MLP")
    assert ar["original_init_model_sha256"]==br["original_init_model_sha256"]
    assert ar["optimizer_updates"]==br["optimizer_updates"]==64
    assert ar["learned_weights_changed"] and br["learned_weights_changed"]
    assert br["sham_history_randomly_misaligned"]
    assert not ar["model_received_source_length_as_INPUT"]
    c,cr=p28.train_model(training,training,p28.SEEDS[0],"history_length_MLP")
    assert cr["source_length_supervision_diagnostic_only"]
    assert c(fx).shape==(24,1)
    assert a(fx).shape==b(fx).shape==(24,2)


def test_factual_mse_and_zero_variance_fail_closed():
    true=torch.tensor([[.1,.4],[.5,.7]],dtype=torch.float32)
    pred=torch.zeros_like(true)
    constant=torch.tensor([[.2,.2]],dtype=torch.float32)
    out=p28.effect_eval(pred,true,constant)
    assert out["mse"]>0
    assert out["source_pair_count"]==2
    with pytest.raises(RuntimeError,match="NO_FACTUAL_PHYSICS_EFFECT_VARIANCE"):
        p28.effect_eval(constant.expand(2,2),constant.expand(2,2),constant)
