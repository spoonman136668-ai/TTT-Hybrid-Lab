"""P29 exact-source and feature-access negatives, source-real sister interventions."""
import json
import numpy as np
import pytest
import torch

from scripts import native_p20_reward_only_ppo_dev as p20
from scripts import native_p27_state_dependent_effect_dev as p27
from scripts import native_p28_physical_history_identifiability_dev as p28
from scripts import native_p29_feature_access_dev as p29


def test_freeze_before_source_and_protected_science_denied(monkeypatch):
    p=p29.frozen()
    assert p["parent"]["exact_sha"]=="15ebf21b5e1799428635a0ea7182ccec604084dd"
    assert p["training"]["model_seeds"]==list(p29.SEEDS)
    assert p["source"]["protected_science_first_use_not_authorized"]
    assert set(p29.TRAIN_SEEDS).isdisjoint(p28.TRAIN_SEEDS)
    assert set(p29.TEST_SEEDS).isdisjoint(p28.TEST_SEEDS)
    assert p["predeclared"]["all_three_seeds_required"]
    monkeypatch.setenv("NATIVE_P29_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="SEALED_SCIENCE_FIRST_USE_DENIED"):
        p29.require_dev()


def test_scope_restores_P27_original_even_on_failure():
    fields=("TRAIN_SEEDS","VAL_SEEDS","EVAL_SEEDS","TRAIN_LENGTHS","HELDOUT_LENGTHS")
    original=[getattr(p27,k) for k in fields]
    with p29.scoped_source():
        assert p27.TRAIN_SEEDS==p29.TRAIN_SEEDS
        assert p27.EVAL_SEEDS==p29.TEST_SEEDS
        assert p27.HELDOUT_LENGTHS==p29.HELDOUT_LENGTHS
    assert [getattr(p27,k) for k in fields]==original
    with pytest.raises(RuntimeError,match="ROLLBACK_TEST"):
        with p29.scoped_source():
            raise RuntimeError("ROLLBACK_TEST")
    assert [getattr(p27,k) for k in fields]==original


def test_past_only_feature_derivation_nonleaking_and_unchanged_original():
    x=torch.zeros((2,p20.HISTORY,6))
    x[:,:,0]=torch.arange(p20.HISTORY).view(1,-1)*.001
    x[:,:,2]=torch.arange(p20.HISTORY).view(1,-1)**2*.002
    x[:,:,4]=torch.tensor([1.,-1.,1.,-1.,1.,-1.,1.,-1.])
    before=x.clone()
    raw,derived=p29.two_views(x)
    assert raw.shape==derived.shape==(2,p20.HISTORY,10)
    assert torch.equal(raw[:,:,:6],derived[:,:,:6])
    assert torch.count_nonzero(raw[:,:,6:])==0
    assert torch.count_nonzero(derived[:,0,6:])==0
    assert torch.count_nonzero(derived[:,1,8:])==0
    assert torch.equal(x,before)
    assert torch.count_nonzero(derived[:,:,6:])>0
    altered=x.clone()
    altered[:,-1,0]+=1234
    altered[:,-1,2]-=5678
    _,a=p29.two_views(altered)
    assert torch.equal(derived[:,:-1,6:],a[:,:-1,6:])
    with pytest.raises(ValueError,match="PAST_ONLY"):
        x_bad=x.clone()
        x_bad[:,:,1]=1000
        p29.two_views(x_bad)


def test_train_only_standardization_and_same_shape_same_capacity():
    torch.manual_seed(1)
    h=torch.randn((9,p20.HISTORY,6))
    h[:,:,1]=0.
    h[:,:,3]=0.
    stats=p29.TrainOnlyNormalizer(h)
    raw,der=stats.views(h)
    assert torch.equal(raw[:,:,:6],der[:,:,:6])
    assert torch.count_nonzero(raw[:,:,6:])==0
    assert bool(torch.isfinite(der).all())
    assert len(stats.stats_sha256)==64
    h2=h*2
    r2,d2=stats.views(h2)
    assert torch.equal(r2[:,:,:6],d2[:,:,:6])
    assert stats.stats_sha256==p29.TrainOnlyNormalizer(h).stats_sha256
    mlp=p29.make_model("raw_MLP")
    feat_mlp=p29.make_model("derived_MLP")
    gru=p29.make_model("raw_GRU")
    feat_gru=p29.make_model("derived_GRU")
    assert sum(p.numel() for p in mlp.parameters())==sum(
        p.numel() for p in feat_mlp.parameters())
    assert sum(p.numel() for p in gru.parameters())==sum(
        p.numel() for p in feat_gru.parameters())
    assert mlp(raw).shape==feat_mlp(der).shape==(9,2)
    assert gru(raw).shape==feat_gru(der).shape==(9,2)
    assert p29.make_model("derived_length_MLP")(der).shape==(9,1)


class PublicCartPoleFake:
    def __init__(self):
        self.unwrapped=self
        self.gravity=9.8
        self.length=.5
        self.masspole=.1
        self.polemass_length=.05
        self.cart=self.theta=self.vcart=self.vtheta=0.
        self.actions=[]
    def reset(self,seed=None):
        self.cart=(seed%101)*.00012
        self.theta=.012
        self.vcart=.004
        self.vtheta=.003
        return self.obs(),{}
    def obs(self):
        return np.asarray([self.cart,self.vcart,self.theta,self.vtheta],
                          dtype=np.float32)
    def step(self,a):
        assert a in (0,1)
        self.actions.append(a)
        force=2*a-1
        self.vcart+=.02*.15*force
        self.vtheta+=.02*1.4*force/self.length
        self.cart+=.02*self.vcart
        self.theta+=.02*self.vtheta
        return self.obs(),1.,False,False,{}
    def close(self):
        return None


class FakeSource:
    def make(self,name,render_mode=None):
        assert name=="CartPole-v1" and render_mode is None
        return PublicCartPoleFake()


def make_small_real_factual_source(monkeypatch):
    monkeypatch.setattr(p29,"TRAIN_SEEDS",range(124000,124003))
    monkeypatch.setattr(p29,"VAL_SEEDS",range(124500,124503))
    monkeypatch.setattr(p29,"TEST_SEEDS",range(125000,125003))
    with p29.scoped_source():
        train,c=p27.source_pairs(FakeSource(),p29.TRAIN_SEEDS,
                                 p29.TRAIN_LENGTHS,"train")
        val,_=p27.source_pairs(FakeSource(),p29.VAL_SEEDS,
                               p29.TRAIN_LENGTHS,"validation")
        test,tc=p27.source_pairs(FakeSource(),p29.TEST_SEEDS,
                                 p29.HELDOUT_LENGTHS,"evaluation")
    return train,val,test,c,tc


def test_real_paired_actions_and_past_only_source_custody(monkeypatch):
    train,val,test,c,tc=make_small_real_factual_source(monkeypatch)
    assert len(train)==36
    assert len(val)==36
    assert len(test)==18
    assert c["valid_pairs_total"]==36
    assert tc["unexecuted_counterfactual_targets"]==0
    assert c["source_world_length_given_to_models"] is False
    assert c["true_velocities_given_to_models"] is False
    assert c["original_real_environment_actions"]>len(train)*16
    assert all(row["first_interventions"]==[0,1] and row["valid"]
               for row in c["raw_factual_real_sister_world_events"])
    assert len(c["all_real_sister_world_events_sha256"])==64
    assert json.loads(json.dumps(c))["valid_pairs_total"]==36


def test_equal_training_budget_derived_sham_and_feature_source_contract(monkeypatch):
    torch.set_num_threads(1)
    train,val,_,_,_=make_small_real_factual_source(monkeypatch)
    h,targets,length=p29.raw_and_targets(train)
    normalizer=p29.TrainOnlyNormalizer(h)
    models={}
    for arm in p29.ARMS:
        m,r=p29.train_arm(train,val,normalizer,arm,p29.SEEDS[0])
        assert r["optimizer_minibatches"]==p29.EPOCHS
        assert r["trained_on_factual_public_pairs"]==len(train)
        assert r["weights_changed"]
        assert r["source_length_is_never_an_input"]
        models[arm]=r
    for family in ("MLP","GRU"):
        assert models["raw_"+family]["initial_weight_sha256"]==models[
            "derived_"+family]["initial_weight_sha256"]==models[
            "shuffled_derived_"+family]["initial_weight_sha256"]
        assert models["raw_"+family]["equal_model_parameters"]==models[
            "derived_"+family]["equal_model_parameters"]
        assert models["shuffled_derived_"+family][
            "sham_derivatives_misaligned_each_epoch"]
    assert models["raw_length_MLP"]["initial_weight_sha256"]==models[
        "derived_length_MLP"]["initial_weight_sha256"]


def test_no_eval_updates_and_nonzero_real_effect(monkeypatch):
    torch.set_num_threads(1)
    train,val,heldout,_,_=make_small_real_factual_source(monkeypatch)
    h,y,l=p29.raw_and_targets(train)
    norm=p29.TrainOnlyNormalizer(h)
    constant=y.mean(dim=0,keepdim=True)
    mean_length=l.mean().reshape(1,1)
    m,_=p29.train_arm(train,val,norm,"derived_MLP",p29.SEEDS[0])
    checksum=p20.model_sha(m)
    e=p29.evaluate_effect(m,heldout,norm,"derived_MLP",constant)
    assert e["real_source_sister_pairs"]==len(heldout)
    assert e["no_eval_training"]
    assert e["constant_effect_mse"]>0
    assert p20.model_sha(m)==checksum
    ml,_=p29.train_arm(train,val,norm,"derived_length_MLP",p29.SEEDS[0])
    length_report=p29.evaluate_length(
        ml,heldout,norm,"derived_length_MLP",mean_length)
    assert length_report["source_length_used_only_as_supervision_target"]
