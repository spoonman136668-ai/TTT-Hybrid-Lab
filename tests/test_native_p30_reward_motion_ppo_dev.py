"""P30: reward-only owner/P20 adapter source integrity and real-control negatives."""
import math
import numpy as np
import pytest
import torch
from scripts import native_p20_reward_only_ppo_dev as p20
from scripts import native_p30_reward_motion_ppo_dev as p30

def row(cart,cart_velocity,theta,omega,action,reward):
    return [cart,cart_velocity,theta,omega,action,reward]

def fake_history():
    return [row(.01+0.02*i,200.+i,.005+.001*i*i,-100-i,
                0 if i==0 else (1 if i%2 else -1),0 if i==0 else 1)
            for i in range(5)]

def test_frozen_protocol_and_protected_science_denial(monkeypatch):
    j=p30.frozen()
    assert j["status"]=="PRE_IMPLEMENTATION_FROZEN_FIRST_USE_DENIED"
    assert j["parent"]["qualified_sha"]=="f86efac4b9fbfe377f28ae82b8dd93bd98a45e09"
    assert j["source"]["no_protected_science_first_use"]
    assert set(p30.SEEDS).isdisjoint(p20.SEEDS)
    assert set(p30.EVAL_SEEDS).isdisjoint(p20.EVAL_SEEDS)
    assert j["train"]["factual_actions_per_arm_seed"]==49152
    assert j["train"]["update_batches_per_arm_seed"]==576
    monkeypatch.setenv("NATIVE_P30_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="PROTECTED_SCIENTIFIC_FIRST_USE_DENIED"):
        p30.require_dev()

def test_mask_both_true_velocities_and_past_only_differencing():
    history=fake_history()
    original=[list(x) for x in history]
    raw=p30.view_tensor(history,"raw",p30.SEEDS[0])
    derived=p30.view_tensor(history,"derived",p30.SEEDS[0])
    assert raw.shape==derived.shape==(1,p30.H,10)
    assert torch.equal(raw[:,:,:6],derived[:,:,:6])
    assert torch.count_nonzero(raw[:,:,6:])==0
    assert torch.count_nonzero(derived[:,:,6:])>0
    assert torch.count_nonzero(raw[:,:,[1,3]])==0
    assert torch.count_nonzero(derived[:,:,[1,3]])==0
    assert torch.count_nonzero(derived[:,p30.H-len(history),6:])==0
    assert torch.count_nonzero(derived[:,p30.H-len(history)+1,8:])==0
    assert history==original
    future=history+[row(99,100,-777,200,1,1)]
    dnext=p30.view_tensor(future,"derived",p30.SEEDS[0])
    assert torch.equal(derived[:,1:,6:],dnext[:,:-1,6:])
    with pytest.raises(ValueError,match="VIEW"):
        p30.view_tensor(history,"privileged_velocity",p30.SEEDS[0])

def test_sham_hash_is_state_independent_and_stable():
    a=fake_history()
    b=[row(r[0]+1000,r[1]-3000,r[2]-2,r[3]+100,r[4],r[5])
       for r in a]
    sham1=p30.view_tensor(a,"placebo",p30.SEEDS[0])
    sham2=p30.view_tensor(b,"placebo",p30.SEEDS[0])
    sham3=p30.view_tensor(a,"placebo",p30.SEEDS[0])
    assert torch.equal(sham1[:,:,6:],sham2[:,:,6:])
    assert torch.equal(sham1,sham3)
    assert not torch.equal(p30.view_tensor(a,"derived",p30.SEEDS[0])[:,:,6:],
                           p30.view_tensor(b,"derived",p30.SEEDS[0])[:,:,6:])
    assert not torch.equal(sham1[:,:,6:],
                           p30.view_tensor(a,"derived",p30.SEEDS[0])[:,:,6:])

def test_original_P20_adapters_restore_exactly_even_after_exception():
    keys=("actor_tensor","model_for_arm","MODEL_CLASSES","TRAIN_ARMS",
          "SEEDS","TRAIN_SEED_BASE")
    original=[getattr(p20,key) for key in keys]
    init=p30.SEEDS[0]
    with p30.scoped_original_ppo("derived",init):
        assert p20.SEEDS==p30.SEEDS
        assert p20.TRAIN_SEED_BASE==p30.TRAIN_BASE
        assert p20.actor_tensor(fake_history()).shape==(1,p30.H,10)
        a=p20.model_for_arm("reward_gru",init)
        b=p20.model_for_arm("wrong_credit_gru",init)
        assert p20.model_sha(a)==p20.model_sha(b)
        assert isinstance(a,p30.RewardGRU10)
        assert len(a.actor.weight)==2
    assert all(getattr(p20,k) is v if k in ("actor_tensor","model_for_arm",
        "MODEL_CLASSES","TRAIN_ARMS","SEEDS") else getattr(p20,k)==v
        for k,v in zip(keys,original))
    with pytest.raises(RuntimeError,match="TEST_RESTORE"):
        with p30.scoped_original_ppo("raw",init):
            raise RuntimeError("TEST_RESTORE")
    assert all(getattr(p20,k) is v if k in ("actor_tensor","model_for_arm",
        "MODEL_CLASSES","TRAIN_ARMS","SEEDS") else getattr(p20,k)==v
        for k,v in zip(keys,original))

def test_equal_actor_dimensions_and_critic_and_unregistered_model():
    x=torch.zeros(3,p30.H,10)
    torch.manual_seed(500)
    a=p30.RewardGRU10()
    p,v=a(x)
    assert p.shape==(3,2) and v.shape==(3,)
    with pytest.raises(ValueError,match="10_INPUT_SHAPE"):
        a(torch.zeros(3,p30.H,6))
    count=sum(t.numel() for t in a.parameters())
    with p30.scoped_original_ppo("raw",p30.SEEDS[0]):
        b=p20.model_for_arm("reward_gru",p30.SEEDS[0])
        assert count==sum(t.numel() for t in b.parameters())
        with pytest.raises(ValueError,match="UNREGISTERED"):
            p20.model_for_arm("reward_mlp",p30.SEEDS[0])

class FakePublicCartPole:
    def __init__(self):
        self.unwrapped=self
        self.length=.5
        self.masspole=.1
        self.polemass_length=.05
        self.gravity=9.8
        self.n=0
        self.cart=self.vcart=self.theta=self.omega=0.
    def reset(self,seed=None):
        self.n=0
        self.cart=(seed%19)*.0001
        self.vcart=.002
        self.theta=.01
        self.omega=.003
        return self.obs(),{}
    def obs(self):
        return np.array([self.cart,self.vcart,self.theta,self.omega],
                        dtype=np.float32)
    def step(self,action):
        assert action in (0,1)
        sign=2*action-1
        self.vcart+=.02*sign
        self.omega+=.02*sign/self.length
        self.cart+=.02*self.vcart
        self.theta+=.02*self.omega
        self.n+=1
        return self.obs(),1.,self.n>=20,False,{}
    def close(self): pass

class FakeGym:
    def make(self,name,render_mode=None):
        assert name=="CartPole-v1" and render_mode is None
        return FakePublicCartPole()

def test_real_episode_action_reward_source_no_teacher_and_invariant_source():
    gym=FakeGym()
    outcomes=[]
    for arm in ("finite_difference_public_two_frame","seeded_random",
                "untrained_derived_gru"):
        model=p30.RewardGRU10().eval() if arm=="untrained_derived_gru" else None
        result=p30.single_real_episode(gym,model,arm,p30.EVAL_SEEDS[0],
                                       "original",p30.SEEDS[0])
        assert 1<=result["true_executed_source_actions"]<=p20.EVAL_HORIZON
        assert result["actual_source_rewards"]==result["true_executed_source_actions"]
        assert len(result["actual_action_consequence_sha256"])==64
        assert not result["given_true_source_velocity"]
        assert not result["given_future_action_consequence"]
        assert not result["training_oracle_action_labels_provided"]
        assert result["eval_optimizer_updates"]==0
        outcomes.append(result)
    assert all(x["initial_public_six_observation"]==
               outcomes[0]["initial_public_six_observation"] for x in outcomes)
    with pytest.raises(ValueError,match="UNREGISTERED_SEALED_SOURCE_TEST"):
        p30.single_real_episode(gym,None,"seeded_random",123,
                                "original",p30.SEEDS[0])
