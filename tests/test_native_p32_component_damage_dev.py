"""P32 original-source reward-only PPO component-freeze and negative controls."""
import copy
import numpy as np
import pytest
import torch
from scripts import native_p20_reward_only_ppo_dev as p20
from scripts import native_p30_reward_motion_ppo_dev as p30
from scripts import native_p32_component_damage_dev as p32

class FakePhysicalOriginalCartPole:
    def __init__(self):
        self.unwrapped=self
        self.gravity=9.8
        self.length=.5
        self.polemass_length=.05
        self.masspole=.1
        self.n=0
        self.x=self.xdot=self.theta=self.omega=0.
    def reset(self,seed=None):
        self.n=0
        self.x=(seed%131)*.0001
        self.xdot=.002
        self.theta=.014
        self.omega=.003
        return self.observe(),{}
    def observe(self):
        return np.asarray([self.x,self.xdot,self.theta,self.omega],
                          dtype=np.float32)
    def step(self,action):
        assert action in (0,1)
        impulse=2*action-1
        self.xdot+=.003*impulse
        self.omega+=.006*impulse/self.length
        self.x+=.02*self.xdot
        self.theta+=.02*self.omega
        self.n+=1
        return self.observe(),1.,self.n>=24,False,{}
    def close(self):
        return None

class FakeSource:
    def make(self,name,render_mode=None):
        assert name=="CartPole-v1" and render_mode is None
        return FakePhysicalOriginalCartPole()

def factual_history():
    return [p20.factual_row(
        (.015+.0001*i,200.+i,.02+.0001*i*i,-100.-i),
        0 if i==0 else (1 if i%2 else -1),
        0. if i==0 else 1.)
        for i in range(5)]

def test_frozen_admission_custody_and_source_seed_boundaries(monkeypatch):
    protocol=p32.frozen()
    assert protocol["parent"]["sha"]=="af574a87a9c81e5dc1b5e703dacf315606494e0a"
    assert protocol["phase"]=="FROZEN_BEFORE_IMPLEMENTATION_AND_PUBLIC_SOURCE_USE"
    assert protocol["source"]["protected_scientific_first_use"] is False
    assert protocol["training"]["model_seeds"]==[4507,4741,4759]
    assert protocol["training"]["train_original_source_reset_base"]==144000
    assert protocol["training"]["arms"]==list(p32.TRAIN_ARMS)
    assert protocol["evaluation"]["arms"]==list(p32.EVAL_ARMS)
    assert set(p32.EVAL_SEEDS).isdisjoint(set(p30.EVAL_SEEDS))
    assert set(p32.EVAL_SEEDS).isdisjoint({140011,140019,140029,140037,140049,140059,140071,140089})
    assert protocol["frozen_rules"]["no_post_result_threshold_tuning"]
    monkeypatch.setenv("NATIVE_P32_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="PROTECTED_FIRST_USE_DENIED"):
        p32.require_dev()

def test_frozen_real_source_observation_mask_and_no_future_information():
    original=factual_history()
    before=[list(row) for row in original]
    with p32.scoped_p30_seeds():
        x=p32.motion(original,p32.SEEDS[0])
        assert x.shape==(1,8,10)
        assert torch.count_nonzero(x[:,:,[1,3]])==0
        assert torch.count_nonzero(x[:,:,6:])>0
        assert original==before
        copy_future=original+[p20.factual_row((7.,1.,5.,3.),1,1.)]
        x2=p32.motion(copy_future,p32.SEEDS[0])
        assert torch.equal(x[:,1:,6:],x2[:,:-1,6:])

def test_source_seed_scope_restored_even_when_training_throws():
    original=p30.SEEDS
    with p32.scoped_p30_seeds():
        assert p30.SEEDS==p32.SEEDS
        for seed in p32.SEEDS:
            assert p32.make_initial(seed)(p32.motion(factual_history(),seed))[0].shape==(1,2)
    assert p30.SEEDS is original
    with pytest.raises(RuntimeError,match="SOURCE_SCOPE_ROLLBACK"):
        with p32.scoped_p30_seeds():
            raise RuntimeError("SOURCE_SCOPE_ROLLBACK")
    assert p30.SEEDS is original

def test_four_component_variants_equal_initial_weights_and_trainable_state():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    initial=p32.make_initial(p32.SEEDS[0])
    state=initial.state_dict()
    assert any(k.startswith("recurrent.") for k in state)
    assert any(k.startswith("actor.") for k in state)
    assert any(k.startswith("critic.") for k in state)
    hashes=[p20.model_sha(p32.make_initial(seed))
            for seed in p32.SEEDS]
    assert len(set(hashes))==3
    assert p20.model_sha(p32.make_initial(p32.SEEDS[0]))==p20.model_sha(initial)

def test_real_factual_source_actions_match_across_component_training_arms(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(p20,"ROLLOUTS",1)
    monkeypatch.setattr(p20,"NUM_ENVS",2)
    monkeypatch.setattr(p20,"STEPS_PER_ROLLOUT",8)
    monkeypatch.setattr(p20,"PPO_EPOCHS",2)
    monkeypatch.setattr(p20,"MINIBATCH",8)
    receipts={}
    models={}
    with p32.scoped_p30_seeds():
        for arm in p32.TRAIN_ARMS:
            trained,initial,rec=p32.train_arm(
                FakeSource(),p32.SEEDS[0],arm)
            assert rec["real_original_train_source_actions"]==16
            assert rec["original_ppo_minibatch_updates"]==4
            assert rec["cold_exact_logits_and_values"]
            assert rec["learning_executed_real_actions_without_expert"]
            assert len(rec["original_actual_action_consequence_sha256"])==64
            assert rec["initial_source_actor_weights_sha256"]==p20.model_sha(initial)
            assert rec["original_PPO_hyperparameters_and_reward_only_unchanged"]
            models[arm]=(trained,initial)
            receipts[arm]=rec
    assert len({r["first_512_real_source_actions_trace_sha256"]
                for r in receipts.values()})==1
    assert len({r["initial_source_actor_weights_sha256"]
                for r in receipts.values()})==1
    assert receipts["actor_head_only_ppo"][
        "components_bit_exact_unchanged_from_original"]["recurrent"] is True
    assert receipts["actor_head_only_ppo"][
        "components_bit_exact_unchanged_from_original"]["actor"] is False
    assert receipts["recurrent_only_ppo"][
        "components_bit_exact_unchanged_from_original"]["recurrent"] is False
    assert receipts["recurrent_only_ppo"][
        "components_bit_exact_unchanged_from_original"]["actor"] is True
    assert receipts["critic_only_frozen_policy"][
        "components_bit_exact_unchanged_from_original"]["recurrent"] is True
    assert receipts["critic_only_frozen_policy"][
        "components_bit_exact_unchanged_from_original"]["actor"] is True
    assert receipts["full_actor_recurrent_ppo"][
        "components_bit_exact_unchanged_from_original"]["actor"] is False
    assert receipts["full_actor_recurrent_ppo"][
        "components_bit_exact_unchanged_from_original"]["recurrent"] is False
    assert receipts["critic_only_frozen_policy"]["trained_parameters_effective"]<(
        receipts["actor_head_only_ppo"]["trained_parameters_effective"]<
        receipts["recurrent_only_ppo"]["trained_parameters_effective"]<
        receipts["full_actor_recurrent_ppo"]["trained_parameters_effective"])
    with p32.scoped_p30_seeds():
        critic,initial=models["critic_only_frozen_policy"]
        probe=[factual_history(),factual_history()[:-1]]
        drift=p32.policy_drift(initial,critic,probe,p32.SEEDS[0])
        assert drift["bit_exact_original_logits"]
        assert drift["initial_to_trained_actor_policy_KL"]==0.
        assert drift["greedy_action_agreement_with_initial"]==1.

def test_exact_original_world_probe_and_evaluation_no_adversarial_velocity(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(p20,"EVAL_HORIZON",25)
    gym=FakeSource()
    probe,receipt=p32.fixed_history_probe(gym)
    assert receipt["original_source_executed_probe_actions"]>500
    assert len(receipt["original_source_factual_probe_sha256"])==64
    assert len(probe)>=500
    with p32.scoped_p30_seeds():
        seed=p32.SEEDS[0]
        m=p32.make_initial(seed)
        init=p32.evaluate_episode(gym,m,p32.EVAL_SEEDS[0],
                   "original",seed,"initial_no_update")
        critic=p32.evaluate_episode(gym,m,p32.EVAL_SEEDS[0],
                   "original",seed,"critic_only_frozen_policy")
        conventional=p32.evaluate_episode(gym,None,p32.EVAL_SEEDS[0],
                   "original",seed,"finite_difference_public_two_frame")
        assert init["source_real_action_consequence_sha256"]==critic[
            "source_real_action_consequence_sha256"]
        assert init["source_original_initial_observation"]==conventional[
            "source_original_initial_observation"]
        assert not init["true_velocity_seen_by_model"]
        assert not init["future_source_observation_given_to_policy"]
        assert init["source_eval_model_optimizer_updates"]==0
        with pytest.raises(ValueError,match="UNREGISTERED_EVALUATION_WORLD"):
            p32.evaluate_episode(gym,m,123,"original",seed,"initial_no_update")
