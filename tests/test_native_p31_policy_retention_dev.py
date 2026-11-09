"""P31 original reward source retention negative and exact-context tests."""
import copy
import math
import numpy as np
import pytest
import torch
from scripts import native_p20_reward_only_ppo_dev as p20
from scripts import native_p30_reward_motion_ppo_dev as p30
from scripts import native_p31_policy_retention_dev as p31

class PhysicalOriginalFake:
    def __init__(self):
        self.unwrapped=self
        self.length=.5
        self.polemass_length=.05
        self.masspole=.1
        self.gravity=9.8
        self.i=0
        self.x=self.xdot=self.theta=self.omega=0.
    def reset(self,seed=None):
        self.i=0
        self.x=.0001*(seed%113)
        self.xdot=.001
        self.theta=.015
        self.omega=.004
        return self.observe(),{}
    def observe(self):
        return np.array((self.x,self.xdot,self.theta,self.omega),
                        dtype=np.float32)
    def step(self,action):
        assert action in (0,1)
        impulse=2*action-1
        self.xdot+=.002*impulse
        self.omega+=.007*impulse/self.length
        self.x+=.02*self.xdot
        self.theta+=.02*self.omega
        self.i+=1
        return self.observe(),1.,self.i>=32,False,{}
    def close(self): return None

class FakeGym:
    def make(self,env_name,render_mode=None):
        assert env_name=="CartPole-v1" and render_mode is None
        return PhysicalOriginalFake()

def history():
    return [
      p20.factual_row((.01+.0001*i,100.+i,.02+.0003*i*i,-100.-i),
                      0 if i==0 else (1 if i%2 else -1),
                      0. if i==0 else 1.)
      for i in range(6)]

def test_frozen_original_negative_and_no_sealed_first_use(monkeypatch):
    frozen=p31.frozen()
    assert frozen["original"]["parent_exact_SHA"]=="8fe7cb4b9e1e4e9b9188065671eed005911819f1"
    assert frozen["phase"]=="PROSPECTIVE_FROZEN_BEFORE_CODE_OR_SOURCE_USE"
    assert frozen["source"]["protected_science_first_use"] is False
    assert frozen["training"]["model_initialization_seeds"]==list(p31.SEEDS)
    assert frozen["training"]["initial_policy_KL_beta"]==1.
    assert frozen["training"]["new_public_original_source_reset_base"]==134000
    assert list(p31.EVAL_SEEDS)==frozen["evaluation"]["new_public_original_reset_seeds"]
    assert set(p31.EVAL_SEEDS).isdisjoint(set(p30.EVAL_SEEDS))
    assert set(p31.PROBE_SEEDS).isdisjoint(set(p31.EVAL_SEEDS))
    assert frozen["frozen_metrics"]["do_not_tune_after_results"]
    monkeypatch.setenv("NATIVE_P31_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="PROTECTED_FIRST_USE_DENIED"):
        p31.require_dev()

def test_original_p30_source_seed_fence_restores_on_success_and_exception():
    initial=p30.SEEDS
    with p31.scoped_p30_seeds():
        assert p30.SEEDS==p31.SEEDS
        for seed in p31.SEEDS:
            model=p31.make_initial(seed)
            assert model(p31.motion(history(),seed))[0].shape==(1,2)
        h=history()
        out=p31.motion(h,p31.SEEDS[1])
        assert out.shape==(1,8,10)
        assert torch.count_nonzero(out[:,:,[1,3]])==0
        assert torch.count_nonzero(out[:,:,6:])>0
    assert p30.SEEDS is initial
    with pytest.raises(RuntimeError,match="P31_RESTORE_TEST"):
        with p31.scoped_p30_seeds():
            raise RuntimeError("P31_RESTORE_TEST")
    assert p30.SEEDS is initial

def test_identical_real_source_initialization_and_frozen_KL_gradient():
    torch.set_num_threads(1)
    with p31.scoped_p30_seeds():
        m=p31.make_initial(p31.SEEDS[0])
        frozen=copy.deepcopy(m)
        w0=p20.model_sha(m)
        w1=p20.model_sha(p31.make_initial(p31.SEEDS[0]))
        assert w0==w1
        x=p31.motion(history(),p31.SEEDS[0])
        logits,_=m(x)
        old,_=frozen(x)
        assert torch.equal(logits,old)
        a=torch.distributions.Categorical(logits=logits)
        loss=-a.log_prob(torch.tensor([0])).mean()+p31.KL_BETA*torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(logits,dim=1),
            torch.nn.functional.softmax(old.detach(),dim=1),
            reduction="batchmean")
        assert bool(torch.isfinite(loss))
        loss.backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in m.parameters())

def test_all_probe_contexts_come_from_actual_world_actions_not_source_oracle():
    torch.set_num_threads(1)
    probe,receipt=p31.fixed_history_probe(FakeGym())
    assert len(probe)>=500
    assert receipt["original_source_executed_probe_actions"]>500
    assert len(receipt["original_source_factual_probe_sha256"])==64
    assert not any(
        len(r)!=6 or r[4] not in (-1,0,1)
        for rows in probe for r in rows)
    assert receipt["model_was_not_trained_using_probe"]

def test_critic_only_updates_preserve_initial_actor_and_all_training_source_actions(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(p20,"ROLLOUTS",1)
    monkeypatch.setattr(p20,"NUM_ENVS",2)
    monkeypatch.setattr(p20,"STEPS_PER_ROLLOUT",8)
    monkeypatch.setattr(p20,"PPO_EPOCHS",2)
    monkeypatch.setattr(p20,"MINIBATCH",8)
    receipts={}
    models={}
    with p31.scoped_p30_seeds():
        for arm in p31.TRAIN_ARMS:
            model,original,rec=p31.train_arm(FakeGym(),p31.SEEDS[0],arm)
            assert rec["real_original_train_source_actions"]==16
            assert rec["original_ppo_minibatch_updates"]==4
            assert rec["learning_executed_real_actions_without_expert"]
            assert rec["cold_exact_logits_and_values"]
            assert rec["original_source_reset_stream_sha256"]
            assert rec["initial_source_actor_weights_sha256"]==p20.model_sha(original)
            models[arm]=(model,original)
            receipts[arm]=rec
        assert len({x["first_512_real_source_actions_trace_sha256"]
                    for x in receipts.values()})==1
        probe=[history(),history()[:-1]]
        critic,initial=models["critic_only_frozen_policy"]
        drift=p31.policy_drift(initial,critic,probe,p31.SEEDS[0])
        assert drift["bit_exact_original_logits"]
        assert drift["initial_to_trained_actor_policy_KL"]==0.
        assert drift["greedy_action_agreement_with_initial"]==1.
        assert receipts["critic_only_frozen_policy"]["trained_parameters_effective"] < (
            receipts["standard_motion_ppo"]["trained_parameters_effective"])
        assert receipts["initial_KL_anchored_motion_ppo"]["KL_beta_to_original_actor"]==1.
        assert receipts["initial_KL_anchored_motion_ppo"]["actual_minibatch_initial_policy_KL_mean"]>=0

def test_original_real_source_eval_no_teacher_and_exact_actor_identity(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(p20,"EVAL_HORIZON",25)
    with p31.scoped_p30_seeds():
        m=p31.make_initial(p31.SEEDS[0])
        baseline=p31.evaluate_episode(FakeGym(),m,p31.EVAL_SEEDS[0],
                                      "original",p31.SEEDS[0],"initial_no_update")
        same=p31.evaluate_episode(FakeGym(),m,p31.EVAL_SEEDS[0],
                                  "original",p31.SEEDS[0],"critic_only_frozen_policy")
        rule=p31.evaluate_episode(FakeGym(),None,p31.EVAL_SEEDS[0],
                                  "original",p31.SEEDS[0],
                                  "finite_difference_public_two_frame")
        assert baseline["source_real_action_consequence_sha256"]==same[
            "source_real_action_consequence_sha256"]
        assert baseline["source_original_initial_observation"]==rule[
            "source_original_initial_observation"]
        assert baseline["actual_source_reward_return"]>0
        assert not baseline["true_velocity_seen_by_model"]
        assert baseline["source_eval_model_optimizer_updates"]==0
        assert len(rule["source_executed_action_choices"])<=25
        with pytest.raises(ValueError,match="UNREGISTERED_EVALUATION_WORLD"):
            p31.evaluate_episode(FakeGym(),m,-999,"original",
                                 p31.SEEDS[0],"initial_no_update")
