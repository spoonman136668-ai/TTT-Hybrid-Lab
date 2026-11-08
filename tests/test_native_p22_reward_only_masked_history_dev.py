"""Negative and interface controls for P22; no protected science or world first-use."""
from pathlib import Path
import pytest
import torch
from scripts import native_p20_reward_only_ppo_dev as p20
from scripts import native_p22_reward_only_masked_history_dev as p22

def history():
    return [
        p20.factual_row([.01, .12, .02, .19], 0, 0.),
        p20.factual_row([.03, -.15, -.04, -.21], 1, 1.),
        p20.factual_row([.04, .08, .08, .14], -1, 1.),
    ]

def test_protocol_registered_before_code_and_sealed_denied(monkeypatch):
    p=p22.frozen()
    assert p["exact_parent"]["sha"]=="ceb54837a8066176a6b591fb3eb101c463c5d02e"
    assert p["source"]["scientific_first_use"] is False
    assert p["acceptance"]["scientific_pass"] is False
    assert p["study"]["eval_seeds"] == list(p22.EVAL_SEEDS)
    assert set(p22.EVAL_SEEDS).isdisjoint(p20.EVAL_SEEDS)
    assert set(p22.SEEDS).isdisjoint(p20.SEEDS)
    monkeypatch.setenv("NATIVE_P22_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="SEALED_SCIENCE_FIRST_USE_DENIED"):
        p22.require_dev()

def test_real_public_velocity_masking_all_history_and_current_only():
    original=history()
    orig_copy=[x[:] for x in original]
    full=p22.tensor_for_view(original,"full_4d")
    masked=p22.tensor_for_view(original,"masked_full")
    latest=p22.tensor_for_view(original,"masked_current")
    two=p22.tensor_for_view(original,"masked_two")
    assert full.shape==masked.shape==latest.shape==two.shape==(1,p20.HISTORY,6)
    assert torch.any(full[:, -3:, 1] != 0)
    assert torch.any(full[:, -3:, 3] != 0)
    assert torch.all(masked[:,:,1] == 0)
    assert torch.all(masked[:,:,3] == 0)
    assert torch.equal(latest[:,-1,:],masked[:,-1,:])
    assert torch.count_nonzero(latest[:,:-1,:])==0
    assert torch.equal(two[:,-2:,:],masked[:,-2:,:])
    assert torch.count_nonzero(two[:,:-2,:])==0
    assert original == orig_copy

def test_failed_views_and_empty_histories_fail_closed():
    with pytest.raises(ValueError,match="UNREGISTERED"):
        p22.tensor_for_view(history(),"privileged_future")
    with pytest.raises(ValueError,match="MALFORMED"):
        p22.tensor_for_view([],"masked_full")
    with pytest.raises(ValueError,match="MALFORMED"):
        p22.tensor_for_view([[1,2,3]],"masked_full")

def test_ppo_original_code_unmodified_after_scope():
    old_tensor=p20.actor_tensor
    old_seeds=p20.SEEDS
    old_reset=p20.TRAIN_SEED_BASE
    with p22.isolated_ppo_view("masked_two"):
        assert p20.SEEDS == p22.SEEDS
        assert p20.TRAIN_SEED_BASE == 84000
        assert torch.equal(p20.actor_tensor(history()),p22.tensor_for_view(history(),"masked_two"))
    assert p20.actor_tensor is old_tensor
    assert p20.SEEDS == old_seeds
    assert p20.TRAIN_SEED_BASE == old_reset

def test_finite_difference_factual_angles_only():
    a=[p20.factual_row([0,.5,.03,5.],0,0.)]
    assert p22.finite_difference(a)==1
    b=a+[p20.factual_row([0,-9.,-.02,-4.],1,1.)]
    assert p22.finite_difference(b)==0
    changed=[list(z) for z in b]
    for x in changed:
        x[1] = -100.
        x[3] = 99.
    assert p22.finite_difference(changed)==p22.finite_difference(b)

def test_preregistered_all_same_param_and_budgets():
    assert len(p22.VIEWS)==5
    assert set(p22.VIEWS.values()) == {
        ("reward_gru","masked_full"),("reward_gru","masked_current"),
        ("reward_gru","masked_two"),("wrong_credit_gru","masked_full"),
        ("reward_gru","full_4d")}
    assert 5*3*p20.ROLLOUTS*p20.NUM_ENVS*p20.STEPS_PER_ROLLOUT==737280
    assert 5*3*p20.ROLLOUTS*p20.PPO_EPOCHS*(
        p20.NUM_ENVS*p20.STEPS_PER_ROLLOUT//p20.MINIBATCH)==8640

class FakeEnv:
    def __init__(self):
        self.unwrapped=self
        self.length=.5
        self.masspole=.1
        self.polemass_length=.05
        self.count=0
    def reset(self,seed=None):
        self.count=0
        return [0.,0.1,.02,.05],{}
    def step(self,action):
        self.count+=1
        return [0.,0.11,.02+.001*action,.05],1.,self.count==4,False,{}
    def close(self):pass

class FakeGym:
    def make(self,name,render_mode=None):
        assert name=="CartPole-v1" and render_mode is None
        return FakeEnv()

def test_factual_step_receipts_and_no_eval_grad():
    for name in ("masked_finite_difference","seeded_random"):
        out=p22.run_episode(FakeGym(),None,"masked_full",p22.EVAL_SEEDS[0],
                            "original",p22.SEEDS[0],name)
        assert out["actual_factual_steps"]==4
        assert out["mean_reward"]==4.
        assert len(out["factual_step_reward_sha256"])==64
        assert out["online_eval_gradient_steps"]==0
