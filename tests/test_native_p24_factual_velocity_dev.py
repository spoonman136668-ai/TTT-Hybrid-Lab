"""P24 hard negatives: finite-difference past-only, no oracle and frozen PPO."""
import pytest
import torch
from scripts import native_p20_reward_only_ppo_dev as p20
from scripts import native_p24_factual_velocity_dev as p24

def rows():
    return [
      p20.factual_row([0.01,700.,0.02,900.],0,0.),
      p20.factual_row([0.03,-800.,0.04,-950.],1,1.),
      p20.factual_row([0.07,700.,0.01,800.],-1,1.),
    ]

def test_frozen_separate_before_implementation_and_science_fence(monkeypatch):
    p=p24.frozen()
    assert p["original"]["P22_sha"]=="25213b4f02ac0771667428cb176fcb6f962d7a1d"
    assert p["original"]["isolated_P21_pr"]==94
    assert p["original"]["isolated_P23_branch"].startswith("engineering/native-p23")
    assert p["source"]["protected_science_first_use"] is False
    assert set(p24.SEEDS).isdisjoint(p20.SEEDS)
    assert set(p24.EVAL_SEEDS).isdisjoint(p20.EVAL_SEEDS)
    monkeypatch.setenv("NATIVE_P24_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="OFFICIAL_FIRST_USE_DENIED"):
        p24.require_dev()

def test_exact_past_only_velocity_reconstruction_and_source_unchanged():
    before=rows()
    original=[x[:] for x in before]
    reconstructed=p24.history_to_tensor(before,"reconstructed")
    inverted=p24.history_to_tensor(before,"inverted")
    masked=p24.history_to_tensor(before,"masked_raw")
    unmasked=p24.history_to_tensor(before,"unmasked")
    assert (reconstructed.shape==masked.shape==unmasked.shape==inverted.shape
            ==(1,p20.HISTORY,6))
    # P19's frozen SCALE: cart vel 0.5, pole vel 5.0.
    assert torch.isclose(reconstructed[0,-2,1],torch.tensor(0.5))
    assert torch.isclose(reconstructed[0,-2,3],torch.tensor(5.0))
    assert torch.isclose(reconstructed[0,-1,1],torch.tensor(1.0))
    assert torch.isclose(reconstructed[0,-1,3],torch.tensor(-7.5))
    assert torch.equal(inverted[:,:,1],-reconstructed[:,:,1])
    assert torch.equal(inverted[:,:,3],-reconstructed[:,:,3])
    assert torch.count_nonzero(masked[:,:,1])==0
    assert torch.count_nonzero(masked[:,:,3])==0
    assert not torch.equal(unmasked[:,: ,1],reconstructed[:,:,1])
    assert before==original

def test_never_use_future_or_new_episode_velocity():
    first=rows()[0]
    v=p24.history_to_tensor([first],"reconstructed")
    assert v[0,-1,1].item()==0
    assert v[0,-1,3].item()==0
    # Changes to historical raw velocities cannot change reconstructed view
    ordinary=rows()
    corrupted=[x[:] for x in ordinary]
    for x in corrupted:
        x[1]=-1e9
        x[3]=1e9
    assert torch.equal(
        p24.history_to_tensor(ordinary,"reconstructed"),
        p24.history_to_tensor(corrupted,"reconstructed"))

def test_invalid_input_rejected():
    with pytest.raises(ValueError,match="UNREGISTERED"):
        p24.history_to_tensor(rows(),"oracle")
    with pytest.raises(ValueError,match="MALFORMED"):
        p24.history_to_tensor([],"reconstructed")
    with pytest.raises(ValueError,match="MALFORMED"):
        p24.history_to_tensor([[0,1,2]],"masked_raw")

def test_original_ppo_source_restored_after_scope():
    old_tensor=p20.actor_tensor
    old_seed=p20.SEEDS
    old_reset=p20.TRAIN_SEED_BASE
    with p24.scoped_ppo("inverted"):
        assert p20.SEEDS==p24.SEEDS
        assert p20.TRAIN_SEED_BASE==86000
        assert torch.equal(p20.actor_tensor(rows()),
                           p24.history_to_tensor(rows(),"inverted"))
    assert p20.actor_tensor is old_tensor
    assert p20.SEEDS==old_seed
    assert p20.TRAIN_SEED_BASE==old_reset

def test_model_budget_and_negative_controls_paired():
    assert len(p24.VIEWS)==5
    assert p24.VIEWS["reconstructed_wrong_reward_credit"]==(
        "wrong_credit_gru","reconstructed")
    assert 3*5*p20.NUM_ENVS*p20.STEPS_PER_ROLLOUT*p20.ROLLOUTS==737280
    assert 3*5*p20.PPO_EPOCHS*p20.ROLLOUTS*(
        p20.NUM_ENVS*p20.STEPS_PER_ROLLOUT//p20.MINIBATCH)==8640
    assert set(p24.EVAL_SEEDS).isdisjoint(p20.EVAL_SEEDS)

class FakeEnv:
    def __init__(self):
        self.unwrapped=self
        self.length=.5
        self.masspole=.1
        self.polemass_length=.05
        self.n=0
    def reset(self,seed=None):
        self.n=0
        return [0.,.1,.02,.05],{}
    def step(self,action):
        self.n+=1
        return [0.,.11,.02+.001*action,.05],1.,self.n==4,False,{}
    def close(self): pass

class FakeGym:
    def make(self,name,render_mode=None):
        assert name=="CartPole-v1" and render_mode is None
        return FakeEnv()

def test_real_action_receipts_and_no_eval_training():
    gym=FakeGym()
    for name in p24.CONTROLS:
        r=p24.factual_episode(gym,None,"masked_raw",name,
              p24.EVAL_SEEDS[0],"original",p24.SEEDS[0])
        assert r["real_return"]==4.
        assert r["real_actions_executed"]==4
        assert r["no_eval_weight_updates"]
        assert not r["model_given_future_obs_or_teacher_action"]
        assert len(r["factual_action_consequence_sha256"])==64
