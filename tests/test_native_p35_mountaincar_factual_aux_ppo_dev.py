"""P35 independent MountainCar actual-action auxiliary outcome / false-credit tests."""
import math
import types
import numpy as np
import pytest
import torch
from scripts import native_p20_reward_only_ppo_dev as p20
from scripts import native_p35_mountaincar_factual_aux_ppo_dev as p35

class FakeMountainCar:
    def __init__(self):
        self.unwrapped=self
        self.action_space=types.SimpleNamespace(n=3)
        self.force=.001;self.gravity=.0025;self.goal_position=.5
        self.position=-.5;self.velocity=0.;self.steps=0
    def reset(self,seed=None):
        self.steps=0
        self.position=-.5+(seed%27)*.0005
        self.velocity=0.
        return np.asarray([self.position,self.velocity],dtype=np.float32),{}
    def step(self,action):
        assert action in (0,1,2)
        self.velocity+=self.force*(action-1)-self.gravity*math.cos(3*self.position)
        self.velocity=max(-.07,min(.07,self.velocity))
        self.position=max(-1.2,min(.6,self.position+self.velocity))
        if self.position==-1.2 and self.velocity<0:self.velocity=0.
        self.steps+=1
        success=self.position>=self.goal_position
        return np.asarray([self.position,self.velocity],dtype=np.float32),-1.,success,self.steps>=200,{}
    def close(self):pass

class FakeGym:
    def make(self,name,render_mode=None):
        assert name=="MountainCar-v0" and render_mode is None
        return FakeMountainCar()

def test_p35_frozen_original_scientific_source_and_protected_guard(monkeypatch):
    d=p35.frozen()
    assert d["phase"]=="PROSPECTIVE_FROZEN_BEFORE_IMPLEMENTATION_AND_ANY_P35_SOURCE_ACTIONS"
    assert d["parent"]["sha"]=="50a90718362fac77d8f39a4037a81f5abe97aa76"
    assert d["source"]["original_mountaincar_code_blob"]=="5e7e11518715197974e8df46a842f1bc66ccd88c"
    assert d["training"]["new_model_initialization_seeds"]==list(p35.SEEDS)
    assert d["training"]["arms"]==list(p35.TRAIN_ARMS)
    assert d["evaluation"]["eval_arms"]==list(p35.EVAL_ARMS)
    assert d["training"]["auxiliary_loss_coefficient"]==.15
    assert d["frozen_scientific_thresholds"]["no_post_result_adjustment_or_sweeps"]
    assert set(p35.SEEDS).isdisjoint({5101,5113,5137})
    assert set(p35.EVAL_SEEDS).isdisjoint({175011,175019,175029,175037,175049,175059,175071,175089})
    monkeypatch.setenv("NATIVE_P35_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="P35_PROTECTED_SCIENCE_FIRST_USE_DENIED"):
        p35.guard()
    monkeypatch.setenv("NATIVE_P35_DEV_ONLY","1")
    assert p35.guard()["source"]["three_actions"]

def test_identical_original_GRU_weights_and_auxiliary_head_does_not_shift_policy_RNG():
    torch.set_num_threads(1)
    seed=p35.SEEDS[0]
    models={arm:p35.make_model(seed,arm) for arm in p35.TRAIN_ARMS}
    core={p35.core_sha(m) for arm,m in models.items()
          if arm!="conventional_history_mlp_ppo"}
    assert len(core)==1
    assert p20.model_sha(models["factual_action_aux_ppo"])==p20.model_sha(models["shuffled_action_aux_ppo"])
    assert p35.core_sha(models["full_recurrent_ppo"])==p35.core_sha(models["factual_action_aux_ppo"])
    assert sum(p.numel() for p in models["factual_action_aux_ppo"].aux_predictor.parameters())==11
    assert not hasattr(models["full_recurrent_ppo"],"aux_predictor")
    # both exact stochastic source action distribution and torch RNG start must
    # equal full PPO after model initialization; extra head gets forked RNG.
    next_rng=[]
    for arm in ("full_recurrent_ppo","factual_action_aux_ppo","shuffled_action_aux_ppo"):
        m=p35.make_model(seed,arm)
        next_rng.append(torch.rand(4))
        with torch.no_grad():
            logit,_=m(torch.ones(2,8,4))
        assert logit.shape==(2,3)
    assert torch.equal(next_rng[0],next_rng[1])
    assert torch.equal(next_rng[1],next_rng[2])

def test_source_factual_acceleration_only_uses_executed_positions_and_prior_delta():
    h=[p35.row(-.5,1,0.,0.),p35.row(-.498,2,-1,.002)]
    new=p35.row(-.495,2,-1,.003)
    result=p35.factual_acceleration(h,new)
    assert result==pytest.approx((.003-.002)/.0035,abs=1e-5)
    assert p35.factual_acceleration(h,p35.row(-.41,2,-1,.088))==5.
    assert p35.factual_acceleration(h,p35.row(-.6,0,-1,-.1))==-5.
    before=p35.past_tensor(h)
    assert before.shape==(1,8,4)
    assert torch.count_nonzero(before[:,:6,:])==0

def test_shuffled_credit_is_not_invertible_global_action_swap():
    torch.set_num_threads(1)
    seed=p35.SEEDS[0]
    m=p35.make_model(seed,"factual_action_aux_ppo")
    x=torch.randn(256,8,4)
    actions=torch.tensor([i%3 for i in range(256)],dtype=torch.long)
    actual=torch.linspace(-1,1,256)
    generator=torch.Generator().manual_seed(123)
    genuine,count=p35.auxiliary_loss(m,x,actions,actual,
                                  "factual_action_aux_ppo",generator)
    assert count==0 and genuine.ndim==0 and torch.isfinite(genuine)
    wrong,wrong_count=p35.auxiliary_loss(m,x,actions,actual,
                                  "shuffled_action_aux_ppo",generator)
    assert 120<=wrong_count<=256
    assert wrong.ndim==0 and torch.isfinite(wrong)
    assert not torch.equal(genuine,wrong)
    # deterministic sham permutation on a separate, seeded generator
    same,repeat_count=p35.auxiliary_loss(m,x,actions,actual,
                    "shuffled_action_aux_ppo",torch.Generator().manual_seed(123))
    assert torch.equal(wrong,same) and repeat_count==wrong_count
    with pytest.raises(ValueError,match="P35_UNREGISTERED_AUXILIARY_ARM"):
        p35.auxiliary_loss(m,x,actions,actual,"full_recurrent_ppo",generator)

def test_six_actual_source_reward_training_arms_preserve_first_rollout_and_aux_controls(monkeypatch):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    monkeypatch.setattr(p35,"ROLLOUTS",1)
    monkeypatch.setattr(p35,"NUM_ENVS",2)
    monkeypatch.setattr(p35,"STEPS",8)
    monkeypatch.setattr(p35,"EPOCHS",2)
    monkeypatch.setattr(p35,"MINIBATCH",8)
    outcomes={}
    for arm in p35.TRAIN_ARMS:
        model,initial,r=p35.train_arm(FakeGym(),p35.SEEDS[0],arm)
        assert r["original_training_actual_source_actions"]==16
        assert r["actual_source_optimizer_minibatches"]==4
        assert r["no_privileged_velocity_or_expert_actions"]
        assert r["components_bitwise_unchanged"]["critic"] is False
        assert len(r["full_source_train_trace_sha256"])==64
        outcomes[arm]=r
    assert len({r["initial_core_actor_recurrent_critic_sha256"]
                for arm,r in outcomes.items()
                if arm!="conventional_history_mlp_ppo"})==1
    assert len({r["first_512_source_action_consequence_sha256"]
                for arm,r in outcomes.items()
                if arm!="conventional_history_mlp_ppo"})==1
    fact=outcomes["factual_action_aux_ppo"]
    sham=outcomes["shuffled_action_aux_ppo"]
    assert fact["auxiliary_prediction_head_changed"]
    assert sham["auxiliary_prediction_head_changed"]
    assert fact["auxiliary_action_samples"]==32
    assert sham["auxiliary_action_samples"]==32
    assert fact["auxiliary_wrong_action_fraction"]==0.
    assert 0.<=sham["auxiliary_wrong_action_fraction"]<=1.
    assert fact["trained_parameters"]==sham["trained_parameters"]
    assert outcomes["critic_only_actor_frozen"]["components_bitwise_unchanged"]["actor"]
    assert outcomes["actor_head_only_ppo"]["components_bitwise_unchanged"]["recurrent"]

def test_frozen_real_reward_criterion_is_not_accidentally_true_when_all_goals_fail():
    sample={c:{arm:{"mean_true_reward":-200.,
                    "actual_goal_successes_of_8":0}
                for arm in p35.EVAL_ARMS} for c in p35.CONDITIONS}
    gates=p35.frozen_gates(sample)
    assert not all(all(x.values()) for x in gates.values())
    for cond in p35.CONDITIONS:
        sample[cond]["factual_action_aux_ppo"]["mean_true_reward"]=-90
        sample[cond]["factual_action_aux_ppo"]["actual_goal_successes_of_8"]=7
        sample[cond]["full_recurrent_ppo"]["mean_true_reward"]=-120
        sample[cond]["shuffled_action_aux_ppo"]["mean_true_reward"]=-130
        sample[cond]["conventional_history_mlp_ppo"]["mean_true_reward"]=-95
    assert all(all(x.values()) for x in p35.frozen_gates(sample).values())
    sample["original"]["shuffled_action_aux_ppo"]["mean_true_reward"]=-93
    assert not p35.frozen_gates(sample)["original"]["factual_vs_shuffled_source_reward_margin20"]

def test_public_MountainCar_actual_source_obs_action_and_negative_control_same():
    gym=FakeGym()
    seed=p35.SEEDS[0]
    for world in p35.CONDITIONS:
        m=p35.make_model(seed,"full_recurrent_ppo")
        initial=p35.eval_episode(gym,p35.EVAL_SEEDS[0],seed,world,"initial_no_update",m)
        critic=p35.eval_episode(gym,p35.EVAL_SEEDS[0],seed,world,"critic_only_actor_frozen",m)
        conventional=p35.eval_episode(gym,p35.EVAL_SEEDS[0],seed,world,
                             "two_frame_conventional_pendulum_momentum_rule")
        assert initial["source_real_action_consequence_sha256"]==critic[
            "source_real_action_consequence_sha256"]
        assert initial["initial_visible_source_row"]==conventional[
            "initial_visible_source_row"]
        assert initial["source_actual_reward_return"]==-initial["source_actual_actions"]
        assert initial["privileged_source_velocity_seen_by_policy"] is False
        assert initial["evaluation_optimizer_updates"]==0
    with pytest.raises(ValueError,match="P35_UNREGISTERED_ACTUAL_SOURCE_EVAL"):
        p35.eval_episode(gym,-1,seed,"original","seeded_random")

def test_cold_fails_closed_on_missing_weights(tmp_path):
    import json
    result={"schema":p35.SCHEMA,"engineering":"PASS_PUBLIC_ORIGINAL_SOURCE_DEV",
        "real_source_training_actions":3*len(p35.TRAIN_ARMS)*49152,
        "real_source_optimizer_minibatches":3*len(p35.TRAIN_ARMS)*576,
        "records":[{"model_seed":s} for s in p35.SEEDS]}
    file=tmp_path/"receipt.json"
    file.write_text(json.dumps(result),encoding="utf-8")
    root=tmp_path/"models"
    root.mkdir()
    with pytest.raises(RuntimeError,match="P35_COLD_SAVED_MODEL_SET_INCOMPLETE"):
        p35.cold_verify(FakeGym(),file,root)
