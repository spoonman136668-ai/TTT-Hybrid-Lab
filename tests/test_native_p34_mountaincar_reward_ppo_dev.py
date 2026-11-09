"""P34 source-original MountainCar reward-only three-action mechanisms and negative controls."""
import hashlib
import math
import types
import numpy as np
import pytest
import torch
from scripts import native_p34_mountaincar_reward_ppo_dev as p34
from scripts import native_p20_reward_only_ppo_dev as p20

class FakeMountainCar:
    def __init__(self):
        self.unwrapped=self
        self.action_space=types.SimpleNamespace(n=3)
        self.force=.001
        self.gravity=.0025
        self.goal_position=.5
        self.t=0
        self.position=-.5
        self.true_velocity=12345.
    def reset(self,seed=None):
        self.t=0
        self.position=-.5+(seed%27)*.0005
        self.true_velocity=.0
        return np.array([self.position,self.true_velocity],dtype=np.float32),{}
    def step(self,action):
        assert action in (0,1,2)
        self.t+=1
        self.true_velocity+=self.force*(action-1)-self.gravity*math.cos(3*self.position)
        self.true_velocity=min(.07,max(-.07,self.true_velocity))
        self.position=min(.6,max(-1.2,self.position+self.true_velocity))
        if self.position<=-1.2 and self.true_velocity<0:
            self.true_velocity=0.
        terminal=self.position>=self.goal_position
        return np.array([self.position,self.true_velocity],dtype=np.float32),-1.0,terminal,self.t>=200,{}
    def close(self): pass

class FakeGym:
    def make(self,name,render_mode=None):
        assert name=="MountainCar-v0" and render_mode is None
        return FakeMountainCar()

def test_exact_original_external_MIT_source_and_dev_fences(monkeypatch):
    c=p34.frozen()
    assert c["parent"]["sha"]=="448df2a5039427e232439de6136d35d4e2b5d795"
    assert c["phase"]=="PROSPECTIVE_FROZEN_BEFORE_IMPLEMENTATION_AND_ORIGINAL_SOURCE_ACTIONS"
    assert c["source"]["original_MountainCar_blob"]=="5e7e11518715197974e8df46a842f1bc66ccd88c"
    assert c["source"]["protected_scientific_first_use"] is False
    assert c["training"]["model_seeds"]==list(p34.SEEDS)
    assert c["training"]["training_arms"]==list(p34.TRAIN_ARMS)
    assert c["evaluation"]["model_evaluation_seeds"]==list(p34.EVAL_SEEDS)
    assert c["thresholds"]["all_3_new_seeds_required"]
    assert c["thresholds"]["no_post_result_threshold_tuning"]
    monkeypatch.setenv("NATIVE_P34_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="P34_PROTECTED_SCIENCE_FIRST_USE_DENIED"):
        p34.guard()
    monkeypatch.setenv("NATIVE_P34_DEV_ONLY","1")
    assert p34.guard()["source"]["actual_source_env"]=="MountainCar-v0"

def test_native_models_305_GRU_300_MLP_and_equal_actors_across_GRU_arms():
    torch.set_num_threads(1)
    gru=p34.MountainHistoryGRU()
    mlp=p34.MountainHistoryMLP()
    assert sum(p.numel() for p in gru.parameters())==305
    assert sum(p.numel() for p in mlp.parameters())==300
    assert not any(isinstance(x,torch.nn.GRU) for x in mlp.modules())
    assert len(list(p34.model_parts(gru)["recurrent"]))>0
    x=torch.randn(3,8,4)
    for model in (gru,mlp):
        logits,critic=model(x)
        assert logits.shape==(3,3) and critic.shape==(3,)
    all_gru=[p34.make_model(p34.SEEDS[0],arm) for arm in
            p34.TRAIN_ARMS if arm!="conventional_history_mlp_ppo"]
    assert len({p20.model_sha(m) for m in all_gru})==1
    assert p20.model_sha(p34.make_model(p34.SEEDS[0],
                      "conventional_history_mlp_ppo"))!=p20.model_sha(all_gru[0])

def test_factual_position_history_no_true_velocity_no_future_or_force_input():
    before=p34.row(-.53,2,-1,.012)
    assert len(before)==4
    assert before[0]==pytest.approx(-.53/1.2)
    assert before[1]==1 and before[2]==-1.
    assert before[3]==pytest.approx(.012/.07)
    base=p34.past_tensor([p34.row(-.5),p34.row(-.49,2,-1,.01)])
    assert base.shape==(1,8,4)
    assert torch.count_nonzero(base[:,:6,:])==0
    assert torch.count_nonzero(base[:,6:,:])>0
    true_vel_secret=[p34.row(-.5),p34.row(-.49,2,-1,.01)]
    same=p34.past_tensor(true_vel_secret)
    assert torch.equal(base,same)
    with pytest.raises(ValueError,match="BAD_CONTINUOUS_FACTUAL_HISTORY"):
        p34.past_tensor([])

def test_real_original_source_three_actions_and_exact_force_guard():
    gym=FakeGym()
    env=p34.source_env(gym,"low_force_x0.75")
    assert env.unwrapped.force==pytest.approx(.00075)
    obs,_=env.reset(seed=p34.EVAL_SEEDS[0])
    history=[p34.row(obs[0])]
    for a in (0,1,2):
        n,r,done,trunc=p34.actual_step(env,history,a)
        assert len(n)==4 and r==-1. and isinstance(done,bool)
        assert isinstance(trunc,bool)
        history.append(n)
    env.close()
    with pytest.raises(ValueError,match="UNREGISTERED_SOURCE_FORCE"):
        p34.source_env(gym,"unknown-world")
    with pytest.raises(ValueError,match="NOT_REAL_THREE_ACTION"):
        env=p34.source_env(gym)
        try:p34.actual_step(env,[p34.row(-.5)],9)
        finally:env.close()

def test_all_4_real_reward_PPO_trained_component_custody_and_first_rollout(monkeypatch):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    monkeypatch.setattr(p34,"ROLLOUTS",1)
    monkeypatch.setattr(p34,"NUM_ENVS",2)
    monkeypatch.setattr(p34,"STEPS",8)
    monkeypatch.setattr(p34,"EPOCHS",2)
    monkeypatch.setattr(p34,"MINIBATCH",8)
    data={}
    for arm in p34.TRAIN_ARMS:
        model,initial,receipt=p34.train_arm(FakeGym(),p34.SEEDS[0],arm)
        assert receipt["original_training_actual_source_actions"]==16
        assert receipt["actual_source_optimizer_minibatches"]==4
        assert receipt["no_privileged_velocity_or_expert_actions"]
        assert len(receipt["full_source_train_trace_sha256"])==64
        assert receipt["components_bitwise_unchanged"]["critic"] is False
        assert model is not initial
        data[arm]=receipt
    assert len({r["initial_model_sha256"] for arm,r in data.items()
                if arm!="conventional_history_mlp_ppo"})==1
    assert len({r["first_512_source_action_consequence_sha256"]
                for arm,r in data.items()
                if arm!="conventional_history_mlp_ppo"})==1
    assert data["actor_head_only_ppo"]["components_bitwise_unchanged"]["recurrent"]
    assert not data["actor_head_only_ppo"]["components_bitwise_unchanged"]["actor"]
    assert not data["full_recurrent_ppo"]["components_bitwise_unchanged"]["recurrent"]
    assert not data["conventional_history_mlp_ppo"]["components_bitwise_unchanged"]["recurrent"]
    assert data["critic_only_actor_frozen"]["components_bitwise_unchanged"]["actor"]
    assert data["critic_only_actor_frozen"]["components_bitwise_unchanged"]["recurrent"]
    assert data["critic_only_actor_frozen"]["trained_parameters"]<(
        data["actor_head_only_ppo"]["trained_parameters"]<
        data["conventional_history_mlp_ppo"]["trained_parameters"]<
        data["full_recurrent_ppo"]["trained_parameters"])

def test_source_frozen_negative_and_physical_reward_success_metrics():
    e=FakeGym()
    for condition in p34.CONDITIONS:
        m=p34.make_model(p34.SEEDS[0],"full_recurrent_ppo")
        a=p34.eval_episode(e,p34.EVAL_SEEDS[0],p34.SEEDS[0],
                           condition,"initial_no_update",m)
        b=p34.eval_episode(e,p34.EVAL_SEEDS[0],p34.SEEDS[0],
                           condition,"critic_only_actor_frozen",m)
        c=p34.eval_episode(e,p34.EVAL_SEEDS[0],p34.SEEDS[0],
                           condition,"two_frame_conventional_pendulum_momentum_rule")
        assert a["source_real_action_consequence_sha256"]==b[
            "source_real_action_consequence_sha256"]
        assert a["initial_visible_source_row"]==c["initial_visible_source_row"]
        assert a["source_actual_reward_return"]==-a["source_actual_actions"]
        assert a["source_actual_actions"]<=200
        assert a["privileged_source_velocity_seen_by_policy"] is False
        assert a["evaluation_optimizer_updates"]==0
    fake={world:{arm:{"mean_true_reward":-90,
                      "actual_goal_successes_of_8":8}
                  for arm in p34.EVAL_ARMS}
          for world in p34.CONDITIONS}
    for world in p34.CONDITIONS:
        fake[world]["actor_head_only_ppo"]["mean_true_reward"]=-80
        fake[world]["full_recurrent_ppo"]["mean_true_reward"]=-110
        fake[world]["conventional_history_mlp_ppo"]["mean_true_reward"]=-90
    assert all(all(x.values()) for x in p34.frozen_gates(fake).values())
    fake["original"]["actor_head_only_ppo"]["actual_goal_successes_of_8"]=5
    assert not p34.frozen_gates(fake)["original"]["source_goal_success_count_min"]

def test_cold_checksum_and_missing_checkpoint_negative(tmp_path):
    rec={"schema":p34.SCHEMA,"engineering":"PASS_PUBLIC_ORIGINAL_SOURCE_DEV",
        "real_source_training_actions":3*4*49152,
        "real_source_optimizer_minibatches":3*4*576,
        "records":[{"model_seed":s} for s in p34.SEEDS]}
    file=tmp_path/"source.json"
    file.write_text(__import__("json").dumps(rec),encoding="utf-8")
    directory=tmp_path/"cold"
    directory.mkdir()
    with pytest.raises(RuntimeError,match="P34_COLD_SAVED_MODEL_SET_INCOMPLETE"):
        p34.cold_verify(FakeGym(),file,directory)
    with pytest.raises(ValueError,match="UNREGISTERED_ACTUAL_SOURCE_EVAL"):
        p34.eval_episode(FakeGym(),-1,p34.SEEDS[0],"original",
                         "seeded_random")
