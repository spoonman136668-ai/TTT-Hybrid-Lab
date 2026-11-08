"""P19 source-safe tests: real trainer entry remains DEV gated."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from scripts import native_p19_expert_calibration_dev as p19


ROOT=Path(__file__).resolve().parents[1]


class FakeEnv:
    def reset(self,seed=None):
        self.t=0
        self.theta=-0.04 if seed%2 else 0.04
        return [0.,0.,self.theta,0.02],{}
    def step(self,action):
        self.t+=1
        self.theta+=0.003*(2*action-1)
        return [0.,0.,self.theta,0.02],1.,self.t==5,False,{}
    def close(self):
        pass


class FakeGym:
    def make(self,name,render_mode=None):
        assert name=="CartPole-v1" and render_mode is None
        return FakeEnv()


def test_frozen_predeclared_future_seeds_are_disjoint_from_p18():
    d=p19.frozen()
    assert d["science"]["authorized"] is False
    assert d["science"]["first_use_consumed"] is False
    assert d["base_P18_sha"]=="675ef40f80c43814cc320d6b9950d37a9395c0f8"
    assert len(p19.TRAIN_SEEDS)==48
    assert len(p19.VAL_SEEDS)==8
    assert len(p19.EVAL_SEEDS)==8
    assert not (set(p19.TRAIN_SEEDS)&set(p19.VAL_SEEDS))
    assert not ((set(p19.TRAIN_SEEDS)|set(p19.VAL_SEEDS)|set(p19.EVAL_SEEDS))&set(__import__(
        "scripts.native_p18_real_cartpole_factual_dev",fromlist=["EVAL_SEEDS"]).EVAL_SEEDS))
    assert d["design"]["epochs"]==p19.EPOCHS==32
    assert d["design"]["actors"]==[
        "trained_history_GRU_489_params",
        "trained_history_MLP_471_params",
        "trained_linear_observation_10_params"]


def test_explicit_owner_development_flag_is_fail_closed(monkeypatch):
    monkeypatch.delenv("NATIVE_P19_DEV_ONLY",raising=False)
    with pytest.raises(RuntimeError,match="P19_SCIENCE_FIRST_USE_NOT_AUTHORIZED"):
        p19.require_dev()
    monkeypatch.setenv("NATIVE_P19_DEV_ONLY","1")
    p19.require_dev()


def test_linear_actor_only_reads_current_public_four_sensor_coordinates():
    net=p19.LinearObservation()
    assert sum(p.numel() for p in net.parameters())==10
    x=torch.zeros((1,8,6),dtype=torch.float32)
    logits,_=net(x)
    x[:,0:7,:]=999.
    x[:,-1,4:]=999.
    changed,_=net(x)
    assert torch.equal(logits,changed)
    with pytest.raises(ValueError,match="P19_LINEAR_INPUT_SHAPE"):
        net(torch.zeros((1,7,6)))


def test_teacher_dataset_is_only_completed_factual_source_actions():
    x,y,receipt=p19.expert_dataset(FakeGym(),[911,912])
    assert x.shape==(10,8,6)
    assert y.shape==(10,)
    assert receipt["actual_teacher_actions"]==10
    assert receipt["actual_teacher_rewards"]==10.
    assert receipt["teacher_labels_from_present_public_sensor_only"]
    assert len(receipt["factual_action_outcome_sha256"])==64
    assert set(y.tolist())=={0,1}
    with pytest.raises(ValueError,match="P19_BAD_DEMONSTRATION_SEEDS"):
        p19.expert_dataset(FakeGym(),[911,911])


def test_real_weight_update_and_fresh_model_state_exact_for_small_offline_unit():
    torch.set_num_threads(1)
    x,y,_=p19.expert_dataset(FakeGym(),[911,912])
    for kind in p19.KINDS:
        m,r=p19.fit(kind,2741,x,y,x,y,epochs=1)
        assert r["optimizer_updates"]==1
        assert r["trained_weight_updated"]
        assert r["fresh_object_state_only_restore_exact"]
        assert not r["real_process_death_cold_restore_proven"]
        assert r["teacher_supervision_not_reward_only"]
        assert 0<=r["validation_agreement"]<=1


def test_real_eval_interface_for_seven_arms_and_finite_completed_episodes():
    torch.manual_seed(1)
    models={k:p19.new_model(k).eval() for k in p19.KINDS}
    models["untrained_gru"]=p19.new_model("history_gru").eval()
    results=[p19.eval_episode(FakeGym(),arm,models,p19.EVAL_SEEDS[0],2741)
             for arm in p19.ARMS]
    assert len({tuple(x["initial_observation"]) for x in results})==1
    assert all(x["real_episode_actions"]==5 for x in results)
    assert all(x["real_episode_reward"]==5. for x in results)
    assert all(x["eval_optimizer_updates"]==0 for x in results)


def test_help_is_sealed_without_scientific_environment():
    p=ROOT/"scripts/native_p19_expert_calibration_dev.py"
    r=subprocess.run([sys.executable,"-I","-B",str(p),"--help"],
                     capture_output=True,text=True,cwd=ROOT,timeout=30)
    assert r.returncode==0,r.stderr
    assert "--s29-root" in r.stdout
    src=p.read_text()
    assert "pip install" not in src
    assert "scientific_first_use" in src
    assert "reward_only_competence_proven" in src
