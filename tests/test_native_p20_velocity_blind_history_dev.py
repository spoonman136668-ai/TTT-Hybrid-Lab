"""P20 public DEV source, velocity-masking and memory-necessity negative tests."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch

from scripts import native_p20_velocity_blind_history_dev as p20
from scripts import native_p19_expert_calibration_dev as p19
from scripts import native_p18_real_cartpole_factual_dev as p18


ROOT=Path(__file__).resolve().parents[1]


class FakeEnv:
    def reset(self,seed=None):
        self.steps=0
        self.theta=-0.03 if seed%2 else 0.03
        return [0.,0.03,self.theta,0.04],{}
    def step(self,action):
        self.steps+=1
        self.theta+=0.002*(2*action-1)
        return [0.,0.03,self.theta,0.04],1.,self.steps==5,False,{}
    def close(self):pass


class FakeGym:
    def make(self,name,render_mode=None):
        assert name=="CartPole-v1" and render_mode is None
        return FakeEnv()


def test_frozen_protocol_distinct_public_development_worlds_and_science_fence():
    d=p20.frozen()
    assert d["stack"]["p18_exact_sha"]=="675ef40f80c43814cc320d6b9950d37a9395c0f8"
    assert d["stack"]["p19_exact_sha"]=="d7e1535f615c7f9f0ca70b2ab9059fdf1f912726"
    assert d["scientific_boundary"]["first_use_authorized"] is False
    assert d["scientific_boundary"]["sealed_scientific_data_consumed"] is False
    assert len(p20.TRAIN_SEEDS)==48 and len(p20.VAL_SEEDS)==8 and len(p20.EVAL_SEEDS)==8
    assert set(p20.TRAIN_SEEDS).isdisjoint(p20.VAL_SEEDS)
    assert set(p20.TRAIN_SEEDS).isdisjoint(p20.EVAL_SEEDS)
    assert set(p20.VAL_SEEDS).isdisjoint(p20.EVAL_SEEDS)
    assert set(p20.EVAL_SEEDS).isdisjoint(p19.EVAL_SEEDS)
    assert set(p20.EVAL_SEEDS).isdisjoint(p18.EVAL_SEEDS)
    assert d["design"]["meaningful_history_requirement_each_seed"]["versus_trained_current_only_gru_mean_reward_gain_min"]==15


def test_science_first_use_is_denied_without_original_owner_dev_flag(monkeypatch):
    monkeypatch.delenv("NATIVE_P20_DEV_ONLY",raising=False)
    with pytest.raises(RuntimeError,match="P20_SCIENCE_FIRST_USE_NOT_AUTHORIZED"):
        p20.require_dev()
    monkeypatch.setenv("NATIVE_P20_DEV_ONLY","1")
    p20.require_dev()


def test_every_history_frame_masks_velocity_and_current_only_is_matched():
    raw=torch.arange(2*8*6,dtype=torch.float32).reshape(2,8,6)
    v=p20.visible_history(raw)
    assert torch.count_nonzero(v[:,:,1])==0
    assert torch.count_nonzero(v[:,:,3])==0
    assert torch.equal(v[:,:,0],raw[:,:,0])
    assert torch.equal(v[:,:,2],raw[:,:,2])
    assert torch.equal(v[:,:,4],raw[:,:,4])
    assert torch.equal(v[:,:,5],raw[:,:,5])
    c=p20.visible_history(raw,current_only=True)
    assert torch.count_nonzero(c[:,:-1,:])==0
    assert torch.equal(c[:,-1,:],v[:,-1,:])
    assert torch.count_nonzero(raw[:,:,1])>0
    with pytest.raises(ValueError,match="P20_PUBLIC_SENSOR_SHAPE"):
        p20.visible_history(raw[:,:7,:])


def test_history_reverse_preserves_last_public_current_event():
    raw=[p18.factual_row([float(i),100+i,.01*i,200+i],1,1.) for i in range(5)]
    a=p20.visible_rows(raw)[0]
    b=p20.visible_rows(raw,reversed_history=True)[0]
    assert torch.equal(a[-1],b[-1])
    assert torch.equal(a[0],torch.zeros(6))
    assert torch.count_nonzero(a[:,1])==0
    assert torch.count_nonzero(b[:,3])==0
    assert not torch.equal(a,b)


def test_finite_difference_uses_only_completed_history_not_hidden_velocity():
    row=p18.factual_row([0.,800.,.01,900.],0,0.)
    assert p20.finite_difference_action([row])==1
    old=p18.factual_row([0.,-800.,.1,-900.],0,0.)
    new=p18.factual_row([0.,+800.,.01,900.],1,1.)
    assert p20.finite_difference_action([old,new])==0
    changed=p18.factual_row([0.,-999.,.01,-999.],1,1.)
    assert p20.finite_difference_action([old,changed])==0


def test_actual_source_interface_eight_arms_action_receipts_without_teacher_leak():
    torch.manual_seed(1)
    models={
        "full_history_gru":p19.new_model("history_gru").eval(),
        "current_only_gru":p19.new_model("history_gru").eval(),
        "full_history_mlp":p19.new_model("history_mlp").eval(),
        "current_only_linear":p19.new_model("linear_observation").eval()}
    outcomes=[p20.actual_episode(FakeGym(),arm,models,p20.EVAL_SEEDS[0],2741)
              for arm in p20.ARMS]
    assert len(outcomes)==8
    assert len({tuple(r["public_reset_observation"]) for r in outcomes})==1
    assert all(r["actual_actions"]==5 and r["actual_reward"]==5.
               and r["eval_optimizer_updates"]==0 for r in outcomes)
    assert len({r["factual_trace_sha256"] for r in outcomes})>=1
    with pytest.raises(ValueError,match="P20_EVALUATION_SOURCE_DENIED"):
        p20.actual_episode(FakeGym(),"hidden_oracle",models,p20.EVAL_SEEDS[0],2741)


def test_cli_help_requires_no_environment_or_paid_dependency():
    p=ROOT/"scripts/native_p20_velocity_blind_history_dev.py"
    result=subprocess.run([sys.executable,"-I","-B",str(p),"--help"],
                         cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert "--s29-root" in result.stdout
    text=p.read_text()
    assert "pip install" not in text
    assert "first_use" in text
    assert "scientific_transfer_or_superiority_established" in text
