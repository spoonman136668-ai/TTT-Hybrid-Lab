"""P28 exact same-source matched-data task-order vs cue diagnostic controls."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts import native_p18_real_cartpole_factual_dev as p18
from scripts import native_p23_exact_training_resume_gravity_dev as p23
from scripts import native_p26_contradictory_action_retention_dev as p26
from scripts import native_p28_context_task_order_dev as p28


ROOT=Path(__file__).resolve().parents[1]
SHA="1234567890abcdef1234567890abcdef12345678"


class FakeEnv:
    def reset(self,seed=None):
        self.t=0
        self.theta=.025 if seed%2 else -.025
        return [0.,0.15,self.theta,.05],{}
    def step(self,physical_action):
        self.t+=1
        self.theta+=.002*(2*physical_action-1)
        return [0.,0.15,self.theta,.05],1.,self.t==5,False,{}
    def close(self):pass


class FakeGym:
    def make(self,name,render_mode=None):
        assert name=="CartPole-v1" and render_mode is None
        return FakeEnv()


def test_frozen_original_P26_failure_new_seeds_science_fence_and_matched_budget():
    q=p28.frozen()
    assert q["parent"]["P26_SHA"]==p28.PARENT
    assert q["parent"]["P26_frozen_dual_result"]=="FAIL 0/3"
    assert q["boundaries"]["scientific_first_use"] is False
    assert q["boundaries"]["FORGE_activation"] is False
    assert q["task_source"]["reward_only_learning"] is False if "reward_only_learning" in q["task_source"] else True
    assert q["training"]["total_updates"]==1216
    assert q["training"]["per_task_labels_seen"]==311296
    assert set(p28.EVAL_OLD).isdisjoint(p28.EVAL_NEW)
    assert set(p28.EVAL_OLD).isdisjoint(p26.EVAL_OLD)
    assert set(p28.EVAL_NEW).isdisjoint(p26.EVAL_NEW)


def test_source_dev_flag_and_candidate_sha_missing_are_denied(monkeypatch):
    monkeypatch.delenv("NATIVE_P28_DEV_ONLY",raising=False)
    with pytest.raises(RuntimeError,match="P28_PROTECTED_SCIENTIFIC_FIRST_USE_DENIED"):
        p28.require_dev()
    monkeypatch.setenv("NATIVE_P28_DEV_ONLY","1")
    p28.require_dev()
    monkeypatch.delenv("EXPECTED_CANDIDATE_SHA",raising=False)
    with pytest.raises(RuntimeError,match="P28_EXACT_SOURCE_SHA_REQUIRED"):
        p28.exact_sha()
    monkeypatch.setenv("EXPECTED_CANDIDATE_SHA",SHA)
    assert p28.exact_sha()==SHA


def test_factual_external_source_returns_two_valid_matched_cue_cohorts():
    data,validation,meta=p28.source_sets(FakeGym())
    ox,oy,nx,ny=data
    assert ox.shape==nx.shape==(240,8,6)
    assert len(oy)==len(ny)==240
    assert torch.all(ox[:,:,1]==1.)
    assert torch.all(nx[:,:,1]==-1.)
    assert torch.count_nonzero(ox[:,:,3])==0
    assert torch.count_nonzero(nx[:,:,3])==0
    assert meta["normal_train"]["factual_physical_actions_executed"]==240
    assert meta["reverse_train"]["factual_physical_actions_executed"]==240
    assert len(validation[1])==len(validation[3])==40
    assert len(meta["source_data_SHA256"])==64


def test_task_index_stream_is_exact_deterministic_and_balanced_across_ordering():
    old,new,digest=p28.matched_indices(23,31,2741,steps=8)
    old2,new2,digest2=p28.matched_indices(23,31,2741,steps=8)
    assert torch.equal(old,old2)
    assert torch.equal(new,new2)
    assert digest==digest2
    assert len(old)==len(new)==8*256
    assert old.min()>=0 and old.max()<23
    assert new.min()>=0 and new.max()<31
    with pytest.raises(ValueError,match="P28_EVEN_UPDATES_REQUIRED"):
        p28.matched_indices(23,31,2741,steps=7)


def test_identical_initial_recurrent_weights_and_context_removed_only_in_negative():
    joint,_=p28.actor("joint_history_gru",2741)
    blocked,_=p28.actor("blocked_history_gru",2741)
    blind,_=p28.actor("joint_cue_blind_gru",2741)
    assert p23.semantic_digest(joint.state_dict())==p23.semantic_digest(blocked.state_dict())
    assert p23.semantic_digest(joint.state_dict())==p23.semantic_digest(blind.state_dict())
    with pytest.raises(ValueError,match="P28_UNREGISTERED_MODEL"):
        p28.actor("unregistered",2741)
    x=torch.randn(3,8,6)
    y=p28.cue_blind(x)
    assert torch.all(y[:,:,1]==1.)
    assert torch.equal(x[:,:,2],y[:,:,2])
    assert not torch.equal(x,y)


def test_all_four_actors_truly_update_equal_factual_indices_and_old_anchor():
    torch.set_num_threads(1)
    g=FakeGym()
    old_x,old_y,_=p26.factual_demonstrations(g,[911,912],False)
    new_x,new_y,_=p26.factual_demonstrations(g,[911,912],True)
    old,new,index_sha=p28.matched_indices(len(old_y),len(new_y),2741,steps=4)
    assert len(index_sha)==64
    for arm in p28.ARMS:
        m,opt,half,proof=p28.train_actor(arm,2741,old_x,old_y,new_x,new_y,
                                         old,new,steps=4)
        assert proof["optimizer_updates"]==4
        assert proof["old_examples_seen"]==proof["reverse_examples_seen"]==1024
        assert proof["final_state_SHA256"]!=proof["initial_state_SHA256"]
        assert proof["state_finite"]
        assert half is not None if arm=="blocked_history_gru" else half is None


def test_cold_checkpoint_and_negative_source_sampler_digest(tmp_path,monkeypatch):
    monkeypatch.setenv("EXPECTED_CANDIDATE_SHA",SHA)
    torch.set_num_threads(1)
    old_x,old_y,_=p26.factual_demonstrations(FakeGym(),[911,912],False)
    new_x,new_y,_=p26.factual_demonstrations(FakeGym(),[911,912],True)
    old,new,stream_sha=p28.matched_indices(len(old_y),len(new_y),2741,steps=2)
    net,opt,_,_=p28.train_actor("joint_history_gru",2741,old_x,old_y,new_x,new_y,
                                 old,new,steps=2)
    item=p28.checkpoint(tmp_path,"joint_history_gru",2741,net,opt,stream_sha,SHA,"a"*64)
    receipt={"factual_source":{"source_data_SHA256":"a"*64}}
    restored=p28.load_model(tmp_path,item,"joint_history_gru",2741,receipt)
    assert p23.semantic_digest(restored.state_dict())==item["model_SHA256"]
    with pytest.raises(RuntimeError,match="P28_COLD_MODEL_IDENTITY_DRIFT"):
        p28.load_model(tmp_path,{**item,"seed":2753},"joint_history_gru",2741,receipt)
    with pytest.raises(RuntimeError,match="P28_CHECKPOINT_FULL_OPTIMIZER_OR_SOURCE_CUSTODY_DRIFT"):
        p28.load_model(tmp_path,{**item,"stream_SHA256":"b"*64},"joint_history_gru",2741,receipt)
    b=bytearray((tmp_path/item["file"]).read_bytes())
    b[len(b)//2]^=1
    (tmp_path/item["file"]).write_bytes(bytes(b))
    with pytest.raises(RuntimeError,match="P28_ARCHIVE_CORRUPTION_DENIED"):
        p28.load_model(tmp_path,item,"joint_history_gru",2741,receipt)


def test_frozen_old_policy_task_action_inversion_real_steps_and_valid_cue():
    net,_=p28.actor("joint_history_gru",2741)
    for reverse,world in ((False,p28.EVAL_OLD[0]),(True,p28.EVAL_NEW[0])):
        r=p28.old_XOR_rollout(FakeGym(),net,reverse,world)
        assert r["actual_actions"]==5
        assert r["reward"]==5.
        assert len(r["source_fact_SHA256"])==64
        assert all(x==p26.physical_action(y,reverse)
                   for x,y in zip(r["actual_physical_actions"],r["chosen_model_actions"]))
    with pytest.raises(ValueError,match="P28_FROZEN_XOR_BAD_EVALUATION_WORLD"):
        p28.old_XOR_rollout(FakeGym(),net,False,p28.EVAL_NEW[0])


def test_cli_does_not_invoke_install_or_protected_science():
    path=ROOT/"scripts/native_p28_context_task_order_dev.py"
    p=subprocess.run([sys.executable,"-I","-B",str(path),"--help"],
                     cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert p.returncode==0,p.stderr
    assert "--mode" in p.stdout
    assert "pip install" not in path.read_text()
    assert "scientific_superiority" in path.read_text()
