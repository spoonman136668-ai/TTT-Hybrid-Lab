"""P28-R1: fresh source-world admission, original model SHA and no training."""
from pathlib import Path
import json
import subprocess
import sys
import pytest
import torch

from scripts import native_p28_r1_cold_factual_eval_dev as r1
from scripts import native_p28_context_task_order_dev as original
from scripts import native_p26_contradictory_action_retention_dev as p26
from scripts import native_p23_exact_training_resume_gravity_dev as p23


ROOT=Path(__file__).resolve().parents[1]
SOURCE_SHA="1234567890abcdef1234567890abcdef12345678"


class FakeEnv:
    def reset(self,seed=None):
        self.steps=0
        self.theta=-.025 if seed%2 else .025
        return [0.,.14,self.theta,.06],{}
    def step(self,physical):
        assert physical in (0,1)
        self.steps+=1
        self.theta+=.002*(2*physical-1)
        return [0.,.14,self.theta,.06],1.,self.steps==4,False,{}
    def close(self):pass


class FakeGym:
    def make(self,name,render_mode=None):
        assert name=="CartPole-v1" and render_mode is None
        return FakeEnv()


def test_original_science_failure_preserved_and_fresh_eval_worlds_disjoint():
    q,original_frozen=r1.frozen()
    assert q["original"]["candidate_SHA"]==r1.TRAIN_SHA
    assert q["original"]["first_owner_run"]==42541253699
    assert q["original"]["training_checkpoint_count"]==15
    assert q["recovery"]["no_new_training_or_optimizer_steps"]
    assert q["recovery"]["no_scientific_first_use"]
    assert len(r1.NORMAL_EVAL)==len(r1.REVERSE_EVAL)==8
    assert set(r1.NORMAL_EVAL).isdisjoint(r1.REVERSE_EVAL)
    assert set(r1.NORMAL_EVAL).isdisjoint(original.EVAL_OLD)
    assert set(r1.REVERSE_EVAL).isdisjoint(original.EVAL_NEW)
    assert q["recovery"]["frozen_joint_both_tasks_min_return"]==original_frozen["preregistered"]["old_and_reverse_min_reward_each"]


def test_owner_source_fence_without_automatic_retraining(monkeypatch):
    monkeypatch.delenv("NATIVE_P28_R1_DEV_ONLY",raising=False)
    with pytest.raises(RuntimeError,match="P28_R1_SCIENCE_FIRST_USE_NOT_AUTHORIZED"):
        r1.require_dev()
    monkeypatch.setenv("NATIVE_P28_R1_DEV_ONLY","1")
    r1.require_dev()
    monkeypatch.delenv("EXPECTED_CANDIDATE_SHA",raising=False)
    with pytest.raises(RuntimeError,match="P28_R1_DISTINCT_EVALUATION_SOURCE_SHA_REQUIRED"):
        r1.exact_source()
    monkeypatch.setenv("EXPECTED_CANDIDATE_SHA",r1.TRAIN_SHA)
    with pytest.raises(RuntimeError,match="P28_R1_DISTINCT_EVALUATION_SOURCE_SHA_REQUIRED"):
        r1.exact_source()
    monkeypatch.setenv("EXPECTED_CANDIDATE_SHA",SOURCE_SHA)
    assert r1.exact_source()==SOURCE_SHA


def test_original_training_manifest_exact_source_checkpoint_set_only(tmp_path):
    m={"schema":original.SCHEMA,"mode":"train",
       "candidate_sha":r1.TRAIN_SHA,"parent_sha":r1.SOURCE_PARENT,
       "entries":[{"kind":f"test{i}","seed":i} for i in range(15)],
       "training":[{} for _ in range(12)],
       "same_indices_exactly_between_all_arms":True,
       "same_old_reverse_sample_counts_and_steps":True,
       "science_first_use":"NOT_RUN","protected_science_worlds_consumed":[]}
    p=tmp_path/"train.json"
    p.write_text(json.dumps(m))
    assert r1.read_manifest(p)["candidate_sha"]==r1.TRAIN_SHA
    m["candidate_sha"]=SOURCE_SHA
    p.write_text(json.dumps(m))
    with pytest.raises(RuntimeError,match="P28_R1_ORIGINAL_TRAINING_RECEIPT_DRIFT"):
        r1.read_manifest(p)


def test_all_eight_arms_use_true_source_actions_on_new_only_registered_worlds():
    gym=FakeGym()
    models={
        "joint_history_gru":original.actor("joint_history_gru",2741)[0].eval(),
        "joint_history_mlp":original.actor("joint_history_mlp",2741)[0].eval(),
        "frozen_old_half_gru_plus_XOR":original.actor("joint_history_gru",2741)[0].eval()}
    for reverse,seed in ((False,r1.NORMAL_EVAL[0]),(True,r1.REVERSE_EVAL[0])):
        results={}
        for arm in r1.EVALUATE:
            net=(models["joint_history_mlp"] if arm=="joint_history_mlp"
                 else models["frozen_old_half_gru_plus_XOR"] if arm=="frozen_old_half_gru_plus_XOR"
                 else models["joint_history_gru"])
            if arm in ("full_public_sensor_teacher","cue_aware_two_frame_strong_conventional"):
                net=None
            result=r1.actual_episode(gym,arm,net,reverse,seed)
            assert result["real_action_count"]==4
            assert result["reward"]==4.
            assert result["online_optimizer_updates"]==0
            assert len(result["source_fact_sha256"])==64
            assert all(p==p26.physical_action(a,reverse) for a,p in zip(
                result["selected_actions"],result["actual_physical_actions"]))
            results[arm]=result
        assert len({tuple(y["initial_observation"]) for y in results.values()})==1
    with pytest.raises(ValueError,match="P28_R1_NEW_SOURCE_EVAL_SEED_OR_ARM_DENIED"):
        r1.actual_episode(gym,"joint_history_gru",models["joint_history_gru"],False,original.EVAL_OLD[0])
    with pytest.raises(ValueError,match="P28_R1_NEW_SOURCE_EVAL_SEED_OR_ARM_DENIED"):
        r1.actual_episode(gym,"unknown",models["joint_history_gru"],False,r1.NORMAL_EVAL[0])


def test_original_torch_checkpoint_source_and_optimizer_are_not_rewritten(tmp_path,monkeypatch):
    monkeypatch.setenv("EXPECTED_CANDIDATE_SHA",r1.TRAIN_SHA)
    torch.set_num_threads(1)
    net,opt=original.actor("joint_history_gru",2741)
    x=torch.zeros((4,8,6))
    y=torch.zeros(4,dtype=torch.long)
    logits,_=net(x)
    loss=torch.nn.functional.cross_entropy(logits,y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    e=original.checkpoint(tmp_path,"joint_history_gru",2741,net,opt,"a"*64,r1.TRAIN_SHA,"b"*64)
    m={"factual_source":{"source_data_SHA256":"b"*64}}
    a=r1.load_original_model(tmp_path,e,"joint_history_gru",2741,m)
    assert p23.semantic_digest(a.state_dict())==e["model_SHA256"]
    with pytest.raises(RuntimeError,match="P28_R1_MODEL_SEED_OR_DATA_IDENTITY_DENIED"):
        r1.load_original_model(tmp_path,{**e,"seed":2753},"joint_history_gru",2741,m)
    raw=bytearray((tmp_path/e["file"]).read_bytes())
    raw[len(raw)//2]^=1
    (tmp_path/e["file"]).write_bytes(raw)
    with pytest.raises(RuntimeError,match="P28_R1_ORIGINAL_CHECKPOINT_CORRUPT"):
        r1.load_original_model(tmp_path,e,"joint_history_gru",2741,m)


def test_help_has_no_training_mode_or_source_installs():
    f=ROOT/"scripts/native_p28_r1_cold_factual_eval_dev.py"
    result=subprocess.run([sys.executable,"-I","-B",str(f),"--help"],
                          cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert "--checkpoint-dir" in result.stdout
    assert "--train-receipt" in result.stdout
    assert "--mode" not in result.stdout
    assert "pip install" not in f.read_text()
    assert "train(gym" not in f.read_text()
