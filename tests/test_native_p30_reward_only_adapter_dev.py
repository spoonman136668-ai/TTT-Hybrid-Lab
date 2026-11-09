"""P30 negative science/source custody and actual intervention learning tests."""
from pathlib import Path
import copy
import json
import subprocess
import sys
import pytest
import torch

from scripts import native_p26_contradictory_action_retention_dev as p26
from scripts import native_p28_r1_cold_factual_eval_dev as r1
from scripts import native_p30_reward_only_adapter_dev as p30
from scripts import native_p23_exact_training_resume_gravity_dev as p23

ROOT=Path(__file__).resolve().parents[1]
SHA="1234567890abcdef1234567890abcdef12345678"


class FakeEnv:
    def reset(self,seed=None):
        self.step_count=0
        self.theta=.02 if seed%2 else -.02
        return [0.,.01,self.theta,.04],{}
    def step(self,action):
        assert action in (0,1)
        self.step_count+=1
        self.theta+=.002*(2*action-1)
        return [0.,.01,self.theta,.04],1.,self.step_count>=4,False,{}
    def close(self):pass


class FakeGym:
    def make(self,name,render_mode=None):
        assert name=="CartPole-v1" and render_mode is None
        return FakeEnv()


def test_preregister_source_seeds_science_denial_and_prior_negative():
    q=p30.frozen()
    assert q["parent"]["P28_R1_qualified_SHA"]==p30.TRAIN_PARENT
    assert q["parent"]["prior_P26_failure_preserved"]=="dual-task rehearsal FAIL 0/3"
    assert q["qualification"]["protected_scientific_first_use"] is False
    assert q["qualification"]["one_use_exact_candidate_only"]
    assert len(p30.SEEDS)==3
    assert len(p30.TRAIN_NORMAL)==len(p30.TRAIN_REVERSE)==12
    assert len(p30.EVAL_NORMAL)==len(p30.EVAL_REVERSE)==8
    assert set(p30.TRAIN_NORMAL).isdisjoint(p30.EVAL_NORMAL)
    assert set(p30.TRAIN_REVERSE).isdisjoint(p30.EVAL_REVERSE)
    assert set(p30.EVAL_NORMAL).isdisjoint(r1.NORMAL_EVAL)


def test_source_sha_and_dev_first_use_are_fail_closed(monkeypatch):
    monkeypatch.delenv("NATIVE_P30_DEV_ONLY",raising=False)
    with pytest.raises(RuntimeError,match="P30_SCIENCE_FIRST_USE_NOT_AUTHORIZED"):
        p30.require_dev()
    monkeypatch.setenv("NATIVE_P30_DEV_ONLY","1")
    p30.require_dev()
    monkeypatch.delenv("EXPECTED_CANDIDATE_SHA",raising=False)
    with pytest.raises(RuntimeError,match="P30_EXACT_NEW_ADAPTER_SOURCE_REQUIRED"):
        p30.exact_sha()
    monkeypatch.setenv("EXPECTED_CANDIDATE_SHA",p30.TRAIN_PARENT)
    with pytest.raises(RuntimeError,match="P30_EXACT_NEW_ADAPTER_SOURCE_REQUIRED"):
        p30.exact_sha()
    monkeypatch.setenv("EXPECTED_CANDIDATE_SHA",SHA)
    assert p30.exact_sha()==SHA


def test_real_source_both_candidate_interventions_execute_and_audit():
    m=r1.p19.new_model("history_gru").eval()
    for reverse,world in ((False,p30.TRAIN_NORMAL[0]),(True,p30.TRAIN_REVERSE[0])):
        a=p30.source_episode(FakeGym(),m,reverse,0,world)
        b=p30.source_episode(FakeGym(),m,reverse,1,world)
        assert a["source_return"]==b["source_return"]==4.
        assert a["factual_actions"]==b["factual_actions"]==4
        assert len(a["factual_consequence_SHA256"])==64
        assert a["original_initial_observation"]==b["original_initial_observation"]
        assert a["physically_executed_actions"][0]!=b["physically_executed_actions"][0]
        for receipt,bit in ((a,0),(b,1)):
            assert all(phys==chosen^int(reverse)
                       for chosen,phys in zip(receipt["chosen_actions"],
                                               receipt["physically_executed_actions"]))
    with pytest.raises(ValueError,match="P30_UNREGISTERED_FACTUAL_WORLD"):
        p30.source_episode(FakeGym(),m,False,0,r1.NORMAL_EVAL[0])
    with pytest.raises(ValueError,match="P30_EXPLICIT_CUE_AND_TWO_PHYSICAL_MAPPING_CANDIDATES_REQUIRED"):
        p30.source_episode(FakeGym(),m,False,2,p30.TRAIN_NORMAL[0])


def factual_pair_examples():
    source=[]
    for cue,seeds,rew0,rew1 in (
        (1.,p30.TRAIN_NORMAL,200.,12.),
        (-1.,p30.TRAIN_REVERSE,13.,200.)):
        for world in seeds:
            source.append({"cue":cue,"world":world,
                "episode_return_bit0":rew0,"episode_return_bit1":rew1,
                "two_truly_executed_real_source_episodes":[{},{}]})
    return source


def test_frozen_reward_only_two_parameter_adapter_vs_same_source_conventional_and_sham():
    torch.set_num_threads(1)
    data=factual_pair_examples()
    assert len(data)==24
    table=p30.table_from_actual_rewards(data)
    assert table["1"]["mapping_bit"]==0
    assert table["-1"]["mapping_bit"]==1
    assert table["1"]["source_episodes_per_candidate"]==12
    learned={}
    for arm in p30.LEARNED:
        model,opt,receipt=p30.train_from_actual_rewards(data,2741,arm)
        assert receipt["optimizer_updates"]==48
        assert receipt["raw_real_reward_pairs"]==24
        assert receipt["final_sha"]!=receipt["initial_sha"]
        assert opt.state
        learned[arm]=model
    good=learned["two_parameter_cue_logistic_adapter"]
    wrong=learned["wrong_credited_two_parameter_logistic"]
    assert p30.chosen_bit("two_parameter_cue_logistic_adapter",1.,good)==0
    assert p30.chosen_bit("two_parameter_cue_logistic_adapter",-1.,good)==1
    assert p30.chosen_bit("wrong_credited_two_parameter_logistic",1.,wrong)==1
    assert p30.chosen_bit("wrong_credited_two_parameter_logistic",-1.,wrong)==0
    assert p30.chosen_bit("conventional_two_context_reward_table",-1.,table=table)==1
    assert p30.chosen_bit("unadapted_identity",-1.)==0
    assert p30.chosen_bit("manually_explicit_XOR_oracle",-1.)==1
    with pytest.raises(ValueError,match="P30_ONLY_FROZEN_PUBLIC_CONTEXTS"):
        p30.chosen_bit("unadapted_identity",0.)


def test_source_intervention_pairs_real_reset_count_without_privileged_label():
    torch.set_num_threads(1)
    m=r1.p19.new_model("history_gru").eval()
    pairs=p30.training_pairs(FakeGym(),m,2741)
    assert len(pairs)==24
    assert sum(v["cue"]==1. for v in pairs)==12
    assert sum(v["cue"]==-1. for v in pairs)==12
    assert all(len(x["two_truly_executed_real_source_episodes"])==2 for x in pairs)
    assert all(x["episode_return_bit0"]==4. for x in pairs)
    assert all(x["episode_return_bit1"]==4. for x in pairs)
    with pytest.raises(RuntimeError,match="P30_TABLE_ACTION_AMBIGUITY"):
        p30.table_from_actual_rewards(pairs)


def test_adapter_checkpoint_has_exact_source_model_and_optimizer_denials(tmp_path,monkeypatch):
    monkeypatch.setenv("EXPECTED_CANDIDATE_SHA",SHA)
    torch.set_num_threads(1)
    pairs=factual_pair_examples()
    net,opt,_=p30.train_from_actual_rewards(pairs,2741,p30.LEARNED[0])
    data_sha="a"*64
    model_sha="b"*64
    entry=p30.checkpoint(tmp_path,2741,p30.LEARNED[0],net,opt,data_sha,model_sha)
    restored=p30.read_checkpoint(tmp_path,entry,p30.LEARNED[0],2741,data_sha,model_sha,SHA)
    assert p23.semantic_digest(net.state_dict())==p23.semantic_digest(restored.state_dict())
    with pytest.raises(RuntimeError,match="P30_SOURCE_OR_ADAPTER_IDENTITY_DENIED"):
        p30.read_checkpoint(tmp_path,{**entry,"seed":2753},p30.LEARNED[0],2741,data_sha,model_sha,SHA)
    with pytest.raises(RuntimeError,match="P30_ADAPTER_OPTIMIZER_OR_SOURCE_DRIFT"):
        p30.read_checkpoint(tmp_path,entry,p30.LEARNED[0],2741,data_sha,model_sha,"c"*40)
    archive=tmp_path/entry["file"]
    b=bytearray(archive.read_bytes())
    b[len(b)//2]^=1
    archive.write_bytes(b)
    with pytest.raises(RuntimeError,match="P30_ADAPTER_ARCHIVE_CORRUPT"):
        p30.read_checkpoint(tmp_path,entry,p30.LEARNED[0],2741,data_sha,model_sha,SHA)


def test_no_source_first_use_from_help_and_immutable_receipts(tmp_path):
    script=ROOT/"scripts/native_p30_reward_only_adapter_dev.py"
    result=subprocess.run([sys.executable,"-I","-B",str(script),"--help"],
                          cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert "--original-checkpoints" in result.stdout
    assert "pip install" not in script.read_text()
    a=tmp_path/"r.json"
    p30.immutable_receipt(a,{"first":True})
    with pytest.raises(FileExistsError):
        p30.immutable_receipt(a,{"first":False})
    assert json.loads(a.read_text())=={"first":True}
