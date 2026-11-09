"""P26 source/fact custody, contradictory actuator mapping, rehearsal negative controls."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from scripts import native_p26_contradictory_action_retention_dev as p26
from scripts import native_p18_real_cartpole_factual_dev as p18
from scripts import native_p23_exact_training_resume_gravity_dev as p23


ROOT=Path(__file__).resolve().parents[1]
SHA="1234567890abcdef1234567890abcdef12345678"


class FakeEnv:
    def reset(self,seed=None):
        self.t=0
        self.theta=-0.03 if seed%2 else 0.03
        return [0.,0.08,self.theta,-0.025 if seed%2 else 0.025],{}
    def step(self,physical_action):
        assert physical_action in (0,1)
        self.t+=1
        self.theta+=.001*(2*physical_action-1)
        return [0.,0.08,self.theta,-0.025 if self.theta<0 else 0.025],1.,self.t==5,False,{}
    def close(self):pass


class FakeGym:
    def make(self,name,render_mode=None):
        assert name=="CartPole-v1" and render_mode is None
        return FakeEnv()


def test_predeclared_continual_learning_no_science_and_independent_reset_splits():
    q=p26.frozen()
    assert q["schema"]=="ttt-hybrid.native-p26.contradictory-action-retention-frozen.v1"
    assert q["source"]["P23_exact_SHA"]==p26.P23_SHA
    assert q["qualification"]["science_first_use_authorized"] is False
    assert q["qualification"]["FORGE_owner_activation"] is False
    assert q["training"]["adaptation_same_gradient_update_count"]
    assert q["training"]["information_budget_caveat"]
    assert len(p26.TRAIN_OLD)==len(p26.TRAIN_NEW)==48
    assert len(p26.EVAL_OLD)==len(p26.EVAL_NEW)==8
    assert set(p26.TRAIN_OLD).isdisjoint(p26.EVAL_OLD)
    assert set(p26.EVAL_OLD).isdisjoint(p26.EVAL_NEW)


def test_science_and_exact_candidate_source_fail_closed(monkeypatch):
    monkeypatch.delenv("NATIVE_P26_DEV_ONLY",raising=False)
    with pytest.raises(RuntimeError,match="P26_SCIENTIFIC_FIRST_USE_NOT_AUTHORIZED"):
        p26.require_dev()
    monkeypatch.setenv("NATIVE_P26_DEV_ONLY","1")
    p26.require_dev()
    monkeypatch.delenv("EXPECTED_CANDIDATE_SHA",raising=False)
    with pytest.raises(RuntimeError,match="P26_EXACT_SOURCE_SHA_REQUIRED"):
        p26.candidate_sha()
    monkeypatch.setenv("EXPECTED_CANDIDATE_SHA",SHA)
    assert p26.candidate_sha()==SHA


def test_source_based_task_cue_masks_all_angular_velocity():
    history=[p18.factual_row([.01*k,200.+k,.005*k,300.+k],1,1.) for k in range(4)]
    a=p26.source_visible(history,False)
    b=p26.source_visible(history,True)
    assert a.shape==b.shape==(1,8,6)
    assert torch.all(a[:,:,1]==1.)
    assert torch.all(b[:,:,1]==-1.)
    assert torch.count_nonzero(a[:,:,3])==0
    assert torch.count_nonzero(b[:,:,3])==0
    assert torch.equal(a[:,:,2],b[:,:,2])
    with pytest.raises(ValueError,match="P26_TASK_CUE_MUST_BE_EXPLICIT"):
        p26.source_visible(history,1)


def test_physical_action_inversion_is_bijective_and_not_just_relabelled_credit():
    for physical in (0,1):
        assert p26.physical_action(physical,False)==physical
        assert p26.physical_action(physical,True)==1-physical
        assert p26.physical_action(p26.physical_action(physical,True),True)==physical
    with pytest.raises(ValueError,match="P26_UNREGISTERED_ACTUATOR_MAPPING"):
        p26.physical_action(2,False)


def test_real_executed_teacher_actions_and_both_context_labels_are_opposite():
    gym=FakeGym()
    a,ay,ma=p26.factual_demonstrations(gym,[911,912],False)
    b,by,mb=p26.factual_demonstrations(gym,[911,912],True)
    assert a.shape==b.shape==(10,8,6)
    assert ay.shape==by.shape==(10,)
    assert torch.equal(by,1-ay)
    assert torch.all(a[:,:,1]==1.)
    assert torch.all(b[:,:,1]==-1.)
    assert ma["factual_physical_actions_executed"]==10
    assert mb["factual_physical_actions_executed"]==10
    assert ma["source_selected_and_physical_action_SHA256"]!=mb["source_selected_and_physical_action_SHA256"]
    assert ma["expert_supervision_not_reward_only"]
    assert mb["expert_supervision_not_reward_only"]
    with pytest.raises(ValueError,match="P26_DUPLICATE_FROZEN_SOURCE_SEEDS"):
        p26.factual_demonstrations(gym,[911,911],False)


def test_rehearsal_and_sham_equal_optimizer_steps_and_real_weight_update():
    torch.set_num_threads(1)
    x,y,_=p26.factual_demonstrations(FakeGym(),[911,912],False)
    new_x,new_y,_=p26.factual_demonstrations(FakeGym(),[911,912],True)
    receipts=[]
    for arm in ("new_only","true_rehearsal","wrong_old_labels_rehearsal"):
        net,opt,gen=p26.model_initial(2741)
        initial=p23.semantic_digest(net.state_dict())
        _,updates=p26.adapt_update(net,opt,new_x,new_y,x,y,2741,arm)
        assert updates==p26.EPOCHS
        assert p23.semantic_digest(net.state_dict())!=initial
        receipts.append(updates)
    assert receipts==[32,32,32]


def test_full_checkpoint_ancestor_rng_optimizer_and_corruption_controls(tmp_path,monkeypatch):
    monkeypatch.setenv("EXPECTED_CANDIDATE_SHA",SHA)
    torch.set_num_threads(1)
    model,opt,gen=p26.model_initial(2741)
    input=torch.zeros(4,8,6)
    y=torch.zeros(4,dtype=torch.long)
    steps=p23.run_epochs(model,opt,gen,input,y,0,1,0)
    data_sha=p23.semantic_digest({"x":input,"y":y})
    item=p26.save_checkpoint(tmp_path,"base_immutable",2741,model,opt,gen,1,steps,data_sha)
    raw=p26.load_checkpoint(tmp_path,item,"base_immutable",2741,1,data_sha)
    assert raw["optimizer"]["state"]
    assert raw["parent_P23_sha"]==p26.P23_SHA
    with pytest.raises(RuntimeError,match="P26_COLD_ARCHIVE_MANIFEST_MISMATCH"):
        p26.load_checkpoint(tmp_path,{**item,"data_sha256":"f"*64},"base_immutable",2741,1,data_sha)
    b=bytearray((tmp_path/item["file"]).read_bytes())
    b[len(b)//2]^=1
    (tmp_path/item["file"]).write_bytes(bytes(b))
    with pytest.raises(RuntimeError,match="P26_COLD_ARCHIVE_CORRUPTION"):
        p26.load_checkpoint(tmp_path,item,"base_immutable",2741,1,data_sha)


def test_real_environment_eval_uses_selected_and_physical_actions_distinctly():
    g=FakeGym()
    net=p26.p19.new_model("history_gru").eval()
    for reverse,world in ((False,p26.EVAL_OLD[0]),(True,p26.EVAL_NEW[0])):
        result=p26.evaluate_episode(g,"new_only",net,reverse,world)
        assert result["actual_external_steps"]==5
        assert result["online_optimizer_updates"]==0
        assert len(result["source_fact_SHA256"])==64
        assert all(p==p26.physical_action(a,reverse)
                   for a,p in zip(result["chosen_model_actions"],result["actual_physical_actions"]))
        conventional=p26.evaluate_episode(g,"cue_aware_two_frame_strong_conventional",None,reverse,world)
        assert conventional["actual_external_steps"]==5
    with pytest.raises(ValueError,match="P26_UNREGISTERED_EVALUATION_RESET"):
        p26.evaluate_episode(g,"new_only",net,False,99999)


def test_immutable_receipt_refuses_replay_and_cli_no_dependency_install(tmp_path):
    p=tmp_path/"receipt.json"
    p26.new_receipt(p,{"ok":True})
    with pytest.raises(FileExistsError):
        p26.new_receipt(p,{"ok":False})
    assert json.loads(p.read_text())=={"ok":True}
    script=ROOT/"scripts/native_p26_contradictory_action_retention_dev.py"
    result=subprocess.run([sys.executable,"-I","-B",str(script),"--help"],
                          cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert "pip install" not in script.read_text()
    assert "--mode" in result.stdout
