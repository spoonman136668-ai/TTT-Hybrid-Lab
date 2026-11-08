"""P23: exact optimizer+sampler+host RNG continuation and negative custody."""
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from scripts import native_p23_exact_training_resume_gravity_dev as p23


ROOT=Path(__file__).resolve().parents[1]
SHA="1234567890abcdef1234567890abcdef12345678"


class FakeEnv:
    def __init__(self):
        self.unwrapped=self
        self.gravity=9.8
    def reset(self,seed=None):
        self.t=0
        self.theta=.01 if seed%2 else -.01
        return [0.,.3,self.theta,.03],{}
    def step(self,action):
        self.t+=1
        self.theta+=.001*(2*action-1)*self.gravity
        return [0.,.3,self.theta,.03],1.,self.t==4,False,{}
    def close(self):pass


class FakeGym:
    def make(self,name,render_mode=None):
        assert name=="CartPole-v1" and render_mode is None
        return FakeEnv()


def test_frozen_worlds_optimizer_fence_and_no_science():
    q=p23.frozen()
    assert q["source"]["parent_P21_SHA"]==p23.P21_PARENT
    assert q["custody"]["scientific_first_use_authorized"] is False
    assert q["custody"]["original_S26_S27_first_use"] is False
    assert q["training"]["total_epochs"]==32
    assert q["training"]["interruption_after_epoch"]==16
    assert p23.WORLD_SEEDS==(81011,81012,81013,81014,81015,81016,81017,81018)
    assert len(p23.GRAVITY)==3
    assert p23.KINDS==("full_history_gru","current_only_gru")
    assert p23.MODEL_SEEDS==(2741,2753,2767)


def test_owner_gate_and_exact_source_sha(monkeypatch):
    monkeypatch.delenv("NATIVE_P23_DEV_ONLY",raising=False)
    with pytest.raises(RuntimeError,match="P23_PROTECTED_SCIENCE_FIRST_USE_DENIED"):
        p23.require_dev()
    monkeypatch.setenv("NATIVE_P23_DEV_ONLY","1")
    p23.require_dev()
    monkeypatch.delenv("EXPECTED_CANDIDATE_SHA",raising=False)
    with pytest.raises(RuntimeError,match="P23_EXACT_CANDIDATE_SOURCE_REQUIRED"):
        p23.candidate_sha()
    monkeypatch.setenv("EXPECTED_CANDIDATE_SHA",SHA)
    assert p23.candidate_sha()==SHA


def test_current_only_sees_no_history_and_all_velocity_dimensions_removed():
    x=torch.randn(4,8,6)
    full=p23.actor_input(x,False)
    only=p23.actor_input(x,True)
    assert torch.count_nonzero(full[:,:,1])==0
    assert torch.count_nonzero(full[:,:,3])==0
    assert torch.count_nonzero(only[:,:-1,:])==0
    assert torch.equal(full[:,-1,:],only[:,-1,:])
    assert p23.factual_input_digest(full,torch.zeros(4,dtype=torch.long))!=p23.factual_input_digest(only,torch.zeros(4,dtype=torch.long))


def test_bit_exact_split_restart_optimizer_sampler_and_host_rng(monkeypatch):
    monkeypatch.setenv("EXPECTED_CANDIDATE_SHA",SHA)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    g=torch.Generator().manual_seed(44)
    x=p23.actor_input(torch.randn((24,8,6),generator=g),False)
    y=torch.randint(0,2,(24,),generator=g)
    digest=p23.factual_input_digest(x,y)

    def initial():
        return p23.model_and_optimizer("full_history_gru",2741)

    net,opt,sampler=initial()
    n=p23.run_epochs(net,opt,sampler,x,y,0,4,0)
    reference=p23.state_payload(net,opt,sampler,"full_history_gru",2741,4,n,digest)
    reference_sha=p23.semantic_digest(reference)

    net,opt,sampler=initial()
    n=p23.run_epochs(net,opt,sampler,x,y,0,2,0)
    half=p23.state_payload(net,opt,sampler,"full_history_gru",2741,2,n,digest)
    blob=io.BytesIO()
    torch.save(half,blob)
    reload=torch.load(io.BytesIO(blob.getvalue()),weights_only=True,map_location="cpu")
    p23.validate_payload(reload,"full_history_gru",2741,2,digest)

    net,opt,sampler=initial()
    net.load_state_dict(reload["model"],strict=True)
    opt.load_state_dict(reload["optimizer"])
    sampler.set_state(reload["sampler_rng"])
    torch.set_rng_state(reload["torch_cpu_rng"])
    import random
    random.setstate(reload["python_rng"])
    p23.restore_numpy_rng(reload["numpy_rng"])
    n=p23.run_epochs(net,opt,sampler,x,y,2,4,reload["optimizer_updates"])
    resumed=p23.state_payload(net,opt,sampler,"full_history_gru",2741,4,n,digest)
    assert p23.semantic_digest(resumed)==reference_sha


def test_checkpoint_source_epoch_rng_and_optimizer_denial(monkeypatch):
    monkeypatch.setenv("EXPECTED_CANDIDATE_SHA",SHA)
    net,opt,sampler=p23.model_and_optimizer("full_history_gru",2741)
    x=torch.zeros(4,8,6)
    y=torch.zeros(4,dtype=torch.long)
    steps=p23.run_epochs(net,opt,sampler,x,y,0,1,0)
    digest=p23.factual_input_digest(x,y)
    item=p23.state_payload(net,opt,sampler,"full_history_gru",2741,1,steps,digest)
    p23.validate_payload(item,"full_history_gru",2741,1,digest)
    for mutation in [
        lambda q:q.update(candidate_SHA="0"*40),
        lambda q:q.update(completed_epochs=0),
        lambda q:q.pop("optimizer"),
        lambda q:q.pop("sampler_rng"),
        lambda q:q.pop("python_rng"),
        lambda q:q.pop("numpy_rng")
    ]:
        damaged=copy.copy(item)
        mutation(damaged)
        with pytest.raises(RuntimeError,match="P23_"):
            p23.validate_payload(damaged,"full_history_gru",2741,1,digest)


def test_real_action_source_unfamiliar_gravity_and_conventional_reference():
    gym=FakeGym()
    net=p23.p19.new_model("history_gru").eval()
    for name,factor in p23.GRAVITY:
        a=p23.rollout(gym,"full_history_gru",factor,p23.WORLD_SEEDS[0],2741,net)
        b=p23.rollout(gym,"full_history_gru",factor,p23.WORLD_SEEDS[0],2741,net)
        assert a==b
        assert a["actual_actions"]==4
        assert a["actual_reward"]==4.
        assert len(a["factual_SHA256"])==64
        assert p23.rollout(gym,"two_frame",factor,p23.WORLD_SEEDS[0],2741)["actual_reward"]==4.
    with pytest.raises(ValueError,match="P23_UNREGISTERED_TRANSFER_WORLD"):
        p23.rollout(gym,"teacher",1.0,1,2741)


def test_manifest_no_replay_and_archive_integrity(tmp_path,monkeypatch):
    monkeypatch.setenv("EXPECTED_CANDIDATE_SHA",SHA)
    p=tmp_path/"data.json"
    p23.new_file(p,{"ok":True})
    with pytest.raises(FileExistsError):
        p23.new_file(p,{"ok":False})
    assert json.loads(p.read_text())=={"ok":True}
    assert p23.sha_bytes(b"bad")!=p23.sha_bytes(b"good")


def test_isolated_cli_help_without_science_or_installs():
    script=ROOT/"scripts/native_p23_exact_training_resume_gravity_dev.py"
    out=subprocess.run([sys.executable,"-I","-B",str(script),"--help"],
                       cwd=ROOT,capture_output=True,text=True,timeout=25)
    assert out.returncode==0,out.stderr
    assert "--mode" in out.stdout and "evaluate" in out.stdout
    assert "pip install" not in script.read_text()
    assert "FORGE_owner_activation" in script.read_text()
