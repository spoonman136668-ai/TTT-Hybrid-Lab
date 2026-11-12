"""F7 no-replay POSIX claim controls (pure local test directory only)."""
from __future__ import annotations
import json
import os
from pathlib import Path
import pytest
from experiments import p85mr_f7_source_once as f7


def test_f7_source_requires_owner_identity_and_exclusive_real_FIT_context():
    with pytest.raises(PermissionError,match="F7_UNTRUSTED_OWNER_OR_REPLAY_ATTEMPT"):
        f7._owner({})
    with pytest.raises(PermissionError,match="F7_UNTRUSTED_OWNER_OR_REPLAY_ATTEMPT"):
        f7._owner({"GITHUB_REPOSITORY":f7.PROJECT})
    assert f7.SOURCE_SHA=="c94e9a4cd3641588b6b8507e71ccee5c27032bdf"
    assert f7.cfg.MAX_ACTIONS==12288
    assert f7.cfg.FIT_WORLDS==128
    assert f7.cfg.frozen()["authority"]["approve_F7_CAL"] is False


def test_f7_local_posix_only_exclusive_claim_fsync_and_denial(tmp_path):
    if os.name!="posix":
        with pytest.raises(PermissionError,match="F7_POSIX_NATIVE_CUSTODY_REQUIRED"):
            f7._fsync_dir(tmp_path)
        return
    original=tmp_path/"ONCE"
    f7._mkdir(original)
    with pytest.raises(FileExistsError):
        f7._mkdir(original)
    i=f7._exclusive(original/"001.intent.json",{
        "session":"fixture:only","effect":"NOT_ATTEMPTED","no_replay":True})
    assert len(i)==64
    with pytest.raises(FileExistsError):
        f7._exclusive(original/"001.intent.json",{"no_replay":False})
    assert json.loads((original/"001.intent.json").read_text())["no_replay"] is True
    link=tmp_path/"alias"
    link.symlink_to(original,target_is_directory=True)
    with pytest.raises(OSError):
        f7._fsync_dir(link)


def test_f7_source_is_not_imported_or_started_in_default_test_suite():
    body=Path(f7.__file__).read_text("utf-8")
    assert "if __name__==\"__main__\":" in body
    # Every environment operation is in owner-only collect() or its caller;
    # there is no startup on importing the module or module-global env object.
    assert "env.do(policy)" in body
    assert "def collect(" in body
    assert "def _owner(" in body
    assert "EXCLUSIVE_ROOT=" in body

def test_f7_original_source_driver_immutable_intent_then_outcome_on_fake_world(tmp_path):
    """Mock environment only: does not import, reset, or act in original source."""
    if os.name!="posix":
        pytest.skip("exact POSIX directory-fsync and write-once only")
    closed=[]
    class FakeEnv:
        def __init__(self):
            self.i=0
            self.state=[0.,0.,0.,0.]
        def reset(self,seed):
            assert seed==f7.cfg.SEED_BASE
            return self.state[:],{}
        def do(self,policy):
            action=policy(self.state[:])
            assert action in (0,1)
            self.i+=1
            self.state=[.01*self.i,0.,.001*self.i,0.]
            return self.state[:],1.0,self.i==2,False,{"action":action}
        def close(self):
            closed.append(True)
    def factory(**kwargs):
        assert kwargs["max_episode_steps"]==96
        return FakeEnv()
    owner={"run_id":"1234567","sha":"f"*40}
    import time
    world=f7._one_episode(factory,None,0,owner,tmp_path,time.monotonic()+30)
    assert len(world["transitions"])==2
    assert closed==[True]
    path=tmp_path/f"fit-000-{f7.cfg.SEED_BASE}"
    assert (path/"001.intent.json").is_file()
    assert (path/"001.outcome.json").is_file()
    assert (path/"002.intent.json").is_file()
    assert (path/"002.outcome.json").is_file()
    assert (path/"world.json").is_file()
    with pytest.raises(FileExistsError):
        f7._one_episode(factory,None,0,owner,tmp_path,time.monotonic()+30)


def test_f7_unknown_effect_is_quarantined_and_session_cannot_replay(tmp_path):
    if os.name!="posix":
        pytest.skip("POSIX only")
    import time
    class UnknownEffect:
        def reset(self,seed):
            return [0.,0.,0.,0.],{}
        def do(self,policy):
            policy([0.,0.,0.,0.])
            raise RuntimeError("UNKNOWN_EFFECT")
        def close(self):
            pass
    def factory(**kwargs):
        return UnknownEffect()
    owner={"run_id":"1234567","sha":"f"*40}
    with pytest.raises(RuntimeError,match="UNKNOWN_EFFECT"):
        f7._one_episode(factory,None,0,owner,tmp_path,time.monotonic()+30)
    path=tmp_path/f"fit-000-{f7.cfg.SEED_BASE}"
    assert (path/"001.intent.json").is_file()
    assert not (path/"001.outcome.json").exists()
    with pytest.raises(FileExistsError):
        f7._one_episode(factory,None,0,owner,tmp_path,time.monotonic()+30)
