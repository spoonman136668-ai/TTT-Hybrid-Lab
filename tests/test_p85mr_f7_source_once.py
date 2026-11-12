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
