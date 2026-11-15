"""R78 full-width FIT 2048->4096 exact-custody engineering, never DEV/OOD."""
import json
from pathlib import Path

import pytest
import torch

from experiments import r74_capability_ceiling as r74
from experiments import r75_cumulative_training as r75
from experiments import r77_verified_resume as r77
from experiments import r78_4096_continuation as r78


def test_original_prospective_r78_freeze_before_execution():
    f=r78.frozen()
    assert r78.git_blob(r78.FREEZE.read_bytes())==r78.FREEZE_BLOB
    assert f["schema"]==r78.SCHEMA
    assert f["parent"]["R77_exact_SHA"]==r78.R77_PARENT_SHA
    assert f["parent"]["R77_artifact_bytes"]==186542465
    assert f["parent"]["R77_artifact_digest"].startswith("sha256:")
    assert f["source_checkpoint"]["checkpoint_original_identity_sha"]==r78.ORIGINAL_IDENTITY_SHA
    assert f["learning"]["stages"]==[
        {"id":1,"start":2048,"end":3072},
        {"id":2,"start":3072,"end":4096}]
    assert f["execution"]["max_wall_minutes_each"]==29
    assert f["execution"]["after_both_G4_windows_ubuntu_pass"]
    assert f["qualification"]["original_R74_science_full_target"].startswith("7 architectures")
    assert not any(f["authority"].values())


def test_execution_unavailable_outside_first_same_repo_pr_runner(monkeypatch):
    for name in ("GITHUB_ACTIONS","GITHUB_EVENT_NAME","GITHUB_HEAD_REF",
                 "GITHUB_REF","GITHUB_REPOSITORY","GITHUB_RUN_ATTEMPT",
                 "RUNNER_ENVIRONMENT","RUNNER_OS","R78_STAGE",
                 "EXPECTED_CANDIDATE_SHA","GITHUB_RUN_ID"):
        monkeypatch.delenv(name,raising=False)
    for stage in (1,2):
        with pytest.raises(PermissionError,match="R78_ONLY_FIRST_SAME_REPO_HOSTED_PR_CPU_JOB"):
            r78.run_identity(stage)
    with pytest.raises(ValueError,match="R78_INVALID_STAGE"):
        r78.run_identity(3)
    with pytest.raises(ValueError,match="R78_UNKNOWN_ARM_OR_STAGE"):
        r78.checkpoint_name("OFFICIAL_GPU_UNQUALIFIED",4096,2)


def test_missing_original_artifact_fail_closed_before_FIT(monkeypatch,tmp_path):
    monkeypatch.setattr(r78,"API",{1:tmp_path/"absent.json",2:tmp_path/"absent2.json"})
    monkeypatch.setattr(r78,"SRC",{1:tmp_path/"r77",2:tmp_path/"s1"})
    monkeypatch.setattr(r78,"run_identity",lambda stage:{"candidate_SHA":"a"*40,"run_id":56789,"stage":stage})
    with pytest.raises(FileNotFoundError,match="R78_OFFICIAL_GITHUB_SOURCE_METADATA_NOT_VERIFIED"):
        r78.preflight(1)
    with pytest.raises(FileNotFoundError,match="R78_OFFICIAL_GITHUB_SOURCE_METADATA_NOT_VERIFIED"):
        r78.preflight(2)
    (tmp_path/"absent.json").write_text(json.dumps({
        "candidate_SHA":"a"*40,"run_id":56789,"stage":1,"not_expired":True,
        "original_artifact_id":r78.frozen()["parent"]["R77_private_artifact_id"],
        "artifact_digest":r78.frozen()["parent"]["R77_artifact_digest"]
    }))
    with pytest.raises(FileNotFoundError,match="R78_R77_RECEIPT_MISSING"):
        r78.preflight(1)


def test_cold_identity_preserves_original_R76_and_R77_lineage():
    for arm in r78.ARMS:
        assert r78.original_identity(arm)==r77.old_identity(arm)
        assert r78.original_identity(arm)["checkout_commit_sha"]==r78.ORIGINAL_IDENTITY_SHA
        assert r78.original_identity(arm)["engine"]==r75.ENGINE
        assert r78.checkpoint_name(arm,3072,1).endswith("-u3072.pt")
        assert r78.checkpoint_name(arm,4096,2).endswith("-u4096.pt")
    assert r78.STAGES[1][0]==2048
    assert r78.STAGES[1][1]==r78.STAGES[2][0]
    assert r78.STAGES[2][1] in r74.TRAIN_STEPS
    assert r78.STAGES[1][1] not in r74.TRAIN_STEPS


def test_no_protected_source_access_and_nonleaking_current_selected_outcome():
    fit=r74.factual_worlds("FIT",count_override=2,sequence=32)
    assert fit["input"].shape==(2,32,13)
    assert not torch.from_numpy(fit["input"]).data_ptr()==torch.from_numpy(fit["selected_reward"]).data_ptr()
    assert r74.factual_causality_negative("FIT",sequence=32)
    for panel in ("R71_HOLDOUT","R73_HOLDOUT","R77_DEV","EXTERNAL_PROTECTED"):
        with pytest.raises(PermissionError,match="R74_PREVIOUS_OR_PROTECTED_SCIENCE_SOURCE_DENIED"):
            r74.factual_worlds(panel,count_override=1,sequence=32)


def test_source_artifact_metadata_fail_closed_without_original_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN",raising=False)
    monkeypatch.setattr(r78,"run_identity",lambda stage:{"candidate_SHA":"a"*40,"run_id":56789,"stage":stage})
    with pytest.raises(PermissionError,match="R78_ORIGINAL_READ_ONLY_GITHUB_TOKEN_MISSING"):
        r78.verified_source_artifact(1)
    with pytest.raises(PermissionError,match="R78_ORIGINAL_READ_ONLY_GITHUB_TOKEN_MISSING"):
        r78.verified_source_artifact(2)
