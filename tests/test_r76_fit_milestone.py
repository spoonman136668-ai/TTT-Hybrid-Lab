"""R76 first max-width 1024 FIT-only checkpoint custody and scope tests.

Full long training is never run inside pytest; it is a separately bounded
single original PR G4 Ubuntu job, protecting 600s mandatory regression SLA.
"""
import json
import os

import pytest

from experiments import r74_capability_ceiling as r74
from experiments import r75_cumulative_training as r75training
from experiments import r76_fit_milestone as r76


def test_original_single_freeze_before_first_training():
    f=r76.frozen()
    assert r76.git_blob(r76.FREEZE.read_bytes())==r76.FREEZE_BLOB
    assert f["schema"]==r76.SCHEMA
    assert f["parent"]["R75_exact_SHA"]==r76.PARENT
    assert f["parent"]["prior_scored_R64_to_R73_unused"]
    assert f["training"]["architecture"]==list(r76.ARMS)
    assert f["training"]["optim_updates"]==1024
    assert f["training"]["selected_real_FIT_events_per_arm"]==262144
    assert f["training"]["width"]==1024
    assert f["training"]["sequence"]==128
    assert f["execution"]["timeout_minutes"]==29
    assert f["predeclared_interpretation"]["science"]=="UNVERIFIED; full R74/R75 target remains both seeds, all seven arms, 65536 updates each before DEV/OOD scoring."
    assert not any(f["authority"].values())
    assert r76.git_blob(b"factual\n")==r76.git_blob(b"factual\r\n")
    with pytest.raises(ValueError,match="R76_UNSUPPORTED_TEXT_CARRIAGE_RETURN"):
        r76.git_blob(b"invalid\roriginal")


def test_training_route_rejects_local_nonowner_and_wrong_runner(monkeypatch):
    for key in ("GITHUB_ACTIONS","GITHUB_EVENT_NAME","GITHUB_HEAD_REF",
                "RUNNER_OS","RUNNER_ENVIRONMENT","GITHUB_RUN_ATTEMPT",
                "GITHUB_REPOSITORY","EXPECTED_CANDIDATE_SHA","GITHUB_REF"):
        monkeypatch.delenv(key,raising=False)
    with pytest.raises(PermissionError,match="R76_ONLY_FIRST_ORIGINAL_HOSTED_UBUNTU_PR_RUN_ALLOWED"):
        r76.admit_cpu_execution()
    monkeypatch.setenv("GITHUB_ACTIONS","true")
    monkeypatch.setenv("GITHUB_EVENT_NAME","pull_request")
    monkeypatch.setenv("GITHUB_HEAD_REF",r76.BRANCH)
    monkeypatch.setenv("RUNNER_OS","Windows")
    monkeypatch.setenv("RUNNER_ENVIRONMENT","github-hosted")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT","1")
    monkeypatch.setenv("GITHUB_REPOSITORY","spoonman136668-ai/TTT-Hybrid-Lab")
    with pytest.raises(PermissionError,match="R76_ONLY_FIRST_ORIGINAL_HOSTED_UBUNTU_PR_RUN_ALLOWED"):
        r76.admit_cpu_execution()
    monkeypatch.setenv("RUNNER_OS","Linux")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT","2")
    with pytest.raises(PermissionError,match="R76_ONLY_FIRST_ORIGINAL_HOSTED_UBUNTU_PR_RUN_ALLOWED"):
        r76.admit_cpu_execution()


def test_protected_heldout_denied_and_original_resume_identity(tmp_path):
    for panel in ("R72_HOLDOUT","R73_HOLDOUT","OLD_DEV","UNPROTECTED_OOD"):
        with pytest.raises(PermissionError,match="R74_PREVIOUS_OR_PROTECTED_SCIENCE_SOURCE_DENIED"):
            r74.factual_worlds(panel,count_override=2,sequence=32)
    n=r75training.checkpoint_path("HybridPilot_FULL",1024,128,r76.SEED)
    assert n.name.startswith("R75-HybridPilot_FULL-width1024-seq128-")
    assert n.suffix==".pt"
    assert r75training.FROZEN_CHECKPOINTS[0]==r76.UPDATES
    assert r76.ACTUAL_SELECTED_FIT_EVENTS==262144
    with pytest.raises(FileNotFoundError,match="R75_INCOMPLETE_65536_TRAINING"):
        r75training.require_all_control_checkpoints_before_holdout(tmp_path,1024,128)


def test_receipt_will_not_pretend_to_prove_unfrozen_science():
    report=json.loads(r76.FREEZE.read_text())
    assert report["training"]["post_training"].startswith("No DEV/OOD")
    assert report["predeclared_interpretation"]["RESOURCE_LIMIT"].startswith("INCOMPLETE_RESOURCE_OR_RUNNER_LIMIT")
    assert report["execution"]["no_paid_API"]
    assert not report["authority"]["external_science_firstuse"]
    assert not report["authority"]["GPU_firstuse"]
