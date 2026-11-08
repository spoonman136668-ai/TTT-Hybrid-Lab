"""Tests of actual P0 factual adaptation + PR53 checkpoint, DEV-only."""
import ast
import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts.native_p3_continuity import (
    CHECKPOINT_AT, MODEL_INITIALIZATIONS, SCHEMA, TOTAL_EVENTS,
    identity, new_model_and_optimizer, restore_completed, run,
    save_completed, transact)
from scripts.native_prequential_gate import NativeCausalEpisode
from native_training_checkpoint import CheckpointRejected


def test_complete_native_training_and_feedback_cold_resume_exact():
    result=run()
    assert result["status"]=="PASS_COLD_RESUME_INTEGRATION_ENGINEERING_ONLY"
    assert result["science_first_use"]=="NOT_RUN"
    assert result["scientific_seeds_consumed"]==[]
    assert result["record_count"]==len(MODEL_INITIALIZATIONS)==3
    assert result["scientific_worlds_constructed"]==0
    for r in result["records"]:
        assert r["status"]=="PASS_EXACT_CONTINUATION_ENGINEERING_ONLY"
        assert all(r["checks"].values())
        assert r["checkpoint_at_completed_step"]==40
        assert r["final_consumed_factual_examples"]==80
        assert r["post_switch_uninterrupted_mse"]==r["post_switch_resumed_mse"]


def test_pending_factual_ticket_blocks_checkpoint(tmp_path):
    model,optimizer=new_model_and_optimizer(MODEL_INITIALIZATIONS[0])
    ep=NativeCausalEpisode(model)
    sensor=torch.zeros((1,1));action=torch.ones((1,1));dt=torch.tensor([[.2]])
    ep.predict(sensor,action,dt)
    with pytest.raises(CheckpointRejected,match="PENDING_CAUSAL"):
        save_completed(tmp_path,identity(),model,optimizer,ep,1,None)


def test_resume_refuses_reset_and_wrong_identity(tmp_path):
    model,opt=new_model_and_optimizer(MODEL_INITIALIZATIONS[0])
    ep=NativeCausalEpisode(model)
    ident=identity()
    root=save_completed(tmp_path,ident,model,opt,ep,0,None)
    transact(model,opt,ep,0)
    with pytest.raises(CheckpointRejected,match="UNDECLARED_RESTART"):
        save_completed(tmp_path,ident,model,opt,ep,1,None)
    wrong=dict(ident);wrong["protocol_sha256"]="f"*64
    with pytest.raises(CheckpointRejected,match="IDENTITY_OR_CURSOR"):
        restore_completed(tmp_path,root,wrong,model,opt)


def test_science_fence_and_lineage_no_source_imports(monkeypatch):
    monkeypatch.setenv("NATIVE_P3_ENGINEERING_ONLY","0")
    with pytest.raises(RuntimeError,match="P3_SCIENCE_FIRST_USE_FORBIDDEN"):
        run()
    source=(Path(__file__).resolve().parents[1]/
            "scripts/native_p3_continuity.py").read_text()
    imports=[n.module or "" for n in ast.walk(ast.parse(source))
             if isinstance(n,ast.ImportFrom)]
    assert not any(name in module for module in imports for name in
                   ("g7_s20","g7_s25","g7_s26","g7_s27"))


def test_frozen_engineering_manifest_references_both_lineages():
    path=Path(__file__).resolve().parents[1]/"experiments/native_p3_continuity_engineering_frozen.json"
    p=json.loads(path.read_text())
    assert p["schema"]=="ttt-hybrid.native-p3.continuity-engineering.v1"
    assert p["parents"]["checkpoint_pr53_sha"]=="c42ad43d6451ada13159ef81e5978de55c7e8b4b"
    assert p["parents"]["fast_memory_pr52_sha"]=="2fb2be9e23361baee915a0450ce79b7d58e7772c"
    assert p["science_first_use"]["authorized"] is False
    assert p["science_first_use"]["seed_assignments"]=={}
    assert p["training"]["analytic_dev_optimizer_steps"]==TOTAL_EVENTS
    assert p["training"]["cold_resume_after_step"]==CHECKPOINT_AT
