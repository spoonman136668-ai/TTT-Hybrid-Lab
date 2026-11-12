"""P110 frozen F6 five-policy prospective plan: engineering only, no source birth."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments import p110_f6_five_arm_readiness as p110


def plan():
    return copy.deepcopy(p110.locked_plan())


def test_p110_frozen_protocol_and_lineage_before_implementation():
    p=plan()
    assert p110.FREEZE_COMMIT=="664ef9dbe7183c267756612a4df642be33f31e30"
    assert p110.FREEZE_BLOB=="05e7ae9486f9183a15ab2b69e2647ce59d2fcf7e"
    assert p["stage"]=="FROZEN_PRE_EFFECT_PROPOSAL_NOT_OWNER_AUTHORITY"
    assert p["parent"]["pr"]==233
    assert p["parent"]["source_head"]=="79623cf0f8c5696bdabea9998194f59bd3890b85"
    assert p["parent"]["freeze_P106B_blob"]=="42db0228c1c4561333058a3b9d849b8c3db1183c"
    assert p["governance"]["status"]=="RECOMMENDATION_ONLY_PENDING_OWNER_SOURCE_APPROVAL"
    assert p["governance"]["alternative_B_retained_unmodified"]==(
        "research/p106-f6-binary-policy-preflight-20261111")
    assert p["upstream"]["source_commit"]==p110.BOUND_UPSTREAM
    assert p["classification"]["authentic_external_source"]=="NOT_RUN"


def test_p110_all_five_arms_and_prospective_effect_budget_true():
    result=p110.source_free_readiness_report()
    assert result["policies"]==5 and result["conventional"]==4
    assert result["FIT_world_seeds"]==48
    assert result["CAL_world_seeds"]==48
    assert result["FIT_sessions"]==240 and result["CAL_sessions"]==240
    assert result["maximum_FIT_factual_actions"]==23040
    assert result["maximum_full_trial_factual_actions"]==46080
    assert result["FIT_training_worlds_for_SSM"]==32
    assert result["FIT_heldout_control_selection_worlds"]==16
    assert result["SSM_training_seeds"]==3
    assert result["protected_worlds_generated"]==0
    assert result["source_effect_attempts"]==0
    assert result["trained_actual_SSM_control_qualified"] is False
    assert result["FIT_firstuse_approved"] is False
    assert result["CAL_firstuse_approved"] is False
    assert result["scientific_pass"] is False


def test_p110_frozen_candidate_never_silently_excludes_strong_control():
    p=plan()
    assert p["experiment"]["policy_order"]==list(p110.ARMS)
    assert p["experiment"]["conventional"]==list(p110.CONVENTIONAL)
    assert "physical_balance" in p110.CONVENTIONAL
    assert "ssm256_model_based" in p110.CONVENTIONAL
    assert p["experiment"]["fixed_epsilon"]==.15
    assert p["experiment"]["tuning_epsilon_on_spent_CAL_or_external_FIT"] is False
    assert p["learning"]["SSM_control"]["FIT_train_blocks"]==[0,1]
    assert p["learning"]["SSM_control"]["FIT_selection_block"]==2
    assert p["learning"]["SSM_control"]["updates_per_seed"]==800
    assert p["learning"]["future_CAL"].startswith("Only if distinct owner authorization")
    p["experiment"]["policy_order"].remove("physical_balance")
    with pytest.raises(ValueError,match="P110_POLICY_OR_SOURCE_SPLIT_DRIFT"):
        p110.validate_prospective_plan(p)
    p=plan()
    p["experiment"]["conventional"].remove("ssm256_model_based")
    with pytest.raises(ValueError,match="P110_POLICY_OR_SOURCE_SPLIT_DRIFT"):
        p110.validate_prospective_plan(p)


def test_p110_authority_and_precise_first_use_prohibitions():
    for section,key,value,reason in (
        ("governance","FIT_firstuse_approved",True,"P110_OWNER_NOT_AUTHORIZED_OR_FORK_DRIFT"),
        ("governance","CAL_firstuse_approved",True,"P110_OWNER_NOT_AUTHORIZED_OR_FORK_DRIFT"),
        ("governance","no_source_effects_granted",False,"P110_OWNER_NOT_AUTHORIZED_OR_FORK_DRIFT"),
        ("governance","owner_SSM_trained",True,"P110_OWNER_NOT_AUTHORIZED_OR_FORK_DRIFT"),
        ("governance","status","APPROVED","P110_OWNER_NOT_AUTHORIZED_OR_FORK_DRIFT"),
        ("acceptance","scientific_pass",True,"P110_OWNER_OR_SCIENTIFIC_BOUNDARY_VIOLATION"),
        ("classification","authentic_external_source","RUN","P110_OWNER_OR_SCIENTIFIC_BOUNDARY_VIOLATION"),
        ("owner_integrity","owner_native_preflight","PASS","P110_OWNER_OR_SCIENTIFIC_BOUNDARY_VIOLATION"),
    ):
        p=plan()
        p[section][key]=value
        with pytest.raises(PermissionError,match=reason):
            p110.validate_prospective_plan(p)
    with pytest.raises(PermissionError,match="P110_EXTERNAL_F6_FIRSTUSE_NOT_AUTHORIZED"):
        p110.deny_external_firstuse("FIT",runner="TTT-HYBRID-GPU-WIN")


def test_p110_no_mutable_data_birth_or_budget_escapes():
    tests=[
        ("split_seeds","FIT_base",1811000000,"P110_NEW_SOURCE_SEED_DRIFT"),
        ("split_seeds","OOD_RESERVED_base",1820100000,"P110_NEW_SOURCE_SEED_DRIFT"),
        ("budget","full_trial_max_sessions",384,"P110_SESSION_OR_EFFECT_BUDGET_DRIFT"),
        ("budget","source_max_actions_per_split",36864,"P110_SESSION_OR_EFFECT_BUDGET_DRIFT"),
        ("budget","first_use_FIT_only_max_sessions",480,"P110_SESSION_OR_EFFECT_BUDGET_DRIFT"),
        ("ssm","FIT_selection_block",1,"P110_SSM_TRAINING_SPLIT_DRIFT"),
        ("ssm","updates_per_seed",0,"P110_SSM_TRAINING_SPLIT_DRIFT"),
        ("ssm","FIT_train_blocks",[0,1,2],"P110_SSM_TRAINING_SPLIT_DRIFT"),
    ]
    for kind,key,val,reason in tests:
        p=plan()
        if kind=="split_seeds":
            p["experiment"]["split_seeds"][key]=val
        elif kind=="budget":
            p["experiment"][key]=val
        elif kind=="ssm":
            p["learning"]["SSM_control"][key]=val
        with pytest.raises(ValueError,match=reason):
            p110.validate_prospective_plan(p)


def test_p110_no_runtime_or_external_source_capability():
    code=Path(p110.__file__).read_text(encoding="utf-8")
    for fragment in (
        "import causal_gym","from causal_gym","import subprocess",
        "import requests","import socket","import urllib",
        "torch.load(","env.reset(","env.do(","run_workflow(",
        "dispatch_runner(","exec(","os.system(",
    ):
        assert fragment not in code
    result=p110.source_free_readiness_report()
    assert result["qualification_class"]=="PROTOCOL_CONSISTENCY_ONLY_NO_OWNER_FIRSTUSE"
    assert result["owner_native_receipts_qualified"] is False
    assert result["protocol_selection_authorized"] is False
    assert result["deployment"] is False
