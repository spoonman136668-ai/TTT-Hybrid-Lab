"""P110: inspect a prospective five-arm F6 release candidate; NEVER run F6.

This is a source-neutral, fail-closed review of previously frozen provenance,
budget, policies, train/selection/calibration split and owner authorization.
It has NO source simulator, checkout, runner dispatch or effect entry point.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

FREEZE_COMMIT = "664ef9dbe7183c267756612a4df642be33f31e30"
FREEZE_BLOB = "05e7ae9486f9183a15ab2b69e2647ce59d2fcf7e"
PARENT_SHA = "79623cf0f8c5696bdabea9998194f59bd3890b85"
ROOT = Path(__file__).resolve().parents[1]
ARMS = ("guarded_voi","a_opt","greedy_goal","physical_balance",
        "ssm256_model_based")
CONVENTIONAL = ARMS[1:]
SSM_SEEDS = (985201,985203,985207)
BOUND_UPSTREAM = "c94e9a4cd3641588b6b8507e71ccee5c27032bdf"
SOURCE_BLOBS = {
    "experiments/p106_f6_source_neutral_freeze.json":
        "3a2cee9454edf2e2f9937e35492d618e30f79de0",
    "experiments/p109_f6_strong_control_freeze.json":
        "293b022bc0d8a430a0aba44908d9d8eba4fbb36b",
    "experiments/p108_f6_confirmed_rls_fixture.py":
        "30353758463a80a9ab0568aab4cebadfe8926609",
    "model.py":"5d9ea57b827aa1796548447868fa5ee14c251b49",
}


def _blob(content: bytes) -> str:
    return hashlib.sha1(b"blob "+str(len(content)).encode("ascii")+b"\0"+content).hexdigest()


def validate_prospective_plan(f: dict) -> dict:
    """Validate internally inconsistent five-arm scopes; no override token.

    This is a review of fixed, non-authorized prospective data. It does
    not grant any rights to construct an environment or to select a fork.
    """
    if not isinstance(f,dict) or f.get("schema") != (
        "ttt-hybrid.p110-f6-five-arm-prospective-release-candidate.v1"):
        raise ValueError("P110_SCHEMA_DRIFT")
    if (f.get("stage")!="FROZEN_PRE_EFFECT_PROPOSAL_NOT_OWNER_AUTHORITY"
        or f["parent"]["source_head"]!=PARENT_SHA
        or f["parent"]["freeze_P106A_blob"] != SOURCE_BLOBS[
            "experiments/p106_f6_source_neutral_freeze.json"]
        or f["parent"]["freeze_P109_blob"] != SOURCE_BLOBS[
            "experiments/p109_f6_strong_control_freeze.json"]):
        raise ValueError("P110_PARENT_OR_FROZEN_PROVENANCE_DRIFT")
    gov=f["governance"]
    if (gov["recommended_design"]!="P106_A+SSM_5_ARM_ONLY"
        or gov["status"]!="RECOMMENDATION_ONLY_PENDING_OWNER_SOURCE_APPROVAL"
        or gov["alternative_B_retained_unmodified"]!=
            "research/p106-f6-binary-policy-preflight-20261111"
        or gov["owner_SSM_trained"] is not False
        or gov["FIT_firstuse_approved"] is not False
        or gov["CAL_firstuse_approved"] is not False
        or gov["OOD_firstuse_approved"] is not False
        or gov["no_source_effects_granted"] is not True
        or gov["no_owner_or_controller_dispatch"] is not True
        or gov["no_new_workflows"] is not True):
        raise PermissionError("P110_OWNER_NOT_AUTHORIZED_OR_FORK_DRIFT")
    if f["upstream"]["source_commit"]!=BOUND_UPSTREAM:
        raise ValueError("P110_UPSTREAM_SHA_DRIFT")
    p=f["experiment"]
    if (p["policy_order"]!=list(ARMS) or
        p["conventional"]!=list(CONVENTIONAL) or
        p["candidate"]!=ARMS[0] or
        p["fixed_epsilon"]!=0.15 or
        p["tuning_epsilon_on_spent_CAL_or_external_FIT"] is not False or
        p["episode_steps_max"]!=96 or
        p["blocks_per_split"]!=3 or
        p["worlds_per_block"]!=16 or
        p["splits_planned"]!=["FIT","CAL"] or
        p["real_world_splits_not_generated"] is not True or
        p["protected_OOD_and_NEGATIVE_no_generation"] is not True):
        raise ValueError("P110_POLICY_OR_SOURCE_SPLIT_DRIFT")
    base=p["split_seeds"]
    if base != {"FIT_base":1820000000,
                "CAL_base":1820100000,
                "OOD_RESERVED_base":1820200000,
                "NEGATIVE_RESERVED_base":1820300000}:
        raise ValueError("P110_NEW_SOURCE_SEED_DRIFT")
    if not p["episode_opportunities_same_across_policies"]:
        raise ValueError("P110_UNEQUAL_EPISODE_OPPORTUNITIES")
    fit={base["FIT_base"]+1000*b+w
         for b in range(3) for w in range(16)}
    cal={base["CAL_base"]+1000*b+w
         for b in range(3) for w in range(16)}
    reserved={base[tag]+1000*b+w
              for tag in ("OOD_RESERVED_base","NEGATIVE_RESERVED_base")
              for b in range(3) for w in range(16)}
    # Integer range checking only: NO external data/source generation.
    if (len(fit)!=48 or len(cal)!=48 or len(reserved)!=96
        or fit&cal or fit&reserved or cal&reserved
        or min(fit|cal)<1820000000 or max(fit|cal)>=1820200000):
        raise ValueError("P110_SEED_COLLISION")
    per_split=len(ARMS)*48
    effect_cap=per_split*p["episode_steps_max"]
    if (p["policy_sessions_per_split"]!=per_split or
        p["source_max_actions_per_split"]!=effect_cap or
        p["first_use_FIT_only_max_sessions"]!=per_split or
        p["first_use_FIT_only_max_effects"]!=effect_cap or
        p["full_trial_max_sessions"]!=2*per_split or
        p["full_trial_max_effect_actions"]!=2*effect_cap):
        raise ValueError("P110_SESSION_OR_EFFECT_BUDGET_DRIFT")
    m=f["learning"]["SSM_control"]
    if (m["seed_list"]!=list(SSM_SEEDS)
        or m["updates_per_seed"]!=800
        or m["FIT_train_blocks"]!=[0,1]
        or m["FIT_selection_block"]!=2
        or m["collect_train_action_policy"]!=(
            "physical_balance for SSM arm on FIT blocks 0-1; those 32 sessions "
            "are already counted in SSM arm; choose exactly one observed action "
            "each step, never extra outcomes")
        or f["learning"]["future_CAL"].startswith("Only if distinct owner authorization") is not True):
        raise ValueError("P110_SSM_TRAINING_SPLIT_DRIFT")
    a=f["acceptance"]
    if (a.get("scientific_pass",False) is not False or
        a["status"]!="PROSPECTIVE_NOT_EXECUTED" or
        f["owner_integrity"]["FIT_owner_approval"]!="REQUIRED_SEPARATELY"
        or f["owner_integrity"]["CAL_owner_approval"]!="REQUIRED_SEPARATELY_AFTER_FIT"
        or f["owner_integrity"]["native_owner_preflight"]!="NOT_VERIFIED"
        or f["classification"]["authentic_external_source"]!="NOT_RUN"
        or f["classification"]["trained_real_SSM"]!="NOT_RUN"
        or f["classification"]["scientific_pass"] is not False):
        raise PermissionError("P110_OWNER_OR_SCIENTIFIC_BOUNDARY_VIOLATION")
    return {
        "recommendation_only":True,
        "recommended_design":gov["recommended_design"],
        "policies":len(ARMS),"conventional":len(CONVENTIONAL),
        "FIT_world_seeds":len(fit),"CAL_world_seeds":len(cal),
        "FIT_sessions":per_split,"CAL_sessions":per_split,
        "maximum_FIT_factual_actions":effect_cap,
        "maximum_full_trial_factual_actions":2*effect_cap,
        "SSM_training_seeds":len(SSM_SEEDS),
        "FIT_training_worlds_for_SSM":32,
        "FIT_heldout_control_selection_worlds":16,
        "protected_worlds_generated":0,
        "actual_source_sessions":0,
        "source_effect_attempts":0,
        "actual_training_performed":False,
        "FIT_firstuse_approved":False,
        "CAL_firstuse_approved":False,
        "scientific_pass":False,
    }


def locked_plan() -> dict:
    raw=(ROOT/"experiments/p110_f6_five_arm_candidate_freeze.json").read_bytes()
    if _blob(raw)!=FREEZE_BLOB:
        raise ValueError("P110_FREEZE_BLOB_CHANGED")
    for path,digest in SOURCE_BLOBS.items():
        if _blob((ROOT/path).read_bytes())!=digest:
            raise ValueError("P110_PROVENANCE_BLOB_DRIFT")
    plan=json.loads(raw)
    validate_prospective_plan(plan)
    return plan


def source_free_readiness_report() -> dict:
    """Only a frozen recommendation receipt, not an owner-run manifest."""
    p=locked_plan()
    result=validate_prospective_plan(p)
    result.update({
        "schema":"ttt-hybrid.p110-f6-five-arm-readiness-report.v1",
        "freeze_commit":FREEZE_COMMIT,
        "freeze_blob":FREEZE_BLOB,
        "source_head":PARENT_SHA,
        "qualification_class":"PROTOCOL_CONSISTENCY_ONLY_NO_OWNER_FIRSTUSE",
        "owner_native_receipts_qualified":False,
        "trained_actual_SSM_control_qualified":False,
        "protocol_selection_authorized":False,
        "deployment":False,
    })
    return result


def deny_external_firstuse(*_args,**_kwargs):
    raise PermissionError("P110_EXTERNAL_F6_FIRSTUSE_NOT_AUTHORIZED")
