"""P111: one-block, 16-world FIT policy selector and SSM receipt checks.

Pure source-neutral fixtures only. A fixture checkpoint is never a trained
SSM and a fixture outcome is never a real CartPoleWind action consequence.
No simulator import, model training, owner dispatch, source access or CAL.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from experiments import p109_f6_strong_control_preflight as p109
from experiments import p110_f6_five_arm_readiness as p110

FREEZE_COMMIT = "3eacee07265e93d5356f080d9e4a513a979ab7b5"
FREEZE_BLOB = "fef0bf267904de982916f25acf2c77c2582691ee"
PARENT_SHA = "72e8c56b084903327085daa88bb21cf147cdf16b"
ARMS = ("guarded_voi", "a_opt", "greedy_goal", "physical_balance",
        "ssm256_model_based")
CONVENTIONAL = ARMS[1:]
SEEDS = (985201, 985203, 985207)
HORIZON = 96
FIT_BASE = 1820000000
TRAIN_DIGEST = hashlib.sha256(b"p111-FAKE-FIT-block0-1-training-dataset").hexdigest()
ROOT = Path(__file__).resolve().parents[1]
ROW_FIELDS = frozenset((
    "split","block","index","policy","survival_steps","action_count",
    "reward_total","physical_loss","wall_ms","terminated","truncated",
    "source_session_sha256","receipt_ref",
))
MANIFEST_FIELDS = frozenset((
    "seed","model_kind","model_source_blob","checkpoint_format",
    "checkpoint_sha256","train_dataset_sha256","train_worlds",
    "train_blocks","train_updates","optimizer","optimizer_steps",
    "fit_only","cal_or_ood_used","crossfit_scope","train_crossfit_mse",
    "zero_crossfit_mse","preheldout_locked","persistent_floats",
    "confirmed_factual_effects","receipt_ref",
))
PINNED = {
    "experiments/p110_f6_five_arm_candidate_freeze.json":
        "05e7ae9486f9183a15ab2b69e2647ce59d2fcf7e",
    "experiments/p110_f6_five_arm_readiness.py":
        "5692e4f28f604d94f0bebb7a9388afcc89a61091",
    "experiments/p109_f6_strong_control_preflight.py":
        "3926bc0f0e647c5627ce627ba1ee0a02a58689e6",
    "experiments/p109_f6_strong_control_freeze.json":
        "293b022bc0d8a430a0aba44908d9d8eba4fbb36b",
    "experiments/p108_f6_confirmed_rls_fixture.py":
        "30353758463a80a9ab0568aab4cebadfe8926609",
    "experiments/p85mr_dynamics_baselines.py":
        "aa7f6c34a68ee3118f0fe9e44659aa3dec5524bd",
    "model.py":
        "5d9ea57b827aa1796548447868fa5ee14c251b49",
}


def locked_protocol():
    raw=(ROOT/"experiments/p111_f6_fit16_freeze.json").read_bytes()
    if p110._blob(raw)!=FREEZE_BLOB:
        raise ValueError("P111_FROZEN_PROTOCOL_BLOB_DRIFT")
    f=json.loads(raw)
    if (f.get("schema")!="ttt-hybrid.p111-f6-heldout16-strong-control-preflight.v1"
        or f["parent"]["exact_precode_sha"]!=PARENT_SHA
        or f["static_interface"]["FIT_selection_block"]!=2
        or f["static_interface"]["worlds_per_arm"]!=16
        or f["static_interface"]["incumbent_arms"]!=list(CONVENTIONAL)
        or f["static_interface"]["candidate_arm"]!=ARMS[0]
        or f["static_interface"]["episode_horizon"]!=HORIZON
        or f["static_interface"]["FIT_seed_base"]!=FIT_BASE
        or f["synthetic_checkpoint_guard"]["seed_list"]!=list(SEEDS)
        or f["synthetic_checkpoint_guard"]["train_worlds"]!=32
        or f["synthetic_checkpoint_guard"]["updates_per_seed"]!=800
        or f["synthetic_checkpoint_guard"]["prediction_MSE_max_ratio_to_zero"]!=0.75
        or f["synthetic_checkpoint_guard"]["preheldout_lock_required"] is not True
        or f["future_owner_gaps"]["real_F6_firstuse_approved"] is not False
        or f["future_owner_gaps"]["dual_F6_frozen_protocol_resolved"] is not False
        or f["classification"]["scientific_pass"] is not False
        or f["classification"]["real_F6_effects"]!=0):
        raise ValueError("P111_FROZEN_SELECTION_OR_AUTHORITY_DRIFT")
    for path,sha in PINNED.items():
        if p110._blob((ROOT/path).read_bytes())!=sha:
            raise ValueError("P111_PARENT_GIT_BLOB_DRIFT")
    plan=p110.locked_plan()
    if (plan["learning"]["SSM_control"]["FIT_selection_block"]!=2
        or plan["experiment"]["policy_order"]!=list(ARMS)
        or plan["owner_integrity"]["FIT_owner_approval"]!="REQUIRED_SEPARATELY"):
        raise ValueError("P111_PARENT_OWNER_SCOPE_DRIFT")
    return f


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),
                      allow_nan=False).encode("utf-8")


def _sha(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _finite(value,reason):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError(reason)
    return float(value)


def _session_id(arm,index):
    if arm not in ARMS or type(index) is not int or not 0<=index<16:
        raise ValueError("P111_UNKNOWN_FIT_POLICY_OR_WORLD")
    return _sha(["p111-test-fixture-only","FIT",FIT_BASE+2000+index,arm])


def make_fixture_episode(arm,index,steps,reward,loss,wall,
                         *,terminated=None,truncated=False):
    """A fabricated record; no model or source action is ever executed."""
    _session_id(arm,index)
    if terminated is None:
        terminated=steps<HORIZON
    row={
        "split":"FIT","block":2,"index":index,"policy":arm,
        "survival_steps":steps,"action_count":steps,
        "reward_total":reward,"physical_loss":loss,"wall_ms":wall,
        "terminated":terminated,"truncated":truncated,
        "source_session_sha256":_session_id(arm,index),
    }
    row["receipt_ref"]="fixture:"+_sha(row)
    return row


def _validate_episode(row,arm,seen):
    if not isinstance(row,dict) or set(row)!=ROW_FIELDS:
        raise ValueError("P111_UNKNOWN_EPISODE_FIELDS")
    if (row["split"]!="FIT" or row["block"]!=2 or row["policy"]!=arm):
        raise ValueError("P111_FIT_BLOCK2_ONLY")
    index=row["index"]
    if type(index) is not int or not 0<=index<16 or index in seen:
        raise ValueError("P111_DUPLICATE_OR_UNKNOWN_HELDOUT_WORLD")
    seen.add(index)
    n=row["survival_steps"]
    if (type(n) is not int or not 1<=n<=HORIZON
        or type(row["action_count"]) is not int or row["action_count"]!=n):
        raise ValueError("P111_FALSE_FACTUAL_ACTION_COUNT")
    term,trunc=row["terminated"],row["truncated"]
    if (type(term) is not bool or type(trunc) is not bool or
        (term and trunc) or (n<HORIZON and not(term or trunc))):
        raise ValueError("P111_DROPPED_EARLY_TERMINATION")
    _finite(row["reward_total"],"P111_INVALID_REALIZED_REWARD")
    if _finite(row["physical_loss"],"P111_INVALID_PHYSICAL_LOSS")<0:
        raise ValueError("P111_INVALID_PHYSICAL_LOSS")
    if _finite(row["wall_ms"],"P111_INVALID_WALL_COST")<=0:
        raise ValueError("P111_INVALID_WALL_COST")
    if row["source_session_sha256"]!=_session_id(arm,index):
        raise ValueError("P111_UNKNOWN_WORLD_SESSION")
    if row["receipt_ref"]!="fixture:"+_sha({
        key:value for key,value in row.items() if key!="receipt_ref"}):
        raise ValueError("P111_CHANGED_EPISODE_RECEIPT")


def _checkpoint_receipt(manifest):
    return "fixture:"+_sha({
        key:value for key,value in manifest.items() if key!="receipt_ref"})


def make_fixture_checkpoint(seed,training_mse,zero_mse,checkpoint_bytes):
    """Fixture helper; no claim that bytes are a trained model."""
    if seed not in SEEDS or not isinstance(checkpoint_bytes,bytes):
        raise ValueError("P111_UNKNOWN_FIXTURE_CHECKPOINT")
    m={
        "seed":seed,"model_kind":"ssm256",
        "model_source_blob":p109.SSM_BLOB,
        "checkpoint_format":"fixture-bytes-only",
        "checkpoint_sha256":hashlib.sha256(checkpoint_bytes).hexdigest(),
        "train_dataset_sha256":TRAIN_DIGEST,
        "train_worlds":32,"train_blocks":[0,1],
        "train_updates":800,"optimizer":"AdamW","optimizer_steps":800,
        "fit_only":True,"cal_or_ood_used":False,
        "crossfit_scope":"TRAIN_ONLY_CROSS_FIT",
        "train_crossfit_mse":training_mse,"zero_crossfit_mse":zero_mse,
        "preheldout_locked":True,"persistent_floats":256,
        "confirmed_factual_effects":1536,
    }
    m["receipt_ref"]=_checkpoint_receipt(m)
    return m


def _validate_checkpoint(manifest,checkpoint_bytes):
    if (not isinstance(manifest,dict) or set(manifest)!=MANIFEST_FIELDS
        or not isinstance(checkpoint_bytes,bytes)
        or not 1<=len(checkpoint_bytes)<=8*1024*1024):
        raise ValueError("P111_BAD_FIXTURE_CHECKPOINT_OR_FIELDS")
    seed=manifest["seed"]
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("P111_WRONG_TRAINED_SSM_SEED")
    if (manifest["model_kind"]!="ssm256"
        or manifest["model_source_blob"]!=p109.SSM_BLOB
        or manifest["checkpoint_format"]!="fixture-bytes-only"
        or manifest["checkpoint_sha256"]!=hashlib.sha256(checkpoint_bytes).hexdigest()
        or manifest["train_dataset_sha256"]!=TRAIN_DIGEST
        or manifest["train_blocks"]!=[0,1]
        or type(manifest["train_worlds"]) is not int
        or manifest["train_worlds"]!=32
        or type(manifest["train_updates"]) is not int
        or manifest["train_updates"]!=800
        or manifest["optimizer"]!="AdamW"
        or type(manifest["optimizer_steps"]) is not int
        or manifest["optimizer_steps"]!=800
        or manifest["fit_only"] is not True
        or manifest["cal_or_ood_used"] is not False
        or manifest["crossfit_scope"]!="TRAIN_ONLY_CROSS_FIT"
        or manifest["preheldout_locked"] is not True
        or type(manifest["persistent_floats"]) is not int
        or manifest["persistent_floats"]!=256
        or type(manifest["confirmed_factual_effects"]) is not int
        or not 32<=manifest["confirmed_factual_effects"]<=3072):
        raise ValueError("P111_UNTRAINED_OR_UNFROZEN_SSM")
    mse=_finite(manifest["train_crossfit_mse"],"P111_BAD_PREDICTOR_COMPETENCE")
    zero=_finite(manifest["zero_crossfit_mse"],"P111_BAD_PREDICTOR_COMPETENCE")
    if mse<0 or zero<=0:
        raise ValueError("P111_BAD_PREDICTOR_COMPETENCE")
    if manifest["receipt_ref"]!=_checkpoint_receipt(manifest):
        raise ValueError("P111_CHANGED_SSM_RECEIPT")
    return mse/zero


def check_fit_block2(episodes,ssm_checkpoints):
    """Dev-facing static fixture selector, NOT an F6 source or CAL gate."""
    locked_protocol()
    if not isinstance(episodes,dict) or set(episodes)!=set(ARMS):
        raise ValueError("P111_FIVE_ARMS_REQUIRED_IN_FIT")
    if (not isinstance(ssm_checkpoints,dict)
        or set(ssm_checkpoints)!=set(SEEDS)):
        raise ValueError("P111_THREE_TRAIN_SEED_RECEIPTS_REQUIRED")
    audited={}
    ratios={}
    for seed in SEEDS:
        item=ssm_checkpoints[seed]
        if not isinstance(item,dict) or set(item)!={"manifest","bytes"}:
            raise ValueError("P111_INVALID_CHECKPOINT_ENTRY")
        m=item["manifest"]
        if not isinstance(m,dict) or m.get("seed")!=seed:
            raise ValueError("P111_WRONG_TRAINED_SSM_SEED")
        ratios[seed]=_validate_checkpoint(m,item["bytes"])
    frozen_seed=min(SEEDS,key=lambda seed:(
        float(ssm_checkpoints[seed]["manifest"]["train_crossfit_mse"]),seed))
    ssm_numerically_competent=ratios[frozen_seed]<=0.75
    for arm in ARMS:
        rows=episodes[arm]
        if not isinstance(rows,list) or len(rows)!=16:
            raise ValueError("P111_REQUIRE_EXACTLY_16_HELDOUT_FIT_WORLDS")
        seen=set()
        for row in rows:
            _validate_episode(row,arm,seen)
        if seen!=set(range(16)):
            raise ValueError("P111_INCOMPLETE_FIT_BLOCK2")
        audited[arm]={
            "mean_survival":sum(row["survival_steps"] for row in rows)/16.,
            "mean_physical_loss":sum(row["physical_loss"] for row in rows)/16.,
            "mean_realized_reward":sum(row["reward_total"] for row in rows)/16.,
            "complete_factual_actions":sum(row["action_count"] for row in rows),
            "total_wall_ms":sum(row["wall_ms"] for row in rows),
            "fixture_receipts_sha256":_sha([row["receipt_ref"] for row in
                                           sorted(rows,key=lambda row:row["index"])]),
        }
    winner=min(CONVENTIONAL,key=lambda arm:(
        -audited[arm]["mean_survival"],
        audited[arm]["mean_physical_loss"],CONVENTIONAL.index(arm)))
    threshold_met=audited[winner]["mean_survival"]>=24
    return {
        "schema":"ttt-hybrid.p111-fit16-fixture-only-audit.v1",
        "classification":"SOURCE_NEUTRAL_ENGINEERING_ONLY",
        "selected_conventional_fixture":winner,
        "selected_ssm_seed_fixture_only":frozen_seed,
        "ssm_train_only_crossfit_mse_ratios_fixture":ratios,
        "ssm_fixture_predictor_competent":ssm_numerically_competent,
        "strongest_conventional_survival_fixture":audited[winner]["mean_survival"],
        "strongest_conventional_minimum_met_fixture":threshold_met,
        "can_claim_fixture_eligibility":bool(threshold_met and ssm_numerically_competent),
        "per_arm_results":audited,
        "equal_FIT_block2_episode_opportunities":True,
        "equal_actual_factual_effects":len(set(
            audited[arm]["complete_factual_actions"] for arm in ARMS))==1,
        "actual_source_attempts":0,"trained_real_ssm":False,
        "authentic_checkpoint_proven":False,"F6_owner_approved":False,
        "P110_owner_protocol_revision_required":True,
        "real_CAL_admitted":False,"scientific_pass":False,"deployment":False,
    }


def refuse_owner_cal_or_source(*_args,**_kwargs):
    raise PermissionError("P111_NO_REAL_F6_OR_CAL_FIRSTUSE_AUTHORITY")
