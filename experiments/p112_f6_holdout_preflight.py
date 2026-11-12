"""P112 source-neutral F6 training/holdout receipt disjointness and scoring.

Engineering fixture only. This module never trains a model or imports
CartPoleWind; the 24/8 holdout is MODEL SELECTION, not an unbiased test
of the selected SSM, and cannot authorize owner F6 first-use.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from experiments import p110_f6_five_arm_readiness as p110
from experiments import p111_f6_fit16_preflight as p111

FREEZE_COMMIT="27e68423f1807574d0370e4eed9d9fb3341a3f4b"
FREEZE_BLOB="76f46135a8f134e1d638615aafad2d1d8f5805c3"
SOURCE_PARENT="85f9dcdad0ad78dfb18a807bef64ebebef1ccbdc"
SEEDS=(985201,985203,985207)
D=4
HORIZON=96
TRAIN_IDS=tuple((0,i) for i in range(16))+tuple((1,i) for i in range(8))
VAL_IDS=tuple((1,i) for i in range(8,16))
TRAIN_FIELDS=frozenset(("split","block","index","policy","action_count",
                        "terminated","truncated","receipt_ref"))
VAL_FIELDS=frozenset(("split","block","index","policy","executed_action",
                      "state_before","state_after","receipt_ref"))
MANIFEST_FIELDS=frozenset(("seed","model_kind","model_source_blob",
    "checkpoint_format","checkpoint_sha256","train_receipts_sha256",
    "train_world_ids","validation_world_ids","optimizer","optimizer_steps",
    "fit_only","cal_or_ood_used","validation_predictions_sha256",
    "validation_wall_ms","receipt_ref"))
PINNED={
    "experiments/p110_f6_five_arm_candidate_freeze.json":
        "05e7ae9486f9183a15ab2b69e2647ce59d2fcf7e",
    "experiments/p110_f6_five_arm_readiness.py":
        "5692e4f28f604d94f0bebb7a9388afcc89a61091",
    "experiments/p111_f6_fit16_freeze.json":
        "fef0bf267904de982916f25acf2c77c2582691ee",
    "experiments/p111_f6_fit16_preflight.py":
        "614ad6073d9bd350b418cb363e4b3e5f40262a0d",
    "experiments/p85mr_dynamics_baselines.py":
        "aa7f6c34a68ee3118f0fe9e44659aa3dec5524bd",
}


def frozen():
    root=Path(__file__).resolve().parents[1]
    raw=(root/"experiments/p112_f6_holdout_freeze.json").read_bytes()
    if p110._blob(raw)!=FREEZE_BLOB:
        raise ValueError("P112_FROZEN_GIT_BLOB_DRIFT")
    f=json.loads(raw)
    d=f["data"]
    a=f["result"]
    if (f.get("schema")!="ttt-hybrid.p112-prospective-f6-train24-validate8-owner-unapproved.v1"
        or f["parent"]["exact_source_sha"]!=SOURCE_PARENT
        or d["train_worlds_n"]!=24 or d["validation_worlds_n"]!=8
        or d["training_model_seeds"]!=list(SEEDS)
        or d["updates_per_seed"]!=800
        or d["extra_reset_do_calls"]!=0
        or d["train_scoring_overlap_allowed"] is not False
        or d["max_world_horizon"]!=HORIZON
        or a["F6_owner_firstuse_authorized"] is not False
        or a["P110_protocol_changed"] is not False
        or a["science_pass"] is not False
        or a["real_source_FIT_runs"]!=0
        or a["real_CAL_runs"]!=0):
        raise ValueError("P112_FROZEN_POLICY_DRIFT")
    for name,blob in PINNED.items():
        if p110._blob((root/name).read_bytes())!=blob:
            raise ValueError("P112_PINNED_PARENT_BLOB_DRIFT")
    p111.locked_protocol()
    return f


def _canonical(record):
    return json.dumps(record,sort_keys=True,separators=(",",":"),
                      allow_nan=False).encode("utf-8")


def _sha(record):
    return hashlib.sha256(_canonical(record)).hexdigest()


def _world_id(row):
    return (row["block"],row["index"])


def _finite(v,code):
    if type(v) not in (float,int) or not math.isfinite(v):
        raise ValueError(code)
    return float(v)


def _state4(v):
    if not isinstance(v,(tuple,list)) or len(v)!=4:
        raise ValueError("P112_INVALID_PHYSICAL_STATE")
    return tuple(_finite(z,"P112_INVALID_PHYSICAL_STATE") for z in v)


def fixture_train_row(block,index,actions):
    """One fabricated episode summary, no action executed."""
    row={"split":"FIT","block":block,"index":index,
         "policy":"physical_balance","action_count":actions,
         "terminated":actions<HORIZON,"truncated":False}
    row["receipt_ref"]="fixture:"+_sha(row)
    return row


def fixture_validation_row(index,action,before,after):
    """One fabricated *selected* consequence per world, not both actions."""
    row={"split":"FIT","block":1,"index":index,
         "policy":"physical_balance","executed_action":action,
         "state_before":list(before),"state_after":list(after)}
    row["receipt_ref"]="fixture:"+_sha(row)
    return row


def _verify_receipt(row):
    h=row.get("receipt_ref") if isinstance(row,dict) else None
    if not isinstance(h,str) or h!="fixture:"+_sha({
        k:v for k,v in row.items() if k!="receipt_ref"}):
        raise ValueError("P112_NOT_EXACT_FIXTURE_RECEIPT")


def validate_worlds(training,validation):
    """Enforce exact disjoint FIT world identities BEFORE checkpoint evaluation."""
    frozen()
    if not isinstance(training,list) or len(training)!=24:
        raise ValueError("P112_REQUIRE_24_TRAIN_WORLDS")
    if not isinstance(validation,list) or len(validation)!=8:
        raise ValueError("P112_REQUIRE_8_HELDOUT_WORLDS")
    train_ids=set()
    val_ids=set()
    for row in training:
        if not isinstance(row,dict) or set(row)!=TRAIN_FIELDS:
            raise ValueError("P112_TRAIN_SCHEMA_OR_ORACLE_DRIFT")
        if (row["split"]!="FIT" or row["policy"]!="physical_balance"
            or type(row["block"]) is not int or type(row["index"]) is not int
            or _world_id(row) not in TRAIN_IDS
            or _world_id(row) in train_ids):
            raise ValueError("P112_TRAIN_WORLD_ID_OR_SPLIT_DRIFT")
        train_ids.add(_world_id(row))
        n=row["action_count"]
        if (type(n) is not int or not 1<=n<=HORIZON
            or type(row["terminated"]) is not bool
            or type(row["truncated"]) is not bool
            or (row["terminated"] and row["truncated"])
            or (n<HORIZON and not(row["terminated"] or row["truncated"]))):
            raise ValueError("P112_TRAIN_FACTUAL_ACTION_OR_TERMINAL_DRIFT")
        _verify_receipt(row)
    if train_ids!=set(TRAIN_IDS):
        raise ValueError("P112_INCOMPLETE_TRAIN_WORLD_GRID")
    for row in validation:
        if not isinstance(row,dict) or set(row)!=VAL_FIELDS:
            raise ValueError("P112_VALIDATION_SCHEMA_OR_ORACLE_DRIFT")
        if (row["split"]!="FIT" or row["policy"]!="physical_balance"
            or type(row["block"]) is not int or type(row["index"]) is not int
            or _world_id(row) not in VAL_IDS
            or _world_id(row) in val_ids
            or _world_id(row) in train_ids):
            raise ValueError("P112_VALIDATION_TRAIN_OVERLAP_OR_SPLIT_DRIFT")
        val_ids.add(_world_id(row))
        if type(row["executed_action"]) is not int or row["executed_action"] not in (0,1):
            raise ValueError("P112_UNQUALIFIED_FACTUAL_ACTION")
        _state4(row["state_before"])
        _state4(row["state_after"])
        _verify_receipt(row)
    if val_ids!=set(VAL_IDS):
        raise ValueError("P112_INCOMPLETE_HELDOUT_WORLD_GRID")
    training_sorted=sorted(training,key=_world_id)
    val_sorted=sorted(validation,key=_world_id)
    return {"train_receipts_sha256":_sha([r["receipt_ref"] for r in training_sorted]),
            "validation_receipts_sha256":_sha([r["receipt_ref"] for r in val_sorted]),
            "train_ids":train_ids,"validation_ids":val_ids,
            "confirmed_train_effects_fixture":sum(r["action_count"] for r in training)}


def fixture_checkpoint(seed,checkpoint_bytes,training,validation,predictions,wall_ms=12.):
    """Fixture-only manifest: a hash CANNOT authenticate trained SSM weights."""
    if type(seed) is not int or seed not in SEEDS or not isinstance(checkpoint_bytes,bytes):
        raise ValueError("P112_UNKNOWN_FIXTURE_CHECKPOINT")
    ids=validate_worlds(training,validation)
    m={"seed":seed,"model_kind":"ssm256",
       "model_source_blob":"aa7f6c34a68ee3118f0fe9e44659aa3dec5524bd",
       "checkpoint_format":"fixture-bytes-only",
       "checkpoint_sha256":hashlib.sha256(checkpoint_bytes).hexdigest(),
       "train_receipts_sha256":ids["train_receipts_sha256"],
       "train_world_ids":[list(k) for k in TRAIN_IDS],
       "validation_world_ids":[list(k) for k in VAL_IDS],
       "optimizer":"AdamW","optimizer_steps":800,"fit_only":True,
       "cal_or_ood_used":False,"validation_predictions_sha256":_sha(predictions),
       "validation_wall_ms":wall_ms}
    m["receipt_ref"]="fixture:"+_sha(m)
    return m


def score_fixture_holdout(training,validation,models):
    """Recompute *factual* error from eight heldout rows, never a caller scalar.

    Model forecasts are fabricated fixture inputs, not independently executed
    learned checkpoints; the selected model's validation score is not an
    independent test because these same eight worlds select its seed.
    """
    meta=validate_worlds(training,validation)
    if not isinstance(models,dict) or set(models)!=set(SEEDS):
        raise ValueError("P112_REQUIRE_ALL_THREE_SEED_MODELS")
    ordered=sorted(validation,key=_world_id)
    true_delta=[tuple(b-a for a,b in zip(_state4(r["state_before"]),
                                        _state4(r["state_after"]))) for r in ordered]
    zero_mse=sum(x*x for row in true_delta for x in row)/(len(ordered)*4)
    if zero_mse<=0:
        raise ValueError("P112_UNINFORMATIVE_ZERO_PREDICTOR")
    score={}
    for seed in SEEDS:
        d=models[seed]
        if not isinstance(d,dict) or set(d)!={"manifest","bytes","predictions"}:
            raise ValueError("P112_MODEL_ENTRY_FIELDS")
        m,ckpt,pred=d["manifest"],d["bytes"],d["predictions"]
        if not isinstance(m,dict) or set(m)!=MANIFEST_FIELDS or not isinstance(ckpt,bytes) or not 1<=len(ckpt)<=8*1024*1024:
            raise ValueError("P112_BAD_CHECKPOINT_FIELDS")
        if (type(m["seed"]) is not int or m["seed"]!=seed
            or m["model_kind"]!="ssm256" or
            m["model_source_blob"]!="aa7f6c34a68ee3118f0fe9e44659aa3dec5524bd"
            or m["checkpoint_format"]!="fixture-bytes-only"
            or m["checkpoint_sha256"]!=hashlib.sha256(ckpt).hexdigest()
            or m["train_receipts_sha256"]!=meta["train_receipts_sha256"]
            or m["train_world_ids"]!=[list(k) for k in TRAIN_IDS]
            or m["validation_world_ids"]!=[list(k) for k in VAL_IDS]
            or m["optimizer"]!="AdamW"
            or type(m["optimizer_steps"]) is not int or m["optimizer_steps"]!=800
            or m["fit_only"] is not True or m["cal_or_ood_used"] is not False):
            raise ValueError("P112_UNTRAINED_OR_CONTAMINATED_CHECKPOINT_FIXTURE")
        if _finite(m["validation_wall_ms"],"P112_BAD_EVALUATION_COST")<=0:
            raise ValueError("P112_BAD_EVALUATION_COST")
        if not isinstance(pred,list) or len(pred)!=8:
            raise ValueError("P112_INCOMPLETE_EIGHT_WORLD_PREDICTIONS")
        if m["validation_predictions_sha256"]!=_sha(pred) or m["receipt_ref"]!="fixture:"+_sha({k:v for k,v in m.items() if k!="receipt_ref"}):
            raise ValueError("P112_TAMPERED_PREDICTION_OR_CHECKPOINT_RECEIPT")
        if not all(isinstance(row,(list,tuple)) for row in pred):
            raise ValueError("P112_BAD_PREDICTED_DELTA")
        forecast=[_state4(row) for row in pred]
        value=sum((p-y)**2 for pred_row,truth in zip(forecast,true_delta)
                  for p,y in zip(pred_row,truth))/(len(ordered)*4)
        score[seed]={"mse_fixture":value,"ratio_to_zero_fixture":value/zero_mse,
                     "evaluation_wall_ms_fixture":float(m["validation_wall_ms"])}
    winner=min(SEEDS,key=lambda seed:(score[seed]["mse_fixture"],seed))
    competent=score[winner]["ratio_to_zero_fixture"]<=0.75
    return {
        "schema":"ttt-hybrid.p112-train24-val8-source-neutral-fixture.v1",
        "classification":"ENGINEERING_FIXTURE_ONLY_NOT_GENUINE_CROSS_FIT",
        "training_worlds_fixture":24,"validation_worlds_fixture":8,
        "train_and_heldout_disjoint":True,
        "source_effects_executed":0,"training_optimizer_steps_executed":0,
        "train_effects_count_fixture":meta["confirmed_train_effects_fixture"],
        "selected_seed_fixture_only":winner,
        "ssm_selected_heldout_mse_fixture":score[winner]["mse_fixture"],
        "zero_predictor_mse_fixture":zero_mse,
        "all_three_seed_validation_metrics_fixture":score,
        "competence_score_is_selection_biased":True,
        "development_fixture_threshold_met":competent,
        "trained_real_ssm":False,"real_checkpoint_authenticated":False,
        "P110_protocol_owner_revision_required":True,
        "P108_cross_world_state_qualified":False,
        "F6_real_firstuse_authorized":False,
        "real_CAL_admitted":False,"scientific_pass":False,"deployment":False,
    }


def deny_external_trial(*_args,**_kwargs):
    raise PermissionError("P112_NO_F6_SOURCE_OR_CAL_FIRSTUSE_AUTHORITY")
