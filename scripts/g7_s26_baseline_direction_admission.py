"""G7-S26 conventional GRU/KRR action-effect *pre-admission*, stdlib only.

This module NEVER trains, calibrates, imports torch, reads frozen scientific
worlds, accesses a runner, or authenticates assertions inside user-provided
receipts. It checks what a subsequently authorized/independently witnessed
baseline training run would need to demonstrate before its science is
considered: true validation paired MSE, directional competence and coverage,
fit/validation disjointness, GRU plateau, train-only KRR calibration,
equal observed streams, source/compute/cold-checkpoint custody.

The directional majority is an additional *pre-first-use sanity gate*, not a
modification of the frozen S26 efficacy or success threshold. Claimed numbers
can be fabricated: PASS means ONLY structurally coherent claimed receipts.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

S26_SCHEMA="ttt-hybrid.g7-s26.reserve-aware-contradiction-timing.v1"
S26_BLOB="34251b2cde6373ed484dbc21e4617c9165047e93"
S25_BLOB="31633b341403729b7ffbb7d73f3a04098ad44380"
S20_BLOB="ac3f9ddd5b879e53e62030707fb6419adaf732ca"
S25_HEAD="108808f7b685107a76fae1ce5605b7ff9e2bac27"
FIT_SEEDS=(7207,7219,7229)
VALIDATION_SEEDS=(7307,7319,7327)
ENGINEERING_SEEDS=(3101,3109,3119)
ARMS=("GRU_fully_trained","RBF_KRR")
MIN_UPDATES=640
MAX_UPDATES=1600
CHECKPOINT_INTERVAL=64
VALIDATION_WINDOW=5
RSS_LIMIT=2147483648
TIME_LIMIT=900
VALID_EFFECT_MIN_SIGN=0.012


class S26PreAdmissionBlocked(ValueError):
    pass


def require(ok,why:str):
    if not ok:
        raise S26PreAdmissionBlocked(why)


def integer(x,why:str,minimum:int=0)->int:
    require(type(x) is int and x>=minimum,why)
    return x


def number(x,why:str,positive:bool=False)->float:
    require(type(x) in (int,float) and math.isfinite(x) and
            (x>0 if positive else x>=0),why)
    return float(x)


def digest(x,why:str,length:int=64)->str:
    require(type(x) is str and len(x)==length and
            all(c in "0123456789abcdef" for c in x),why)
    return x


def frozen_protocol(path:Path)->dict:
    """Read the frozen JSON, never open data from its reserved seeds."""
    p=json.loads(path.read_text(encoding="utf-8"))
    require(p.get("schema")==S26_SCHEMA,"S26_FROZEN_SCHEMA_DRIFT")
    provenance=p["rationale"]
    require(provenance["s25_baseline_head"]==S25_HEAD and
            provenance["s25_protocol_git_blob"]==S25_BLOB and
            provenance["s20_world_git_blob"]==S20_BLOB,
            "S26_FROZEN_SOURCE_DRIFT")
    roots=p["evaluation"]["science_seed_namespace_reserved_unconsumed"]
    require(tuple(roots["fit"])==FIT_SEEDS and
            tuple(roots["validation"])==VALIDATION_SEEDS and
            tuple(p["evaluation"]["dev_only_seed_set"])==ENGINEERING_SEEDS,
            "S26_FROZEN_SPLIT_DRIFT")
    groups=[value for name,value in roots.items() if isinstance(value,list)]
    require(len({s for group in groups for s in group})==
            sum(map(len,groups)) and
            not any(s in ENGINEERING_SEEDS for group in groups for s in group),
            "S26_FROZEN_SPLITS_COLLIDE")
    authority=p["authority"]
    require(authority["scientific_first_use_authorized"] is False and
            authority["no_scientific_training_or_fit"] is True and
            authority["no_heldout_seed_consumption"] is True and
            authority["no_merge_or_promotion"] is True,
            "S26_UNAUTHORIZED_FIRST_USE")
    require(p["budgets"]["training_source_min_updates"]==MIN_UPDATES and
            p["budgets"]["training_source_max_updates"]==MAX_UPDATES and
            p["budgets"]["max_rss_bytes"]==RSS_LIMIT and
            p["budgets"]["max_cpu_wall_seconds_per_arm"]==TIME_LIMIT,
            "S26_FROZEN_RESOURCE_DRIFT")
    return p


def _action_effect_claim(row:dict)->dict:
    """Directional readiness must be independently testable from pair MSE."""
    metric=row.get("action_effect_validation")
    require(type(metric) is dict,
            "S26_MISSING_PAIRED_ACTION_EFFECT_VALIDATION")
    n=integer(metric.get("paired_world_steps"),"S26_MISSING_VALIDATION_PAIRS",16)
    significant=integer(metric.get("true_effect_significant"),
                        "S26_INVALID_SIGNIFICANT_EFFECT_COUNT")
    selected=integer(metric.get("model_effect_significant"),
                     "S26_INVALID_PREDICTED_CONTRAST_COUNT")
    correct=integer(metric.get("sign_correct"),
                    "S26_INVALID_SIGN_CORRECT_COUNT")
    wrong=integer(metric.get("sign_wrong"),
                  "S26_INVALID_SIGN_WRONG_COUNT")
    uninformative=integer(metric.get("uninformative_model_effect"),
                          "S26_INVALID_UNINFORMATIVE_EFFECT_COUNT")
    require(0<significant<=n and 0<=selected<=n and
            correct+wrong+uninformative==significant and
            correct+wrong<=selected and
            (correct+wrong)>=16,
            "S26_ACTION_EFFECT_CONFUSION_MATRIX_INCONSISTENT")
    require(correct>wrong,
            "S26_BASELINE_WRONG_ACTION_EFFECT_DIRECTION")
    # Avoid the degenerate always-near-zero predictor: a correct-sign fraction
    # on a handful of predictions is not direction readiness.
    require((correct+wrong)*2>significant,
            "S26_BASELINE_SIGN_COVERAGE_UNINFORMATIVE")
    require(metric.get("predictions_made_before_observation") is True and
            metric.get("same_state_paired_action_contrast") is True and
            metric.get("validation_only_not_heldout") is True,
            "S26_NONCAUSAL_ACTION_EFFECT_VALIDATION")
    effect_mae=number(metric.get("mean_absolute_effect_error"),
                      "S26_INVALID_EFFECT_MAGNITUDE_ERROR")
    require(effect_mae>=0,"S26_INVALID_EFFECT_MAGNITUDE_ERROR")
    return {"paired":n,"significant":significant,
            "correct":correct,"wrong":wrong,"uninformative":uninformative}


def _common_claim(row:dict,arm:str,fit:int,validation:int,sha:str):
    require(type(row) is dict,"S26_BASELINE_RECEIPT_NOT_DICT")
    require(row.get("arm")==arm,"S26_BASELINE_ARM_SUBSTITUTION")
    require(row.get("fit_seed")==fit and row.get("validation_seed")==validation,
            "S26_BASELINE_FIT_VALIDATION_ROLE_DRIFT")
    require(row.get("candidate_sha")==sha and row.get("s25_ancestor_sha")==S25_HEAD,
            "S26_BASELINE_SOURCE_SHA_DRIFT")
    require(row.get("frozen_s26_protocol_blob")==S26_BLOB and
            row.get("frozen_s25_protocol_blob")==S25_BLOB and
            row.get("s20_physics_source_blob")==S20_BLOB,
            "S26_BASELINE_FROZEN_BLOB_DRIFT")
    for k in ("fit_stream_sha256","validation_stream_sha256",
              "checkpoint_sha256","model_source_sha256"):
        digest(row.get(k),"S26_MISSING_OR_INVALID_"+k.upper())
    require(row["fit_stream_sha256"]!=row["validation_stream_sha256"],
            "S26_FIT_VALIDATION_STREAM_COLLISION")
    require(row.get("training_uses_fit_only") is True and
            row.get("model_selection_uses_validation_only") is True and
            row.get("evaluation_uses_future_observation") is False,
            "S26_BASELINE_TARGET_LEAKAGE_OR_SPLIT_DRIFT")
    require(row.get("actual_checkpoint_restored") is True,
            "S26_BASELINE_NO_COLD_CHECKPOINT_RESTORE")
    require(integer(row.get("fit_pair_exposures"),"S26_INVALID_TRAIN_EXPOSURE",32)>0
            and integer(row.get("validation_pair_exposures"),
                        "S26_INVALID_VALIDATION_EXPOSURE",32)>0,
            "S26_BASELINE_INSUFFICIENT_EXPOSURES")
    number(row.get("train_wall_seconds"),"S26_INVALID_TRAIN_WALL",positive=True)
    number(row.get("validate_wall_seconds"),"S26_INVALID_VALIDATION_WALL",positive=True)
    number(row.get("train_cpu_seconds"),"S26_INVALID_TRAIN_CPU",positive=True)
    number(row.get("validate_cpu_seconds"),"S26_INVALID_VALIDATION_CPU",positive=True)
    require(row["train_wall_seconds"]+row["validate_wall_seconds"]<=TIME_LIMIT,
            "S26_BASELINE_EXECUTION_TIME_CAP_EXCEEDED")
    require(integer(row.get("peak_rss_bytes"),"S26_INVALID_RSS",1)<=RSS_LIMIT,
            "S26_BASELINE_MEMORY_BUDGET_EXCEEDED")
    integer(row.get("checkpoint_bytes"),"S26_INVALID_CHECKPOINT_SIZE",1)
    metric=number(row.get("validation_paired_mse"),
                  "S26_INVALID_PAIRED_MSE",positive=True)
    baseline=number(row.get("naive_validation_paired_mse"),
                    "S26_INVALID_NAIVE_PAIRED_MSE",positive=True)
    require(metric<baseline,"S26_NOT_BETTER_THAN_NAIVE_PAIRED_MSE")
    require(not any(x in str(k).lower() for k in row.keys()
                    for x in ("heldout","oracle","future_target","test_seed")),
            "S26_BASELINE_FORBIDDEN_SCIENCE_ORACLE_FIELD")
    return _action_effect_claim(row)


def _gru(row:dict):
    updates=integer(row.get("optimizer_updates"),"S26_GRU_INVALID_UPDATES")
    require(MIN_UPDATES<=updates<=MAX_UPDATES and
            updates%CHECKPOINT_INTERVAL==0,
            "S26_GRU_UNDERTRAINED_OR_BUDGET")
    require(row.get("batch_worlds")==32 and
            row.get("width")==32 and
            row.get("optimizer")=="AdamW_lr0.001_wd0.01" and
            row.get("converged") is True and
            row.get("trainer_status")=="BASELINE_VALIDATION_PLATEAU",
            "S26_GRU_FROZEN_TRAINER_OR_PLATEAU_MISSING")
    history=row.get("validation_history")
    require(type(history) is list and len(history)>=VALIDATION_WINDOW,
            "S26_GRU_MISSING_FIVE_VALIDATION_CHECKS")
    previous=0
    values=[]
    for v in history:
        require(type(v) is dict,"S26_GRU_VALIDATION_NOT_OBJECT")
        step=integer(v.get("step"),"S26_GRU_INVALID_VALIDATION_STEP",MIN_UPDATES)
        mse=number(v.get("paired_mse"),
                   "S26_GRU_INVALID_CHECKPOINT_MSE",positive=True)
        require(previous==0 or step-previous==CHECKPOINT_INTERVAL,
                "S26_GRU_VALIDATION_NOT_SPACED_64")
        previous=step
        values.append(mse)
    require(previous==updates,"S26_GRU_VALIDATION_NOT_AT_FINAL_UPDATE")
    last=values[-VALIDATION_WINDOW:]
    require((max(last)-min(last))/min(last)<0.001,
            "S26_GRU_PLATEAU_NOT_ESTABLISHED")
    require(abs(values[-1]-row["validation_paired_mse"])<1e-8,
            "S26_GRU_FINAL_PAIR_MSE_DOES_NOT_MATCH")


def _krr(row:dict):
    require(row.get("gamma") in (0.5,2.0,6.0) and
            row.get("ridge") in (0.01,0.08),
            "S26_KERNEL_UNREGISTERED_CALIBRATION_GRID")
    require(row.get("normalization_train_only") is True and
            row.get("calibration_validation_only") is True and
            row.get("history_steps")==32 and
            row.get("numeric_query_width")==3 and
            integer(row.get("train_query_samples"),
                    "S26_KERNEL_TRAIN_QUERY_COUNT_INVALID",16)>=16,
            "S26_KERNEL_CALIBRATION_OR_INPUT_INVALID")


def validate_claimed_baselines(protocol:dict,candidate_sha:str,records:list[dict])->dict:
    """Logical receipt check ONLY. NO proof records are authentic or executed."""
    digest(candidate_sha,"S26_EXPECTED_COMMIT_SHA_INVALID",length=40)
    require(protocol["schema"]==S26_SCHEMA and
            protocol["authority"]["scientific_first_use_authorized"] is False,
            "S26_FROZEN_FIRST_USE_GATE_DRIFT")
    require(type(records) is list and len(records)==6,
            "S26_INCOMPLETE_BASELINE_ARM_SEED_MATRIX")
    lookup={}
    for r in records:
        require(type(r) is dict,"S26_BASELINE_RECEIPT_NOT_DICT")
        key=(r.get("arm"),r.get("fit_seed"))
        require(key not in lookup,"S26_DUPLICATE_CONTROL_OR_SOURCE_SEED")
        lookup[key]=r
    captured={}
    fits=set(); validations=set(); checkpoints=set()
    for fit,val in zip(FIT_SEEDS,VALIDATION_SEEDS):
        for arm in ARMS:
            record=lookup.get((arm,fit))
            require(record is not None,
                    "S26_MISSING_FROZEN_CONVENTIONAL_CONTROL")
            captured[(fit,arm)]=_common_claim(record,arm,fit,val,candidate_sha)
            if arm==ARMS[0]:
                _gru(record)
            else:
                _krr(record)
            checkpoints.add(record["checkpoint_sha256"])
        a=lookup[(ARMS[0],fit)]
        b=lookup[(ARMS[1],fit)]
        require(a["fit_stream_sha256"]==b["fit_stream_sha256"] and
                a["validation_stream_sha256"]==b["validation_stream_sha256"] and
                a["validation_pair_exposures"]==b["validation_pair_exposures"] and
                a["fit_pair_exposures"]==b["fit_pair_exposures"],
                "S26_GRU_KERNEL_NOT_MATCHED_ON_SOURCE_EXPOSURE")
        fits.add(a["fit_stream_sha256"])
        validations.add(a["validation_stream_sha256"])
    require(len(fits)==3 and len(validations)==3 and
            not (fits&validations),
            "S26_CROSS_SEED_DATASET_REPLAY_OR_LEAKAGE")
    require(len(checkpoints)==6,"S26_REUSED_BASELINE_CHECKPOINT")
    return {
        "schema":"ttt-hybrid.g7-s26.baseline-direction-pre-admission.v1",
        "status":"PASS_CLAIMED_RECEIPT_SHAPES_ONLY_NEVER_SCIENCE",
        "expected_s26_fit_seed_count":3,
        "expected_strong_control_arms":list(ARMS),
        "claimed_fit_validation_records":len(records),
        "claimed_val_action_effect_sign_direction_positive":True,
        "claimed_gru_converged":True,
        "claimed_kernel_train_normalization_and_val_calibration":True,
        "independent_receipt_authenticity_verified":False,
        "baseline_training_performed_here":False,
        "scientific_first_use":"NOT_RUN",
        "trained_baseline_science_accepted":False,
        "full_windows_linux_regression":"UNVERIFIED",
        "deployment":"NOT_AUTHORIZED",
    }
