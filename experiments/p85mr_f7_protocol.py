"""Frozen F7 new FIT-only native experiment; never imports/starts simulator."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FREEZE_BLOB="31c6107766389b73a228c36cb2bfa2e1a0de1f24"
PARENT_SHA="4d5c49be278077fe3b0be80080d39128a153aa77"
SEED_BASE=1821000000
FIT_WORLDS=128
TRAIN_WORLDS=96
VAL_WORLDS=32
MAX_ACTIONS=12288
HORIZON=96
MODEL_SEEDS=(985201,985203,985207)
ARMS=("original_four_block_HybridPilot",
      "original_four_block_with_16D_raw_action_key_bypass",
      "original_256state_SSM")
SCALES=(2.4,3.0,.2095,3.5)


def git_blob(data):
    return hashlib.sha1(b"blob "+str(len(data)).encode("ascii")+b"\0"+data).hexdigest()


def frozen():
    raw=(ROOT/"experiments/p85mr_f7_native_freshfit_freeze.json").read_bytes()
    if git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("F7_FROZEN_NEW_SOURCE_DRIFT")
    f=json.loads(raw)
    a=f["scope"]; d=f["frozen_development"];o=f["owner"]; auth=f["authority"]
    if (f.get("schema")!="ttt-hybrid.f7-native-credit-bypass-fresh-fit-firstuse.v1"
        or a["f5_original_sha"]!=PARENT_SHA
        or a["new_seed_base"]!=SEED_BASE
        or a["first_use_max_sessions"]!=FIT_WORLDS
        or a["first_use_max_factual_actions"]!=MAX_ACTIONS
        or a["source_max_episode_horizon"]!=HORIZON
        or a["train_world_indices"]!="0..95"
        or a["validation_world_indices"]!="96..127"
        or d["arms"]!=list(ARMS)
        or d["model_seeds"]!=list(MODEL_SEEDS)
        or d["training_steps_per_seed"]!=800
        or d["fit_train_worlds"]!=TRAIN_WORLDS
        or d["fit_validation_worlds"]!=VAL_WORLDS
        or o["source_branch"]!="research/p85mr-f7-native-raw-bypass-freshfit-20261112"
        or o["one_shot_push_only_run_number_1_attempt_1"] is not True
        or auth["approve_only_F7_FIT_new_source"] is not True
        or auth["approve_F7_CAL"] is not False
        or auth["approve_F6_A5"] is not False
        or auth["science_effects_started"] is not False
        or auth["scientific_pass"] is not False):
        raise ValueError("F7_SCOPE_OR_AUTHORITY_DRIFT")
    if git_blob((ROOT/"model.py").read_bytes())!=a["model_blob"]:
        raise ValueError("F7_ORIGINAL_MODEL_DRIFT")
    if len(set(seed(i) for i in range(FIT_WORLDS)))!=FIT_WORLDS:
        raise ValueError("F7_SEED_COLLISION")
    reserved=[a[k] for k in ("future_CAL_reserved_seed_base",
                            "future_OOD_reserved_seed_base",
                            "future_NEGATIVE_reserved_seed_base")]
    if len(set(reserved+[SEED_BASE]))!=4 or any(abs(x-SEED_BASE)<10000 for x in reserved):
        raise ValueError("F7_RESERVED_SEED_SCOPE")
    return f


def seed(index):
    if type(index) is not int or not 0<=index<FIT_WORLDS:
        raise ValueError("F7_OUT_OF_SCOPE_WORLD")
    return SEED_BASE+index


def state4(values):
    if (not isinstance(values,(tuple,list)) or len(values)!=4
        or any(type(x) not in (int,float) or not math.isfinite(x) for x in values)):
        raise ValueError("F7_INVALID_PUBLIC_STATE")
    return tuple(float(v) for v in values)


def factual_policy(index,step,values):
    """Declared exploratory balance policy; no future or unchosen outcomes."""
    q=state4(values)
    if type(step) is not int or not 1<=step<=HORIZON:
        raise ValueError("F7_INVALID_STEP")
    balanced=int(q[2]+.5*q[3]>0)
    explore=int(((seed(index)+step*7)%8)==0)
    return balanced^explore


def no_external_from_CI(*args,**kwargs):
    raise PermissionError("F7_SOURCE_MAY_RUN_ONLY_ONE_SHOT_EXACT_OWNER_ROUTE")


def proposal_receipt():
    f=frozen()
    return {"status":"F7_FROZEN_FIT_ONLY_NO_REAL_EFFECT",
      "seed_base":SEED_BASE,"total_sessions":FIT_WORLDS,
      "max_factual_actions":MAX_ACTIONS,"model_seeds":MODEL_SEEDS,
      "future_CAL_authorized":False,
      "owner_custody_qualified":False,"source_firstuse_executed":False,
      "science_pass":False}
