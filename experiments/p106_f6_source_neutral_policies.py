"""P106 source-neutral 2-action policy kernel for a *future* F6 first-use.

No simulator import, source session, network, runner dispatch, or execution.
Original frozen protocol commit 389bf0a3c6e023309d63a5d01a7f353ed0ffe0ec.
PR CI can only test predetermined in-memory states, estimates and seed integers.
The first on-policy CartPoleWind experiment is NOT authorized by this code.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREEZE_COMMIT = "389bf0a3c6e023309d63a5d01a7f353ed0ffe0ec"
FREEZE_BLOB = "3a2cee9454edf2e2f9937e35492d618e30f79de0"
PARENT_SHA = "955d032ac21b71b9d5e02061ae9727600205d569"
DIM = 16
SCALES = (2.4, 3.0, .2095, 3.5)
GOAL_WEIGHTS = (.1, .05, 3., .25)
SIGMA = (1.,1.,1.,1.,1.,2.,2.) + (0.,)*9
ACTIONS = (0,1)
POLICIES = ("guarded_voi","a_opt","greedy_goal","physical_balance")
EPSILONS = (.05,.15,.30)
FUTURE_SEEDS = {"FIT":1820000000,"CAL":1820100000}
RESERVED_BASES = {"OOD":1820200000,"NEGATIVE":1820300000}
ORIGINAL_BLOBS = {
    "model.py":"5d9ea57b827aa1796548447868fa5ee14c251b49",
    "experiments/p105_authentic_action_support_freeze.json":
        "cd1bb40fd74df9d693067798fa5360a0cb205cbb",
    "experiments/p105_f3_action_support_preflight.py":
        "ab1bcb17da2e4eade5b0422e627cedb7b41da544",
}


def _git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob "+str(len(raw)).encode("ascii")+b"\0"+raw).hexdigest()


def locked_protocol() -> dict:
    """Custody/eligibility only. No scientific data are born here."""
    raw=(ROOT/"experiments/p106_f6_source_neutral_freeze.json").read_bytes()
    if _git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("P106_FROZEN_PROTOCOL_BLOB_DRIFT")
    cfg=json.loads(raw)
    if (cfg.get("schema") !=
        "ttt-hybrid.p106-f6-source-neutral-voi-policy-readiness-freeze.v1"
        or cfg.get("parent",{}).get("sha")!=PARENT_SHA
        or cfg["authority"]["fresh_F6_external_first_use_authorized"] is not False
        or cfg["source"]["source_firstuse_now"] is not False
        or cfg["outcome"]["external_source_executed"] is not False
        or cfg["predeclared_future_cohorts"]["FIT"]["base"]!=FUTURE_SEEDS["FIT"]
        or cfg["predeclared_future_cohorts"]["CAL"]["base"]!=FUTURE_SEEDS["CAL"]
        or cfg["predeclared_future_cohorts"]["OOD_RESERVED"]["base"]!=RESERVED_BASES["OOD"]
        or cfg["predeclared_future_cohorts"]["negative_RESERVED"]["base"]!=RESERVED_BASES["NEGATIVE"]
        or cfg["selector"]["epsilon_grid_FIT_only"]!=list(EPSILONS)
        or cfg["source"]["sha"]!="c94e9a4cd3641588b6b8507e71ccee5c27032bdf"):
        raise ValueError("P106_PROTOCOL_SCOPE_OR_AUTH_DRIFT")
    for filename,expected in ORIGINAL_BLOBS.items():
        if _git_blob((ROOT/filename).read_bytes())!=expected:
            raise ValueError("P106_ORIGINAL_MODEL_OR_P105_CUSTODY_DRIFT")
    return cfg


def _number(value):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError("P106_NONFINITE_OR_NONNUMERIC")
    return float(value)


def _vector(value, n):
    if (not isinstance(value,(tuple,list)) or len(value)!=n):
        raise ValueError("P106_VECTOR_SHAPE")
    return tuple(_number(x) for x in value)


def _matrix(value):
    if (not isinstance(value,(tuple,list)) or len(value)!=DIM):
        raise ValueError("P106_MATRIX_SHAPE")
    return tuple(_vector(row,DIM) for row in value)


def action_key(state4, desired_action):
    """Pre-action P80 feature transform (same formulas, 16D continuous).

    Only caller-supplied observation and candidate discrete action; never
    outcome, next observation, hidden wind or source model.
    """
    s=_vector(state4,4)
    if type(desired_action) is not int or desired_action not in ACTIONS:
        raise ValueError("P106_NOT_DISCRETE_REAL_ACTION")
    k=[1.0]+[math.tanh(x/scale) for x,scale in zip(s,SCALES)]
    k.extend((float(desired_action==0),float(desired_action==1)))
    k.extend([0.]*9)
    norm=math.sqrt(math.fsum(x*x for x in k))
    if norm<1e-12:
        raise ValueError("P106_ZERO_KEY")
    return tuple(x/norm for x in k)


def future_seed_schedule(split, block, index):
    """Return only integer proposed seeds; no source construction or effects."""
    locked_protocol()
    if (split not in FUTURE_SEEDS or type(block) is not int
        or not 0<=block<3 or type(index) is not int or not 0<=index<16):
        raise ValueError("P106_SEALED_OR_UNKNOWN_SOURCE")
    seed=FUTURE_SEEDS[split]+1000*block+index
    if seed>=2**31 or seed < 0:
        raise ValueError("P106_SEED_OUT_OF_RANGE")
    return seed


def proposed_partition():
    """Pure integer check only; protected cohorts are never instantiated."""
    locked_protocol()
    fit={future_seed_schedule("FIT",block,w) for block in range(3) for w in range(16)}
    cal={future_seed_schedule("CAL",block,w) for block in range(3) for w in range(16)}
    if (len(fit)!=48 or len(cal)!=48 or fit&cal or
        min(fit|cal)<=1810500063 or max(fit|cal)>=1820200000):
        raise ValueError("P106_SEED_COLLISION_WITH_SPENT_OR_RESERVED")
    return {"fit_seeds":len(fit),"cal_seeds":len(cal),
            "unique_future_fit_cal":len(fit|cal),"overlap":0,
            "spent_F3_replayed":False,"reserved_source_created":False,
            "source_events_created":False}


def _model_vector(W,k):
    return tuple(math.fsum(W[i][j]*k[j] for j in range(DIM))
                 for i in range(DIM))


def _quadratic(P,k):
    pk=tuple(math.fsum(P[i][j]*k[j] for j in range(DIM))
             for i in range(DIM))
    variance=math.fsum(k[i]*pk[i] for i in range(DIM))
    if variance< -1e-10:
        raise ValueError("P106_NEGATIVE_POSTERIOR_VARIANCE")
    den=1.+variance
    if den<=1e-12:
        raise ValueError("P106_INVALID_INFO_DENOMINATOR")
    utility=math.fsum(SIGMA[j]*pk[j]*pk[j] for j in range(DIM))/den
    return utility


def _physical_penalty(state,W,k):
    expected=_model_vector(W,k)
    # Model estimates the original P80 tanh-normalized 4 physical deltas.
    # Clipping is disclosed as an assumption, not a witnessed consequence.
    future=[]
    for v,e,scale in zip(state,expected,SCALES):
        safe=max(-.98,min(.98,e))
        future.append(v+math.atanh(safe)*scale)
    return math.fsum(weight*((v/scale)**2) for weight,v,scale
                     in zip(GOAL_WEIGHTS,future,SCALES))


def select_policy(policy, state4, weights16, inverse_precision16,
                  *, epsilon=.05):
    """One legal model-based choice; returns *action*, not future effect.

    This policy is source-neutral and makes no observation of unselected
    outcomes. The authenticated first-use runner is NOT implemented here.
    """
    if policy not in POLICIES or epsilon not in EPSILONS:
        raise ValueError("P106_UNQUALIFIED_POLICY_OR_EPSILON")
    state=_vector(state4,4)
    W=_matrix(weights16)
    P=_matrix(inverse_precision16)
    # Exactly symmetric PSD posterior precision should be delivered by a
    # separately qualified confirmed-action RLS. This checks symmetry and
    # local candidate variance; no unverified claim of full PSD proof.
    if any(abs(P[i][j]-P[j][i])>1e-9
           for i in range(DIM) for j in range(i+1,DIM)):
        raise ValueError("P106_NON_SYMMETRIC_PRECISION")
    if policy=="physical_balance":
        return int(state[2]+.5*state[3]>0)
    keys=[action_key(state,a) for a in ACTIONS]
    info=[_quadratic(P,k) for k in keys]
    costs=[_physical_penalty(state,W,k) for k in keys]
    opt=max(ACTIONS,key=lambda a: info[a])
    greedy=min(ACTIONS,key=lambda a: costs[a])
    if policy=="a_opt":
        return opt
    if policy=="greedy_goal":
        return greedy
    return opt if costs[opt]-costs[greedy] <= epsilon*max(costs[greedy],1e-6) else greedy


def refuse_unapproved_source_firstuse(*_args,**_kwargs):
    """A caller must use a separately authorized, qualified owner route."""
    raise PermissionError("P106_F6_EXTERNAL_FIRSTUSE_NOT_AUTHORIZED")


def engineering_receipt():
    cfg=locked_protocol()
    part=proposed_partition()
    return {
        "schema":"ttt-hybrid.p106-f6-source-neutral-engineering.v1",
        "status":"PREFLIGHT_ONLY_NOT_SCIENCE",
        "freeze_commit":FREEZE_COMMIT,"freeze_blob":FREEZE_BLOB,
        "source_sha":cfg["source"]["sha"],
        "new_source_firstuse_authorized":False,
        "new_source_effects_executed":False,"science_pass":False,
        "runner_dispatched":False,"owner_workflow_created":False,
        "source_neutral_policy_count":len(POLICIES),
        "proposed_seed_partition":part,
    }
