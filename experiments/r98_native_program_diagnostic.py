"""R98: unseen composition of public causal primitives inside native Hybrid stem.

PURE TEST-only development: independent fresh R85 physics; input is lawful
preintervention public role/operator instructions and noisy A cues; all
models train ONLY on the realized reward of one selected physical action.
No R85-R97 scored source, weight reuse, hidden parity/mux supervision or
official native GPU/independent-science claims.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import r81_paired_history as r81
from experiments import r85_delayed_composition as r85
from experiments import r86_native_compositional_value as r86
from experiments import r87_public_composition_diagnostic as r87
from experiments import r88_causal_event_access as r88
from experiments import r95_learned_relation_diagnostic as r95
from experiments import r97_unseen_binding_diagnostic as r97

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r98_native_program_freeze.json"
FREEZE_BLOB="44a2b0802efc81e2360df2a44aeed4f4507f1967"
SCHEMA="ttt-hybrid.r98-native-internal-compositional-transfer.testonly.v1"
PARENT="ebfca5928edf6bcec7f0e3ec9340255bfad44680"
BRANCH="research/r98-native-internal-compositional-operator-20261116"
SEEDS=(1001911,1001923)
SALT={"FIT":25390111,"VALID":25390211,"DEV_SEEN":25390311,
      "DEV_COMPOSE":25390411,"OOD_SHIFT":25390511,"ROLE_REASSIGN":25390611}
COUNT={"FIT":2048,"VALID":256,"DEV_SEEN":256,
       "DEV_COMPOSE":256,"OOD_SHIFT":256,"ROLE_REASSIGN":256}
NEGATIVE_SALT=25390711
BATCH_SALT=25390811
BOOTSTRAP_SALT=25390911
UNIT_SALT=25399991
TRAIN_PROGRAMS=((0,0),(0,1),(1,0))
COMPOSE_PROGRAM=(1,1)
ROLE_BINDINGS=((0,1,2),(1,2,3),(2,3,0))
HELDOUT_BINDING=(3,0,1)
ARMS=("NATIVE32_RAW_TAGGED_EVENTS","NATIVE32_FLAT_STEM_ADAPTER",
      "NATIVE32_INTERNAL_ROLE_PROGRAM_STEM",
      "NATIVE32_EXTERNAL_ROLE_PROGRAM_SIDECAR",
      "MLP64_PUBLIC_RAW_PROGRAM","ROLE_SET_PUBLIC_PROGRAM",
      "GRU64_PUBLIC_TAGGED_EVENTS","MLP64_PUBLIC_CANONICAL_UPPER",
      "ROLE_SET_PROGRAM_CORRUPTED_NEGATIVE")
(NATIVE_RAW,NATIVE_FLAT,NATIVE_INTERNAL,NATIVE_SIDECAR,
 MLP,SET,GRU,UPPER,NEGATIVE)=ARMS
STEPS=3072
BATCH=16
MAX_SECONDS=260


def frozen():
    if r85.git_blob(FREEZE.read_bytes())!=FREEZE_BLOB:
        raise RuntimeError("R98_PREIMPLEMENTATION_FROZEN_GIT_BLOB_CHANGED")
    r97.frozen()
    f=json.loads(FREEZE.read_text(encoding="utf-8"))
    p,s,t,e,x=f["parent"],f["source"],f["training"],f["evaluation"],f["execution"]
    checks=e["mechanism_checks"]
    if (f["schema"]!=SCHEMA
        or f["stage"]!="PROSPECTIVE_FREEZE_BEFORE_IMPLEMENTATION_TESTS_OR_FRESH_SOURCE_FIRST_USE"
        or p["r97_qualified_nonreplaying_SHA"]!=PARENT
        or p["r97_frozen_blob"]!=r97.FREEZE_BLOB
        or p["r97_original_scored_SHA"]!=r97.ORIGINAL_R97_SPENT_SHA
        or p["r97_original_scored_run"]!=r97.ORIGINAL_R97_SPENT_RUN_ID
        or p["r97_frozen_status"]!="R97_UNSEEN_BINDING_CONVENTIONAL_SIGNAL_ONLY"
        or p["previous_scored_source_weights_reused"]
        or s["seeds"]!=list(SEEDS) or s["counts"]!=COUNT
        or s["salts"]!={**SALT,"TAG_NEGATIVE":NEGATIVE_SALT,
                         "BATCH":BATCH_SALT,"BOOTSTRAP":BOOTSTRAP_SALT,
                         "UNIT_ONLY":UNIT_SALT}
        or s["known_train_programs"]!=[list(i) for i in TRAIN_PROGRAMS]
        or s["never_trained_program"]!=list(COMPOSE_PROGRAM)
        or s["data_reuse"]
        or f["arms"]!=list(ARMS)
        or t["full_steps_each_arm"]!=STEPS or t["batch"]!=BATCH
        or t["whole_CPU_budget_seconds_per_seed"]!=MAX_SECONDS
        or e["seed_count"]!=2 or e["bootstrap_resamples"]!=2000
        or list(checks.values())!=[.5,.2,.2,.1,.8,.3,.1,.2,.65,.8]
        or e["independent_scientific_pass"]
        or not x["two_fresh_seed_jobs_only_in_existing_G4"]
        or not x["requires_full_G4_Linux_and_Windows_PASS_before_scored_source"]
        or not f["authority"]["scoped_CPU_testonly_diagnostic"]
        or any(bool(v) for k,v in f["authority"].items()
               if k!="scoped_CPU_testonly_diagnostic")):
        raise PermissionError("R98_FROZEN_PROTOCOL_OR_AUTHORITY_CHANGED")
    salts=set(s["salts"].values())
    if len(salts)!=len(s["salts"]) or salts & set(r97.frozen()["source"]["salts"].values()):
        raise RuntimeError("R98_SPENT_OR_DUPLICATED_SOURCE_SALT")
    return f


def program(panel:str,index:int):
    if panel not in SALT or not isinstance(index,int) or not 0<=index<COUNT[panel]:
        raise PermissionError("R98_PROGRAM_ONLY_FROZEN_NEW_SOURCE")
    if panel in ("FIT","VALID","DEV_SEEN"):
        return TRAIN_PROGRAMS[index%len(TRAIN_PROGRAMS)]
    return COMPOSE_PROGRAM


def role_binding(panel:str,index:int):
    if panel not in SALT or not isinstance(index,int) or not 0<=index<COUNT[panel]:
        raise PermissionError("R98_BINDING_ONLY_FROZEN_NEW_SOURCE")
    if panel in ("FIT","VALID","DEV_SEEN"):
        return ROLE_BINDINGS[index%len(ROLE_BINDINGS)]
    return HELDOUT_BINDING


def law_of(factors:np.ndarray,roles:tuple[int,int,int],op:tuple[int,int])->int:
    if (factors.shape!=(4,) or not np.all(np.isin(factors,(-1,1)))
        or len(roles)!=3 or len(set(roles))!=3
        or any(not 0<=k<4 for k in roles)
        or op not in ((0,0),(0,1),(1,0),(1,1))):
        raise ValueError("R98_INVALID_PRIVATE_CAUSAL_PROGRAM")
    if op[0]==0:
        b=int(factors[roles[1]] if factors[roles[0]]>0 else factors[roles[2]])
    else:
        b=int(np.prod(factors))
    return b if op[1]==0 else -b


def _chosen(seed:int,index:int):
    if seed not in SEEDS or not isinstance(index,int) or index<0:
        raise PermissionError("R98_UNFROZEN_FACTUAL_INTERVENTION")
    return r85.CALIBRATE if (index//4+SEEDS.index(seed))%2==0 else r85.EXPLOIT_POS


def balance_preflight():
    out={}
    for seed in SEEDS:
        out[str(seed)]={}
        for panel,n in COUNT.items():
            seen={}
            for i in range(n):
                key=(i%2,_chosen(seed,i))
                seen[key]=seen.get(key,0)+1
            if len(seen)!=4 or set(seen.values())!={n//4}:
                raise RuntimeError("R98_PUBLIC_COLD_BY_FACTUAL_ACTION_DESIGN_UNBALANCED")
            out[str(seed)][panel]={"joint_cold_and_action_cells":4,"count_per_cell":n//4}
    return out


def _admit(seed:int):
    frozen()
    if (seed not in SEEDS or os.getenv("GITHUB_ACTIONS")!="true"
        or os.getenv("GITHUB_EVENT_NAME")!="pull_request"
        or os.getenv("GITHUB_REPOSITORY")!="spoonman136668-ai/TTT-Hybrid-Lab"
        or os.getenv("GITHUB_HEAD_REF")!=BRANCH
        or os.getenv("GITHUB_JOB")!="r98-internal-program"
        or os.getenv("R98_SEED")!=str(seed)
        or os.getenv("RUNNER_OS")!="Linux"
        or os.getenv("RUNNER_ENVIRONMENT")!="github-hosted"
        or os.getenv("GITHUB_RUN_ATTEMPT")!="1"
        or "/merge" not in os.getenv("GITHUB_REF","")):
        raise PermissionError("R98_ORIGINAL_HOSTED_G4_FIRST_ATTEMPT_ONLY")
    sha=os.getenv("EXPECTED_CANDIDATE_SHA","")
    actual=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
                          capture_output=True,text=True,check=False,timeout=10)
    clean=subprocess.run(["git","diff","--quiet","HEAD","--"],cwd=ROOT,
                         capture_output=True,check=False,timeout=10)
    if len(sha)!=40 or actual.returncode or actual.stdout.strip()!=sha or clean.returncode:
        raise PermissionError("R98_NOT_EXACT_CLEAN_CANDIDATE_SOURCE")
    return sha


def unit_world(i:int):
    if not isinstance(i,int) or not 0<=i<12:
        raise PermissionError("R98_UNIT_ONLY_NONSCORING_SOURCE")
    rng=np.random.default_rng(np.random.SeedSequence([UNIT_SALT,i]))
    world=r85._make_synthetic_world(i,rng)
    op=TRAIN_PROGRAMS[i%3] if i<9 else COMPOSE_PROGRAM
    roles=ROLE_BINDINGS[i%3] if i<9 else HELDOUT_BINDING
    factors=rng.choice([-1,1],size=4).astype(np.int64)
    world["factors_a"]=factors
    world["r98_op"]=op
    world["r98_roles"]=roles
    world["law_a"]=law_of(factors,roles,op)
    return world,_chosen(SEEDS[0],i)


def _world(panel:str,seed:int,index:int,sha:str):
    if panel not in SALT:
        raise PermissionError("R98_PREVIOUS_SPENT_OR_PROTECTED_SOURCE")
    if seed not in SEEDS or not isinstance(index,int) or not 0<=index<COUNT[panel]:
        raise PermissionError("R98_UNREGISTERED_SEED_OR_INDEX")
    if (not sha or sha!=os.getenv("EXPECTED_CANDIDATE_SHA")
        or os.getenv("GITHUB_ACTIONS")!="true"
        or os.getenv("GITHUB_HEAD_REF")!=BRANCH
        or os.getenv("GITHUB_JOB")!="r98-internal-program"
        or os.getenv("R98_SEED")!=str(seed)
        or os.getenv("RUNNER_OS")!="Linux"
        or os.getenv("RUNNER_ENVIRONMENT")!="github-hosted"
        or os.getenv("GITHUB_RUN_ATTEMPT")!="1"):
        raise PermissionError("R98_FRESH_PHYSICAL_SOURCE_NOT_ADMITTED")
    rng=np.random.default_rng(np.random.SeedSequence([SALT[panel],seed,index]))
    world=r85._make_synthetic_world(index%r85.N,rng)
    factors=rng.choice([-1,1],size=4).astype(np.int64)
    op=program(panel,index)
    roles=role_binding(panel,index)
    world["factors_a"]=factors
    world["r98_op"]=op
    world["r98_roles"]=roles
    world["law_a"]=law_of(factors,roles,op)
    if panel=="OOD_SHIFT":
        world=r86._shift_public_world(world,index)
    return world,_chosen(seed,index)


def tag_for(roles):
    if len(roles)!=3 or len(set(roles))!=3 or any(not 0<=r<4 for r in roles):
        raise ValueError("R98_INVALID_PUBLIC_ROLE_ASSIGNMENT")
    tags=np.zeros(4,np.int64)
    for k,role in enumerate(roles,1):
        tags[role]=k
    return tags


def _public_view(world:dict,chosen:int):
    """Only PREACTION public task description, never private law or reward."""
    x,ret=r85._rollout(world,chosen)
    tags=tag_for(world["r98_roles"])
    op,neg=world["r98_op"]
    for t,(phase,role,marker) in world["probe"].items():
        if phase=="A" and t+1<=world["pivot"]:
            x[t+1,8]=float(int(tags[role])+4*op+8*neg)/15.
    source_times=sorted(t for t in world["probe"] if t<world["pivot"])
    return r86._compressed(x,source_times),float(ret)


def _raw(slots:np.ndarray)->np.ndarray:
    if slots.shape!=(5,13) or slots.dtype!=np.float32:
        raise ValueError("R98_PUBLIC_A_SLOTS_GEOMETRY")
    codes=np.rint(slots[:4,8]*15).astype(np.int64)
    tags=codes%4
    op=(codes//4)%2
    neg=(codes//8)%2
    if (set(tags)!={0,1,2,3} or len(set(op))!=1 or len(set(neg))!=1
        or not np.all((codes>=0)&(codes<16))
        or not np.array_equal(np.rint(slots[:4,9]*4).astype(int),np.arange(1,5))
        or not np.all(slots[:4,5]==1) or not np.all(slots[:4,11]==1)):
        raise RuntimeError("R98_PUBLIC_TAG_OR_OPERATOR_NOT_VALID")
    raw=np.zeros(14,np.float32)
    raw[:4]=slots[:4,10]
    raw[4:8]=tags/3.
    raw[8]=op[0]
    raw[9]=neg[0]
    raw[10]=slots[-1,r85.CALIBRATE]
    raw[11]=slots[-1,r85.EXPLOIT_POS]
    raw[12]=slots[-1,12]
    if (raw[10],raw[11]) not in ((1.,0.),(0.,1.)):
        raise RuntimeError("R98_UNKNOWN_PUBLIC_ACTUAL_ACTION")
    return raw


def _private_public_rule(raw:np.ndarray)->np.ndarray:
    if raw.ndim!=2 or raw.shape[1]!=14:
        raise ValueError("R98_RULE_ONLY_PUBLIC_14D")
    signs=np.where(raw[:,:4]>0,1,-1)
    tags=np.rint(raw[:,4:8]*3).astype(int)
    result=np.empty(len(raw),np.int8)
    for i in range(len(raw)):
        if set(tags[i])!={0,1,2,3}:
            raise RuntimeError("R98_PUBLIC_TAG_MISSING")
        roles={tag:int(np.flatnonzero(tags[i]==tag)[0]) for tag in (1,2,3)}
        base=(int(signs[i,roles[2]] if signs[i,roles[1]]>0 else signs[i,roles[3]])
              if raw[i,8]==0 else int(np.prod(signs[i])))
        result[i]=base if raw[i,9]==0 else -base
    return result


def _canonical(raw:np.ndarray)->np.ndarray:
    if raw.ndim!=2 or raw.shape[1]!=14:
        raise ValueError("R98_CANONICAL_PUBLIC_INPUT_NOT_14D")
    x=raw.copy()
    x[:,:10]=0.
    x[:,0]=_private_public_rule(raw)
    return x


def _blind(raw:np.ndarray)->np.ndarray:
    x=raw.copy()
    x[:,4:10]=0.
    return x


def _corrupt(raw:np.ndarray,seed:int,panel:str)->np.ndarray:
    if panel not in COUNT or seed not in SEEDS or raw.shape!=(COUNT[panel],14):
        raise PermissionError("R98_OPERATOR_DISRUPTION_OUTSIDE_FROZEN_PANEL")
    out=raw.copy()
    for i in range(len(raw)):
        rng=np.random.default_rng(np.random.SeedSequence(
            [NEGATIVE_SALT,seed,SALT[panel],i]))
        out[i,8:10]=rng.integers(0,2,size=2)
    if not np.array_equal(out[:,:8],raw[:,:8]) or not np.array_equal(
            out[:,10:],raw[:,10:]):
        raise RuntimeError("R98_NEGATIVE_CHANGED_FACTUAL_PROBES_OR_ACTION")
    return out


def _audit(law:np.ndarray,raw:np.ndarray,corrupted:np.ndarray,
           chosen:np.ndarray|None):
    if law.shape!=(len(raw),) or raw.shape!=(len(raw),14) or corrupted.shape!=raw.shape:
        raise ValueError("R98_EVALUATOR_AUDIT_SHAPE")
    rates={}
    for pivot in np.unique(raw[:,12]):
        selected=raw[:,12]==pivot
        rate=float(np.mean(law[selected]>0))
        rates[str(float(pivot))]=rate
        if not .35<=rate<=.65:
            raise RuntimeError("R98_PRIVATE_LAW_PUBLIC_PIVOT_SOURCE_INVALID")
    if chosen is not None:
        for action in (r85.CALIBRATE,r85.EXPLOIT_POS):
            v=float(np.mean(law[chosen==action]>0))
            if not .35<=v<=.65:
                raise RuntimeError("R98_PRIVATE_LAW_FACTUAL_ACTION_SOURCE_INVALID")
    same=float(np.mean(_private_public_rule(corrupted)==law))
    return {"law_positive_by_public_pivot":rates,
            "corrupted_public_program_implied_law_agreement":same,
            "negative_is_instruction_disruption_not_independent_null":True,
            "valid":True}
