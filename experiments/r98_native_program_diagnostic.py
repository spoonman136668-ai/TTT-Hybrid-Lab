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


def _factual(panel:str,seed:int,sha:str):
    if panel not in ("FIT","VALID"):
        raise PermissionError("R98_FIT_CANNOT_ACCESS_HELDOUT_OR_OLD_SCORED_PANEL")
    n=COUNT[panel]
    slots=np.empty((n,5,13),np.float32)
    features=np.empty((n,14),np.float32)
    returns=np.empty(n,np.float32)
    law=np.empty(n,np.int8)
    actions=np.empty(n,np.int8)
    for i in range(n):
        world,chosen=_world(panel,seed,i,sha)
        public,ret=_public_view(world,chosen)
        observed=r88._public_slots(public)
        slots[i],features[i]=observed,_raw(observed)
        returns[i],law[i],actions[i]=ret,world["law_a"],chosen
    corrupted=_corrupt(features,seed,panel)
    audit=_audit(law,features,corrupted,actions)
    return {"slots":slots,"raw":features,"blind":_blind(features),
            "canonical":_canonical(features),"negative":corrupted,
            "target":returns,"logged_action":actions,
            "audit":audit,"only_one_observed_factual_return":True}


def _scored(panel:str,seed:int,sha:str):
    if panel not in ("DEV_SEEN","DEV_COMPOSE","OOD_SHIFT","ROLE_REASSIGN"):
        raise PermissionError("R98_EVALUATOR_ONLY_FRESH_POSTTRAIN_PANEL")
    n=COUNT[panel]
    slots0=np.empty((n,5,13),np.float32);slots2=np.empty_like(slots0)
    raw0=np.empty((n,14),np.float32);raw2=np.empty_like(raw0)
    reward0=np.empty(n,np.float64);reward2=np.empty_like(reward0)
    law=np.empty(n,np.int8);cold=np.empty(n,bool)
    counter=None
    if panel=="ROLE_REASSIGN":
        counter={"slots0":np.empty_like(slots0),"slots2":np.empty_like(slots2),
                 "raw0":np.empty_like(raw0),"raw2":np.empty_like(raw2),
                 "reward0":np.empty_like(reward0),"reward2":np.empty_like(reward2),
                 "law":np.empty_like(law),"cold":np.empty_like(cold)}
    for i in range(n):
        world,_=_world(panel,seed,i,sha)
        p0,r0=_public_view(world,r85.CALIBRATE)
        p2,r2=_public_view(world,r85.EXPLOIT_POS)
        if (not np.array_equal(p0[:-1],p2[:-1])
            or not np.array_equal(p0[-1,4:],p2[-1,4:])
            or p0[-1,0]!=1. or p2[-1,2]!=1.):
            raise RuntimeError("R98_PUBLIC_PREFIX_CONTAINS_ALTERNATE_ACTION")
        slots0[i],slots2[i]=r88._public_slots(p0),r88._public_slots(p2)
        raw0[i],raw2[i]=_raw(slots0[i]),_raw(slots2[i])
        reward0[i],reward2[i]=r0,r2
        law[i],cold[i]=world["law_a"],bool(i%2)
        if counter is not None:
            swapped=copy.deepcopy(world)
            swapped["r98_op"]=(0,1)
            swapped["law_a"]=law_of(
                swapped["factors_a"],swapped["r98_roles"],swapped["r98_op"])
            q0,t0=_public_view(swapped,r85.CALIBRATE)
            q2,t2=_public_view(swapped,r85.EXPLOIT_POS)
            if not np.array_equal(p0[:,9:12],q0[:,9:12]):
                raise RuntimeError("R98_PROGRAM_SWITCH_CHANGED_PHYSICAL_NOISY_A_PROBES")
            counter["slots0"][i],counter["slots2"][i]=r88._public_slots(q0),r88._public_slots(q2)
            counter["raw0"][i],counter["raw2"][i]=_raw(counter["slots0"][i]),_raw(counter["slots2"][i])
            counter["reward0"][i],counter["reward2"][i]=t0,t2
            counter["law"][i],counter["cold"][i]=swapped["law_a"],bool(i%2)
    corrupted0=_corrupt(raw0,seed,panel)
    corrupted2=_corrupt(raw2,seed,panel)
    audit=_audit(law,raw0,corrupted0,None)
    result={"slots0":slots0,"slots2":slots2,"raw0":raw0,"raw2":raw2,
            "blind0":_blind(raw0),"blind2":_blind(raw2),
            "canonical0":_canonical(raw0),"canonical2":_canonical(raw2),
            "negative0":corrupted0,"negative2":corrupted2,
            "reward0":reward0,"reward2":reward2,
            "law":law,"cold":cold,"audit":audit}
    if counter is not None:
        counter["blind0"]=_blind(counter["raw0"])
        counter["blind2"]=_blind(counter["raw2"])
        counter["canonical0"]=_canonical(counter["raw0"])
        counter["canonical2"]=_canonical(counter["raw2"])
        counter["negative0"]=_corrupt(counter["raw0"],seed,panel)
        counter["negative2"]=_corrupt(counter["raw2"],seed,panel)
        result["counter"]=counter
    return result


class Flat14Value(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(14,64),nn.GELU(),
                               nn.Linear(64,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,raw):
        if raw.ndim!=2 or raw.shape[1]!=14:
            raise RuntimeError("R98_MLP_INPUT_NOT_PUBLIC_14D")
        return self.net(raw)[:,0]


class RoleProgramContext(nn.Module):
    """Shared learned per-role public-cue/opcode encoder, 32 native channels."""
    def __init__(self):
        super().__init__()
        self.event=nn.Sequential(nn.Linear(5,32),nn.GELU(),
                                 nn.Linear(32,32),nn.GELU())
        self.program=nn.Sequential(nn.Linear(34,64),nn.GELU(),
                                   nn.Linear(64,32))
    def forward(self,raw):
        if raw.ndim!=2 or raw.shape[1]!=14:
            raise RuntimeError("R98_CONTEXT_INPUT_NOT_PUBLIC_14D")
        tags=torch.round(raw[:,4:8]*3).long()
        if bool((tags<0).any()) or bool((tags>3).any()):
            raise RuntimeError("R98_PUBLIC_TAG_OUTSIDE_4_ROLE_VOCABULARY")
        onehot=F.one_hot(tags,num_classes=4).to(raw.dtype)
        events=torch.cat((raw[:,:4].unsqueeze(2),onehot),dim=2)
        seen=self.event(events).sum(dim=1)
        return self.program(torch.cat((seen,raw[:,8:10]),dim=1))


class RoleProgramValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.context=RoleProgramContext()
        self.head=nn.Sequential(nn.Linear(35,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,raw):
        ctx=self.context(raw)
        return self.head(torch.cat((ctx,raw[:,10:13]),dim=1))[:,0]


class FlatStemContext(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(14,64),nn.GELU(),nn.Linear(64,32))
    def forward(self,raw):
        return self.net(raw)


def _native_trunk(core,stemmed):
    """Run original native SSM→causal attention→TTT→depth→head unchanged."""
    z=stemmed
    if core.ssm is not None:
        z=core.ssm(z)
    if core.attn is not None:
        z=core.attn(z)
    if core.ttt is not None:
        z=core.ttt(z)
    if core.depth is not None:
        z=core.depth(z)
    return core.head(z)[:,-1,0]


class NativeStemInjection(nn.Module):
    """Inside last pre-SSM native stem state, NOT after the native value head."""
    def __init__(self,core,adapter):
        super().__init__()
        self.core=core
        self.adapter=adapter
    def forward(self,slots,raw):
        stemmed=self.core.stem(slots)
        context=self.adapter(raw)
        if context.shape!=(len(slots),stemmed.shape[-1]):
            raise RuntimeError("R98_INTERNAL_NATIVE_STEM_CONTEXT_DIMENSION_DRIFT")
        # No in-place autograd mutation; only the pivot native stem receives
        # an extra lawful public role/program context before causal modules.
        mixed=torch.cat((stemmed[:,:-1],stemmed[:,-1:]+context[:,None,:]),dim=1)
        return _native_trunk(self.core,mixed)


class NativeExternalSidecar(nn.Module):
    def __init__(self,core):
        super().__init__()
        self.core=core
        self.encoder=RoleProgramContext()
        self.head=nn.Sequential(nn.Linear(35,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,slots,raw):
        extra=self.head(torch.cat((self.encoder(raw),raw[:,10:13]),dim=1))[:,0]
        return self.core(slots)[:,-1,0]+extra


def _models(seed:int):
    if seed not in SEEDS:
        raise PermissionError("R98_NOT_FROZEN_MODEL_SEED")
    result={}
    result[NATIVE_RAW]=r95._native(seed)
    torch.manual_seed(seed+61)
    result[NATIVE_FLAT]=NativeStemInjection(r95._native(seed),FlatStemContext())
    torch.manual_seed(seed+62)
    result[NATIVE_INTERNAL]=NativeStemInjection(r95._native(seed),RoleProgramContext())
    torch.manual_seed(seed+62)
    result[NATIVE_SIDECAR]=NativeExternalSidecar(r95._native(seed))
    torch.manual_seed(seed+71)
    result[MLP]=Flat14Value()
    torch.manual_seed(seed+72)
    result[SET]=RoleProgramValue()
    torch.manual_seed(seed+73)
    result[GRU]=r81.GRU(64)
    torch.manual_seed(seed+74)
    result[UPPER]=Flat14Value()
    torch.manual_seed(seed+72)
    result[NEGATIVE]=RoleProgramValue()
    cores=[result[NATIVE_RAW],result[NATIVE_FLAT].core,
           result[NATIVE_INTERNAL].core,result[NATIVE_SIDECAR].core]
    states=[dict(c.named_parameters()) for c in cores]
    if any(states[0].keys()!=other.keys() for other in states[1:]) or any(
        not torch.equal(states[0][k],other[k])
        for other in states[1:] for k in states[0]):
        raise RuntimeError("R98_INTERNAL_AND_EXTERNAL_ORIGINAL_NATIVE_CORES_DIFFER")
    # Same model+initial seed for program-disrupted control.
    a,b=[dict(result[name].named_parameters()) for name in (SET,NEGATIVE)]
    if a.keys()!=b.keys() or any(not torch.equal(a[k],b[k]) for k in a):
        raise RuntimeError("R98_MATCHED_PROGRAM_NEGATIVE_DIFFERENT_INITIAL_TENSORS")
    return result


def _arrays(data,arm,suffix=""):
    key=("blind" if arm==MLP and False else
         "canonical" if arm==UPPER else "negative" if arm==NEGATIVE else "raw")
    return data["slots"+suffix],data[key+suffix]


def _forward(model,arm,slots,raw):
    if arm in (NATIVE_RAW,GRU):
        y=model(slots)[:,-1,0]
    elif arm in (NATIVE_FLAT,NATIVE_INTERNAL,NATIVE_SIDECAR):
        y=model(slots,raw)
    else:
        y=model(raw)
    if y.shape!=(len(slots),):
        raise RuntimeError("R98_NATIVE_OR_CONVENTIONAL_RETURN_SHAPE")
    return y


def _predict(model,arm,data,suffix=""):
    slots,raw=_arrays(data,arm,suffix)
    model.eval()
    found=[]
    with torch.no_grad():
        for i in range(0,len(raw),32):
            x=torch.from_numpy(slots[i:i+32])
            f=torch.from_numpy(raw[i:i+32])
            found.extend(_forward(model,arm,x,f).tolist())
    model.train()
    arr=np.asarray(found,np.float64)
    if arr.shape!=(len(raw),) or not np.isfinite(arr).all():
        raise RuntimeError("R98_NONFINITE_OR_PARTIAL_VALUE_PREDICTION")
    return arr


def _train(model,arm,fit,valid,batches):
    slots,raw=_arrays(fit,arm)
    x=torch.from_numpy(slots)
    f=torch.from_numpy(raw)
    target=torch.from_numpy(fit["target"])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    before=float(np.mean((_predict(model,arm,valid)-valid["target"])**2))
    tic=time.perf_counter()
    for ix in batches:
        loss=F.mse_loss(_forward(model,arm,x[ix],f[ix]),target[ix])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("R98_NONFINITE_FACTUAL_RETURN_TRAINING_LOSS")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
    elapsed=time.perf_counter()-tic
    train=float(np.mean((_predict(model,arm,fit)-fit["target"])**2))
    val=float(np.mean((_predict(model,arm,valid)-valid["target"])**2))
    if not all(math.isfinite(v) for v in (elapsed,before,train,val)):
        raise RuntimeError("R98_NONFINITE_COMPLETED_FROZEN_ARM")
    return {"updates":len(batches),"factual_exposures":len(batches)*BATCH,
            "params":sum(p.numel() for p in model.parameters()),
            "initial_VALID_MSE":before,"final_FIT_MSE":train,
            "final_VALID_MSE":val,"fit_wall_seconds":elapsed}


def _score(models,data):
    actions={};metrics={};returns={}
    for name,model in models.items():
        v0=_predict(model,name,data,"0")
        v2=_predict(model,name,data,"2")
        actions[name]=np.where(v0>v2,r85.CALIBRATE,r85.EXPLOIT_POS)
    public_sign=_private_public_rule(data["raw0"])
    public_action=np.where(public_sign>0,r85.CALIBRATE,r85.EXPLOIT_POS)
    true_action=np.where(data["law"]>0,r85.CALIBRATE,r85.EXPLOIT_POS)
    actions["PUBLIC_PROGRAM_RULE_EVALUATOR_ONLY"]=public_action
    actions["PRIVATE_PHYSICAL_LAW_EVALUATOR_ONLY"]=true_action
    actions["ALWAYS_CALIBRATE"]=np.full(len(public_action),r85.CALIBRATE)
    actions["ALWAYS_EXPLOIT_POS"]=np.full(len(public_action),r85.EXPLOIT_POS)
    oracle=np.maximum(data["reward0"],data["reward2"])
    for name,pick in actions.items():
        realized=np.where(pick==r85.CALIBRATE,data["reward0"],data["reward2"])
        returns[name]=realized
        metrics[name]={"mean_realized_reward":float(realized.mean()),
                       "cold_mean_reward":float(realized[data["cold"]].mean()),
                       "oracle_regret":float((oracle-realized).mean()),
                       "public_program_agreement":float(np.mean(pick==public_action)),
                       "true_physical_law_agreement":float(np.mean(pick==true_action))}
    metrics["PHYSICAL_PAIRED_ORACLE_EVALUATOR_ONLY"]={
        "mean_realized_reward":float(oracle.mean()),
        "cold_mean_reward":float(oracle[data["cold"]].mean()),
        "oracle_regret":0.}
    return {"metrics":metrics,"actions":actions,"returns":returns}


def _bootstrap_lower(a,b,seed):
    if a.shape!=(COUNT["DEV_COMPOSE"],) or b.shape!=a.shape:
        raise RuntimeError("R98_FROZEN_PAIRED_BOOTSTRAP_SHAPE")
    rng=np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SALT,seed]))
    rows=rng.integers(0,len(a),size=(2000,len(a)))
    return float(np.quantile((a-b)[rows].mean(axis=1),.025))


def original_ci(seed:int):
    sha=_admit(seed)
    preflight=balance_preflight()
    torch.set_num_threads(1)
    path=ROOT/f"local_runs/r98_internal_program_seed{seed}.json"
    if path.exists():
        raise FileExistsError("R98_ORIGINAL_NEW_SCIENCE_REPLAY_OR_UNCERTAIN_EFFECT_DENIED")
    started=time.perf_counter()
    result={"schema":SCHEMA,"seed":seed,"candidate_source_SHA":sha,
            "freeze_blob":FREEZE_BLOB,"R97_parent_SHA":PARENT,
            "platform":platform.platform(),"torch":str(torch.__version__),
            "preflight":preflight[str(seed)],
            "status":"R98_ENGINEERING_OR_RESOURCE_INCOMPLETE_NO_SCIENTIFIC_VERDICT",
            "trained":{},"source_audits":{},
            "all_nine_frozen_before_new_DEV_OOD":False,
            "new_DEV_COMPOSE_OOD_ROLE_REASSIGN_first_use":False,
            "older_scored_source_or_weights_used":False,
            "scientific_pass":False}
    try:
        fit=_factual("FIT",seed,sha)
        valid=_factual("VALID",seed,sha)
        if not fit["only_one_observed_factual_return"]:
            raise RuntimeError("R98_ILLEGAL_UNCHOSEN_BRANCH_TRAINING_TARGET")
        result["source_audits"]["FIT"]=fit["audit"]
        result["source_audits"]["VALID"]=valid["audit"]
        rng=np.random.default_rng(np.random.SeedSequence([BATCH_SALT,seed]))
        batches=rng.integers(0,COUNT["FIT"],size=(STEPS,BATCH))
        models=_models(seed)
        for arm in ARMS:
            result["trained"][arm]=_train(models[arm],arm,fit,valid,batches)
            if time.perf_counter()-started>MAX_SECONDS:
                raise TimeoutError("R98_PRESET_CPU_RESOURCE_CEILING_EXCEEDED")
        if set(result["trained"])!=set(ARMS):
            raise RuntimeError("R98_FROZEN_ARM_INCOMPLETE")
        result["all_nine_frozen_before_new_DEV_OOD"]=True
        # First fresh heldout / shifted world exists ONLY after every arm's
        # full factual-return optimizer training and all weights freezing.
        result["new_DEV_COMPOSE_OOD_ROLE_REASSIGN_first_use"]=True
        seen=_scored("DEV_SEEN",seed,sha)
        dev=_scored("DEV_COMPOSE",seed,sha)
        ood=_scored("OOD_SHIFT",seed,sha)
        switched=_scored("ROLE_REASSIGN",seed,sha)
        result["source_audits"].update({
            "DEV_SEEN":seen["audit"],"DEV_COMPOSE":dev["audit"],
            "OOD_SHIFT":ood["audit"],"ROLE_REASSIGN":switched["audit"]})
        scored_seen=_score(models,seen)
        scored_dev=_score(models,dev)
        scored_ood=_score(models,ood)
        before=_score(models,switched)
        after=_score(models,switched["counter"])
        dm=scored_dev["metrics"];om=scored_ood["metrics"]
        changed=switched["law"]!=switched["counter"]["law"]
        if not changed.any():
            raise RuntimeError("R98_PHYSICAL_PROGRAM_SWITCH_DID_NOT_CHANGE_LAW")
        native_before=before["actions"][NATIVE_INTERNAL]
        native_after=after["actions"][NATIVE_INTERNAL]
        physical_before=np.where(switched["law"]>0,r85.CALIBRATE,r85.EXPLOIT_POS)
        physical_after=np.where(switched["counter"]["law"]>0,r85.CALIBRATE,r85.EXPLOIT_POS)
        conventional=max(dm[x]["mean_realized_reward"] for x in (
            MLP,SET,GRU,UPPER))
        measures={
            "internal_minus_original_DEV":dm[NATIVE_INTERNAL]["mean_realized_reward"]-dm[NATIVE_RAW]["mean_realized_reward"],
            "internal_minus_flat_stem_DEV":dm[NATIVE_INTERNAL]["mean_realized_reward"]-dm[NATIVE_FLAT]["mean_realized_reward"],
            "internal_minus_external_sidecar_DEV":dm[NATIVE_INTERNAL]["mean_realized_reward"]-dm[NATIVE_SIDECAR]["mean_realized_reward"],
            "internal_minus_strongest_conventional_DEV":dm[NATIVE_INTERNAL]["mean_realized_reward"]-conventional,
            "internal_true_law_accuracy_DEV":dm[NATIVE_INTERNAL]["true_physical_law_agreement"],
            "internal_minus_original_shifted_OOD_cold":om[NATIVE_INTERNAL]["cold_mean_reward"]-om[NATIVE_RAW]["cold_mean_reward"],
            "internal_minus_original_paired_bootstrap_lower95":_bootstrap_lower(
                scored_dev["returns"][NATIVE_INTERNAL],scored_dev["returns"][NATIVE_RAW],seed),
            "program_switch_physical_law_change_fraction":float(changed.mean()),
            "program_switch_internal_action_flips_when_law_changes":float(np.mean(
                native_before[changed]!=native_after[changed])),
            "program_switch_internal_true_law_accuracy_both":float((
                (native_before==physical_before).sum()+
                (native_after==physical_after).sum())/(2*len(changed)))}
        f=frozen()["evaluation"]
        t=f["mechanism_checks"]
        checks={
          "beats_original_native":measures["internal_minus_original_DEV"]>=t["composed_internal_minus_original_native_DEV_reward_min"],
          "beats_flat_native_stem":measures["internal_minus_flat_stem_DEV"]>=t["composed_internal_minus_flat_native_stem_DEV_reward_min"],
          "beats_same_role_encoder_external_sidecar":measures["internal_minus_external_sidecar_DEV"]>=t["composed_internal_minus_external_role_sidecar_DEV_reward_min"],
          "beats_strongest_conventional":measures["internal_minus_strongest_conventional_DEV"]>=t["composed_internal_vs_strongest_conventional_DEV_reward_margin_min"],
          "accurate_on_new_composition":measures["internal_true_law_accuracy_DEV"]>=t["composed_internal_true_law_accuracy_min"],
          "shifted_cold_gain":measures["internal_minus_original_shifted_OOD_cold"]>=t["composed_internal_vs_original_native_shifted_cold_reward_min"],
          "paired_bootstrap_positive":measures["internal_minus_original_paired_bootstrap_lower95"]>=t["composed_internal_minus_original_native_DEV_bootstrap_lower95_min"],
          "physical_program_switch_changes_law":measures["program_switch_physical_law_change_fraction"]>=t["program_switch_true_law_change_fraction_min"],
          "model_changes_action_with_program":measures["program_switch_internal_action_flips_when_law_changes"]>=t["program_switch_internal_action_change_on_true_change_min"],
          "two_programs_true_law_accuracy":measures["program_switch_internal_true_law_accuracy_both"]>=t["program_switch_internal_true_law_accuracy_both_min"]}
        result["DEV_SEEN"]=scored_seen["metrics"]
        result["DEV_COMPOSE"]=dm
        result["OOD_SHIFT"]=om
        result["ROLE_REASSIGN"]={"original":before["metrics"],
            "changed_program":after["metrics"],
            "true_law_changed_worlds":int(changed.sum())}
        result["metrics"]=measures
        result["frozen_mechanism_checks"]={k:bool(v) for k,v in checks.items()}
        result["status"]=f["statuses"]["both"] if all(checks.values()) else f["statuses"]["no_go"]
    except BaseException as e:
        result["error_type"]=type(e).__name__
        result["error_message"]=str(e)
        result["status"]=(frozen()["evaluation"]["statuses"]["invalid"]
            if "R98_PRIVATE_LAW_" in str(e) or "R98_NEGATIVE_CHANGED" in str(e)
            else frozen()["evaluation"]["statuses"]["incomplete"])
        raise
    finally:
        result["whole_process_seconds"]=time.perf_counter()-started
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("x",encoding="utf-8") as f:
            json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
            f.write("\n")
    return result


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",required=True,type=int,choices=SEEDS)
    args=parser.parse_args()
    x=original_ci(args.seed)
    print(json.dumps({"seed":args.seed,"status":x["status"],
                      "arms":len(x["trained"]),
                      "scored":x["new_DEV_COMPOSE_OOD_ROLE_REASSIGN_first_use"],
                      "scientific_pass":False},sort_keys=True))
