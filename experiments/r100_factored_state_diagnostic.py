"""R100 — factorized PUBLIC causal-program prior versus factual-return-sign feedback.

R100 TEST-only CPU research. The MUX/PARITY/NEGATION algebra and public role
canonicalization are EXPLICITLY ENGINEERED structural prior. This is NOT
new rule discovery or a production native-model promotion. The auxiliary
feedback comes solely from the ONE actual selected action's factual return.
Never reopen R85–R99 scored source, saved weights, hidden law or oracle.
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
from experiments import r88_causal_event_access as r88
from experiments import r95_learned_relation_diagnostic as r95
from experiments import r98_native_program_diagnostic as r98
from experiments import r99_microfeedback_diagnostic as r99

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r100_factored_state_freeze.json"
FREEZE_BLOB="9c9ffd09fadc95d9f94662560480d0ba6e4be54b"
SCHEMA="ttt-hybrid.r100-factored-native-causal-state.cpu-testonly.v1"
PARENT="772217b0dbf750e0749999fc560fec8f4b4bb7f1"
BRANCH="research/r100-factored-native-causal-state-20261116"
SEEDS=(1002311,1002323)
SALT={"FIT":25410111,"VALID":25410211,"DEV_SEEN":25410311,
      "DEV_COMPOSE":25410411,"OOD_SHIFT":25410511,
      "PROGRAM_SWITCH":25410611}
COUNT={"FIT":2048,"VALID":256,"DEV_SEEN":256,"DEV_COMPOSE":256,
       "OOD_SHIFT":256,"PROGRAM_SWITCH":256}
RETURN_SHUFFLE_SALT=25410711
BATCH_SALT=25410811
BOOTSTRAP_SALT=25410911
UNIT_SALT=25419991
TRAIN_PROGRAMS=((0,0),(0,1),(1,0))
COMPOSE_PROGRAM=(1,1)
TRAIN_BINDINGS=((0,1,2),(1,2,3),(2,3,0))
HELDOUT_BINDING=(3,0,1)
ARMS=("NATIVE32_ORIGINAL","NATIVE32_UNSTRUCTURED_STEM",
      "NATIVE32_FACTORED_NO_AUX","NATIVE32_FACTORED_FACTUAL_SIGN_AUX",
      "NATIVE32_FACTORED_SHUFFLED_SIGN_AUX",
      "NATIVE32_FACTORED_OPCODE_MASK",
      "NATIVE32_FACTORED_EXTERNAL_VALUE",
      "MLP64_RAW_PUBLIC","ROLE_SET_PUBLIC","GRU64_PUBLIC_EVENTS",
      "MLP64_ENGINEERED_PUBLIC_RULE_UPPER")
(NATIVE_RAW,UNSTRUCTURED,NOAUX,AUX,SHUFFLED,OPMASK,EXTERNAL,
 MLP,SET,GRU,UPPER)=ARMS
FACTORED_INTERNAL_ARMS=(NOAUX,AUX,SHUFFLED,OPMASK)
STEPS=3072
BATCH=16
LAMBDA=.15
MAX_SECONDS=300


def frozen():
    if r85.git_blob(FREEZE.read_bytes())!=FREEZE_BLOB:
        raise RuntimeError("R100_PREIMPLEMENTATION_FROZEN_GIT_BLOB_DRIFT")
    r99.frozen()
    f=json.loads(FREEZE.read_text(encoding="utf-8"))
    p,s,t,e,x=f["parent"],f["source"],f["training"],f["evaluation"],f["execution"]
    if (f["schema"]!=SCHEMA
        or f["stage"]!="PROSPECTIVE_FREEZE_BEFORE_R100_CODE_TESTS_OR_SOURCE_FIRST_USE"
        or p["qualified_r99_sealed_SHA"]!=PARENT
        or p["r99_freeze_blob"]!=r99.FREEZE_BLOB
        or p["r99_spent_SHA"]!=r99.ORIGINAL_R99_SPENT_SHA
        or p["r99_spent_run"]!=r99.ORIGINAL_R99_SPENT_RUN_ID
        or p["r99_science_status"]!="R99_INTERNAL_MICROFEEDBACK_NO_GO"
        or p["prior_scored_sources_or_weights_reused"]
        or s["seeds"]!=list(SEEDS)
        or s["salts"]!={**SALT,"RETURN_SHUFFLE":RETURN_SHUFFLE_SALT,
                         "BATCH":BATCH_SALT,"BOOTSTRAP":BOOTSTRAP_SALT,
                         "UNIT_ONLY":UNIT_SALT}
        or s["counts"]!=COUNT
        or s["train_programs"]!=[list(v) for v in TRAIN_PROGRAMS]
        or s["heldout_program"]!=list(COMPOSE_PROGRAM)
        or s["train_role_bindings"]!=[list(v) for v in TRAIN_BINDINGS]
        or s["heldout_roles"]!=list(HELDOUT_BINDING)
        or s["older_scored_sources_reused"]
        or f["arms"]!=list(ARMS)
        or t["steps_each_arm"]!=STEPS
        or t["batch"]!=BATCH
        or t["outcome_aux_weight"]!=LAMBDA
        or t["whole_CPU_budget_seconds_per_seed"]!=MAX_SECONDS
        or e["independent_seeds"]!=2 or e["bootstrap_draws"]!=2000
        or list(e["structural_checks"].values())!=[.5,.4,.75,.5,.8,.4,.1,.2,.7,.8]
        or list(e["feedback_checks"].values())!=[.2,.2,.08,.15]
        or list(e["native_specific_checks"].values())!=[.15,.15]
        or e["independent_scientific_pass"]
        or not x["only_two_branch_scoped_seed_jobs_appended"]
        or not x["requires_full_G4_Linux_Windows_before_fresh_scoring"]
        or not f["authority"]["CPU_TEST_ONLY_scoped_original_first_use"]
        or any(bool(v) for k,v in f["authority"].items()
               if k!="CPU_TEST_ONLY_scoped_original_first_use")):
        raise PermissionError("R100_FROZEN_HYPOTHESIS_SOURCE_OR_OWNER_SCOPE_DRIFT")
    new=set(s["salts"].values())
    older=set(r99.frozen()["source"]["salts"].values())
    if len(new)!=len(s["salts"]) or bool(new&older):
        raise RuntimeError("R100_SPENT_OR_DUPLICATE_SOURCE_SALT")
    return f


def program(panel,index):
    if panel not in SALT or not isinstance(index,int) or not 0<=index<COUNT[panel]:
        raise PermissionError("R100_PROGRAM_OUTSIDE_NEW_FROZEN_COHORT")
    return TRAIN_PROGRAMS[index%3] if panel in ("FIT","VALID","DEV_SEEN") else COMPOSE_PROGRAM


def roles(panel,index):
    if panel not in SALT or not isinstance(index,int) or not 0<=index<COUNT[panel]:
        raise PermissionError("R100_ROLES_OUTSIDE_NEW_FROZEN_COHORT")
    return TRAIN_BINDINGS[index%3] if panel in ("FIT","VALID","DEV_SEEN") else HELDOUT_BINDING


def chosen(seed,index):
    if seed not in SEEDS or not isinstance(index,int) or index<0:
        raise PermissionError("R100_UNREGISTERED_FACTUAL_ACTION")
    return r85.CALIBRATE if (index//4+SEEDS.index(seed))%2==0 else r85.EXPLOIT_POS


def design_preflight():
    report={}
    for seed in SEEDS:
        report[str(seed)]={}
        for panel,n in COUNT.items():
            buckets={}
            for i in range(n):
                key=(i%2,chosen(seed,i))
                buckets[key]=buckets.get(key,0)+1
            if len(buckets)!=4 or set(buckets.values())!={n//4}:
                raise RuntimeError("R100_COLD_ACTION_STRATA_UNBALANCED")
            report[str(seed)][panel]={"four_action_time_cells":4,"per_cell":n//4}
    return report


def _admit(seed):
    frozen()
    if (seed not in SEEDS or os.getenv("GITHUB_ACTIONS")!="true"
        or os.getenv("GITHUB_EVENT_NAME")!="pull_request"
        or os.getenv("GITHUB_REPOSITORY")!="spoonman136668-ai/TTT-Hybrid-Lab"
        or os.getenv("GITHUB_HEAD_REF")!=BRANCH
        or os.getenv("GITHUB_JOB")!="r100-factored-native"
        or os.getenv("R100_SEED")!=str(seed)
        or os.getenv("RUNNER_OS")!="Linux"
        or os.getenv("RUNNER_ENVIRONMENT")!="github-hosted"
        or os.getenv("GITHUB_RUN_ATTEMPT")!="1"
        or "/merge" not in os.getenv("GITHUB_REF","")):
        raise PermissionError("R100_GITHUB_HOSTED_CPU_ORIGINAL_FIRST_USE_ONLY")
    sha=os.getenv("EXPECTED_CANDIDATE_SHA","")
    head=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
                        text=True,capture_output=True,timeout=10,check=False)
    clean=subprocess.run(["git","diff","--quiet","HEAD","--"],cwd=ROOT,
                         capture_output=True,timeout=10,check=False)
    if len(sha)!=40 or head.returncode or head.stdout.strip()!=sha or clean.returncode:
        raise PermissionError("R100_EXACT_SHA_OR_CLEAN_SOURCE_MISMATCH")
    return sha


def unit_world(i):
    if not isinstance(i,int) or not 0<=i<12:
        raise PermissionError("R100_INDEPENDENT_UNIT_FIXTURE_ONLY")
    rng=np.random.default_rng(np.random.SeedSequence([UNIT_SALT,i]))
    world=r85._make_synthetic_world(i,rng)
    op=TRAIN_PROGRAMS[i%3] if i<9 else COMPOSE_PROGRAM
    bind=TRAIN_BINDINGS[i%3] if i<9 else HELDOUT_BINDING
    factors=rng.choice([-1,1],size=4).astype(np.int64)
    world["factors_a"]=factors
    world["r98_op"]=op
    world["r98_roles"]=bind
    world["law_a"]=r98.law_of(factors,bind,op)
    return world,chosen(SEEDS[0],i)


def _world(panel,seed,index,sha):
    if panel not in SALT:
        raise PermissionError("R100_OLD_SPENT_OR_PROTECTED_SOURCE_FORBIDDEN")
    if seed not in SEEDS or not isinstance(index,int) or not 0<=index<COUNT[panel]:
        raise PermissionError("R100_UNFROZEN_SEED_OR_WORLD_INDEX")
    if (not sha or sha!=os.getenv("EXPECTED_CANDIDATE_SHA")
        or os.getenv("GITHUB_ACTIONS")!="true"
        or os.getenv("GITHUB_HEAD_REF")!=BRANCH
        or os.getenv("GITHUB_JOB")!="r100-factored-native"
        or os.getenv("R100_SEED")!=str(seed)
        or os.getenv("RUNNER_OS")!="Linux"
        or os.getenv("RUNNER_ENVIRONMENT")!="github-hosted"
        or os.getenv("GITHUB_RUN_ATTEMPT")!="1"):
        raise PermissionError("R100_NEW_SCIENCE_SOURCE_NOT_ADMITTED")
    rng=np.random.default_rng(np.random.SeedSequence([SALT[panel],seed,index]))
    world=r85._make_synthetic_world(index%r85.N,rng)
    factors=rng.choice([-1,1],size=4).astype(np.int64)
    op=program(panel,index)
    bind=roles(panel,index)
    world["factors_a"]=factors
    world["r98_op"]=op
    world["r98_roles"]=bind
    world["law_a"]=r98.law_of(factors,bind,op)
    if panel=="OOD_SHIFT":
        world=r86._shift_public_world(world,index)
    return world,chosen(seed,index)


def _public(world,action):
    """Only R98/R99 pure public observation, never old scored source."""
    return r99._observed(world,action)


def _mask_program(slots,raw):
    """Ablate public program bits BOTH in raw features and native event rows."""
    if raw.ndim!=2 or raw.shape[1]!=14 or slots.ndim!=3 or slots.shape[1:]!=(5,13):
        raise ValueError("R100_PROGRAM_ABLATION_INPUT_GEOMETRY")
    a=raw.copy();x=slots.copy()
    a[:,8:10]=0.
    tag=np.rint(raw[:,4:8]*3)
    x[:,:4,8]=tag/15.
    return x,a


def _derange(raw,action,seed,panel):
    if (panel not in ("FIT","VALID") or seed not in SEEDS
        or raw.shape!=(COUNT[panel],14)
        or action.shape!=(COUNT[panel],)):
        raise PermissionError("R100_AUX_RETURN_SHUFFLE_UNFROZEN_PANEL")
    rng=np.random.default_rng(np.random.SeedSequence([RETURN_SHUFFLE_SALT,seed,SALT[panel]]))
    perm=np.arange(len(raw))
    groups={}
    for i in range(len(raw)):
        groups.setdefault((float(raw[i,12]),int(action[i])),[]).append(i)
    for values in groups.values():
        g=np.asarray(values,np.int64)
        if len(g)<8:
            raise RuntimeError("R100_SHUFFLE_PUBLIC_STRATUM_TOO_SMALL")
        for trial in range(256):
            p=rng.permutation(g)
            if not np.any(p==g):
                perm[g]=p
                break
        else:
            raise RuntimeError("R100_INDEPENDENT_DERANGEMENT_NOT_FOUND")
    if (np.any(perm==np.arange(len(raw)))
        or not np.array_equal(raw[perm,12],raw[:,12])
        or not np.array_equal(action[perm],action)):
        raise RuntimeError("R100_RETURN_SHUFFLE_ACTION_TIME_OR_FIXED_POINTS")
    return perm


def _audit(law,raw,action=None,perm=None):
    if law.shape!=(len(raw),) or raw.shape!=(len(raw),14):
        raise ValueError("R100_PRIVATE_SOURCE_AUDIT_SHAPE")
    rates={}
    for pivot in np.unique(raw[:,12]):
        selected=raw[:,12]==pivot
        rate=float(np.mean(law[selected]>0))
        if not .35<=rate<=.65:
            raise RuntimeError("R100_PRIVATE_LAW_PUBLIC_PIVOT_SOURCE_INVALID")
        rates[str(float(pivot))]=rate
    if action is not None:
        for a in (r85.CALIBRATE,r85.EXPLOIT_POS):
            rate=float(np.mean(law[action==a]>0))
            if not .35<=rate<=.65:
                raise RuntimeError("R100_PRIVATE_LAW_CHOSEN_ACTION_SOURCE_INVALID")
    donor_match=None
    if perm is not None:
        donor_match=float(np.mean(law==law[perm]))
        if not .35<=donor_match<=.65:
            raise RuntimeError("R100_RETURN_SHUFFLE_PRIVATE_LAW_CONTROL_INVALID")
    return {"private_law_plus_by_public_pivot":rates,
            "donor_private_law_match":donor_match,
            "return_shuffle_not_independent_causal_null":True,
            "source_valid":True}


def _factual(panel,seed,sha):
    if panel not in ("FIT","VALID"):
        raise PermissionError("R100_TRAIN_ONLY_NEW_FIT_VALID_NOT_DEV")
    n=COUNT[panel]
    slots=np.empty((n,5,13),np.float32)
    raw=np.empty((n,14),np.float32)
    factual=np.empty(n,np.float32)
    law=np.empty(n,np.int8)
    action=np.empty(n,np.int8)
    for i in range(n):
        world,a=_world(panel,seed,i,sha)
        observed,features,reward=_public(world,a)
        slots[i],raw[i],factual[i],law[i],action[i]=(
            observed,features,reward,world["law_a"],a)
    # A fixed seed/public-stratum matching permutation, chosen BEFORE any
    # target or true-law inspection. It scrambles ONLY auxiliary targets.
    perm=_derange(raw,action,seed,panel)
    audit=_audit(law,raw,action,perm)
    return {"slots":slots,"raw":raw,"canonical":r98._canonical(raw),
            "factual":factual,"aux_shuffled_factual":factual[perm].copy(),
            "action":action,"audit":audit,
            "only_one_realized_chosen_action_return":True}


def _scored(panel,seed,sha):
    if panel not in ("DEV_SEEN","DEV_COMPOSE","OOD_SHIFT","PROGRAM_SWITCH"):
        raise PermissionError("R100_EVALUATOR_ONLY_NEW_HELDOUT_PANELS")
    n=COUNT[panel]
    slots0=np.empty((n,5,13),np.float32)
    slots2=np.empty_like(slots0)
    raw0=np.empty((n,14),np.float32)
    raw2=np.empty_like(raw0)
    ret0=np.empty(n,np.float64)
    ret2=np.empty_like(ret0)
    law=np.empty(n,np.int8)
    cold=np.empty(n,bool)
    counter=None
    if panel=="PROGRAM_SWITCH":
        counter={"slots0":np.empty_like(slots0),
                 "slots2":np.empty_like(slots2),
                 "raw0":np.empty_like(raw0),"raw2":np.empty_like(raw2),
                 "reward0":np.empty_like(ret0),"reward2":np.empty_like(ret2),
                 "law":np.empty_like(law),"cold":np.empty_like(cold)}
    for i in range(n):
        world,_=_world(panel,seed,i,sha)
        x0,r0,val0=_public(world,r85.CALIBRATE)
        x2,r2,val2=_public(world,r85.EXPLOIT_POS)
        if (not np.array_equal(x0[:-1],x2[:-1])
            or not np.array_equal(x0[-1,4:],x2[-1,4:])
            or x0[-1,0]!=1 or x2[-1,2]!=1):
            raise RuntimeError("R100_PUBLIC_PREFIX_LEAKS_UNCHOSEN_ACTION")
        slots0[i],slots2[i]=x0,x2
        raw0[i],raw2[i]=r0,r2
        ret0[i],ret2[i]=val0,val2
        law[i],cold[i]=world["law_a"],bool(i%2)
        if counter is not None:
            w=copy.deepcopy(world)
            w["r98_op"]=(0,1)
            w["law_a"]=r98.law_of(w["factors_a"],w["r98_roles"],(0,1))
            z0,u0,rr0=_public(w,r85.CALIBRATE)
            z2,u2,rr2=_public(w,r85.EXPLOIT_POS)
            if not np.array_equal(x0[:4,9:12],z0[:4,9:12]):
                raise RuntimeError("R100_PROGRAM_INTERVENTION_ALTERED_NOISY_FACTORS")
            counter["slots0"][i],counter["slots2"][i]=z0,z2
            counter["raw0"][i],counter["raw2"][i]=u0,u2
            counter["reward0"][i],counter["reward2"][i]=rr0,rr2
            counter["law"][i],counter["cold"][i]=w["law_a"],bool(i%2)
    audit=_audit(law,raw0)
    result={"slots0":slots0,"slots2":slots2,"raw0":raw0,"raw2":raw2,
            "canonical0":r98._canonical(raw0),
            "canonical2":r98._canonical(raw2),
            "reward0":ret0,"reward2":ret2,
            "law":law,"cold":cold,"audit":audit}
    if counter is not None:
        counter["canonical0"]=r98._canonical(counter["raw0"])
        counter["canonical2"]=r98._canonical(counter["raw2"])
        result["counter"]=counter
    return result


class FactorAlgebra(nn.Module):
    """Deliberate observable-operator inductive bias, NOT inferred law type.

    Four PUBLIC noisy A observations are calibrated by a shared learned
    scalar. PUBLIC role tags select operands. MUX/PARITY/NEGATE operations
    are explicitly hardcoded in differentiable algebra. The hidden true
    factor signs, law and unchosen returns never enter training.
    """
    def __init__(self):
        super().__init__()
        self.gain=nn.Parameter(torch.tensor(2.5))
        self.bias=nn.Parameter(torch.tensor(0.))
        self.selector_sharpness=nn.Parameter(torch.tensor(5.))

    def forward(self,raw):
        if raw.ndim!=2 or raw.shape[1]!=14:
            raise RuntimeError("R100_ALGEBRA_NEEDS_PUBLIC_14D")
        tags=torch.round(raw[:,4:8]*3).long()
        if not bool(torch.all(torch.sort(tags,dim=1).values==torch.arange(
                4,device=raw.device).unsqueeze(0))):
            raise RuntimeError("R100_ALGEBRA_PUBLIC_ROLES_INCOMPLETE")
        calib=torch.tanh(self.gain*raw[:,:4]+self.bias)
        ix=[torch.argmax((tags==k).to(torch.int64),dim=1,keepdim=True)
            for k in (1,2,3)]
        sel,yes,no=(torch.gather(calib,1,j)[:,0] for j in ix)
        prob=torch.sigmoid(self.selector_sharpness*sel)
        mux=prob*yes+(1.-prob)*no
        parity=torch.prod(calib,dim=1)
        # Public binary bits are themselves legally observed instructions.
        base=(1.-raw[:,8])*mux+raw[:,8]*parity
        return (1.-2.*raw[:,9])*base


class NativeFactorInternal(nn.Module):
    """Factorized internal context enters BEFORE original native SSM/TTT."""
    def __init__(self,core):
        super().__init__()
        self.core=core
        self.algebra=FactorAlgebra()
        self.projection=nn.Sequential(nn.Linear(4,32),nn.Tanh(),
                                      nn.Linear(32,32))
        # All variants compute this head: only the objective differs.
        self.sign_head=nn.Sequential(nn.Linear(4,16),nn.Tanh(),
                                     nn.Linear(16,1))

    def forward(self,slots,raw):
        if slots.ndim!=3 or slots.shape[1:]!=(5,13):
            raise RuntimeError("R100_NATIVE_INTERNAL_PUBLIC_EVENT_SHAPE")
        factor=self.algebra(raw)
        features=torch.stack((factor,raw[:,10],raw[:,11],raw[:,12]),dim=1)
        state=self.core.stem(slots)
        context=self.projection(features)
        if context.shape!=(len(slots),32):
            raise RuntimeError("R100_NATIVE_FACTORIZATION_CONTEXT_WIDTH")
        state=torch.cat((state[:,:-1],state[:,-1:]+context[:,None]),dim=1)
        value=r98._native_trunk(self.core,state)
        aux=self.sign_head(features)[:,0]
        if value.shape!=(len(slots),) or aux.shape!=value.shape:
            raise RuntimeError("R100_NATIVE_FACTORED_VALUE_OR_MICRO_SHAPE")
        return value,aux


class NativeFactorExternal(nn.Module):
    """Identical program prior in a scalar output head, not native proof."""
    def __init__(self,core):
        super().__init__()
        self.core=core
        self.algebra=FactorAlgebra()
        self.value=nn.Sequential(nn.Linear(4,32),nn.Tanh(),
                                 nn.Linear(32,1))

    def forward(self,slots,raw):
        z=self.algebra(raw)
        features=torch.stack((z,raw[:,10],raw[:,11],raw[:,12]),dim=1)
        return self.core(slots)[:,-1,0]+self.value(features)[:,0]


def _models(seed):
    if seed not in SEEDS:
        raise PermissionError("R100_MODEL_SEED_NOT_PRECOMMITTED")
    result={}
    result[NATIVE_RAW]=r95._native(seed)
    torch.manual_seed(seed+201)
    result[UNSTRUCTURED]=r98.NativeStemInjection(
        r95._native(seed),r98.RoleProgramContext())
    for arm in FACTORED_INTERNAL_ARMS:
        core=r95._native(seed)
        torch.manual_seed(seed+202)
        result[arm]=NativeFactorInternal(core)
    core=r95._native(seed)
    torch.manual_seed(seed+202)
    result[EXTERNAL]=NativeFactorExternal(core)
    torch.manual_seed(seed+203)
    result[MLP]=r98.Flat14Value()
    torch.manual_seed(seed+204)
    result[SET]=r98.RoleProgramValue()
    torch.manual_seed(seed+205)
    result[GRU]=r81.GRU(64)
    torch.manual_seed(seed+206)
    result[UPPER]=r98.Flat14Value()
    cores=[result[NATIVE_RAW],result[UNSTRUCTURED].core]
    cores.extend(result[arm].core for arm in FACTORED_INTERNAL_ARMS)
    cores.append(result[EXTERNAL].core)
    base=dict(cores[0].named_parameters())
    if any(set(dict(c.named_parameters()))!=set(base) for c in cores[1:]):
        raise RuntimeError("R100_CORE_PARAMETER_NAMES_DIFFER")
    for core in cores[1:]:
        d=dict(core.named_parameters())
        if any(not torch.equal(d[k],base[k]) for k in base):
            raise RuntimeError("R100_IDENTICAL_ORIGINAL_NATIVE_INIT_DRIFT")
    initial=dict(result[NOAUX].named_parameters())
    for arm in (AUX,SHUFFLED,OPMASK):
        other=dict(result[arm].named_parameters())
        if other.keys()!=initial.keys() or any(
            not torch.equal(other[k],initial[k]) for k in initial):
            raise RuntimeError("R100_FACTORED_VARIANTS_NOT_BITWISE_EQUAL_INIT")
    if any(not torch.equal(result[NOAUX].algebra.state_dict()[k],
                           result[EXTERNAL].algebra.state_dict()[k])
           for k in result[NOAUX].algebra.state_dict()):
        raise RuntimeError("R100_INTERNAL_VS_EXTERNAL_FACTORS_NOT_EQUAL_INIT")
    return result


def _arrays(source,arm,suffix=""):
    slots=source["slots"+suffix]
    raw=source["raw"+suffix]
    if arm==UPPER:
        raw=source["canonical"+suffix]
    if arm==OPMASK:
        return _mask_program(slots,raw)
    return slots,raw


def _forward(model,arm,slots,raw,include_aux=False):
    if arm in FACTORED_INTERNAL_ARMS:
        value,aux=model(slots,raw)
        if include_aux:
            return value,aux
    elif arm in (NATIVE_RAW,GRU):
        value=model(slots)[:,-1,0]
    elif arm in (UNSTRUCTURED,EXTERNAL):
        value=model(slots,raw)
    else:
        value=model(raw)
    if value.shape!=(len(slots),):
        raise RuntimeError("R100_MODEL_OUTPUT_SHAPE_NONFROZEN")
    return value


def _predict(model,arm,panel,suffix=""):
    slots,raw=_arrays(panel,arm,suffix)
    model.eval()
    found=[]
    with torch.no_grad():
        for i in range(0,len(slots),32):
            x=torch.from_numpy(slots[i:i+32])
            f=torch.from_numpy(raw[i:i+32])
            found.extend(_forward(model,arm,x,f).tolist())
    model.train()
    out=np.asarray(found,np.float64)
    if out.shape!=(len(slots),) or not np.isfinite(out).all():
        raise RuntimeError("R100_NONFINITE_OR_INCOMPLETE_PREDICTION")
    return out


def _train(model,arm,fit,valid,indices):
    slots,raw=_arrays(fit,arm)
    x=torch.from_numpy(slots)
    features=torch.from_numpy(raw)
    correct=torch.from_numpy(fit["factual"])
    shuffled=torch.from_numpy(fit["aux_shuffled_factual"])
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    before=float(np.mean((_predict(model,arm,valid)-valid["factual"])**2))
    started=time.perf_counter()
    for take in indices:
        if arm in FACTORED_INTERNAL_ARMS:
            value,logit=_forward(model,arm,x[take],features[take],include_aux=True)
            loss=F.mse_loss(value,correct[take])
            if arm==AUX:
                loss=loss+LAMBDA*F.binary_cross_entropy_with_logits(
                    logit,(correct[take]>0).float())
            elif arm==SHUFFLED:
                loss=loss+LAMBDA*F.binary_cross_entropy_with_logits(
                    logit,(shuffled[take]>0).float())
            # NOAUX/OPMASK carry IDENTICAL sign heads, but no aux gradient.
        else:
            value=_forward(model,arm,x[take],features[take])
            loss=F.mse_loss(value,correct[take])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("R100_FROZEN_FACTUAL_TRAINING_LOSS_NONFINITE")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
    elapsed=time.perf_counter()-started
    train=float(np.mean((_predict(model,arm,fit)-fit["factual"])**2))
    val=float(np.mean((_predict(model,arm,valid)-valid["factual"])**2))
    if not all(math.isfinite(z) for z in (before,train,val,elapsed)):
        raise RuntimeError("R100_TRAINING_ARM_INCOMPLETE_OR_NONFINITE")
    return {"steps":len(indices),"factual_selected_return_exposures":len(indices)*BATCH,
            "params":sum(p.numel() for p in model.parameters()),
            "validation_initial_MSE":before,"FIT_final_MSE":train,
            "validation_final_MSE":val,"training_seconds":elapsed,
            "aux_loss_weight":LAMBDA if arm in (AUX,SHUFFLED) else 0.,
            "aux_target":"SIGN_OF_ACTUALLY_REALIZED_RETURN" if arm==AUX
                else "DONOR_FACTUAL_RETURN_SIGN" if arm==SHUFFLED else "NONE"}
