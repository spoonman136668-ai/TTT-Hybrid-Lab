"""R99: internal factual-consequence microfeedback without teacher labels.

Isolated CPU TEST-only experiment. Deep internal intermediate native heads
receive ONLY the SAME REALIZED chosen-action reward as the final value head.
Controls include bitwise same original native+adapter/head initial tensors with
zero auxiliary loss, shuffled factual-return auxiliary targets, and single
stem-only feedback. Original model.py and official weights are unchanged.

No R85-R98 scored physical source or prior learned checkpoint is reopened.
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

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r99_microfeedback_freeze.json"
FREEZE_BLOB="52531b8f3d83c17d1cc5948058ad34b53899f7c5"
SCHEMA="ttt-hybrid.r99-factual-internal-microfeedback.testonly.v1"
PARENT="f9276dac0f9e83bae0eb7443881735e930e14ba8"
BRANCH="research/r99-internal-factual-microfeedback-20261116"
SEEDS=(1002111,1002123)
SALT={"FIT":25400111,"VALID":25400211,"DEV_SEEN":25400311,
      "DEV_COMPOSE":25400411,"OOD_SHIFT":25400511,
      "PROGRAM_SWITCH":25400611}
COUNT={"FIT":2048,"VALID":256,"DEV_SEEN":256,
       "DEV_COMPOSE":256,"OOD_SHIFT":256,"PROGRAM_SWITCH":256}
SHUFFLE_SALT=25400711
BATCH_SALT=25400811
BOOTSTRAP_SALT=25400911
UNIT_SALT=25409991
TRAIN_PROGRAMS=((0,0),(0,1),(1,0))
COMPOSE_PROGRAM=(1,1)
TRAIN_BINDINGS=((0,1,2),(1,2,3),(2,3,0))
HELDOUT_BINDING=(3,0,1)
ARMS=("NATIVE32_ORIGINAL_MAIN",
      "NATIVE32_INTERNAL_MAIN_ONLY",
      "NATIVE32_INTERNAL_FACTUAL_MICROFEEDBACK",
      "NATIVE32_INTERNAL_DERANGED_MICROFEEDBACK",
      "NATIVE32_INTERNAL_STEM_ONLY_MICROFEEDBACK",
      "MLP64_PUBLIC_RAW_PROGRAM",
      "ROLE_SET_PUBLIC_PROGRAM",
      "GRU64_PUBLIC_TAGGED_EVENTS",
      "MLP64_PUBLIC_CANONICAL_UPPER")
(NATIVE_RAW,MAIN_ONLY,FACTUAL,DERANGED,STEM_ONLY,
 MLP,SET,GRU,UPPER)=ARMS
MICRO_ARMS=(MAIN_ONLY,FACTUAL,DERANGED,STEM_ONLY)
STEPS=3072
BATCH=16
AUX_WEIGHT=.15
MAX_SECONDS=260


def frozen():
    if r85.git_blob(FREEZE.read_bytes())!=FREEZE_BLOB:
        raise RuntimeError("R99_PREIMPLEMENTATION_FROZEN_PROTOCOL_GIT_BLOB_DRIFT")
    r98.frozen()
    f=json.loads(FREEZE.read_text(encoding="utf-8"))
    p,s,a,t,e,x=f["parent"],f["source"],f["architecture"],f["training"],f["evaluation"],f["execution"]
    gates=e["hypothesis_checks"]
    if (f["schema"]!=SCHEMA
        or f["stage"]!="FREEZE_BEFORE_IMPLEMENTATION_TESTS_OR_PHYSICAL_SOURCE_FIRST_USE"
        or p["r98_final_nonreplaying_SHA"]!=PARENT
        or p["r98_original_scored_SHA"]!=r98.ORIGINAL_R98_SPENT_SHA
        or p["r98_original_scored_run"]!=r98.ORIGINAL_R98_SPENT_RUN_ID
        or p["r98_freeze_blob"]!=r98.FREEZE_BLOB
        or p["r98_status"]!="R98_INTERNAL_COMPOSITION_NO_GO"
        or p["reuse_prior_scored_source_or_weights"]
        or s["seeds"]!=list(SEEDS)
        or s["counts"]!=COUNT
        or s["salts"]!={**SALT,"AUX_SHUFFLE":SHUFFLE_SALT,
                         "BATCH":BATCH_SALT,"BOOTSTRAP":BOOTSTRAP_SALT,
                         "UNIT_ONLY":UNIT_SALT}
        or s["train_programs"]!=[list(i) for i in TRAIN_PROGRAMS]
        or s["new_composition"]!=list(COMPOSE_PROGRAM)
        or s["training_bindings"]!=[list(i) for i in TRAIN_BINDINGS]
        or s["heldout_binding"]!=list(HELDOUT_BINDING)
        or s["old_scored_or_reserved_sources_used"]
        or f["arms"]!=list(ARMS)
        or t["full_steps_per_arm"]!=STEPS
        or t["batch"]!=BATCH
        or t["auxiliary_loss_weight"]!=AUX_WEIGHT
        or t["aux_heads"]!=4
        or t["whole_process_CPU_limit_seconds_per_seed"]!=MAX_SECONDS
        or e["seeds_required"]!=2
        or e["bootstrap_draws"]!=2000
        or list(gates.values())!=[.30,.30,.15,.50,.10,.35,.80,.25,.10,.20,.65,.80]
        or e["independent_scientific_pass"]
        or not a["original_model_unchanged"]
        or not x["two_original_CPU_fresh_seed_jobs"]
        or not x["requires_full_same_candidate_G4_Linux_and_Windows_PASS_before_physics_first_use"]
        or not f["authority"]["scoped_CPU_TEST_only_original_first_use"]
        or any(bool(v) for k,v in f["authority"].items()
               if k!="scoped_CPU_TEST_only_original_first_use")):
        raise PermissionError("R99_FROZEN_SOURCE_HYPOTHESIS_OR_OWNER_AUTHORITY_DRIFT")
    salts=set(s["salts"].values())
    if len(salts)!=len(s["salts"]) or salts & set(r98.frozen()["source"]["salts"].values()):
        raise RuntimeError("R99_FRESH_SOURCE_SALTS_COLLIDE_WITH_SPENT_DATA")
    return f


def program(panel,index):
    if panel not in SALT or not isinstance(index,int) or not 0<=index<COUNT[panel]:
        raise PermissionError("R99_PROGRAM_OUTSIDE_FROZEN_NEW_PANELS")
    return TRAIN_PROGRAMS[index%3] if panel in ("FIT","VALID","DEV_SEEN") else COMPOSE_PROGRAM


def binding(panel,index):
    if panel not in SALT or not isinstance(index,int) or not 0<=index<COUNT[panel]:
        raise PermissionError("R99_ROLE_BINDING_OUTSIDE_FROZEN_PANELS")
    return TRAIN_BINDINGS[index%3] if panel in ("FIT","VALID","DEV_SEEN") else HELDOUT_BINDING


def chosen(seed,index):
    if seed not in SEEDS or not isinstance(index,int) or index<0:
        raise PermissionError("R99_FACTUAL_ACTION_NOT_FROZEN")
    return r85.CALIBRATE if ((index//4)+SEEDS.index(seed))%2==0 else r85.EXPLOIT_POS


def design_preflight():
    result={}
    for seed in SEEDS:
        result[str(seed)]={}
        for panel,n in COUNT.items():
            buckets={}
            for i in range(n):
                key=(i%2,chosen(seed,i))
                buckets[key]=buckets.get(key,0)+1
            if len(buckets)!=4 or set(buckets.values())!={n//4}:
                raise RuntimeError("R99_COLD_FACTUAL_ACTION_DESIGN_UNBALANCED")
            result[str(seed)][panel]={"cold_action_strata":4,"each_count":n//4}
    return result


def _admit(seed):
    frozen()
    if (seed not in SEEDS
        or os.getenv("GITHUB_ACTIONS")!="true"
        or os.getenv("GITHUB_EVENT_NAME")!="pull_request"
        or os.getenv("GITHUB_REPOSITORY")!="spoonman136668-ai/TTT-Hybrid-Lab"
        or os.getenv("GITHUB_HEAD_REF")!=BRANCH
        or os.getenv("GITHUB_JOB")!="r99-factual-microfeedback"
        or os.getenv("R99_SEED")!=str(seed)
        or os.getenv("RUNNER_OS")!="Linux"
        or os.getenv("RUNNER_ENVIRONMENT")!="github-hosted"
        or os.getenv("GITHUB_RUN_ATTEMPT")!="1"
        or "/merge" not in os.getenv("GITHUB_REF","")):
        raise PermissionError("R99_ORIGINAL_G4_HOSTED_FIRST_ATTEMPT_CPU_ONLY")
    sha=os.getenv("EXPECTED_CANDIDATE_SHA","")
    head=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
                        capture_output=True,text=True,timeout=10,check=False)
    clean=subprocess.run(["git","diff","--quiet","HEAD","--"],cwd=ROOT,
                         capture_output=True,text=True,timeout=10,check=False)
    if len(sha)!=40 or head.returncode or head.stdout.strip()!=sha or clean.returncode:
        raise PermissionError("R99_SOURCE_SHA_OR_CLEAN_WORKTREE_NOT_EXACT")
    return sha


def unit_world(i):
    if not isinstance(i,int) or not 0<=i<12:
        raise PermissionError("R99_ONLY_INDEPENDENT_NONSCORING_UNIT_WORLD")
    rng=np.random.default_rng(np.random.SeedSequence([UNIT_SALT,i]))
    world=r85._make_synthetic_world(i,rng)
    op=TRAIN_PROGRAMS[i%3] if i<9 else COMPOSE_PROGRAM
    roles=TRAIN_BINDINGS[i%3] if i<9 else HELDOUT_BINDING
    factors=rng.choice([-1,1],size=4).astype(np.int64)
    world["factors_a"]=factors
    world["r98_op"]=op
    world["r98_roles"]=roles
    world["law_a"]=r98.law_of(factors,roles,op)
    return world,chosen(SEEDS[0],i)


def _world(panel,seed,index,sha):
    if panel not in SALT:
        raise PermissionError("R99_PREVIOUS_SPENT_OR_PROTECTED_SCIENCE_SOURCE_DENIED")
    if seed not in SEEDS or not isinstance(index,int) or not 0<=index<COUNT[panel]:
        raise PermissionError("R99_UNREGISTERED_SOURCE_SEED_INDEX")
    if (not sha or sha!=os.getenv("EXPECTED_CANDIDATE_SHA")
        or os.getenv("GITHUB_ACTIONS")!="true"
        or os.getenv("GITHUB_HEAD_REF")!=BRANCH
        or os.getenv("GITHUB_JOB")!="r99-factual-microfeedback"
        or os.getenv("R99_SEED")!=str(seed)
        or os.getenv("RUNNER_OS")!="Linux"
        or os.getenv("RUNNER_ENVIRONMENT")!="github-hosted"
        or os.getenv("GITHUB_RUN_ATTEMPT")!="1"):
        raise PermissionError("R99_FIRST_USE_PHYSICAL_SOURCE_NOT_ADMITTED")
    rng=np.random.default_rng(np.random.SeedSequence([SALT[panel],seed,index]))
    world=r85._make_synthetic_world(index%r85.N,rng)
    factors=rng.choice([-1,1],size=4).astype(np.int64)
    op=program(panel,index)
    roles=binding(panel,index)
    world["factors_a"]=factors
    world["r98_op"]=op
    world["r98_roles"]=roles
    world["law_a"]=r98.law_of(factors,roles,op)
    if panel=="OOD_SHIFT":
        world=r86._shift_public_world(world,index)
    return world,chosen(seed,index)


def _observed(world,action):
    """Reusable R98 pure public observation function, NOT its scored source."""
    public,ret=r98._public_view(world,action)
    slots=r88._public_slots(public)
    raw=r98._raw(slots)
    return slots,raw,ret


def derange(raw,action,seed,panel):
    """Fixed before reading return/private law: zero-fixed-point public strata."""
    if (panel not in ("FIT","VALID") or seed not in SEEDS
        or raw.shape!=(COUNT[panel],14) or action.shape!=(COUNT[panel],)):
        raise PermissionError("R99_AUX_DERANGEMENT_ONLY_FRESH_FACTUAL_PANEL")
    rng=np.random.default_rng(np.random.SeedSequence([SHUFFLE_SALT,seed,SALT[panel]]))
    ids=np.arange(len(raw))
    groups={}
    for i in range(len(raw)):
        key=(float(raw[i,12]),int(action[i]))
        groups.setdefault(key,[]).append(i)
    for members in groups.values():
        g=np.asarray(members,np.int64)
        if len(g)<8:
            raise RuntimeError("R99_AUX_DONOR_PUBLIC_STRATUM_TOO_SMALL")
        for _ in range(256):
            p=rng.permutation(g)
            if not np.any(p==g):
                ids[g]=p
                break
        else:
            raise RuntimeError("R99_NO_PRECOMMITTED_DERANGEMENT_FOUND")
    if np.any(ids==np.arange(len(raw))):
        raise RuntimeError("R99_AUX_TARGET_SOURCE_SELF_DONOR_INVALID")
    if (not np.array_equal(raw[ids,12],raw[:,12])
        or not np.array_equal(action[ids],action)):
        raise RuntimeError("R99_AUX_DONOR_DISRUPTED_PUBLIC_TIME_OR_ACTION")
    return ids


def _audit(laws,raw,actions=None,donor=None):
    if laws.shape!=(len(raw),) or raw.shape!=(len(raw),14):
        raise RuntimeError("R99_PRIVATE_LAW_SOURCE_AUDIT_SHAPE")
    strata={}
    for pivot in np.unique(raw[:,12]):
        mask=raw[:,12]==pivot
        p=float(np.mean(laws[mask]>0))
        if not .35<=p<=.65:
            raise RuntimeError("R99_PRIVATE_LAW_PUBLIC_PIVOT_INVALID")
        strata[str(float(pivot))]=p
    if actions is not None:
        for a in (r85.CALIBRATE,r85.EXPLOIT_POS):
            rate=float(np.mean(laws[actions==a]>0))
            if not .35<=rate<=.65:
                raise RuntimeError("R99_PRIVATE_LAW_FACTUAL_ACTION_INVALID")
    alignment=None
    if donor is not None:
        alignment=float(np.mean(laws==laws[donor]))
        if not .35<=alignment<=.65:
            raise RuntimeError("R99_AUX_SHUFFLE_PRIVATE_LAW_ALIGNMENT_INVALID")
    return {"private_law_plus_by_public_pivot":strata,
            "donor_private_law_match":alignment,
            "aux_shuffle_preserves_public_action_time":True,
            "valid":True}


def _factual(panel,seed,sha):
    if panel not in ("FIT","VALID"):
        raise PermissionError("R99_FACTUAL_TRAINING_CANNOT_OPEN_UNSEEN_OR_SCORED_PANEL")
    n=COUNT[panel]
    slots=np.empty((n,5,13),np.float32)
    raw=np.empty((n,14),np.float32)
    target=np.empty(n,np.float32)
    law=np.empty(n,np.int8)
    actions=np.empty(n,np.int8)
    for i in range(n):
        world,a=_world(panel,seed,i,sha)
        obs,features,value=_observed(world,a)
        slots[i],raw[i]=obs,features
        target[i],law[i],actions[i]=value,world["law_a"],a
    # The derangement is seeded and uses ONLY legally public raw and factual
    # actions. It never inspects rewards or private law before its selection.
    perm=derange(raw,actions,seed,panel)
    audit=_audit(law,raw,actions,perm)
    return {"slots":slots,"raw":raw,"canonical":r98._canonical(raw),
            "target":target,"negative_aux_target":target[perm].copy(),
            "factual_action":actions,"source_audit":audit,
            "only_one_factually_observed_return_per_world":True}


def _scored(panel,seed,sha):
    if panel not in ("DEV_SEEN","DEV_COMPOSE","OOD_SHIFT","PROGRAM_SWITCH"):
        raise PermissionError("R99_SCORER_CANNOT_ACCESS_FIT_OR_SPENT_PANEL")
    n=COUNT[panel]
    slots0=np.empty((n,5,13),np.float32)
    slots2=np.empty_like(slots0)
    raw0=np.empty((n,14),np.float32)
    raw2=np.empty_like(raw0)
    r0=np.empty(n,np.float64)
    r2=np.empty_like(r0)
    law=np.empty(n,np.int8)
    cold=np.empty(n,bool)
    counter=None
    if panel=="PROGRAM_SWITCH":
        counter={"slots0":np.empty_like(slots0),
                 "slots2":np.empty_like(slots2),
                 "raw0":np.empty_like(raw0),
                 "raw2":np.empty_like(raw2),
                 "reward0":np.empty_like(r0),
                 "reward2":np.empty_like(r2),
                 "law":np.empty_like(law),
                 "cold":np.empty_like(cold)}
    for i in range(n):
        world,_=_world(panel,seed,i,sha)
        x0,y0,value0=_observed(world,r85.CALIBRATE)
        x2,y2,value2=_observed(world,r85.EXPLOIT_POS)
        if (not np.array_equal(x0[:-1],x2[:-1])
            or not np.array_equal(x0[-1,4:],x2[-1,4:])
            or x0[-1,0]!=1 or x2[-1,2]!=1):
            raise RuntimeError("R99_PHYSICAL_PUBLIC_INPUT_LEAKED_UNCHOSEN_ACTION")
        slots0[i],raw0[i]=x0,y0
        slots2[i],raw2[i]=x2,y2
        r0[i],r2[i]=value0,value2
        law[i],cold[i]=world["law_a"],bool(i%2)
        if counter is not None:
            switch=copy.deepcopy(world)
            switch["r98_op"]=(0,1)
            switch["law_a"]=r98.law_of(switch["factors_a"],
                                      switch["r98_roles"],(0,1))
            z0,u0,w0=_observed(switch,r85.CALIBRATE)
            z2,u2,w2=_observed(switch,r85.EXPLOIT_POS)
            if (not np.array_equal(x0[:4,9:12],z0[:4,9:12])
                or not np.array_equal(x0[-1,4:],z0[-1,4:])):
                raise RuntimeError("R99_PROGRAM_SWITCH_ALTERED_FACTUAL_PROBE_OR_NOISE")
            counter["slots0"][i],counter["slots2"][i]=z0,z2
            counter["raw0"][i],counter["raw2"][i]=u0,u2
            counter["reward0"][i],counter["reward2"][i]=w0,w2
            counter["law"][i],counter["cold"][i]=switch["law_a"],bool(i%2)
    audit=_audit(law,raw0)
    result={"slots0":slots0,"slots2":slots2,"raw0":raw0,"raw2":raw2,
            "canonical0":r98._canonical(raw0),
            "canonical2":r98._canonical(raw2),
            "reward0":r0,"reward2":r2,"law":law,"cold":cold,
            "source_audit":audit}
    if counter is not None:
        counter["canonical0"]=r98._canonical(counter["raw0"])
        counter["canonical2"]=r98._canonical(counter["raw2"])
        result["counter"]=counter
    return result


class NativeInternalMicro(nn.Module):
    """Same role/program adapter inside native stem and four actual-value probes.

    Intermediate probes receive the same observed chosen-action return,
    never an engineered hidden-rule/parity/mux label. Every arm computes all
    four heads: loss routing, not parameter capacity or forward, is ablated.
    """
    def __init__(self,core):
        super().__init__()
        self.core=core
        self.adapter=r98.RoleProgramContext()
        self.micro=nn.ModuleList(nn.Linear(32,1) for _ in range(4))

    def forward(self,slots,raw,include_micro=True):
        if slots.ndim!=3 or slots.shape[1:]!=(5,13) or raw.shape!=(len(slots),14):
            raise RuntimeError("R99_NATIVE_PUBLIC_EVENT_GEOMETRY_DRIFT")
        stem=self.core.stem(slots)
        context=self.adapter(raw)
        if context.shape!=(len(slots),stem.shape[-1]):
            raise RuntimeError("R99_NATIVE_ROLE_CONTEXT_DIMENSION_DRIFT")
        z=torch.cat((stem[:,:-1],stem[:,-1:]+context.unsqueeze(1)),dim=1)
        features=[z[:,-1]]
        if self.core.ssm is not None:
            z=self.core.ssm(z)
        features.append(z[:,-1])
        if self.core.attn is not None:
            z=self.core.attn(z)
        features.append(z[:,-1])
        if self.core.ttt is not None:
            z=self.core.ttt(z)
        features.append(z[:,-1])
        if self.core.depth is not None:
            z=self.core.depth(z)
        main=self.core.head(z)[:,-1,0]
        # Compute ALL four heads even for lambda=0/one-stage ablations.
        aux=torch.stack([h(v)[:,0] for h,v in zip(self.micro,features)],dim=1)
        if main.shape!=(len(slots),) or aux.shape!=(len(slots),4):
            raise RuntimeError("R99_NATIVE_MAIN_OR_MICROFEEDBACK_SHAPE")
        return (main,aux) if include_micro else main


def _models(seed):
    if seed not in SEEDS:
        raise PermissionError("R99_MODEL_SEED_NOT_FROZEN")
    models={}
    models[NATIVE_RAW]=r95._native(seed)
    for arm in MICRO_ARMS:
        core=r95._native(seed)
        torch.manual_seed(seed+62)
        models[arm]=NativeInternalMicro(core)
    torch.manual_seed(seed+71)
    models[MLP]=r98.Flat14Value()
    torch.manual_seed(seed+72)
    models[SET]=r98.RoleProgramValue()
    torch.manual_seed(seed+73)
    models[GRU]=r81.GRU(64)
    torch.manual_seed(seed+74)
    models[UPPER]=r98.Flat14Value()
    native_params=dict(models[NATIVE_RAW].named_parameters())
    for arm in MICRO_ARMS:
        c=dict(models[arm].core.named_parameters())
        if c.keys()!=native_params.keys() or any(
            not torch.equal(c[k],native_params[k]) for k in c):
            raise RuntimeError("R99_NATIVE_CORE_INITIAL_WEIGHTS_NOT_IDENTICAL")
    baseline=dict(models[MAIN_ONLY].named_parameters())
    for arm in (FACTUAL,DERANGED,STEM_ONLY):
        p=dict(models[arm].named_parameters())
        if p.keys()!=baseline.keys() or any(
            not torch.equal(p[k],baseline[k]) for k in baseline):
            raise RuntimeError("R99_MICROFEEDBACK_ABLATION_STARTING_TENSORS_DIFFER")
    return models


def _arrays(panel,arm,suffix=""):
    return (panel["slots"+suffix],
            panel[("canonical" if arm==UPPER else "raw")+suffix])


def _forward(model,arm,slots,raw,aux=False):
    if arm in MICRO_ARMS:
        result=model(slots,raw)
        if aux:
            return result
        value=result[0]
    elif arm in (NATIVE_RAW,GRU):
        value=model(slots)[:,-1,0]
    else:
        value=model(raw)
    if value.shape!=(len(slots),):
        raise RuntimeError("R99_MODEL_VALUE_SHAPE_DRIFT")
    return value


def _predict(model,arm,panel,suffix=""):
    slots,raw=_arrays(panel,arm,suffix)
    model.eval()
    out=[]
    with torch.no_grad():
        for k in range(0,len(slots),32):
            x=torch.from_numpy(slots[k:k+32])
            f=torch.from_numpy(raw[k:k+32])
            out.extend(_forward(model,arm,x,f).tolist())
    model.train()
    out=np.asarray(out,np.float64)
    if out.shape!=(len(slots),) or not np.isfinite(out).all():
        raise RuntimeError("R99_NONFINITE_OR_INCOMPLETE_PREDICTION")
    return out


def _train(model,arm,fit,valid,batches):
    x,y=_arrays(fit,arm)
    slots=torch.from_numpy(x)
    raw=torch.from_numpy(y)
    factual=torch.from_numpy(fit["target"])
    negative=torch.from_numpy(fit["negative_aux_target"])
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    before=float(np.mean((_predict(model,arm,valid)-valid["target"])**2))
    started=time.perf_counter()
    for batch in batches:
        if arm in MICRO_ARMS:
            main,aux=_forward(model,arm,slots[batch],raw[batch],aux=True)
            loss=F.mse_loss(main,factual[batch])
            if arm==FACTUAL:
                loss=loss+AUX_WEIGHT*F.mse_loss(aux,factual[batch,None].expand_as(aux))
            elif arm==DERANGED:
                loss=loss+AUX_WEIGHT*F.mse_loss(aux,negative[batch,None].expand_as(aux))
            elif arm==STEM_ONLY:
                loss=loss+AUX_WEIGHT*F.mse_loss(aux[:,0],factual[batch])
            # MAIN_ONLY: same four auxiliary forward heads / parameter count,
            # with no intermediate microfeedback loss or gradient.
        else:
            prediction=_forward(model,arm,slots[batch],raw[batch])
            loss=F.mse_loss(prediction,factual[batch])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("R99_FACTUAL_RETURN_OR_AUXILIARY_LOSS_NONFINITE")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
    elapsed=time.perf_counter()-started
    train=float(np.mean((_predict(model,arm,fit)-fit["target"])**2))
    val=float(np.mean((_predict(model,arm,valid)-valid["target"])**2))
    if not all(math.isfinite(z) for z in (before,train,val,elapsed)):
        raise RuntimeError("R99_FROZEN_ARM_TRAINING_NONFINITE")
    return {"steps":len(batches),
            "factual_action_return_exposures":len(batches)*BATCH,
            "params":sum(p.numel() for p in model.parameters()),
            "VALID_before_MSE":before,"FIT_after_MSE":train,
            "VALID_after_MSE":val,"training_seconds":elapsed,
            "aux_loss_weight":AUX_WEIGHT if arm in (FACTUAL,DERANGED,STEM_ONLY) else 0.,
            "aux_target":"FACTUAL" if arm in (FACTUAL,STEM_ONLY) else
                        "DERANGED_FACTUAL" if arm==DERANGED else "NONE"}
