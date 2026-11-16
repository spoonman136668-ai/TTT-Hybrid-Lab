"""R96: independently sampled, noncommutative role-mux action consequences.

Fresh TEST-only worlds; no prior scored R85-R95 source, weights or private-law
training labels. Only factual chosen-action return teaches each learner.
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

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r96_role_mux_freeze.json"
FREEZE_BLOB="523447556b00de3761097d8592217418a70cd766"
SCHEMA="ttt-hybrid.r96-independent-mux-role-learning.testonly.v1"
PARENT="ea09a41dc13e4329c6d1a711d3a023ba297bdfa7"
BRANCH="research/r96-independent-mux-role-learning-20261116"
SEEDS=(1001511,1001523)
SALT={"FIT":25370111,"VALID":25370211,"DEV":25370311,
      "OOD_SHIFT":25370411,"ROLE_SWAP":25370511}
COUNT={"FIT":2048,"VALID":256,"DEV":256,"OOD_SHIFT":256,"ROLE_SWAP":256}
DERANGE_SALT=25370611
BOOTSTRAP_SALT=25370711
BATCH_SALT=25370811
UNIT_SALT=25379991
ARMS=("NATIVE32_RAW_SLOTS5","NATIVE32_FLAT_SIDECAR",
      "NATIVE32_ROLE_GATE_SIDECAR","MLP64_RAW_CUES",
      "MLP_ROLE_GATE_RAW_CUES","MLP64_SORTED_CUES_NEGATIVE",
      "MLP_ROLE_GATE_DERANGED_A_NEGATIVE","GRU64_SLOTS5",
      "MLP64_PUBLIC_MUX_UPPER")
NATIVE_RAW,NATIVE_FLAT,NATIVE_GATE,MLP_RAW,MLP_GATE,SORTED,DERANGED,GRU,UPPER=ARMS
STEPS=4096
BATCH=16
MAX_SECONDS=260


def frozen():
    if r85.git_blob(FREEZE.read_bytes())!=FREEZE_BLOB:
        raise RuntimeError("R96_PROSPECTIVE_FREEZE_GIT_BLOB_DRIFT")
    r95.frozen()
    f=json.loads(FREEZE.read_text(encoding="utf-8"))
    p,s,m,t,e,x=f["parent"],f["source"],f["model"],f["training"],f["evaluation"],f["execution"]
    if (f["schema"]!=SCHEMA
        or f["stage"]!="PROSPECTIVE_FREEZE_BEFORE_IMPLEMENTATION_TESTS_AND_SOURCE_FIRST_USE"
        or p["R95_final_engineering_sha"]!=PARENT
        or p["R95_frozen_blob"]!=r95.FREEZE_BLOB
        or p["R95_spent_original_sha"]!=r95.ORIGINAL_R95_SPENT_SHA
        or p["R95_spent_run_id"]!=r95.ORIGINAL_R95_SPENT_RUN_ID
        or p["R95_status"]!="R95_RAW_RELATION_NO_GO"
        or p["old_scored_source_or_weights_reused"]
        or s["seeds"]!=list(SEEDS)
        or s["salts"]!={**SALT,"DERANGE":DERANGE_SALT,"BOOTSTRAP":BOOTSTRAP_SALT,
                         "BATCH":BATCH_SALT,"UNIT":UNIT_SALT}
        or s["world_counts"]!=COUNT
        or s["previous_scored_or_reserved_access"]
        or f["arms"]!=list(ARMS)
        or t["steps_per_arm"]!=STEPS or t["batch"]!=BATCH
        or t["per_seed_CPU_resource_ceiling_seconds"]!=MAX_SECONDS
        or e["bootstrap_samples"]!=2000
        or list(e["mechanism_checks"].values())!=[.75,.75,.15,.35,.85,.50,.10,.20,.65,.80]
        or list(e["native_checks"].values())!=[.20,.50,.20,.30]
        or e["independent_scientific_pass"]
        or not x["two_original_first_use_seed_jobs_in_existing_workflow"]
        or x["new_workflow_file"]
        or not f["authority"]["scoped_cpu_testonly_research_allowed"]
        or any(bool(v) for k,v in f["authority"].items()
               if k!="scoped_cpu_testonly_research_allowed")):
        raise PermissionError("R96_FROZEN_SOURCE_METHOD_OR_AUTHORITY_MISMATCH")
    salts=set(s["salts"].values())
    if len(salts)!=len(s["salts"]) or salts & set(r95.frozen()["source"]["salts"].values()):
        raise RuntimeError("R96_SPENT_OR_REPEATED_SALT_REFUSED")
    return f


def mux(factors:np.ndarray)->int:
    if factors.shape!=(4,) or not np.all(np.isin(factors,(-1,1))):
        raise ValueError("R96_PRIVATE_PHYSICAL_MUX_BINARY_FACTOR_SHAPE")
    return int(factors[1] if factors[0]>0 else factors[2])


def _chosen(seed:int,index:int)->int:
    if seed not in SEEDS or not isinstance(index,int) or index<0:
        raise PermissionError("R96_UNFROZEN_LOGGED_ACTION")
    return r85.CALIBRATE if ((index//4)+SEEDS.index(seed))%2==0 else r85.EXPLOIT_POS


def balance_preflight():
    result={}
    for seed in SEEDS:
        result[str(seed)]={}
        for panel,n in COUNT.items():
            cells={}
            for i in range(n):
                key=(i%2,((i%r85.N)//2)%2,_chosen(seed,i))
                cells[key]=cells.get(key,0)+1
            if len(cells)!=8 or set(cells.values())!={n//8}:
                raise RuntimeError("R96_ACTION_PIVOT_INDEX_STRATUM_DESIGN_IMBALANCE")
            result[str(seed)][panel]={"joint_design_cells":8,"count_per_cell":n//8}
    return result


def _admit(seed:int):
    frozen()
    if (seed not in SEEDS or os.getenv("GITHUB_ACTIONS")!="true"
        or os.getenv("GITHUB_EVENT_NAME")!="pull_request"
        or os.getenv("GITHUB_REPOSITORY")!="spoonman136668-ai/TTT-Hybrid-Lab"
        or os.getenv("GITHUB_HEAD_REF")!=BRANCH
        or os.getenv("GITHUB_JOB")!="r96-independent-role-mux"
        or os.getenv("R96_SEED")!=str(seed)
        or os.getenv("RUNNER_OS")!="Linux"
        or os.getenv("RUNNER_ENVIRONMENT")!="github-hosted"
        or os.getenv("GITHUB_RUN_ATTEMPT")!="1"
        or "/merge" not in os.getenv("GITHUB_REF","")):
        raise PermissionError("R96_ONLY_ORIGINAL_HOSTED_G4_FIRST_ATTEMPT_SCIENCE")
    sha=os.getenv("EXPECTED_CANDIDATE_SHA","")
    actual=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
                          capture_output=True,text=True,timeout=10,check=False)
    clean=subprocess.run(["git","diff","--quiet","HEAD","--"],cwd=ROOT,
                         capture_output=True,timeout=10,check=False)
    if len(sha)!=40 or actual.returncode or actual.stdout.strip()!=sha or clean.returncode:
        raise PermissionError("R96_NOT_EXACT_CLEAN_SOURCE_CANDIDATE")
    return sha


def unit_world(index:int):
    if not isinstance(index,int) or not 0<=index<8:
        raise PermissionError("R96_UNIT_ONLY_NONSCORING_SOURCE")
    rng=np.random.default_rng(np.random.SeedSequence([UNIT_SALT,index]))
    w=r85._make_synthetic_world(index,rng)
    a=rng.choice([-1,1],size=4).astype(np.int64)
    w["factors_a"]=a
    w["law_a"]=mux(a)
    return w,_chosen(SEEDS[0],index)


def _world(panel:str,seed:int,index:int,sha:str):
    if panel not in SALT:
        raise PermissionError("R96_OLD_SPENT_OR_PROTECTED_SOURCE_DENIED")
    if seed not in SEEDS or not isinstance(index,int) or not 0<=index<COUNT[panel]:
        raise PermissionError("R96_UNREGISTERED_SEED_PANEL_OR_WORLD")
    if (not sha or sha!=os.getenv("EXPECTED_CANDIDATE_SHA")
        or os.getenv("GITHUB_ACTIONS")!="true"
        or os.getenv("GITHUB_HEAD_REF")!=BRANCH
        or os.getenv("GITHUB_JOB")!="r96-independent-role-mux"
        or os.getenv("R96_SEED")!=str(seed)
        or os.getenv("RUNNER_OS")!="Linux"
        or os.getenv("RUNNER_ENVIRONMENT")!="github-hosted"
        or os.getenv("GITHUB_RUN_ATTEMPT")!="1"):
        raise PermissionError("R96_NEW_SOURCE_NOT_ADMITTED")
    rng=np.random.default_rng(np.random.SeedSequence([SALT[panel],seed,index]))
    w=r85._make_synthetic_world(index%r85.N,rng)
    a=rng.choice([-1,1],size=4).astype(np.int64)
    w["factors_a"]=a
    w["law_a"]=mux(a)
    if panel=="OOD_SHIFT":
        w=r86._shift_public_world(w,index)
    return w,_chosen(seed,index)


def _upper(raw:np.ndarray):
    if raw.ndim!=2 or raw.shape[1]!=8:
        raise ValueError("R96_PUBLIC_UPPER_REQUIRES_RAW_8D")
    out=raw.copy()
    cues=np.where(raw[:,:4]>0,1.,-1.)
    out[:,:4]=0
    out[:,0]=np.where(cues[:,0]>0,cues[:,1],cues[:,2])
    return out


def _sorted(raw:np.ndarray):
    if raw.ndim!=2 or raw.shape[1]!=8:
        raise ValueError("R96_ROLE_BLIND_REQUIRES_RAW_8D")
    out=raw.copy()
    out[:,:4]=np.sort(out[:,:4],axis=1)
    return out


def _derange(raw:np.ndarray,seed:int,panel:str,chosen:np.ndarray|None):
    if panel not in COUNT or seed not in SEEDS or raw.shape!=(COUNT[panel],8):
        raise PermissionError("R96_DERANGE_ONLY_FROZEN_NEW_PUBLIC_PANEL")
    if chosen is not None and chosen.shape!=(len(raw),):
        raise ValueError("R96_DERANGE_LOGGED_ACTION_SHAPE")
    rng=np.random.default_rng(np.random.SeedSequence([DERANGE_SALT,seed,SALT[panel]]))
    out=np.arange(len(raw))
    groups={}
    for i,row in enumerate(raw):
        key=(float(row[6]),int(chosen[i]) if chosen is not None else -1)
        groups.setdefault(key,[]).append(i)
    for indices in groups.values():
        ids=np.asarray(indices,dtype=np.int64)
        if len(ids)<8:
            raise RuntimeError("R96_DERANGEMENT_GROUP_TOO_SMALL")
        for _ in range(200):
            donor=rng.permutation(ids)
            if not np.any(donor==ids):
                out[ids]=donor
                break
        else:
            raise RuntimeError("R96_DERANGEMENT_REJECTION_LIMIT")
    if np.any(out==np.arange(len(raw))):
        raise RuntimeError("R96_DERANGEMENT_HAS_FIXED_POINT")
    return out


def _audit(law:np.ndarray,raw:np.ndarray,perm:np.ndarray|None,chosen=None):
    if law.shape!=(len(raw),):
        raise RuntimeError("R96_PRIVATE_LAW_AUDIT_GEOMETRY")
    rates={}
    for pivot in np.unique(raw[:,6]):
        ids=np.flatnonzero(raw[:,6]==pivot)
        p=float(np.mean(law[ids]>0))
        rates[str(float(pivot))]=p
        if not .4<=p<=.6:
            raise RuntimeError("R96_PRIVATE_LAW_UNBALANCED_BY_PUBLIC_TIME")
    if chosen is not None:
        for a in (r85.CALIBRATE,r85.EXPLOIT_POS):
            p=float(np.mean(law[chosen==a]>0))
            if not .4<=p<=.6:
                raise RuntimeError("R96_PRIVATE_LAW_UNBALANCED_BY_SELECTED_ACTION")
    eq=None
    if perm is not None:
        eq=float(np.mean(law==law[perm]))
        if not .4<=eq<=.6:
            raise RuntimeError("R96_NEGATIVE_LAW_ALIGNMENT_NOT_INDEPENDENT")
    return {"law_plus_rate_by_public_pivot":rates,"same_law_by_deranged_donor":eq,"pass":True}


def _factual(panel:str,seed:int,sha:str):
    if panel not in ("FIT","VALID"):
        raise PermissionError("R96_CANNOT_TRAIN_ON_NEW_SCORED_DEV")
    n=COUNT[panel]
    slots=np.empty((n,5,13),np.float32)
    raw=np.empty((n,8),np.float32)
    target=np.empty(n,np.float32)
    chosen=np.empty(n,np.int8)
    law=np.empty(n,np.int8)
    for i in range(n):
        w,a=_world(panel,seed,i,sha)
        prefix,ret=r86._view(w,a)
        slot=r88._public_slots(prefix)
        slots[i]=slot
        raw[i]=r95._raw_from_slots(slot)
        target[i],chosen[i],law[i]=ret,a,w["law_a"]
    permutation=_derange(raw,seed,panel,chosen)
    audit=_audit(law,raw,permutation,chosen)
    neg=raw.copy()
    neg[:,:4]=raw[permutation,:4]
    if not np.array_equal(neg[:,4:7],raw[:,4:7]):
        raise RuntimeError("R96_SHUFFLED_NEGATIVE_CHANGED_FACTUAL_ACTION")
    return {"slots":slots,"raw":raw,"upper":_upper(raw),"sorted":_sorted(raw),
            "deranged":neg,"target":target,"action":chosen,
            "audit":audit,"only_one_factual_return_per_world":True}


def _scored(panel:str,seed:int,sha:str):
    if panel not in ("DEV","OOD_SHIFT","ROLE_SWAP"):
        raise PermissionError("R96_NEW_PAIRED_SCORING_NOT_FOR_FIT")
    n=COUNT[panel]
    slots0=np.empty((n,5,13),np.float32);slots2=np.empty_like(slots0)
    raw0=np.empty((n,8),np.float32);raw2=np.empty_like(raw0)
    r0=np.empty(n,np.float64);r2=np.empty(n,np.float64)
    truth=np.empty(n,np.int8)
    cold=np.empty(n,bool)
    counter=None
    if panel=="ROLE_SWAP":
        counter={"slots0":np.empty_like(slots0),"slots2":np.empty_like(slots2),
                 "raw0":np.empty_like(raw0),"raw2":np.empty_like(raw2),
                 "reward0":np.empty_like(r0),"reward2":np.empty_like(r2),
                 "law":np.empty_like(truth),"cold":np.empty_like(cold)}
    for i in range(n):
        w,_=_world(panel,seed,i,sha)
        prefix0,ret0=r86._view(w,r85.CALIBRATE)
        prefix2,ret2=r86._view(w,r85.EXPLOIT_POS)
        if (not np.array_equal(prefix0[:-1],prefix2[:-1])
            or not np.array_equal(prefix0[-1,4:],prefix2[-1,4:])
            or prefix0[-1,0]!=1. or prefix2[-1,2]!=1.):
            raise RuntimeError("R96_COUNTERFACTUAL_ACTION_LEAK_IN_PUBLIC_PREFIX")
        slots0[i],slots2[i]=r88._public_slots(prefix0),r88._public_slots(prefix2)
        raw0[i],raw2[i]=r95._raw_from_slots(slots0[i]),r95._raw_from_slots(slots2[i])
        r0[i],r2[i],truth[i],cold[i]=ret0,ret2,w["law_a"],bool(i%2)
        if counter is not None:
            swapped=copy.deepcopy(w)
            a=swapped["factors_a"].copy()
            a[1],a[2]=a[2],a[1]
            swapped["factors_a"]=a
            swapped["law_a"]=mux(a)
            p0,s0=r86._view(swapped,r85.CALIBRATE)
            p2,s2=r86._view(swapped,r85.EXPLOIT_POS)
            counter["slots0"][i],counter["slots2"][i]=r88._public_slots(p0),r88._public_slots(p2)
            counter["raw0"][i]=r95._raw_from_slots(counter["slots0"][i])
            counter["raw2"][i]=r95._raw_from_slots(counter["slots2"][i])
            counter["reward0"][i],counter["reward2"][i]=s0,s2
            counter["law"][i]=swapped["law_a"]
            counter["cold"][i]=bool(i%2)
    perm=_derange(raw0,seed,panel,None)
    audit=_audit(truth,raw0,perm,None)
    deranged0=raw0.copy();deranged2=raw2.copy()
    deranged0[:,:4]=raw0[perm,:4];deranged2[:,:4]=raw0[perm,:4]
    result={"slots0":slots0,"slots2":slots2,"raw0":raw0,"raw2":raw2,
            "upper0":_upper(raw0),"upper2":_upper(raw2),
            "sorted0":_sorted(raw0),"sorted2":_sorted(raw2),
            "deranged0":deranged0,"deranged2":deranged2,
            "reward0":r0,"reward2":r2,"law":truth,"cold":cold,"audit":audit}
    if counter is not None:
        counter["upper0"]=_upper(counter["raw0"]);counter["upper2"]=_upper(counter["raw2"])
        counter["sorted0"]=_sorted(counter["raw0"]);counter["sorted2"]=_sorted(counter["raw2"])
        # For role-swap diagnostics, deranged-control predictions are not used
        # as evidence; keep evaluation shapes valid without a new adaptive draw.
        counter["deranged0"]=counter["sorted0"];counter["deranged2"]=counter["sorted2"]
        result["counter"]=counter
    return result


class RoleGateValue(nn.Module):
    """Learn a generic cue-dependent selector, not a coded if-then mux."""
    def __init__(self):
        super().__init__()
        self.gate=nn.Sequential(nn.Linear(4,32),nn.GELU(),nn.Linear(32,4))
        self.candidate=nn.Sequential(nn.Linear(4,32),nn.GELU(),nn.Linear(32,4))
        self.readout=nn.Sequential(nn.Linear(4,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,x:torch.Tensor)->torch.Tensor:
        if x.ndim!=2 or x.shape[1]!=8:
            raise RuntimeError("R96_ROLE_GATE_EXPECTS_PUBLIC_8D")
        cue=x[:,:4]
        weights=F.softmax(self.gate(cue),dim=1)
        rel=(weights*self.candidate(cue)).sum(dim=1,keepdim=True)
        return self.readout(torch.cat((rel,x[:,4:7]),dim=1))[:,0]


class NativeSidecar(nn.Module):
    """Additive experimental value readout only; core remains unchanged."""
    def __init__(self,core:nn.Module,head:nn.Module):
        super().__init__()
        self.core=core
        self.head=head
    def forward(self,slots,raw):
        return self.core(slots)[:,-1,0]+self.head(raw)


def _models(seed:int):
    if seed not in SEEDS:
        raise PermissionError("R96_MODEL_SEED_NOT_FROZEN")
    out={}
    out[NATIVE_RAW]=r95._native(seed)
    out[NATIVE_FLAT]=NativeSidecar(r95._native(seed),r87.PublicValueMLP())
    out[NATIVE_GATE]=NativeSidecar(r95._native(seed),RoleGateValue())
    torch.manual_seed(seed+42)
    out[MLP_RAW]=r87.PublicValueMLP()
    torch.manual_seed(seed+43)
    out[MLP_GATE]=RoleGateValue()
    torch.manual_seed(seed+42)
    out[SORTED]=r87.PublicValueMLP()
    torch.manual_seed(seed+43)
    out[DERANGED]=RoleGateValue()
    torch.manual_seed(seed+64)
    out[GRU]=r81.GRU(64)
    torch.manual_seed(seed+42)
    out[UPPER]=r87.PublicValueMLP()
    a,b,c=[dict(m.named_parameters()) for m in
           (out[NATIVE_RAW],out[NATIVE_FLAT].core,out[NATIVE_GATE].core)]
    if not(a.keys()==b.keys()==c.keys()) or any(
        not torch.equal(a[k],b[k]) or not torch.equal(a[k],c[k]) for k in a):
        raise RuntimeError("R96_NATIVE_INITIAL_CORE_WEIGHTS_DIFFER")
    x,y=dict(out[MLP_GATE].named_parameters()),dict(out[DERANGED].named_parameters())
    if x.keys()!=y.keys() or any(not torch.equal(x[k],y[k]) for k in x):
        raise RuntimeError("R96_SHUFFLED_GATE_HAS_DIFFERENT_INITIAL_WEIGHTS")
    return out


def _arrays(data,arm,suffix=""):
    key="sorted" if arm==SORTED else "deranged" if arm==DERANGED else "upper" if arm==UPPER else "raw"
    return data["slots"+suffix],data[key+suffix]


def _forward(m,arm,slots,raw):
    if arm in (NATIVE_RAW,GRU):
        y=m(slots)[:,-1,0]
    elif arm in (NATIVE_FLAT,NATIVE_GATE):
        y=m(slots,raw)
    else:
        y=m(raw)
    if y.shape!=(len(slots),):
        raise RuntimeError("R96_PREDICTION_SHAPE_CHANGED")
    return y


def _predict(m,arm,data,suffix=""):
    slots,raw=_arrays(data,arm,suffix)
    m.eval()
    v=[]
    with torch.no_grad():
        for k in range(0,len(raw),32):
            x=torch.from_numpy(slots[k:k+32]);y=torch.from_numpy(raw[k:k+32])
            v.extend(_forward(m,arm,x,y).tolist())
    m.train()
    v=np.asarray(v,np.float64)
    if v.shape!=(len(raw),) or not np.isfinite(v).all():
        raise RuntimeError("R96_NONFINITE_PREDICTION")
    return v


def _train(m,arm,fit,valid,batches):
    slot,raw=_arrays(fit,arm)
    x=torch.from_numpy(slot);f=torch.from_numpy(raw)
    y=torch.from_numpy(fit["target"])
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    before=float(np.mean((_predict(m,arm,valid)-valid["target"])**2))
    t=time.perf_counter()
    for indices in batches:
        loss=F.mse_loss(_forward(m,arm,x[indices],f[indices]),y[indices])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("R96_NONFINITE_TRAINING_LOSS")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(),1.)
        opt.step()
    duration=time.perf_counter()-t
    train=float(np.mean((_predict(m,arm,fit)-fit["target"])**2))
    val=float(np.mean((_predict(m,arm,valid)-valid["target"])**2))
    if not all(math.isfinite(z) for z in (train,val,before,duration)):
        raise RuntimeError("R96_NONFINITE_COMPLETED_FIT")
    return {"updates":len(batches),"observed_factual_exposures":len(batches)*BATCH,
            "params":sum(p.numel() for p in m.parameters()),
            "VALID_initial_MSE":before,"FIT_final_MSE":train,
            "VALID_final_MSE":val,"fit_seconds":duration}
