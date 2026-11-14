"""R67 FIT-trained factual causal memory; prospective DEV source and controls.

R66 supplied only the fixed public-bank algorithm, not any previously scored
data. No hidden change points, law families, evaluator labels, future effects,
or noncausal proxy are inputs to the learner. A selected effect is observed
ONLY after the prediction is scored. DEV is original, scored once after FIT
training, with no post-result adjustment.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import p85mr_f8r66_bank_identity as r66

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r67_trained_protected_memory_freeze.json"
FREEZE_BLOB="0f4bb710c00d424fd96de4e4dc11ca0745500627"
SCHEMA="ttt-hybrid.f8r67-trained-protected-causal-memory-dev-freeze.v1"
PARENT="efe7081d1a32e5ce96236bdf0a5c9024658d9afb"
SALT={"FIT":23910000,"CAL":23920000,"DEV":23930000,"PROXY_NEGATIVE":23940000}
COUNTS={"FIT":128,"CAL":32,"DEV":96}
SEEDS=(985403,985409)
ARMS=("TRAINED_SPARSE_DIRECTION","TRAINED_SPARSE_NONDIRECTION",
      "TRAINED_GRU36","FROZEN_R66_PROTECTED_ROLE_BANK")
TRAINED=ARMS[:3]
CONTROL=ARMS[1:]
ABLATION="DIRECTIONAL_ERASE"
N=48
D=4
FAST=36
UPDATES=96
BATCH=8
GAIN=.65
LR=.01
WD=.005
BOOT_SEED=23949999
B_INDICES=(20,21,24,25,28,29)
RETURN_INDICES=(36,37,40,41,44,45)


def git_blob(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def locked():
    raw=FREEZE.read_bytes()
    if git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("F8R67_FROZEN_BLOB_DRIFT")
    f=json.loads(raw)
    src,arms,tr,metrics=f["source"],f["arms"],f["training"],f["metrics"]
    if (f["schema"]!=SCHEMA or f["stage"]!="FROZEN_BEFORE_R67_SOURCE_TRAINING_OR_DEV_SCORE"
        or f["parent"]["R66_SHA"]!=PARENT
        or f["parent"]["no_R66_or_earlier_scored_world_reuse"] is not True
        or src["salts"]!=SALT or src["cases"]!=COUNTS or src["event_count"]!=N
        or src["action_roles"]!=[0,1,2,3]
        or tr["seeds"]!=list(SEEDS)
        or tr["optimizer_steps_each_trainable_arm_seed"]!=UPDATES
        or tr["FIT_batch_worlds"]!=BATCH or tr["lr"]!=LR
        or tr["weight_decay"]!=WD
        or metrics["require_both_seeds"] is not True
        or metrics["eligible_controls"]!=list(CONTROL)
        or metrics["bootstrap"]["resamples"]!=2000
        or metrics["bootstrap"]["seed"]!=BOOT_SEED
        or arms["persistent_fast_state_candidate_and_fixed"]!=FAST
        or arms["gru_hidden_fast_state"]!=FAST
        or r66.FREEZE_BLOB!=f["parent"]["R66_freeze_blob"]
        or any(f["authority"].values())):
        raise ValueError("F8R67_FROZEN_CONTRACT_DRIFT")
    return f


def worlds(panel):
    if panel not in COUNTS:
        raise PermissionError("F8R67_PROTECTED_OR_OLD_SOURCE_DENIED")
    rng=np.random.default_rng(SALT[panel])
    n=COUNTS[panel]
    switched=n//2
    modes=["stable"]*(n//2)
    if panel in ("FIT","CAL"):
        modes+=["rotation"]*(switched//2)+["gain"]*(switched//2)
    else:
        modes+=["rotation"]*(switched//2)+["crossrole"]*(switched//2)
    modes=[modes[i] for i in rng.permutation(n)]
    roles=np.arange(N,dtype=np.int64)%D
    out=[]
    for i,mechanism in enumerate(modes):
        law_a=rng.normal(0,.75,(D,D))+np.eye(D)*.25
        law_b=law_a.copy()
        if mechanism=="rotation":
            law_b[:2,:2]=np.stack((-law_a[1,:2],law_a[0,:2]),axis=0)
        elif mechanism=="gain":
            law_b[:2,:2]=1.35*law_a[:2,:2]
        elif mechanism=="crossrole":
            law_b[:,0]=.60*law_a[:,0]+.80*law_a[:,1]
            law_b[:,1]=-.80*law_a[:,0]+.60*law_a[:,1]
        effect=np.empty((N,D),dtype=np.float64)
        for t,role in enumerate(roles):
            law=law_b if 16<=t<32 else law_a
            effect[t]=law[:,role]+rng.normal(0,.04,D)
        proxy=rng.normal(0,.8,(N,D))
        out.append({"panel":panel,"index":i,"roles":roles.copy(),
                    "effects":effect,"proxy":proxy,
                    "mechanism_evaluator_only":mechanism,
                    "switched_evaluator_only":mechanism!="stable"})
    return out


def proxy_negative(dev):
    if len(dev)!=COUNTS["DEV"] or any(x["panel"]!="DEV" for x in dev):
        raise PermissionError("F8R67_PROXY_NEEDS_ORIGINAL_DEV")
    rng=np.random.default_rng(SALT["PROXY_NEGATIVE"])
    return [{**w,"proxy":rng.normal(0,3,(N,D))} for w in dev]


def factual_batch(source,indices):
    if not len(indices) or any(int(i)<0 or int(i)>=len(source) for i in indices):
        raise ValueError("F8R67_BAD_PUBLIC_WORLD_INDEX")
    batch=[source[int(i)] for i in indices]
    return (torch.from_numpy(np.stack([w["roles"] for w in batch]).astype(np.int64)),
            torch.from_numpy(np.stack([w["effects"] for w in batch]).astype(np.float64)))


def _validate(roles,effects):
    if (roles.dtype!=torch.int64 or effects.dtype!=torch.float64
        or roles.ndim!=2 or roles.shape[1]!=N
        or effects.shape!=(roles.shape[0],N,D)
        or not bool(((roles>=0)&(roles<D)).all())
        or not bool(torch.isfinite(effects).all())):
        raise ValueError("F8R67_BAD_FACTUAL_BATCH")


class SparseBankGate(nn.Module):
    """5 trained globals; 2 x 4 x 4 factual banks and 4 role selectors."""

    def __init__(self,seed:int,mode:str):
        super().__init__()
        if seed not in SEEDS or mode not in (TRAINED[0],TRAINED[1]):
            raise ValueError("F8R67_UNREGISTERED_LEARNED_GATE")
        self.mode=mode
        generator=torch.Generator(device="cpu").manual_seed(seed+int(mode==TRAINED[1]))
        self.theta=nn.Parameter((torch.rand(5,generator=generator,dtype=torch.float64)-.5)*.1)

    def forward(self,roles,effects,*,erase_direction=False,details=False):
        _validate(roles,effects)
        batch=roles.shape[0]
        w0=effects.new_zeros((batch,D,D))
        w1=effects.new_zeros((batch,D,D))
        selector=effects.new_zeros((batch,D))
        losses=[]
        for t in range(N):
            action=F.one_hot(roles[:,t],num_classes=D).to(effects.dtype)
            active=(selector*action).sum(-1)
            v0=(w0*action[:,None,:]).sum(-1)
            v1=(w1*action[:,None,:]).sum(-1)
            current=(1-active[:,None])*v0+active[:,None]*v1
            other=active[:,None]*v0+(1-active[:,None])*v1
            observed=effects[:,t]
            losses.append((current-observed).square().mean(-1))
            c=F.cosine_similarity(current,observed,dim=-1,eps=1e-9)
            altc=F.cosine_similarity(other,observed,dim=-1,eps=1e-9)
            if self.mode==TRAINED[0]:
                extra=(c.abs(),altc.abs()) if erase_direction else (c,altc)
            else:
                extra=(torch.log1p(current.square().mean(-1)),
                       torch.log1p(other.square().mean(-1)))
            features=torch.stack((
                torch.ones_like(c),
                torch.log1p((observed-current).square().mean(-1)),
                torch.log1p((observed-other).square().mean(-1)),
                extra[0],extra[1]),dim=-1)
            soft=torch.sigmoid(features@self.theta)
            hard=(soft>.5).to(effects.dtype)
            flip=hard-soft.detach()+soft
            new_active=active*(1-flip)+(1-active)*flip
            selector=selector*(1-action)+new_active[:,None]*action
            w0=w0+((1-new_active)*GAIN)[:,None,None]*(observed-v0)[:,:,None]*action[:,None,:]
            w1=w1+(new_active*GAIN)[:,None,None]*(observed-v1)[:,:,None]*action[:,None,:]
        errors=torch.stack(losses,dim=1)
        if not details:
            return errors.mean()
        state=torch.cat((w0.reshape(batch,-1),w1.reshape(batch,-1),selector),dim=-1)
        if state.shape!=(batch,FAST) or not bool(torch.isfinite(state).all()):
            raise ValueError("F8R67_FAST_STATE_DRIFT")
        return {"errors":errors.detach().cpu().numpy(),
                "state":state.detach().cpu().numpy()}


class GRU36(nn.Module):
    """Stronger conventional 36-float state control; more trainable params."""

    def __init__(self,seed:int):
        super().__init__()
        if seed not in SEEDS:
            raise ValueError("F8R67_GRU_SEED_NOT_FROZEN")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.cell=nn.GRUCell(D+D,FAST).double()
            self.read=nn.Linear(FAST+D,D).double()

    def forward(self,roles,effects,*,details=False):
        _validate(roles,effects)
        b=roles.shape[0]
        h=effects.new_zeros((b,FAST))
        losses=[]
        for t in range(N):
            action=F.one_hot(roles[:,t],num_classes=D).to(effects.dtype)
            prediction=self.read(torch.cat((h,action),dim=-1))
            observed=effects[:,t]
            losses.append((prediction-observed).square().mean(-1))
            h=self.cell(torch.cat((action,observed),dim=-1),h)
        errors=torch.stack(losses,dim=1)
        if not details:
            return errors.mean()
        return {"errors":errors.detach().cpu().numpy(),
                "state":h.detach().cpu().numpy()}


def init(arm,seed):
    if arm in TRAINED[:2]:
        return SparseBankGate(seed,arm)
    if arm==TRAINED[2]:
        return GRU36(seed)
    raise ValueError("F8R67_NOT_TRAINABLE_ARM")


def train_one(arm,seed,fit):
    if arm not in TRAINED or seed not in SEEDS or len(fit)!=COUNTS["FIT"]:
        raise ValueError("F8R67_TRAIN_REQUIRES_FIT_AND_FROZEN_SEED")
    t0=time.perf_counter()
    model=init(arm,seed)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    draw=np.random.default_rng(23919999+seed).integers(0,len(fit),(UPDATES,BATCH))
    check_roles,check_effects=factual_batch(fit,draw[0])
    with torch.no_grad():
        initial=float(model(check_roles,check_effects))
    for ids in draw:
        roles,effects=factual_batch(fit,ids)
        opt.zero_grad(set_to_none=True)
        loss=model(roles,effects)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R67_NONFINITE_FIT_LOSS")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
    with torch.no_grad():
        final=float(model(check_roles,check_effects))
    n=sum(p.numel() for p in model.parameters())
    return model,{"seed":seed,"arm":arm,"optimizer_steps":UPDATES,
                  "FIT_factual_selected_events":UPDATES*BATCH*N,
                  "trainable_parameters":n,"fast_state_floats":FAST,
                  "train_initial_MSE":initial,"train_final_MSE":final,
                  "train_wall_seconds":time.perf_counter()-t0,
                  "weights_sha256":hashlib.sha256(b"".join(
                      p.detach().cpu().numpy().astype("<f8").tobytes()
                      for p in model.parameters())).hexdigest()}


def score_one(arm,model,source,*,erase_direction=False):
    if not source or any(x["panel"]!=source[0]["panel"] for x in source):
        raise ValueError("F8R67_INVALID_EVALUATION_PANEL")
    started=time.perf_counter()
    if arm==ARMS[-1]:
        if model is not None:
            raise ValueError("F8R67_FIXED_CONTROL_HAS_NO_TRAINED_WEIGHTS")
        raw=[]
        for w in source:
            factual=r66.update_role_bank("protected_role_bank",w["roles"],w["effects"])
            raw.append((factual["errors"],factual["state_hash"]))
    else:
        roles,effects=factual_batch(source,range(len(source)))
        if model is None:
            raise ValueError("F8R67_MISSING_TRAINED_MODEL")
        with torch.no_grad():
            z=(model(roles,effects,erase_direction=erase_direction,details=True)
               if arm in TRAINED[:2] else model(roles,effects,details=True))
        raw=[(z["errors"][i],hashlib.sha256(z["state"][i]
               .astype("<f8").tobytes()).hexdigest()) for i in range(len(source))]
    rows=[]
    for world,(errors,state_hash) in zip(source,raw):
        e=np.asarray(errors,dtype=np.float64)
        rows.append({"world":world["index"],"switched":world["switched_evaluator_only"],
                     "family":world["mechanism_evaluator_only"],
                     "B":float(np.mean(e[list(B_INDICES)])),
                     "A_return":float(np.mean(e[list(RETURN_INDICES)])),
                     "stable":float(np.mean(e)),
                     "state_hash":state_hash})
    switched=[v for v in rows if v["switched"]]
    stable=[v for v in rows if not v["switched"]]
    if len(switched)!=len(stable):
        raise ValueError("F8R67_UNBALANCED_DEV_SOURCE")
    B=float(np.mean([v["B"] for v in switched]))
    A=float(np.mean([v["A_return"] for v in switched]))
    S=float(np.mean([v["stable"] for v in stable]))
    return {"B_MSE":B,"A_return_MSE":A,"stable_MSE":S,"joint_MSE":B+A+S,
            "fast_state_floats":FAST,"factual_events":N*len(source),
            "eval_wall_seconds":time.perf_counter()-started,"per_world":rows}


def judge(scores,seed,frozen,trained):
    panel=scores["DEV"][seed]
    strongest=min(CONTROL,key=lambda a:panel[a]["joint_MSE"])
    base=panel[strongest]
    cand=panel[ARMS[0]]
    abl=panel[ABLATION]
    rel=lambda c,b: c/max(b,1e-12)-1
    gain=1-cand["joint_MSE"]/max(base["joint_MSE"],1e-12)
    reg_B=rel(cand["B_MSE"],base["B_MSE"])
    reg_A=rel(cand["A_return_MSE"],base["A_return_MSE"])
    reg_S=rel(cand["stable_MSE"],base["stable_MSE"])
    erasure=rel(abl["joint_MSE"],cand["joint_MSE"])
    dev_pair=[]
    for a,b in zip(cand["per_world"],base["per_world"]):
        if a["world"]!=b["world"] or a["switched"]!=b["switched"]:
            raise ValueError("F8R67_MISALIGNED_WORLD_PAIRS")
        dev_pair.append(
            2*((a["B"]+a["A_return"])-(b["B"]+b["A_return"]))
            if a["switched"] else 2*(a["stable"]-b["stable"]))
    boot=frozen["metrics"]["bootstrap"]
    rng=np.random.default_rng(boot["seed"]+seed)
    res=rng.integers(0,len(dev_pair),(boot["resamples"],len(dev_pair)))
    upper=float(np.quantile(np.asarray(dev_pair)[res].mean(axis=1),.975))
    limits=frozen["metrics"]["pass_all"]
    checks={
     "joint_5pct_gain":gain>=limits["joint_relative_gain_vs_strongest_min"],
     "B_nonregression":reg_B<=limits["B_relative_regression_max"],
     "A_return_nonregression":reg_A<=limits["A_return_relative_regression_max"],
     "stable_nonregression":reg_S<=limits["stable_relative_regression_max"],
     "directional_erase_5pct_worse":erasure>=limits["directional_erase_joint_relative_degradation_min"],
     "paired_bootstrap_upper_negative":upper<0,
     "train_step_and_factual_event_parity":all(
         trained[seed][arm]["optimizer_steps"]==UPDATES
         and trained[seed][arm]["FIT_factual_selected_events"]==UPDATES*BATCH*N
         for arm in TRAINED)
    }
    return {"seed":seed,"strongest_eligible_control":strongest,
            "joint_relative_gain":gain,"B_relative_regression":reg_B,
            "A_return_relative_regression":reg_A,"stable_relative_regression":reg_S,
            "directional_erasure_relative_degradation":erasure,
            "paired_world_bootstrap_upper_95pct":upper,"checks":checks}


def run():
    freeze=locked()
    fit,cal,dev=(worlds(panel) for panel in ("FIT","CAL","DEV"))
    if (any(sum(int(w["switched_evaluator_only"]) for w in s)!=len(s)//2
            for s in (fit,cal,dev)) or
        sum(w["mechanism_evaluator_only"]=="crossrole" for w in dev)!=24 or
        any(w["mechanism_evaluator_only"]=="crossrole" for w in fit+cal)):
        raise ValueError("F8R67_SOURCE_CONTRACT_FAILED")
    training={}
    scores={"CAL":{},"DEV":{}}
    verdicts=[]
    wall_start=time.perf_counter()
    for seed in SEEDS:
        trained={}
        training[seed]={}
        for arm in TRAINED:
            trained[arm],training[seed][arm]=train_one(arm,seed,fit)
        scores["CAL"][seed]={}
        scores["DEV"][seed]={}
        for arm in ARMS:
            model=trained.get(arm)
            scores["CAL"][seed][arm]=score_one(arm,model,cal)
            scores["DEV"][seed][arm]=score_one(arm,model,dev)
        scores["DEV"][seed][ABLATION]=score_one(ARMS[0],trained[ARMS[0]],dev,
                                               erase_direction=True)
        verdict=judge(scores,seed,freeze,training)
        # Negative proxy intervention and true cold reinitialization with the
        # *same trained weights* and no hidden mutable model memory.
        proxy=proxy_negative(dev)
        proxy_ok=True
        cold_ok=True
        for arm in ARMS:
            original=scores["DEV"][seed][arm]
            model=trained.get(arm)
            def stable_signature(row):
                return [(x["world"],x["B"],x["A_return"],
                         x["stable"],x["state_hash"]) for x in row["per_world"]]
            proxy_ok &= stable_signature(original)==stable_signature(
                score_one(arm,model,proxy))
            cold_ok &= stable_signature(original)==stable_signature(
                score_one(arm,model,dev))
        verdict["checks"]["proxy_invariance"]=bool(proxy_ok)
        verdict["checks"]["cold_restore_exact"]=bool(cold_ok)
        verdict["all_DEV_frozen_checks"]=all(verdict["checks"].values())
        verdicts.append(verdict)
    passed=all(x["all_DEV_frozen_checks"] for x in verdicts)
    return {
        "schema":SCHEMA,"freeze_blob":FREEZE_BLOB,"parent":PARENT,
        "classification":freeze["classification"],
        "source":{"FIT":len(fit),"CAL":len(cal),"DEV":len(dev),
                  "switched_DEV":48,"unfamiliar_crossrole_DEV":24,
                  "protected_OOD_generated":0},
        "training":{str(seed):training[seed] for seed in SEEDS},
        "scores":{panel:{str(seed):{
              arm:{k:v for k,v in scores[panel][seed][arm].items()
                   if k!="per_world"}
              for arm in scores[panel][seed]}
              for seed in SEEDS} for panel in ("CAL","DEV")},
        "verdicts":verdicts,
        "DEV_admission":"DEV_TRAINED_MECHANISM_SIGNAL_ONLY" if passed else "DEV_NO_GO",
        "independent_scientific_pass":False,
        "CUDA_used":False,"original_external_firstuse":False,
        "deployment":"NONE",
        "whole_process_wall_seconds":time.perf_counter()-wall_start
    }


def retain_ci_receipt(result):
    if os.getenv("GITHUB_ACTIONS")!="true":
        return
    dest=ROOT/"local_runs/p85mr_f8r67_trained_protected_memory_dev.json"
    dest.parent.mkdir(exist_ok=True)
    if dest.exists():
        raise FileExistsError("F8R67_REFUSE_OVERWRITE_ORIGINAL")
    dest.write_text(json.dumps(result,sort_keys=True,indent=2,
                               allow_nan=False)+"\n")


def deny_external_firstuse(*_args,**_kwargs):
    raise PermissionError("F8R67_EXTERNAL_SCIENTIFIC_FIRSTUSE_DENIED")
