"""F8-R65: prospective FIT-only trained two-bank causal memory vs matched controls.

Read frozen JSON before changing this. All scored DEV worlds are new and
completely disjoint from R64. This is author-synthetic development; independent
scientific acceptance, external OOD and GPU first-use are not authorized.

Each inference predicts terminal consequence BEFORE factual action feedback.
The 33-float fast state is two 4x4 banks plus a persistent scalar bank gate.
No private role-change, phase flag, hidden effect, proxy or future label enters
prediction, routing or training. All trainable arms receive the same FIT draws
and identical 96 AdamW steps per model seed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/p85mr_f8r65_trained_two_bank_retention_freeze.json"
FREEZE_BLOB = "73588768695ab76a671e9997a50778942b21f00a"
SCHEMA = "ttt-hybrid.f8r65-two-bank-trained-retention-dev-freeze.v1"
PARENT = "fed5b9b1bcd58ae151b62d96b44dfdf0665695cc"
SALT = {"FIT":23710000,"CAL":23720000,"DEV":23730000,"PROXY_NEGATIVE":23740000}
COUNT = {"FIT":96,"CAL":32,"DEV":64}
SEEDS = (985301,985307)
TRAINED = ("directional","nondirectional_surprise","abs_direction_ablation","random_negative")
FIXED = "fixed_two_bank_nearest"
ARMS = TRAINED + (FIXED,)
D = 4
N = 48
STEPS = 96
BATCH = 12
LR = 0.02
WD = 0.005
WRITE_GAIN = 0.55
P0 = 0.05


def git_blob(data):
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()


def locked():
    raw = FREEZE.read_bytes()
    if git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("F8R65_FROZEN_BLOB_DRIFT")
    f=json.loads(raw)
    if (f["schema"]!=SCHEMA
            or f["stage"]!="PROSPECTIVE_FREEZE_BEFORE_R65_SOURCE_IMPLEMENTATION_OR_SCORING"
            or f["parent"]["base_R63_sha"]!=PARENT
            or f["parent"]["R64_source_and_DEV_spent"] is not True
            or f["source"]["salts"]!=SALT
            or f["source"]["worlds"]!=COUNT
            or f["source"]["event_count"]!=N
            or tuple(f["train"]["model_seeds"])!=SEEDS
            or f["train"]["optimizer_steps_per_seed_per_trainable_arm"]!=STEPS
            or f["train"]["FIT_world_batch"]!=BATCH
            or f["train"]["lr"]!=LR
            or f["train"]["weight_decay"]!=WD
            or f["architectures"]["fast_state_floats_per_world"]!=33
            or f["architectures"]["theta_count_per_trainable_arm"]!=5
            or set(f["architectures"]["trained_arms"])!=set(TRAINED)
            or f["metrics"]["both_seeds_required"] is not True
            or f["metrics"]["post_score_tuning_allowed"] is not False
            or any(f["authority"].values())):
        raise ValueError("F8R65_FROZEN_PROTOCOL_DRIFT")
    return f


def worlds(panel):
    if panel not in COUNT:
        raise PermissionError("F8R65_PROTECTED_OR_UNREGISTERED_SOURCE_DENIED")
    rng=np.random.default_rng(SALT[panel])
    n=COUNT[panel]
    switched=np.array([False]*(n//2)+[True]*(n//2))
    rng.shuffle(switched)
    roles=np.arange(N,dtype=np.int64)%D
    result=[]
    for idx in range(n):
        law_a=rng.normal(0,.75,(D,D)) + np.eye(D)*.25
        law_b=law_a.copy()
        # Physically different from R64: only the first two terminal effect
        # coordinates of roles 0/1 rotate 90 degrees, not full sign reversal.
        if switched[idx]:
            law_b[:2,:2]=np.stack((-law_a[1,:2],law_a[0,:2]),axis=0)
        observed=np.empty((N,D),dtype=np.float64)
        for t,role in enumerate(roles):
            selected=law_b if 16<=t<32 else law_a
            observed[t]=selected[:,role]+rng.normal(0,.06,D)
        proxy=rng.normal(0,.8,(N,D))
        result.append({"panel":panel,"index":idx,"roles":roles.copy(),
                       "effects":observed,"proxy":proxy,
                       "switched_evaluator_only":bool(switched[idx])})
    return result


def proxy_negative(dev):
    if len(dev)!=COUNT["DEV"] or any(w["panel"]!="DEV" for w in dev):
        raise PermissionError("F8R65_PROXY_REQUIRES_ORIGINAL_DEV")
    rng=np.random.default_rng(SALT["PROXY_NEGATIVE"])
    return [{**w, "proxy":rng.normal(0,3,(N,D))} for w in dev]


def tensor_batch(worlds_source,indices):
    batch=[worlds_source[int(i)] for i in indices]
    if not batch or len(batch)>COUNT["FIT"]:
        raise ValueError("F8R65_BAD_BATCH")
    a=np.stack([w["roles"] for w in batch])
    y=np.stack([w["effects"] for w in batch])
    return (torch.from_numpy(a.astype(np.int64)),
            torch.from_numpy(y.astype(np.float64)))


def _cos(a,b):
    return F.cosine_similarity(a,b,dim=-1,eps=1e-9)


def _random_features(panel,indices,t):
    result=[]
    for i in indices:
        fields=[]
        for k in range(2):
            raw=hashlib.sha256(f"F8R65_RANDOM_V1/{panel}/{i}/{t}/{k}".encode()).digest()
            fields.append(2*(int.from_bytes(raw[:8],"little") / (1<<64)) -1)
        result.append(fields)
    return torch.tensor(result,dtype=torch.float64)


def features(arm,pred,old0,old1,y,panel,indices,t):
    residual=torch.log1p((y-pred).square().mean(-1))
    magnitude=torch.log1p(pred.square().mean(-1))
    if arm=="directional":
        extra=torch.stack((_cos(old0,y),_cos(old1,y)),dim=-1)
    elif arm=="abs_direction_ablation":
        extra=torch.stack((_cos(old0,y).abs(),_cos(old1,y).abs()),dim=-1)
    elif arm=="nondirectional_surprise":
        extra=torch.stack((torch.log1p((old0-y).square().sum(-1)),
                           torch.log1p((old1-y).square().sum(-1))),dim=-1)
    elif arm=="random_negative":
        extra=_random_features(panel,indices,t).to(y.device)
    else:
        raise ValueError("F8R65_UNREGISTERED_FEATURE_MODE")
    return torch.cat((torch.ones_like(residual)[:,None],
                      residual[:,None],magnitude[:,None],extra),dim=-1)


def rollout(arm,theta,roles,effects,panel,indices,details=False):
    if arm not in ARMS:
        raise ValueError("F8R65_UNREGISTERED_ARM")
    batch,steps=roles.shape
    if (steps!=N or effects.shape!=(batch,N,D) or
            roles.dtype!=torch.int64 or effects.dtype!=torch.float64
            or not bool(torch.isfinite(effects).all()) or
            not bool(((roles>=0)&(roles<D)).all()) or
            panel not in COUNT):
        raise ValueError("F8R65_MALFORMED_SELECTED_FACTS")
    if arm in TRAINED and (theta is None or theta.shape!=(5,)):
        raise ValueError("F8R65_BAD_TRAINED_GATE")
    if arm==FIXED and theta is not None:
        raise ValueError("F8R65_FIXED_ARM_HAS_PARAMETERS")
    w0=effects.new_zeros((batch,D,D))
    w1=effects.new_zeros((batch,D,D))
    p=effects.new_full((batch,),P0)
    losses=[]
    for t in range(N):
        role=roles[:,t]
        action=F.one_hot(role,num_classes=D).to(effects.dtype)
        v0=(w0*action[:,None,:]).sum(-1)
        v1=(w1*action[:,None,:]).sum(-1)
        pred=(1-p[:,None])*v0+p[:,None]*v1
        observed=effects[:,t]
        losses.append((pred-observed).square().mean(-1))
        if arm==FIXED:
            diff=(v0-observed).square().mean(-1)-(v1-observed).square().mean(-1)
            new_p=torch.sigmoid(torch.clamp(diff*4,-8,8))
        else:
            feat=features(arm,pred,v0,v1,observed,panel,indices,t)
            new_p=torch.sigmoid(feat@theta)
        p=new_p
        # No in-place writes: fully differentiable through 48 selected events.
        w0=w0+((1-p)*WRITE_GAIN)[:,None,None]*(observed-v0)[:,:,None]*action[:,None,:]
        w1=w1+(p*WRITE_GAIN)[:,None,None]*(observed-v1)[:,:,None]*action[:,None,:]
    matrix=torch.stack(losses,dim=1)
    if not details:
        return matrix.mean()
    raw=torch.cat((w0.reshape(batch,-1),w1.reshape(batch,-1),p[:,None]),dim=1)
    if raw.shape!=(batch,33) or not bool(torch.isfinite(raw).all()):
        raise ValueError("F8R65_FAST_STATE_DRIFT")
    return {"errors":matrix.detach().cpu().numpy(),
            "state":raw.detach().cpu().numpy(),
            "writes":N,"fast_state_floats":33}


def train_one(arm,seed,fit):
    if arm not in TRAINED or seed not in SEEDS or len(fit)!=COUNT["FIT"]:
        raise ValueError("F8R65_TRAIN_FIT_SEED_ONLY")
    t0=time.perf_counter()
    g=torch.Generator(device="cpu").manual_seed(seed)
    theta=nn.Parameter((torch.rand(5,generator=g,dtype=torch.float64)-.5)*.1)
    optimizer=torch.optim.AdamW([theta],lr=LR,weight_decay=WD)
    draws=np.random.default_rng(23719999+seed).integers(0,len(fit),(STEPS,BATCH))
    last=None
    for ids in draws:
        roles,effects=tensor_batch(fit,ids)
        optimizer.zero_grad(set_to_none=True)
        loss=rollout(arm,theta,roles,effects,"FIT",ids)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R65_NONFINITE_TRAINING")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta],1.0)
        optimizer.step()
        last=float(loss.detach())
    weights=theta.detach().cpu().clone()
    if weights.numel()!=5 or not bool(torch.isfinite(weights).all()):
        raise ValueError("F8R65_BAD_WEIGHTS")
    return {"arm":arm,"seed":seed,"theta":weights,
            "FIT_last_batch_MSE":last,
            "train_wall_seconds":time.perf_counter()-t0,
            "optimizer_steps":STEPS,
            "selected_factual_events":STEPS*BATCH*N,
            "weights_sha256":hashlib.sha256(weights.numpy().astype("<f8").tobytes()).hexdigest()}


def score_one(arm,theta,worlds_dev):
    if not worlds_dev or len(worlds_dev)>COUNT["FIT"]:
        raise ValueError("F8R65_SCORE_SOURCE_REJECTED")
    panel=worlds_dev[0]["panel"]
    if any(w["panel"]!=panel for w in worlds_dev):
        raise ValueError("F8R65_MIXED_PANELS")
    roles,effects=tensor_batch(worlds_dev,range(len(worlds_dev)))
    with torch.no_grad():
        result=rollout(arm,theta,roles,effects,panel,list(range(len(worlds_dev))),True)
    values=[]
    for idx,w in enumerate(worlds_dev):
        e=result["errors"][idx]
        values.append({
            "index":w["index"],
            "switched":w["switched_evaluator_only"],
            "B_changed":float(np.mean(e[16:32][np.arange(16)%4<2])),
            "A_return_first":float(np.mean(e[32:36])),
            "stable_all":float(np.mean(e)),
            "fast_state_sha256":hashlib.sha256(
                result["state"][idx].astype("<f8").tobytes()).hexdigest()
        })
    switched=[v for v in values if v["switched"]]
    stable=[v for v in values if not v["switched"]]
    if len(switched)!=len(stable):
        raise ValueError("F8R65_UNBALANCED_SCORE_WORLD")
    B=float(np.mean([v["B_changed"] for v in switched]))
    A=float(np.mean([v["A_return_first"] for v in switched]))
    S=float(np.mean([v["stable_all"] for v in stable]))
    return {"B_changed_MSE":B,"A_return_first_MSE":A,
            "stable_world_MSE":S,"joint":B+A+S,
            "per_world":values,"writes_per_world":N,
            "fast_state_floats":33}


def judge(result,seed):
    scored=result["DEV"][seed]
    eligible=("nondirectional_surprise","random_negative",FIXED)
    best=min(eligible,key=lambda name:scored[name]["joint"])
    cand=scored["directional"]
    ctrl=scored[best]
    abl=scored["abs_direction_ablation"]
    checks={
      "joint_gain_at_least_5pct":1-cand["joint"]/ctrl["joint"]>=.05,
      "B_change_not_worse_5pct":cand["B_changed_MSE"]/ctrl["B_changed_MSE"]-1<=.05,
      "A_return_not_worse_5pct":cand["A_return_first_MSE"]/ctrl["A_return_first_MSE"]-1<=.05,
      "stable_not_worse_5pct":cand["stable_world_MSE"]/ctrl["stable_world_MSE"]-1<=.05,
      "direction_ablation_5pct_worse":abl["joint"]/cand["joint"]-1>=.05
    }
    # To bootstrap an equally weighted three-metric joint across switched
    # and stable cohorts, each switched world contributes 2*(B+A), each
    # stable world contributes 2*S. Means of 64 reproduce the frozen joint.
    deltas=[]
    for c,b in zip(cand["per_world"],ctrl["per_world"]):
        if c["index"]!=b["index"] or c["switched"]!=b["switched"]:
            raise ValueError("F8R65_MISALIGNED_PAIRED_WORLDS")
        if c["switched"]:
            d=2*((c["B_changed"]+c["A_return_first"])-
                 (b["B_changed"]+b["A_return_first"]))
        else:
            d=2*(c["stable_all"]-b["stable_all"])
        deltas.append(d)
    rng=np.random.default_rng(23749999+seed)
    draws=rng.integers(0,len(deltas),(1000,len(deltas)))
    upper=float(np.quantile(np.asarray(deltas)[draws].mean(1),.975))
    checks["paired_bootstrap_upper_negative"]=upper<0
    return {"seed":seed,"strongest_control":best,"checks":checks,
            "joint_relative_gain":1-cand["joint"]/ctrl["joint"],
            "B_relative_regression":cand["B_changed_MSE"]/ctrl["B_changed_MSE"]-1,
            "A_return_relative_regression":cand["A_return_first_MSE"]/ctrl["A_return_first_MSE"]-1,
            "stable_relative_regression":cand["stable_world_MSE"]/ctrl["stable_world_MSE"]-1,
            "direction_ablation_relative_decline":abl["joint"]/cand["joint"]-1,
            "paired_bootstrap_upper_95pct":upper,
            "all_scientific_DEV_conditions_met":all(checks.values())}


def run():
    frozen=locked()
    fit=worlds("FIT")
    cal=worlds("CAL")
    dev=worlds("DEV")
    proxy=proxy_negative(dev)
    trained={}
    scores={"CAL":{},"DEV":{}}
    time_start=time.perf_counter()
    for seed in SEEDS:
        trained[seed]={}
        for arm in TRAINED:
            trained[seed][arm]=train_one(arm,seed,fit)
        scores["CAL"][seed]={}
        scores["DEV"][seed]={}
        for arm in ARMS:
            theta=trained[seed][arm]["theta"] if arm in TRAINED else None
            scores["CAL"][seed][arm]=score_one(arm,theta,cal)
            scores["DEV"][seed][arm]=score_one(arm,theta,dev)
    checks={}
    for seed in SEEDS:
        checks[seed]={"proxy_invariance":True,"cold_restore_exact":True}
        for arm in ARMS:
            theta=trained[seed][arm]["theta"] if arm in TRAINED else None
            clean=scores["DEV"][seed][arm]
            proxy_result=score_one(arm,theta,proxy)
            repeated=score_one(arm,theta,dev)
            def receipt(x):
                return [(v["index"],v["B_changed"],v["A_return_first"],v["stable_all"],
                         v["fast_state_sha256"]) for v in x["per_world"]]
            checks[seed]["proxy_invariance"] &= receipt(clean)==receipt(proxy_result)
            checks[seed]["cold_restore_exact"] &= receipt(clean)==receipt(repeated)
    verdicts=[judge(scores,seed) for seed in SEEDS]
    for v in verdicts:
        v["checks"].update(checks[v["seed"]])
        v["all_scientific_DEV_conditions_met"]=all(v["checks"].values())
    success=all(v["all_scientific_DEV_conditions_met"] for v in verdicts)
    return {"schema":SCHEMA,"frozen_blob":FREEZE_BLOB,
            "parent":PARENT,"classification":frozen["classification"],
            "source":{"FIT":len(fit),"CAL":len(cal),"DEV":len(dev),
                      "switched_DEV":sum(w["switched_evaluator_only"] for w in dev)},
            "train_receipts":{
               str(seed):{a:{k:v for k,v in trained[seed][a].items() if k!="theta"}
                          for a in TRAINED}
               for seed in SEEDS},
            "score":{panel:{str(seed):{a:{k:v for k,v in scores[panel][seed][a].items()
                                           if k!="per_world"}
                                 for a in ARMS}
                           for seed in SEEDS}
                     for panel in ("CAL","DEV")},
            "judgments":verdicts,
            "DEV_frozen_admission":"DEV_TRAINED_MECHANISM_SIGNAL_ONLY" if success
                                   else "DEV_NO_GO",
            "no_R64_DEV_reuse":True,
            "no_protected_OOD_source":True,
            "external_firstuse":False,
            "GPU_firstuse":False,
            "independent_scientific_pass":False,
            "deployment":"NONE",
            "whole_run_wall_seconds":time.perf_counter()-time_start}


def retain_ci_receipt(result):
    if os.getenv("GITHUB_ACTIONS")!="true":
        return
    path=ROOT/"local_runs/p85mr_f8r65_trained_two_bank_retention_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R65_REFUSE_RESULT_OVERWRITE")
    path.write_text(json.dumps(result,sort_keys=True,allow_nan=False,indent=2)+"\n")


def deny_external_firstuse(*args,**kwargs):
    raise PermissionError("F8R65_PROTECTED_SCIENCE_FIRSTUSE_DENIED")
