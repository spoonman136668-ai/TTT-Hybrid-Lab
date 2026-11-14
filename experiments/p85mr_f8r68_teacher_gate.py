"""F8-R68: prospective factual teacher imitation for causal memory switch.

Fresh author-synthetic DEV only. Teacher switch decisions are reconstructed
exclusively from factual post-action terminal evidence and the original R66
public-state memory rules. No private phase, law, regime, future consequence,
or irrelevant sensor proxy is ever an input to any learned selector.
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
FREEZE=ROOT/"experiments/p85mr_f8r68_teacher_gate_freeze.json"
FREEZE_BLOB="cfd3ee9a795a5207835657680cb3836c0cf24890"
SCHEMA="ttt-hybrid.f8r68-factual-teacher-bank-gate-freeze.v1"
PARENT="efe7081d1a32e5ce96236bdf0a5c9024658d9afb"
SALT={"FIT":24010000,"CAL":24020000,"DEV":24030000,"PROXY_NEGATIVE":24040000}
COUNTS={"FIT":128,"CAL":32,"DEV":96}
SEEDS=(985509,985519)
TRAINED=("FULL_ORDERED_MLP","ORDER_INVARIANT_MLP","FULL_ORDERED_LINEAR","SHUFFLED_LABEL_MLP")
TEACHER="FIXED_R66_ORACLE_TEACHER"
ARMS=TRAINED+(TEACHER,)
G=0.65
CONF=0.20
NOVELTY=0.75
MARGIN=0.15
N=48
D=4
FAST=36
INPUT=10
UPDATES=128
BATCH=16
LR=.015
WD=.001
B_IDX=(20,21,24,25,28,29)
A_IDX=(36,37,40,41,44,45)
BOOT_SEED=24049999


def git_blob(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def locked():
    raw=FREEZE.read_bytes()
    if git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("F8R68_FREEZE_BLOB_DRIFT")
    f=json.loads(raw)
    src,teacher,train,metrics=f["source"],f["teacher"],f["training"],f["metrics"]
    if (f["schema"]!=SCHEMA
            or f["stage"]!="PROSPECTIVE_FREEZE_BEFORE_R68_IMPLEMENTATION_AND_SCORING"
            or f["parent"]["exact_R66_SHA"]!=PARENT
            or f["parent"]["R67_result"]!="DEV_NO_GO"
            or not f["parent"]["no_spent_cohorts_or_test_tuning"]
            or src["salts"]!=SALT or src["worlds"]!=COUNTS
            or src["scored_B_indices"]!=list(B_IDX)
            or src["scored_A_return_indices"]!=list(A_IDX)
            or train["seeds"]!=list(SEEDS)
            or train["optimizer_steps_per_seed_per_trainable_arm"]!=UPDATES
            or train["FIT_world_batch_per_step"]!=BATCH
            or train["lr"]!=LR or train["weight_decay"]!=WD
            or f["arms"]["shared_fast_state_floats"]!=FAST
            or (teacher["no_private_flags"] is not True)
            or metrics["bootstrap"]["seed"]!=BOOT_SEED
            or metrics["bootstrap"]["resamples"]!=2000
            or metrics["post_result_tuning_forbidden"] is not True
            or any(f["authority"].values())
            or r66.GAIN!=G or r66.NOVELTY!=NOVELTY
            or r66.CONF!=CONF or r66.MARGIN!=MARGIN):
        raise ValueError("F8R68_FROZEN_PROTOCOL_DRIFT")
    return f


def worlds(panel):
    if panel not in COUNTS:
        raise PermissionError("F8R68_PROTECTED_OR_SPENT_SOURCE_DENIED")
    rng=np.random.default_rng(SALT[panel])
    count=COUNTS[panel]
    modes=["stable"]*(count//2)
    switched=count//2
    modes+=["rotation"]*(switched//2)
    modes+=(["gain"] if panel!="DEV" else ["swap"])*(switched//2)
    modes=[modes[i] for i in rng.permutation(count)]
    roles=np.arange(N,dtype=np.int64)%D
    result=[]
    for index,mechanism in enumerate(modes):
        law_a=rng.normal(0,.75,(D,D))+np.eye(D)*.25
        law_b=law_a.copy()
        if mechanism=="rotation":
            law_b[:2,:2]=np.stack((-law_a[1,:2],law_a[0,:2]),axis=0)
        elif mechanism=="gain":
            law_b[:2,:2]=1.35*law_a[:2,:2]
        elif mechanism=="swap":
            law_b[:,0]=law_a[:,1]
            law_b[:,1]=law_a[:,0]
        selected=np.empty((N,D),dtype=np.float64)
        for t,role in enumerate(roles):
            law=law_b if 16<=t<32 else law_a
            selected[t]=law[:,role]+rng.normal(0,.04,D)
        proxy=rng.normal(0,.8,(N,D))
        result.append({"panel":panel,"index":index,"roles":roles.copy(),
                       "effects":selected,"proxy":proxy,
                       "evaluator_family_only":mechanism,
                       "evaluator_switched_only":mechanism!="stable"})
    return result


def negative_proxy(dev):
    if len(dev)!=COUNTS["DEV"] or any(w["panel"]!="DEV" for w in dev):
        raise PermissionError("F8R68_PROXY_SOURCE_MUST_BE_DEV")
    rng=np.random.default_rng(SALT["PROXY_NEGATIVE"])
    return [{**w,"proxy":rng.normal(0,3,(N,D))} for w in dev]


def observable_features(active_norm,other_norm,active_error,other_error,mode):
    vals=(float(active_norm),float(other_norm),float(active_error),
          float(other_error))
    if any(not np.isfinite(v) or v<0 for v in vals):
        raise ValueError("F8R68_INVALID_OBSERVED_FEATURE")
    na,no,ea,eo=vals
    if mode=="ordered":
        return np.array((np.log1p(na),np.log1p(no),np.log1p(ea),np.log1p(eo),
                         ea-eo,ea-NOVELTY,na-CONF,no-CONF,ea-eo-MARGIN,1),
                        dtype=np.float64)
    if mode=="invariant":
        return np.array((np.log1p(min(na,no)),np.log1p(max(na,no)),
                         np.log1p(min(ea,eo)),np.log1p(max(ea,eo)),
                         abs(ea-eo),abs(ea-NOVELTY),abs(na-CONF),
                         abs(no-CONF),abs(ea-eo-MARGIN),1),dtype=np.float64)
    raise ValueError("F8R68_INVALID_FEATURE_MODE")


def teacher_trace(world):
    """Teacher labels are factual post-event decisions, not evaluator flags."""
    roles=world["roles"]
    y=world["effects"]
    if (roles.shape!=(N,) or roles.dtype!=np.int64
            or not np.array_equal(roles,np.arange(N)%D)
            or y.shape!=(N,D) or not np.isfinite(y).all()):
        raise ValueError("F8R68_MALFORMED_SELECTED_FACTS")
    banks=[np.zeros((D,D),dtype=np.float64),np.zeros((D,D),dtype=np.float64)]
    selector=np.zeros(D,dtype=np.float64)
    errors=np.zeros(N)
    labels=np.zeros(N,dtype=np.float64)
    ordered=np.zeros((N,INPUT))
    invariant=np.zeros((N,INPUT))
    changes=0
    for t in range(N):
        role=int(roles[t])
        active=int(selector[role])
        old=banks[active][:,role].copy()
        other=banks[1-active][:,role].copy()
        observed=y[t]
        errors[t]=float(np.mean((old-observed)**2))
        na,no=np.linalg.norm(old),np.linalg.norm(other)
        ea,eo=np.linalg.norm(old-observed),np.linalg.norm(other-observed)
        ordered[t]=observable_features(na,no,ea,eo,"ordered")
        invariant[t]=observable_features(na,no,ea,eo,"invariant")
        move=bool((no>=CONF and eo+MARGIN<ea) or
                  (na>=CONF and ea>NOVELTY and no<CONF))
        labels[t]=float(move)
        if move:
            active=1-active
            changes+=1
            selector[role]=float(active)
        banks[active][:,role]+=G*(observed-banks[active][:,role])
    state=np.concatenate([banks[0].reshape(-1),banks[1].reshape(-1),selector])
    return {"labels":labels,"ordered":ordered,"invariant":invariant,
            "errors":errors,"switches":changes,"state_hash":hashlib.sha256(
                state.astype("<f8").tobytes()).hexdigest()}


def teacher_identity(worlds_source, traces):
    for w,t in zip(worlds_source,traces):
        ref=r66.update_role_bank("protected_role_bank",w["roles"],w["effects"])
        if (not np.array_equal(ref["errors"],t["errors"])
                or ref["state_hash"]!=t["state_hash"]
                or ref["bank_switch_events"]!=t["switches"]
                or ref["allocated_fast_floats"]!=FAST
                or ref["factual_selected_events"]!=N):
            raise ValueError("F8R68_TEACHER_NOT_IDENTICAL_TO_QUALIFIED_R66")
    return True


def arm_mode(arm):
    if arm=="ORDER_INVARIANT_MLP":
        return "invariant"
    if arm in TRAINED:
        return "ordered"
    raise ValueError("F8R68_UNREGISTERED_LEARNER")


class BinaryGate(nn.Module):
    def __init__(self,arm,seed):
        super().__init__()
        if arm not in TRAINED or seed not in SEEDS:
            raise ValueError("F8R68_TRAINABLE_ARM_OR_SEED_DENIED")
        offset=TRAINED.index(arm)*997
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+offset)
            self.net=(nn.Linear(INPUT,1).double() if arm=="FULL_ORDERED_LINEAR"
                      else nn.Sequential(nn.Linear(INPUT,16),nn.Tanh(),
                                         nn.Linear(16,1)).double())
        self.arm=arm

    def forward(self,x):
        return self.net(x).squeeze(-1)


def training_arrays(fit_trace,arm,seed):
    mode=arm_mode(arm)
    full=np.stack([x[mode] for x in fit_trace])
    original=np.stack([x["labels"] for x in fit_trace])
    if arm=="SHUFFLED_LABEL_MLP":
        perm=np.random.default_rng(24059999+seed).permutation(original.size)
        labels=original.reshape(-1)[perm].reshape(original.shape)
    else:
        labels=original
    return torch.from_numpy(full),torch.from_numpy(labels)


def train_one(arm,seed,fit_trace):
    if arm not in TRAINED or seed not in SEEDS or len(fit_trace)!=COUNTS["FIT"]:
        raise ValueError("F8R68_FIT_ONLY_SUPERVISION")
    x,y=training_arrays(fit_trace,arm,seed)
    fit_positive=sum(int(t["labels"].sum()) for t in fit_trace)
    fit_total=COUNTS["FIT"]*N
    if fit_positive<=0 or fit_positive==fit_total:
        raise ValueError("F8R68_UNIDENTIFIABLE_TEACHER_TARGET")
    pos_weight=min(12.,max(1.,(fit_total-fit_positive)/fit_positive))
    weight=torch.tensor(pos_weight,dtype=torch.float64)
    model=BinaryGate(arm,seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    draws=np.random.default_rng(24019999+seed).integers(
        0,COUNTS["FIT"],size=(UPDATES,BATCH))
    started=time.perf_counter()
    with torch.no_grad():
        x0=x[draws[0]].reshape(-1,INPUT)
        y0=y[draws[0]].reshape(-1)
        initial=float(F.binary_cross_entropy_with_logits(
            model(x0),y0,pos_weight=weight))
    last=0.0
    for indices in draws:
        bx=x[indices].reshape(-1,INPUT)
        by=y[indices].reshape(-1)
        optimizer.zero_grad(set_to_none=True)
        logit=model(bx)
        loss=F.binary_cross_entropy_with_logits(logit,by,pos_weight=weight)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R68_NONFINITE_FIT_LOSS")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        last=float(loss.detach())
    with torch.no_grad():
        final=float(F.binary_cross_entropy_with_logits(
            model(x0),y0,pos_weight=weight))
    count=sum(p.numel() for p in model.parameters())
    if count not in (193,11):
        raise ValueError("F8R68_PARAMETER_COUNT_DRIFT")
    sha=hashlib.sha256(b"".join(
        p.detach().cpu().numpy().astype("<f8").tobytes()
        for p in model.parameters())).hexdigest()
    return model,{"seed":seed,"arm":arm,"train_wall_seconds":time.perf_counter()-started,
                  "FIT_teacher_label_positive":fit_positive,
                  "FIT_teacher_label_total":fit_total,
                  "FIT_positive_weight":pos_weight,
                  "trainable_global_parameters":count,
                  "persistent_fast_floats":FAST,
                  "optimizer_steps":UPDATES,
                  "FIT_world_batch":BATCH,
                  "FIT_selected_factual_event_exposures":UPDATES*BATCH*N,
                  "BCE_initial":initial,"BCE_last_batch":last,"BCE_final_first_batch":final,
                  "trained_weight_SHA256":sha}


def teacher_state_accuracy(model,source_trace):
    if not source_trace:
        raise ValueError("F8R68_NO_FRESH_TEACHER_TRACES")
    arm=model.arm
    mode=arm_mode(arm)
    x=torch.from_numpy(np.concatenate([z[mode] for z in source_trace]))
    y=np.concatenate([z["labels"] for z in source_trace]).astype(bool)
    with torch.no_grad():
        pred=(model(x).cpu().numpy()>0)
    positive=int(y.sum())
    negative=int((~y).sum())
    if not positive or not negative:
        raise ValueError("F8R68_MISSING_POSITIVE_OR_NEGATIVE_LABELS")
    recall=float(pred[y].mean())
    specificity=float((~pred[~y]).mean())
    return {"teacher_positive":positive,"teacher_negative":negative,
            "positive_recall":recall,"negative_specificity":specificity,
            "balanced_accuracy":(recall+specificity)/2}


def student_rollout(model,source):
    """Independent closed-loop state; no teacher predictions or labels read."""
    if not source or any(w["panel"]!=source[0]["panel"] for w in source):
        raise ValueError("F8R68_BAD_EVAL_SOURCE")
    began=time.perf_counter()
    n=len(source)
    role=np.stack([w["roles"] for w in source])
    outcomes=np.stack([w["effects"] for w in source])
    w0=np.zeros((n,D,D),dtype=np.float64)
    w1=np.zeros((n,D,D),dtype=np.float64)
    selector=np.zeros((n,D),dtype=np.int64)
    errors=np.zeros((n,N),dtype=np.float64)
    switches=np.zeros(n,dtype=np.int64)
    for t in range(N):
        feat=np.zeros((n,INPUT),dtype=np.float64)
        for i in range(n):
            a=int(selector[i,role[i,t]])
            current=(w0 if a==0 else w1)[i,:,role[i,t]].copy()
            other=(w1 if a==0 else w0)[i,:,role[i,t]].copy()
            observed=outcomes[i,t]
            errors[i,t]=float(np.mean((current-observed)**2))
            feat[i]=observable_features(
                np.linalg.norm(current),np.linalg.norm(other),
                np.linalg.norm(current-observed),np.linalg.norm(other-observed),
                arm_mode(model.arm))
        with torch.no_grad():
            flags=(model(torch.from_numpy(feat)).cpu().numpy()>0)
        for i in range(n):
            rr=int(role[i,t])
            if flags[i]:
                selector[i,rr]=1-selector[i,rr]
                switches[i]+=1
            bank=w0 if selector[i,rr]==0 else w1
            bank[i,:,rr]+=G*(outcomes[i,t]-bank[i,:,rr])
    hashes=[hashlib.sha256(np.concatenate([
        w0[i].reshape(-1),w1[i].reshape(-1),
        selector[i].astype(np.float64)]).astype("<f8").tobytes()).hexdigest()
        for i in range(n)]
    return errors,hashes,switches,time.perf_counter()-began


def score(arm,model,src,traces):
    scored_started=time.perf_counter()
    if arm==TEACHER:
        if model is not None:
            raise ValueError("F8R68_TEACHER_NEVER_USES_TRAINED_MODEL")
        errors=np.stack([t["errors"] for t in traces])
        hashes=[t["state_hash"] for t in traces]
        switches=[t["switches"] for t in traces]
        wall=0.
    else:
        errors,hashes,switches,wall=student_rollout(model,src)
    rows=[]
    for w,e,h,nchange in zip(src,errors,hashes,switches):
        rows.append({"world":w["index"],"family":w["evaluator_family_only"],
                     "switched":w["evaluator_switched_only"],
                     "B":float(np.mean(e[list(B_IDX)])),
                     "A_return":float(np.mean(e[list(A_IDX)])),
                     "stable":float(np.mean(e)),
                     "state_hash":h,"bank_switches":int(nchange)})
    sw=[x for x in rows if x["switched"]]
    stable=[x for x in rows if not x["switched"]]
    swap=[x for x in sw if x["family"]=="swap"]
    if len(sw)!=len(stable) or (src[0]["panel"]=="DEV" and len(swap)!=24):
        raise ValueError("F8R68_UNBALANCED_SCORER")
    mean=lambda items,key:float(np.mean([x[key] for x in items]))
    B=mean(sw,"B")
    A=mean(sw,"A_return")
    S=mean(stable,"stable")
    wall=time.perf_counter()-scored_started
    return {"B_MSE":B,"A_return_MSE":A,"stable_MSE":S,
            "joint_MSE":B+A+S,
            "swap_A_return_MSE":mean(swap,"A_return") if swap else None,
            "allocated_fast_floats":FAST,"selected_event_count":len(src)*N,
            "total_bank_switches":int(sum(switches)),"eval_wall_seconds":wall,
            "per_world":rows}


def judge(scores,accuracy,training,seed,freeze):
    d=scores["DEV"][seed]
    cand=d[TRAINED[0]]
    teacher=d[TEACHER]
    eligible=(TRAINED[1],TRAINED[2],TRAINED[3])
    best=min(eligible,key=lambda a:d[a]["joint_MSE"])
    control=d[best]
    change=lambda x,y:x/max(y,1e-12)-1
    improvement=1-cand["joint_MSE"]/max(control["joint_MSE"],1e-12)
    pairs=[]
    for x,y in zip(cand["per_world"],control["per_world"]):
        if x["world"]!=y["world"] or x["switched"]!=y["switched"]:
            raise ValueError("F8R68_PAIRED_WORLD_MISMATCH")
        if x["switched"]:
            pairs.append(2*((x["B"]+x["A_return"])-(y["B"]+y["A_return"])))
        else:
            pairs.append(2*(x["stable"]-y["stable"]))
    boot=freeze["metrics"]["bootstrap"]
    draw=np.random.default_rng(boot["seed"]+seed).integers(
        0,len(pairs),(boot["resamples"],len(pairs)))
    upper=float(np.quantile(np.asarray(pairs)[draw].mean(axis=1),.975))
    a=accuracy["DEV"][seed][TRAINED[0]]
    l=freeze["metrics"]["required_all_both_seeds"]
    checks={
      "teacher_trace_exact_R66":bool(training["teacher_identity"]),
      "teacher_state_balanced_acc":a["balanced_accuracy"]>=l["student_teacher_state_balanced_accuracy_min"],
      "teacher_state_positive_recall":a["positive_recall"]>=l["student_positive_recall_min"],
      "teacher_state_specificity":a["negative_specificity"]>=l["student_negative_specificity_min"],
      "best_trained_joint_gain":improvement>=l["candidate_joint_improvement_over_strongest_trained_min"],
      "A_return_near_fixed_teacher":change(cand["A_return_MSE"],teacher["A_return_MSE"])<=l["A_return_relative_regression_vs_fixed_teacher_max"],
      "B_near_fixed_teacher":change(cand["B_MSE"],teacher["B_MSE"])<=l["B_relative_regression_vs_fixed_teacher_max"],
      "stable_near_fixed_teacher":change(cand["stable_MSE"],teacher["stable_MSE"])<=l["stable_relative_regression_vs_fixed_teacher_max"],
      "swap_return_near_fixed_teacher":change(cand["swap_A_return_MSE"],teacher["swap_A_return_MSE"])<=l["unseen_swap_A_return_relative_regression_vs_fixed_teacher_max"],
      "paired_joint_upper_negative":upper<0,
      "FIT_optimizer_and_event_parity":all(
          training["receipts"][seed][arm]["optimizer_steps"]==UPDATES and
          training["receipts"][seed][arm]["FIT_selected_factual_event_exposures"]==UPDATES*BATCH*N
          for arm in TRAINED)
    }
    return {"seed":seed,"best_trained_control":best,
            "relative_joint_improvement":improvement,
            "A_return_vs_fixed_regression":change(cand["A_return_MSE"],teacher["A_return_MSE"]),
            "B_vs_fixed_regression":change(cand["B_MSE"],teacher["B_MSE"]),
            "stable_vs_fixed_regression":change(cand["stable_MSE"],teacher["stable_MSE"]),
            "swap_return_vs_fixed_regression":change(cand["swap_A_return_MSE"],teacher["swap_A_return_MSE"]),
            "bootstrap_upper95_candidate_minus_best":upper,"checks":checks}


def run():
    started=time.perf_counter()
    freeze=locked()
    src={panel:worlds(panel) for panel in COUNTS}
    if any(sum(int(x["evaluator_switched_only"]) for x in arr)!=len(arr)//2
           for arr in src.values()):
        raise ValueError("F8R68_SOURCE_BALANCE_VIOLATION")
    if (sum(x["evaluator_family_only"]=="swap" for x in src["DEV"])!=24
        or any(x["evaluator_family_only"]=="swap" for x in src["FIT"]+src["CAL"])):
        raise ValueError("F8R68_HELD_OUT_CAUSAL_FAMILY_BROKEN")
    # The independent DEV outcomes and teacher labels are not inspected
    # until ALL seeds and comparator arms have frozen their FIT-trained weights.
    traces={panel:[teacher_trace(w) for w in src[panel]]
            for panel in ("FIT","CAL")}
    teacher_identity_all=all(teacher_identity(src[panel],traces[panel])
                             for panel in ("FIT","CAL"))
    training={"teacher_identity":teacher_identity_all,"receipts":{}}
    accuracy={"CAL":{},"DEV":{}}
    scores={"CAL":{},"DEV":{}}
    negatives={}
    verdicts=[]
    models_by_seed={}
    for seed in SEEDS:
        models={}
        training["receipts"][seed]={}
        for arm in TRAINED:
            models[arm],training["receipts"][seed][arm]=train_one(
                arm,seed,traces["FIT"])
        models_by_seed[seed]=models
    traces["DEV"]=[teacher_trace(w) for w in src["DEV"]]
    training["teacher_identity"] &= teacher_identity(src["DEV"],traces["DEV"])
    negative=negative_proxy(src["DEV"])
    for seed in SEEDS:
        models=models_by_seed[seed]
        for panel in ("CAL","DEV"):
            accuracy[panel][seed]={}
            scores[panel][seed]={}
            for arm in ARMS:
                m=models.get(arm)
                if m is not None:
                    accuracy[panel][seed][arm]=teacher_state_accuracy(m,traces[panel])
                scores[panel][seed][arm]=score(arm,m,src[panel],traces[panel])
        controls={}
        for arm in ARMS:
            model=models.get(arm)
            original=scores["DEV"][seed][arm]
            negative_trace=[teacher_trace(w) for w in negative] if arm==TEACHER else traces["DEV"]
            proxy_score=score(arm,model,negative,negative_trace)
            repeat=score(arm,model,src["DEV"],traces["DEV"])
            signature=lambda x:[(v["world"],v["B"],v["A_return"],
                v["stable"],v["state_hash"]) for v in x["per_world"]]
            controls[arm]={
                "proxy_invariance":signature(original)==signature(proxy_score),
                "cold_replay_exact":signature(original)==signature(repeat)}
        negatives[seed]=controls
        verdict=judge(scores,accuracy,training,seed,freeze)
        verdict["checks"]["proxy_negative_invariance"]=all(
            controls[a]["proxy_invariance"] for a in ARMS)
        verdict["checks"]["cold_start_replay_exact"]=all(
            controls[a]["cold_replay_exact"] for a in ARMS)
        verdict["all_frozen_checks"]=all(verdict["checks"].values())
        verdicts.append(verdict)
    passed=all(v["all_frozen_checks"] for v in verdicts)
    return {
        "schema":SCHEMA,"freeze_blob":FREEZE_BLOB,"parent":PARENT,
        "classification":freeze["classification"],
        "source":{"FIT":len(src["FIT"]),"CAL":len(src["CAL"]),
                  "DEV":len(src["DEV"]),"switched_DEV":48,
                  "held_out_swap_DEV":24,"protected_external_OOD":0},
        "FIT_teacher_positive_count":sum(int(x["labels"].sum()) for x in traces["FIT"]),
        "training":{str(seed):training["receipts"][seed] for seed in SEEDS},
        "accuracy":{panel:{str(seed):accuracy[panel][seed] for seed in SEEDS}
                    for panel in ("CAL","DEV")},
        "scores":{panel:{str(seed):{arm:{
                    k:v for k,v in scores[panel][seed][arm].items()
                    if k!="per_world"} for arm in ARMS}
                 for seed in SEEDS}
                 for panel in ("CAL","DEV")},
        "verdicts":verdicts,
        "DEV_admission":"DEV_TEACHER_IMITATION_MECHANISM_SIGNAL_ONLY" if passed
                        else "DEV_NO_GO",
        "independent_scientific_pass":False,
        "CUDA_used":False,"external_firstuse":False,
        "deployment":"NONE",
        "whole_process_wall_seconds":time.perf_counter()-started
    }


def retain_ci_receipt(result):
    if os.getenv("GITHUB_ACTIONS")!="true":
        return
    output=ROOT/"local_runs/p85mr_f8r68_teacher_gate_dev.json"
    output.parent.mkdir(exist_ok=True)
    if output.exists():
        raise FileExistsError("F8R68_NEVER_OVERWRITE_SPENT_DEV")
    output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")


def deny_external_firstuse(*_args,**_kwargs):
    raise PermissionError("F8R68_EXTERNAL_FIRST_USE_DENIED")
