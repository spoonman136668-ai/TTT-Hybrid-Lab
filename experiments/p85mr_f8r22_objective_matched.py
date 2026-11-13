"""R22: objective-matched native/GRU/SSM causal memory, source-neutral.

All source effects are from the pre-existing author-generated mathematical R3
law. Not external first use, not F7 replay, not a scientific PASS.
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

from experiments import p85mr_f8r20_factorial_transfer as r20
from experiments import p85mr_f8r19_balanced_return as r19
from experiments import p85mr_f8r18_matched_aba as r18
from p85mr_dynamics_baselines import new_baseline, prequential as conventional_prequential

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r22_objective_matched_freeze.json"
FREEZE_BLOB="c7c5cd9bb3e3590a58ccff92439c763305659651"
SCHEMA="ttt-hybrid.f8r22-matched-objective-recurrent.v1"
COHORTS={"fit":(19710000,32,8),"dev":(19720000,8,8),
         "id":(19730000,12,8),"ood":(19740000,12,16)}
ARMS=("native_focus20","native_uniform","native_returnonly",
      "gru_focus20","ssm_focus20","mlp_focus20")
SEEDS=(985201,985203,985207)
GOALS=(-.2,.2)
UPDATES=384
BATCH=8
CHECKPOINTS=(96,192,288,384)
BOOT=10000
BOOT_SEED=19749999
LAMBDA=1e-9


def locked():
    raw=FREEZE.read_bytes()
    actual=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
    if actual!=FREEZE_BLOB:
        raise ValueError("F8R22_FROZEN_BLOB_DRIFT")
    f=json.loads(raw)
    w,m,s,a=f["worlds"],f["models"],f["score"],f["authority"]
    if (f["schema"]!=SCHEMA
        or f["status"]!="FROZEN_BEFORE_IMPLEMENTATION_TRAINING_AND_RESULTS"
        or f["classification"]!="SOURCE_NEUTRAL_AUTHOR_GENERATED_DEVELOPMENT_ONLY"
        or f["parent"]["sha"]!="dbed2b1bca5a07d74634929b8ec1d84efc21da77"
        or f["parent"]["r20_source_git_blob"]!="78700dabd9b4301318148b4536e9df21c64d2617"
        or f["parent"]["r20_freeze_git_blob"]!="8a181c6fd71c9dc6d4c4d640396d19920eb74175"
        or m["arms"]!=list(ARMS)
        or m["seeds"]!=list(SEEDS)
        or m["model_count"]!=18
        or m["updates_per_model"]!=UPDATES
        or m["total_optimizer_updates"]!=6912
        or m["batch_worlds"]!=BATCH
        or m["sampler_seed"]!=985299
        or m["evaluation_checkpoints"]!=list(CHECKPOINTS)
        or m["same_focused_objective_native_GRU_SSM_MLP"] is not True
        or w["matched_B_groups"]!={k:v[1] for k,v in COHORTS.items()}
        or w["seed_bases"]!={k:v[0] for k,v in COHORTS.items()}
        or w["B_steps"]!={k:v[2] for k,v in COHORTS.items()}
        or w["FIT_episodes"]!=256
        or w["FIT_selected_effects"]!=2560
        or w["independent_crossed_goal"]!=list(GOALS)
        or s["candidate"]!="native_focus20"
        or s["paired_bootstrap_world_groups"]!=12
        or s["bootstrap_resamples"]!=BOOT
        or s["bootstrap_seed"]!=BOOT_SEED
        or s["independent_scientific_pass"] is not False
        or not all(v is False for v in a.values())):
        raise ValueError("F8R22_SCIENCE_OR_AUTHORITY_DRIFT")
    r20.locked()
    if r20.git_blob((ROOT/"experiments/p85mr_f8r21_repro_trace.py").read_bytes()) != "6a6f801e83798c21e625325eee2da3f817e326fa":
        raise ValueError("F8R22_INHERITED_R21_SOURCE_DRIFT")
    return f


def episode(split:str,g:int,b:int,a:int,action:int|None=None):
    if split not in COHORTS or type(g) is not int or not 0<=g<COHORTS[split][1]:
        raise ValueError("F8R22_INVALID_SPLIT_OR_GROUP")
    if type(a) is not int or a not in (-1,1) or type(b) is not int or b not in (-1,1):
        raise ValueError("F8R22_INVALID_REGIME")
    if ((split=="fit" and (type(action) is not int or action not in (0,1)))
        or (split!="fit" and action is not None)):
        raise ValueError("F8R22_INVALID_FACTUAL_RETURN")
    seed=COHORTS[split][0]+g
    actions,effects=[],[]
    def record(s,selected,next_s,final=False):
        actions.append(r18.r5._action_key((s,0.,0.,0.),selected))
        effects.append(r18._observed(s,next_s,final=final))
    record(.2,0,r20.r3.synthetic_effect(.2,0,a))
    s=0.
    for t in range(COHORTS[split][2]):
        selected=r20.r3.behavior(s)^int((seed+5*t+int(b==1))%3==0)
        nxt=r20.r3.synthetic_effect(s,selected,b)
        record(s,selected,nxt)
        s=nxt
    if split=="fit":
        record(0.,action,r20.r3.synthetic_effect(0.,action,a),True)
    return {"action":torch.tensor(actions,dtype=torch.float32),
            "effect":torch.tensor(effects,dtype=torch.float32),
            "split":split,"group":g,"z_a":a,"z_b":b,
            "selected_return":action}


def cohort(split):
    locked()
    if split=="fit":
        return [episode(split,g,b,a,q) for g in range(COHORTS[split][1])
                for b in (-1,1) for a in (-1,1) for q in (0,1)]
    return [episode(split,g,b,a) for g in range(COHORTS[split][1])
            for b in (-1,1) for a in (-1,1)]


def factorial_audit(eps,split):
    n=COHORTS[split][1]
    if len(eps)!=(8*n if split=="fit" else 4*n):
        raise ValueError("F8R22_WORLD_PANEL_INCOMPLETE")
    if split=="fit":
        counts={(a,b,q):0 for a in (-1,1) for b in (-1,1) for q in (0,1)}
        for ep in eps:
            counts[(ep["z_a"],ep["z_b"],ep["selected_return"])]+=1
            if ep["effect"].shape!=(10,16):
                raise ValueError("F8R22_FIT_SHAPE")
        if set(counts.values())!={n}:
            raise ValueError("F8R22_FACTUAL_SUPPORT_GAP")
        return {"eight_factorial_cells":8,"per_cell_factual_FIT":n}
    for i in range(0,len(eps),2):
        neg,pos=eps[i:i+2]
        if (neg["group"]!=pos["group"] or neg["z_a"]!=-1 or pos["z_a"]!=1
            or neg["z_b"]!=pos["z_b"]
            or not torch.equal(neg["action"],pos["action"])
            or not torch.equal(neg["effect"][1:],pos["effect"][1:])
            or torch.equal(neg["effect"][0],pos["effect"][0])
            or neg["effect"].shape!=(1+COHORTS[split][2],16)):
            raise ValueError("F8R22_IDENTICAL_B_CAUSAL_PAIR_FAILED")
    return {"matched_B_pairs":len(eps)//2,"group_clusters":n}


def factual(ep):
    return {"action":ep["action"],"effect":ep["effect"]}


def model_for(arm,seed):
    if arm not in ARMS or type(seed) is not int or seed not in SEEDS:
        raise ValueError("F8R22_UNREGISTERED_ARM_SEED")
    if arm.startswith("native"):
        return r19._model("native_return_focus",seed).cpu()
    if arm.startswith("gru"):
        return new_baseline("gru256",seed).cpu()
    return r19._model("ssm_return_focus" if arm.startswith("ssm") else "mlp_return_focus",seed).cpu()


def predict(arm,model,data):
    return conventional_prequential(model,data) if arm=="gru_focus20" else (
        r19._pred("native_return_focus" if arm.startswith("native") else (
                   "ssm_return_focus" if arm.startswith("ssm") else "mlp_return_focus"),model,data))


def objective(arm,model,data,valid):
    if arm=="native_uniform":
        return r19._objective("native_uniform",model,data,valid)
    pred=predict(arm,model,data)
    if (pred.shape!=data["effect"].shape or pred.shape[1]!=10
        or valid.shape!=(BATCH,10) or not bool(valid[:,-1].all())):
        raise ValueError("F8R22_BAD_FACTUAL_RETURN_TENSOR")
    focused=(pred[:,-1,:4]-data["effect"][:,-1,:4]).square().mean()
    if arm=="native_returnonly":
        return focused
    mapped=("native_return_focus" if arm.startswith("native")
            else "ssm_return_focus" if arm.startswith("ssm")
            else "mlp_return_focus")
    if arm=="gru_focus20":
        base=r18.r5.f4._physical_loss(model,data,valid)[0]
    else:
        base=r18._objective("bypass_surrogate" if arm.startswith("native") else (
            "ssm256" if arm.startswith("ssm") else "mlp7"),model,data,valid)
    # Use exactly one common 20× factual return focus on ALL candidate/
    # conventional arms. Their base physical objectives are all existing
    # masked source-neutral factual objectives; not classifier labels.
    return base+20.*focused


def _hypothetical(arm,model,eps,*,erase_A=False,late_B_only=False):
    aa,ee=[],[]
    for ep in eps:
        for act in (0,1):
            current=torch.tensor([r18.r5._action_key((0.,0.,0.,0.),act)],
                                 dtype=torch.float32)
            acts=torch.cat((ep["action"],current))
            obs=torch.cat((ep["effect"],torch.zeros((1,16),dtype=torch.float32)))
            if erase_A or late_B_only:
                obs[0]=0.
            if late_B_only:
                obs[1:-3]=0.
            aa.append(acts);ee.append(obs)
    with torch.no_grad():
        pred=predict(arm,model,{"action":torch.stack(aa),"effect":torch.stack(ee)})
    if pred.shape!=(len(eps)*2,2+COHORTS[eps[0]["split"]][2],16):
        raise ValueError("F8R22_BAD_PREACTION_SHAPE")
    v=pred[:,-1,0].detach().cpu().numpy()
    if not np.isfinite(v).all():
        raise ValueError("F8R22_NONFINITE_PREACTION_PREDICTION")
    return np.asarray([2.4*math.atanh(max(-.98,min(.98,float(z)))) for z in v],
                      dtype=np.float64).reshape(len(eps),2)


def score(arm,model,eps,split):
    factorial_audit(eps,split)
    model.eval()
    t0=time.perf_counter()
    normal=_hypothetical(arm,model,eps)
    erased=_hypothetical(arm,model,eps,erase_A=True)
    short=_hypothetical(arm,model,eps,late_B_only=True)
    actions,losses,erased_losses,short_losses=[],[],[],[]
    for i,ep in enumerate(eps):
        for goal in GOALS:
            acts=[min((0,1),key=lambda act:((pred[i,act]-goal)**2,act))
                  for pred in (normal,erased,short)]
            actions.append(acts[0])
            losses.append(r20.regret(ep["z_a"],goal,acts[0]))
            erased_losses.append(r20.regret(ep["z_a"],goal,acts[1]))
            short_losses.append(r20.regret(ep["z_a"],goal,acts[2]))
    def groupmeans(v):
        return [float(np.mean(v[k*8:k*8+8])) for k in range(COHORTS[split][1])]
    contrasts=sum(int(actions[2*(4*g+2*b)+goal]!=
                      actions[2*(4*g+2*b)+goal+2])
                  for g in range(COHORTS[split][1]) for b in range(2) for goal in range(2))
    return {"mean_regret":float(np.mean(losses)),
            "pergroup_regret":groupmeans(losses),
            "mean_erased_A_regret":float(np.mean(erased_losses)),
            "mean_last3_B_only_regret":float(np.mean(short_losses)),
            "A_sign_choice_contrasts":contrasts,
            "contrast_opportunities":4*COHORTS[split][1],
            "selected_actions":actions,
            "inference_seconds":time.perf_counter()-t0}


def fit_ridge(fit,*,use_early):
    x,y,one=[],[],[]
    for ep in fit:
        delta=r19._physical_delta(ep["effect"][-1])
        if ep["selected_return"]==1:
            one.append(delta)
        else:
            features=([1.,r19._physical_delta(ep["effect"][0])] if use_early
                      else [1.]+[r19._physical_delta(z) for z in ep["effect"][-4:-1]])
            x.append(features);y.append(delta)
    if len(x)!=128 or len(one)!=128:
        raise ValueError("F8R22_INCOMPLETE_FIT_RIDGE_DATA")
    X=np.asarray(x,dtype=np.float64)
    beta=np.linalg.solve(X.T@X+LAMBDA*np.eye(X.shape[1]),X.T@np.asarray(y))
    return {"coef":beta.tolist(),"action1_mean":float(np.mean(one)),"use_early":use_early}


def ridge_score(fitted,eps,split):
    factorial_audit(eps,split)
    losses,choices=[],[]
    beta=np.asarray(fitted["coef"])
    for ep in eps:
        features=([1.,r19._physical_delta(ep["effect"][0])] if fitted["use_early"]
                  else [1.]+[r19._physical_delta(z) for z in ep["effect"][-3:]])
        pred=[float(beta@np.asarray(features)),fitted["action1_mean"]]
        for goal in GOALS:
            act=min((0,1),key=lambda a:((pred[a]-goal)**2,a))
            choices.append(act)
            losses.append(r20.regret(ep["z_a"],goal,act))
    return {"mean_regret":float(np.mean(losses)),"selected_actions":choices}


def train_one(arm,seed,fit,dev,ids,ood,batches):
    model=model_for(arm,seed)
    initial=r18.r5.f5.hash_weights(model)
    opt=torch.optim.AdamW(model.parameters(),lr=.002 if arm.startswith("native") else .001,
                          weight_decay=.01)
    t0=time.perf_counter()
    losses=[];curve=[]
    for update,ix in enumerate(batches,1):
        data,valid=r18.r5.f4._padded([factual(fit[i]) for i in ix])
        model.train();opt.zero_grad(set_to_none=True)
        loss=objective(arm,model,data,valid)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R22_NONFINITE_FIT_LOSS")
        loss.backward()
        grads=[p.grad for p in model.parameters()]
        if (not any(g is not None and bool((g!=0).any()) for g in grads)
            or any(g is not None and not bool(torch.isfinite(g).all()) for g in grads)):
            raise ValueError("F8R22_BAD_GRADIENTS")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        losses.append(float(loss.detach()))
        if update in CHECKPOINTS:
            d=score(arm,model,dev,"dev")
            curve.append({"updates":update,"DEV_regret":d["mean_regret"],
                          "loss_last32":float(np.mean(losses[-32:])),
                          "DEV_contrasts":d["A_sign_choice_contrasts"]})
    wall=time.perf_counter()-t0
    trained=r18.r5.f5.hash_weights(model)
    if trained==initial:
        raise ValueError("F8R22_UNTRAINED_CHECKPOINT")
    id_score=score(arm,model,ids,"id")
    ood_score=score(arm,model,ood,"ood")
    cold=model_for(arm,seed)
    cold.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    cold.eval()
    other_id=score(arm,cold,ids,"id")
    other_ood=score(arm,cold,ood,"ood")
    if (r18.r5.f5.hash_weights(cold)!=trained
        or id_score["selected_actions"]!=other_id["selected_actions"]
        or ood_score["selected_actions"]!=other_ood["selected_actions"]
        or id_score["pergroup_regret"]!=other_id["pergroup_regret"]
        or ood_score["pergroup_regret"]!=other_ood["pergroup_regret"]):
        raise ValueError("F8R22_COLD_RESTORE_ACTION_PARITY_DRIFT")
    return {"arm":arm,"seed":seed,"parameters":sum(p.numel() for p in model.parameters()),
            "updates":UPDATES,"original_weight_hash":initial,"trained_weight_hash":trained,
            "train_seconds":wall,"first32_loss":float(np.mean(losses[:32])),
            "last32_loss":float(np.mean(losses[-32:])),
            "dev_convergence_curve":curve,"id":id_score,"ood":ood_score,
            "cold_restored_exact_action_parity":True}


def _decision(rows,controls):
    conventional=[r for r in rows if r["arm"] in
                  ("gru_focus20","ssm_focus20","mlp_focus20")]
    best=min(conventional,key=lambda row:(
        row["dev_convergence_curve"][-1]["DEV_regret"],row["arm"],row["seed"]))
    rng=np.random.default_rng(BOOT_SEED)
    draws=rng.integers(0,12,size=(BOOT,12))
    out=[]
    for row in rows:
        if row["arm"]!="native_focus20":continue
        halves={}
        for split in ("id","ood"):
            fixed=controls[split]["pergroup"]["always1"]
            conv=best[split]["pergroup_regret"]
            ref=conv if float(np.mean(conv))<float(np.mean(fixed)) else fixed
            val=row[split]
            gain=np.asarray(ref)-np.asarray(val["pergroup_regret"])
            lower=float(np.percentile(gain[draws].mean(axis=1),2.5))
            passed=bool(val["mean_regret"]<=.0135
                and val["mean_regret"]<=best[split]["mean_regret"]-.001
                and val["A_sign_choice_contrasts"]>=24
                and val["mean_erased_A_regret"]-val["mean_regret"]>=.005
                and lower>0)
            halves[split]={"candidate_regret":val["mean_regret"],
                "dev_selected_conventional_regret":best[split]["mean_regret"],
                "bootstrap_2p5_paired_group_gain":lower,
                "probe_erasure_regret_increase":val["mean_erased_A_regret"]-val["mean_regret"],
                "contrast":val["A_sign_choice_contrasts"],"pass":passed}
        out.append({"seed":row["seed"],"two_horizon_pass":
                    halves["id"]["pass"] and halves["ood"]["pass"],
                    "details":halves})
    return {"strongest_DEV_conventional":{"arm":best["arm"],"seed":best["seed"],
            "DEV_regret":best["dev_convergence_curve"][-1]["DEV_regret"]},
            "seed_decisions":out,
            "native_frozen_DEV_PASS":sum(v["two_horizon_pass"] for v in out)>=2}


def run():
    locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit,dev,ids,ood=(cohort(k) for k in COHORTS)
    identities={k:factorial_audit(v,k) for k,v in (
        ("fit",fit),("dev",dev),("id",ids),("ood",ood))}
    # The original R20 analytical controls validate matched worlds, but
    # their function hardcodes R20 split counters. Compute the identical
    # analytic fixed decision policy on newly registered groups instead.
    controls={}
    for name,eps in (("id",ids),("ood",ood)):
        pergroup={key:[] for key in ("always0","always1","goal_positive","goal_negative")}
        for g in range(COHORTS[name][1]):
            worlds=[ep for ep in eps if ep["group"]==g]
            for key in pergroup:
                vals=[]
                for ep in worlds:
                    for goal in GOALS:
                        act={"always0":0,"always1":1,
                             "goal_positive":int(goal>0),
                             "goal_negative":int(goal<0)}[key]
                        vals.append(r20.regret(ep["z_a"],goal,act))
                pergroup[key].append(float(np.mean(vals)))
        controls[name]={"pergroup":pergroup,
                        "means":{k:float(np.mean(v)) for k,v in pergroup.items()}}
        if not math.isclose(controls[name]["means"]["always1"],.015,abs_tol=1e-12):
            raise ValueError("F8R22_INVALID_HISTORYFREE_BOUND")
    probe_reg=fit_ridge(fit,use_early=True)
    recent_reg=fit_ridge(fit,use_early=False)
    learned={k:{"early":ridge_score(probe_reg,v,k),
                "late_B_only":ridge_score(recent_reg,v,k)}
             for k,v in (("id",ids),("ood",ood))}
    if any(learned[k]["early"]["mean_regret"]>=.001 for k in ("id","ood")):
        raise ValueError("F8R22_FIT_ONLY_CAUSAL_MEMORY_NOT_IDENTIFIABLE")
    g=torch.Generator(device="cpu");g.manual_seed(985299)
    batches=torch.randint(0,len(fit),(UPDATES,BATCH),generator=g).tolist()
    start=time.perf_counter()
    rows=[train_one(arm,seed,fit,dev,ids,ood,batches)
          for arm in ARMS for seed in SEEDS]
    decision=_decision(rows,controls)
    return {"schema":SCHEMA,"freeze_blob":FREEZE_BLOB,
            "classification":"SOURCE_NEUTRAL_AUTHOR_GENERATED_DEVELOPMENT_ONLY",
            "model_count":len(rows),"optimizer_updates":len(rows)*UPDATES,
            "synthetic_selected_FIT_effects":sum(len(x["effect"]) for x in fit),
            "synthetic_heldout_observed_effects":
                sum(len(x["effect"]) for x in ids+ood),
            "original_external_source_effects":0,"spent_F7_replay":False,
            "identity_audits":identities,"analytical_controls":controls,
            "FIT_only_learned_memory_controls":learned,
            "models":rows,"decision":decision,
            "conventional_convergence_independently_qualified":False,
            "scientific_pass":False,"deployment":"NONE",
            "train_score_wall_seconds":time.perf_counter()-start}


def retain_ci_receipt(result):
    if os.environ.get("GITHUB_ACTIONS")!="true":return
    dest=ROOT/"local_runs/p85mr_f8r22_matched_objective_dev.json"
    dest.parent.mkdir(exist_ok=True)
    if dest.exists():
        raise FileExistsError("F8R22_EVIDENCE_REPLAY_OVERWRITE_REFUSED")
    dest.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")


def refuse_original_firstuse(*args,**kwargs):
    raise PermissionError("F8R22_ORIGINAL_SOURCE_FIRSTUSE_NOT_AUTHORIZED")
