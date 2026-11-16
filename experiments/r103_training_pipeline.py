"""R103 frozen 11-arm candidate training/scoring, **not** original first-use.

Only isolated UNIT_ONLY fixtures may be used for engineering. The real
prospective source calls R103._world which ALWAYS refuses first-use; no owner
authorization, scientific source, runner, or scoring is activated here.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import r85_delayed_composition as r85
from experiments import r103_relational_binding_diagnostic as r103

ROOT=Path(__file__).resolve().parents[1]
TRAIN=("FIT","VALID")
SCORED=("DEV_FAMILIAR","DEV_UNKNOWN","OOD_SHIFT","DELAYED_COLD","REVERSAL")
ARMS=r103.ARMS
SEEDS=r103.SEEDS
STEPS=3072
BATCH=16
SUPPORT=r103.SUPPORT


def protocol():
    f=r103.frozen()
    if (f["source"]["counts"]!=r103.PANEL_COUNTS
        or f["source"]["support_interventions_total"]!=SUPPORT
        or f["training"]["steps_per_arm"]!=STEPS
        or f["training"]["batch_size"]!=BATCH
        or f["training"]["training_all_11_fully_trained_required_before_DEV"] is not True
        or f["training"]["paired_bootstrap_draws"]!=2000
        or tuple(f["arms"])!=ARMS
        or f["evaluation"]["independent_scientific_pass"] is not False):
        raise PermissionError("R103_PROSPECTIVE_PIPELINE_FREEZE_MISMATCH")
    return f


def _observed_shuffled_outcomes(slots,actions,rewards,panel,seed,index):
    """Fixed public action/time strata, real nontrivial derangement.

    Never inspect rewards, hidden family, private factors or future query to
    choose the permutation. No original scored source called.
    """
    f=protocol()
    if (panel not in r103.PANEL_COUNTS or seed not in SEEDS
        or not isinstance(index,int) or not 0<=index<r103.PANEL_COUNTS[panel]
        or slots.shape!=(SUPPORT,5,13)
        or actions.shape!=(SUPPORT,) or rewards.shape!=(SUPPORT,)):
        raise PermissionError("R103_NONREGISTERED_SHUFFLE_SOURCE")
    key=[(int(actions[i]),float(slots[i,-1,12])) for i in range(SUPPORT)]
    order=np.arange(SUPPORT)
    rng=np.random.default_rng(np.random.SeedSequence([
        f["source"]["salts"]["SHUFFLE"],
        f["source"]["salts"][panel],seed,index]))
    for tag in sorted(set(key)):
        members=np.array([i for i in range(SUPPORT) if key[i]==tag])
        if len(members)<2:
            raise RuntimeError("R103_NO_SUFFICIENT_PUBLIC_STRATUM_SHUFFLE")
        order[members]=np.roll(members,int(rng.integers(1,len(members))))
    if (not np.array_equal(np.sort(order),np.arange(SUPPORT))
        or any(order[i]==i or key[i]!=key[order[i]] for i in range(SUPPORT))):
        raise RuntimeError("R103_SHUFFLE_IS_FIXED_OR_ACTION_LEAK")
    return rewards[order].copy(),order


def episode(panel,seed,index,sha=None,*,paired=False):
    """16 independent REAL factual worlds, same latent hidden rule family.

    Blocked for ALL registered source panels until separate verified first-use.
    """
    f=protocol()
    if (panel not in r103.PANEL_COUNTS or seed not in SEEDS
        or not isinstance(index,int) or not 0<=index<r103.PANEL_COUNTS[panel]):
        raise PermissionError("R103_UNREGISTERED_PROSPECTIVE_EPISODE")
    if bool(paired)!=(panel in SCORED):
        raise PermissionError("R103_FORBIDDEN_TRAIN_COUNTERFACTUAL_QUERY")
    slots=[]; rewards=[]; actions=[]
    for event in range(SUPPORT):
        action=r103.support_action(event)
        family=r103.family_for_panel(panel,index,event)
        world=r103._world(panel,seed,index,sha,event=event,family=family)
        cues,ret=r103.public_view(world,action)
        slots.append(cues);rewards.append(ret);actions.append(action)
    s=np.stack(slots).astype(np.float32)
    ret=np.asarray(rewards,np.float32)
    acts=np.asarray(actions,np.int8)
    shuffled,order=_observed_shuffled_outcomes(
        s,acts,ret,panel,seed,index)
    qworld=r103._world(panel,seed,index,sha,event=SUPPORT,
                      family=r103.family_for_panel(panel,index,SUPPORT))
    out={"support_slots":s,"support_rewards":ret,
         "support_actions":acts,"support_rewards_shuffled":shuffled,
         "shuffle_permutation_evaluator_only":order}
    if paired:
        q0,r0=r103.public_view(qworld,r85.CALIBRATE)
        q2,r2=r103.public_view(qworld,r85.EXPLOIT_POS)
        if (not np.array_equal(q0[:4],q2[:4])
            or not np.array_equal(q0[-1,4:],q2[-1,4:])):
            raise RuntimeError("R103_QUERY_UNCHOSEN_PREACTION_LEAK")
        out.update({"query0":q0,"query2":q2,
                    "reward0":float(r0),"reward2":float(r2),
                    "private_law":int(qworld["law_a"]),
                    "private_family":r103.family_for_panel(panel,index,SUPPORT),
                    "cold":bool(qworld["pivot"]==r85.COLD)})
    else:
        action=r103.query_action(seed,index)
        query,return_actual=r103.public_view(qworld,action)
        out.update({"query":query,"target":float(return_actual),
                    "query_factual_action":action})
        if set(out)&{"query0","query2","reward0","reward2","private_law","private_family"}:
            raise RuntimeError("R103_TRAINING_PRIVATE_OR_UNCHOSEN_LEAK")
    return out


def source(panel,seed,sha=None):
    """Original scored generation is not admitted on this branch."""
    f=protocol()
    if panel not in r103.PANEL_COUNTS or seed not in SEEDS:
        raise PermissionError("R103_PROTECTED_SOURCE_SCOPE_NOT_REGISTERED")
    records=[episode(panel,seed,j,sha,paired=(panel in SCORED))
             for j in range(f["source"]["counts"][panel])]
    result={k:np.stack([row[k] for row in records])
            for k in ("support_slots","support_rewards",
                      "support_rewards_shuffled","support_actions")}
    if panel in SCORED:
        for k in ("query0","query2","reward0","reward2",
                  "private_law","private_family","cold"):
            result[k]=np.asarray([row[k] for row in records])
    else:
        for k in ("query","target","query_factual_action"):
            result[k]=np.asarray([row[k] for row in records])
        if set(result)&{"query0","query2","reward0","reward2","private_law"}:
            raise RuntimeError("R103_FIT_VALID_EVALUATOR_LEAK")
    return result


def _inputs(data,index,arm,action=None,permuted_roles=False):
    if arm not in ARMS:
        raise PermissionError("R103_TRAINING_ARM_NOT_REGISTERED")
    s=data["support_slots"][index]
    y=(data["support_rewards_shuffled"][index] if arm==r103.SHUFFLED
       else data["support_rewards"][index])
    if action is None:
        q=data["query"][index]
    elif action==r85.CALIBRATE:
        q=data["query0"][index]
    elif action==r85.EXPLOIT_POS:
        q=data["query2"][index]
    else: raise PermissionError("R103_UNREGISTERED_EVALUATOR_ACTION")
    if permuted_roles:
        s,q=permute_public_roles(s,q)
    return (torch.from_numpy(np.ascontiguousarray(s,dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(y,dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(q,dtype=np.float32)))


def permute_public_roles(support_slots,query):
    """True physical-role relabel (cues + operand tags + role identities).

    A mere row reversal without permuted role IDs is not causal invariance.
    """
    if (support_slots.ndim!=4 or support_slots.shape[1:]!=(SUPPORT,5,13)
        or query.ndim!=3 or query.shape[1:]!=(5,13)
        or len(support_slots)!=len(query)):
        raise ValueError("R103_INVALID_PUBLIC_ROLE_PERMUTATION_INPUTS")
    role_map=np.array([2,0,3,1],np.int64)
    order=np.argsort(role_map)
    new_s=support_slots.copy()
    new_q=query.copy()
    new_s[:,:,:4,:]=support_slots[:,:,order,:]
    new_q[:,:4,:]=query[:,order,:]
    names=np.array([.25,.5,.75,1.],np.float32)
    new_s[:,:,:4,9]=names[None,None,:]
    new_q[:,:4,9]=names[None,:]
    if (not np.array_equal(new_s[:,:,-1,:],support_slots[:,:,-1,:])
        or not np.array_equal(new_q[:,-1,:],query[:,-1,:])
        or not np.array_equal(np.sort(new_s[:,:,:4,8],axis=2),
                              np.sort(support_slots[:,:,:4,8],axis=2))):
        raise RuntimeError("R103_ROLE_RELABEL_CHANGED_FACTUAL_CAUSAL_QUERY")
    return new_s,new_q


def predict(model,arm,data,action=None,permuted_roles=False):
    model.eval()
    values=[]
    with torch.no_grad():
        for i in range(0,len(data["support_slots"]),32):
            subset=np.arange(i,min(len(data["support_slots"]),i+32))
            x,y,q=_inputs(data,subset,arm,action,permuted_roles)
            result=model(x,y,q)
            if result.shape!=(len(subset),):
                raise RuntimeError("R103_TRAINED_VALUE_SHAPE_DRIFT")
            values.append(result.cpu().numpy())
    model.train()
    out=np.concatenate(values).astype(np.float64)
    if not np.isfinite(out).all():
        raise RuntimeError("R103_NONFINITE_TRAINED_VALUE")
    return out


def fit_arm(model,arm,fit,valid,indices):
    """Actual factual FIT only; all 3072 updates, no DEV scoring."""
    if (arm not in ARMS or indices.shape!=(STEPS,BATCH)
        or set(fit)&{"reward0","reward2","query0","query2","private_law","private_family"}
        or set(valid)&{"reward0","reward2","query0","query2","private_law","private_family"}):
        raise PermissionError("R103_TRAINING_INVALID_OR_PRIVILEGED_INPUT")
    if len(fit["target"])!=protocol()["source"]["counts"]["FIT"]:
        raise RuntimeError("R103_FULL_FIT_COHORT_SIZE_INVALID")
    optimiser=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    first_valid=float(np.mean((predict(model,arm,valid)-valid["target"])**2))
    started=time.perf_counter()
    labels=torch.from_numpy(fit["target"].astype(np.float32))
    for ix in indices:
        s,r,q=_inputs(fit,ix,arm)
        v=model(s,r,q)
        loss=F.mse_loss(v,labels[ix])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("R103_FACTUAL_FIT_LOSS_NONFINITE")
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimiser.step()
    seconds=time.perf_counter()-started
    train_mse=float(np.mean((predict(model,arm,fit)-fit["target"])**2))
    valid_mse=float(np.mean((predict(model,arm,valid)-valid["target"])**2))
    if not all(math.isfinite(n) for n in (seconds,first_valid,train_mse,valid_mse)):
        raise RuntimeError("R103_TRAINING_TIME_OR_MSE_NONFINITE")
    total=sum(p.numel() for p in model.parameters() if p.requires_grad)
    active=sum(p.numel() for p in model.parameters()
               if p.requires_grad and p.grad is not None)
    if arm in (r103.GRU,r103.ROLE_SET,r103.MLP,r103.STATIC) and active!=total:
        raise RuntimeError("R103_CONVENTIONAL_HAS_UNUSED_TRAINABLE_WEIGHTS")
    return {"optimizer_updates":STEPS,"factual_query_examples":STEPS*BATCH,
            "factual_support_examples":STEPS*BATCH*SUPPORT,
            "trainable_parameters":total,"active_gradient_parameters":active,
            "dormant_gradient_parameters":total-active,
            "FIT_MSE":train_mse,"VALID_MSE":valid_mse,
            "VALID_before_MSE":first_valid,"validation_report_only":True,
            "training_CPU_seconds":seconds,
            "actual_FLOPs_not_measured":True}


def score(models,panel,data):
    """Evaluator ONLY, never inspect paired outcome during training."""
    if (panel not in SCORED or set(data)&{"query","target"}
        or set(models)!=set(ARMS)):
        raise PermissionError("R103_EVALUATOR_SCORED_SOURCE_INVALID")
    n=len(data["reward0"])
    raw,chosen,metrics={},{},{}
    oracle=np.maximum(data["reward0"],data["reward2"])
    true_action=np.where(data["private_law"]>0,r85.CALIBRATE,r85.EXPLOIT_POS)
    for arm in ARMS:
        value0=predict(models[arm],arm,data,action=r85.CALIBRATE)
        value2=predict(models[arm],arm,data,action=r85.EXPLOIT_POS)
        action=np.where(value0>value2,r85.CALIBRATE,r85.EXPLOIT_POS)
        realized=np.where(action==r85.CALIBRATE,data["reward0"],data["reward2"])
        raw[arm],chosen[arm]=realized,action
        metrics[arm]={"realized_reward_mean":float(np.mean(realized)),
                      "regret_to_oracle":float(np.mean(oracle-realized)),
                      "true_law_query_action_agreement":
                          float(np.mean(action==true_action))}
        if np.any(data["cold"]):
            metrics[arm]["cold_reward_mean"]=float(np.mean(realized[data["cold"]]))
    renamed=np.where(
        predict(models[r103.NATIVE],r103.NATIVE,data,action=r85.CALIBRATE,permuted_roles=True)>
        predict(models[r103.NATIVE],r103.NATIVE,data,action=r85.EXPLOIT_POS,permuted_roles=True),
        r85.CALIBRATE,r85.EXPLOIT_POS)
    return {"panel":panel,"metrics":metrics,"raw_reward":raw,"choices":chosen,
            "role_relabel_action_agreement":float(
                np.mean(chosen[r103.NATIVE]==renamed)),
            "oracle_mean":float(np.mean(oracle)),"physical_episodes":n}


def bootstrap_lower(a,b,seed):
    if (a.ndim!=1 or b.shape!=a.shape or seed not in SEEDS):
        raise ValueError("R103_BOOTSTRAP_NONMATCHED_COHORT")
    rng=np.random.default_rng(np.random.SeedSequence([
        protocol()["source"]["salts"]["BOOTSTRAP"],seed]))
    samples=rng.integers(0,len(a),size=(2000,len(a)))
    return float(np.quantile((a-b)[samples].mean(axis=1),.025))


def measures(seed,scores):
    """All twelve R103 precommitted gates, with original first-use denied."""
    if seed not in SEEDS or not set(SCORED)<=set(scores):
        raise PermissionError("R103_SCIENCE_SOURCE_OR_SEED_MISSING")
    dev=scores["DEV_UNKNOWN"];cold=scores["DELAYED_COLD"]
    ood=scores["OOD_SHIFT"];rev=scores["REVERSAL"]
    def mean(panel,arm):
        return panel["metrics"][arm]["realized_reward_mean"]
    conventional=(r103.GRU,r103.ROLE_SET,r103.MLP,r103.STATIC)
    strongest=max(conventional,key=lambda x:mean(dev,x))
    strongest_ood=max(conventional,key=lambda x:mean(ood,x))
    check_values={
        "native_unknown_query_action_accuracy_min":
            dev["metrics"][r103.NATIVE]["true_law_query_action_agreement"],
        "native_minus_strongest_conventional_reward_min":
            mean(dev,r103.NATIVE)-mean(dev,strongest),
        "paired_bootstrap_lower95_native_minus_strongest_min":
            bootstrap_lower(dev["raw_reward"][r103.NATIVE],
                            dev["raw_reward"][strongest],seed),
        "native_minus_r102_scalar_gate_reward_min":
            mean(dev,r103.NATIVE)-mean(dev,r103.SCALAR),
        "native_minus_ungated_same_capacity_reward_min":
            mean(dev,r103.NATIVE)-mean(dev,r103.UNGATED),
        "native_minus_shuffled_feedback_reward_min":
            mean(dev,r103.NATIVE)-mean(dev,r103.SHUFFLED),
        "native_minus_zero_outcomes_reward_min":
            mean(dev,r103.NATIVE)-mean(dev,r103.ZERO),
        "native_minus_cold_reset_reward_min":
            mean(cold,r103.NATIVE)-mean(cold,r103.RESET),
        "native_minus_external_matched_reward_min":
            mean(dev,r103.NATIVE)-mean(dev,r103.EXTERNAL),
        "native_minus_strongest_OOD_reward_min":
            mean(ood,r103.NATIVE)-mean(ood,strongest_ood),
        "reversal_native_minus_zero_outcome_reward_min":
            mean(rev,r103.NATIVE)-mean(rev,r103.ZERO),
        "role_relabel_action_agreement_min":
            dev["role_relabel_action_agreement"],
    }
    bounds=protocol()["evaluation"]["checks"]
    if set(bounds)!=set(check_values):
        raise RuntimeError("R103_POSTHOC_THRESHOLD_OR_METRIC_MAPPING_DRIFT")
    checks={k:bool(np.isfinite(check_values[k]) and check_values[k]>=bounds[k])
            for k in bounds}
    return {"measured":check_values,"checks":checks,"all_pass":all(checks.values()),
            "strongest_nonengineered":strongest,"strongest_ood":strongest_ood,
            "frozen_source":True}


def original_ci(*_args,**_kwargs):
    """R103 original FIT/VALID/DEV not authorized; no action or side effects."""
    raise PermissionError("R103_PROTECTED_ORIGINAL_FIRST_USE_NOT_AUTHORIZED")


if __name__=="__main__":
    raise PermissionError("R103_PROTECTED_ORIGINAL_FIRST_USE_NOT_AUTHORIZED")
