"""F8-R73: prospective rare factual calibration action learning diagnostic.

A fresh-synthetic, tokenless action-consequence diagnostic using selected real
intervention histories only. Factual public-time teacher imitation is NOT
learning a new causal rule. The weights of all models are fit on new FIT
before new HOLDOUT worlds are even constructed. No protected external use.
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

from experiments import p85mr_f8r70_stateful_intervention as r70
from experiments import p85mr_f8r71_trained_causal_action as r71
from experiments import p85mr_f8r72_gradient_diagnostic as r72

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r73_rare_intervention_freeze.json"
FREEZE_BLOB="b77dab412f8183d4680f10d1561eb75060d446a5"
SCHEMA="ttt-hybrid.f8r73-rare-intervention-learning.v1"
PARENT="9981ed29e7b31cfb64ae9cc0a3b08de2d99ddb4f"
SALTS={"FIT":24510000,"HOLDOUT":24530000,"NEGATIVE":24540000}
COUNTS={"FIT":128,"HOLDOUT":64}
ARMS=("STRUCTURED_CAUSAL_MEMORY","GRU16","MLP16")
METHODS=("UNWEIGHTED_FACTUAL_BC","FROZEN_CLASS_WEIGHT_FACTUAL_BC",
         "SHIFTED_LABEL_NEGATIVE_BC","SELECTED_RETURN_REINFORCE")
NAMES=("UNWEIGHTED","WEIGHTED","SHUFFLED","PG")
SEEDS=(986101,986103)
UPDATES=64
BATCH=8
STEPS=40
ACTIONS=(0,2,3)
BOOT_SEED=24549999
CLASS_WEIGHTS=(37./3.,1.,1.)
FIXED=("R70_CALIBRATE_CAUSAL","R70_NO_CALIBRATE_REACTIVE")


def git_blob(data):
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()


def frozen():
    raw=FREEZE.read_bytes()
    if git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("F8R73_ORIGINAL_FREEZE_HASH_DRIFT")
    f=json.loads(raw)
    p=f["parent"]
    s=f["source"]
    t=f["training"]
    sc=f["scoring"]
    if (f["schema"]!=SCHEMA
        or f["stage"]!="PROSPECTIVE_FROZEN_BEFORE_R73_SOURCE_TRAINING_AND_HOLDOUT"
        or p["R72_SHA"]!=PARENT or p["R72_freeze_blob"]!=r72.FREEZE_BLOB
        or p["R72_result"]!="ENGINEERING_PASS_DEV_DIAGNOSTIC_NO_GO"
        or p["all_R64_to_R72_scored_cohorts_spent"] is not True
        or s["salts"]!=SALTS or s["counts"]!=COUNTS
        or s["steps"]!=STEPS or s["actions"]!=4
        or s["selected_observation_dim"]!=4
        or s["protected_external_OOD"]
        or t["arms"]!=list(ARMS)
        or t["objectives"]!=list(METHODS)
        or t["seeds"]!=list(SEEDS)
        or t["optimizer_steps"]!=UPDATES or t["FIT_worlds_per_step"]!=BATCH
        or t["selected_observed_events_per_model"]!=UPDATES*BATCH*STEPS
        or t["learning_rate"]!=.015 or t["weight_decay"]!=.001
        or list(t["class_weights_for_3_action_logits"].values())!=list(CLASS_WEIGHTS)
        or sc["seed_bootstrap"]!=BOOT_SEED
        or sc["paired_bootstrap_resamples"]!=2000
        or sc["no_post_result_tuning"] is not True
        or any(f["authority"].values())):
        raise ValueError("F8R73_ORIGINAL_FROZEN_CONTRACT_DRIFT")
    return f


def worlds(panel):
    if panel not in COUNTS:
        raise PermissionError("F8R73_SPENT_OR_PROTECTED_SOURCE_DENIED")
    rng=np.random.default_rng(SALTS[panel])
    n=COUNTS[panel]
    labels=[(sw,sg) for sw in (False,True) for sg in (-1,1)
            for _ in range(n//4)]
    order=rng.permutation(n)
    source=[]
    for index,chosen in enumerate(order):
        switched,initial=labels[int(chosen)]
        b=int(rng.integers(9,19))
        a=int(rng.integers(22,36))
        hidden=np.full(STEPS,initial,dtype=np.int64)
        if switched:
            hidden[b:a]=-initial
        runtime="FIT" if panel=="FIT" else "DEV"
        source.append({
          "panel":runtime,"source_panel":runtime,
          "R73_source_panel":panel,"index":int(index),
          "switched_evaluator_only":bool(switched),
          "initial_sign_evaluator_only":int(initial),
          "B_start_evaluator_only":b,
          "A_return_start_evaluator_only":a,
          "latent_law_evaluator_only":hidden,
          "potential_innovations_evaluator_only":rng.normal(0,1,(STEPS,4,4)),
          "unobserved_nuisance_proxy":rng.normal(0,1,(STEPS,4))
        })
    return source


def negative_proxy(holdout):
    if (len(holdout)!=COUNTS["HOLDOUT"]
        or any(w.get("R73_source_panel")!="HOLDOUT" for w in holdout)):
        raise PermissionError("F8R73_NEGATIVE_REQUIRES_FRESH_HOLDOUT")
    rng=np.random.default_rng(SALTS["NEGATIVE"])
    return [{**w,"unobserved_nuisance_proxy":rng.normal(0,5,(STEPS,4))}
            for w in holdout]


class Model(nn.Module):
    """R71 architectural interfaces, freshly initialized at R73 seeds."""
    def __init__(self,arm,seed):
        super().__init__()
        if arm not in ARMS or seed not in SEEDS:
            raise ValueError("F8R73_INVALID_TRAINED_MODEL")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+997*ARMS.index(arm))
            if arm==ARMS[0]:
                self.head=nn.Linear(8,3).double()
            elif arm==ARMS[1]:
                self.cell=nn.GRUCell(9,16).double()
                self.head=nn.Linear(16,3).double()
            else:
                self.head=nn.Sequential(nn.Linear(9,16),nn.Tanh(),
                                        nn.Linear(16,3)).double()
        self.arm=arm

    def initial(self):
        return torch.zeros(16,dtype=torch.float64)

    def logits(self,history,t,carry):
        if self.arm==ARMS[0]:
            return self.head(r71._memory_features(carry,t)),carry
        last_action=history[-1][0] if history else None
        last_obs=history[-1][1] if history else np.zeros(4)
        public=r71._public_input(last_action,last_obs,t)
        if self.arm==ARMS[1]:
            updated=self.cell(public.unsqueeze(0),carry.unsqueeze(0)).squeeze(0)
            return self.head(updated),updated
        return self.head(public),carry


def teacher(world):
    if world.get("R73_source_panel") not in COUNTS:
        raise PermissionError("F8R73_FACTUAL_TEACHER_WRONG_SOURCE")
    z=r71.fixed_rollout("R70_CALIBRATE_CAUSAL",world)
    if len(z["actions"])!=STEPS or len(z["observations"])!=STEPS:
        raise ValueError("F8R73_INVALID_FACTUAL_TEACHER_HISTORY")
    if tuple(i for i,a in enumerate(z["actions"]) if a==0)!=(0,16,32):
        raise ValueError("F8R73_FIXED_PUBLIC_TEACHER_DRIFT")
    return z["actions"],z["observations"]


def factual_loss(model,world,method):
    if method not in METHODS[:3]:
        raise ValueError("F8R73_ILLEGAL_BC_METHOD")
    actions,observations=teacher(world)
    hist=[]
    state=model.initial()
    loss=[]
    ws=[]
    for t,(action,selected) in enumerate(zip(actions,observations)):
        logits,state=model.logits(hist,t,state)
        target=actions[(t+5)%STEPS] if method==METHODS[2] else action
        idx=ACTIONS.index(target)
        weight=1. if method==METHODS[0] else CLASS_WEIGHTS[idx]
        loss.append(-F.log_softmax(logits,dim=0)[idx]*weight)
        ws.append(weight)
        hist.append((action,selected))
        if model.arm==ARMS[0]:
            state=r71._update_factual_memory(state,action,selected)
    return torch.stack(loss).sum()/sum(ws)


def _fit_batches(seed,n):
    return np.random.default_rng(24519999+seed).integers(0,n,(UPDATES,BATCH))


def train(arm,method,seed,fit):
    if (arm not in ARMS or method not in METHODS or seed not in SEEDS
        or len(fit)!=COUNTS["FIT"]
        or any(w.get("R73_source_panel")!="FIT" for w in fit)):
        raise PermissionError("F8R73_UNREGISTERED_OR_NON_FIT_TRAINING")
    start=time.perf_counter()
    model=Model(arm,seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.015,weight_decay=.001)
    act_gen=torch.Generator(device="cpu").manual_seed(
        24515555+seed+ARMS.index(arm)*997)
    first=None
    last=None
    for idx in _fit_batches(seed,len(fit)):
        if method==METHODS[3]:
            actual=[r71.rollout(model,fit[int(i)],greedy=False,action_gen=act_gen)
                    for i in idx]
            loss=r72.policy_gradient_loss(actual)
        else:
            loss=torch.stack([factual_loss(model,fit[int(i)],method)
                              for i in idx]).mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R73_NONFINITE_FIT_LOSS")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if first is None: first=float(loss.detach())
        last=float(loss.detach())
    weights=hashlib.sha256(b"".join(
        p.detach().numpy().astype("<f8").tobytes() for p in model.parameters()
    )).hexdigest()
    return model,{
      "arm":arm,"method":method,"seed":seed,
      "steps":UPDATES,"FIT_worlds_per_step":BATCH,
      "FIT_selected_event_exposures":UPDATES*BATCH*STEPS,
      "trainable_params":int(sum(p.numel() for p in model.parameters())),
      "allocated_state_floats":16 if arm!=ARMS[2] else 0,
      "initial_loss":first,"final_loss":last,"weights_sha256":weights,
      "train_wall_seconds":time.perf_counter()-start
    }


def action_agreement(model,source):
    tp=fp=fn=correct=total=0
    for world in source:
        actions,observations=teacher(world)
        hist=[]
        state=model.initial()
        with torch.no_grad():
            for t,(true,selected) in enumerate(zip(actions,observations)):
                logits,state=model.logits(hist,t,state)
                predicted=ACTIONS[int(torch.argmax(logits).item())]
                correct+=int(predicted==true)
                total+=1
                tp+=int(predicted==0 and true==0)
                fp+=int(predicted==0 and true!=0)
                fn+=int(predicted!=0 and true==0)
                hist.append((true,selected))
                if model.arm==ARMS[0]:
                    state=r71._update_factual_memory(state,true,selected)
    positives=total*3//STEPS
    negatives=total-positives
    return {"factual_action_agreement":correct/total,
      "calibration_recall":tp/(tp+fn),
      "calibration_precision":tp/max(tp+fp,1),
      "calibration_false_positive_rate":fp/negatives,
      "calibration_f1":2*tp/max(2*tp+fp+fn,1),
      "calibration_true_positive":tp,
      "calibration_false_positive":fp,
      "observed_factual_actions":total
    }


def row_identity(rows):
    return tuple((x["world"],x["actions"],x["episode_reward"],
                  x["late_return_reward"],x["observation_sha256"],
                  x["physical_sha256"]) for x in rows)


def score(model,arm,source):
    return r71.score_one(arm,model,source)


def bootstrap_lower(a,b,seed):
    if len(a)!=len(b) or len(a)!=COUNTS["HOLDOUT"]:
        raise ValueError("F8R73_UNPAIRED_HOLDOUT")
    diffs=[]
    for x,y in zip(a,b):
        if (x["world"],x["switched"])!=(y["world"],y["switched"]):
            raise ValueError("F8R73_UNPAIRED_WORLD_IDS")
        diffs.append(x["episode_reward"]-y["episode_reward"])
    rng=np.random.default_rng(BOOT_SEED+seed)
    ids=rng.integers(0,len(diffs),(2000,len(diffs)))
    return float(np.quantile(np.asarray(diffs)[ids].mean(axis=1),.025))


def adjudicate(seed,scores,agreements,source_checks,freeze):
    prefix="GRU16"
    weighted=scores[prefix+"_WEIGHTED"]
    unweighted=scores[prefix+"_UNWEIGHTED"]
    pg=scores[prefix+"_PG"]
    shuffled=scores[prefix+"_SHUFFLED"]
    reference=scores["R70_CALIBRATE_CAUSAL"]
    w=agreements[prefix+"_WEIGHTED"]
    lim=freeze["scoring"]["frozen_admission_all"]
    wu=weighted["episode_reward"]-unweighted["episode_reward"]
    wp=weighted["episode_reward"]-pg["episode_reward"]
    ws=weighted["episode_reward"]-shuffled["episode_reward"]
    bootstrap_unweighted=bootstrap_lower(weighted["per_world"],unweighted["per_world"],seed)
    bootstrap_pg=bootstrap_lower(weighted["per_world"],pg["per_world"],seed)
    fraction=weighted["episode_reward"]/max(abs(reference["episode_reward"]),1e-12)
    checks={
       "weighted_calibration_recall":w["calibration_recall"]>=lim["GRU_weighted_calibration_recall_min"],
       "weighted_calibration_precision":w["calibration_precision"]>=lim["GRU_weighted_calibration_precision_min"],
       "weighted_calibration_false_positive":w["calibration_false_positive_rate"]<=lim["GRU_weighted_calibration_false_positive_max"],
       "weighted_fraction_of_fixed_reward":fraction>=lim["GRU_weighted_closed_loop_reward_fraction_of_fixed_min"],
       "weighted_reward_over_unweighted":wu>=lim["GRU_weighted_reward_gain_over_unweighted_min"],
       "weighted_reward_over_policy_gradient":wp>=lim["GRU_weighted_reward_gain_over_reward_gradient_min"],
       "weighted_reward_over_shifted_negative":ws>=lim["GRU_weighted_reward_gain_over_shifted_negative_min"],
       "bootstrap_lower_weighted_vs_unweighted":bootstrap_unweighted>0,
       "bootstrap_lower_weighted_vs_pg":bootstrap_pg>0,
       "source_balance_and_FIT_HOLDOUT_nonoverlap":bool(source_checks["source_balance_FIT_HOLDOUT_disjoint"]),
       "same_FIT_steps_draws_factual_exposures":bool(source_checks["same_FIT_optimizer_steps_world_draws_actual_selected_exposures"]),
       "negative_unobserved_proxy_invariance":bool(source_checks["unobserved_proxy_negative_invariance"]),
       "exact_cold_replay":bool(source_checks["exact_cold_replay"]),
       "original_causal_effect":bool(source_checks["original_causal_transition_effect_preserved"])
    }
    return {"seed":seed,"metrics":{
        "weighted_reward":weighted["episode_reward"],
        "unweighted_reward":unweighted["episode_reward"],
        "policy_gradient_reward":pg["episode_reward"],
        "shifted_negative_reward":shuffled["episode_reward"],
        "fixed_positive_reference_reward":reference["episode_reward"],
        "weighted_reward_fraction_of_fixed":fraction,
        "weighted_minus_unweighted_reward":wu,
        "weighted_minus_policy_gradient_reward":wp,
        "weighted_minus_shifted_negative_reward":ws,
        "bootstrap_lower95_weighted_minus_unweighted":bootstrap_unweighted,
        "bootstrap_lower95_weighted_minus_policy_gradient":bootstrap_pg,
        **w
       },"checks":checks,"all_frozen_checks":all(checks.values())}


def run():
    started=time.perf_counter()
    freeze=frozen()
    fit=worlds("FIT")
    training={}
    models={}
    # Train and freeze all model weights before the new scored source exists.
    for seed in SEEDS:
        label=str(seed)
        training[label]={}
        models[label]={}
        for arm in ARMS:
            for method,name in zip(METHODS,NAMES):
                key=arm+"_"+name
                model,receipt=train(arm,method,seed,fit)
                models[label][key]=model
                training[label][key]=receipt
    holdout=worlds("HOLDOUT")
    digest=lambda w:hashlib.sha256(w["potential_innovations_evaluator_only"].astype("<f8").tobytes()).hexdigest()
    disjoint=set(map(digest,fit)).isdisjoint(map(digest,holdout))
    balanced=all(
      len(z)==COUNTS[panel]
      and sum(int(w["switched_evaluator_only"]) for w in z)==len(z)//2
      and sum(int(w["initial_sign_evaluator_only"]==1) for w in z)==len(z)//2
      and all(sum(int(w["switched_evaluator_only"]==sw
                      and w["initial_sign_evaluator_only"]==sg)
              for w in z)==len(z)//4 for sw in (False,True) for sg in (-1,1))
      for panel,z in (("FIT",fit),("HOLDOUT",holdout)))
    if not disjoint or not balanced:
        raise ValueError("F8R73_FROZEN_SOURCE_BALANCE_OR_CUSTODY_FAILURE")
    negative=negative_proxy(holdout)
    physical=[]
    for world in holdout:
        _,fixed=r70.transition(world,0,0,0)
        _,sham=r70.transition(world,0,1,0)
        m=int(world["latent_law_evaluator_only"][1])
        action=2 if m>0 else 3
        reward_fixed,_=r70.transition(world,1,action,fixed)
        reward_sham,_=r70.transition(world,1,action,sham)
        physical.append(float(reward_fixed[0]-reward_sham[0]))
    physical_ok=bool(np.allclose(physical,.8,atol=1e-12,rtol=0.))
    if not physical_ok:
        raise ValueError("F8R73_PHYSICAL_CAUSAL_TRANSITION_BROKEN")
    fixed_scores={name:score(None,name,holdout) for name in FIXED}
    original_fixed_ok=all(
        row_identity(fixed_scores[n]["per_world"])==
        row_identity(score(None,n,holdout)["per_world"])==
        row_identity(score(None,n,negative)["per_world"])
        for n in FIXED)
    all_scores={}
    all_agree={}
    decisions=[]
    for seed in SEEDS:
        label=str(seed)
        outcomes={**fixed_scores}
        agreements={}
        replay=bool(original_fixed_ok)
        allbudgets=True
        for arm in ARMS:
            for method,name in zip(METHODS,NAMES):
                key=arm+"_"+name
                model=models[label][key]
                a=score(model,arm,holdout)
                outcomes[key]=a
                agreements[key]=action_agreement(model,holdout)
                cold=score(model,arm,holdout)
                noised=score(model,arm,negative)
                replay &= (row_identity(a["per_world"])==
                           row_identity(cold["per_world"])==
                           row_identity(noised["per_world"]))
                allbudgets &= a["selected_action_events"]==len(holdout)*STEPS
        controls={
          "source_balance_FIT_HOLDOUT_disjoint":disjoint and balanced,
          "same_FIT_optimizer_steps_world_draws_actual_selected_exposures":allbudgets
            and all(info["steps"]==UPDATES
                    and info["FIT_worlds_per_step"]==BATCH
                    and info["FIT_selected_event_exposures"]==UPDATES*BATCH*STEPS
                    for info in training[label].values()),
          "unobserved_proxy_negative_invariance":bool(replay),
          "exact_cold_replay":bool(replay),
          "original_causal_transition_effect_preserved":physical_ok
        }
        result=adjudicate(seed,outcomes,agreements,controls,freeze)
        decisions.append(result)
        all_scores[label]={key:{k:v for k,v in val.items() if k!="per_world"}
                           for key,val in outcomes.items()}
        all_agree[label]=agreements
    verdict=("DEV_RARE_SCRIPTED_ACTION_LEARNABILITY_SIGNAL_ONLY" if all(
        x["all_frozen_checks"] for x in decisions) else "DEV_DIAGNOSTIC_NO_GO")
    return {
        "schema":SCHEMA,"freeze_blob":FREEZE_BLOB,"parent":PARENT,
        "classification":freeze["classification"],
        "source":{"FIT":len(fit),"HOLDOUT":len(holdout),
                  "switched_HOLDOUT":len(holdout)//2,
                  "protected_external_OOD":0},
        "training":training,"scores":all_scores,"factual_teacher_agreement":all_agree,
        "verdicts":decisions,"DEV_admission":verdict,
        "independent_scientific_pass":False,
        "native_causal_law_discovery":False,
        "full_hybrid_substrate_qualification":False,
        "external_science_firstuse":False,
        "GPU_firstuse":False,
        "deployment":"NONE",
        "whole_process_wall_seconds":time.perf_counter()-started
    }


def retain_ci_receipt(result):
    if os.getenv("GITHUB_ACTIONS")!="true":
        return
    path=ROOT/"local_runs/p85mr_f8r73_rare_intervention_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R73_ORIGINAL_SCORED_HOLDOUT_ALREADY_EXISTS")
    path.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")


def deny_protected_firstuse(*_args,**_kwargs):
    raise PermissionError("F8R73_PROTECTED_EXTERNAL_GPU_FIRSTUSE_DENIED")
