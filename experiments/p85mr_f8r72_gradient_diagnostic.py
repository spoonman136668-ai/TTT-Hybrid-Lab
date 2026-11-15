"""F8-R72 prospective factual learning-signal diagnostic on NEW synthetic worlds.

We compare R71 reward-only REINFORCE training to public, factual *scripted
behavior imitation* for the same three CPU policy architectures, exposing
whether the budget can learn an OBSERVABLE fixed policy at all. Imitating the
fixed rule is not learning new causal law and is never classified as science.
No R71 original scored worlds, weights or private labels are consumed.
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

from experiments import p85mr_f8r71_trained_causal_action as r71

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r72_gradient_diagnostic_freeze.json"
FREEZE_BLOB="68155d115d8216f766319c76dd253a02ad71e56f"
SCHEMA="ttt-hybrid.f8r72-factual-gradient-diagnostics.v1"
PARENT="220377e19217a5535da27c34ed0b8bd50d0185c6"
SALTS={"FIT":24410000,"HOLDOUT":24430000,"PROXY_NEGATIVE":24440000}
COUNTS={"FIT":128,"HOLDOUT":48}
SEEDS=(985601,985607)
ARMS=r71.TRAINED
METHODS=("SELECTED_RETURN_REINFORCE","FACTUAL_TEACHER_BEHAVIOR_CLONE")
NEGATIVE="STRUCTURED_SHUFFLED_LABEL_BC"
FIXED=("R70_CALIBRATE_CAUSAL","R70_NO_CALIBRATE_REACTIVE")
UPDATES=32
BATCH=8
STEPS=40
BOOT_SEED=24449999


def blob_sha(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def locked():
    raw=FREEZE.read_bytes()
    if blob_sha(raw)!=FREEZE_BLOB:
        raise ValueError("F8R72_ORIGINAL_FREEZE_BLOB_DRIFT")
    f=json.loads(raw)
    c=f["source_custody"]
    tr=f["training"]
    sc=f["scoring"]
    a=f["arms"]
    if (f["schema"]!=SCHEMA
        or f["stage"]!="PROSPECTIVE_FREEZE_BEFORE_R72_SOURCE_TRAINING_OR_HOLDOUT"
        or c["parent_R71_SHA"]!=PARENT
        or c["R71_original_result"]!="ENGINEERING_PASS_DEV_NO_GO"
        or not c["all_R64_to_R71_scored_worlds_spent"]
        or c["new_source_salts"]!=SALTS or c["counts"]!=COUNTS
        or c["external_OOD_used"]
        or a["architectures"]!=list(ARMS) or a["methods"]!=list(METHODS)
        or a["policy_action_set"]!=list(r71.ROLLOUT_ACTIONS)
        or tr["seeds"]!=list(SEEDS)
        or tr["steps_per_seed_architecture_method"]!=UPDATES
        or tr["FIT_world_batch"]!=BATCH
        or tr["factual_events_per_seed_architecture_method"]!=UPDATES*BATCH*STEPS
        or tr["learning_rate"]!=.015
        or tr["weight_decay"]!=.001
        or sc["bootstrap"]["seed"]!=BOOT_SEED
        or sc["bootstrap"]["resamples"]!=2000
        or sc["no_post_score_changes"] is not True
        or any(f["authority"].values())):
        raise ValueError("F8R72_FROZEN_SEMANTIC_CONTRACT_DRIFT")
    return f


def worlds(panel):
    """R72 source owns its own seeds; runtime panel alias is for R70 interface."""
    if panel not in COUNTS:
        raise PermissionError("F8R72_SPENT_OR_PROTECTED_SOURCE_DENIED")
    rng=np.random.default_rng(SALTS[panel])
    n=COUNTS[panel]
    balanced=[(sw,sg) for sw in (False,True) for sg in (-1,1)
              for _ in range(n//4)]
    order=rng.permutation(n)
    worlds_out=[]
    for i,slot in enumerate(order):
        sw,sg=balanced[int(slot)]
        b=int(rng.integers(9,19))
        a=int(rng.integers(22,36))
        latent=np.full(STEPS,sg,dtype=np.int64)
        if sw:
            latent[b:a]=-sg
        innovations=rng.normal(0,1,(STEPS,4,4))
        nuisance=rng.normal(0,1,(STEPS,4))
        runtime="FIT" if panel=="FIT" else "DEV"
        worlds_out.append({
            "panel":runtime,"source_panel":runtime,
            "R72_source_panel":panel,
            "index":int(i),
            "switched_evaluator_only":bool(sw),
            "initial_sign_evaluator_only":int(sg),
            "B_start_evaluator_only":b,"A_return_start_evaluator_only":a,
            "latent_law_evaluator_only":latent,
            "potential_innovations_evaluator_only":innovations,
            "unobserved_nuisance_proxy":nuisance
        })
    return worlds_out


def negative_proxy(holdout):
    if (len(holdout)!=COUNTS["HOLDOUT"]
        or any(w.get("R72_source_panel")!="HOLDOUT" for w in holdout)):
        raise PermissionError("F8R72_NEGATIVE_REQUIRES_FRESH_HOLDOUT")
    rng=np.random.default_rng(SALTS["PROXY_NEGATIVE"])
    return [{**w,"unobserved_nuisance_proxy":rng.normal(0,5,(STEPS,4))}
            for w in holdout]


def teacher_history(world):
    """Only public scripted fixed teacher and its actually chosen consequences."""
    if world.get("R72_source_panel") not in COUNTS:
        raise PermissionError("F8R72_TEACHER_SOURCE_DENIED")
    sample=r71.fixed_rollout("R70_CALIBRATE_CAUSAL",world)
    acts=sample["actions"]
    observations=sample["observations"]
    if len(acts)!=STEPS or len(observations)!=STEPS:
        raise ValueError("F8R72_INVALID_TEACHER_EPISODE")
    if any(a not in r71.ROLLOUT_ACTIONS for a in acts):
        raise ValueError("F8R72_TEACHER_SELECTED_SHAM")
    return acts,observations


def teacher_loss(model,world,*,shift_labels=False):
    """Behavior cloning uses actual selected teacher transitions, not oracle law."""
    acts,observations=teacher_history(world)
    history=[]
    carry=model.initial()
    losses=[]
    for t,(a,ob) in enumerate(zip(acts,observations)):
        logits,carry=model.logits(history,t,carry)
        target_action=acts[(t+5)%STEPS] if shift_labels else a
        target=r71.ROLLOUT_ACTIONS.index(target_action)
        losses.append(-F.log_softmax(logits,dim=0)[target])
        history.append((a,ob))
        if model.arm==ARMS[0]:
            carry=r71._update_factual_memory(carry,a,ob)
    return torch.stack(losses).mean()


def policy_gradient_loss(trajectories):
    rtg=np.stack([r71._return_to_go(t["rewards"]) for t in trajectories])
    advantage=torch.tensor((rtg-rtg.mean(axis=0))/8.,dtype=torch.float64)
    logp=torch.stack([torch.stack(t["logprobs"]) for t in trajectories])
    entropy=torch.stack([torch.stack(t["entropies"]) for t in trajectories])
    return -(logp*advantage).mean()-.005*entropy.mean()


def _fit_batches(seed,fit):
    rng=np.random.default_rng(24419999+seed)
    return rng.integers(0,len(fit),(UPDATES,BATCH))


def gradient_quality(arm,seed,fit):
    """Untrained initial factual policy gradient from disjoint FIT8 batches.

    Descriptive diagnostics only: no source selection, hyperparameter tuning,
    acceptance/promotion based on this value. The gradient is reward-only.
    """
    model=r71.TrainablePolicy(arm,seed)
    draws=np.random.default_rng(24417777+seed).permutation(len(fit))[:2*BATCH]
    gradients=[]
    norms=[]
    for k in range(2):
        action_gen=torch.Generator(device="cpu").manual_seed(
            24415555+seed+100*r71.TRAINED.index(arm)+k)
        batch=[r71.rollout(model,fit[int(i)],greedy=False,action_gen=action_gen)
               for i in draws[k*BATCH:(k+1)*BATCH]]
        loss=policy_gradient_loss(batch)
        model.zero_grad(set_to_none=True)
        loss.backward()
        grad=torch.cat([p.grad.reshape(-1).detach() for p in model.parameters()])
        norm=float(grad.norm().item())
        gradients.append(grad)
        norms.append(norm)
    cosine=float(F.cosine_similarity(
        gradients[0].reshape(1,-1),gradients[1].reshape(1,-1)).item())
    return {"initial_gradient_cosine_disjoint_FIT_batches":cosine,
            "initial_gradient_norm_batchA":norms[0],
            "initial_gradient_norm_batchB":norms[1],
            "FIT_worlds_per_gradient_batch":BATCH}


def train_one(arm,seed,method,fit):
    if (arm not in ARMS or seed not in SEEDS
        or method not in METHODS+(NEGATIVE,)
        or (method==NEGATIVE and arm!=ARMS[0])
        or len(fit)!=COUNTS["FIT"]
        or any(w.get("R72_source_panel")!="FIT" for w in fit)):
        raise PermissionError("F8R72_TRAIN_ONLY_FRESH_FIT_FROZEN_ARMS")
    t0=time.perf_counter()
    model=r71.TrainablePolicy(arm,seed)
    optim=torch.optim.AdamW(model.parameters(),lr=.015,weight_decay=.001)
    draws=_fit_batches(seed,fit)
    action_gen=torch.Generator(device="cpu").manual_seed(
        24415000+seed+1337*ARMS.index(arm))
    first=None
    last=None
    for batch_ids in draws:
        if method==METHODS[0]:
            trajectories=[r71.rollout(
                model,fit[int(i)],greedy=False,action_gen=action_gen)
                for i in batch_ids]
            loss=policy_gradient_loss(trajectories)
        else:
            loss=torch.stack([teacher_loss(
                model,fit[int(i)],shift_labels=(method==NEGATIVE))
                for i in batch_ids]).mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R72_NONFINITE_TRAINING_LOSS")
        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        optim.step()
        if first is None:
            first=float(loss.detach())
        last=float(loss.detach())
    weight=hashlib.sha256(b"".join(
        p.detach().cpu().numpy().astype("<f8").tobytes()
        for p in model.parameters())).hexdigest()
    return model,{
       "arm":arm,"seed":seed,"method":method,
       "steps":UPDATES,"FIT_worlds_per_step":BATCH,
       "FIT_actual_selected_event_exposures":UPDATES*BATCH*STEPS,
       "trainable_parameters":int(sum(p.numel() for p in model.parameters())),
       "allocated_persistent_floats":16 if arm!=ARMS[2] else 0,
       "loss_first":first,"loss_last":last,
       "trained_weight_sha256":weight,
       "fit_training_seconds":time.perf_counter()-t0
    }


def action_agreement(model,source):
    """Teacher-forced *public* action imitation: separate from closed-loop reward."""
    correct=total=probes=probe_hits=false_probes=nonprobe=0
    for w in source:
        acts,obs=teacher_history(w)
        history=[]
        carry=model.initial()
        with torch.no_grad():
            for t,(a,y) in enumerate(zip(acts,obs)):
                logits,carry=model.logits(history,t,carry)
                prediction=r71.ROLLOUT_ACTIONS[int(torch.argmax(logits).item())]
                correct+=int(prediction==a)
                total+=1
                if t in (0,16,32):
                    probes+=1
                    probe_hits+=int(prediction==0)
                else:
                    nonprobe+=1
                    false_probes+=int(prediction==0)
                history.append((a,y))
                if model.arm==ARMS[0]:
                    carry=r71._update_factual_memory(carry,a,y)
    return {
       "teacher_factual_action_agreement":correct/total,
       "public_calibration_recall":probe_hits/probes,
       "outside_calibration_false_positive_rate":false_probes/nonprobe,
       "observed_factual_selected_events":total
    }


def _score(arm,model,source):
    if arm=="R70_CALIBRATE_CAUSAL" or arm=="R70_NO_CALIBRATE_REACTIVE":
        s=r71.score_one(arm,None,source)
    elif arm in ARMS:
        s=r71.score_one(arm,model,source)
    else:
        raise PermissionError("F8R72_UNREGISTERED_SCORE_ARM")
    return s


def _digest_rows(rows):
    return tuple((v["world"],v["actions"],v["episode_reward"],
                  v["late_return_reward"],v["observation_sha256"],
                  v["physical_sha256"]) for v in rows)


def verdict(seed,scores,agreement,source_check,freeze):
    fixed=scores["R70_CALIBRATE_CAUSAL"]
    no_calib=scores["R70_NO_CALIBRATE_REACTIVE"]
    bc=scores["GRU16_BC"]
    pg=scores["GRU16_PG"]
    neg=agreement["STRUCTURED_SHUFFLED_LABEL_BC"]
    bc_accuracy=agreement["GRU16_BC"]
    teacher_vs_bc=bc["episode_reward"]/max(abs(fixed["episode_reward"]),1e-12)
    net_gain=bc["episode_reward"]-pg["episode_reward"]
    per_pg={v["world"]:v["episode_reward"] for v in pg["per_world"]}
    paired=np.asarray([v["episode_reward"]-per_pg[v["world"]]
                       for v in bc["per_world"]],dtype=np.float64)
    rng=np.random.default_rng(BOOT_SEED+seed)
    indices=rng.integers(0,len(paired),(2000,len(paired)))
    low=float(np.quantile(paired[indices].mean(axis=1),.025))
    limits=freeze["scoring"]["all_required_for_mechanism_signal_each_seed"]
    checks={
        "bc_action_agreement":bc_accuracy["teacher_factual_action_agreement"]>=limits["GRU16_BC_factual_action_agreement_min"],
        "bc_calibration_recall":bc_accuracy["public_calibration_recall"]>=limits["GRU16_BC_calibration_recall_min"],
        "bc_calibration_false_positive":bc_accuracy["outside_calibration_false_positive_rate"]<=limits["GRU16_BC_calibration_false_positive_max"],
        "bc_reward_fraction_teacher":teacher_vs_bc>=limits["GRU16_BC_net_reward_fraction_of_fixed_calibration_min"],
        "bc_net_vs_RL":net_gain>=limits["GRU16_BC_net_reward_over_GRU16_RL_min_absolute"],
        "bc_vs_shuffled_agreement":bc_accuracy["teacher_factual_action_agreement"]-neg["teacher_factual_action_agreement"]>=limits["GRU16_BC_action_agreement_over_shuffled_structured_min"],
        "bootstrap_lower_positive":low>0,
        "all_source_qualifications_and_negative_replays":bool(source_check)
    }
    return {
       "seed":seed,"checks":checks,"all_frozen_checks":all(checks.values()),
       "GRU_BC_closed_loop_reward":bc["episode_reward"],
       "GRU_PG_closed_loop_reward":pg["episode_reward"],
       "fixed_reference_reward":fixed["episode_reward"],
       "fixed_no_calibration_reward":no_calib["episode_reward"],
       "GRU_BC_over_PG_reward":net_gain,
       "GRU_BC_fraction_of_teacher_reward":teacher_vs_bc,
       "GRU_BC_minus_PG_bootstrap_lower95":low,
       "GRU_BC_minus_shuffled_structured_action_agreement":
           bc_accuracy["teacher_factual_action_agreement"]-
           neg["teacher_factual_action_agreement"]
    }


def run():
    started=time.perf_counter()
    f=locked()
    fit=worlds("FIT")
    # Lock the training corpus and ALL model weights BEFORE any HOLDOUT
    # source outcomes, private laws or potential innovations are generated.
    all_models={}
    training={}
    gradient={}
    for seed in SEEDS:
        gradient[str(seed)]={arm:gradient_quality(arm,seed,fit) for arm in ARMS}
        all_models[str(seed)]={}
        training[str(seed)]={}
        for arm in ARMS:
            for method in METHODS+(NEGATIVE,):
                if method==NEGATIVE and arm!=ARMS[0]:
                    continue
                key=arm+("_PG" if method==METHODS[0] else "_BC"
                         if method==METHODS[1] else "_SHUFFLED_LABEL_BC")
                model,info=train_one(arm,seed,method,fit)
                all_models[str(seed)][key]=model
                training[str(seed)][key]=info
    holdout=worlds("HOLDOUT")
    id_hash=lambda w:hashlib.sha256(
        w["potential_innovations_evaluator_only"].astype("<f8").tobytes()).hexdigest()
    unique=set(map(id_hash,fit)).isdisjoint(map(id_hash,holdout))
    balanced=all(
        len(arr)==COUNTS[panel]
        and sum(w["switched_evaluator_only"] for w in arr)==len(arr)//2
        and sum(w["initial_sign_evaluator_only"]==1 for w in arr)==len(arr)//2
        and all(sum(w["switched_evaluator_only"]==sw and
                    w["initial_sign_evaluator_only"]==sign for w in arr)==len(arr)//4
                for sw in (False,True) for sign in (-1,1))
        for panel,arr in (("FIT",fit),("HOLDOUT",holdout)))
    if not unique or not balanced:
        raise ValueError("F8R72_SOURCE_ADMISSION_FAILED")
    perturbed=negative_proxy(holdout)
    all_scores={}
    all_agreement={}
    decisions=[]
    fixed_scores={name:_score(name,None,holdout) for name in FIXED}
    fixed_replay=all(
       _digest_rows(fixed_scores[name]["per_world"])==
       _digest_rows(_score(name,None,holdout)["per_world"])==
       _digest_rows(_score(name,None,perturbed)["per_world"])
       for name in FIXED)
    for seed in SEEDS:
        key=str(seed)
        scores={name:fixed_scores[name] for name in FIXED}
        agreement={}
        replay=bool(fixed_replay)
        for name,model in all_models[key].items():
            scored=_score(model.arm,model,holdout)
            scores[name]=scored
            agreement[name]=action_agreement(model,holdout)
            fresh=_score(model.arm,model,holdout)
            negative=_score(model.arm,model,perturbed)
            replay &= (
                _digest_rows(scored["per_world"])==
                _digest_rows(fresh["per_world"])==
                _digest_rows(negative["per_world"]))
        source_check=(
            unique and balanced and replay
            and all(v["selected_action_events"]==len(holdout)*STEPS
                    for v in scores.values())
            and all(info["steps"]==UPDATES
                    and info["FIT_actual_selected_event_exposures"]==UPDATES*BATCH*STEPS
                    for info in training[key].values()))
        decision=verdict(seed,scores,agreement,source_check,f)
        decision["source_nonoverlap"]=bool(unique)
        decision["balanced_laws"]=bool(balanced)
        decision["cold_replay_and_unobserved_proxy_negative"]=bool(replay)
        decisions.append(decision)
        all_scores[key]={n:{k:v for k,v in score.items() if k!="per_world"}
                         for n,score in scores.items()}
        all_agreement[key]=agreement
    return {
       "schema":SCHEMA,"freeze_blob":FREEZE_BLOB,"parent":PARENT,
       "classification":f["classification"],
       "source":{"FIT":len(fit),"HOLDOUT":len(holdout),
                 "switched_HOLDOUT":len(holdout)//2,
                 "protected_external_OOD":0},
       "initial_selected_reward_gradient_diagnostics":gradient,
       "training":training,"scores":all_scores,
       "teacher_forced_factual_action_agreement":all_agreement,
       "verdicts":decisions,
       "DEV_admission":"DEV_SCRIPTED_BEHAVIOR_LEARNABILITY_SIGNAL_ONLY"
          if all(x["all_frozen_checks"] for x in decisions)
          else "DEV_DIAGNOSTIC_NO_GO",
       "independent_scientific_pass":False,
       "novel_causal_law_learning_proven":False,
       "full_hybrid_substrate_qualification":False,
       "GPU_firstuse":False,"external_scientific_firstuse":False,
       "deployment":"NONE",
       "whole_process_wall_seconds":time.perf_counter()-started
    }


def retain_ci_receipt(report):
    if os.environ.get("GITHUB_ACTIONS")!="true":
        return
    path=ROOT/"local_runs/p85mr_f8r72_gradient_diagnostic_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R72_NO_OVERWRITE_SCORED_HOLDOUT")
    path.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")


def deny_protected_firstuse(*_args,**_kwargs):
    raise PermissionError("F8R72_PROTECTED_GPU_OR_EXTERNAL_FIRSTUSE_DENIED")
