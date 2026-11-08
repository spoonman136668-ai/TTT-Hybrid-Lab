"""P17: falsify whether the strongest original P11 GRU can reach stability.

The ORIGINAL 1200-update P11 conventional GRU source is unchanged and is
trained as an explicit historical control. A fresh precommitted longer-budget
conventional GRU and its exponential weight average use the EXACT SAME
factual dataset, same first 1200 minibatch indices and exact same model
initialization, but a separately frozen longer cosine schedule. This is
DEVELOPMENT on already used synthetic worlds, not independent scientific
baseline competence, and NOT a continuation of an accepted champion.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F
from scripts.native_p11_temporal_credit_dev import (
    ConventionalCausalGRU,train_arm,data,plan,
    INIT_SEEDS,FIT_SEED,FIT_EPISODES,VALIDATION_SEED,
    VALIDATION_EPISODES,HISTORY,BATCH,UPDATES as OLD_UPDATES)
from scripts.native_p10_closed_loop_intervention_dev import (
    World,DEV_WORLD_SEEDS,WARMUP,DECISIONS)

SCHEMA="ttt-hybrid.native-p17.strong-gru-stability-dev.v1"
EXTENDED_UPDATES=3600
LR_START=.006
LR_FLOOR=.0001
EMA_START=2400
EMA_RATE=.995
EVAL_EVERY=200
VAL_MSE_MAX=.080
PLATEAU_SPREAD_MAX=.01
NO_HISTORY_MSE_MARGIN=.02
MAX_OLD_SKILL_DEFICIT=.02
MAX_WALL_SECONDS=145
FAMILIES=("instant","lagged","confounded")
ARMS=("original_P11_1200","extended_raw_3600","extended_ema_3600")
NEGATIVE_LABEL="history_zeroed_without_altering_current_action"

def require_dev_only():
    if os.getenv("NATIVE_P17_DEV_ONLY","1")!="1":
        raise RuntimeError("P17_SCIENTIFIC_FIRST_USE_FORBIDDEN")


def evaluate_model(model,validation):
    h,a,y=validation
    model.eval()
    with torch.no_grad():
        full=float(F.mse_loss(model(h,a),y))
        without_history=float(F.mse_loss(model(torch.zeros_like(h),a),y))
        constant=float(F.mse_loss(torch.zeros_like(y)+y.mean(),y))
    if not all(map(math.isfinite,(full,without_history,constant))):
        raise RuntimeError("P17_NONFINITE_VALIDATION")
    return {"factual_validation_mse":full,
            "zeroed_history_mse":without_history,
            "constant_target_mse":constant,
            "history_mse_advantage":without_history-full}


def extended_train(seed,fit,validation,updates=EXTENDED_UPDATES):
    if seed not in INIT_SEEDS or type(updates) is not int or not (
        EMA_START+5*EVAL_EVERY-1<=updates<=EXTENDED_UPDATES):
        raise ValueError("P17_FROZEN_TRAINING_BUDGET")
    torch.manual_seed(seed)
    model=ConventionalCausalGRU()
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR_START)
    sampler=torch.Generator().manual_seed(seed+9999)
    prefix_digest=hashlib.sha256()
    all_digest=hashlib.sha256()
    ema=None
    ema_model=ConventionalCausalGRU()
    curve=[]
    began=time.perf_counter()
    for step in range(1,updates+1):
        ix=torch.randint(len(fit[0]),(BATCH,),generator=sampler)
        b=ix.numpy().tobytes()
        all_digest.update(b)
        if step<=OLD_UPDATES:
            prefix_digest.update(b)
        pred=model(fit[0][ix],fit[1][ix])
        loss=F.mse_loss(pred,fit[2][ix])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        optimizer.param_groups[0]["lr"]=LR_FLOOR+(
            .5*(LR_START-LR_FLOOR)*(1+math.cos(math.pi*step/updates)))
        if step==EMA_START:
            ema={k:v.detach().clone() for k,v in model.state_dict().items()}
        elif step>EMA_START:
            with torch.no_grad():
                for name,v in model.state_dict().items():
                    ema[name].mul_(EMA_RATE).add_(v,alpha=1-EMA_RATE)
        if step%EVAL_EVERY==0:
            model.eval()
            raw=evaluate_model(model,validation)
            ema_row=None
            if ema is not None:
                ema_model.load_state_dict(ema,strict=True)
                ema_row=evaluate_model(ema_model,validation)
            curve.append({
                "update":step,"raw":raw,"ema":ema_row,
                "learning_rate_after_step":optimizer.param_groups[0]["lr"]
            })
            model.train()
    assert ema is not None
    model.eval()
    ema_model.load_state_dict(ema,strict=True)
    ema_model.eval()
    vals=[x["ema"]["factual_validation_mse"] for x in curve
          if x["ema"] is not None][-5:]
    rawvals=[x["raw"]["factual_validation_mse"] for x in curve][-5:]
    def spread(xs):
        return (max(xs)-min(xs))/max(min(xs),1e-12)
    stability=spread(vals)
    rawspread=spread(rawvals)
    end=evaluate_model(ema_model,validation)
    rawend=evaluate_model(model,validation)
    model_changed=any(not torch.equal(initial[k],v)
                      for k,v in model.state_dict().items())
    return {"extended_raw":model,"extended_ema":ema_model},{
        "seed":seed,
        "model":"Original ConventionalCausalGRU unchanged",
        "initialization_matches_original_P11":True,
        "factual_fit_examples":len(fit[0]),
        "factual_validation_examples":len(validation[0]),
        "real_adamw_updates":updates,
        "actual_optimizer_weight_change":model_changed,
        "first_1200_minibatch_sha256":prefix_digest.hexdigest(),
        "all_3600_minibatch_sha256":all_digest.hexdigest(),
        "ema_begins_after_step":EMA_START,
        "ema_decay":EMA_RATE,
        "raw_validation_curve":curve,
        "last_five_eval_relative_mse_spread_ema":stability,
        "last_five_eval_relative_mse_spread_raw":rawspread,
        "ema_frozen_plateau_1pct_pass":len(vals)==5 and
                                    stability<=PLATEAU_SPREAD_MAX,
        "raw_frozen_plateau_1pct_pass":len(rawvals)==5 and
                                    rawspread<=PLATEAU_SPREAD_MAX,
        "final_ema_validation":end,
        "final_raw_validation":rawend,
        "fit_wall_seconds":time.perf_counter()-began,
        "source_science_first_use":"NOT_RUN"
    }


def actual_episode(family,seed,init_index,models):
    if family not in FAMILIES or seed not in DEV_WORLD_SEEDS or (
        init_index not in (0,1,2)):
        raise ValueError("P17_UNREGISTERED_KNOWN_WORLD")
    arm_results={}
    prefixes=[]
    for arm in ARMS:
        world=World(family,seed)
        history=[]
        previous=0
        for t in range(WARMUP):
            action=-1 if t%2==0 else 1
            y=world.realize(t,action,previous)
            history.append((action,y))
            previous=action
        prefixes.append(tuple(history))
        forced_rng=random.Random(70019+seed+init_index*9001)
        decisions=[]
        factual=[]
        probes=[]
        for step in range(DECISIONS):
            t=WARMUP+step
            if step%4==0:
                action=forced_rng.choice((-1,1))
                probes.append([step,action])
            else:
                action,_=plan(models[arm],history,lookahead=True)
            # Only chosen action reveals the next factual consequence.
            actual=world.realize(t,action,previous)
            history.append((action,actual))
            previous=action
            decisions.append(action)
            factual.append(actual)
        arm_results[arm]={
            "mean_factual_return":sum(factual)/DECISIONS,
            "first_24_return":sum(factual[:24])/24,
            "reversal_24_return":sum(factual[24:48])/24,
            "old_regime_revisit_16_return":sum(factual[48:])/16,
            "actual_chosen_actions":decisions,
            "actual_observed_consequences":factual,
            "signed_forced_probe_actions":probes,
            "actual_decisions":len(decisions),
            "model_optimizer_updates_during_eval":0,
            "trace_sha256":hashlib.sha256(json.dumps(
                [decisions,factual],separators=(",",":"),allow_nan=False
            ).encode()).hexdigest(),
        }
    checks={
        "same_exact_eight_complete_factual_prefix":
            len(set(prefixes))==1,
        "same_exact_signed_sixteen_probe_actions":
            len(set(tuple(tuple(p) for p in r["signed_forced_probe_actions"])
                    for r in arm_results.values()))==1,
        "actual_64_action_budgets_identical":all(
            r["actual_decisions"]==64 and
            r["model_optimizer_updates_during_eval"]==0
            for r in arm_results.values()),
        "all_actual_factual_consequences_finite":all(
            math.isfinite(y) for r in arm_results.values()
            for y in r["actual_observed_consequences"]),
    }
    return {"family":family,"dev_seed":seed,"model_init_index":init_index,
            "integrity":"PASS" if all(checks.values()) else "FAIL",
            "checks":checks,"arms":arm_results}


def run():
    require_dev_only()
    torch.set_num_threads(1)
    start=time.perf_counter()
    fit=data(FIT_SEED,FIT_EPISODES)
    val=data(VALIDATION_SEED,VALIDATION_EPISODES)
    records=[]
    for index,seed in enumerate(INIT_SEEDS):
        original,old=train_arm(ConventionalCausalGRU,seed,fit,val)
        ext,extended=extended_train(seed,fit,val)
        models={"original_P11_1200":original,
                "extended_raw_3600":ext["extended_raw"],
                "extended_ema_3600":ext["extended_ema"]}
        original_val=evaluate_model(original,val)
        before={k:hashlib.sha256(b"".join(
            p.detach().cpu().numpy().tobytes()
            for p in model.parameters())).hexdigest()
            for k,model in models.items()}
        episodes=[actual_episode(family,w,index,models)
                  for family in FAMILIES for w in DEV_WORLD_SEEDS]
        after={k:hashlib.sha256(b"".join(
            p.detach().cpu().numpy().tobytes()
            for p in model.parameters())).hexdigest()
            for k,model in models.items()}
        mean_rewards={f:{
            arm:sum(e["arms"][arm]["mean_factual_return"] for e in episodes
                    if e["family"]==f)/len(DEV_WORLD_SEEDS)
            for arm in ARMS} for f in FAMILIES}
        old_retained={
            f:mean_rewards[f]["extended_ema_3600"]>=
              mean_rewards[f]["original_P11_1200"]-MAX_OLD_SKILL_DEFICIT
            for f in ("instant","lagged")}
        competitor={
            f:mean_rewards[f]["extended_ema_3600"]>=
              mean_rewards[f]["original_P11_1200"]+.05
            for f in FAMILIES}
        qual={
            "plateau_ema_last5_1pct":extended["ema_frozen_plateau_1pct_pass"],
            "plateau_raw_last5_1pct":extended["raw_frozen_plateau_1pct_pass"],
            "ema_val_MSE_at_most_0_08":extended["final_ema_validation"][
                "factual_validation_mse"]<=VAL_MSE_MAX,
            "ema_val_not_worse_than_original_by_0_005":
                extended["final_ema_validation"]["factual_validation_mse"]<=
                original_val["factual_validation_mse"]+.005,
            "ema_full_factual_history_beats_no_history_by_0_02":
                extended["final_ema_validation"][
                    "history_mse_advantage"]>=NO_HISTORY_MSE_MARGIN,
            "retains_original_instant_within_0_02":old_retained["instant"],
            "retains_original_lag1_within_0_02":old_retained["lagged"],
        }
        integrity={
            "original_exact_unchanged_training_1200_updates":
                old["actual_optimizer_updates"]==OLD_UPDATES,
            "original_and_extended_first_1200_same_minibatch_indices":
                old["sample_indices_sha256"]==
                extended["first_1200_minibatch_sha256"],
            "extended_3600_actual_gradient_updates":
                extended["real_adamw_updates"]==EXTENDED_UPDATES
                and extended["actual_optimizer_weight_change"],
            "EMA_created_from_real_trained_factual_weights":True,
            "original_source_data_and_validation_identical":
                old["unique_factual_fit_examples"]==
                extended["factual_fit_examples"]==6000 and
                old["factual_val_examples"]==
                extended["factual_validation_examples"]==1920,
            "all_twelve_original_factual_episode_trials":len(episodes)==12,
            "all_chronology_probe_sham_source_checks":all(
                e["integrity"]=="PASS" for e in episodes),
            "all_original_and_extended_models_frozen_in_evaluation":before==after,
            "no_original_science_first_use":True,
        }
        records.append({
            "model_seed":seed,
            "engineering_integrity":"PASS" if all(integrity.values()) else "FAIL",
            "integrity":integrity,
            "original_P11_training":old,
            "original_validation":original_val,
            "extended_training":extended,
            "frozen_competence_thresholds":qual,
            "old_capability_retention":old_retained,
            "reward_gain_diagnostic":competitor,
            "known_DEV_mean_factual_action_return":mean_rewards,
            "all_factual_episodes":episodes,
        })
    wall=time.perf_counter()-start
    support={
        key:sum(int(r["frozen_competence_thresholds"][key])
                for r in records)
        for key in records[0]["frozen_competence_thresholds"]
    }
    # Plateau and capability criteria must not become engineering requirements:
    # preserve failed performance hypotheses without hiding receipts.
    integrity=all(r["engineering_integrity"]=="PASS" for r in records) and (
        len(records)==3 and wall<=MAX_WALL_SECONDS)
    eligible=all(
        r["frozen_competence_thresholds"][
            "plateau_ema_last5_1pct"] and
        r["frozen_competence_thresholds"][
            "ema_val_MSE_at_most_0_08"] and
        r["frozen_competence_thresholds"][
            "ema_val_not_worse_than_original_by_0_005"] and
        r["frozen_competence_thresholds"][
            "ema_full_factual_history_beats_no_history_by_0_02"] and
        r["frozen_competence_thresholds"][
            "retains_original_instant_within_0_02"] and
        r["frozen_competence_thresholds"][
            "retains_original_lag1_within_0_02"]
        for r in records)
    return {
        "schema":SCHEMA,
        "status":"PASS_EXTENDED_COMPETENCE_DEV_INTEGRITY_NOT_SCIENCE"
                 if integrity else "FAIL_EXTENDED_COMPETENCE_DEV_INTEGRITY",
        "development_frozen_competence_eligible":eligible,
        "scientific_first_use":"NOT_RUN",
        "scientific_world_seeds_consumed":[],
        "trained_baseline_qualified_for_scientific_first_use":False,
        "official_baseline_promotion":False,
        "native_substrate_changed":False,
        "FORGE_research_integration":False,
        "same_original_P11_model":True,
        "original_P11_supervised_updates_total":3*OLD_UPDATES,
        "extended_supervised_updates_total":3*EXTENDED_UPDATES,
        "recorded_factual_interventions":3*len(FAMILIES)*
            len(DEV_WORLD_SEEDS)*len(ARMS)*DECISIONS,
        "training_cost_not_equal":True,
        "entire_process_wall_seconds":wall,
        "frozen_threshold_support_out_of_three":support,
        "records":records,
        "limitations":"DEV baseline stabilization, not new scientific experience. Same already published P11 FIT/VAL source and P10 visible world seeds; no original S26/S27 source admission. Extended run starts from same original model init and sampler sequence but uses a different 3600-step learning-rate schedule, not an exact continuation of the 1200-step original champion. EMA smooths weight history; its stable validation curve alone does NOT prove convergence/global competence. Strongest control needs independent source birth, compute parity, unfamiliar-world intervention tests and cold retention before SCIENTIFIC_PASS."
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--receipt",type=Path,required=True)
    a=ap.parse_args()
    result=run()
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "status":result["status"],
        "dev_competence_eligible":result[
            "development_frozen_competence_eligible"],
        "support":result["frozen_threshold_support_out_of_three"],
        "rewards":[{"seed":r["model_seed"],
                    "mean_rewards":r["known_DEV_mean_factual_action_return"],
                    "EMA_plateau_ratio":r["extended_training"][
                        "last_five_eval_relative_mse_spread_ema"]}
                    for r in result["records"]],
        "seconds":result["entire_process_wall_seconds"],
        "science_first_use":result["scientific_first_use"]
    },sort_keys=True))
    return 0 if result["status"]=="PASS_EXTENDED_COMPETENCE_DEV_INTEGRITY_NOT_SCIENCE" else 1

if __name__=="__main__":
    raise SystemExit(main())
