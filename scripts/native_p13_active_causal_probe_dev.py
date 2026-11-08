"""Native P13 active intervention value: a falsifiable causal-probing DEV assay.

A purposeful D-optimal lag-disambiguation probe competes with random probes
at the EXACT same fixed decision positions and equal online feedback updates.
The prior P12 false negative ('permuted' lag columns merely relabeled
features) is corrected here: an intentionally broken control shuffles the
temporal ACTION HISTORY inside the factual learning feature, not coefficient
column names. All worlds are KNOWN synthetic engineering DEV, NOT scientific
first-use. No production substrate, checkpoint or FORGE change.
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
from scripts.native_p12_online_causal_delay_dev import (
    CausalDelayWorld,FAMILIES,DEV_SEEDS,OBS_PREFIX,DECISIONS,
    FactualLagEstimator,features)
from scripts.native_p11_temporal_credit_dev import (
    ConventionalCausalGRU,train_arm,data,plan,INIT_SEEDS,
    FIT_SEED,FIT_EPISODES,VALIDATION_SEED,VALIDATION_EPISODES,
    UPDATES,HISTORY)

SCHEMA="ttt-hybrid.native-p13.active-lag-probing-dev.v1"
ARMS=("active_probe","passive_random_probe","no_probe",
      "scrambled_feedback_lags","adaptive_gru","frozen_gru",
      "random","privileged_diagnostic")
PROBE_INTERVAL=4
PROBES_PER_EPISODE=16
ONLINE_ADAMW_LR=.001
MIN_INFORMATION_SCORE_GAP=0.
MAX_PROCESS_SECONDS=115


def guard_dev():
    if os.environ.get("NATIVE_P13_DEV_ONLY","1")!="1":
        raise RuntimeError("P13_ORIGINAL_SCIENTIFIC_FIRST_USE_FORBIDDEN")


def information_scores(learner:FactualLagEstimator,previous:list[int])->dict[int,float]:
    """Pre-outcome D-optimal design proxy: x^T Gram^-1 x for +/- action.

    Both active and passive controllers compute both scores, so their online
    matrix/feature work matches; passive discards scores to choose randomly.
    """
    gram=learner.gram
    scores={}
    for action in (-1,1):
        x=features(action,previous,"full")
        v=float(torch.dot(x,torch.linalg.solve(gram,x)))
        if not math.isfinite(v) or v<0:
            raise RuntimeError("P13_NONFINITE_INFORMATION_SCORE")
        scores[action]=v
    return scores


def action_from_model(learner:FactualLagEstimator,rng:random.Random,
                      previous:list[int],step:int,*,policy:str)->tuple[int,bool,dict|None]:
    if policy not in ("active","passive","no_probe","scrambled"):
        raise ValueError("P13_INVALID_POLICY")
    is_probe=(step%PROBE_INTERVAL==0 and policy!="no_probe")
    if is_probe:
        info=information_scores(learner,previous)
        if policy=="active":
            if abs(info[1]-info[-1])<=MIN_INFORMATION_SCORE_GAP:
                return rng.choice((-1,1)),True,info
            return (1 if info[1]>info[-1] else -1),True,info
        return rng.choice((-1,1)),True,info
    # Four causal future slots: current action's effects on outcomes at
    # t, t+1, t+2, t+3. Future lags/regimes remain UNOBSERVED.
    beta=learner.coefficients()[1:].tolist()
    direction=sum(beta)
    if abs(direction)<.08:
        return rng.choice((-1,1)),False,None
    return (1 if direction>0 else -1),False,None


def update_with_actual_feedback(learner:FactualLagEstimator,action:int,
                                past:list[int],actual:float,
                                *,scramble_rng:random.Random|None):
    # Previous ACTIONS are history, never outcome labels or latent regimes.
    if scramble_rng is None:
        actual_past=past
        changed=False
    else:
        actual_past=list(past)
        # Entire prior history scrambled, destroying the relation between
        # a_(t-k) and observed y_t. Matrix shape/updates remain equal.
        scramble_rng.shuffle(actual_past)
        changed=actual_past[-3:]!=past[-3:] if len(past)>=3 else False
    learner.observe(action,actual_past,actual,scored=True)
    return changed


def evaluate(family:str,world_seed:int,model_seed:int,trained_gru)->dict:
    if family not in FAMILIES or world_seed not in DEV_SEEDS or (
        model_seed not in INIT_SEEDS):
        raise ValueError("P13_WORLD_OR_MODEL_SOURCE_DENIED")
    records={}
    prefixes=[]
    for arm in ARMS:
        world=CausalDelayWorld(family,world_seed)
        history=[]
        for t in range(OBS_PREFIX):
            act=1 if t%2 else -1
            outcome=world.realize(t,act)
            history.append((act,outcome))
        prefixes.append(tuple(history))
        learner=(FactualLagEstimator("full",adapt=True)
                 if arm in ("active_probe","passive_random_probe","no_probe",
                            "scrambled_feedback_lags") else None)
        if learner:
            for t,(act,y) in enumerate(history):
                learner.observe(act,[x for x,_ in history[:t]],y,scored=False)
        net=copy.deepcopy(trained_gru) if arm in ("adaptive_gru","frozen_gru") else None
        optimizer=(torch.optim.AdamW(net.parameters(),lr=ONLINE_ADAMW_LR)
                   if arm=="adaptive_gru" else None)
        policy_rng=random.Random(world_seed+model_seed*1009+
                                 97*ARMS.index(arm))
        history_rng=random.Random(world_seed+model_seed*3701+71337)
        choices=[]
        outcomes=[]
        probe_count=0
        shuffled_count=0
        update_count=0
        prequential_error=[]
        probe_records=[]
        start=time.perf_counter()
        for step in range(DECISIONS):
            t=OBS_PREFIX+step
            past_actions=[x for x,_ in history]
            prehistory=list(history)
            if learner:
                mode=("active" if arm=="active_probe" else
                      "passive" if arm=="passive_random_probe" else
                      "scrambled" if arm=="scrambled_feedback_lags" else
                      "no_probe")
                action,is_probe,info=action_from_model(
                    learner,policy_rng,past_actions,step,policy=mode)
                if is_probe:
                    probe_count+=1
                    probe_records.append({"step":step,"choice":action,
                                          "information_scores":info})
            elif net is not None:
                if step%PROBE_INTERVAL==0:
                    action=policy_rng.choice((-1,1))
                    probe_count+=1
                else:
                    action,_=plan(net,history,lookahead=True)
            elif arm=="random":
                action=policy_rng.choice((-1,1))
            else:
                # Privileged world sign/lag is NEVER a baseline control.
                lag=world.lag(t)
                looked_ahead=min(OBS_PREFIX+DECISIONS-1,t+(lag or 0))
                action=world.sign(looked_ahead)
            assert action in (-1,1)
            # Crucial: reveal exactly one FACTUAL action consequence now.
            actual=world.realize(t,action)
            if learner:
                changed=update_with_actual_feedback(
                    learner,action,past_actions,actual,
                    scramble_rng=(history_rng if arm=="scrambled_feedback_lags"
                                  else None))
                shuffled_count+=int(changed)
                update_count+=1
            if optimizer:
                net.train()
                h=torch.tensor(prehistory[-HISTORY:],dtype=torch.float32).reshape(
                    1,HISTORY,2)
                act=torch.tensor([[float(action)]])
                target=torch.tensor([[actual]],dtype=torch.float32)
                pred=net(h,act)
                loss=torch.nn.functional.mse_loss(pred,target)
                prequential_error.append(float(loss.detach()))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
                optimizer.step()
                net.eval()
                update_count+=1
            choices.append(action)
            outcomes.append(actual)
            history.append((action,actual))
        elapsed=time.perf_counter()-start
        coeff=learner.coefficients().tolist() if learner else None
        records[arm]={
            "mean_factual_return":sum(outcomes)/DECISIONS,
            "first_phase_return":sum(outcomes[:24])/24,
            "changed_phase_return":sum(outcomes[24:48])/24,
            "old_phase_revisit_return":sum(outcomes[48:])/16,
            "all_chosen_actions":choices,
            "all_factual_consequences":outcomes,
            "scored_action_count":len(choices),
            "online_actual_feedback_updates":update_count,
            "fixed_probe_opportunities_used":probe_count,
            "probe_records":probe_records,
            "history_alignments_deliberately_broken":shuffled_count,
            "learned_online_coefficients":coeff,
            "sham_max_abs_spurious_action_coeff":(
                max(abs(x) for x in coeff[1:])
                if family=="sham_no_action" and coeff else None),
            "adaptive_gru_prequential_mse":(
                sum(prequential_error)/len(prequential_error)
                if prequential_error else None),
            "online_policy_wall_seconds":elapsed,
            "trace_sha256":hashlib.sha256(json.dumps(
                [choices,outcomes],separators=(",",":"),allow_nan=False
            ).encode()).hexdigest()
        }
    checks={
        "common_eight_factual_start_observations":len(set(prefixes))==1,
        "same_sixtyfour_action_opportunities":all(
            r["scored_action_count"]==DECISIONS for r in records.values()),
        "all_learning_online_policies_64_feedback_updates":all(
            records[a]["online_actual_feedback_updates"]==64 for a in
            ("active_probe","passive_random_probe","no_probe",
             "scrambled_feedback_lags","adaptive_gru")),
        "exactly_sixteen_precommitted_probe_slots_for_matched_arms":all(
            records[a]["fixed_probe_opportunities_used"]==PROBES_PER_EPISODE
            for a in ("active_probe","passive_random_probe",
                      "scrambled_feedback_lags","adaptive_gru","frozen_gru")),
        "frozen_and_random_never_receive_gradient_updates":all(
            records[a]["online_actual_feedback_updates"]==0 for a in
            ("frozen_gru","random","privileged_diagnostic")),
        "scramble_really_breaks_alignment":(
            records["scrambled_feedback_lags"][
                "history_alignments_deliberately_broken"]>0),
        "all_factual_consequences_finite":all(
            math.isfinite(v) for r in records.values()
            for v in r["all_factual_consequences"]),
        "sham_has_identical_rewards_regardless_of_decisions":(
            all(r["all_factual_consequences"]==
                records["random"]["all_factual_consequences"]
                for r in records.values()) if family=="sham_no_action" else True)
    }
    return {"family":family,"development_world_seed":world_seed,
            "model_seed":model_seed,
            "integrity":"PASS" if all(checks.values()) else "FAIL",
            "checks":checks,"arms":records}


def run()->dict:
    guard_dev()
    torch.set_num_threads(1)
    began=time.perf_counter()
    fit=data(FIT_SEED,FIT_EPISODES)
    validation=data(VALIDATION_SEED,VALIDATION_EPISODES)
    records=[]
    for init in INIT_SEEDS:
        # Unchanged actual P11 conventional control, same original source
        # and optimizer; its validation plateau may fail and is recorded.
        gru,trained=train_arm(ConventionalCausalGRU,init,fit,validation)
        cases=[evaluate(family,seed,init,gru)
               for family in FAMILIES for seed in DEV_SEEDS]
        integrity=(
            trained["actual_optimizer_updates"]==UPDATES and
            trained["gradient_changed_weights"] and len(cases)==16 and
            all(q["integrity"]=="PASS" for q in cases))
        aggregated={family:{
            a:sum(q["arms"][a]["mean_factual_return"] for q in cases
                  if q["family"]==family)/len(DEV_SEEDS)
            for a in ARMS} for family in FAMILIES}
        hypotheses={}
        for family,vals in aggregated.items():
            sham=family=="sham_no_action"
            sham_coeff=([q["arms"]["active_probe"][
                "sham_max_abs_spurious_action_coeff"]
                for q in cases if q["family"]==family] if sham else [])
            hypotheses[family]={
                "active_gt_equal_probe_passive_plus_0_05":
                    vals["active_probe"]>=vals["passive_random_probe"]+.05,
                "active_gt_no_probe_plus_0_05":
                    vals["active_probe"]>=vals["no_probe"]+.05,
                "active_gt_misaligned_feedback_plus_0_05":
                    vals["active_probe"]>=vals["scrambled_feedback_lags"]+.05,
                "active_gt_adaptive_conventional_GRU_plus_0_05":
                    vals["active_probe"]>=vals["adaptive_gru"]+.05,
                "active_gt_seeded_random_plus_0_05":
                    vals["active_probe"]>=vals["random"]+.05,
                "sham_false_action_causality_below_0_12":
                    max(sham_coeff)<=.12 if sham else None}
        records.append({
            "trained_GRU_initialization":init,
            "integrity":"PASS" if integrity else "FAIL",
            "original_trained_GRU":trained,
            "aggregate_factual_action_return":aggregated,
            "frozen_directional_hypotheses":hypotheses,
            "all_development_episodes":cases})
    duration=time.perf_counter()-began
    counts={
        family:{k:(sum(int(q["frozen_directional_hypotheses"][family][k])
                       for q in records)
                   if records[0]["frozen_directional_hypotheses"][family][k]
                   is not None else None)
                for k in records[0]["frozen_directional_hypotheses"][family]}
        for family in FAMILIES}
    passed=(len(records)==3 and all(r["integrity"]=="PASS"
           for r in records) and duration<=MAX_PROCESS_SECONDS)
    return {"schema":SCHEMA,
            "status":"PASS_ACTIVE_FACTUAL_PROBING_DEV_NOT_SCIENCE"
                     if passed else "FAIL_ACTIVE_FACTUAL_PROBING_INTEGRITY",
            "scientific_first_use":"NOT_RUN",
            "scientific_seeds_consumed":[],
            "native_substrate_changed":False,
            "models_promoted":False,
            "original_training_updates":len(INIT_SEEDS)*UPDATES,
            "actual_scored_factual_interventions":
                len(INIT_SEEDS)*len(FAMILIES)*len(DEV_SEEDS)*
                len(ARMS)*DECISIONS,
            "probe_slots_per_matched_policy":PROBES_PER_EPISODE,
            "actual_online_adaptation_updates_each_learner_per_episode":64,
            "original_trained_GRU_plateau_independently_qualified":False,
            "distinct_training_and_evaluation_sources_authentic_science":False,
            "development_families":list(FAMILIES),
            "model_initializations":list(INIT_SEEDS),
            "world_seeds":list(DEV_SEEDS),
            "hypothesis_support_out_of_three":counts,
            "records":records,
            "wall_process_seconds":duration,
            "limitations":"Only authored known synthetic P12 worlds. Controller uses hand-coded 0..3 lag hypotheses and D-optimal design; not non-programmed causal discovery. Original P11 GRU receives 64 online AdamW updates but greater training FLOPs than 5x5 ridge. Different model initializations share the same procedural world RNG seeds, so all-three counts are not independent world replications. Time/energy budget partly measured, FLOPs not matched, no mature cold retention or original-world science."}


def main()->int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--receipt",type=Path,required=True)
    args=parser.parse_args()
    r=run()
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(r,sort_keys=True,indent=2)+"\n")
    print(json.dumps({
        "status":r["status"],
        "actual_scored_factual_interventions":r[
            "actual_scored_factual_interventions"],
        "support":r["hypothesis_support_out_of_three"],
        "per_seed_mean_rewards":[{"seed":x["trained_GRU_initialization"],
            "return_by_world":x["aggregate_factual_action_return"]}
            for x in r["records"]],
        "process_seconds":r["wall_process_seconds"],
        "science_first_use":r["scientific_first_use"]
        },sort_keys=True))
    return 0 if r["status"]=="PASS_ACTIVE_FACTUAL_PROBING_DEV_NOT_SCIENCE" else 1


if __name__=="__main__":
    raise SystemExit(main())
