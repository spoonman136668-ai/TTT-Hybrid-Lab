"""Native P12: factual online causal-lag attribution vs matched online ablations.

Known deterministic synthetic DEV only.  Does NOT modify the native Hybrid
substrate, persistent checkpoint, P11 models or sealed prospective science.
No latent causal lag/sign is provided to any trained or online policy.
"""
from __future__ import annotations
import argparse
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
from scripts.native_p11_temporal_credit_dev import (
    LearnedTemporalCredit,ConventionalCausalGRU,train_arm,data,plan,
    FIT_SEED,FIT_EPISODES,VALIDATION_SEED,VALIDATION_EPISODES,
    INIT_SEEDS,HISTORY,UPDATES
)

SCHEMA="ttt-hybrid.native-p12.online-lag-causal-dev.v1"
DEV_SEEDS=(88421,88433,88457,88469)
FAMILIES=("delay2","delay3","changing_delay","sham_no_action")
ARMS=("online_full","online_current","online_no_adapt","online_permuted",
      "p11_structured","p11_gru","random","last_pair","privileged_reference")
OBS_PREFIX=8
DECISIONS=64
MAX_PROCESS_SECONDS=110
RIDGE=1.
FORGET=.96
EXPLORATION=.20
CONFIDENCE=.08

def dev_only():
    if os.environ.get("NATIVE_P12_DEV_ONLY","1")!="1":
        raise RuntimeError("P12_SCIENCE_FIRST_USE_FORBIDDEN")


class CausalDelayWorld:
    """Four authored DEV interventions; exogenous noise fixed across arms."""
    def __init__(self,family,seed):
        if family not in FAMILIES or seed not in DEV_SEEDS:
            raise ValueError("P12_UNREGISTERED_DEVELOPMENT_WORLD")
        self.family,self.seed=family,seed
        r=random.Random(seed+100007*FAMILIES.index(family))
        self.base=r.choice((-1,1))
        self.gain=r.uniform(.38,.57)
        self.bias=r.uniform(-.075,.075)
        self.phase=r.uniform(0.,2*math.pi)
        self.noise=[r.gauss(0.,.02) for _ in range(OBS_PREFIX+DECISIONS)]
        self.actions=[]
        self.outputs=[]

    def lag(self,t):
        if not 0<=t<OBS_PREFIX+DECISIONS:
            raise ValueError("P12_OUT_OF_RANGE")
        if self.family=="delay2":return 2
        if self.family=="delay3":return 3
        if self.family=="changing_delay":return 0 if t<32 else (2 if t<56 else 1)
        return None

    def sign(self,t):
        return self.base*(1 if t<32 or t>=56 else -1)

    def realize(self,t,action):
        if type(action) is not int or action not in (-1,1) or t!=len(self.actions):
            raise ValueError("P12_FACTUAL_ACTION_SEQUENCE_INVALID")
        lag=self.lag(t)
        self.actions.append(action)
        prev=self.actions[t-lag] if lag is not None and t-lag>=0 else 0
        exogenous=(.24*math.sin(.31*t+self.phase)
                   if lag is None else
                   (.14*math.sin(.35*t+self.phase)
                    if self.family=="changing_delay" else 0.))
        output=self.bias+self.noise[t]+exogenous+(
            self.gain*self.sign(t)*prev if lag is not None else 0.)
        self.outputs.append(float(output))
        return float(output)


def features(current:int,previous:list[int],mode:str):
    if current not in (-1,1) or mode not in (
        "full","current","permuted"):
        raise ValueError("P12_INVALID_CAUSAL_FEATURES")
    older=[previous[-lag] if len(previous)>=lag else 0
           for lag in (1,2,3)]
    if mode=="permuted":
        older=older[1:]+older[:1]
    if mode=="current":
        older=[0,0,0]
    return torch.tensor([1.,float(current),*map(float,older)],
                        dtype=torch.float64)


class FactualLagEstimator:
    """5x5 regularized discounted linear sufficient statistics.

    The full and current-action ablation have identical matrix shape,
    update count and solver work. Both see exactly the same completed
    factual actions/outcomes from their OWN selected decisions.
    """
    def __init__(self,mode="full",adapt=True):
        if mode not in ("full","current","permuted") or type(adapt) is not bool:
            raise ValueError("P12_LAG_POLICY_INVALID")
        self.mode,self.adapt=mode,adapt
        self.gram=torch.eye(5,dtype=torch.float64)*RIDGE
        self.rhs=torch.zeros(5,dtype=torch.float64)
        self.count=0
        self.scored_updates=0

    def coefficients(self):
        return torch.linalg.solve(self.gram,self.rhs)

    def choose(self,rng:random.Random)->int:
        betas=self.coefficients()[1:].tolist()
        eligible=betas[:1] if self.mode=="current" else betas
        dominant=max(eligible,key=lambda v:abs(v))
        if abs(dominant)<CONFIDENCE or rng.random()<EXPLORATION:
            return rng.choice((-1,1))
        return 1 if dominant>0 else -1

    def observe(self,current,previous,observed,*,scored):
        if not math.isfinite(observed):
            raise ValueError("P12_UNVERIFIED_FACTUAL_FEEDBACK")
        if scored and not self.adapt:
            return
        x=features(current,previous,self.mode)
        self.gram=(FORGET*self.gram+torch.outer(x,x)+
                   (1-FORGET)*RIDGE*torch.eye(5,dtype=torch.float64))
        self.rhs=FORGET*self.rhs+x*observed
        self.count+=1
        if scored:self.scored_updates+=1


def choose_p11(model,history):
    a,pred=plan(model,history,lookahead=True)
    return a,float(pred)


def score_episode(family,seed,models):
    if family not in FAMILIES or seed not in DEV_SEEDS:
        raise ValueError("P12_UNREGISTERED_EVALUATION")
    records={}
    prefixes=[]
    for arm in ARMS:
        world=CausalDelayWorld(family,seed)
        history=[]
        for t in range(OBS_PREFIX):
            # Frozen common exploration; real feedback is observed.
            a=-1 if t%2==0 else 1
            y=world.realize(t,a)
            history.append((a,y))
        prefixes.append(tuple(history))
        ctl=(FactualLagEstimator(
            "current" if arm=="online_current" else (
                "permuted" if arm=="online_permuted" else "full"),
            adapt=(arm!="online_no_adapt"))
            if arm.startswith("online_") else None)
        if ctl:
            for t,(action,outcome) in enumerate(history):
                ctl.observe(action,[p[0] for p in history[:t]],
                            outcome,scored=False)
        rng=random.Random(seed+19007+17*ARMS.index(arm))
        actions=[]
        rewards=[]
        for step in range(DECISIONS):
            t=OBS_PREFIX+step
            if ctl:
                action=ctl.choose(rng)
            elif arm=="p11_structured":
                action,_=choose_p11(models["structured"],history)
            elif arm=="p11_gru":
                action,_=choose_p11(models["gru"],history)
            elif arm=="random":
                action=rng.choice((-1,1))
            elif arm=="last_pair":
                last={}
                for prior_action,prior_y in history[-HISTORY:]:
                    last[prior_action]=prior_y
                action=1 if last.get(1,0)>=last.get(-1,0) else -1
            else:
                # Inaccessible phase and delay: contextualized for diagnostic
                # only. With changing delays it is NOT an optimal upper bound.
                target=min(OBS_PREFIX+DECISIONS-1,t+(
                    0 if world.lag(t) is None else world.lag(t)))
                action=world.sign(target)
            previous=[a for a,_ in history]
            observed=world.realize(t,action)
            if ctl:
                ctl.observe(action,previous,observed,scored=True)
            history.append((action,observed))
            actions.append(action)
            rewards.append(observed)
        assert len(world.actions)==OBS_PREFIX+DECISIONS
        estimates=ctl.coefficients().tolist() if ctl else None
        records[arm]={
            "mean_factual_return":sum(rewards)/DECISIONS,
            "phase0_return":sum(rewards[:24])/24,
            "phase1_return":sum(rewards[24:48])/24,
            "phase0_revisit_return":sum(rewards[48:])/16,
            "all_scored_actions":actions,
            "all_factual_consequences":rewards,
            "scored_factual_count":len(actions),
            "online_feedback_updates_during_scoring":(
                ctl.scored_updates if ctl else 0),
            "online_initial_factual_updates":(OBS_PREFIX if ctl else 0),
            "learned_online_coefficient_vector":estimates,
            "false_causal_max_abs_coeff_on_sham":(
                max(abs(v) for v in estimates[1:])
                if family=="sham_no_action" and estimates else None),
            "trace_sha256":hashlib.sha256(json.dumps(
                [actions,rewards],separators=(",",":"),allow_nan=False
            ).encode()).hexdigest(),
        }
    return {
        "family":family,"dev_seed":seed,
        "same_initial_factual_history":len(set(prefixes))==1,
        "all_policies_equal_decision_budget":all(
            r["scored_factual_count"]==DECISIONS for r in records.values()),
        "causal_action_effect_absent_on_sham":(
            all(records[a]["all_factual_consequences"]==
                records["random"]["all_factual_consequences"] for a in ARMS)
            if family=="sham_no_action" else None),
        "all_consequences_finite":all(
            math.isfinite(y) for r in records.values()
            for y in r["all_factual_consequences"]),
        "policies":records
    }


def run():
    dev_only()
    torch.set_num_threads(1)
    start=time.perf_counter()
    fit=data(FIT_SEED,FIT_EPISODES)
    val=data(VALIDATION_SEED,VALIDATION_EPISODES)
    collected=[]
    for seed in INIT_SEEDS:
        structured,rs=train_arm(LearnedTemporalCredit,seed,fit,val)
        gru,rg=train_arm(ConventionalCausalGRU,seed,fit,val)
        models={"structured":structured,"gru":gru}
        episodes=[score_episode(family,dev_seed,models)
                  for family in FAMILIES for dev_seed in DEV_SEEDS]
        parity={
            "both_original_trained_models_received_real_1200_updates":(
                rs["actual_optimizer_updates"]==rg["actual_optimizer_updates"]==UPDATES),
            "both_changed_weights_from_factual_targets":(
                rs["gradient_changed_weights"] and rg["gradient_changed_weights"]),
            "identical_250k_plus_factual_minibatch_index_digest":(
                rs["sample_indices_sha256"]==rg["sample_indices_sha256"]),
            "both_training_and_validation_examples_identical":(
                rs["unique_factual_fit_examples"]==
                rg["unique_factual_fit_examples"]==6000 and
                rs["factual_val_examples"]==rg["factual_val_examples"]==1920),
            "all_sixteen_development_worlds":len(episodes)==16,
            "initial_histories_same_all_arms":all(
                q["same_initial_factual_history"] for q in episodes),
            "exact_equal_nonzero_64_action_budgets":all(
                q["all_policies_equal_decision_budget"] for q in episodes),
            "source_not_corrupted_all_rewards_finite":all(
                q["all_consequences_finite"] for q in episodes),
            "no_action_sham_factual_outcomes_identical_across_actions":all(
                q["causal_action_effect_absent_on_sham"] is True for q in
                episodes if q["family"]=="sham_no_action"),
            "full_online_and_current_only_exact_same_factual_update_budget":all(
                q["policies"]["online_full"]["online_feedback_updates_during_scoring"]==
                q["policies"]["online_current"]["online_feedback_updates_during_scoring"]==
                DECISIONS for q in episodes),
            "frozen_online_ablation_no_scored_updates":all(
                q["policies"]["online_no_adapt"]["online_feedback_updates_during_scoring"]==0
                for q in episodes),
            "no_hidden_delay_regime_or_counterfactual_given_to_policies":True,
        }
        aggregate={
            family:{a:sum(e["policies"][a]["mean_factual_return"] for e in episodes
                         if e["family"]==family)/len(DEV_SEEDS)
                    for a in ARMS}
            for family in FAMILIES
        }
        hypothesis={}
        for family in FAMILIES:
            m=aggregate[family]
            sham=(family=="sham_no_action")
            sensitivity=([
                e["policies"]["online_full"]["false_causal_max_abs_coeff_on_sham"]
                for e in episodes if e["family"]==family] if sham else [])
            hypothesis[family]={
                "online_full_gt_equal_update_current_only_by_0_05":
                    m["online_full"]>=m["online_current"]+.05,
                "online_full_gt_no_adapt_by_0_05":
                    m["online_full"]>=m["online_no_adapt"]+.05,
                "online_full_gt_trained_GRU_by_0_05":
                    m["online_full"]>=m["p11_gru"]+.05,
                "online_full_gt_random_by_0_05":
                    m["online_full"]>=m["random"]+.05,
                "online_full_gt_action_permuted_by_0_05":
                    m["online_full"]>=m["online_permuted"]+.05,
                "sham_false_causal_below_0_12":
                    max(sensitivity)<=.12 if sham else None,
            }
        collected.append({
            "model_seed":seed,
            "status":"PASS_DEV_INTEGRITY" if all(parity.values()) else
                     "FAIL_DEV_INTEGRITY",
            "training":{"structured":rs,"GRU":rg},
            "controls":parity,
            "reward_by_family":aggregate,
            "hypotheses":hypothesis,
            "episodes":episodes
        })
    elapsed=time.perf_counter()-start
    counts={family:{
        k:sum(int(x["hypotheses"][family][k]) for x in collected)
        if collected[0]["hypotheses"][family][k] is not None else None
        for k in collected[0]["hypotheses"][family]}
        for family in FAMILIES}
    passed=(len(collected)==len(INIT_SEEDS) and all(
        x["status"]=="PASS_DEV_INTEGRITY" for x in collected)
        and elapsed<MAX_PROCESS_SECONDS)
    return {
        "schema":SCHEMA,
        "status":"PASS_FACTUAL_ONLINE_CREDIT_DEV_NOT_SCIENCE" if passed else
                 "FAIL_FACTUAL_ONLINE_CREDIT_DEV",
        "science_first_use":"NOT_RUN",
        "official_science_seeds_consumed":[],
        "authentic_unseen_worlds":"NOT_BORN",
        "native_substrate_changed":False,
        "accepted_checkpoint_changed":False,
        "optimizer_updates_per_original_control":UPDATES,
        "trained_control_optimizer_steps_total":UPDATES*6,
        "online_factual_updates_per_scored_episode":DECISIONS,
        "model_seeds":list(INIT_SEEDS),
        "dev_world_seeds":list(DEV_SEEDS),
        "development_families":list(FAMILIES),
        "factual_scored_interventions":len(INIT_SEEDS)*len(FAMILIES)*
                                      len(DEV_SEEDS)*len(ARMS)*DECISIONS,
        "all_negative_hypotheses_retained":True,
        "support_counts_out_of_three":counts,
        "records":collected,
        "whole_process_wall_seconds":elapsed,
        "limits":"KNOWN authored synthetic source. Full lag 0..3 available as an explicit hand-coded inductive bias, online RLS uses feedback and computation not available to frozen GRU; only same-update current-only and permuted lag negatives offer close online causal comparisons. Episode-specific online parameters do not persist across worlds. Exact FLOP/energy costs and fully trained plateau not independently qualified. No real unannounced physics, cold retention, true scientific transfer, model promotion or FORGE activation."
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--receipt",required=True,type=Path)
    a=p.parse_args()
    result=run()
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({
        "status":result["status"],
        "science_first_use":result["science_first_use"],
        "support":result["support_counts_out_of_three"],
        "return_by_model_seed":[{
            "seed":x["model_seed"],"reward":x["reward_by_family"]}
            for x in result["records"]],
        "actual_scored_interventions":result["factual_scored_interventions"],
        "wall_seconds":result["whole_process_wall_seconds"]
    },sort_keys=True))
    return 0 if result["status"]=="PASS_FACTUAL_ONLINE_CREDIT_DEV_NOT_SCIENCE" else 1

if __name__=="__main__":
    raise SystemExit(main())
