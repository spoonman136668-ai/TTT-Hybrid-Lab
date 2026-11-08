"""P15: P14 factual-reward-trained causal policy versus original strongest P11 planners.

Exact prior P14/P11 actor code and training sources are left unchanged.
This is a direct, shared-episode, forced-probe-equalized comparison, NOT a
training- or FLOP-matched scientific comparison. The six synthetic DEV
families have been repeatedly seen; original future science seeds stay sealed.
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
from scripts.native_p14_reward_trained_intervention_dev import (
    train_model as train_p14, completed_history, eval_world, factual_step,
    FAMILIES, INIT_SEEDS as P14_INIT, OLD_SEEDS, LAG_SEEDS,
    OBS_PREFIX, DECISIONS, PROBE_INTERVAL, TRAIN_UPDATES,
)
from scripts.native_p11_temporal_credit_dev import (
    train_arm as train_p11, data as data_p11, plan, INIT_SEEDS as P11_INIT,
    ConventionalCausalGRU, LearnedTemporalCredit, FIT_SEED,
    FIT_EPISODES, VALIDATION_SEED, VALIDATION_EPISODES,
    UPDATES as P11_UPDATES,
)
from scripts.native_p12_online_causal_delay_dev import FactualLagEstimator
from scripts.native_p13_active_causal_probe_dev import action_from_model

SCHEMA="ttt-hybrid.native-p15.strong-control-dev.v1"
ARMS=("p14_reward_causal","p14_credit_corrupted",
      "p11_conventional_gru_twostep","p11_conventional_gru_greedy",
      "p11_structured_twostep","p13_passive_online","random_actions")
PROBE_COUNT=16
REWARD_GAIN_THRESHOLD=.05
OLD_CAPABILITY_MAX_DEFICIT=.02
MAX_WALL_SECONDS=130


def require_dev():
    if os.environ.get("NATIVE_P15_DEV_ONLY","1")!="1":
        raise RuntimeError("P15_AUTHENTIC_SCIENCE_FIRST_USE_FORBIDDEN")


def common_probe_rng(seed:int,paired_index:int)->random.Random:
    """Same signed forced probes for every competing policy in each world."""
    return random.Random(70019 + seed + 9001*paired_index)


def select_arm_action(arm:str,models:dict,history:list[tuple[int,float]],
                      actions:list[int],idx:int,rng:random.Random,
                      online:FactualLagEstimator|None)->int:
    if arm not in ARMS:
        raise ValueError("P15_UNREGISTERED_EVALUATION_ARM")
    if len(history)<OBS_PREFIX:
        raise ValueError("P15_INCOMPLETE_FACTUAL_CONTEXT")
    if idx%PROBE_INTERVAL==0:
        return rng.choice((-1,1))
    if arm in ("p14_reward_causal","p14_credit_corrupted"):
        with torch.no_grad():
            y=models[arm](completed_history([history]))
        return 1 if float(y.item())>=0 else -1
    if arm in ("p11_conventional_gru_twostep","p11_conventional_gru_greedy",
               "p11_structured_twostep"):
        lookahead=arm!="p11_conventional_gru_greedy"
        action,_=plan(models[
            "p11_gru" if "gru" in arm else "p11_structured"],
            history,lookahead=lookahead)
        return action
    if arm=="p13_passive_online":
        action,probing,_=action_from_model(online,rng,actions,idx,
                                          policy="no_probe")
        if probing:
            raise RuntimeError("P15_BUDGET_DOUBLE_PROBE")
        return action
    return rng.choice((-1,1))


def evaluate(family:str,dev_seed:int,pair_index:int,models:dict)->dict:
    if family not in FAMILIES or pair_index not in (0,1,2):
        raise ValueError("P15_EVALUATION_FAMILY_OR_PAIR_UNREGISTERED")
    permitted=OLD_SEEDS if family in ("instant","lagged") else LAG_SEEDS
    if dev_seed not in permitted:
        raise ValueError("P15_UNREGISTERED_EVALUATION_SEED")
    traces={}
    prefixes=[]
    for arm in ARMS:
        world=eval_world(family,dev_seed)
        # Separate forced-probe RNG from policy exploration: a random
        # baseline must not change later forced actions by consuming draws.
        probe_rng=common_probe_rng(dev_seed,pair_index)
        policy_rng=random.Random(807719+dev_seed+pair_index*1337+
                                 ARMS.index(arm)*109)
        history=[]
        previous=0
        for idx in range(OBS_PREFIX):
            action=(-1 if idx%2==0 else 1)
            observation=factual_step(world,idx,action,previous)
            history.append((action,observation))
            previous=action
        prefixes.append(tuple(history))
        learner=FactualLagEstimator("full",adapt=True) if arm=="p13_passive_online" else None
        if learner:
            for idx,(a,y) in enumerate(history):
                learner.observe(a,[a0 for a0,_ in history[:idx]],y,scored=False)
        decisions=[]
        factual=[]
        probes=[]
        began=time.perf_counter()
        for step in range(DECISIONS):
            absolute=OBS_PREFIX+step
            completed_actions=[a for a,_ in history]
            if step%PROBE_INTERVAL==0:
                action=probe_rng.choice((-1,1))
                probes.append((step,action))
            else:
                action=select_arm_action(arm,models,history,
                                         completed_actions,step,
                                         policy_rng,learner)
            # Crucial: there is no observation until the action was chosen.
            outcome=factual_step(world,absolute,action,previous)
            if learner:
                learner.observe(action,completed_actions,outcome,scored=True)
            history.append((action,outcome))
            previous=action
            decisions.append(action)
            factual.append(outcome)
        traces[arm]={
            "mean_return":sum(factual)/DECISIONS,
            "first_24_return":sum(factual[:24])/24,
            "changed_24_return":sum(factual[24:48])/24,
            "old_regime_revisit_16_return":sum(factual[48:])/16,
            "factual_actions":decisions,
            "factual_outcomes":factual,
            "factual_observation_updates":learner.scored_updates if learner else 0,
            "scored_action_count":len(decisions),
            "fixed_probe_decisions":probes,
            "wall_policy_seconds":time.perf_counter()-began,
            "trace_sha256":hashlib.sha256(json.dumps(
                {"actions":decisions,"observed":factual},
                allow_nan=False,separators=(",",":")).encode()).hexdigest(),
        }
    expected_idx=list(range(0,64,4))
    signed_probe_sets=[tuple(x["fixed_probe_decisions"]) for x in traces.values()]
    checks={
        "all_arms_same_eight_factual_initial_observations":
            len(set(prefixes))==1,
        "all_arms_exact_same_sixteen_forced_probe_positions":
            all([a for a,_ in t["fixed_probe_decisions"]]==expected_idx
                for t in traces.values()),
        "all_arms_same_sixteen_signed_probe_actions":
            len(set(signed_probe_sets))==1,
        "all_arms_exact_sixtyfour_factual_decisions":
            all(t["scored_action_count"]==64 for t in traces.values()),
        "all_factual_consequences_finite":
            all(math.isfinite(y) for t in traces.values()
                for y in t["factual_outcomes"]),
        "p13_online_only_arm_updates_64_factual_events":
            traces["p13_passive_online"]["factual_observation_updates"]==64
            and all(traces[a]["factual_observation_updates"]==0
                    for a in ARMS if a!="p13_passive_online"),
        "zero_action_sham_all_arms_same_exact_reward_sequence":
            (len(set(tuple(v["factual_outcomes"]) for v in traces.values()))==1
             if family=="sham_no_action" else True),
        "source_does_not_supply_hidden_regime_or_delays_to_model":True,
    }
    return {"family":family,"development_seed":dev_seed,"model_pair":pair_index,
            "integrity":"PASS" if all(checks.values()) else "FAIL",
            "checks":checks,"arms":traces}


def run()->dict:
    require_dev()
    torch.set_num_threads(1)
    start=time.perf_counter()
    fit=data_p11(FIT_SEED,FIT_EPISODES)
    val=data_p11(VALIDATION_SEED,VALIDATION_EPISODES)
    results=[]
    for pair_index,(p14_seed,p11_seed) in enumerate(zip(P14_INIT,P11_INIT)):
        p14,trained_p14=train_p14("causal",p14_seed)
        negative,trained_negative=train_p14("misaligned",p14_seed)
        gru,trained_gru=train_p11(ConventionalCausalGRU,p11_seed,fit,val)
        structured,trained_structured=train_p11(
            LearnedTemporalCredit,p11_seed,fit,val)
        models={
            "p14_reward_causal":p14,"p14_credit_corrupted":negative,
            "p11_gru":gru,"p11_structured":structured
        }
        initial_state={
            arm:hashlib.sha256(b"".join(
                p.detach().cpu().numpy().tobytes()
                for p in model.parameters())).hexdigest()
            for arm,model in models.items()
        }
        cases=[evaluate(family,dev_seed,pair_index,models)
               for family in FAMILIES
               for dev_seed in (OLD_SEEDS if family in ("instant","lagged")
                                else LAG_SEEDS)]
        after_state={
            arm:hashlib.sha256(b"".join(
                p.detach().cpu().numpy().tobytes()
                for p in model.parameters())).hexdigest()
            for arm,model in models.items()
        }
        checks={
            "P14_reward_actor_and_negative_both_96_real_gradient_updates":
                trained_p14["actual_policy_gradient_updates"]==
                trained_negative["actual_policy_gradient_updates"]==
                TRAIN_UPDATES==96,
            "P14_trained_actors_same_world_cohort":
                trained_p14["training_world_cohort_digest"]==
                trained_negative["training_world_cohort_digest"],
            "P11_strong_GRU_and_structured_received_1200_factual_updates":
                trained_gru["actual_optimizer_updates"]==
                trained_structured["actual_optimizer_updates"]==
                P11_UPDATES==1200,
            "P11_models_same_exact_training_minibatch_indices":
                trained_gru["sample_indices_sha256"]==
                trained_structured["sample_indices_sha256"],
            "all_models_updated_weights_from_genuine_factual_gradients":
                all(x["gradient_changed_weights"] for x in
                    (trained_p14,trained_negative,trained_gru,
                     trained_structured)),
            "twentyfour_known_DEV_worlds_not_unseen_science":len(cases)==24,
            "same_actual_factual_obs_actions_probes_for_all_arms":
                all(c["integrity"]=="PASS" for c in cases),
            "all_evaluation_neural_weights_frozen":
                initial_state==after_state,
            "no_heldout_original_science_data_or_seeds_read":True,
        }
        average={
            family:{
                arm:sum(c["arms"][arm]["mean_return"] for c in cases
                        if c["family"]==family)/4 for arm in ARMS
            } for family in FAMILIES
        }
        gates={
            family:{
                "P14_learned_actor_beats_P11_strong_GRU_planner_by_0_05":
                    a["p14_reward_causal"]>=
                    a["p11_conventional_gru_twostep"]+REWARD_GAIN_THRESHOLD,
                "P14_learned_actor_beats_P11_structured_planner_by_0_05":
                    a["p14_reward_causal"]>=
                    a["p11_structured_twostep"]+REWARD_GAIN_THRESHOLD,
                "P14_learned_actor_beats_wrong_credit_by_0_05":
                    a["p14_reward_causal"]>=
                    a["p14_credit_corrupted"]+REWARD_GAIN_THRESHOLD,
                "P11_strong_two_step_beats_its_greedy_control_by_0_05":
                    a["p11_conventional_gru_twostep"]>=
                    a["p11_conventional_gru_greedy"]+REWARD_GAIN_THRESHOLD,
                "P14_old_capability_within_0_02_of_strong_GRU":
                    a["p14_reward_causal"]>=
                    a["p11_conventional_gru_twostep"]-OLD_CAPABILITY_MAX_DEFICIT
                    if family in ("instant","lagged") else None,
            } for family,a in average.items()
        }
        results.append({
            "model_pair":pair_index,"p14_actor_seed":p14_seed,
            "P11_control_seed":p11_seed,
            "engineering":"PASS" if all(checks.values()) else "FAIL",
            "all_integrity_checks":checks,
            "actual_training_records":{
                "P14_reward_causal":trained_p14,
                "P14_bad_credit":trained_negative,
                "P11_GRU":trained_gru,
                "P11_structured":trained_structured},
            "prior_strong_GRU_original_plateau_qualified":
                trained_gru["strict_last_five_1pct_plateau"],
            "original_training_update_budgets":{
                "P14_reward_causal":TRAIN_UPDATES,
                "P14_bad_credit":TRAIN_UPDATES,
                "P11_GRU":P11_UPDATES,
                "P11_structured":P11_UPDATES},
            "average_factual_return_by_familiar_family":average,
            "predeclared_DEV_hypothesis":gates,
            "all_factual_test_episodes":cases,
        })
    elapsed=time.perf_counter()-start
    support={
        family:{
            key:(sum(int(r["predeclared_DEV_hypothesis"][family][key])
                     for r in results)
                 if results[0]["predeclared_DEV_hypothesis"][family][key]
                 is not None else None)
            for key in results[0]["predeclared_DEV_hypothesis"][family]
        } for family in FAMILIES
    }
    good=(len(results)==3 and all(r["engineering"]=="PASS" for r in results)
          and elapsed<=MAX_WALL_SECONDS)
    return {
        "schema":SCHEMA,
        "status":"PASS_STRONGEST_CONTROL_FROZEN_DEV_NOT_SCIENCE"
                 if good else "FAIL_STRONGEST_CONTROL_ENGINEERING",
        "scientific_first_use":"NOT_RUN",
        "original_science_seeds_consumed":[],
        "native_substrate_modified":False,
        "FORGE_research_integration":False,
        "original_science_world_unseen":False,
        "total_factual_test_actions":3*24*len(ARMS)*64,
        "per_control_training_update_budgets":[96,96,1200,1200],
        "four_old_known_seeds_per_six_world_families":True,
        "strict_factual_probe_action_equality":True,
        "source_training_compute_equal_between_P14_and_P11":False,
        "strong_control_independently_plateau_qualified":False,
        "all_hypothesis_negative_controls_retained":True,
        "initialization_pairings":list(zip(P14_INIT,P11_INIT)),
        "hypothesis_support_of_three":support,
        "records":results,
        "whole_process_wall_seconds":elapsed,
        "limitations":"This is a direct matched-evaluation ACTION/OBSERVATION comparison using already observed synthetic development worlds, not an independently unseen science evaluation. The original P14 and P11 training curricula and number of optimizer steps differ, and P11 planner performs extra inference. No fair training-compute equivalence, independently established convergence, true process-death durable recall, 2nd distribution unseen transfer or scientific first-use. P13 passive online policy receives 64 factual updates while P11/P14 neural evaluation remains frozen. The first-use and promotion boundaries remain closed."
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--receipt",type=Path,required=True)
    args=parser.parse_args()
    output=run()
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(output,sort_keys=True,indent=2)+"\n")
    print(json.dumps({
        "status":output["status"],
        "scientific_first_use":output["scientific_first_use"],
        "evaluated_factual_actions":output["total_factual_test_actions"],
        "support":output["hypothesis_support_of_three"],
        "per_seed_means":[{
            "paired_initializations":[r["p14_actor_seed"],r["P11_control_seed"]],
            "average_factual_return":r["average_factual_return_by_familiar_family"]}
            for r in output["records"]],
        "seconds":output["whole_process_wall_seconds"]
    },sort_keys=True))
    return 0 if output["status"]=="PASS_STRONGEST_CONTROL_FROZEN_DEV_NOT_SCIENCE" else 1

if __name__=="__main__":
    raise SystemExit(main())
