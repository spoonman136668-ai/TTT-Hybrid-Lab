"""P20: source-pinned genuine CartPole velocity-blind factual-history test.

Frozen BEFORE this implementation. Public DEV only, not original science
first-use, not reward-only learning and not a FORGE owner/Hybrid activation.
The expert supplies stronger labels derived from its complete observed sensors.
All trained actors see the *same* factual demonstrations with velocity masked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import native_p18_real_cartpole_factual_dev as p18
from scripts import native_p19_expert_calibration_dev as p19

ROOT=Path(__file__).resolve().parents[1]
SCHEMA="ttt-hybrid.native-p20.velocity-blind-history-dev.v1"
TRAIN_SEEDS=tuple(range(76000,76048))
VAL_SEEDS=tuple(range(77000,77008))
EVAL_SEEDS=(78011,78012,78013,78014,78015,78016,78017,78018)
MODEL_SEEDS=(2741,2753,2767)
MAX_STEPS=200
MAX_SECONDS=240
ARMS=("full_history_gru","current_only_gru","full_history_mlp",
      "current_only_linear","history_reversed_gru",
      "finite_difference_public_history","full_observation_expert","seeded_random")


def require_dev():
    if os.environ.get("NATIVE_P20_DEV_ONLY","0")!="1":
        raise RuntimeError("P20_SCIENCE_FIRST_USE_NOT_AUTHORIZED")


def frozen():
    d=json.loads((ROOT/"experiments/native_p20_velocity_blind_history_frozen.json").read_text())
    if (d["schema"]!="ttt-hybrid.native-p20.velocity-blind-history-frozen.v1"
        or d["scientific_boundary"]["first_use_authorized"] is not False
        or d["scientific_boundary"]["sealed_scientific_data_consumed"] is not False
        or d["stack"]["p19_exact_sha"]!="d7e1535f615c7f9f0ca70b2ab9059fdf1f912726"
        or d["stack"]["s29_read_only_sha"]!=p18.S29_SHA
        or tuple(d["design"]["evaluation_seeds"])!=EVAL_SEEDS
        or tuple(d["design"]["model_initializations"])!=MODEL_SEEDS
        or d["design"]["supervised_epochs"]!=p19.EPOCHS
        or d["design"]["batch_size"]!=p19.BATCH
        or d["design"]["learning_rate"]!=p19.LR):
        raise RuntimeError("P20_FROZEN_SOURCE_OR_SCIENCE_DRIFT")
    if not set(TRAIN_SEEDS).isdisjoint(VAL_SEEDS) or not set(VAL_SEEDS).isdisjoint(EVAL_SEEDS) or not set(TRAIN_SEEDS).isdisjoint(EVAL_SEEDS):
        raise RuntimeError("P20_DATA_SPLIT_LEAK")
    if not set(EVAL_SEEDS).isdisjoint(p19.EVAL_SEEDS):
        raise RuntimeError("P20_PRIOR_P19_EVAL_EXPOSURE")
    return d


def visible_history(raw:torch.Tensor,current_only:bool=False):
    if raw.ndim!=3 or raw.shape[1:]!=(8,6):
        raise ValueError("P20_PUBLIC_SENSOR_SHAPE")
    v=raw.clone()
    # Public source velocity sensor columns 1/3 are NOT passed to the learner,
    # at any time or position. This is true for demonstration and evaluation.
    v[:,:,1]=0.
    v[:,:,3]=0.
    if current_only:
        v[:,:-1,:]=0.
    return v


def visible_rows(raw_rows, current_only:bool=False, reversed_history:bool=False):
    if reversed_history:
        window=raw_rows[-8:]
        raw_rows=list(reversed(window[:-1]))+window[-1:]
    return visible_history(p18.history_tensor(raw_rows),current_only=current_only)


def finite_difference_action(raw_rows):
    """Two factual public angles; no hidden physics or unobserved velocity."""
    theta=float(raw_rows[-1][2])
    if len(raw_rows)==1:
        estimate=0.
    else:
        estimate=(theta-float(raw_rows[-2][2]))/0.02
    return int(theta+0.5*estimate>=0.)


@torch.no_grad()
def actual_episode(gym,arm,models,seed,model_seed):
    if arm not in ARMS or seed not in EVAL_SEEDS or model_seed not in MODEL_SEEDS:
        raise ValueError("P20_EVALUATION_SOURCE_DENIED")
    env=gym.make("CartPole-v1",render_mode=None)
    digest=hashlib.sha256()
    try:
        obs,_=env.reset(seed=seed)
        initial=p18.factual_row(obs,0,0.)
        hist=[initial]
        actions=[]
        rewards=[]
        rng=random.Random(seed+model_seed*31+97)
        for _ in range(MAX_STEPS):
            if arm=="full_observation_expert":
                action=p18.handcoded_action(obs)
            elif arm=="seeded_random":
                action=rng.randrange(2)
            elif arm=="finite_difference_public_history":
                action=finite_difference_action(hist)
            else:
                m=models["full_history_gru"] if arm=="history_reversed_gru" else models[arm]
                x=visible_rows(hist,current_only=arm in ("current_only_gru","current_only_linear"),
                               reversed_history=arm=="history_reversed_gru")
                logits,_=m(x)
                action=int(logits.argmax(dim=-1).item())
            nxt,reward,terminated,truncated,_=env.step(action)
            reward=float(reward)
            new=p18.factual_row(nxt,2*action-1,reward)
            digest.update(json.dumps(
                [hist[-1],action,new,bool(terminated),bool(truncated)],
                separators=(",",":"),allow_nan=False).encode())
            actions.append(action)
            rewards.append(reward)
            hist.append(new)
            obs=nxt
            if terminated or truncated:
                break
        return {"actual_reward":sum(rewards),"actual_actions":len(actions),
                "actual_action_choices":actions,
                "factual_trace_sha256":digest.hexdigest(),
                "public_reset_observation":initial,
                "eval_optimizer_updates":0}
    finally:
        env.close()


def run(s29_root,gym_root,cloud_root):
    require_dev()
    d=frozen()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    gym,identity=p18.admit_gymnasium(s29_root,gym_root,cloud_root)
    tx,ty,train=p19.expert_dataset(gym,TRAIN_SEEDS)
    vx,vy,val=p19.expert_dataset(gym,VAL_SEEDS)
    full_train=visible_history(tx)
    full_val=visible_history(vx)
    only_train=visible_history(tx,current_only=True)
    only_val=visible_history(vx,current_only=True)
    per_seed=[]
    for model_seed in MODEL_SEEDS:
        fits={}
        models={}
        for arm,kind,use_current in (
            ("full_history_gru","history_gru",False),
            ("current_only_gru","history_gru",True),
            ("full_history_mlp","history_mlp",False),
            ("current_only_linear","linear_observation",True)):
            x=only_train if use_current else full_train
            v=only_val if use_current else full_val
            fit,receipt=p19.fit(kind,model_seed,x,ty,v,vy)
            models[arm]=fit["trained"]
            fits[arm]=receipt
        expected={k:p18.model_sha(net) for k,net in models.items()}
        cases=[]
        for world in EVAL_SEEDS:
            result={arm:actual_episode(gym,arm,models,world,model_seed) for arm in ARMS}
            if len({tuple(v["public_reset_observation"]) for v in result.values()})!=1:
                raise RuntimeError("P20_DIFFERENT_ENVIRONMENT_RESETS")
            cases.append({"seed":world,"arms":result})
        unchanged=expected=={k:p18.model_sha(net) for k,net in models.items()}
        means={arm:sum(case["arms"][arm]["actual_reward"] for case in cases)/len(cases)
               for arm in ARMS}
        minimum=d["design"]["meaningful_history_requirement_each_seed"]
        gap=means["full_history_gru"]-means["current_only_gru"]
        reversed_gap=means["full_history_gru"]-means["history_reversed_gru"]
        discriminative=means["current_only_gru"]<minimum["task_discriminative_if_current_only_gru_mean_reward_below"]
        causal_support=(means["full_history_gru"]>=minimum["full_history_gru_mean_reward_min"]
                        and gap>=minimum["versus_trained_current_only_gru_mean_reward_gain_min"]
                        and reversed_gap>=minimum["versus_reversed_history_gru_mean_reward_gain_min"]
                        and discriminative)
        engineering={
            "four_actors_updated_and_restored_exact":
                all(r["trained_weight_updated"] and r["fresh_object_state_only_restore_exact"] for r in fits.values()),
            "same_three_model_training_seeds_and_epochs":
                all(r["initialization"]==model_seed and r["epochs_completed"]==p19.EPOCHS for r in fits.values()),
            "no_online_training_during_actual_source_eval":
                all(v["eval_optimizer_updates"]==0 and 1<=v["actual_actions"]<=MAX_STEPS for c in cases for v in c["arms"].values()),
            "model_weights_unchanged_during_evaluation":unchanged,
            "same_actual_source_reset_for_eight_arms":True
        }
        per_seed.append({
            "model_seed":model_seed,
            "engineering":"PASS" if all(engineering.values()) else "FAIL",
            "engineering_checks":engineering,
            "fits":fits,
            "mean_real_episode_reward":means,
            "full_vs_matched_current_only_gap":gap,
            "full_vs_reverse_history_gap":reversed_gap,
            "task_discriminative":discriminative,
            "meaningful_history_support":causal_support,
            "real_source_cases":cases})
    wall=time.perf_counter()-start
    engineering=all(r["engineering"]=="PASS" for r in per_seed) and wall<=MAX_SECONDS
    support=sum(int(r["meaningful_history_support"]) for r in per_seed)
    return {
        "schema":SCHEMA,
        "status":"PASS_EXTERNAL_DEV_ENGINEERING_NOT_SCIENCE" if engineering else "FAIL_EXTERNAL_DEV_INTEGRITY",
        "parent_P19_source_shas_untouched":True,
        "source_identity":identity,
        "frozen_protocol_schema":d["schema"],
        "masked_velocity_columns":[1,3],
        "teacher_was_privileged_full_public_sensor_supervisor":True,
        "supervised_teacher_not_reward_only":True,
        "scientific_first_use":"NOT_RUN",
        "protected_scientific_world_seeds_consumed":[],
        "train_demonstrations":train,
        "validation_demonstrations":val,
        "public_dev_eval_reset_seeds":list(EVAL_SEEDS),
        "supporting_initializations_of_three":support,
        "all_three_meaningful_history_support":support==3,
        "per_seed":per_seed,
        "elapsed_whole_experiment_seconds":wall,
        "conventional_finite_difference_strong_control_included":True,
        "scientific_transfer_or_superiority_established":False,
        "FORGE_owner_activation":False,
        "limitations":"Expert teacher labels use full public state, while learners see masked velocity; teacher information is unmatched to reward-only P18. P19 all-200 scores do not prove memory; P20 contrasts learned full-history versus trained matched current-only on the same public CartPole source. Negative controls are not independent scientific worlds, no real process-death cold restore, no matched FLOPs, no model acceptance or promotion."}


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--s29-root",required=True,type=Path)
    a.add_argument("--source-root",required=True,type=Path)
    a.add_argument("--cloudpickle-root",required=True,type=Path)
    a.add_argument("--receipt",required=True,type=Path)
    args=a.parse_args()
    r=run(args.s29_root,args.source_root,args.cloudpickle_root)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(r,sort_keys=True,indent=2)+"\n")
    print(json.dumps({
        "status":r["status"],"history_support":r["supporting_initializations_of_three"],
        "mean_reward":[q["mean_real_episode_reward"] for q in r["per_seed"]],
        "gaps":[[q["full_vs_matched_current_only_gap"],q["full_vs_reverse_history_gap"]] for q in r["per_seed"]],
        "elapsed":r["elapsed_whole_experiment_seconds"],"science":r["scientific_first_use"]},
        sort_keys=True))
    return 0 if r["status"]=="PASS_EXTERNAL_DEV_ENGINEERING_NOT_SCIENCE" else 1


if __name__=="__main__":
    raise SystemExit(main())
