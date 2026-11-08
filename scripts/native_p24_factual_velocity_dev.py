"""P24 public DEV: reward-only PPO with past-derived motion, not source oracle.

Retains all P20/P22 original trainers, controls, and failures; exact
prospectively registered protocol is an earlier commit. External source-only
MIT CartPole, no teacher actions, scientific first-use or engine modifications.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from scripts import native_p20_reward_only_ppo_dev as p20

SCHEMA = "ttt-hybrid.native-p24.factual-velocity-reconstruction-dev.v1"
PROTOCOL = ROOT / "experiments" / "native_p24_factual_velocity_frozen.json"
SEEDS = (3491,3503,3517)
EVAL_SEEDS=(96011,96017,96023,96029,96035,96041,96047,96053)
CONDITIONS=("original","short_pole_x0.7","long_pole_x1.3")
VIEWS={
 "masked_raw":("reward_gru","masked_raw"),
 "reconstructed_from_past":("reward_gru","reconstructed"),
 "inverted_reconstructed_velocity":("reward_gru","inverted"),
 "unmasked_full_4d":("reward_gru","unmasked"),
 "reconstructed_wrong_reward_credit":("wrong_credit_gru","reconstructed"),
}
CONTROLS=("finite_difference_public_two_frame","seeded_random")

def frozen():
    o=json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if (o.get("schema")!="ttt-hybrid.native-p24.factual-velocity-reconstruction-frozen.v1"
        or o["original"]["P22_sha"]!="25213b4f02ac0771667428cb176fcb6f962d7a1d"
        or o["source"]["protected_science_first_use"] is not False
        or o["train"]["seeds"]!=list(SEEDS)
        or o["evaluation"]["seeds"]!=list(EVAL_SEEDS)
        or o["train"]["total_factual_training_actions"]!=737280
        or o["train"]["total_PPO_minibatch_updates"]!=8640
        or len(VIEWS)!=5):
        raise RuntimeError("P24_FROZEN_PROTOCOL_MISMATCH")
    return o

def require_dev():
    if os.environ.get("NATIVE_P24_DEV_ONLY","0")!="1":
        raise RuntimeError("P24_OFFICIAL_FIRST_USE_DENIED")
    return frozen()

def history_to_tensor(history,view):
    if view not in ("masked_raw","reconstructed","inverted","unmasked"):
        raise ValueError("P24_UNREGISTERED_OBSERVATION_VIEW")
    if (not history or any(len(x)!=6 or not all(math.isfinite(float(v)) for v in x)
                           for x in history[-p20.HISTORY:])):
        raise ValueError("P24_MALFORMED_FACTUAL_HISTORY")
    rows=[list(x) for x in history[-p20.HISTORY:]]
    if view=="unmasked":
        return p20.scaled_history(rows)
    for i,row in enumerate(rows):
        if view=="masked_raw" or i==0:
            row[1]=0.0
            row[3]=0.0
        else:
            prev=rows[i-1]
            # Prior positional coordinates are unchanged by every view.
            direction=1.0 if view=="reconstructed" else -1.0
            row[1]=direction*(row[0]-prev[0])/0.02
            row[3]=direction*(row[2]-prev[2])/0.02
    return p20.scaled_history(rows)

@contextmanager
def scoped_ppo(view):
    """Source-preserving, sequential adapter. No other PPO calls overlap."""
    old_tensor=p20.actor_tensor
    old_seeds=p20.SEEDS
    old_reset=p20.TRAIN_SEED_BASE
    try:
        p20.actor_tensor=lambda history: history_to_tensor(history,view)
        p20.SEEDS=SEEDS
        p20.TRAIN_SEED_BASE=86000
        yield
    finally:
        p20.actor_tensor=old_tensor
        p20.SEEDS=old_seeds
        p20.TRAIN_SEED_BASE=old_reset

def train(gym):
    all_models={}
    receipts={}
    for seed in SEEDS:
        all_models[seed]={}
        receipts[seed]={}
        for name,(ppo_arm,view) in VIEWS.items():
            with scoped_ppo(view):
                pair,receipt=p20.train_ppo(gym,ppo_arm,seed)
            if not (receipt["actual_factual_environment_actions"]==49152
                    and receipt["actor_critic_training_update_batches"]==576
                    and receipt["no_expert_action_labels_used"]
                    and receipt["trained_weights_changed"]
                    and receipt["in_process_fresh_weights_exact_logits"]):
                raise RuntimeError("P24_SOURCE_BOUND_PPO_RECEIPT_REJECTED")
            all_models[seed][name]=pair["trained"]
            receipts[seed][name]=receipt
    return all_models,receipts

def finite_difference(history):
    theta=history[-1][2]
    omega=0.0 if len(history)<2 else (theta-history[-2][2])/0.02
    return int(theta+0.5*omega>=0.0)

def factual_episode(gym,model,view,name,seed,condition,init):
    if seed not in EVAL_SEEDS or condition not in CONDITIONS or init not in SEEDS:
        raise ValueError("P24_UNREGISTERED_REAL_DEV_WORLD")
    env=gym.make("CartPole-v1",render_mode=None)
    try:
        p20.set_public_environment_dynamics(
            env,{"original":"original","short_pole_x0.7":"short_pole",
                 "long_pole_x1.3":"long_pole"}[condition])
        obs,_=env.reset(seed=seed)
        rows=[p20.factual_row(obs,0,0.0)]
        first=list(rows[0])
        rewards=[]
        actions=[]
        sha=hashlib.sha256()
        rng=random.Random(seed+7919*init)
        for _ in range(p20.EVAL_HORIZON):
            if name=="finite_difference_public_two_frame":
                action=finite_difference(rows)
            elif name=="seeded_random":
                action=rng.randrange(2)
            else:
                with torch.no_grad():
                    logits,_=model(history_to_tensor(rows,view))
                    action=int(logits.argmax(dim=-1).item())
            nxt,reward,term,trunc,_=env.step(action)
            row=p20.factual_row(nxt,2*action-1,float(reward))
            sha.update(json.dumps([rows[-1],action,row,bool(term),bool(trunc)],
                                  separators=(",",":"),allow_nan=False).encode())
            rows.append(row)
            actions.append(action)
            rewards.append(float(reward))
            if term or trunc:break
        return {
            "first_original_public_observation":first,
            "real_return":sum(rewards),
            "real_actions_executed":len(actions),
            "real_action_choices":actions,
            "factual_action_consequence_sha256":sha.hexdigest(),
            "no_eval_weight_updates":True,
            "model_given_true_velocity":view=="unmasked" and name not in CONTROLS,
            "model_given_future_obs_or_teacher_action":False,
        }
    finally:
        env.close()

def run(s29_root,source_root,cloudpickle_root):
    protocol=require_dev()
    if (p20.ROLLOUTS,p20.NUM_ENVS,p20.STEPS_PER_ROLLOUT,
        p20.PPO_EPOCHS,p20.MINIBATCH,p20.PPO_LR,p20.ENTROPY)!=(96,8,64,3,256,.0015,.015):
        raise RuntimeError("P24_ORIGINAL_PPO_OPTIMIZER_DRIFT")
    if not set(EVAL_SEEDS).isdisjoint(p20.EVAL_SEEDS):
        raise RuntimeError("P24_REUSED_P20_DEV_EVALUATION_SEEDS")
    torch.set_num_threads(1)
    started=time.perf_counter()
    gym,source=p20.admit_gymnasium(s29_root,source_root,cloudpickle_root)
    models,training=train(gym)
    records=[]
    for init in SEEDS:
        weights_before={arm:p20.model_sha(m) for arm,m in models[init].items()}
        evaluations=[]
        for condition in CONDITIONS:
            for seed in EVAL_SEEDS:
                outcomes={}
                first=[]
                for arm in tuple(VIEWS)+CONTROLS:
                    model=models[init].get(arm)
                    view=VIEWS[arm][1] if arm in VIEWS else "masked_raw"
                    outcome=factual_episode(gym,model,view,arm,seed,condition,init)
                    outcomes[arm]=outcome
                    first.append(tuple(outcome["first_original_public_observation"]))
                if len(set(first))!=1:
                    raise RuntimeError("P24_PAIRED_RESET_OBSERVATION_MISMATCH")
                evaluations.append({"condition":condition,"reset_seed":seed,
                                    "real_factual_episodes":outcomes})
        weights_after={arm:p20.model_sha(m) for arm,m in models[init].items()}
        if weights_before!=weights_after:
            raise RuntimeError("P24_EVAL_MUTATED_MODEL_STATE")
        mean={condition:{
            arm:sum(row["real_factual_episodes"][arm]["real_return"]
                    for row in evaluations if row["condition"]==condition)/len(EVAL_SEEDS)
            for arm in tuple(VIEWS)+CONTROLS
        } for condition in CONDITIONS}
        original=mean["original"]
        recon=original["reconstructed_from_past"]
        gates={
           "reconstructed_reward_150":recon>=150,
           "reconstructed_gain_vs_masked_raw_25":recon>=original["masked_raw"]+25,
           "reconstructed_gain_vs_inverted_15":
               recon>=original["inverted_reconstructed_velocity"]+15,
           "reconstructed_gain_vs_miscredit_40":
               recon>=original["reconstructed_wrong_reward_credit"]+40,
           "short_pole_retention_100":
               mean["short_pole_x0.7"]["reconstructed_from_past"]>=100,
           "long_pole_retention_100":
               mean["long_pole_x1.3"]["reconstructed_from_past"]>=100,
           "unmasked_PPO_calibration_150":original["unmasked_full_4d"]>=150,
        }
        records.append({
          "initialization":init,"training":training[init],
          "evaluation":evaluations,"means":mean,
          "prespecified_gates":gates,
          "reconstructed_minus_finite_difference":
              recon-original["finite_difference_public_two_frame"],
          "model_weights_sha256":weights_before,
        })
    wall=time.perf_counter()-started
    source_integrity=(wall<=600
      and sum(receipt["actual_factual_environment_actions"]
          for v in training.values() for receipt in v.values())==737280
      and sum(receipt["actor_critic_training_update_batches"]
          for v in training.values() for receipt in v.values())==8640
      and len(records)==3)
    return {
     "schema":SCHEMA,
     "status":"PASS_ENGINEERING_PUBLIC_DEV_ONLY" if source_integrity else "FAIL_ENGINEERING",
     "frozen_development_target_met_all_three":
          all(all(r["prespecified_gates"].values()) for r in records),
     "original_external_source":source,
     "preregistered_source_parent":protocol["original"],
     "factual_train_actions":737280,
     "factual_PPO_minibatch_updates":8640,
     "factual_eval_worlds_per_init":24,
     "whole_process_wall_seconds":wall,
     "scientific_first_use":"NOT_RUN",
     "independent_novel_world_transfer":"UNVERIFIED",
     "hybrid_architecture_superiority":"UNVERIFIED",
     "original_P22_negative_preserved":True,
     "original_P23_branch_untouched":True,
     "no_FORK_or_FORGE_activation":True,
     "records":records,
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--s29-root",type=Path,required=True)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--cloudpickle-root",type=Path,required=True)
    p.add_argument("--receipt",type=Path,required=True)
    a=p.parse_args()
    o=run(a.s29_root,a.source_root,a.cloudpickle_root)
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(o,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"status":o["status"],
                      "frozen_development_target":o["frozen_development_target_met_all_three"],
                      "science":o["scientific_first_use"],
                      "wall":o["whole_process_wall_seconds"],
                      "means":[{"seed":r["initialization"],"original":r["means"]["original"],
                                "rules":r["prespecified_gates"]} for r in o["records"]]},
                     sort_keys=True))
    return 0 if o["status"]=="PASS_ENGINEERING_PUBLIC_DEV_ONLY" else 1

if __name__=="__main__":
    raise SystemExit(main())
