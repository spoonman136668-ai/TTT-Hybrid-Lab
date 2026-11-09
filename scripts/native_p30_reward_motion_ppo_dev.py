"""P30 public DEV: actual reward-only PPO with motion features, not teacher labels.

P20 original PPO source and owner physics code remain unchanged; four same-size
GRU actors receive identical raw source observations and equal reward-only
factual action/optimizer budgets. P29 positive supervised effect result and
P22/P24 reward-only negatives are retained, never relabeled.
"""
from __future__ import annotations
from contextlib import contextmanager
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
from torch import nn

from scripts import native_p20_reward_only_ppo_dev as p20

SCHEMA="ttt-hybrid.native-p30.reward-only-motion-ppo-dev.v1"
PROTOCOL=ROOT/"experiments/native_p30_reward_motion_ppo_frozen.json"
SEEDS=(4507,4517,4523)
TRAIN_BASE=130000
EVAL_SEEDS=(138011,138019,138029,138037,138049,138059,138071,138089)
CONDITIONS=("original","short_pole_x0.7","long_pole_x1.3")
VIEWS={
  "masked_raw_reward_gru":("reward_gru","raw"),
  "derived_motion_reward_gru":("reward_gru","derived"),
  "placebo_action_hash_reward_gru":("reward_gru","placebo"),
  "derived_wrong_credit_gru":("wrong_credit_gru","derived"),
}
CONTROLS=("untrained_derived_gru","finite_difference_public_two_frame","seeded_random")
SIX_SCALE=(.5,.5,10.,5.,1.,1.)
MOTION_SCALE=(.5,1.0,.05,.05)
DT=.02
H=p20.HISTORY

def frozen():
    p=json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if (p.get("schema")!="ttt-hybrid.native-p30.reward-only-motion-feature-ppo-frozen.v1"
        or p["status"]!="PRE_IMPLEMENTATION_FROZEN_FIRST_USE_DENIED"
        or p["parent"]["qualified_sha"]!="f86efac4b9fbfe377f28ae82b8dd93bd98a45e09"
        or p["source"]["no_protected_science_first_use"] is not True
        or p["train"]["new_model_seeds"]!=list(SEEDS)
        or p["train"]["new_reset_base"]!=TRAIN_BASE
        or p["train"]["all_arms"]!=list(VIEWS)
        or p["evaluation"]["seeds"]!=list(EVAL_SEEDS)
        or p["evaluation"]["conditions"]!=list(CONDITIONS)
        or p["evaluation"]["agents"]!=list(VIEWS)+list(CONTROLS)
        or p["train"]["factual_actions_per_arm_seed"]!=49152
        or p["train"]["update_batches_per_arm_seed"]!=576):
        raise RuntimeError("P30_FROZEN_PPO_SOURCE_OR_HYPOTHESIS_DRIFT")
    return p

def require_dev():
    if os.environ.get("NATIVE_P30_DEV_ONLY")!="1":
        raise RuntimeError("P30_PROTECTED_SCIENTIFIC_FIRST_USE_DENIED")
    p=frozen()
    if (p20.ROLLOUTS,p20.NUM_ENVS,p20.STEPS_PER_ROLLOUT,
        p20.PPO_EPOCHS,p20.MINIBATCH,p20.PPO_LR,p20.ENTROPY)!=(
            96,8,64,3,256,.0015,.015):
        raise RuntimeError("P30_ORIGINAL_REWARD_PPO_BUDGET_OR_OPTIMIZER_DRIFT")
    return p

def placebo_values(history,seed,index):
    """Deterministic placebo depends on prior real action/reward only, not state."""
    if seed not in SEEDS or index<0 or index>=H:
        raise ValueError("P30_UNREGISTERED_PLACEBO_SOURCE")
    previous=[(int(row[4]),float(row[5])) for row in history]
    token=json.dumps(["P30-NO-STATE-SOURCE",seed,len(history),index,previous],
                     separators=(",",":"),allow_nan=False)
    raw=hashlib.sha256(token.encode()).digest()
    return tuple((int.from_bytes(raw[2*k:2*k+2],"big")/32767.5)-1.
                 for k in range(4))

def view_tensor(history,view,seed):
    """Only actual pre-action history, with raw true velocity irrevocably zeroed."""
    if view not in ("raw","derived","placebo") or seed not in SEEDS:
        raise ValueError("P30_UNREGISTERED_REWARD_POLICY_VIEW")
    if not history or len(history)>200000:
        raise ValueError("P30_INVALID_HISTORY_SIZE")
    window=[list(row) for row in history[-H:]]
    if any(len(row)!=6 or not all(math.isfinite(float(v)) for v in row)
           for row in window):
        raise ValueError("P30_NONFINITE_OR_MALFORMED_SOURCE_HISTORY")
    out=torch.zeros((1,H,10),dtype=torch.float32)
    pad=H-len(window)
    for i,row in enumerate(window):
        dest=pad+i
        # Source-observed real velocities exist in the original row but are
        # stripped in ALL reward-learning arms including the feature-positive.
        for k in (0,2,4,5):
            out[0,dest,k]=float(row[k])*SIX_SCALE[k]
        if view=="derived":
            if i>=1:
                prior=window[i-1]
                out[0,dest,6]=((row[0]-prior[0])/DT)*MOTION_SCALE[0]
                out[0,dest,7]=((row[2]-prior[2])/DT)*MOTION_SCALE[1]
            if i>=2:
                prev1,prev2=window[i-1],window[i-2]
                out[0,dest,8]=((row[0]-2*prev1[0]+prev2[0])/(DT*DT))*MOTION_SCALE[2]
                out[0,dest,9]=((row[2]-2*prev1[2]+prev2[2])/(DT*DT))*MOTION_SCALE[3]
        elif view=="placebo":
            out[0,dest,6:10]=torch.tensor(
                placebo_values(window,seed,i),dtype=torch.float32)
    if not bool(torch.isfinite(out).all()):
        raise RuntimeError("P30_NONFINITE_PAST_ONLY_REWARD_FEATURE")
    return out

class RewardGRU10(nn.Module):
    """Identical architecture and parameter count across P30 control arms."""
    def __init__(self):
        super().__init__()
        self.recurrent=nn.GRU(10,9,batch_first=True)
        self.actor=nn.Linear(9,2)
        self.critic=nn.Linear(9,1)
    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(H,10):
            raise ValueError("P30_REWARD_GRU_10_INPUT_SHAPE")
        hidden=self.recurrent(x)[0][:,-1,:]
        return self.actor(hidden),self.critic(hidden).reshape(-1)

@contextmanager
def scoped_original_ppo(view,seed):
    """Only a sequential local adapter, restores every original P20 reference."""
    if view not in ("raw","derived","placebo") or seed not in SEEDS:
        raise ValueError("P30_UNREGISTERED_PPO_SCOPE")
    old={
       "actor_tensor":p20.actor_tensor,
       "model_for_arm":p20.model_for_arm,
       "MODEL_CLASSES":p20.MODEL_CLASSES,
       "TRAIN_ARMS":p20.TRAIN_ARMS,
       "SEEDS":p20.SEEDS,
       "TRAIN_SEED_BASE":p20.TRAIN_SEED_BASE
    }
    def model_for_arm(arm,s):
        if arm not in ("reward_gru","wrong_credit_gru") or s!=seed:
            raise ValueError("P30_UNREGISTERED_EXACT_TRAINING_MODEL")
        torch.manual_seed(s)
        return RewardGRU10()
    try:
        p20.actor_tensor=lambda history:view_tensor(history,view,seed)
        p20.model_for_arm=model_for_arm
        p20.MODEL_CLASSES={"reward_gru":RewardGRU10,
                           "wrong_credit_gru":RewardGRU10}
        p20.TRAIN_ARMS=("reward_gru","wrong_credit_gru")
        p20.SEEDS=SEEDS
        p20.TRAIN_SEED_BASE=TRAIN_BASE
        yield
    finally:
        for k,v in old.items():
            setattr(p20,k,v)

def train(gym):
    trained={};receipts={};untrained={}
    for seed in SEEDS:
        trained[seed]={};receipts[seed]={}
        for name,(original_arm,view) in VIEWS.items():
            with scoped_original_ppo(view,seed):
                pair,receipt=p20.train_ppo(gym,original_arm,seed)
            if (receipt["actual_factual_environment_actions"]!=49152
                or receipt["actor_critic_training_update_batches"]!=576
                or not receipt["true_environment_action_consequence_reward_only"]
                or not receipt["no_expert_action_labels_used"]
                or not receipt["trained_weights_changed"]
                or not receipt["in_process_fresh_weights_exact_logits"]
                or receipt["reward_credit_time_and_env_scrambled"]!=(
                    name=="derived_wrong_credit_gru")):
                raise RuntimeError("P30_ORIGINAL_PPO_FACTUAL_TRAINING_RECEIPT_FAIL")
            trained[seed][name]=pair["trained"]
            if name=="derived_motion_reward_gru":
                untrained[seed]=pair["untrained"]
            receipts[seed][name]=receipt
        # All P30 learners begin at identical weights by design.
        starts=[]
        for arm in VIEWS:
            with scoped_original_ppo(VIEWS[arm][1],seed):
                torch.manual_seed(seed)
                model=RewardGRU10()
            starts.append(p20.model_sha(model))
        if len(set(starts))!=1:
            raise RuntimeError("P30_INITIAL_WEIGHT_MISMATCH")
    return trained,untrained,receipts

def finite_difference_public(history):
    theta=history[-1][2]
    omega=(theta-history[-2][2])/DT if len(history)>=2 else 0.
    return int(theta+0.5*omega>=0.)

def single_real_episode(gym,model,arm,seed,condition,init):
    if (seed not in EVAL_SEEDS or condition not in CONDITIONS or
        init not in SEEDS or arm not in tuple(VIEWS)+CONTROLS):
        raise ValueError("P30_UNREGISTERED_SEALED_SOURCE_TEST")
    env=gym.make("CartPole-v1",render_mode=None)
    try:
        descriptor=p20.set_public_environment_dynamics(
            env,{"original":"original","short_pole_x0.7":"short_pole",
                 "long_pole_x1.3":"long_pole"}[condition])
        sensor,_=env.reset(seed=seed)
        history=[p20.factual_row(sensor,0,0.)]
        initial=history[0].copy()
        rng=random.Random(seed+17411*init)
        sha=hashlib.sha256()
        actions=[]
        rewards=[]
        for _ in range(p20.EVAL_HORIZON):
            if arm=="finite_difference_public_two_frame":
                choice=finite_difference_public(history)
            elif arm=="seeded_random":
                choice=rng.randrange(2)
            else:
                view=("derived" if arm=="untrained_derived_gru"
                      else VIEWS[arm][1])
                with torch.no_grad():
                    logits,_=model(view_tensor(history,view,init))
                choice=int(logits.argmax(dim=1).item())
            sensor,reward,terminated,truncated,_=env.step(choice)
            observed=p20.factual_row(sensor,2*choice-1,float(reward))
            sha.update(json.dumps([history[-1],choice,observed,
                        bool(terminated),bool(truncated)],
                        separators=(",",":"),allow_nan=False).encode())
            history.append(observed)
            actions.append(choice)
            rewards.append(float(reward))
            if terminated or truncated:
                break
        return {
           "seed":seed,"condition":condition,
           "source_physics_pole_ratio":descriptor["pole_length_ratio"],
           "initial_public_six_observation":initial,
           "actual_source_rewards":sum(rewards),
           "true_executed_source_actions":len(actions),
           "actual_source_action_choices":actions,
           "actual_action_consequence_sha256":sha.hexdigest(),
           "training_oracle_action_labels_provided":False,
           "given_true_source_velocity":False,
           "given_future_action_consequence":False,
           "eval_optimizer_updates":0
        }
    finally:
        env.close()

def run(s29_root,source_root,cloudpickle_root):
    spec=require_dev()
    started=time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    gym,source=p20.admit_gymnasium(s29_root,source_root,cloudpickle_root)
    trained,untrained,train_receipts=train(gym)
    records=[]
    for seed in SEEDS:
        before={name:p20.model_sha(m) for name,m in trained[seed].items()}
        before["untrained_derived_gru"]=p20.model_sha(untrained[seed])
        all_episodes=[]
        for condition in CONDITIONS:
            for reset in EVAL_SEEDS:
                seen=[]
                outcomes={}
                for arm in tuple(VIEWS)+CONTROLS:
                    model=(trained[seed][arm] if arm in VIEWS else
                           untrained[seed] if arm=="untrained_derived_gru" else None)
                    result=single_real_episode(gym,model,arm,reset,condition,seed)
                    outcomes[arm]=result
                    seen.append(tuple(result["initial_public_six_observation"]))
                if len(set(seen))!=1:
                    raise RuntimeError("P30_PAIRED_REAL_SOURCE_RESET_MISMATCH")
                all_episodes.append({"condition":condition,"reset_seed":reset,
                                     "actual_source_episodes":outcomes})
        after={name:p20.model_sha(m) for name,m in trained[seed].items()}
        after["untrained_derived_gru"]=p20.model_sha(untrained[seed])
        if before!=after:
            raise RuntimeError("P30_HELDOUT_SOURCE_EVAL_UPDATED_MODEL")
        means={
            condition:{arm:sum(
                row["actual_source_episodes"][arm]["actual_source_rewards"]
                for row in all_episodes if row["condition"]==condition)/len(EVAL_SEEDS)
                for arm in tuple(VIEWS)+CONTROLS}
            for condition in CONDITIONS}
        original=means["original"]
        target=original["derived_motion_reward_gru"]
        gates={
          "derived_real_source_original_reward_150":target>=150,
          "derived_over_equal_raw_original_25":
              target>=original["masked_raw_reward_gru"]+25,
          "derived_over_action_hash_placebo_original_25":
              target>=original["placebo_action_hash_reward_gru"]+25,
          "derived_over_wrong_reward_credit_original_40":
              target>=original["derived_wrong_credit_gru"]+40,
          "derived_over_untrained_original_75":
              target>=original["untrained_derived_gru"]+75,
          "derived_short_source_physics_reward_100":
              means["short_pole_x0.7"]["derived_motion_reward_gru"]>=100,
          "derived_long_source_physics_reward_100":
              means["long_pole_x1.3"]["derived_motion_reward_gru"]>=100
        }
        records.append({
           "model_seed":seed,
           "trained_original_factual_PPO_receipts":train_receipts[seed],
           "source_reset_and_execution":all_episodes,
           "real_source_return_means_by_condition":means,
           "prospectively_frozen_rules":gates,
           "derived_outperforms_fixed_two_frame_rule_all_conditions":
              all(means[cond]["derived_motion_reward_gru"]>=
                  means[cond]["finite_difference_public_two_frame"]
                  for cond in CONDITIONS),
           "trained_model_weights_not_updated_during_eval":True
        })
    wall=time.perf_counter()-started
    expected_train=49152*len(SEEDS)*len(VIEWS)
    expected_updates=576*len(SEEDS)*len(VIEWS)
    actual_train=sum(
       v["actual_factual_environment_actions"] for r in records
       for v in r["trained_original_factual_PPO_receipts"].values())
    actual_updates=sum(
       v["actor_critic_training_update_batches"] for r in records
       for v in r["trained_original_factual_PPO_receipts"].values())
    result={
      "schema":SCHEMA,
      "engineering":"PASS_PUBLIC_DEV_ENGINEERING_ONLY"
          if wall<=spec["qualification"]["experiment_wall_cap_seconds"]
          and actual_train==expected_train and actual_updates==expected_updates
          else "FAIL_P30_ENGINEERING",
      "frozen_reward_only_policy_acquisition_all_seeds":
          all(all(r["prospectively_frozen_rules"].values()) for r in records),
      "strong_conventional_architectural_superiority_demonstrated":False,
      "original_p20_PPO_train_source_unchanged":True,
      "P29_supervised_effect_result_preserved":True,
      "P24_reward_only_null_preserved":True,
      "independent_P26_source_untouched":True,
      "protected_science_first_use":"NOT_RUN",
      "accepted_model_promotion":False,
      "FORGE_activation":False,
      "source_admission":source,
      "original_factual_train_actions":actual_train,
      "original_factual_PPO_minibatches":actual_updates,
      "original_real_eval_actions":sum(
         outcome["true_executed_source_actions"]
         for rec in records for r in rec["source_reset_and_execution"]
         for outcome in r["actual_source_episodes"].values()),
      "whole_process_wall_seconds":wall,
      "records":records
    }
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--s29-root",type=Path,required=True)
    parser.add_argument("--source-root",type=Path,required=True)
    parser.add_argument("--cloudpickle-root",type=Path,required=True)
    parser.add_argument("--receipt",type=Path,required=True)
    args=parser.parse_args()
    result=run(args.s29_root,args.source_root,args.cloudpickle_root)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("P30_ORIGINAL_REWARD_ONLY_PPO_DEV_SUMMARY="+json.dumps({
       "engineering":result["engineering"],
       "frozen_reward_only_policy_acquisition":
           result["frozen_reward_only_policy_acquisition_all_seeds"],
       "science_first_use":result["protected_science_first_use"],
       "total_real_training_actions":result["original_factual_train_actions"],
       "total_real_optimizer_updates":result["original_factual_PPO_minibatches"],
       "total_actual_eval_source_actions":result["original_real_eval_actions"],
       "whole_process_seconds":result["whole_process_wall_seconds"],
       "seeds":[{
          "seed":r["model_seed"],
          "frozen_rules":r["prospectively_frozen_rules"],
          "real_source_returns":r["real_source_return_means_by_condition"],
          "conventional_rule_outperformed_all_conditions":
              r["derived_outperforms_fixed_two_frame_rule_all_conditions"]
       } for r in result["records"]]
    },sort_keys=True))
    return 0 if result["engineering"]=="PASS_PUBLIC_DEV_ENGINEERING_ONLY" else 1

if __name__=="__main__":
    raise SystemExit(main())
