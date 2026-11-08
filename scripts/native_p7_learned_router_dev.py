"""P7: train context inference from factual transition errors, no context labels.

Independent synthetic development data only. A learned mixture of signed
continuous-action experts reads the eight completed (action, outcome delta)
pairs and current action. NO latent ID, phase position or future outcome can
enter the training input. This is a standalone prototype, NOT integration into
NativeContinuousHybrid, science first-use, or unfamiliar-world transfer.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SCHEMA="ttt-hybrid.native-p7.learned-factual-history-dev.v1"
HISTORY=8
UPDATES=600
BATCH=256
LR=.006
INIT_SEEDS=(303,307,311)
TRAIN_DEV_SEED=51031
VALIDATION_DEV_SEED=51079
SCRAMBLE_DEV_SEED=51997
TRAIN_EPISODES=150
VALID_EPISODES=48
EVENTS_PER_EPISODE=48
WALL_BUDGET_SECONDS=65


def synthetic_dev(seed:int,episodes:int):
    """Factual sequential episodes; latent context returned for SCORING only.

    Both regimes have identical measured current sensor/action/time marginals.
    A random switch changes signed action gain. Gain/intercept/noise vary per
    synthetic episode. Features are snapshotted BEFORE the current result.
    """
    if seed not in (TRAIN_DEV_SEED,VALIDATION_DEV_SEED) or not 0<episodes<=200:
        raise ValueError("P7_UNREGISTERED_DEV_SOURCE")
    rng=np.random.default_rng(seed)
    h,a,y,diagnostic=[],[],[],[]
    for _ in range(episodes):
        intercept=float(rng.uniform(-.08,.08))
        pos=float(rng.uniform(.35,.60))
        neg=-float(rng.uniform(.35,.60))
        latent=int(rng.choice((-1,1)))
        complete=[]
        for t in range(EVENTS_PER_EPISODE):
            if t and rng.random()<.10:
                latent=-latent
            factual_action=int(rng.choice((-1,1)))
            # Do not allow any current factual outcome into prediction input.
            prior=tuple(complete[-HISTORY:]) if t>=HISTORY else None
            delta=intercept+(pos if latent==1 else neg)*factual_action
            delta+=float(rng.normal(0,.025))
            if prior is not None:
                h.append([z for pair in prior for z in pair])
                a.append([factual_action])
                y.append([delta])
                diagnostic.append(latent)
            complete.append((factual_action,delta))
    return (torch.tensor(np.asarray(h),dtype=torch.float32),
            torch.tensor(a,dtype=torch.float32),
            torch.tensor(y,dtype=torch.float32),
            torch.tensor(diagnostic,dtype=torch.int8))


class LearnedRouter(nn.Module):
    """Trainable history-conditioned gate, trained only with factual MSE.

    Two signed experts have explicit +/-0.45 inductive-bias initialization.
    Both expert magnitudes and router weights are optimized, but never with
    latent regime labels or next-event context classification targets.
    """
    def __init__(self):
        super().__init__()
        self.router=nn.Sequential(
            nn.Linear(HISTORY*2,24),nn.Tanh(),
            nn.Linear(24,16),nn.Tanh(),nn.Linear(16,1))
        self.intercept=nn.Parameter(torch.tensor(0.))
        self.experts=nn.Parameter(torch.tensor([.45,-.45]))

    def forward(self,completed_history,factual_action):
        probability=self.router(completed_history).sigmoid()
        signed_gain=probability*self.experts[0]+(1-probability)*self.experts[1]
        return self.intercept+factual_action*signed_gain,probability


class ConventionalMLP(nn.Module):
    """Separately trained, near-parameter-matched direct history predictor."""
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(HISTORY*2+1,24),nn.Tanh(),
            nn.Linear(24,16),nn.Tanh(),nn.Linear(16,1))

    def forward(self,completed_history,factual_action):
        return self.net(torch.cat((completed_history,factual_action),dim=1)),None


def handcoded_prior_pair(history,factual_action):
    """P6-like fixed-slope negative: last OBSERVED opposite-action pair only."""
    pairs=history.reshape(-1,HISTORY,2)
    action,delta=pairs[:,:,0],pairs[:,:,1]
    positions=torch.arange(HISTORY).reshape(1,HISTORY)
    plus=torch.where(action>0,positions,-1).max(dim=1).values
    minus=torch.where(action<0,positions,-1).max(dim=1).values
    both=(plus>=0)&(minus>=0)
    ix=torch.arange(history.shape[0])
    slope=.5*(delta[ix,plus.clamp(min=0)]-
              delta[ix,minus.clamp(min=0)])
    choice=torch.where(both,slope>0,torch.ones_like(both))
    gain=torch.where(choice,.45,-.45).to(factual_action.dtype)
    return factual_action*gain[:,None],choice


def eval_model(model,data):
    with torch.no_grad():
        predictions,probability=model(data[0],data[1])
        mse=float(F.mse_loss(predictions,data[2]))
        if probability is None:
            return mse,None
        inferred=torch.where(probability[:,0]>.5,1,-1)
        return mse,float((inferred==data[3]).float().mean())


def train_one(kind,seed,training,validation):
    if seed not in INIT_SEEDS or kind not in (LearnedRouter,ConventionalMLP):
        raise ValueError("P7_UNREGISTERED_ARM_OR_INIT")
    torch.manual_seed(seed)
    model=kind()
    original={k:v.detach().clone() for k,v in model.state_dict().items()}
    prior_mse,prior_sign=eval_model(model,validation)
    optim=torch.optim.AdamW(model.parameters(),lr=LR)
    sampler=torch.Generator().manual_seed(seed+9999)
    count=len(training[0])
    evaluations=[]
    start=time.perf_counter()
    for update in range(1,UPDATES+1):
        ix=torch.randint(count,(BATCH,),generator=sampler)
        pred,_=model(training[0][ix],training[1][ix])
        loss=F.mse_loss(pred,training[2][ix])
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        optim.step()
        if update%100==0:
            mse,sign=eval_model(model,validation)
            evaluations.append({"update":update,"validation_mse":mse,
                                "diagnostic_sign_accuracy":sign})
    after_mse,after_sign=eval_model(model,validation)
    updated=any(not torch.equal(original[k],v)
                for k,v in model.state_dict().items())
    count_parameters=sum(p.numel() for p in model.parameters())
    return model,{
        "initialization":seed,
        "arm":kind.__name__,
        "training_updates":UPDATES,
        "training_samples_in_minibatches":UPDATES*BATCH,
        "distinct_training_examples":count,
        "parameters":count_parameters,
        "initial_validation_mse":prior_mse,
        "initial_diagnostic_context_accuracy":prior_sign,
        "final_validation_mse":after_mse,
        "final_diagnostic_context_accuracy":after_sign,
        "model_weights_changed_by_real_gradients":updated,
        "validation_checkpoints":evaluations,
        "strict_last_five_checkpoints_mse_spread_at_most_one_pct":(
            max(v["validation_mse"] for v in evaluations[-5:])<=
            min(v["validation_mse"] for v in evaluations[-5:])*1.01),
        "training_loop_wall_seconds":time.perf_counter()-start,
    }


def run():
    if os.environ.get("NATIVE_P7_DEV_ONLY","1")!="1":
        raise RuntimeError("P7_SCIENCE_FIRST_USE_FORBIDDEN")
    torch.set_num_threads(1)
    start=time.perf_counter()
    training=synthetic_dev(TRAIN_DEV_SEED,TRAIN_EPISODES)
    validation=synthetic_dev(VALIDATION_DEV_SEED,VALID_EPISODES)
    shuffled=torch.randperm(len(validation[0]),generator=
                           torch.Generator().manual_seed(SCRAMBLE_DEV_SEED))
    hand_pred,hand_choice=handcoded_prior_pair(validation[0],validation[1])
    hand_mse=float(F.mse_loss(hand_pred,validation[2]))
    hand_accuracy=float((torch.where(hand_choice,1,-1)==validation[3]).float().mean())
    records=[]
    for seed in INIT_SEEDS:
        learned,lr=train_one(LearnedRouter,seed,training,validation)
        _,mlp=train_one(ConventionalMLP,seed,training,validation)
        with torch.no_grad():
            scrambled_pred,_=learned(validation[0][shuffled],validation[1])
            zero_pred,_=learned(torch.zeros_like(validation[0]),validation[1])
            scrambled_mse=float(F.mse_loss(scrambled_pred,validation[2]))
            no_history_mse=float(F.mse_loss(zero_pred,validation[2]))
        checks={
            "factual_gradient_weight_change":(
                lr["model_weights_changed_by_real_gradients"] and
                mlp["model_weights_changed_by_real_gradients"]),
            "identical_optimizer_update_budget":lr["training_updates"]==
                mlp["training_updates"]==UPDATES,
            "identical_train_validation_examples":lr["distinct_training_examples"]==
                mlp["distinct_training_examples"]==len(training[0]),
            "trainable_parameter_counts_within_five_pct":
                abs(lr["parameters"]-mlp["parameters"])/
                max(lr["parameters"],mlp["parameters"])<.05,
            "no_context_label_used_in_training":True,
            "no_current_outcome_in_prediction_history":True,
            "all_losses_finite":all(math.isfinite(x) for x in (
                lr["final_validation_mse"],mlp["final_validation_mse"],
                scrambled_mse,no_history_mse)),
        }
        diagnostics={
            "learned_better_than_its_untrained_initialization":
                lr["final_validation_mse"]<.8*lr["initial_validation_mse"],
            "learned_better_than_p6_style_handcoded":
                lr["final_validation_mse"]<hand_mse,
            "learned_better_than_fully_trained_mlp":
                lr["final_validation_mse"]<mlp["final_validation_mse"],
            "learned_better_than_shuffled_history":
                lr["final_validation_mse"]<scrambled_mse,
            "learned_better_than_no_history":
                lr["final_validation_mse"]<no_history_mse,
            "strict_plateau":
                lr["strict_last_five_checkpoints_mse_spread_at_most_one_pct"],
        }
        records.append({
            "initialization":seed,
            "status":"PASS_DEV_INTEGRITY" if all(checks.values()) else
                     "FAIL_DEV_INTEGRITY",
            "checks":checks,"diagnostics":diagnostics,
            "learned":lr,"trained_conventional":mlp,
            "shuffled_history_mse":scrambled_mse,
            "no_history_mse":no_history_mse,
        })
    elapsed=time.perf_counter()-start
    integrity=(len(records)==len(INIT_SEEDS) and
               all(r["status"]=="PASS_DEV_INTEGRITY" for r in records)
               and elapsed<WALL_BUDGET_SECONDS)
    counts={n:sum(bool(r["diagnostics"][n]) for r in records)
            for n in records[0]["diagnostics"]}
    return {
        "schema":SCHEMA,
        "status":"PASS_TRAINABLE_DEV_INTEGRITY_NOT_SCIENCE" if integrity
                 else "FAIL_TRAINABLE_DEV_INTEGRITY_NOT_SCIENCE",
        "scientific_first_use":"NOT_RUN",
        "scientific_seeds_consumed":[],
        "worlds_constructed":0,
        "scientific_transfer":"NOT_TESTED",
        "context_identity_used_for_training":"NEVER",
        "training_target":"FACTUAL_OBSERVED_CONTINUOUS_SENSOR_DIFFERENCE",
        "training_example_count":len(training[0]),
        "independent_dev_validation_example_count":len(validation[0]),
        "independent_dev_source_seeds":TRAIN_DEV_SEED!=VALIDATION_DEV_SEED,
        "handcoded_prior_pair_validation_mse":hand_mse,
        "handcoded_prior_pair_diagnostic_sign_accuracy":hand_accuracy,
        "support_counts_of_three":counts,
        "records":records,
        "whole_process_wall_seconds":elapsed,
        "limitations":"Known synthetic family; train/validation use different development RNG seeds but NOT sealed independent world sources. Router has initialized signed experts and a hand-designed eight-event input; no native fast-memory integration, no science or transfer. Conventional MLP shares factual examples, AdamW updates and near-equal parameter count but FLOPs not independently matched. Preserve nonplateau and failures."
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--receipt",type=Path,required=True)
    a=parser.parse_args()
    result=run()
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({
        "status":result["status"],
        "scientific_first_use":result["scientific_first_use"],
        "support_counts_of_three":result["support_counts_of_three"],
        "training_example_count":result["training_example_count"],
        "validation_example_count":result["independent_dev_validation_example_count"],
        "whole_process_wall_seconds":result["whole_process_wall_seconds"],
        "validation_mse":[{
            "initialization":r["initialization"],
            "learned":r["learned"]["final_validation_mse"],
            "conventional":r["trained_conventional"]["final_validation_mse"],
            "shuffled":r["shuffled_history_mse"],
            "no_history":r["no_history_mse"],
            "context_sign_accuracy":r["learned"]["final_diagnostic_context_accuracy"]
        } for r in result["records"]],
    },sort_keys=True))
    return 0 if result["status"]=="PASS_TRAINABLE_DEV_INTEGRITY_NOT_SCIENCE" else 1

if __name__=="__main__":
    raise SystemExit(main())
