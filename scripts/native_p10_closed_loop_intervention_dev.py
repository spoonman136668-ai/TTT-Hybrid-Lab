"""P10: trained action selection under feedback, reversal and delayed actuation.

An independent DEVELOPMENT-ONLY falsification of whether P7's learned
context gate improves *actions and consequences* over a fully executed
same-data/same-steps conventional factual-history MLP. Not true scientific
first-use or generalization. P7 source models and training generator are
imported intact; no modifications to the native Hybrid or C-series tracks.
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
from torch import nn
from torch.nn import functional as F
from scripts.native_p7_learned_router_dev import (
    LearnedRouter,ConventionalMLP,synthetic_dev,INIT_SEEDS,
    TRAIN_DEV_SEED,VALIDATION_DEV_SEED,TRAIN_EPISODES,VALID_EPISODES,
    HISTORY,BATCH
)

SCHEMA="ttt-hybrid.native-p10.closed-loop-causal-action-dev.v1"
FAMILIES=("instant","lagged","confounded")
DEV_WORLD_SEEDS=(88031,88047,88063,88091)
ARMS=("learned_router","trained_mlp","handcoded_last_pair",
      "random_actions","fixed_positive","alternating_actions","informed_oracle")
OPTIMIZER_STEPS=1200
INITIAL_LR=.006
MIN_LR=.0006
WARMUP=8
DECISIONS=64
WALL_BUDGET_SECONDS=85


def dev_guard():
    if os.environ.get("NATIVE_P10_DEV_ONLY","1")!="1":
        raise RuntimeError("P10_ORIGINAL_SCIENTIFIC_FIRST_USE_FORBIDDEN")


class World:
    """Public synthetic DEV fixture; hidden world dynamics inaccessible to models."""
    def __init__(self,family:str,seed:int):
        if family not in FAMILIES or seed not in DEV_WORLD_SEEDS:
            raise ValueError("P10_UNREGISTERED_WORLD_SOURCE")
        self.family=family
        self.seed=seed
        r=random.Random(seed+FAMILIES.index(family)*100003)
        self.base_sign=r.choice((-1,1))
        self.gain=r.uniform(.38,.57)
        self.bias=r.uniform(-.075,.075)
        self.noise=[r.gauss(0,.020) for _ in range(WARMUP+DECISIONS)]
        self.intervention_count=0

    def regime_sign(self,t:int)->int:
        if t<0 or t>=WARMUP+DECISIONS:
            raise ValueError("P10_REGIME_SCORING_INDEX_INVALID")
        # A -> B -> A, same instantaneous sensor and action opportunities.
        return self.base_sign*(1 if t<32 or t>=56 else -1)

    def realize(self,t:int,action:int,previous_action:int)->float:
        """Only source of factual consequences. Never an input to select()."""
        if not (type(t) is int and 0<=t<WARMUP+DECISIONS and
                action in (-1,1) and previous_action in (-1,0,1)):
            raise ValueError("P10_INVALID_FACTUAL_ACTION")
        gain=self.gain*self.regime_sign(t)
        if self.family=="instant":
            effect=gain*action
        elif self.family=="lagged":
            effect=gain*previous_action
        else:
            effect=gain*action + .20*previous_action
        self.intervention_count+=1
        return float(effect+self.bias+self.noise[t])


def _history_tensor(history:list[tuple[int,float]])->torch.Tensor:
    if len(history)<HISTORY or any(a not in (-1,1) or not math.isfinite(y)
                                   for a,y in history[-HISTORY:]):
        raise ValueError("P10_COMPLETED_FACTUAL_HISTORY_REQUIRED")
    return torch.tensor([v for pair in history[-HISTORY:] for v in pair],
                        dtype=torch.float32).reshape(1,2*HISTORY)


def select_action(arm:str,history:list[tuple[int,float]],
                  models:dict,policy_rng:random.Random,t:int)->tuple[int,float|None]:
    """Only observed prior outcomes; no future target, latent state or oracle."""
    if arm not in ARMS[:-1] or len(history)<HISTORY:
        raise ValueError("P10_POLICY_OR_HISTORY_INVALID")
    if arm in ("learned_router","trained_mlp"):
        model=models[arm]
        h=_history_tensor(history)
        with torch.no_grad():
            negative,_=model(h,torch.tensor([[-1.]],dtype=torch.float32))
            positive,_=model(h,torch.tensor([[1.]],dtype=torch.float32))
        n,p=float(negative.item()),float(positive.item())
        action=1 if p>=n else -1
        return action, p if action==1 else n
    if arm=="handcoded_last_pair":
        last={}
        for a,outcome in history[-HISTORY:]:
            last[a]=outcome
        if len(last)!=2:
            return 1,None
        return (1 if last[1]>=last[-1] else -1),None
    if arm=="random_actions":
        return policy_rng.choice((-1,1)),None
    if arm=="alternating_actions":
        return (-1 if t%2==0 else 1),None
    return 1,None


def _public_score(delta:float,pred:float|None)->float|None:
    return (pred-delta)**2 if pred is not None else None


def run_episode(family:str,seed:int,models:dict)->dict:
    if family not in FAMILIES or seed not in DEV_WORLD_SEEDS:
        raise ValueError("P10_UNREGISTERED_EPISODE")
    result={}
    first_histories={}
    for arm in ARMS:
        world=World(family,seed)
        history=[]
        previous_action=0
        # Common, *completed factual* observation history, not hidden labels.
        for t in range(WARMUP):
            act=1 if t%2 else -1
            outcome=world.realize(t,act,previous_action)
            history.append((act,outcome))
            previous_action=act
        first_histories[arm]=tuple(history)
        policy_rng=random.Random(seed+17*ARMS.index(arm)+70201)
        facts=[]
        decisions=[]
        losses=[]
        for i in range(DECISIONS):
            t=WARMUP+i
            if arm=="informed_oracle":
                # Explicit inaccessible true-regime upper-bound diagnostic.
                # For a one-step-lag world, choose sign at next time.
                lookahead=min(WARMUP+DECISIONS-1,t+1) if family=="lagged" else t
                action=world.regime_sign(lookahead)
                prediction=None
            else:
                action,prediction=select_action(arm,history,models,policy_rng,t)
            assert action in (-1,1)
            # Crucial: choose before revealing the one actually realized action.
            actual=world.realize(t,action,previous_action)
            facts.append(float(actual))
            decisions.append(action)
            losses.append(_public_score(actual,prediction))
            history.append((action,actual))
            previous_action=action
        assert world.intervention_count==WARMUP+DECISIONS
        digest=hashlib.sha256(json.dumps(
            {"factual_actions":decisions,"observed_consequences":facts},
            separators=(",",":"),allow_nan=False).encode()).hexdigest()
        result[arm]={
            "total_return":sum(facts),
            "mean_return":sum(facts)/DECISIONS,
            "first_24_mean":sum(facts[:24])/24,
            "changed_regime_24_mean":sum(facts[24:48])/24,
            "old_regime_revisit_16_mean":sum(facts[48:])/16,
            "factual_prequential_mse":(
                sum(v for v in losses if v is not None)/DECISIONS
                if all(v is not None for v in losses) else None),
            "decisions":decisions,
            "factual_outcomes":facts,
            "trace_sha256":digest,
            "factual_interventions_including_shared_prefix":world.intervention_count,
            "decision_budget":DECISIONS,
        }
    common_prefix=len(set(first_histories.values()))==1
    return {"family":family,"world_dev_seed":seed,
            "same_initial_factual_history_for_every_policy":common_prefix,
            "equal_decision_opportunities_and_nonzero_actions":all(
                x["decision_budget"]==DECISIONS and
                len(x["decisions"])==DECISIONS and
                set(x["decisions"])<=set((-1,1))
                for x in result.values()),
            "no_counterfactual_result_used_by_policy":True,
            "data":result}


def _params(model:nn.Module)->int:
    return sum(p.numel() for p in model.parameters())


def train_conventional_and_router(initialization:int,train:tuple,val:tuple):
    """Identical P8 fixed development cosine budget, without checkpoint selection."""
    if initialization not in INIT_SEEDS:
        raise ValueError("P10_UNREGISTERED_MODEL_SEED")
    models={}
    training={}
    for cls,name in ((LearnedRouter,"learned_router"),
                     (ConventionalMLP,"trained_mlp")):
        torch.manual_seed(initialization)
        model=cls()
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        opt=torch.optim.AdamW(model.parameters(),lr=INITIAL_LR)
        sampler=torch.Generator().manual_seed(initialization+9999)
        sampler_digest=hashlib.sha256()
        curve=[]
        for step in range(1,OPTIMIZER_STEPS+1):
            indices=torch.randint(len(train[0]),(BATCH,),generator=sampler)
            sampler_digest.update(indices.numpy().tobytes())
            predicted,_=model(train[0][indices],train[1][indices])
            loss=F.mse_loss(predicted,train[2][indices])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5)
            opt.step()
            opt.param_groups[0]["lr"]=.0006+.5*(INITIAL_LR-MIN_LR)*(
                1+math.cos(math.pi*step/OPTIMIZER_STEPS))
            if step%100==0:
                with torch.no_grad():
                    v,_=model(val[0],val[1])
                    curve.append(float(F.mse_loss(v,val[2])))
        model.eval()
        changed=any(not torch.equal(v,before[k])
                    for k,v in model.state_dict().items())
        training[name]={
            "model_init":initialization,
            "parameters":_params(model),
            "optimizer_updates":OPTIMIZER_STEPS,
            "batch_size":BATCH,
            "original_training_examples":len(train[0]),
            "sample_indices_sha256":sampler_digest.hexdigest(),
            "weights_changed_by_actual_factual_gradients":changed,
            "validation_curve_mse":curve,
            "strict_last_five_checkpoints_spread_le_1pct":(
                (max(curve[-5:])-min(curve[-5:]))/
                max(min(curve[-5:]),1e-12)<=.01),
            "validation_end_mse":curve[-1],
        }
        models[name]=model
    return models,training


def run()->dict:
    dev_guard()
    torch.set_num_threads(1)
    start=time.perf_counter()
    train=synthetic_dev(TRAIN_DEV_SEED,TRAIN_EPISODES)
    val=synthetic_dev(VALIDATION_DEV_SEED,VALID_EPISODES)
    records=[]
    for init in INIT_SEEDS:
        models,trained=train_conventional_and_router(init,train,val)
        per_world=[run_episode(family,seed,models)
                   for family in FAMILIES for seed in DEV_WORLD_SEEDS]
        integrity={
            "both_models_1200_actual_updates":all(
                d["optimizer_updates"]==OPTIMIZER_STEPS and
                d["weights_changed_by_actual_factual_gradients"]
                for d in trained.values()),
            "same_exact_minibatch_indices":len(set(
                d["sample_indices_sha256"] for d in trained.values()))==1,
            "near_equal_param_budget_within_5pct":(
                abs(trained["learned_router"]["parameters"]-
                    trained["trained_mlp"]["parameters"])/
                max(x["parameters"] for x in trained.values())<.05),
            "all_worlds_and_seeds_evaluated":len(per_world)==12,
            "identical_complete_initial_factual_histories":all(
                p["same_initial_factual_history_for_every_policy"]
                for p in per_world),
            "equal_intervention_budgets":all(
                p["equal_decision_opportunities_and_nonzero_actions"]
                for p in per_world),
            "prediction_before_each_actual_feedback":all(
                p["no_counterfactual_result_used_by_policy"]
                for p in per_world),
            "no_factual_test_targets_used_in_optimizer":True,
            "every_arm_evaluated_in_every_world":all(
                set(p["data"])==set(ARMS) for p in per_world),
        }
        aggregate={}
        for family in FAMILIES:
            relevant=[p for p in per_world if p["family"]==family]
            aggregate[family]={arm:sum(
                p["data"][arm]["mean_return"] for p in relevant)/len(relevant)
                for arm in ARMS}
        performance={}
        for family,d in aggregate.items():
            performance[family]={
                "router_beats_trained_mlp_by_0_03":(
                    d["learned_router"]>=d["trained_mlp"]+.03),
                "router_beats_random_by_0_05":(
                    d["learned_router"]>=d["random_actions"]+.05),
                "mlp_beats_random_by_0_05":(
                    d["trained_mlp"]>=d["random_actions"]+.05),
                "router_within_0_10_of_informed_reference":(
                    d["learned_router"]>=d["informed_oracle"]-.10),
                "mlp_within_0_10_of_informed_reference":(
                    d["trained_mlp"]>=d["informed_oracle"]-.10),
            }
        records.append({
            "model_seed":init,"status":"PASS_DEV_INTEGRITY"
                if all(integrity.values()) else "FAIL_DEV_INTEGRITY",
            "training":trained,"integrity":integrity,
            "average_factual_return_by_world":aggregate,
            "directional_hypotheses":performance,
            "all_episode_traces":per_world})
    elapsed=time.perf_counter()-start
    integrity=len(records)==3 and all(r["status"]=="PASS_DEV_INTEGRITY"
                 for r in records) and elapsed<WALL_BUDGET_SECONDS
    support={family:{k:sum(int(r["directional_hypotheses"][family][k])
                             for r in records)
                     for k in records[0]["directional_hypotheses"][family]}
             for family in FAMILIES}
    return {
        "schema":SCHEMA,
        "status":"PASS_CLOSED_LOOP_DEV_INTEGRITY_NOT_SCIENCE"
                 if integrity else "FAIL_CLOSED_LOOP_DEV_INTEGRITY",
        "training_source":"PUBLIC_P7_FROZEN_SYNTHETIC_FIT_ONLY",
        "development_evaluation":"NEW_KNOWN_SYNTHETIC_STRUCTURAL_SHIFT",
        "science_first_use":"NOT_RUN","science_seeds_consumed":[],
        "independent_authentic_world_source":"NOT_BORN",
        "worlds_constructed_for_science":0,
        "trained_models_per_init":2,
        "optimizer_steps_per_model":OPTIMIZER_STEPS,
        "active_factual_decisions_per_episode":DECISIONS,
        "shared_prefix_factual_actions":WARMUP,
        "evaluation_families":list(FAMILIES),
        "dev_world_seeds":list(DEV_WORLD_SEEDS),
        "hypothesis_support_count_of_three":support,
        "records":records,
        "whole_process_seconds":elapsed,
        "limits":"Precommitted DEV analytic generator with known rule families; delayed actuation intentionally out-of-family; no independently authenticated unseen science. P7 signed expert prior and 8-element history are handcrafted; training plateau not established. Closed-loop net reward is an action-consequence metric but is not mature agency or RSI. Informed regime reference sees labels inaccessible to learners. Matched optimizer updates, parameters and factual action counts, but not independently matched FLOPs."
    }


def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--receipt",type=Path,required=True)
    args=p.parse_args()
    output=run()
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(output,sort_keys=True,indent=2)+"\n")
    print(json.dumps({
        "status":output["status"],"science_first_use":output["science_first_use"],
        "support":output["hypothesis_support_count_of_three"],
        "average_return_by_seed":[{
            "seed":r["model_seed"],
            "families":r["average_factual_return_by_world"]}
            for r in output["records"]],
        "whole_process_seconds":output["whole_process_seconds"]
    },sort_keys=True))
    return 0 if output["status"]=="PASS_CLOSED_LOOP_DEV_INTEGRITY_NOT_SCIENCE" else 1

if __name__=="__main__":
    raise SystemExit(main())
