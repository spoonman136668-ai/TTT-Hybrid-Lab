"""Native P11: learn temporal causal credit from factual consequences.

Development only. Compare a structured learned current/previous-action credit
readout against a separately trained GRU with the SAME completed history,
factual training stream, optimizer steps and two-step action planner.

This is not the native Hybrid itself, not an independently born unseen science
world, and does not qualify a model for autonomous FORGE research.
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

from scripts.native_p10_closed_loop_intervention_dev import (
    World,FAMILIES,DEV_WORLD_SEEDS,WARMUP,DECISIONS
)

SCHEMA="ttt-hybrid.native-p11.temporal-credit-development.v1"
HISTORY=8
FIT_SEED=54131
VALIDATION_SEED=54179
FIT_EPISODES=150
VALIDATION_EPISODES=48
EVENTS=48
INIT_SEEDS=(353,359,367)
BATCH=256
UPDATES=1200
LR_START=.006
LR_FLOOR=.0006
WALL_LIMIT_SECONDS=110
ARMS=("structured_lookahead","gru_lookahead","structured_greedy",
      "gru_greedy","structured_scrambled","random_action",
      "handcoded_last_pair","informed_oracle")


def require_dev():
    if os.getenv("NATIVE_P11_DEV_ONLY","1")!="1":
        raise RuntimeError("P11_SCIENTIFIC_SOURCE_FIRST_USE_DENIED")


class LearnedTemporalCredit(nn.Module):
    """Learns current + previous action coefficients from completed history.

    Both coefficients and intercept are ordinary gradient-trained outputs.
    The two candidate causal slots are a hand-built inductive bias, not a
    discovered autonomous causal graph or an inferred latent lag label.
    """
    def __init__(self):
        super().__init__()
        self.context=nn.Sequential(
            nn.Linear(HISTORY*2,24),nn.Tanh(),
            nn.Linear(24,16),nn.Tanh(),nn.Linear(16,3))

    def forward(self,history:torch.Tensor,action:torch.Tensor)->torch.Tensor:
        if history.ndim!=3 or history.shape[1:]!=(HISTORY,2) or (
            action.shape!=(history.shape[0],1)):
            raise ValueError("P11_CONTEXT_ACTION_SHAPE")
        coeff=self.context(history.reshape(-1,HISTORY*2))
        previous=history[:,-1,0:1]
        return coeff[:,0:1]+coeff[:,1:2]*action+coeff[:,2:3]*previous


class ConventionalCausalGRU(nn.Module):
    """Independently trained conventional recurrence, near-matched parameters."""
    def __init__(self):
        super().__init__()
        self.gru=nn.GRU(2,12,batch_first=True)
        self.readout=nn.Sequential(
            nn.Linear(14,16),nn.Tanh(),nn.Linear(16,1))

    def forward(self,history:torch.Tensor,action:torch.Tensor)->torch.Tensor:
        if history.ndim!=3 or history.shape[1:]!=(HISTORY,2) or (
            action.shape!=(history.shape[0],1)):
            raise ValueError("P11_CONTEXT_ACTION_SHAPE")
        hidden=self.gru(history)[0][:,-1,:]
        return self.readout(torch.cat((hidden,history[:,-1,0:1],action),dim=1))


def data(seed:int,episodes:int)->tuple:
    """Every supervised input precedes the current *factual* outcome.

    Unobserved synthetic lag and regime variables affect labels only. No
    delay identity or regime sign is provided to either model in training.
    """
    if seed not in (FIT_SEED,VALIDATION_SEED) or (
        type(episodes) is not int or not 1<=episodes<=200):
        raise ValueError("P11_UNREGISTERED_DATA_SOURCE")
    rng=random.Random(seed)
    hx,acts,targets=[],[],[]
    for _ in range(episodes):
        sign=rng.choice((-1,1))
        gain=rng.uniform(.38,.57)
        intercept=rng.uniform(-.075,.075)
        lag_fraction=rng.choice((0.,.5,1.))
        completed=[]
        previous=0
        for t in range(EVENTS):
            if t and rng.random()<.07:
                sign=-sign
            action=rng.choice((-1,1))
            # Construct strictly from prior completed events (never current y).
            prior=tuple(completed[-HISTORY:]) if t>=HISTORY else None
            y=(intercept+gain*sign*((1-lag_fraction)*action+
                 lag_fraction*previous)+rng.gauss(0,.020))
            if prior is not None:
                hx.append([v for a,z in prior for v in (a,z)])
                acts.append([action])
                targets.append([y])
            completed.append((action,y))
            previous=action
    return (
        torch.tensor(hx,dtype=torch.float32).reshape(-1,HISTORY,2),
        torch.tensor(acts,dtype=torch.float32),
        torch.tensor(targets,dtype=torch.float32),
    )


def train_arm(cls,seed,fit,val,updates=UPDATES):
    if cls not in (LearnedTemporalCredit,ConventionalCausalGRU) or (
        seed not in INIT_SEEDS or type(updates) is not int or
        not 1<=updates<=UPDATES):
        raise ValueError("P11_UNREGISTERED_TRAINING")
    torch.manual_seed(seed)
    model=cls()
    original={k:v.detach().clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR_START)
    sampler=torch.Generator().manual_seed(seed+9999)
    sha=hashlib.sha256()
    curve=[]
    start=time.perf_counter()
    for step in range(1,updates+1):
        ix=torch.randint(len(fit[0]),(BATCH,),generator=sampler)
        sha.update(ix.numpy().tobytes())
        prediction=model(fit[0][ix],fit[1][ix])
        loss=F.mse_loss(prediction,fit[2][ix])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        optimizer.param_groups[0]["lr"]=LR_FLOOR+(
            .5*(LR_START-LR_FLOOR)*(1+math.cos(math.pi*step/updates)))
        if step%100==0 or step==updates:
            model.eval()
            with torch.no_grad():
                estimated=model(val[0],val[1])
                value=float(F.mse_loss(estimated,val[2]))
            model.train()
            curve.append({"update":step,"validation_mse":value})
    model.eval()
    weights_updated=any(not torch.equal(original[k],v)
                        for k,v in model.state_dict().items())
    losses=[x["validation_mse"] for x in curve[-5:]]
    spread=(max(losses)-min(losses))/max(min(losses),1e-12)
    record={
        "class":cls.__name__,"seed":seed,
        "actual_optimizer_updates":updates,
        "batch":BATCH,"unique_factual_fit_examples":len(fit[0]),
        "factual_val_examples":len(val[0]),
        "sample_indices_sha256":sha.hexdigest(),
        "model_parameters":sum(p.numel() for p in model.parameters()),
        "gradient_changed_weights":weights_updated,
        "dev_validation_curve":curve,
        "final_validation_mse":losses[-1],
        "strict_last_five_1pct_plateau":len(losses)==5 and spread<=.01,
        "last_five_spread_ratio":spread,
        "fit_wall_seconds":time.perf_counter()-start,
        "training_target":"ONLY_OBSERVED_FACTUAL_CONTINUOUS_CONSEQUENCE",
    }
    return model,record


def history_tensor(history:list[tuple[int,float]])->torch.Tensor:
    if len(history)<HISTORY or any(a not in (-1,1) or not math.isfinite(y)
                                   for a,y in history[-HISTORY:]):
        raise ValueError("P11_COMPLETED_OBSERVATIONS_REQUIRED")
    return torch.tensor(history[-HISTORY:],dtype=torch.float32).reshape(1,HISTORY,2)


def plan(model,history,*,lookahead=True,scramble=False)->tuple[int,float]:
    """One-step planning horizon with action-dependent next observations.

    For candidate a_t: forecast the CURRENT factual consequence; append a
    hypothetical prediction only for *planning* at t+1; compare best next
    action. Never append that hypothetical prediction to real history.
    After selection the caller observes and appends ONLY realized y_t.
    Both neural competitors use EXACTLY this same planning algorithm.
    """
    h=history_tensor(history)
    if scramble:
        h=torch.flip(h,dims=(1,))
    candidates=[]
    with torch.no_grad():
        for candidate in (-1,1):
            a=torch.tensor([[float(candidate)]])
            present=model(h,a)
            score=present
            if lookahead:
                hypothetic=torch.cat((
                    h[:,1:,:],
                    torch.stack((a.reshape(()),present.reshape(())))
                         .reshape(1,1,2)),dim=1)
                v_neg=model(hypothetic,torch.tensor([[-1.]]))
                v_pos=model(hypothetic,torch.tensor([[1.]]))
                score=score+torch.maximum(v_neg,v_pos)
            candidates.append((candidate,float(score.item()),float(present.item())))
    selected=max(candidates,key=lambda item:(item[1],item[0]))
    return selected[0],selected[2]


def evaluate_episode(family:str,seed:int,models:dict)->dict:
    if family not in FAMILIES or seed not in DEV_WORLD_SEEDS:
        raise ValueError("P11_UNREGISTERED_WORLD_EVALUATION")
    record={}
    identical_prefix=[]
    for arm in ARMS:
        world=World(family,seed)
        completed=[]
        previous=0
        for t in range(WARMUP):
            a=-1 if t%2==0 else 1
            observed=world.realize(t,a,previous)
            completed.append((a,observed))
            previous=a
        identical_prefix.append(tuple(completed))
        rand=random.Random(seed+101+ARMS.index(arm)*17)
        actions=[]
        consequences=[]
        pred_errors=[]
        for step in range(DECISIONS):
            t=WARMUP+step
            if arm=="informed_oracle":
                # Inaccessible regime reference, never a learning comparator.
                t_target=min(WARMUP+DECISIONS-1,t+1) if family=="lagged" else t
                a=world.regime_sign(t_target)
                predicted=None
            elif arm=="random_action":
                a=rand.choice((-1,1))
                predicted=None
            elif arm=="handcoded_last_pair":
                last={}
                for a0,y0 in completed[-HISTORY:]:
                    last[a0]=y0
                a=(1 if last.get(1,0.)>=last.get(-1,0.) else -1)
                predicted=None
            else:
                model=models["structured" if arm.startswith("structured") else "gru"]
                a,predicted=plan(
                    model,completed,
                    lookahead=arm not in ("structured_greedy","gru_greedy"),
                    scramble=arm=="structured_scrambled")
            # Policy has chosen: only NOW is the factual consequence revealed.
            observed=world.realize(t,a,previous)
            actions.append(a)
            consequences.append(observed)
            if predicted is not None:
                pred_errors.append((predicted-observed)**2)
            completed.append((a,observed))
            previous=a
        assert world.intervention_count==WARMUP+DECISIONS
        record[arm]={
            "mean_factual_return":sum(consequences)/DECISIONS,
            "first_24_return":sum(consequences[:24])/24,
            "reversal_24_return":sum(consequences[24:48])/24,
            "cold_revisit_16_return":sum(consequences[48:])/16,
            "factual_prequential_mse":(
                sum(pred_errors)/len(pred_errors) if pred_errors else None),
            "factual_actions":actions,
            "factual_consequences":consequences,
            "decisions":len(actions),"updates_during_evaluation":0,
            "receipt_sha256":hashlib.sha256(json.dumps(
                {"actions":actions,"factual_consequences":consequences},
                separators=(",",":"),allow_nan=False).encode()).hexdigest()
        }
    return {"family":family,"world_seed":seed,
            "prefix_exactly_equal_for_all_arms":len(set(identical_prefix))==1,
            "all_64_factual_action_budgets_equal":all(
                x["decisions"]==DECISIONS and x["updates_during_evaluation"]==0
                for x in record.values()),
            "all_factual_outcomes_finite":all(math.isfinite(y)
                for x in record.values() for y in x["factual_consequences"]),
            "data":record}


def run()->dict:
    require_dev()
    torch.set_num_threads(1)
    start=time.perf_counter()
    train=data(FIT_SEED,FIT_EPISODES)
    valid=data(VALIDATION_SEED,VALIDATION_EPISODES)
    assert len(train[0])==6000 and len(valid[0])==1920
    records=[]
    for seed in INIT_SEEDS:
        structured,rs=train_arm(LearnedTemporalCredit,seed,train,valid)
        gru,rg=train_arm(ConventionalCausalGRU,seed,train,valid)
        models={"structured":structured,"gru":gru}
        episodes=[evaluate_episode(family,world_seed,models)
                  for family in FAMILIES for world_seed in DEV_WORLD_SEEDS]
        params=[rs["model_parameters"],rg["model_parameters"]]
        integrity={
            "actual_same_optimizer_updates":(
                rs["actual_optimizer_updates"]==rg["actual_optimizer_updates"]==UPDATES),
            "same_factual_minibatch_index_digest":(
                rs["sample_indices_sha256"]==rg["sample_indices_sha256"]),
            "both_models_changed_by_factual_gradients":(
                rs["gradient_changed_weights"] and rg["gradient_changed_weights"]),
            "parameters_within_five_percent":(
                (max(params)-min(params))/max(params)<.05),
            "12_dev_worlds_all_evaluated":len(episodes)==12,
            "identical_initial_factual_history_all_arms":all(
                q["prefix_exactly_equal_for_all_arms"] for q in episodes),
            "equal_action_budget_per_episode":all(
                q["all_64_factual_action_budgets_equal"] for q in episodes),
            "all_factual_results_finite":all(
                q["all_factual_outcomes_finite"] for q in episodes),
            "all_seven_plus_oracle_controls":all(set(q["data"])==set(ARMS)
                for q in episodes),
            "no_heldout_consequence_or_lag_label_trained":True,
        }
        summaries={}
        diagnostic={}
        for family in FAMILIES:
            examples=[x for x in episodes if x["family"]==family]
            average={arm:sum(q["data"][arm]["mean_factual_return"]
                             for q in examples)/len(examples) for arm in ARMS}
            summaries[family]=average
            diagnostic[family]={
                "structured_beats_GRU_by_0_05":(
                    average["structured_lookahead"]>=average["gru_lookahead"]+.05),
                "structured_beats_random_by_0_05":(
                    average["structured_lookahead"]>=average["random_action"]+.05),
                "GRU_beats_random_by_0_05":(
                    average["gru_lookahead"]>=average["random_action"]+.05),
                "structured_lookahead_beats_structured_greedy_by_0_05":(
                    average["structured_lookahead"]>=average["structured_greedy"]+.05),
                "GRU_lookahead_beats_GRU_greedy_by_0_05":(
                    average["gru_lookahead"]>=average["gru_greedy"]+.05),
                "structured_beats_scrambled_history_by_0_05":(
                    average["structured_lookahead"]>=average["structured_scrambled"]+.05),
                "structured_within_0_10_of_informed_ref":(
                    average["structured_lookahead"]>=average["informed_oracle"]-.10),
                "GRU_within_0_10_of_informed_ref":(
                    average["gru_lookahead"]>=average["informed_oracle"]-.10),
            }
        records.append({
            "initialization":seed,
            "integrity":"PASS" if all(integrity.values()) else "FAIL",
            "checks":integrity,"training":{"structured":rs,"GRU":rg},
            "development_factual_return":summaries,
            "directional_diagnostics":diagnostic,
            "episodes":episodes
        })
    wall=time.perf_counter()-start
    supports={family:{
        key:sum(int(r["directional_diagnostics"][family][key])
                for r in records)
        for key in records[0]["directional_diagnostics"][family]}
        for family in FAMILIES}
    passed=len(records)==3 and all(r["integrity"]=="PASS" for r in records) and (
        wall<=WALL_LIMIT_SECONDS)
    return {
        "schema":SCHEMA,
        "status":"PASS_DEV_TEMPORAL_CREDIT_INTEGRITY_NOT_SCIENCE"
                 if passed else "FAIL_DEV_TEMPORAL_CREDIT_INTEGRITY",
        "scientific_first_use":"NOT_RUN",
        "scientific_world_seeds_consumed":[],
        "genuine_science_unfamiliar_transfer":"NOT_TESTED",
        "native_substrate_unchanged":True,
        "no_model_promotion":True,
        "fit_examples":len(train[0]),"validation_examples":len(valid[0]),
        "models_trained_per_initialization":2,
        "updates_per_model":UPDATES,
        "total_optimizer_steps":UPDATES*len(records)*2,
        "evaluation_world_families":list(FAMILIES),
        "known_dev_worlds_per_family":len(DEV_WORLD_SEEDS),
        "scored_factual_interventions":len(records)*len(FAMILIES)*len(
            DEV_WORLD_SEEDS)*len(ARMS)*DECISIONS,
        "directional_support_out_of_three":supports,
        "records":records,
        "whole_process_wall_seconds":wall,
        "limits":"Known synthetic development causal families: training explicitly includes random lag fractions, so delayed-action evaluation is not authentic novel science. Temporal causal slot design is a prior. Two-step model-based planner is shared and uses fictive results ONLY for internal planning; never trains on alternatives. Evaluator shows per-policy actual actions and outcomes. Source/optimizer/parameter budgets near-matched; FLOPs and independent baseline competence not authenticated. All hypothesis failures preserved."
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--receipt",required=True,type=Path)
    a=p.parse_args()
    r=run()
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(r,sort_keys=True,indent=2)+"\n")
    print(json.dumps({
        "status":r["status"],
        "scientific_first_use":r["scientific_first_use"],
        "optimizer_steps":r["total_optimizer_steps"],
        "support":r["directional_support_out_of_three"],
        "average_return_by_seed":[{
            "seed":x["initialization"],
            "return":x["development_factual_return"]} for x in r["records"]],
        "seconds":r["whole_process_wall_seconds"]
    },sort_keys=True))
    return 0 if r["status"]=="PASS_DEV_TEMPORAL_CREDIT_INTEGRITY_NOT_SCIENCE" else 1


if __name__=="__main__":
    raise SystemExit(main())
