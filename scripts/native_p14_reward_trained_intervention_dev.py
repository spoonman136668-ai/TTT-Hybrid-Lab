"""P14 DEV: actual reward-trained token-free causal intervention policy.

Compare one learned causal-feature policy, an equal-information conventional
GRU policy and a same-size reward-misaligned training negative. Each receives
its own actual chosen action consequences in randomized synthetic training
worlds; gradients use realized returns, NEVER source latent lag or privileged
counterfactual targets. In evaluation all policies receive the same 16 forced
probe opportunities, exogenous noise and 64 scored factual actions.

All worlds are developer-authored. No independently born scientific source,
no model-promotion, no FORGE activation, no official first-use.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import io
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
from scripts.native_p10_closed_loop_intervention_dev import (
    World as OldWorld, DEV_WORLD_SEEDS as OLD_SEEDS)
from scripts.native_p12_online_causal_delay_dev import (
    CausalDelayWorld, DEV_SEEDS as LAG_SEEDS,
    FactualLagEstimator, OBS_PREFIX,DECISIONS)
from scripts.native_p13_active_causal_probe_dev import action_from_model

SCHEMA="ttt-hybrid.native-p14.reward-trained-intervention-dev.v1"
INIT_SEEDS=(1211,1223,1237)
TRAIN_WORLD_SEED_BASE=211000
TRAIN_UPDATES=96
BATCH_WORLDS=12
TRAIN_HORIZON=32
HISTORY=8
PROBE_INTERVAL=4
PROBE_SLOTS=16
DISCOUNT=.97
ENTROPY_COEF=.01
TRAIN_LR=.003
GRAD_CLIP=3.
MAX_SECONDS=115
FAMILIES=("instant","lagged","delay2","delay3","changing_delay",
          "sham_no_action")
ARMS=("trained_causal","trained_gru","reward_misaligned",
      "p13_active","p13_passive","random")
TRAIN_CLASSES=("causal","gru","misaligned")


def only_dev():
    if os.getenv("NATIVE_P14_DEV_ONLY","1")!="1":
        raise RuntimeError("P14_SCIENCE_FIRST_USE_DENIED")


class CausalFeaturePolicy(nn.Module):
    """Hand-built causal correlation features; actual action policy IS learned.

    The 8 pairs are completed factual observations. Correlations are
    computed across action at t-lag and factual y_t, no future labels.
    """
    def __init__(self):
        super().__init__()
        self.policy=nn.Sequential(nn.Linear(20,20),nn.Tanh(),nn.Linear(20,1))

    def forward(self,h):
        if h.ndim!=3 or h.shape[1:]!=(HISTORY,2):
            raise ValueError("P14_CAUSAL_INPUT_SHAPE")
        history_action=h[:,:,0]
        history_outcome=h[:,:,1]
        correlations=[]
        for lag in range(4):
            if lag==0:
                x=history_action
                y=history_outcome
            else:
                x=history_action[:,:-lag]
                y=history_outcome[:,lag:]
            correlations.append((x*y).mean(dim=1,keepdim=True))
        data=torch.cat((h.reshape(-1,16),*correlations),dim=1)
        return self.policy(data).reshape(-1)


class ConventionalGRUPolicy(nn.Module):
    """No explicit causal coefficients; equal factual history, 431 params."""
    def __init__(self):
        super().__init__()
        self.recurrent=nn.GRU(2,10,batch_first=True)
        self.readout=nn.Linear(10,1)

    def forward(self,h):
        if h.ndim!=3 or h.shape[1:]!=(HISTORY,2):
            raise ValueError("P14_GRU_INPUT_SHAPE")
        return self.readout(self.recurrent(h)[0][:,-1,:]).reshape(-1)


class TrainingWorld:
    """Known synthetic DEV family; no latent labels passed to policies."""
    def __init__(self,seed):
        r=random.Random(seed)
        self.lag=r.randrange(4)
        self.base=r.choice((-1,1))
        self.gain=r.uniform(.38,.57)
        self.bias=r.uniform(-.075,.075)
        self.change_point=r.randrange(18,35)
        self.noise=[r.gauss(0.,.02) for _ in range(HISTORY+TRAIN_HORIZON)]
        self.actions=[]
        self.factual=[]

    def realize(self,action):
        if action not in (-1,1):
            raise ValueError("P14_ACTION_NOT_FACTUAL")
        t=len(self.actions)
        if t>=len(self.noise):
            raise ValueError("P14_TRAINING_SOURCE_EXHAUSTED")
        self.actions.append(action)
        old=self.actions[t-self.lag] if t>=self.lag else 0
        sign=self.base*(1 if t<self.change_point else -1)
        y=self.bias+sign*self.gain*old+self.noise[t]
        self.factual.append(float(y))
        return float(y)


def completed_history(histories):
    if not histories or any(len(h)<HISTORY for h in histories):
        raise ValueError("P14_INCOMPLETE_OBSERVATION")
    return torch.tensor([h[-HISTORY:] for h in histories],
                        dtype=torch.float32)


def world_batch(step,seed):
    if seed not in INIT_SEEDS or type(step) is not int or (
        step<0 or step>=TRAIN_UPDATES):
        raise ValueError("P14_TRAIN_WORLD_BUDGET")
    worlds=[TrainingWorld(TRAIN_WORLD_SEED_BASE+
             (step*BATCH_WORLDS+j)*97) for j in range(BATCH_WORLDS)]
    histories=[]
    for w in worlds:
        h=[]
        for i in range(HISTORY):
            action=1 if i%2 else -1
            h.append((action,w.realize(action)))
        histories.append(h)
    return worlds,histories


def reward_alignment_negative(returns):
    """Same reward histogram but incorrect action/world/time attribution."""
    if returns.ndim!=2 or returns.shape!=(
        TRAIN_HORIZON,BATCH_WORLDS):
        raise ValueError("P14_NEGATIVE_REWARD_SHAPE")
    return torch.roll(returns,shifts=(TRAIN_HORIZON//2,1),dims=(0,1))


def one_gradient_step(model,opt,step,seed,*,corrupt=False):
    worlds,histories=world_batch(step,seed)
    logs=[]
    entropies=[]
    rewards=[]
    for _ in range(TRAIN_HORIZON):
        h=completed_history(histories)
        logits=model(h)
        dist=torch.distributions.Bernoulli(logits=logits)
        sampled=dist.sample()
        logs.append(dist.log_prob(sampled))
        entropies.append(dist.entropy())
        actual_actions=[int(x)*2-1 for x in sampled.detach().tolist()]
        actual_rewards=[]
        for j,(w,action) in enumerate(zip(worlds,actual_actions)):
            outcome=w.realize(action)
            actual_rewards.append(outcome)
            histories[j].append((action,outcome))
        rewards.append(torch.tensor(actual_rewards,dtype=torch.float32))
    r=torch.stack(rewards)
    returns=torch.zeros_like(r)
    running=torch.zeros(BATCH_WORLDS)
    for t in reversed(range(TRAIN_HORIZON)):
        running=r[t]+DISCOUNT*running
        returns[t]=running
    if corrupt:
        returns=reward_alignment_negative(returns)
    advantage=(returns-returns.mean(dim=1,keepdim=True))
    advantage=advantage/(advantage.std().clamp(min=1e-4))
    logp=torch.stack(logs)
    entropy=torch.stack(entropies)
    loss=-(logp*advantage.detach()).mean()-(ENTROPY_COEF*entropy.mean())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
    opt.step()
    return {"loss":float(loss.detach()),
            "true_factual_mean_reward":float(r.mean()),
            "all_actions_observed":all(len(w.actions)==HISTORY+TRAIN_HORIZON
                                       for w in worlds),
            "factual_reward_count":int(r.numel())}


def state_digest(model):
    digest=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        digest.update(k.encode())
        digest.update(v.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_model(kind,seed,updates=TRAIN_UPDATES):
    if kind not in TRAIN_CLASSES or seed not in INIT_SEEDS or (
        type(updates) is not int or not 1<=updates<=TRAIN_UPDATES):
        raise ValueError("P14_UNREGISTERED_POLICY_TRAINING")
    torch.manual_seed(seed)
    cls=ConventionalGRUPolicy if kind=="gru" else CausalFeaturePolicy
    model=cls()
    before=state_digest(model)
    optim=torch.optim.AdamW(model.parameters(),lr=TRAIN_LR)
    curve=[]
    start=time.perf_counter()
    for step in range(updates):
        row=one_gradient_step(model,optim,step,seed,
                              corrupt=(kind=="misaligned"))
        if (step+1)%12==0 or step+1==updates:
            curve.append({"update":step+1,
                          "true_training_reward_per_action":row[
                              "true_factual_mean_reward"],
                          "reinforce_loss":row["loss"]})
    model.eval()
    restored=cls()
    restored.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    restored.eval()
    sentinel=torch.linspace(-.5,.5,16).reshape(1,8,2)
    with torch.no_grad():
        same=torch.equal(model(sentinel),restored(sentinel))
    return restored,{
        "actor":kind,"initialization":seed,
        "parameter_count":sum(p.numel() for p in model.parameters()),
        "actual_policy_gradient_updates":updates,
        "training_factual_actions":updates*BATCH_WORLDS*TRAIN_HORIZON,
        "training_world_cohort_digest":hashlib.sha256(json.dumps(
            [TRAIN_WORLD_SEED_BASE+(i*BATCH_WORLDS+j)*97
             for i in range(updates) for j in range(BATCH_WORLDS)]
        ).encode()).hexdigest(),
        "trained_weight_sha256":state_digest(restored),
        "gradient_changed_weights":state_digest(restored)!=before,
        "in_process_frozen_object_restoration_bit_identical":same,
        "training_curve":curve,
        "frozen_factual_validation_plateau_qualified":False,
        "optimizer":"REINFORCE actual realized reward-to-go + entropy, AdamW",
        "negative_reward_alignment_corrupted":kind=="misaligned",
        "training_wall_seconds":time.perf_counter()-start,
    }


def eval_world(family,seed):
    if family in ("instant","lagged"):
        if seed not in OLD_SEEDS:
            raise ValueError("P14_OLD_DEV_SEED_MISMATCH")
        return OldWorld(family,seed)
    if seed not in LAG_SEEDS:
        raise ValueError("P14_LAG_DEV_SEED_MISMATCH")
    return CausalDelayWorld(family,seed)


def factual_step(world,t,action,previous):
    if isinstance(world,CausalDelayWorld):
        return world.realize(t,action)
    return world.realize(t,action,previous)


def evaluate_episode(family,seed,model_seed,models):
    if family not in FAMILIES or model_seed not in INIT_SEEDS:
        raise ValueError("P14_EVALUATION_SOURCE_DENIED")
    out={}
    prefixes=[]
    for arm in ARMS:
        world=eval_world(family,seed)
        h=[]
        prev=0
        for t in range(OBS_PREFIX):
            a=1 if t%2 else -1
            y=factual_step(world,t,a,prev)
            h.append((a,y))
            prev=a
        prefixes.append(tuple(h))
        online=FactualLagEstimator("full",adapt=True) if arm in (
            "p13_active","p13_passive") else None
        if online:
            for t,(a,y) in enumerate(h):
                online.observe(a,[v for v,_ in h[:t]],y,scored=False)
        rng=random.Random(seed+model_seed*9001+70019)
        probes=[]
        decisions=[]
        realized=[]
        for i in range(DECISIONS):
            t=OBS_PREFIX+i
            previous=[a for a,_ in h]
            if arm in ("trained_causal","trained_gru","reward_misaligned"):
                # Same predetermined forced action, in the same 16 probe slots.
                if i%PROBE_INTERVAL==0:
                    action=rng.choice((-1,1))
                    probes.append({"index":i,"action":action})
                else:
                    m=models[arm]
                    with torch.no_grad():
                        logits=m(completed_history([h]))
                    action=1 if float(logits.item())>=0 else -1
            elif online:
                mode="active" if arm=="p13_active" else "passive"
                action,probing,info=action_from_model(
                    online,rng,previous,i,policy=mode)
                if probing:
                    probes.append({"index":i,"action":action})
            else:
                action=rng.choice((-1,1))
                if i%PROBE_INTERVAL==0:
                    probes.append({"index":i,"action":action})
            # Exactly one true consequence, revealed only AFTER choosing.
            y=factual_step(world,t,action,prev)
            if online:
                online.observe(action,previous,y,scored=True)
            h.append((action,y))
            prev=action
            decisions.append(action)
            realized.append(y)
        out[arm]={
            "mean_realized_reward":sum(realized)/DECISIONS,
            "first_24":sum(realized[:24])/24,
            "reversed_24":sum(realized[24:48])/24,
            "old_regime_revisit_16":sum(realized[48:])/16,
            "decisions":decisions,
            "only_factual_outcomes":realized,
            "probes":probes,
            "scored_actions":DECISIONS,
            "online_factual_updates":online.scored_updates if online else 0,
            "trace_sha256":hashlib.sha256(json.dumps(
                [decisions,realized],separators=(",",":"),allow_nan=False
            ).encode()).hexdigest()
        }
    checks={
        "same_eight_completed_prior_factual_observations":
            len(set(prefixes))==1,
        "all_same_sixtyfour_action_opportunities":
            all(v["scored_actions"]==DECISIONS for v in out.values()),
        "exact_same_sixteen_probe_positions":
            all([x["index"] for x in v["probes"]]==list(range(0,64,4))
                for v in out.values()),
        "trained_actors_same_sixteen_probe_actions":
            len(set(tuple(x["action"] for x in out[a]["probes"])
                    for a in ("trained_causal","trained_gru",
                              "reward_misaligned")))==1,
        "old_online_controls_64_true_updates":
            out["p13_active"]["online_factual_updates"]==
            out["p13_passive"]["online_factual_updates"]==64,
        "trained_actor_weights_frozen_during_evaluation":
            all(out[a]["online_factual_updates"]==0 for a in
                ("trained_causal","trained_gru","reward_misaligned")),
        "all_observed_consequences_finite":all(math.isfinite(y)
            for v in out.values() for y in v["only_factual_outcomes"]),
        "sham_exact_identical_reward_all_actions":
            (all(v["only_factual_outcomes"]==
                 out["random"]["only_factual_outcomes"]
                 for v in out.values()) if family=="sham_no_action" else True)
    }
    return {"family":family,"dev_world_seed":seed,"model_seed":model_seed,
            "integrity":"PASS" if all(checks.values()) else "FAIL",
            "checks":checks,"policies":out}


def run():
    only_dev()
    torch.set_num_threads(1)
    start=time.perf_counter()
    records=[]
    for seed in INIT_SEEDS:
        models={}
        trained={}
        for kind,arm in (("causal","trained_causal"),("gru","trained_gru"),
                         ("misaligned","reward_misaligned")):
            models[arm],trained[arm]=train_model(kind,seed)
        local=[]
        for family in FAMILIES:
            eval_seeds=OLD_SEEDS if family in ("instant","lagged") else LAG_SEEDS
            for w in eval_seeds:
                local.append(evaluate_episode(family,w,seed,models))
        params=[trained[a]["parameter_count"] for a in (
            "trained_causal","trained_gru")]
        checks={
            "all_three_policies_received_96_real_gradient_updates":
                all(r["actual_policy_gradient_updates"]==TRAIN_UPDATES
                    and r["gradient_changed_weights"]
                    for r in trained.values()),
            "same_training_factual_world_cohort_fingerprint":
                len(set(r["training_world_cohort_digest"]
                        for r in trained.values()))==1,
            "parameters_near_equal_within_5pct":
                (max(params)-min(params))/max(params)<.05,
            "independent_restored_weight_object_bit_identical":
                all(r["in_process_frozen_object_restoration_bit_identical"]
                    for r in trained.values()),
            "all_six_families_four_world_seeds":len(local)==24,
            "all_factual_closed_loop_actions_fully_qualified":
                all(x["integrity"]=="PASS" for x in local),
            "negative_receives_wrong_temporal_world_reward_assignment":
                trained["reward_misaligned"][
                    "negative_reward_alignment_corrupted"] is True,
        }
        averages={family:{arm:sum(q["policies"][arm]["mean_realized_reward"]
                for q in local if q["family"]==family)/4
                for arm in ARMS} for family in FAMILIES}
        hypotheses={family:{
            "causal_beats_equal_gradient_gru_by_0_05":
                v["trained_causal"]>=v["trained_gru"]+.05,
            "causal_beats_reward_corrupted_by_0_05":
                v["trained_causal"]>=v["reward_misaligned"]+.05,
            "causal_beats_random_by_0_05":
                v["trained_causal"]>=v["random"]+.05,
            "causal_beats_p13_active_by_0_05":
                v["trained_causal"]>=v["p13_active"]+.05,
            "GRU_beats_random_by_0_05":
                v["trained_gru"]>=v["random"]+.05,
        } for family,v in averages.items()}
        records.append({
            "model_initialization":seed,
            "status":"PASS_DEV_INTEGRITY" if all(checks.values())
                     else "FAIL_DEV_INTEGRITY",
            "training_records":trained,
            "integrity_checks":checks,
            "family_avg_factual_rewards":averages,
            "directional_hypotheses":hypotheses,
            "all_actual_factual_eval_episodes":local
        })
    elapsed=time.perf_counter()-start
    counts={f:{k:sum(int(x["directional_hypotheses"][f][k])
                    for x in records)
               for k in records[0]["directional_hypotheses"][f]}
            for f in FAMILIES}
    integrity=(len(records)==3 and all(r["status"]=="PASS_DEV_INTEGRITY"
               for r in records) and elapsed<MAX_SECONDS)
    return {"schema":SCHEMA,
            "status":"PASS_REWARD_TRAINED_INTERVENTION_DEV_NOT_SCIENCE"
                     if integrity else "FAIL_INTERVENTION_DEV_INTEGRITY",
            "scientific_first_use":"NOT_RUN",
            "official_scientific_seeds_consumed":[],
            "trained_actor_and_conventional_real_optimizer_steps":
                len(INIT_SEEDS)*len(TRAIN_CLASSES)*TRAIN_UPDATES,
            "actual_training_factual_decisions":
                len(INIT_SEEDS)*len(TRAIN_CLASSES)*TRAIN_UPDATES*
                BATCH_WORLDS*TRAIN_HORIZON,
            "actual_eval_factual_decisions":
                len(INIT_SEEDS)*len(FAMILIES)*4*len(ARMS)*DECISIONS,
            "trained_models_per_initialization":3,
            "equal_sixteen_eval_probe_slots":True,
            "source_novelty_authentic_science":False,
            "trained_baseline_plateau_independently_verified":False,
            "cold_durable_process_restart_tested":False,
            "network_weights_promoted":False,
            "FORGE_integration":False,
            "model_initializations":list(INIT_SEEDS),
            "development_families":list(FAMILIES),
            "directional_support_count_of_three":counts,
            "records":records,
            "whole_process_wall_seconds":elapsed,
            "limits":"Both learned policies trained on the same synthetic procedural generator but on distinct on-policy factual outcomes from their selected actions. Synthetic source and evaluation rules are known and neither is independently authenticated unseen science. The causality features and eight-event history are engineered priors, not native independent cognition; REINFORCE trained action-selection is real but not scientifically superior until stable fully trained controls, matched FLOPs/whole cost, heldout source birth, cold process retention and independent first-use. The corrupted-reward negative has identical optimizer work but deliberately INVALID factual target pairing."}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--receipt",type=Path,required=True)
    args=ap.parse_args()
    report=run()
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "status":report["status"],
        "scientific_first_use":report["scientific_first_use"],
        "training_factual_decisions":report["actual_training_factual_decisions"],
        "eval_factual_decisions":report["actual_eval_factual_decisions"],
        "support":report["directional_support_count_of_three"],
        "reward_by_init":[{
            "seed":x["model_initialization"],
            "mean_rewards":x["family_avg_factual_rewards"]}
            for x in report["records"]],
        "whole_seconds":report["whole_process_wall_seconds"]
    },sort_keys=True))
    return 0 if report["status"]=="PASS_REWARD_TRAINED_INTERVENTION_DEV_NOT_SCIENCE" else 1


if __name__=="__main__":
    raise SystemExit(main())
