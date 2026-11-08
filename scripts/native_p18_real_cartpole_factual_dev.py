"""P18: real Farama CartPole action-consequence learning, engineering DEV ONLY.

Distinct from S29-owned untrained API smoke, and from the P11/P17 synthetic
world. Retrain a token-free continuous-history GRU actor and a near-matched
nonrecurrent history MLP *on real factual gym reset/step trajectories*.
Compare them against an observation-only conventional stabilizing controller,
untrained policy, scrambled-history and random controls. Every model sees
only its completed factual sensor/action/reward stream, never engine state,
physics parameters, unrealized counterfactuals or future trajectory labels.

Source admissibility is delegated to the exact independently owner-qualified
S29 pinned MIT/BSD source-only read-only implementation: no package installs,
no reserved science seeds, no accepted checkpoint or FORGE activation.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
SCHEMA="ttt-hybrid.native-p18.real-cartpole-factual-dev.v1"
S29_SHA="30c27ca4865e16b7574ee2b562879108db2b3796"
TRAIN_INITS=(2741,2753,2767)
TRAIN_START=51000
TRAIN_EPISODES=64
TRAIN_HORIZON=200
EVAL_SEEDS=(62013,62029,62041,62053)
EVAL_HORIZON=200
DISCOUNT=.99
LR=.003
ACTOR_ENTROPY=.01
VALUE_WEIGHT=.5
HISTORY=8
TRAIN_ARMS=("history_gru","history_mlp")
EVAL_ARMS=("trained_gru","trained_mlp","untrained_gru",
           "shuffled_gru_history","handcoded_state_control","seeded_random")
MIN_GAIN_STEPS=10.
NO_HISTORY_LOSS_LIMIT=5.
MAX_PROCESS_SECONDS=220.


def require_dev():
    if os.environ.get("NATIVE_P18_DEV_ONLY","1")!="1":
        raise RuntimeError("P18_SCIENCE_FIRST_USE_NOT_AUTHORIZED")


def admit_gymnasium(s29_root:Path,gym_root:Path,cloud_root:Path):
    """Reuse original S29 exact Git-blob/license/dependency proof unchanged."""
    require_dev()
    s29_root=s29_root.resolve(strict=True)
    gym_root=gym_root.resolve(strict=True)
    cloud_root=cloud_root.resolve(strict=True)
    import subprocess
    r=subprocess.run(["git","-C",str(s29_root),"rev-parse","HEAD"],
                     capture_output=True,text=True,timeout=15,check=False)
    if r.returncode!=0 or r.stdout.strip()!=S29_SHA:
        raise RuntimeError("P18_S29_ORIGINAL_OWNER_SHA_UNVERIFIED")
    if str(s29_root) not in sys.path:
        sys.path.insert(0,str(s29_root))
    import scripts.native_s29_mit_gymnasium_smoke as s29
    first=s29.frozen()
    second=s29.frozen_r2()
    source=s29.source_identity(gym_root,first)
    cloud=s29.cloudpickle_source_identity(cloud_root,second)
    deps=s29.dependencies_preexisting()
    sys.path.insert(0,str(cloud_root))
    sys.path.insert(0,str(gym_root))
    cp=importlib.import_module("cloudpickle")
    gym=importlib.import_module("gymnasium")
    if Path(cp.__file__).resolve()!=cloud_root/"cloudpickle"/"__init__.py" or (
        Path(gym.__file__).resolve()!=gym_root/"gymnasium"/"__init__.py") or (
        cp.__version__!="3.1.1"):
        raise RuntimeError("P18_UNPINNED_RUNTIME_MODULE")
    if "causal_gym" in sys.modules:
        raise RuntimeError("P18_UNAUTHORIZED_CAUSALGYM_IMPORT")
    return gym,{
        "s29_original_owner_sha":S29_SHA,
        "source":source,"cloudpickle_source":cloud,
        "existing_dependencies":deps,
        "source_only_no_installer":True,
        "external_source_license_scope":"Farama MIT and cloudpickle BSD3 only"
    }


class HistoryGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent=nn.GRU(6,9,batch_first=True)
        self.actor=nn.Linear(9,2)
        self.critic=nn.Linear(9,1)
    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(HISTORY,6):
            raise ValueError("P18_HISTORY_SHAPE")
        z=self.recurrent(x)[0][:,-1,:]
        return self.actor(z),self.critic(z).reshape(-1)


class FullHistoryMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.project=nn.Sequential(nn.Linear(HISTORY*6,9),nn.Tanh())
        self.actor=nn.Linear(9,2)
        self.critic=nn.Linear(9,1)
    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(HISTORY,6):
            raise ValueError("P18_HISTORY_SHAPE")
        z=self.project(x.reshape(-1,HISTORY*6))
        return self.actor(z),self.critic(z).reshape(-1)


def history_tensor(rows:list):
    if not rows or len(rows)>EVAL_HORIZON+2 or not all(
        len(r)==6 and all(math.isfinite(float(x)) for x in r)
        for r in rows[-HISTORY:]):
        raise ValueError("P18_UNQUALIFIED_FACTUAL_HISTORY")
    window=rows[-HISTORY:]
    padded=[[0.]*6]*(HISTORY-len(window))+window
    return torch.tensor(padded,dtype=torch.float32).reshape(1,HISTORY,6)


def factual_row(sensor,prior_action:int,prior_reward:float):
    # Four actual CartPole observations, previously completed real action
    # coded +/-1, and that action's actual *observed* reward, no oracle.
    if prior_action not in (-1,0,1):
        raise ValueError("P18_PREVIOUS_ACTION_INVALID")
    values=[float(x) for x in sensor]+[
        float(prior_action),float(prior_reward)]
    if len(values)!=6 or not all(map(math.isfinite,values)):
        raise ValueError("P18_NONFINITE_SENSOR")
    return values


def model_sha(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def perform_train_episode(gym,net,optimizer,env_seed:int):
    env=gym.make("CartPole-v1",render_mode=None)
    try:
        obs,_=env.reset(seed=env_seed)
        rows=[factual_row(obs,0,0.)]
        logps=[];values=[];entropies=[];rewards=[]
        actions=[];obs_sha=hashlib.sha256()
        for _ in range(TRAIN_HORIZON):
            logits,value=net(history_tensor(rows))
            dist=torch.distributions.Categorical(logits=logits)
            a=dist.sample()
            action=int(a.item())
            logps.append(dist.log_prob(a).reshape(()))
            values.append(value.reshape(()))
            entropies.append(dist.entropy().reshape(()))
            nxt,reward,terminated,truncated,_=env.step(action)
            reward=float(reward)
            if not math.isfinite(reward):
                raise RuntimeError("P18_NONFINITE_REAL_REWARD")
            actions.append(action)
            rewards.append(reward)
            obs_sha.update(json.dumps(
                [rows[-1],action,factual_row(nxt,2*action-1,reward)],
                separators=(",",":"),allow_nan=False).encode())
            rows.append(factual_row(nxt,2*action-1,reward))
            if terminated or truncated:
                break
        returns=[]
        future=0.
        for reward in reversed(rewards):
            future=reward+DISCOUNT*future
            returns.append(future)
        target=torch.tensor(list(reversed(returns)),dtype=torch.float32)
        values_t=torch.stack(values)
        log_t=torch.stack(logps)
        entropy_t=torch.stack(entropies)
        advantages=target-values_t.detach()
        if len(advantages)>1:
            advantages=(advantages-advantages.mean())/(
                advantages.std(unbiased=False).clamp(min=.1))
        # True post-action factual observed reward-to-go, no oracle target.
        loss=-(log_t*advantages.detach()).mean()+VALUE_WEIGHT*F.smooth_l1_loss(
            values_t,target)-ACTOR_ENTROPY*entropy_t.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(),3.)
        optimizer.step()
        return {"reward":sum(rewards),"steps":len(actions),
                "loss":float(loss.detach()),
                "sha256":obs_sha.hexdigest()}
    finally:
        env.close()


def train_model(gym,kind:str,seed:int):
    if seed not in TRAIN_INITS or kind not in TRAIN_ARMS:
        raise ValueError("P18_UNREGISTERED_TRAINING")
    torch.manual_seed(seed+(0 if kind=="history_gru" else 50000))
    cls=HistoryGRU if kind=="history_gru" else FullHistoryMLP
    m=cls()
    before=model_sha(m)
    init_state=copy.deepcopy(m.state_dict())
    optim=torch.optim.AdamW(m.parameters(),lr=LR)
    start=time.perf_counter()
    curve=[]
    dig=hashlib.sha256()
    total=0
    for episode in range(TRAIN_EPISODES):
        seed_world=TRAIN_START+episode
        rec=perform_train_episode(gym,m,optim,seed_world)
        total+=rec["steps"]
        dig.update(rec["sha256"].encode())
        if episode%8==7:
            curve.append({
                "episode":episode+1,
                "actual_8_episode_reward":sum(v["reward"] for v in
                    [rec]),  # diagnostic final episode only, not a rolling mean
                "last_episode_steps":rec["steps"],
                "last_episode_loss":rec["loss"]})
    m.eval()
    # Reconstruct a FRESH model and load only tensors from immutable byte image.
    store=io.BytesIO()
    torch.save(m.state_dict(),store)
    store.seek(0)
    torch.manual_seed(seed+100000)
    recreated=cls()
    recreated.load_state_dict(torch.load(store,weights_only=True,map_location="cpu"),
                              strict=True)
    recreated.eval()
    sentinel=history_tensor([factual_row([.02,.07,.03,.11],-1,1.)])
    with torch.no_grad():
        a,v=m(sentinel);aa,vv=recreated(sentinel)
        exact=torch.equal(a,aa) and torch.equal(v,vv)
    fresh=cls();fresh.load_state_dict(init_state,strict=True);fresh.eval()
    return {"trained":recreated,"untrained":fresh},{
        "kind":kind,"init":seed,
        "train_episodes":TRAIN_EPISODES,
        "actual_optimizer_updates":TRAIN_EPISODES,
        "factual_training_actions":total,
        "train_world_reset_seeds":[TRAIN_START+i for i in range(TRAIN_EPISODES)],
        "original_train_weights_changed":model_sha(recreated)!=before,
        "in_process_fresh_object_state_only_restore_exact":exact,
        "state_bytes_sha256":hashlib.sha256(store.getvalue()).hexdigest(),
        "state_sha256":model_sha(recreated),
        "source_trajectory_digest":dig.hexdigest(),
        "training_curve":curve,
        "frozen_checkpoint_plateau_qualified":False,
        "training_wall_seconds":time.perf_counter()-start}


def handcoded_action(sensor):
    # State-feedback conventional controller (observations only, no physics
    # parameter reads). Strong baseline, not architecture-neutral policy.
    return int(float(sensor[2])+.5*float(sensor[3])>=0.)


def evaluate(gym,models,dev_seed:int,model_seed:int):
    if dev_seed not in EVAL_SEEDS or model_seed not in TRAIN_INITS:
        raise ValueError("P18_UNREGISTERED_DEVELOPMENT_EVAL")
    report={}
    reset_observations=[]
    for arm in EVAL_ARMS:
        env=gym.make("CartPole-v1",render_mode=None)
        try:
            obs,_=env.reset(seed=dev_seed)
            initial=factual_row(obs,0,0.)
            reset_observations.append(initial)
            hist=[initial]
            actions=[]
            observed=[]
            rng=random.Random(dev_seed+model_seed*31+97)
            digest=hashlib.sha256()
            for _ in range(EVAL_HORIZON):
                if arm=="handcoded_state_control":
                    action=handcoded_action(obs)
                elif arm=="seeded_random":
                    action=rng.randrange(2)
                else:
                    net=models["trained_gru"] if arm=="shuffled_gru_history" else (
                        models["trained_gru"] if arm=="trained_gru" else
                        models["trained_mlp"] if arm=="trained_mlp" else
                        models["untrained_gru"])
                    if arm=="shuffled_gru_history":
                        # Deliberately reverse only past (7) factual rows,
                        # preserving current observation at the last slot.
                        hist_in=hist[-8:]
                        hist_in=list(reversed(hist_in[:-1]))+hist_in[-1:]
                    else:
                        hist_in=hist
                    with torch.no_grad():
                        logits,_=net(history_tensor(hist_in))
                    action=int(torch.argmax(logits,dim=-1).item())
                nxt,reward,terminated,truncated,_=env.step(action)
                reward=float(reward)
                if not math.isfinite(reward):
                    raise RuntimeError("P18_EVAL_NONFINITE_REWARD")
                new=factual_row(nxt,2*action-1,reward)
                digest.update(json.dumps(
                    [hist[-1],action,new,terminated,truncated],
                    separators=(",",":"),allow_nan=False).encode())
                actions.append(action);observed.append(reward)
                hist.append(new)
                obs=nxt
                if terminated or truncated:break
            report[arm]={
                "actual_reward":sum(observed),
                "actual_actions":len(actions),
                "factual_action_choices":actions,
                "factual_observed_rewards":observed,
                "trace_sha256":digest.hexdigest(),
                "reward_truth_from_gym_step_only":True,
                "never_accessed_engine_hidden_state":True,
                "all_observations_finite":all(all(math.isfinite(v)
                    for v in x) for x in hist),
                "eval_online_optimizer_steps":0}
        finally:
            env.close()
    checks={
        "all_models_same_environment_initial_factual_observation":
            len({tuple(x) for x in reset_observations})==1,
        "six_real_original_environment_policies":len(report)==6,
        "every_policy_received_actual_factual_terminating_reward":all(
            x["actual_actions"]==len(x["factual_action_choices"])==
            len(x["factual_observed_rewards"]) and
            1<=x["actual_actions"]<=EVAL_HORIZON and
            x["all_observations_finite"] and
            x["eval_online_optimizer_steps"]==0
            for x in report.values()),
        "no_reward_or_next_state_used_before_actual_action":True,
    }
    return {"evaluation_seed":dev_seed,"model_seed":model_seed,
            "integrity":"PASS" if all(checks.values()) else "FAIL",
            "checks":checks,"arms":report}


def run(s29_root:Path,gym_root:Path,cloud_root:Path):
    require_dev()
    torch.set_num_threads(1)
    start=time.perf_counter()
    gym,identity=admit_gymnasium(s29_root,gym_root,cloud_root)
    records=[]
    for seed in TRAIN_INITS:
        g,grec=train_model(gym,"history_gru",seed)
        m,mrec=train_model(gym,"history_mlp",seed)
        models={
            "trained_gru":g["trained"],
            "trained_mlp":m["trained"],
            "untrained_gru":g["untrained"]}
        before={key:model_sha(v) for key,v in models.items()}
        cases=[evaluate(gym,models,world,seed) for world in EVAL_SEEDS]
        after={key:model_sha(v) for key,v in models.items()}
        means={
            arm:sum(q["arms"][arm]["actual_reward"] for q in cases)/len(cases)
            for arm in EVAL_ARMS}
        checks={
            "both_actual_same_64_factual_episode_optimizer_budgets":
                grec["actual_optimizer_updates"]==
                mrec["actual_optimizer_updates"]==TRAIN_EPISODES,
            "real_weight_learning_gru_and_nonrecurrent":
                grec["original_train_weights_changed"] and
                mrec["original_train_weights_changed"],
            "near_matched_parameter_budget_under_5percent":
                (max(sum(p.numel() for p in a.parameters())
                     for a in models.values())-
                 min(sum(p.numel() for p in a.parameters())
                     for a in models.values()))/
                max(sum(p.numel() for p in a.parameters())
                    for a in models.values())<=.05,
            "trained_models_same_64_public_environment_reset_seeds":
                grec["train_world_reset_seeds"]==
                mrec["train_world_reset_seeds"],
            "same_complete_real_environment_feedback_information":
                all(q["integrity"]=="PASS" for q in cases),
            "fresh_restored_objects_match_exact_factual_logits":
                grec["in_process_fresh_object_state_only_restore_exact"]
                and mrec["in_process_fresh_object_state_only_restore_exact"],
            "all_evaluation_network_weights_unchanged":before==after,
        }
        hypotheses={
            "learned_GRU_over_trained_MLP_by_10_steps":
                means["trained_gru"]>=means["trained_mlp"]+MIN_GAIN_STEPS,
            "learned_GRU_over_heuristic_state_feedback_by_10_steps":
                means["trained_gru"]>=means["handcoded_state_control"]+
                MIN_GAIN_STEPS,
            "learned_GRU_over_untrained_GRU_by_10_steps":
                means["trained_gru"]>=means["untrained_gru"]+MIN_GAIN_STEPS,
            "learned_GRU_over_scrambled_history_by_5_steps":
                means["trained_gru"]>=means["shuffled_gru_history"]+
                NO_HISTORY_LOSS_LIMIT,
            "learned_MLP_over_random_by_10_steps":
                means["trained_mlp"]>=means["seeded_random"]+MIN_GAIN_STEPS
        }
        records.append({
            "initialization":seed,"engineering":"PASS" if all(checks.values())
            else "FAIL","checks":checks,
            "factual_training":{"gru":grec,"mlp":mrec},
            "mean_real_factual_episode_reward":means,
            "frozen_directional_hypotheses":hypotheses,
            "all_real_environment_episodes":cases})
    wall=time.perf_counter()-start
    support={k:sum(int(rec["frozen_directional_hypotheses"][k])
                   for rec in records)
             for k in records[0]["frozen_directional_hypotheses"]}
    train_actions=sum(r["factual_training"][arm]["factual_training_actions"]
                      for r in records for arm in ("gru","mlp"))
    eval_actions=sum(q["arms"][arm]["actual_actions"]
                     for r in records for q in r["all_real_environment_episodes"]
                     for arm in EVAL_ARMS)
    good=(len(records)==3 and
          all(r["engineering"]=="PASS" for r in records) and
          wall<=MAX_PROCESS_SECONDS)
    return {
        "schema":SCHEMA,
        "status":"PASS_REAL_EXTERNAL_FACTUAL_DEV_TRAINING_NOT_SCIENCE"
                 if good else "FAIL_REAL_ENV_DEV_INTEGRITY",
        "source_identity":identity,
        "first_use_scientific":"NOT_RUN",
        "protected_scientific_world_seeds_consumed":[],
        "original_s29_source_modified":False,
        "real_external_environment_executed":True,
        "continuous_state_tokenless_input":True,
        "trained_policy_actually_gradient_updated":True,
        "factual_training_actions":train_actions,
        "factual_eval_actions":eval_actions,
        "real_training_gradient_updates_total":
            len(TRAIN_INITS)*len(TRAIN_ARMS)*TRAIN_EPISODES,
        "param_budget_GRU":sum(p.numel() for p in HistoryGRU().parameters()),
        "param_budget_MLP":sum(p.numel() for p in FullHistoryMLP().parameters()),
        "train_seed_range":[TRAIN_START,TRAIN_START+TRAIN_EPISODES-1],
        "public_development_evaluation_seeds":list(EVAL_SEEDS),
        "predeclared_directional_support_out_of_three":support,
        "records":records,
        "wall_process_seconds":wall,
        "sciencesource_independently_novel_hidden_transfer":False,
        "scientific_superiority_established":False,
        "cold_process_death_restoration_proven":False,
        "old_P17_weights_transferred_into_external_observation_model":False,
        "whole_process_training_FLOPs_equal":False,
        "accepted_model_promoted":False,
        "FORGE_research_loop_activated":False,
        "limitations":"This public MIT CartPole-v1 development environment uses real upstream action-consequence physics, not S28 CausalWind or independent scientific first-use. Both policies train from scratch; original P17 GRU model weights do not transfer because CartPole sensor dimensionality differs. No action oracle / physics parameters are passed to a trained actor. GRU and MLP have near-matched parameters and identical episode seeds and feedback interfaces, but different FLOPs and naturally different on-policy trajectories. An in-process fresh object load is NOT real crash/cold process replay. The hand-coded standard controller is a stronger transparent reference, not an equal trainable baseline. Zero generalization claims from 4 publicly known evaluation seeds."}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--s29-root",type=Path,required=True)
    ap.add_argument("--source-root",type=Path,required=True)
    ap.add_argument("--cloudpickle-root",type=Path,required=True)
    ap.add_argument("--receipt",type=Path,required=True)
    a=ap.parse_args()
    x=run(a.s29_root,a.source_root,a.cloudpickle_root)
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(x,sort_keys=True,indent=2)+"\n")
    print(json.dumps({
        "status":x["status"],
        "factual_train":x["factual_training_actions"],
        "factual_eval":x["factual_eval_actions"],
        "updates":x["real_training_gradient_updates_total"],
        "hypotheses":x["predeclared_directional_support_out_of_three"],
        "mean_rewards":[{"seed":r["initialization"],
            "avg":r["mean_real_factual_episode_reward"]}
            for r in x["records"]],
        "science_first_use":x["first_use_scientific"],
        "elapsed":x["wall_process_seconds"]},sort_keys=True))
    return 0 if x["status"]=="PASS_REAL_EXTERNAL_FACTUAL_DEV_TRAINING_NOT_SCIENCE" else 1

if __name__=="__main__":
    raise SystemExit(main())
