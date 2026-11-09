"""P35 true-source action-consequence prediction auxiliary vs shuffled attribution.

Original pinned Gymnasium MountainCar-v0, source goal, time-limit and -1 reward.
Original velocity is masked at every policy input; the model sees public
position, its own prior action/reward and finite differences of past positions.
No tokenizer, transformer, model promotion, or protected scientific first use.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
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
sys.path.insert(0,str(ROOT))
from scripts import native_p20_reward_only_ppo_dev as p20

SCHEMA="ttt-hybrid.native-p35.original-mountaincar-factual-auxiliary-ppo-dev.v1"
PROTOCOL=ROOT/"experiments/native_p35_mountaincar_factual_aux_frozen.json"
SEEDS=(5209,5227,5237)
TRAIN_BASE=184000
EVAL_SEEDS=(185011,185019,185029,185037,185049,185059,185071,185089)
CONDITIONS=("original","low_force_x0.75","high_force_x1.25")
FORCE_RATIOS={"original":1.,"low_force_x0.75":.75,"high_force_x1.25":1.25}
TRAIN_ARMS=("full_recurrent_ppo","factual_action_aux_ppo",
            "shuffled_action_aux_ppo","actor_head_only_ppo",
            "conventional_history_mlp_ppo","critic_only_actor_frozen")
AUX_ARMS=("factual_action_aux_ppo","shuffled_action_aux_ppo")
AUX_LOSS_WEIGHT=0.15
AUX_TARGET_CLIP=5.0
EVAL_ARMS=("initial_no_update",)+TRAIN_ARMS+(
           "two_frame_conventional_pendulum_momentum_rule","seeded_random")
ROLLOUTS=96
NUM_ENVS=8
STEPS=64
EPOCHS=3
MINIBATCH=256
HORIZON=200
H=8

class MountainHistoryGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent=nn.GRU(4,7,batch_first=True)
        self.actor=nn.Linear(7,3)
        self.critic=nn.Linear(7,1)
    def forward(self,x):
        z,_=self.recurrent(x)
        h=z[:,-1,:]
        return self.actor(h),self.critic(h).reshape(-1)

class MountainHistoryMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent=nn.Sequential(nn.Flatten(),nn.Linear(32,8),nn.Tanh())
        self.actor=nn.Linear(8,3)
        self.critic=nn.Linear(8,1)
    def forward(self,x):
        z=self.recurrent(x)
        return self.actor(z),self.critic(z).reshape(-1)

def frozen():
    d=json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if (d.get("schema")!="ttt-hybrid.native-p35.original-mountaincar-factual-action-consequence-auxiliary-frozen.v1"
        or d["phase"]!="PROSPECTIVE_FROZEN_BEFORE_IMPLEMENTATION_AND_ANY_P35_SOURCE_ACTIONS"
        or d["parent"]["sha"]!="50a90718362fac77d8f39a4037a81f5abe97aa76"
        or d["source"]["original_mountaincar_code_blob"]!="5e7e11518715197974e8df46a842f1bc66ccd88c"
        or d["source"]["protected_scientific_first_use"] is not False
        or d["training"]["new_model_initialization_seeds"]!=list(SEEDS)
        or d["training"]["arms"]!=list(TRAIN_ARMS)
        or d["training"]["train_source_seed_base"]!=TRAIN_BASE
        or d["evaluation"]["new_original_source_reset_seeds"]!=list(EVAL_SEEDS)
        or list(d["evaluation"]["conditions"])!=list(CONDITIONS)
        or d["evaluation"]["eval_arms"]!=list(EVAL_ARMS)
        or d["training"]["auxiliary_loss_coefficient"]!=AUX_LOSS_WEIGHT
        or d["training"]["auxiliary_target_clipping"]!=AUX_TARGET_CLIP
        or d["frozen_scientific_thresholds"]["no_post_result_adjustment_or_sweeps"] is not True):
        raise RuntimeError("P35_FROZEN_ORIGINAL_FACTUAL_AUX_SOURCE_PROTOCOL_DRIFT")
    return d

def guard():
    if os.environ.get("NATIVE_P35_DEV_ONLY")!="1":
        raise RuntimeError("P35_PROTECTED_SCIENCE_FIRST_USE_DENIED")
    d=frozen()
    if (p20.PPO_LR,p20.PPO_CLIP,p20.ENTROPY,p20.VALUE_COEF,
        p20.MAX_GRAD_NORM,p20.GAMMA,p20.GAE_LAMBDA)!=(
        .0015,.2,.015,.5,1.,.99,.95):
        raise RuntimeError("P35_ORIGINAL_REWARD_ONLY_PPO_SOURCE_DRIFT")
    if (sum(p.numel() for p in MountainHistoryGRU().parameters()),
        sum(p.numel() for p in MountainHistoryMLP().parameters()))!=(305,300):
        raise RuntimeError("P35_GRU_MLP_TOTAL_PARAMETER_BUDGET_DRIFT")
    if (p20.PPO_EPOCHS,p20.MINIBATCH)!=(EPOCHS,MINIBATCH):
        raise RuntimeError("P35_ORIGINAL_PPO_OPTIMIZER_CADENCE_DRIFT")
    return d

def make_model(seed,arm):
    if seed not in SEEDS or arm not in TRAIN_ARMS:
        raise ValueError("P35_UNREGISTERED_MODEL_OR_ARCHITECTURE")
    torch.manual_seed(seed)
    m=MountainHistoryMLP() if arm=="conventional_history_mlp_ppo" else MountainHistoryGRU()
    if arm in AUX_ARMS:
        # Initialize a separate identical auxiliary head without changing
        # the stochastic PPO policy's global RNG, initial weights, or first rollout.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+70000)
            m.aux_predictor=nn.Linear(7+3,1)
    m.eval()
    return m

def core_sha(model):
    """Exact SHA of actor/recurrent/critic state excluding auxiliary-only parameters."""
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        if not name.startswith("aux_predictor."):
            h.update(name.encode("utf-8"))
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def factual_acceleration(pre_action_history, factual_new_row):
    """Only public prior/current/next positions; evaluated AFTER real source step."""
    last=pre_action_history[-1]
    previous_displacement=last[3]*.07
    actual_displacement=(factual_new_row[0]-last[0])*1.2
    return max(-AUX_TARGET_CLIP,min(AUX_TARGET_CLIP,
               (actual_displacement-previous_displacement)/.0035))

def auxiliary_loss(model, x, actual_actions, factual_targets, arm, generator):
    if arm not in AUX_ARMS:
        raise ValueError("P35_UNREGISTERED_AUXILIARY_ARM")
    action_values=actual_actions.long()
    if arm=="shuffled_action_aux_ppo":
        order=torch.randperm(len(action_values),generator=generator)
        action_values=action_values[order]
    mismatch=int((action_values!=actual_actions.long()).sum())
    hidden,_=model.recurrent(x)
    z=hidden[:,-1,:]
    encoded=F.one_hot(action_values,num_classes=3).to(dtype=z.dtype)
    prediction=model.aux_predictor(torch.cat((z,encoded),dim=1)).reshape(-1)
    if prediction.shape!=factual_targets.shape:
        raise RuntimeError("P35_UNPAIRED_REAL_SOURCE_AUX_OUTCOME_SHAPE")
    return F.smooth_l1_loss(prediction,factual_targets),mismatch

def row(position,prior_action=1,reward=0.,delta=0.):
    return (float(position)/1.2,float(prior_action-1),float(reward),
            float(delta)/.07)

def past_tensor(history):
    if not history or any(len(z)!=4 for z in history):
        raise ValueError("P35_BAD_CONTINUOUS_FACTUAL_HISTORY")
    rows=list(history[-H:])
    return torch.tensor([[ (0.,0.,0.,0.) for _ in range(H-len(rows))]+rows],
                        dtype=torch.float32)

def source_env(gym,condition="original"):
    if condition not in CONDITIONS:
        raise ValueError("P35_UNREGISTERED_SOURCE_FORCE")
    env=gym.make("MountainCar-v0",render_mode=None)
    original=env.unwrapped
    if (not math.isclose(float(original.force),.001,abs_tol=1e-12)
        or not math.isclose(float(original.gravity),.0025,abs_tol=1e-12)
        or not math.isclose(float(original.goal_position),.5,abs_tol=1e-12)
        or original.action_space.n!=3):
        env.close()
        raise RuntimeError("P35_UPSTREAM_MOUNTAINCAR_SOURCE_PHYSICS_DRIFT")
    original.force=.001*FORCE_RATIOS[condition]
    return env

def actual_step(env,history,action):
    if action not in (0,1,2):
        raise ValueError("P35_NOT_REAL_THREE_ACTION")
    obs,reward,terminated,truncated,_=env.step(action)
    p=float(obs[0])
    last=float(history[-1][0])*1.2
    new=row(p,action,reward,p-last)
    if not math.isfinite(p) or not math.isfinite(float(reward)):
        raise RuntimeError("P35_NONFINITE_PHYSICAL_SOURCE_ACTION_OR_REWARD")
    return new,float(reward),bool(terminated),bool(truncated)

def model_parts(m):
    return {k:tuple(k2 for k2 in m.state_dict() if k2.startswith(k+"."))
            for k in ("recurrent","actor","critic")}

def train_arm(gym,seed,arm):
    if arm not in TRAIN_ARMS or seed not in SEEDS:
        raise ValueError("P35_UNREGISTERED_FACTUAL_MODEL_TRAIN")
    m=make_model(seed,arm)
    initial=copy.deepcopy(m)
    init_sha=p20.model_sha(initial)
    init_core_sha=core_sha(initial)
    freeze=({"critic_only_actor_frozen":("recurrent","actor"),
             "actor_head_only_ppo":("recurrent",)}.get(arm,()))
    for name,p in m.named_parameters():
        if name.startswith(tuple(x+"." for x in freeze)):
            p.requires_grad_(False)
    active=[p for p in m.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(active,lr=p20.PPO_LR,eps=1e-5)
    num_params=sum(p.numel() for p in active)
    real_actions=0;updates=0
    aux_generator=torch.Generator().manual_seed(seed+79001)
    aux_mismatch_total=0;aux_samples_total=0;aux_losses=[]
    source_sha=hashlib.sha256()
    reset_sha=hashlib.sha256()
    first_sha=hashlib.sha256()
    finished_returns=[];finished_success=[]
    histories=[];envs=[];episode_index=[0]*NUM_ENVS
    current_returns=[0.]*NUM_ENVS
    curves=[]
    begin=time.perf_counter()
    for i in range(NUM_ENVS):
        e=source_env(gym)
        seed_i=TRAIN_BASE+1000*i
        obs,_=e.reset(seed=seed_i)
        reset_sha.update(str(seed_i).encode())
        histories.append([row(obs[0])])
        envs.append(e)
    try:
        for rollout in range(ROLLOUTS):
            xs=[];taken=[];old_logp=[];old_v=[];rewards=[];dones=[];acceleration_targets=[]
            m.eval()
            for t in range(STEPS):
                x=torch.cat([past_tensor(h) for h in histories])
                with torch.no_grad():
                    logits,values=m(x)
                    d=torch.distributions.Categorical(logits=logits)
                    actions=d.sample()
                    logp=d.log_prob(actions)
                real_reward=[];real_done=[];real_factual_acceleration=[]
                for j,e in enumerate(envs):
                    a=int(actions[j])
                    new,reward,terminated,truncated=actual_step(e,histories[j],a)
                    done=terminated or truncated
                    real_factual_acceleration.append(factual_acceleration(histories[j],new))
                    event=json.dumps([histories[j][-1],a,new,terminated,truncated],
                        separators=(",",":"),allow_nan=False).encode()
                    source_sha.update(event)
                    if rollout==0:
                        first_sha.update(event)
                    histories[j].append(new)
                    real_actions+=1
                    current_returns[j]+=reward
                    real_reward.append(reward)
                    real_done.append(float(done))
                    if done:
                        finished_returns.append(current_returns[j])
                        finished_success.append(terminated)
                        current_returns[j]=0.
                        episode_index[j]+=1
                        new_seed=TRAIN_BASE+1000*j+episode_index[j]
                        reset_sha.update(str(new_seed).encode())
                        ob,_=e.reset(seed=new_seed)
                        histories[j]=[row(ob[0])]
                xs.append(x);taken.append(actions.detach())
                old_logp.append(logp.detach());old_v.append(values.detach())
                rewards.append(torch.tensor(real_reward,dtype=torch.float32))
                dones.append(torch.tensor(real_done,dtype=torch.float32))
                acceleration_targets.append(torch.tensor(real_factual_acceleration,dtype=torch.float32))
            with torch.no_grad():
                _,last_val=m(torch.cat([past_tensor(h) for h in histories]))
            values=torch.stack(old_v)
            rewards_t=torch.stack(rewards)
            done_t=torch.stack(dones)
            advantages=torch.zeros_like(rewards_t)
            gae=torch.zeros(NUM_ENVS)
            next_val=last_val
            for t in reversed(range(STEPS)):
                nonterminal=1.-done_t[t]
                delta=(rewards_t[t]+p20.GAMMA*next_val*nonterminal-
                       values[t])
                gae=(delta+p20.GAMMA*p20.GAE_LAMBDA*nonterminal*gae)
                advantages[t]=gae
                next_val=values[t]
            returns=(advantages+values).reshape(-1).detach()
            advantage=((advantages-advantages.mean())/
                       advantages.std(unbiased=False).clamp(min=1e-6)).reshape(-1).detach()
            flat_x=torch.cat(xs)
            flat_action=torch.stack(taken).reshape(-1)
            flat_old_lp=torch.stack(old_logp).reshape(-1)
            flat_acceleration=torch.stack(acceleration_targets).reshape(-1)
            m.train()
            generator=torch.Generator().manual_seed(seed+77*rollout+16111)
            for _ in range(EPOCHS):
                order=torch.randperm(len(flat_action),generator=generator)
                for offset in range(0,len(flat_action),MINIBATCH):
                    ix=order[offset:offset+MINIBATCH]
                    logit,val=m(flat_x[ix])
                    if arm=="critic_only_actor_frozen":
                        loss=p20.VALUE_COEF*F.smooth_l1_loss(val,returns[ix])
                    else:
                        dist=torch.distributions.Categorical(logits=logit)
                        ratio=(dist.log_prob(flat_action[ix])-flat_old_lp[ix]).exp()
                        unclipped=ratio*advantage[ix]
                        clipped=ratio.clamp(1-p20.PPO_CLIP,1+p20.PPO_CLIP)*advantage[ix]
                        loss=(-torch.minimum(unclipped,clipped).mean()
                              +p20.VALUE_COEF*F.smooth_l1_loss(val,returns[ix])
                              -p20.ENTROPY*dist.entropy().mean())
                    if arm in AUX_ARMS:
                        aux_loss,mismatch=auxiliary_loss(m,flat_x[ix],flat_action[ix],
                            flat_acceleration[ix],arm,aux_generator)
                        if not bool(torch.isfinite(aux_loss)):
                            raise RuntimeError("P35_NONFINITE_FACTUAL_SOURCE_AUX_PREDICTION")
                        loss=loss+AUX_LOSS_WEIGHT*aux_loss
                        aux_mismatch_total+=mismatch
                        aux_samples_total+=len(ix)
                        aux_losses.append(float(aux_loss.detach()))
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("P35_NONFINITE_ORIGINAL_REWARD_PPO_LOSS")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad=nn.utils.clip_grad_norm_(active,p20.MAX_GRAD_NORM)
                    if not bool(torch.isfinite(grad)):
                        raise RuntimeError("P35_NONFINITE_SOURCE_OPTIMIZER_GRADIENT")
                    optimizer.step()
                    updates+=1
            if (rollout+1)%16==0:
                recent=finished_returns[-32:]
                curves.append({
                    "rollouts":rollout+1,
                    "original_real_source_training_actions":real_actions,
                    "completed_source_episodes":len(finished_returns),
                    "recent_32_stochastic_source_reward_mean":
                        (sum(recent)/len(recent) if recent else None),
                    "recent_32_stochastic_source_goal_successes":
                        sum(finished_success[-32:]),
                    "last_original_reward_PPO_loss":float(loss.detach())})
    finally:
        for e in envs:e.close()
    parts=model_parts(m)
    aux_changed=(arm in AUX_ARMS and not all(torch.equal(initial.state_dict()[k],m.state_dict()[k]) for k in initial.state_dict() if k.startswith("aux_predictor.")))
    unchanged={name:all(torch.equal(initial.state_dict()[key],m.state_dict()[key])
                        for key in keys) for name,keys in parts.items()}
    if not all(unchanged[name] for name in freeze):
        raise RuntimeError("P35_FROZEN_TRAINING_COMPONENT_CHANGED")
    if unchanged["critic"]:
        raise RuntimeError("P35_CRITIC_WAS_NOT_TRAINED")
    if (arm in ("full_recurrent_ppo","conventional_history_mlp_ppo",
                "actor_head_only_ppo","factual_action_aux_ppo",
                "shuffled_action_aux_ppo") and unchanged["actor"]):
        raise RuntimeError("P35_TRAINABLE_ACTOR_DID_NOT_UPDATE")
    if (arm in ("full_recurrent_ppo","conventional_history_mlp_ppo",
                "factual_action_aux_ppo","shuffled_action_aux_ppo")
        and unchanged["recurrent"]):
        raise RuntimeError("P35_TRAINABLE_RECURRENT_OR_MLP_TRUNK_DID_NOT_UPDATE")
    if arm in AUX_ARMS and (not aux_changed or aux_samples_total!=updates*MINIBATCH):
        raise RuntimeError("P35_GENUINE_AUX_TRAIN_OR_REAL_SOURCE_CREDIT_MISSING")
    if arm=="factual_action_aux_ppo" and aux_mismatch_total!=0:
        raise RuntimeError("P35_ACTUAL_SOURCE_ACTION_AUX_CREDIT_CHANGED")
    if (arm=="shuffled_action_aux_ppo" and aux_samples_total>=512
        and aux_mismatch_total<aux_samples_total//2):
        raise RuntimeError("P35_SHUFFLED_AUX_ACTION_CREDIT_NOT_BROKEN")
    m.eval()
    return m,initial,{
        "seed":seed,"arm":arm,
        "initial_model_sha256":init_sha,
        "initial_core_actor_recurrent_critic_sha256":init_core_sha,
        "auxiliary_prediction_loss_mean":(sum(aux_losses)/len(aux_losses) if aux_losses else None),
        "auxiliary_action_samples":aux_samples_total,
        "auxiliary_wrong_action_fraction":(aux_mismatch_total/aux_samples_total if aux_samples_total else None),
        "auxiliary_prediction_head_changed":aux_changed,
        "trained_model_sha256":p20.model_sha(m),
        "all_parameters":sum(p.numel() for p in m.parameters()),
        "trained_parameters":num_params,
        "original_training_actual_source_actions":real_actions,
        "actual_source_optimizer_minibatches":updates,
        "first_512_source_action_consequence_sha256":first_sha.hexdigest(),
        "full_source_train_trace_sha256":source_sha.hexdigest(),
        "source_reset_seed_trace_sha256":reset_sha.hexdigest(),
        "components_bitwise_unchanged":unchanged,
        "frozen_component_names":list(freeze),
        "finished_original_source_training_episodes":len(finished_returns),
        "original_source_stochastic_training_curve":curves,
        "train_seconds":time.perf_counter()-begin,
        "no_privileged_velocity_or_expert_actions":True
    }

def rule(history):
    if len(history)==1:return 2
    return 0 if history[-1][3]<0. else 2

@torch.no_grad()
def eval_episode(gym,seed,model_seed,condition,arm,model=None):
    if seed not in EVAL_SEEDS or model_seed not in SEEDS or condition not in CONDITIONS or arm not in EVAL_ARMS:
        raise ValueError("P35_UNREGISTERED_ACTUAL_SOURCE_EVAL")
    env=source_env(gym,condition)
    try:
        ob,_=env.reset(seed=seed)
        history=[row(ob[0])]
        initial=history[0]
        rng=random.Random(seed+19991*model_seed)
        sha=hashlib.sha256()
        choices=[];total=0.;success=False
        for _ in range(HORIZON):
            if arm=="two_frame_conventional_pendulum_momentum_rule":
                action=rule(history)
            elif arm=="seeded_random":
                action=rng.randrange(3)
            else:
                logits,_=model(past_tensor(history))
                action=int(logits.argmax(dim=-1))
            new,reward,terminal,truncated=actual_step(env,history,action)
            sha.update(json.dumps([history[-1],action,new,terminal,truncated],
                      separators=(",",":"),allow_nan=False).encode())
            choices.append(action);total+=reward;history.append(new)
            if terminal or truncated:
                success=bool(terminal)
                break
        return {
            "source_reset_seed":seed,"condition":condition,
            "force_ratio":FORCE_RATIOS[condition],
            "initial_visible_source_row":initial,
            "source_actual_reward_return":total,
            "goal_reached_before_public_horizon":success,
            "source_actual_actions":len(choices),
            "original_source_actual_action_choices":choices,
            "source_real_action_consequence_sha256":sha.hexdigest(),
            "privileged_source_velocity_seen_by_policy":False,
            "privileged_force_ratio_seen_by_policy":False,
            "evaluation_optimizer_updates":0
        }
    finally:
        env.close()

def models_for_eval(seed):
    d={"initial_no_update":make_model(seed,"full_recurrent_ppo")}
    return d

def all_worlds(gym,seed,models):
    episodes=[]
    for condition in CONDITIONS:
        for world in EVAL_SEEDS:
            results={}
            first=[]
            for arm in EVAL_ARMS:
                v=eval_episode(gym,world,seed,condition,arm,models.get(arm))
                results[arm]=v
                first.append(tuple(v["initial_visible_source_row"]))
            if len(set(first))!=1:
                raise RuntimeError("P35_HELDOUT_INITIAL_PHYSICAL_SOURCE_MISMATCH")
            if (results["initial_no_update"]["source_real_action_consequence_sha256"]
                !=results["critic_only_actor_frozen"]["source_real_action_consequence_sha256"]):
                raise RuntimeError("P35_CRITIC_ONLY_REAL_ACTION_TRACES_NOT_EQUAL")
            episodes.append({"condition":condition,"source_reset_seed":world,
                             "original_source_factual_episodes":results})
    return episodes

def world_metrics(episodes):
    return {condition:{arm:{
        "mean_true_reward":sum(
            e["original_source_factual_episodes"][arm]["source_actual_reward_return"]
            for e in episodes if e["condition"]==condition)/len(EVAL_SEEDS),
        "actual_goal_successes_of_8":sum(
            bool(e["original_source_factual_episodes"][arm][
                "goal_reached_before_public_horizon"])
            for e in episodes if e["condition"]==condition)
        } for arm in EVAL_ARMS} for condition in CONDITIONS}

def frozen_gates(means):
    d=frozen()["frozen_scientific_thresholds"]
    result={}
    for condition in CONDITIONS:
        a=means[condition]["factual_action_aux_ppo"]
        sham=means[condition]["shuffled_action_aux_ppo"]
        full=means[condition]["full_recurrent_ppo"]
        mlp=means[condition]["conventional_history_mlp_ppo"]
        minimum=(d["per_seed_factual_aux_original_goal_success_min_of_8"]
             if condition=="original" else
             d["per_seed_factual_aux_low_and_high_success_min_of_8"])
        metrics={
            "source_goal_success_count_min":a["actual_goal_successes_of_8"]>=minimum,
            "factual_vs_shuffled_source_reward_margin20":a["mean_true_reward"]>=
                sham["mean_true_reward"]+d[
                  "per_seed_factual_reward_advantage_over_shuffled_aux_each_world"],
            "factual_within15_trained_neural_MLP":a["mean_true_reward"]>=
                mlp["mean_true_reward"]-d[
                  "per_seed_factual_noninferior_to_trained_MLP_reward_points"]
        }
        if condition=="original":
            metrics["factual_vs_full_PPO_source_reward_margin20"]=(
               a["mean_true_reward"]>=full["mean_true_reward"]+
               d["per_seed_factual_reward_advantage_over_full_PPO_original"])
        result[condition]=metrics
    return result

def train(gym,checkpoint_dir):
    if checkpoint_dir.exists():
        raise RuntimeError("P35_CHECKPOINT_ROOT_ALREADY_EXISTS_NO_SOURCE_REPLAY")
    checkpoint_dir.mkdir(parents=True)
    started=time.perf_counter()
    records=[]
    for seed in SEEDS:
        models=models_for_eval(seed)
        initsha=p20.model_sha(models["initial_no_update"])
        trained={};checkpoints={};receipts={}
        for arm in TRAIN_ARMS:
            m,initial,r=train_arm(gym,seed,arm)
            if r["original_training_actual_source_actions"]!=ROLLOUTS*NUM_ENVS*STEPS or r[
                "actual_source_optimizer_minibatches"]!=ROLLOUTS*EPOCHS*2:
                raise RuntimeError("P35_FACTUAL_ACTION_OR_PPO_TRAINING_BUDGET_CHANGED")
            if arm!="conventional_history_mlp_ppo" and core_sha(initial)!=core_sha(models["initial_no_update"]):
                raise RuntimeError("P35_IDENTICAL_INITIAL_RECURRENT_SOURCE_NOT_MATCHED")
            name=f"model_seed_{seed}_{arm}.pt"
            path=checkpoint_dir/name
            torch.save(m.state_dict(),path)
            fresh=make_model(seed,arm)
            fresh.load_state_dict(torch.load(path,weights_only=True,map_location="cpu"))
            fresh.eval()
            if p20.model_sha(fresh)!=p20.model_sha(m):
                raise RuntimeError("P35_INPROCESS_COLD_TRAINED_MODEL_SHA_MISMATCH")
            checkpoints[arm]={
                "filename":name,
                "checkpoint_file_SHA256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "trained_weights_SHA256":p20.model_sha(m)
            }
            trained[arm]=m;receipts[arm]=r
        gru_initial={receipts[arm]["initial_core_actor_recurrent_critic_sha256"]
                     for arm in TRAIN_ARMS
                     if arm!="conventional_history_mlp_ppo"}
        gru_first={receipts[arm]["first_512_source_action_consequence_sha256"]
                   for arm in TRAIN_ARMS
                   if arm!="conventional_history_mlp_ppo"}
        if len(gru_initial)!=1 or len(gru_first)!=1:
            raise RuntimeError("P35_EQUAL_ORIGINAL_GRU_FIRST_ACTION_SOURCE_FAIL")
        models.update(trained)
        episodes=all_worlds(gym,seed,models)
        means=world_metrics(episodes)
        records.append({
            "model_seed":seed,"new_non_CartPole_source_task":True,
            "equal_first_real_GRU_source_trace_SHA256":next(iter(gru_first)),
            "original_real_train_receipts":receipts,
            "source_eval_episodes":episodes,
            "original_source_condition_means":means,
            "frozen_source_criteria_by_condition":frozen_gates(means),
            "checkpoints":checkpoints
        })
    actual_actions=sum(r["original_training_actual_source_actions"] for rec in records
                      for r in rec["original_real_train_receipts"].values())
    updates=sum(r["actual_source_optimizer_minibatches"] for rec in records
                for r in rec["original_real_train_receipts"].values())
    passing_seeds=sum(all(all(x.values()) for x in r["frozen_source_criteria_by_condition"].values())
                      for r in records)
    success=passing_seeds>=frozen()["frozen_scientific_thresholds"]["at_least_n_new_model_seeds_passing_all_worlds"]
    wall=time.perf_counter()-started
    good=(actual_actions==3*len(TRAIN_ARMS)*49152 and updates==3*len(TRAIN_ARMS)*576
          and wall<=frozen()["qualification"]["max_public_source_training_plus_evaluation_seconds"])
    return {
        "schema":SCHEMA,
        "source_admission":"ORIGINAL_PINNED_MIT_FARAMA_MOUNTAINCAR",
        "engineering":"PASS_PUBLIC_ORIGINAL_SOURCE_DEV" if good else "FAIL_P35_ENGINEERING",
        "protected_scientific_first_use":"NOT_RUN",
        "factual_aux_advantage_precommitted_2_of_3_science_target":success,
        "factual_aux_seeds_passing_all_source_worlds":passing_seeds,
        "non_CartPole_action_consequence_task_executed":True,
        "no_scientific_architecture_or_FORGE_claim":True,
        "real_source_training_actions":actual_actions,
        "real_source_optimizer_minibatches":updates,
        "real_source_evaluation_actions":sum(
            x["source_actual_actions"] for rec in records
            for ep in rec["source_eval_episodes"]
            for x in ep["original_source_factual_episodes"].values()),
        "original_train_and_eval_wall_seconds":wall,
        "records":records
    }

def cold_verify(gym,receipt_path,checkpoint_dir):
    result=json.loads(receipt_path.read_text(encoding="utf-8"))
    if (result["schema"]!=SCHEMA
        or result["engineering"]!="PASS_PUBLIC_ORIGINAL_SOURCE_DEV"
        or result["real_source_training_actions"]!=3*len(TRAIN_ARMS)*49152
        or result["real_source_optimizer_minibatches"]!=3*len(TRAIN_ARMS)*576
        or len(result["records"])!=3):
        raise RuntimeError("P35_COLD_ORIGINAL_SOURCE_RECEIPT_CHANGED")
    if len(list(checkpoint_dir.glob("*.pt")))!=3*len(TRAIN_ARMS):
        raise RuntimeError("P35_COLD_SAVED_MODEL_SET_INCOMPLETE")
    episodes=0;actions=0
    with torch.no_grad():
        for rec in result["records"]:
            seed=rec["model_seed"]
            if seed not in SEEDS:
                raise RuntimeError("P35_COLD_UNREGISTERED_ORIGINAL_SEED")
            for arm in TRAIN_ARMS:
                info=rec["checkpoints"][arm]
                path=checkpoint_dir/info["filename"]
                if path.parent.resolve()!=checkpoint_dir.resolve() or not path.is_file():
                    raise RuntimeError("P35_COLD_SOURCE_WEIGHTS_MISSING_OR_ESCAPE")
                if hashlib.sha256(path.read_bytes()).hexdigest()!=info[
                    "checkpoint_file_SHA256"]:
                    raise RuntimeError("P35_COLD_WEIGHTS_FILE_SHA_NOT_EXACT")
                m=make_model(seed,arm)
                m.load_state_dict(torch.load(path,weights_only=True,map_location="cpu"),
                                  strict=True)
                m.eval()
                initial_sha=p20.model_sha(m)
                if initial_sha!=info["trained_weights_SHA256"]:
                    raise RuntimeError("P35_COLD_WEIGHTS_STATE_SHA_NOT_EXACT")
                for entry in rec["source_eval_episodes"]:
                    target=entry["original_source_factual_episodes"][arm]
                    got=eval_episode(gym,entry["source_reset_seed"],seed,
                        entry["condition"],arm,m)
                    if (got["original_source_actual_action_choices"]!=target[
                            "original_source_actual_action_choices"]
                        or got["source_real_action_consequence_sha256"]!=target[
                            "source_real_action_consequence_sha256"]
                        or got["source_actual_reward_return"]!=target[
                            "source_actual_reward_return"]
                        or got["goal_reached_before_public_horizon"]!=target[
                            "goal_reached_before_public_horizon"]):
                        raise RuntimeError("P35_COLD_REAL_MOUNTAINCAR_ACTION_CONSEQUENCE_CHANGED")
                    actions+=got["source_actual_actions"];episodes+=1
                if p20.model_sha(m)!=initial_sha:
                    raise RuntimeError("P35_COLD_EVALUATION_UPDATED_MODEL")
    if episodes!=3*len(TRAIN_ARMS)*len(CONDITIONS)*len(EVAL_SEEDS):
        raise RuntimeError("P35_COLD_WORLD_COVERAGE_INCOMPLETE")
    return {
        "schema":"ttt-hybrid.native-p35.cold-source-inference-replay.v1",
        "status":"PASS_EXACT_ACTUAL_MOUNTAINCAR_ACTION_REWARD_TRACE_REPLAY",
        "fresh_python_process":True,
        "total_cold_models":3*len(TRAIN_ARMS),
        "cold_original_source_episodes":episodes,
        "cold_original_source_actions":actions,
        "no_gradient_updates":True,
        "protected_scientific_first_use":"NOT_RUN"
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--s29-root",type=Path,required=True)
    parser.add_argument("--source-root",type=Path,required=True)
    parser.add_argument("--cloudpickle-root",type=Path,required=True)
    parser.add_argument("--receipt",type=Path,required=True)
    parser.add_argument("--checkpoint-dir",type=Path,required=True)
    parser.add_argument("--mode",choices=("train","cold"),required=True)
    args=parser.parse_args()
    guard()
    if not args.receipt.is_absolute() or not args.checkpoint_dir.is_absolute():
        raise RuntimeError("P35_OUTPUT_PATHS_MUST_BE_ABSOLUTE")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    gym,source_receipt=p20.admit_gymnasium(
        args.s29_root,args.source_root,args.cloudpickle_root)
    if args.mode=="train":
        if args.receipt.exists():
            raise RuntimeError("P35_CONSUMED_PUBLIC_SOURCE_RESULT_ALREADY_EXISTS")
        result=train(gym,args.checkpoint_dir)
        result["original_source_custody"]=source_receipt
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        args.receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",
                                encoding="utf-8")
        print("P35_ORIGINAL_MOUNTAINCAR_REWARD_RESULT="+json.dumps({
            "engineering":result["engineering"],
            "factual_aux_replication_success":result["factual_aux_advantage_precommitted_2_of_3_science_target"],
            "seeds_passing":result["factual_aux_seeds_passing_all_source_worlds"],
            "real_train_actions":result["real_source_training_actions"],
            "actual_ppo_updates":result["real_source_optimizer_minibatches"],
            "eval_actions":result["real_source_evaluation_actions"],
            "wall_seconds":result["original_train_and_eval_wall_seconds"],
            "seeds":[{"seed":r["model_seed"],
                      "source_means":r["original_source_condition_means"],
                      "criteria":r["frozen_source_criteria_by_condition"]}
                      for r in result["records"]]
        },sort_keys=True))
        return 0 if result["engineering"]=="PASS_PUBLIC_ORIGINAL_SOURCE_DEV" else 1
    else:
        receipt=cold_verify(gym,args.receipt,args.checkpoint_dir)
        audit=args.receipt.parent/"native_p35_mountaincar_cold_audit.json"
        if audit.exists():
            raise RuntimeError("P35_COLD_REPLAY_ALREADY_CONSUMED")
        audit.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",
                         encoding="utf-8")
        print("P35_FRESH_PROCESS_COLD_REAL_SOURCE_REPLAY="+json.dumps(receipt,sort_keys=True))
        return 0

if __name__=="__main__":
    raise SystemExit(main())
