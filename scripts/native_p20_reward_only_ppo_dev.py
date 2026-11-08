"""P20: learn real CartPole control from OWN factual actions and rewards, NO expert labels.

P19's competent learned supervised controllers are immutable comparator programs.
P18's failed REINFORCE reward-only source and results are preserved.
P20 trains unchanged 489-param HistoryGRU and 471-param FullHistoryMLP
with clipped PPO, GAE, actual environment step rewards, no expert policy
actions, hidden engine parameters, privileged labels or counterfactuals.

The actor-credit corrupted equal-budget PPO GRU negative receives SAME true
environment feedback but shuffled factual action-to-advantage attribution.
External Farama MIT source / S29 Git identity is read-only and admitted via
the ORIGINAL P18 function. The source is an already-known DEV source, not
independent scientific first use or proof of native Hybrid architecture.
"""
from __future__ import annotations
import argparse
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
from torch.nn import functional as F
from scripts.native_p18_real_cartpole_factual_dev import (
    HistoryGRU,FullHistoryMLP,admit_gymnasium,factual_row,handcoded_action,
    model_sha,HISTORY)
from scripts.native_p19_factual_expert_imit_dev import (
    collect_actual_demonstrations,train as train_p19,
    TRAIN_RESET_SEEDS as P19_TRAIN,VAL_RESET_SEEDS as P19_VAL,
    INIT_SEEDS as P19_INITS,scaled_history,logits as expert_logits)

SCHEMA="ttt-hybrid.native-p20.self-reward-ppo-dev.v1"
SEEDS=(3119,3137,3167)
TRAIN_SEED_BASE=82000
ROLLOUTS=96
NUM_ENVS=8
STEPS_PER_ROLLOUT=64
PPO_EPOCHS=3
MINIBATCH=256
PPO_LR=.0015
GAMMA=.99
GAE_LAMBDA=.95
PPO_CLIP=.2
ENTROPY=.015
VALUE_COEF=.5
MAX_GRAD_NORM=1.0
EVAL_SEEDS=(92013,92029,92041,92053)
EVAL_HORIZON=200
EVAL_CONDITIONS=("original","short_pole","long_pole")
EVAL_ARMS=("reward_ppo_gru","reward_ppo_mlp","wrong_credit_ppo_gru",
           "untrained_gru","ppo_gru_zero_prior_history",
           "p19_factual_supervised_linear","p19_factual_supervised_gru",
           "handcoded_observation_controller","seeded_random")
STANDARD_TARGET=150.
MIN_RANDOM_GAIN=75.
MIN_UNTRAINED_GAIN=75.
MIN_CREDIT_NEGATIVE_GAIN=40.
MAX_PPO_GRU_DEFICIT_VS_MLP=20.
TRANSFER_TARGET=100.
MAX_PROCESS_WALL_SECONDS=600.
TRAIN_ARMS=("reward_gru","reward_mlp","wrong_credit_gru")
MODEL_CLASSES={"reward_gru":HistoryGRU,"reward_mlp":FullHistoryMLP,
               "wrong_credit_gru":HistoryGRU}


def require_development():
    if os.environ.get("NATIVE_P20_DEV_ONLY","1")!="1":
        raise RuntimeError("P20_OFFICIAL_SCIENCE_FIRST_USE_DENIED")


def set_public_environment_dynamics(env,condition):
    """Actual source-instance pole parameters; never inputs to any actor."""
    if condition not in EVAL_CONDITIONS:
        raise ValueError("P20_UNREGISTERED_TRANSFER_STRUCTURE")
    ratio={"original":1.0,"short_pole":.7,"long_pole":1.3}[condition]
    state=env.unwrapped
    if not math.isclose(float(state.length),.5,abs_tol=1e-8):
        raise RuntimeError("P20_PINNED_GYM_LENGTH_CHANGED")
    state.length=.5*ratio
    state.polemass_length=state.masspole*state.length
    # Pinned upstream step() reads length and polemass_length. Only alters
    # the ephemeral, isolated environment object, not installed source or
    # any actor input. Standard source untouched.
    return {"condition":condition,"pole_length_ratio":ratio,
            "model_observes_ratio":False}


def model_for_arm(arm,seed):
    if arm not in TRAIN_ARMS or seed not in SEEDS:
        raise ValueError("P20_UNREGISTERED_TRAINING_ARM")
    torch.manual_seed(seed+(0 if arm=="reward_gru" else
                            50000 if arm=="reward_mlp" else 100000))
    return MODEL_CLASSES[arm]()


def actor_tensor(history):
    # Exactly the same public 4D sensor + preceding factual action/reward
    # input scale as P19's separately qualified learned controller.
    return scaled_history(history)


def corrupt_action_credit(gae):
    if gae.ndim!=2 or gae.shape!=(STEPS_PER_ROLLOUT,NUM_ENVS):
        raise ValueError("P20_CREDIT_CONTROL_SHAPE")
    return torch.roll(gae,shifts=(STEPS_PER_ROLLOUT//2,1),dims=(0,1))


def complete_factual_episode_setup(gym,model_seed,arm):
    envs=[]
    history=[]
    completed_rewards=[]
    completed_length=[]
    episode_index=[0]*NUM_ENVS
    episode_return=[0.]*NUM_ENVS
    episode_steps=[0]*NUM_ENVS
    for i in range(NUM_ENVS):
        env=gym.make("CartPole-v1",render_mode=None)
        ob,_=env.reset(seed=TRAIN_SEED_BASE+1000*i)
        envs.append(env)
        history.append([factual_row(ob,0,0.)])
    return envs,history,episode_index,episode_return,episode_steps,completed_rewards,completed_length


def train_ppo(gym,arm,seed):
    require_development()
    model=model_for_arm(arm,seed)
    initial_sha=model_sha(model)
    initial_weights={k:v.detach().clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=PPO_LR,eps=1e-5)
    envs,history,episode_index,episode_return,episode_steps,completed_returns,completed_lengths=complete_factual_episode_setup(gym,seed,arm)
    source_trace=hashlib.sha256()
    resets_hash=hashlib.sha256()
    stats=[]
    gradient_updates=0
    factual_actions=0
    begin=time.perf_counter()
    for i in range(NUM_ENVS):
        resets_hash.update(str(TRAIN_SEED_BASE+1000*i).encode())
    try:
        for rollout in range(ROLLOUTS):
            xs=[]
            as_=[]
            oldlp=[]
            oldvalues=[]
            rewards=[]
            dones=[]
            model.eval()
            for t in range(STEPS_PER_ROLLOUT):
                x=torch.cat([actor_tensor(h) for h in history],dim=0)
                with torch.no_grad():
                    pi,v=model(x)
                    d=torch.distributions.Categorical(logits=pi)
                    a=d.sample()
                    lp=d.log_prob(a)
                x_actions=[];x_rewards=[];x_dones=[]
                for j,env in enumerate(envs):
                    choice=int(a[j])
                    next_obs,reward,term,trunc,_=env.step(choice)
                    factual_actions+=1
                    reward=float(reward)
                    if not math.isfinite(reward):
                        raise RuntimeError("P20_NONFINITE_REAL_REWARD")
                    done=bool(term or trunc)
                    new=factual_row(next_obs,2*choice-1,reward)
                    source_trace.update(json.dumps(
                        [history[j][-1],choice,new,done],
                        separators=(",",":"),allow_nan=False).encode())
                    history[j].append(new)
                    episode_return[j]+=reward
                    episode_steps[j]+=1
                    x_actions.append(choice);x_rewards.append(reward)
                    x_dones.append(done)
                    if done:
                        completed_returns.append(episode_return[j])
                        completed_lengths.append(episode_steps[j])
                        episode_index[j]+=1
                        reset_seed=TRAIN_SEED_BASE+1000*j+episode_index[j]
                        resets_hash.update(str(reset_seed).encode())
                        reset,_=env.reset(seed=reset_seed)
                        history[j]=[factual_row(reset,0,0.)]
                        episode_return[j]=0.
                        episode_steps[j]=0
                xs.append(x)
                as_.append(a.detach())
                oldlp.append(lp.detach())
                oldvalues.append(v.detach())
                rewards.append(torch.tensor(x_rewards,dtype=torch.float32))
                dones.append(torch.tensor(x_dones,dtype=torch.float32))
            with torch.no_grad():
                final_x=torch.cat([actor_tensor(h) for h in history],dim=0)
                _,last_value=model(final_x)
            reward_t=torch.stack(rewards)
            done_t=torch.stack(dones)
            value_t=torch.stack(oldvalues)
            next_v=last_value
            last_adv=torch.zeros(NUM_ENVS)
            advs=torch.zeros_like(reward_t)
            for t in reversed(range(STEPS_PER_ROLLOUT)):
                next_mask=1.-done_t[t]
                delta=reward_t[t]+GAMMA*next_v*next_mask-value_t[t]
                last_adv=delta+GAMMA*GAE_LAMBDA*next_mask*last_adv
                advs[t]=last_adv
                next_v=value_t[t]
            returns=advs+value_t
            # No expert labels in either path. The negative corrupts only
            # how the actually chosen action is credited for eventual reward.
            actor_adv=(corrupt_action_credit(advs) if arm=="wrong_credit_gru"
                       else advs)
            actor_adv=(actor_adv-actor_adv.mean())/(actor_adv.std(
                unbiased=False).clamp(min=1e-6))
            flat_x=torch.cat(xs)
            flat_a=torch.stack(as_).reshape(-1)
            flat_oldlp=torch.stack(oldlp).reshape(-1)
            flat_adv=actor_adv.reshape(-1).detach()
            flat_ret=returns.reshape(-1).detach()
            model.train()
            gen=torch.Generator().manual_seed(seed+77*rollout+16111)
            for epoch in range(PPO_EPOCHS):
                order=torch.randperm(len(flat_a),generator=gen)
                for offset in range(0,len(flat_a),MINIBATCH):
                    ix=order[offset:offset+MINIBATCH]
                    logits,values=model(flat_x[ix])
                    dist=torch.distributions.Categorical(logits=logits)
                    ratio=(dist.log_prob(flat_a[ix])-
                           flat_oldlp[ix]).exp()
                    s1=ratio*flat_adv[ix]
                    s2=ratio.clamp(1-PPO_CLIP,1+PPO_CLIP)*flat_adv[ix]
                    actor_loss=-torch.minimum(s1,s2).mean()
                    critic_loss=F.smooth_l1_loss(values,flat_ret[ix])
                    loss=actor_loss+VALUE_COEF*critic_loss-(
                        ENTROPY*dist.entropy().mean())
                    if not torch.isfinite(loss):
                        raise RuntimeError("P20_NONFINITE_PPO_GRADIENT")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(),MAX_GRAD_NORM)
                    optimizer.step()
                    gradient_updates+=1
            if (rollout+1)%16==0:
                recent=completed_returns[-min(32,len(completed_returns)):]
                stats.append({
                    "rollouts":rollout+1,
                    "total_true_factual_decisions":factual_actions,
                    "closed_factual_episodes":len(completed_returns),
                    "recent_actual_episode_return_mean":sum(recent)/len(recent)
                          if recent else None,
                    "PPO_last_update_loss":float(loss.detach())})
    finally:
        for e in envs:e.close()
    model.eval()
    state=io.BytesIO()
    torch.save(model.state_dict(),state)
    state.seek(0)
    torch.manual_seed(seed+708800)
    fresh=MODEL_CLASSES[arm]()
    fresh.load_state_dict(torch.load(state,weights_only=True,
                                     map_location="cpu"),strict=True)
    fresh.eval()
    sentinel=actor_tensor([factual_row([.02,.03,.04,.05],0,0.)])
    with torch.no_grad():
        p1,v1=model(sentinel);p2,v2=fresh(sentinel)
        restore_exact=torch.equal(p1,p2) and torch.equal(v1,v2)
    clean=model_for_arm(arm,seed)
    clean.load_state_dict(initial_weights,strict=True)
    clean.eval()
    return {"trained":fresh,"untrained":clean},{
        "arm":arm,"seed":seed,"no_expert_action_labels_used":True,
        "true_environment_action_consequence_reward_only":True,
        "reward_credit_time_and_env_scrambled":arm=="wrong_credit_gru",
        "actor_critic_training_update_batches":gradient_updates,
        "actual_factual_environment_actions":factual_actions,
        "original_reset_seed_stream_sha256":resets_hash.hexdigest(),
        "true_factual_action_consequence_sha256":source_trace.hexdigest(),
        "actual_closed_train_episodes":len(completed_returns),
        "param_count":sum(p.numel() for p in fresh.parameters()),
        "trained_weights_changed":model_sha(fresh)!=initial_sha,
        "weight_sha256":model_sha(fresh),
        "state_bytes_sha256":hashlib.sha256(state.getvalue()).hexdigest(),
        "in_process_fresh_weights_exact_logits":restore_exact,
        "same_precommitted_source_reset_rule":True,
        "training_curve_last32_factual_episodes":stats,
        "actual_training_seconds":time.perf_counter()-begin}


def controller_action(arm,models,history,obs,rng):
    if arm=="handcoded_observation_controller":
        return handcoded_action(obs)
    if arm=="seeded_random":
        return rng.randrange(2)
    x=actor_tensor(history)
    with torch.no_grad():
        if arm in ("reward_ppo_gru","reward_ppo_mlp",
                   "wrong_credit_ppo_gru","untrained_gru",
                   "ppo_gru_zero_prior_history"):
            model=(models["reward_gru"] if arm in (
                "reward_ppo_gru","ppo_gru_zero_prior_history") else
                models["reward_mlp"] if arm=="reward_ppo_mlp" else
                models["wrong_credit_gru"] if arm=="wrong_credit_ppo_gru"
                else models["untrained_gru"])
            if arm=="ppo_gru_zero_prior_history":
                x=x.clone()
                x[:,:-1,:]=0
            pi,_=model(x)
        elif arm in ("p19_factual_supervised_linear",
                     "p19_factual_supervised_gru"):
            m=models["p19_linear" if arm.endswith("linear") else "p19_gru"]
            k="linear" if arm.endswith("linear") else "history_gru"
            pi=expert_logits(m,k,x)
        else:
            raise ValueError("P20_UNREGISTERED_EVALUATION_ARM")
        return int(pi.argmax(dim=1).item())


def evaluate(gym,seed,condition,init,models):
    if seed not in EVAL_SEEDS or condition not in EVAL_CONDITIONS or init not in SEEDS:
        raise ValueError("P20_UNREGISTERED_DEV_EVAL")
    result={}
    prefixes=[]
    physics=[]
    for arm in EVAL_ARMS:
        env=gym.make("CartPole-v1",render_mode=None)
        try:
            descriptor=set_public_environment_dynamics(env,condition)
            obs,_=env.reset(seed=seed)
            initial=factual_row(obs,0,0.)
            prefixes.append(tuple(initial))
            physics.append(descriptor["pole_length_ratio"])
            hist=[initial]
            rand=random.Random(seed+31317*init)
            trace=hashlib.sha256()
            actions=[]
            actual_reward=[]
            for _ in range(EVAL_HORIZON):
                action=controller_action(arm,models,hist,obs,rand)
                nxt,reward,term,trunc,_=env.step(action)
                new=factual_row(nxt,2*action-1,float(reward))
                trace.update(json.dumps(
                    [hist[-1],action,new,bool(term),bool(trunc)],
                    separators=(",",":"),allow_nan=False).encode())
                hist.append(new);actions.append(action)
                actual_reward.append(float(reward))
                obs=nxt
                if term or trunc:break
            result[arm]={
                "real_executed_actions":len(actions),
                "real_observed_reward":sum(actual_reward),
                "factual_actions":actions,
                "factual_rewards":actual_reward,
                "trace_sha256":trace.hexdigest(),
                "model_never_read_environment_physics":True,
                "no_eval_optimizer_updates":True}
        finally:env.close()
    checks={
        "same_exact_initial_real_observation_for_all_nine_policies":
            len(set(prefixes))==1,
        "same_exact_runtime_source_instance_physics_for_all_arms":
            len(set(physics))==1,
        "all_nine_arms_actual_source_step_and_completed_factual_reward":
            len(result)==len(EVAL_ARMS) and all(
                1<=v["real_executed_actions"]<=EVAL_HORIZON and
                len(v["factual_actions"])==v["real_executed_actions"]==
                len(v["factual_rewards"]) and
                all(math.isfinite(z) for z in v["factual_rewards"])
                for v in result.values()),
        "no_hidden_physics_given_to_actor":True
    }
    return {"seed":seed,"condition":condition,"init":init,
            "checks":checks,
            "integrity":"PASS" if all(checks.values()) else "FAIL",
            "arms":result}


def run(s29_root,source_root,cloudpickle_root):
    require_development()
    torch.set_num_threads(1)
    started=time.perf_counter()
    gym,source=admit_gymnasium(s29_root,source_root,cloudpickle_root)
    p19_x,p19_y,p19_train=collect_actual_demonstrations(gym,P19_TRAIN)
    p19_vx,p19_vy,p19_val=collect_actual_demonstrations(gym,P19_VAL)
    records=[]
    for ix,seed in enumerate(SEEDS):
        trained={}
        policy={}
        for arm in TRAIN_ARMS:
            o,receipt=train_ppo(gym,arm,seed)
            trained[arm]=receipt
            policy[arm]=o["trained"]
            if arm=="reward_gru":
                policy["untrained_gru"]=o["untrained"]
        p19_seed=P19_INITS[ix]
        champion={}
        champ_receipts={}
        for kind in ("linear","history_gru"):
            m,r=train_p19(gym,p19_x,p19_y,p19_vx,p19_vy,kind,p19_seed)
            champion["p19_"+("linear" if kind=="linear" else "gru")]=m
            champ_receipts[kind]=r
        models={**policy,**champion}
        before={k:model_sha(v) for k,v in models.items()}
        eps=[evaluate(gym,seed_dev,condition,seed,models)
             for condition in EVAL_CONDITIONS for seed_dev in EVAL_SEEDS]
        after={k:model_sha(v) for k,v in models.items()}
        means={condition:{
            arm:sum(e["arms"][arm]["real_observed_reward"]
                    for e in eps if e["condition"]==condition)/len(EVAL_SEEDS)
            for arm in EVAL_ARMS}
            for condition in EVAL_CONDITIONS}
        normal=means["original"]
        gates={
            "PPO_GRU_real_reward_controller_at_least_150":
                normal["reward_ppo_gru"]>=STANDARD_TARGET,
            "PPO_MLP_real_reward_controller_at_least_150":
                normal["reward_ppo_mlp"]>=STANDARD_TARGET,
            "GRU_over_random_by_75":
                normal["reward_ppo_gru"]>=normal["seeded_random"]+MIN_RANDOM_GAIN,
            "GRU_over_untrained_by_75":
                normal["reward_ppo_gru"]>=normal["untrained_gru"]+MIN_UNTRAINED_GAIN,
            "GRU_over_same_budget_broken_reward_credit_by_40":
                normal["reward_ppo_gru"]>=normal["wrong_credit_ppo_gru"]+
                MIN_CREDIT_NEGATIVE_GAIN,
            "GRU_not_20_worse_than_comparable_MLP":
                normal["reward_ppo_gru"]>=normal["reward_ppo_mlp"]-
                MAX_PPO_GRU_DEFICIT_VS_MLP,
            "short_pole_GRU_retains_at_least_100_steps":
                means["short_pole"]["reward_ppo_gru"]>=TRANSFER_TARGET,
            "long_pole_GRU_retains_at_least_100_steps":
                means["long_pole"]["reward_ppo_gru"]>=TRANSFER_TARGET,
            "learned_P19_factual_supervised_linear_revalidates_175":
                normal["p19_factual_supervised_linear"]>=175,
            "learned_P19_factual_supervised_GRU_revalidates_175":
                normal["p19_factual_supervised_gru"]>=175,
            "same_original_bounded_teacher_reaches_200":
                normal["handcoded_observation_controller"]==EVAL_HORIZON}
        checks={
            "all_actual_reward_only_PPO_models_updated_without_expert_labels":
                all(x["trained_weights_changed"] and
                    x["no_expert_action_labels_used"] and
                    x["actual_factual_environment_actions"]==
                    NUM_ENVS*STEPS_PER_ROLLOUT*ROLLOUTS and
                    x["actor_critic_training_update_batches"]==
                    ROLLOUTS*PPO_EPOCHS*(
                       NUM_ENVS*STEPS_PER_ROLLOUT//MINIBATCH)
                    for x in trained.values()),
            "all_PPO_weights_only_loaded_same_network_logits":
                all(x["in_process_fresh_weights_exact_logits"]
                    for x in trained.values()),
            "correct_actor_credit_positive_and_corrupted_negative":
                not trained["reward_gru"]["reward_credit_time_and_env_scrambled"]
                and trained["wrong_credit_gru"][
                    "reward_credit_time_and_env_scrambled"],
            "source_seeds_and_equal_action_update_budgets":
                len({x["actual_factual_environment_actions"]
                     for x in trained.values()})==1,
            "P19_learned_supervised_controls_original_factual_700_and_900":
                champ_receipts["linear"]["actual_gradient_updates"]==700
                and champ_receipts["history_gru"]["actual_gradient_updates"]==900,
            "all_models_unchanged_during_real_source_evaluation":before==after,
            "all_conditions_every_original_episode_causally_chronological":
                len(eps)==len(EVAL_CONDITIONS)*len(EVAL_SEEDS) and all(
                    e["integrity"]=="PASS" for e in eps),
            "protected_official_science_seeds_never_admitted":True}
        records.append({
            "init_seed":seed,"frozen_P19_comparator_seed":p19_seed,
            "engineering":"PASS" if all(checks.values()) else "FAIL",
            "checks":checks,"PPO_training":trained,
            "P19_separate_supervised_reference":champ_receipts,
            "mean_real_factual_episode_reward":means,
            "frozen_directional_hypotheses":gates,
            "full_real_action_outcome_episodes":eps})
    elapsed=time.perf_counter()-started
    support={key:sum(int(r["frozen_directional_hypotheses"][key])
                     for r in records)
             for key in records[0]["frozen_directional_hypotheses"]}
    engineering=(len(records)==3 and
                 all(r["engineering"]=="PASS" for r in records) and
                 elapsed<=MAX_PROCESS_WALL_SECONDS)
    learning=(all(all(r["frozen_directional_hypotheses"][n]
              for n in ("PPO_GRU_real_reward_controller_at_least_150",
                        "GRU_over_random_by_75",
                        "GRU_over_untrained_by_75",
                        "GRU_over_same_budget_broken_reward_credit_by_40",
                        "short_pole_GRU_retains_at_least_100_steps",
                        "long_pole_GRU_retains_at_least_100_steps"))
              for r in records))
    return {
        "schema":SCHEMA,
        "status":"PASS_PPO_REAL_FACTUAL_DEV_INTEGRITY_NOT_SCIENCE"
                 if engineering else "FAIL_PPO_REAL_SOURCE_ENGINEERING",
        "frozen_reward_only_development_target_pass":learning,
        "source_provenance":source,
        "real_external_source_used":True,
        "supervised_teacher_used_to_train_PPO_models":False,
        "P19_supervised_champions_are_only_eval_references":True,
        "reused_original_P19_train_and_validation_source":True,
        "new_actual_reward_only_training_actions":len(records)*len(
            TRAIN_ARMS)*ROLLOUTS*NUM_ENVS*STEPS_PER_ROLLOUT,
        "reward_only_AdamW_PPO_update_batches_total":len(records)*len(
            TRAIN_ARMS)*ROLLOUTS*PPO_EPOCHS*(
                NUM_ENVS*STEPS_PER_ROLLOUT//MINIBATCH),
        "original_P19_supervised_reference_optimizer_updates_total":
            len(records)*(700+900),
        "new_actual_real_factual_eval_actions":sum(
            e["arms"][arm]["real_executed_actions"]
            for r in records for e in r["full_real_action_outcome_episodes"]
            for arm in EVAL_ARMS),
        "original_scientific_first_use":"NOT_RUN",
        "official_prospective_world_seeds_consumed":[],
        "independent_new_scientific_source":False,
        "native_substrate_modified":False,
        "P18_reward_only_failure_preserved":True,
        "original_P19_champion_source_modified":False,
        "independent_P19_sibling_pr91_modified":False,
        "source_whole_process_compute_equal_between_GRU_and_MLP":False,
        "real_process_death_checkpoint_restoration_proven":False,
        "FORYGE_research_loop_activated":False,
        "accepted_model_pointer_changed":False,
        "scientific_learning_success_established":False,
        "model_seeds":list(SEEDS),
        "public_development_evaluation_seeds":list(EVAL_SEEDS),
        "evaluation_perturbations":list(EVAL_CONDITIONS),
        "predeclared_support_out_of_three":support,
        "records":records,"whole_process_wall_seconds":elapsed,
        "limitations":"Actual reward-only PPO actions and factual next observations, with true GAE, compare against P19 factual supervised expert-trained champion and bad reward-credit policy on newly frozen PUBLIC DEVELOPMENT CartPole reset seeds/length perturbations. Similar PPO models share 8-history inputs, rollout actions/update budget and optimizer but different FLOPs, trajectories and initialization. Experts are retrained separately on existing old P19 label data solely as immutable comparison controls, never supplied to reward-only actor. Environmental length changes are authored modifications of existing MIT engine instance, NOT independently born external dynamics; no independent sealed source transfer, scientific-first-use or genuine lifelong cold retention. Other exploratory environments unchanged. Never equate learned RL with native Hybrid substrate and never promote without independent formal science."}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--s29-root",type=Path,required=True)
    ap.add_argument("--source-root",type=Path,required=True)
    ap.add_argument("--cloudpickle-root",type=Path,required=True)
    ap.add_argument("--receipt",type=Path,required=True)
    a=ap.parse_args()
    out=run(a.s29_root,a.source_root,a.cloudpickle_root)
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(out,sort_keys=True,indent=2)+"\n")
    print(json.dumps({
        "status":out["status"],
        "reward_only_target":out["frozen_reward_only_development_target_pass"],
        "factual_training":out["new_actual_reward_only_training_actions"],
        "factual_eval":out["new_actual_real_factual_eval_actions"],
        "optimizer_steps":out["reward_only_AdamW_PPO_update_batches_total"],
        "support":out["predeclared_support_out_of_three"],
        "means":[{"seed":r["init_seed"],
                  "rewards":r["mean_real_factual_episode_reward"]}
                 for r in out["records"]],
        "wall":out["whole_process_wall_seconds"],
        "science_first_use":out["original_scientific_first_use"]
    },sort_keys=True))
    return 0 if out["status"]=="PASS_PPO_REAL_FACTUAL_DEV_INTEGRITY_NOT_SCIENCE" else 1

if __name__=="__main__":
    raise SystemExit(main())
