"""P32 public DEV: whether original reward-only PPO destroys initially useful actions.

Same actual original CartPole source, model initialization, temporal input,
source reset streams, 96x8x64 factual actions and PPO optimizer budget.
Original P20 and P30 sources never changed. One immutable frozen β=1.0
initial-policy KL ablation; critic-only actor-frozen control and no-update
reference. No expert decisions/true velocities/sealed-science first-use.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
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
from torch.nn import functional as F
from scripts import native_p20_reward_only_ppo_dev as p20
from scripts import native_p30_reward_motion_ppo_dev as p30

SCHEMA="ttt-hybrid.native-p32.reward-only-component-damage-dev.v1"
PROTOCOL=ROOT/"experiments/native_p32_component_damage_frozen.json"
SEEDS=(4507,4741,4759)
TRAIN_BASE=144000
EVAL_SEEDS=(145011,145019,145029,145037,145049,145059,145071,145089)
PROBE_SEEDS=range(146000,146064)
CONDITIONS=("original","short_pole_x0.7","long_pole_x1.3")
TRAIN_ARMS=("full_actor_recurrent_ppo","actor_head_only_ppo",
            "recurrent_only_ppo","critic_only_frozen_policy")
EVAL_ARMS=("initial_no_update",)+TRAIN_ARMS+(
           "finite_difference_public_two_frame","seeded_random")

def frozen():
    p=json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if (p.get("schema")!="ttt-hybrid.native-p32.actor-vs-recurrent-source-reward-frozen.v1"
        or p["phase"]!="FROZEN_BEFORE_IMPLEMENTATION_AND_PUBLIC_SOURCE_USE"
        or p["parent"]["sha"]!="af574a87a9c81e5dc1b5e703dacf315606494e0a"
        or p["source"]["protected_scientific_first_use"] is not False
        or p["training"]["model_seeds"]!=list(SEEDS)
        or p["training"]["train_original_source_reset_base"]!=TRAIN_BASE
        or p["training"]["arms"]!=list(TRAIN_ARMS)
        or p["training"]["source_actions_per_trained_arm_seed"]!=49152
        or p["training"]["optimizer_minibatches_per_trained_arm_seed"]!=576
        or p["evaluation"]["original_heldout_reset_seeds"]!=list(EVAL_SEEDS)
        or p["evaluation"]["conditions"]!=list(CONDITIONS)
        or p["evaluation"]["arms"]!=list(EVAL_ARMS)
        or p["evaluation"]["new_physical_probe_reset_seeds"]!=
            [PROBE_SEEDS.start,PROBE_SEEDS.stop-1]):
        raise RuntimeError("P32_FROZEN_COMPONENT_DAMAGE_PROTOCOL_DRIFT")
    return p

def require_dev():
    if os.environ.get("NATIVE_P32_DEV_ONLY")!="1":
        raise RuntimeError("P32_PROTECTED_FIRST_USE_DENIED")
    p=frozen()
    if (p20.ROLLOUTS,p20.NUM_ENVS,p20.STEPS_PER_ROLLOUT,
        p20.PPO_EPOCHS,p20.MINIBATCH,p20.PPO_LR,
        p20.GAMMA,p20.GAE_LAMBDA,p20.PPO_CLIP,p20.ENTROPY,
        p20.VALUE_COEF,p20.MAX_GRAD_NORM)!=(
            96,8,64,3,256,.0015,.99,.95,.2,.015,.5,1.0):
        raise RuntimeError("P32_ORIGINAL_PPO_SOURCE_OR_OPTIMIZER_DRIFT")
    if (p30.H,p30.DT,p30.MOTION_SCALE)!=(8,.02,(.5,1.0,.05,.05)):
        raise RuntimeError("P32_ORIGINAL_P30_MOTION_VIEW_DRIFT")
    return p

@contextmanager
def scoped_p30_seeds():
    """P30's seed fence is expanded only within this sequential source study."""
    old=p30.SEEDS
    try:
        p30.SEEDS=SEEDS
        yield
    finally:
        p30.SEEDS=old

def motion(history,seed):
    if seed not in SEEDS:
        raise ValueError("P32_UNREGISTERED_MODEL_SEED")
    return p30.view_tensor(history,"derived",seed)

def make_initial(seed):
    if seed not in SEEDS:
        raise ValueError("P32_UNREGISTERED_MODEL_INITIALIZATION")
    torch.manual_seed(seed)
    model=p30.RewardGRU10()
    model.eval()
    return model

def fixed_history_probe(gym):
    """Actual public source history; not fitting model, not an expert demonstration."""
    histories=[]
    digest=hashlib.sha256()
    real_actions=0
    for seed in PROBE_SEEDS:
        env=gym.make("CartPole-v1",render_mode=None)
        try:
            obs,_=env.reset(seed=seed)
            history=[p20.factual_row(obs,0,0.)]
            rng=random.Random(seed+133113)
            for n in range(16):
                histories.append([list(z) for z in history])
                a=rng.randrange(2)
                next_obs,reward,terminated,truncated,_=env.step(a)
                real_actions+=1
                new=p20.factual_row(next_obs,2*a-1,float(reward))
                digest.update(json.dumps([seed,n,history[-1],a,new,
                    bool(terminated),bool(truncated)],separators=(",",":"),
                    allow_nan=False).encode())
                history.append(new)
                if terminated or truncated:
                    break
        finally:
            env.close()
    if len(histories)<500:
        raise RuntimeError("P32_INSUFFICIENT_FIXED_SOURCE_DRIFT_PROBE")
    return histories,{
        "original_source_public_reset_seed_range":[PROBE_SEEDS.start,PROBE_SEEDS.stop-1],
        "original_source_executed_probe_actions":real_actions,
        "original_source_factual_probe_sha256":digest.hexdigest(),
        "model_was_not_trained_using_probe":True
    }

@torch.no_grad()
def policy_drift(initial,trained,histories,seed):
    initial.eval();trained.eval()
    before=[];after=[]
    for h in histories:
        x=motion(h,seed)
        p0,_=initial(x)
        p1,_=trained(x)
        before.append(p0.squeeze(0))
        after.append(p1.squeeze(0))
    left=torch.stack(before)
    right=torch.stack(after)
    same=torch.equal(left,right)
    KL=(0.0 if same else max(0.,float(F.kl_div(
          F.log_softmax(right,dim=-1),
          F.softmax(left,dim=-1),reduction="batchmean"))))
    agreement=float((left.argmax(dim=1)==right.argmax(dim=1)).float().mean())
    return {"original_source_probe_histories":len(histories),
            "initial_to_trained_actor_policy_KL":KL,
            "greedy_action_agreement_with_initial":agreement,
            "bit_exact_original_logits":same,
            "no_probe_fit_or_training":True}

def train_arm(gym,seed,arm):
    if seed not in SEEDS or arm not in TRAIN_ARMS:
        raise ValueError("P32_UNREGISTERED_SOURCE_REWARD_TRAIN")
    model=make_initial(seed)
    reference=copy.deepcopy(model)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    init_sha=p20.model_sha(reference)
    for name,p in model.named_parameters():
        frozen=(
          (arm=="critic_only_frozen_policy" and
           (name.startswith("recurrent.") or name.startswith("actor.")))
          or (arm=="actor_head_only_ppo" and name.startswith("recurrent."))
          or (arm=="recurrent_only_ppo" and name.startswith("actor."))
        )
        if frozen:
            p.requires_grad_(False)
    optim_params=[par for par in model.parameters() if par.requires_grad]
    optimizer=torch.optim.AdamW(optim_params,lr=p20.PPO_LR,eps=1e-5)
    effective_parameters=sum(x.numel() for x in optim_params)

    envs=[]
    histories=[]
    episode_count=[0]*p20.NUM_ENVS
    episode_returns=[0.]*p20.NUM_ENVS
    closed_returns=[]
    source_trace=hashlib.sha256()
    resets_trace=hashlib.sha256()
    first_rollout_trace=hashlib.sha256()
    action_count=0
    optimizer_updates=0
    cumulative_gradient_norm=0.
    snapshots=[]
    started=time.perf_counter()
    for i in range(p20.NUM_ENVS):
        e=gym.make("CartPole-v1",render_mode=None)
        reset_seed=TRAIN_BASE+1000*i
        obs,_=e.reset(seed=reset_seed)
        resets_trace.update(str(reset_seed).encode())
        histories.append([p20.factual_row(obs,0,0.)])
        envs.append(e)
    try:
        for rollout in range(p20.ROLLOUTS):
            xs=[];actions=[];old_lp=[];old_values=[];rewards=[];dones=[]
            model.eval()
            for t in range(p20.STEPS_PER_ROLLOUT):
                tensor=torch.cat([motion(h,seed) for h in histories])
                with torch.no_grad():
                    logits,value=model(tensor)
                    sampler=torch.distributions.Categorical(logits=logits)
                    selected=sampler.sample()
                    lp=sampler.log_prob(selected)
                observed_rewards=[];observed_dones=[]
                for j,env in enumerate(envs):
                    action=int(selected[j])
                    next_obs,reward,terminated,truncated,_=env.step(action)
                    reward=float(reward)
                    if not math.isfinite(reward):
                        raise RuntimeError("P32_NONFINITE_REAL_SOURCE_REWARD")
                    action_count+=1
                    done=bool(terminated or truncated)
                    actual=p20.factual_row(next_obs,2*action-1,reward)
                    event=json.dumps([histories[j][-1],action,actual,done],
                        separators=(",",":"),allow_nan=False).encode()
                    source_trace.update(event)
                    if rollout==0:
                        first_rollout_trace.update(event)
                    histories[j].append(actual)
                    episode_returns[j]+=reward
                    observed_rewards.append(reward)
                    observed_dones.append(done)
                    if done:
                        closed_returns.append(episode_returns[j])
                        episode_returns[j]=0.
                        episode_count[j]+=1
                        next_seed=TRAIN_BASE+1000*j+episode_count[j]
                        resets_trace.update(str(next_seed).encode())
                        reset,_=env.reset(seed=next_seed)
                        histories[j]=[p20.factual_row(reset,0,0.)]
                xs.append(tensor)
                actions.append(selected.detach())
                old_lp.append(lp.detach())
                old_values.append(value.detach())
                rewards.append(torch.tensor(observed_rewards,dtype=torch.float32))
                dones.append(torch.tensor(observed_dones,dtype=torch.float32))
            with torch.no_grad():
                x_end=torch.cat([motion(h,seed) for h in histories])
                _,bootstrap=model(x_end)
            rews=torch.stack(rewards)
            ends=torch.stack(dones)
            estimated=torch.stack(old_values)
            next_v=bootstrap
            advantage=torch.zeros_like(rews)
            last=torch.zeros(p20.NUM_ENVS)
            for t in reversed(range(p20.STEPS_PER_ROLLOUT)):
                mask=1.-ends[t]
                delta=rews[t]+p20.GAMMA*next_v*mask-estimated[t]
                last=delta+p20.GAMMA*p20.GAE_LAMBDA*mask*last
                advantage[t]=last
                next_v=estimated[t]
            ret=advantage+estimated
            actor_adv=(advantage-advantage.mean())/advantage.std(
                unbiased=False).clamp(min=1e-6)
            flat_x=torch.cat(xs)
            flat_actions=torch.stack(actions).reshape(-1)
            flat_lp=torch.stack(old_lp).reshape(-1)
            flat_adv=actor_adv.reshape(-1).detach()
            flat_returns=ret.reshape(-1).detach()
            model.train()
            generator=torch.Generator().manual_seed(seed+77*rollout+16111)
            for epoch in range(p20.PPO_EPOCHS):
                order=torch.randperm(len(flat_actions),generator=generator)
                for offset in range(0,len(flat_actions),p20.MINIBATCH):
                    ix=order[offset:offset+p20.MINIBATCH]
                    logits,value=model(flat_x[ix])
                    dist=torch.distributions.Categorical(logits=logits)
                    if arm=="critic_only_frozen_policy":
                        loss=p20.VALUE_COEF*F.smooth_l1_loss(value,flat_returns[ix])
                    else:
                        ratio=(dist.log_prob(flat_actions[ix])-flat_lp[ix]).exp()
                        s1=ratio*flat_adv[ix]
                        s2=ratio.clamp(1-p20.PPO_CLIP,1+p20.PPO_CLIP)*flat_adv[ix]
                        actor_loss=-torch.minimum(s1,s2).mean()
                        critic_loss=F.smooth_l1_loss(value,flat_returns[ix])
                        loss=actor_loss+p20.VALUE_COEF*critic_loss-(
                            p20.ENTROPY*dist.entropy().mean())

                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("P32_NONFINITE_SOURCE_PPO_LOSS")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm=nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        p20.MAX_GRAD_NORM)
                    if not bool(torch.isfinite(grad_norm)):
                        raise RuntimeError("P32_NONFINITE_ACTOR_CRITIC_GRAD")
                    cumulative_gradient_norm+=float(grad_norm)
                    optimizer.step()
                    optimizer_updates+=1
            if (rollout+1)%16==0:
                recent=closed_returns[-min(32,len(closed_returns)):]
                snapshots.append({
                    "rollouts":rollout+1,"actual_actions":action_count,
                    "finished_original_source_episodes":len(closed_returns),
                    "recent_32_original_source_episode_mean_return":
                        sum(recent)/len(recent) if recent else None,
                    "final_PPO_loss":float(loss.detach())})
    finally:
        for e in envs:
            e.close()
    model.eval()
    state=io.BytesIO()
    torch.save(model.state_dict(),state)
    data=state.getvalue()
    torch.manual_seed(seed+728800)
    cold=p30.RewardGRU10()
    cold.load_state_dict(torch.load(io.BytesIO(data),
                                    weights_only=True,map_location="cpu"),
                         strict=True)
    cold.eval()
    sentinel=motion([p20.factual_row([.02,.03,.04,.05],0,0.)],seed)
    with torch.no_grad():
        logits,values=model(sentinel)
        re_logits,re_values=cold(sentinel)
    cold_equal=torch.equal(logits,re_logits) and torch.equal(values,re_values)
    if not cold_equal:
        raise RuntimeError("P32_COLD_RELOADED_REWARD_POLICY_NOT_EXACT")
    components={
        "recurrent":tuple(k for k in reference.state_dict()
                          if k.startswith("recurrent.")),
        "actor":tuple(k for k in reference.state_dict()
                      if k.startswith("actor.")),
        "critic":tuple(k for k in reference.state_dict()
                       if k.startswith("critic."))
    }
    unchanged={name:all(torch.equal(reference.state_dict()[k],
                                   cold.state_dict()[k])
                        for k in keys)
               for name,keys in components.items()}
    expected_frozen={
        "full_actor_recurrent_ppo":(),
        "actor_head_only_ppo":("recurrent",),
        "recurrent_only_ppo":("actor",),
        "critic_only_frozen_policy":("recurrent","actor")
    }[arm]
    if not all(unchanged[name] for name in expected_frozen):
        raise RuntimeError("P32_FROZEN_COMPONENT_MUTATED")
    if unchanged["critic"]:
        raise RuntimeError("P32_REWARD_VALUE_HEAD_DID_NOT_UPDATE")

    return cold,reference,{
       "model_seed":seed,"arm":arm,
       "initial_source_actor_weights_sha256":init_sha,
       "trained_source_actor_critic_weights_sha256":p20.model_sha(cold),
       "cold_exact_logits_and_values":cold_equal,
       "cold_checkpoint_sha256":hashlib.sha256(data).hexdigest(),
       "real_original_train_source_actions":action_count,
       "original_ppo_minibatch_updates":optimizer_updates,
       "original_source_reset_stream_sha256":resets_trace.hexdigest(),
       "original_actual_action_consequence_sha256":source_trace.hexdigest(),
       "first_512_real_source_actions_trace_sha256":first_rollout_trace.hexdigest(),
       "total_real_original_source_closed_episodes":len(closed_returns),
       "trained_parameters_effective":effective_parameters,
       "all_model_parameters":sum(p.numel() for p in cold.parameters()),
       "components_bit_exact_unchanged_from_original":unchanged,
       "frozen_component_names":list(expected_frozen),
       "critic_only_actor_recurrent_frozen":arm=="critic_only_frozen_policy",
       "original_PPO_hyperparameters_and_reward_only_unchanged":True,
       "mean_original_ppo_unclipped_gradient_norm":
         cumulative_gradient_norm/optimizer_updates,
       "train_curve_actual_source_episodes":snapshots,
       "learning_executed_real_actions_without_expert":True,
       "source_train_wall_seconds":time.perf_counter()-started
    }

def conventional_action(history):
    return p30.finite_difference_public(history)

@torch.no_grad()
def evaluate_episode(gym,model,seed,condition,init,arm):
    if (seed not in EVAL_SEEDS or condition not in CONDITIONS
        or init not in SEEDS or arm not in EVAL_ARMS):
        raise ValueError("P32_UNREGISTERED_EVALUATION_WORLD")
    env=gym.make("CartPole-v1",render_mode=None)
    try:
        descriptor=p20.set_public_environment_dynamics(
            env,{"original":"original","short_pole_x0.7":"short_pole",
                 "long_pole_x1.3":"long_pole"}[condition])
        obs,_=env.reset(seed=seed)
        history=[p20.factual_row(obs,0,0.)]
        initial=list(history[0])
        rng=random.Random(seed+24113*init)
        sha=hashlib.sha256()
        actions=[]
        rewards=[]
        for _ in range(p20.EVAL_HORIZON):
            if arm=="finite_difference_public_two_frame":
                action=conventional_action(history)
            elif arm=="seeded_random":
                action=rng.randrange(2)
            else:
                logits,_=model(motion(history,init))
                action=int(logits.argmax(dim=1))
            obs,reward,term,trunc,_=env.step(action)
            current=p20.factual_row(obs,2*action-1,float(reward))
            sha.update(json.dumps([history[-1],action,current,bool(term),
                bool(trunc)],separators=(",",":"),allow_nan=False).encode())
            history.append(current)
            actions.append(action);rewards.append(float(reward))
            if term or trunc:
                break
        return {
          "original_source_reset_seed":seed,
          "source_condition":condition,
          "pole_length_ratio":descriptor["pole_length_ratio"],
          "source_original_initial_observation":initial,
          "actual_source_reward_return":sum(rewards),
          "actual_source_actions":len(actions),
          "source_executed_action_choices":actions,
          "source_real_action_consequence_sha256":sha.hexdigest(),
          "true_velocity_seen_by_model":False,
          "future_source_observation_given_to_policy":False,
          "source_eval_model_optimizer_updates":0
        }
    finally:
        env.close()

def run(s29_root,source_root,cloudpickle_root):
    contract=require_dev()
    start=time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    gym,admission=p20.admit_gymnasium(s29_root,source_root,cloudpickle_root)
    with scoped_p30_seeds():
        probe,probe_receipt=fixed_history_probe(gym)
        records=[]
        for seed in SEEDS:
            initial=make_initial(seed)
            original_initial_sha=p20.model_sha(initial)
            baseline={}
            for condition in CONDITIONS:
                baseline[condition]=[
                    evaluate_episode(gym,initial,s,condition,seed,"initial_no_update")
                    for s in EVAL_SEEDS]
            models={}
            trained_receipts={}
            initial_copies={}
            for arm in TRAIN_ARMS:
                model,original,r=train_arm(gym,seed,arm)
                if p20.model_sha(original)!=original_initial_sha or r[
                    "initial_source_actor_weights_sha256"]!=original_initial_sha:
                    raise RuntimeError("P32_DIFFERENT_INITIAL_POLICY_ACROSS_ARMS")
                models[arm]=model
                initial_copies[arm]=original
                trained_receipts[arm]=r
            init_first={r["first_512_real_source_actions_trace_sha256"]
                for r in trained_receipts.values()}
            if len(init_first)!=1:
                raise RuntimeError("P32_IDENTICAL_START_ACTUAL_SOURCE_ROLLOUT_DRIFT")
            drift={arm:policy_drift(initial,model,probe,seed)
                   for arm,model in models.items()}
            if (not drift["critic_only_frozen_policy"]["bit_exact_original_logits"]
                or drift["critic_only_frozen_policy"][
                   "initial_to_trained_actor_policy_KL"]!=0.):
                raise RuntimeError("P32_CRITIC_ONLY_CHANGED_INITIAL_ACTOR")
            evaluations=[]
            for condition in CONDITIONS:
                for world in EVAL_SEEDS:
                    episode={}
                    first=[]
                    for arm in EVAL_ARMS:
                        if arm=="initial_no_update":
                            result=next(x for x in baseline[condition]
                                        if x["original_source_reset_seed"]==world)
                        else:
                            result=evaluate_episode(
                                gym,models.get(arm),world,condition,seed,arm)
                        episode[arm]=result
                        first.append(tuple(result["source_original_initial_observation"]))
                    if len(set(first))!=1:
                        raise RuntimeError("P32_DIFFERENT_INITIAL_OBSERVATION_IN_PAIRED_WORLD")
                    if (episode["initial_no_update"]["source_real_action_consequence_sha256"]!=
                        episode["critic_only_frozen_policy"]["source_real_action_consequence_sha256"]):
                        raise RuntimeError("P32_CRITIC_ONLY_ACTUAL_SOURCE_ACTION_TRACE_CHANGED")
                    evaluations.append({
                        "source_condition":condition,
                        "source_reset_seed":world,
                        "source_episodes_from_actually_executed_actions":episode
                    })
            means={
                cond:{arm:sum(e["source_episodes_from_actually_executed_actions"][arm][
                          "actual_source_reward_return"]
                        for e in evaluations if e["source_condition"]==cond)/len(EVAL_SEEDS)
                      for arm in EVAL_ARMS}
                for cond in CONDITIONS}
            o=means["original"]
            initial_admitted=o["initial_no_update"]>=150
            full_damage=(initial_admitted and
                o["initial_no_update"]-o["full_actor_recurrent_ppo"]>=25)
            component_damage={
                arm:initial_admitted and
                    o["initial_no_update"]-o[arm]>=25
                for arm in ("actor_head_only_ppo","recurrent_only_ppo")
            }
            retention={
                arm:(initial_admitted and full_damage
                     and o[arm]>=o["initial_no_update"]-20
                     and all(
                        means[condition][arm]>=
                        means[condition]["full_actor_recurrent_ppo"]+25
                        for condition in CONDITIONS))
                for arm in ("actor_head_only_ppo","recurrent_only_ppo")
            }
            component_class=(
                "NO_FROZEN_FULL_PPO_DAMAGE" if not full_damage else
                "BOTH_COMPONENTS_INDIVIDUALLY_SUFFICIENT" if all(component_damage.values()) else
                "ACTOR_HEAD_UPDATES_SUFFICIENT" if component_damage["actor_head_only_ppo"] else
                "RECURRENT_UPDATES_SUFFICIENT" if component_damage["recurrent_only_ppo"] else
                "ONLY_JOINT_UPDATES_DAMAGE_OR_ON_POLICY_INTERACTION")
            acquired={
                arm:(
                    o[arm]>=150
                    and all(means[condition][arm]>=100 and
                        means[condition][arm]>=
                        means[condition]["full_actor_recurrent_ppo"]+25
                        for condition in CONDITIONS))
                for arm in ("actor_head_only_ppo","recurrent_only_ppo")
            }
            records.append({
                "model_seed":seed,
                "selected_historically_favorable_initialization":seed==4507,
                "initial_source_policy_weights_sha256":original_initial_sha,
                "equal_first_actual_512_source_actions_sha256":next(iter(init_first)),
                "actual_train_receipts":trained_receipts,
                "source_real_eval_episodes":evaluations,
                "real_source_reward_means_by_condition":means,
                "source_factual_drift_probe":drift,
                "fresh_initial_source_qualified_for_damage_at_150":initial_admitted,
                "full_PPO_damage_25_on_fresh_original":full_damage,
                "component_sufficient_for_selected_initial_damage_25":component_damage,
                "component_retained_selected_initial_20_and_vs_full_25":retention,
                "frozen_component_diagnostic_class":component_class,
                "all_condition_general_acquisition_gates":acquired,
                "no_heldout_training_or_epoch_selection":True
            })
    wall=time.perf_counter()-start
    true_actions=sum(r["real_original_train_source_actions"]
                     for rec in records for r in rec["actual_train_receipts"].values())
    updates=sum(r["original_ppo_minibatch_updates"]
                for rec in records for r in rec["actual_train_receipts"].values())
    good=(wall<=contract["qualification"]["max_public_source_process_seconds"]
          and true_actions==3*4*49152 and updates==3*4*576
          and all(rec["source_factual_drift_probe"]["critic_only_frozen_policy"][
              "bit_exact_original_logits"] for rec in records))
    all_acquisition=all(
        all(rec["all_condition_general_acquisition_gates"].values())
        for rec in records)
    return {
      "schema":SCHEMA,
      "engineering":"PASS_ORIGINAL_PUBLIC_DEV_ONLY" if good else "FAIL_P32_ENGINEERING",
      "original_source_admission":admission,
      "protected_scientific_first_use":"NOT_RUN",
      "original_P31_positive_and_negative_results_unchanged":True,
      "independent_P30_P26_lanes_unmodified":True,
      "no_model_promotion_or_FORGE_activation":True,
      "frozen_component_general_acquisition_all_three_seeds":all_acquisition,
      "selected_seed_4507_original_initial_admitted":
          records[0]["fresh_initial_source_qualified_for_damage_at_150"],
      "selected_seed_4507_full_PPO_damage":
          records[0]["full_PPO_damage_25_on_fresh_original"],
      "selected_seed_4507_actor_head_only_damage":
          records[0]["component_sufficient_for_selected_initial_damage_25"][
              "actor_head_only_ppo"],
      "selected_seed_4507_recurrent_only_damage":
          records[0]["component_sufficient_for_selected_initial_damage_25"][
              "recurrent_only_ppo"],
      "selected_seed_4507_attribution":
          records[0]["frozen_component_diagnostic_class"],
      "original_source_probe":probe_receipt,
      "total_original_real_training_actions":true_actions,
      "total_original_source_optimizer_minibatches":updates,
      "total_original_real_eval_actions":sum(
          item["actual_source_actions"] for rec in records
          for block in rec["source_real_eval_episodes"]
          for item in block["source_episodes_from_actually_executed_actions"].values()),
      "whole_real_source_wall_seconds":wall,
      "records":records
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--s29-root",type=Path,required=True)
    parser.add_argument("--source-root",type=Path,required=True)
    parser.add_argument("--cloudpickle-root",type=Path,required=True)
    parser.add_argument("--receipt",type=Path,required=True)
    args=parser.parse_args()
    result=run(args.s29_root,args.source_root,args.cloudpickle_root)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",
                           encoding="utf-8")
    summaries=[{
      "model_seed":r["model_seed"],
      "real_reward_means_all_conditions":r["real_source_reward_means_by_condition"],
      "probe_actor_KL":{
          arm:x["initial_to_trained_actor_policy_KL"]
          for arm,x in r["source_factual_drift_probe"].items()},
      "selected_initial_admitted":r["fresh_initial_source_qualified_for_damage_at_150"],
      "full_PPO_damage_25":r["full_PPO_damage_25_on_fresh_original"],
      "actor_head_only_damage_25":
          r["component_sufficient_for_selected_initial_damage_25"]["actor_head_only_ppo"],
      "recurrent_only_damage_25":
          r["component_sufficient_for_selected_initial_damage_25"]["recurrent_only_ppo"],
      "attribution":r["frozen_component_diagnostic_class"],
      "component_retention":r["component_retained_selected_initial_20_and_vs_full_25"],
      "broad_reward_acquisition":r["all_condition_general_acquisition_gates"]
    } for r in result["records"]]
    print("P32_FROZEN_COMPONENT_DAMAGE_SOURCE_RESULT="+json.dumps({
       "engineering":result["engineering"],
       "selected_initial_admitted":result["selected_seed_4507_original_initial_admitted"],
       "selected_full_PPO_damages":result["selected_seed_4507_full_PPO_damage"],
       "selected_component_attribution":result["selected_seed_4507_attribution"],
       "all_seeds_general_acquisition":
          result["frozen_component_general_acquisition_all_three_seeds"],
       "source_training_actions":result["total_original_real_training_actions"],
       "minibatch_updates":result["total_original_source_optimizer_minibatches"],
       "real_eval_actions":result["total_original_real_eval_actions"],
       "wall_seconds":result["whole_real_source_wall_seconds"],
       "seeds":summaries
    },sort_keys=True))
    return 0 if result["engineering"]=="PASS_ORIGINAL_PUBLIC_DEV_ONLY" else 1

if __name__=="__main__":
    raise SystemExit(main())
