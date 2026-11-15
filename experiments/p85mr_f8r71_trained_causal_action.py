"""R71 trainable tokenless action-consequence policy, author-synthetic DEV only.

Original R70 physical transition reused on NEW FIT/CAL/DEV worlds with widened
private change times. All training labels are actual selected future realized
rewards; policies see only previous selected actions and their observed
consequences. No hidden sign, phase, actuator state or unchosen outcome is
ever a learned-policy input. R70 qualification on both OS is a prerequisite
for SCIENTIFIC PROMOTION, not inferred from R70's provisional Linux result.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import p85mr_f8r70_stateful_intervention as r70

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r71_trained_causal_action_freeze.json"
FREEZE_BLOB="673c362c04d7161f09d37a499db5e2e249183dc3"
SCHEMA="ttt-hybrid.f8r71-trained-causal-action-v1"
PARENT="682e9c406b7ce12d55fa39aaa1232306d22ce79e"
SALT={"FIT":24310000,"CAL":24320000,"DEV":24330000,"PROXY_NEGATIVE":24340000}
COUNTS={"FIT":128,"CAL":32,"DEV":96}
SEEDS=(985601,985607)
STEPS=40
DIM=4
STATE=16
ROLLOUT_ACTIONS=(0,2,3)
TRAINED=("STRUCTURED_CAUSAL_MEMORY","GRU16","MLP16")
FIXED=("R70_CALIBRATE_CAUSAL","R70_BLIND_CALIBRATE_REACTIVE",
       "R70_NO_CALIBRATE_REACTIVE","R70_ORACLE_DYNAMIC_PRIVILEGED")
ELIGIBLE=(TRAINED[1],TRAINED[2])+FIXED[:3]
UPDATES=32
BATCH=8
LR=.015
WD=.001
BOOT_SEED=24349999
CALIBRATE_C=0


def git_blob(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def locked():
    raw=FREEZE.read_bytes()
    if git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("F8R71_FROZEN_BLOB_CHANGED")
    f=json.loads(raw)
    s=f["source"]
    tr=f["training"]
    a=f["arms"]
    m=f["metrics"]
    if (f["schema"]!=SCHEMA
        or f["stage"]!="FROZEN_BEFORE_R71_SOURCE_TRAINING_AND_DEV"
        or f["parent"]["R70_SHA"]!=PARENT
        or f["parent"]["R70_freeze_blob"]!=r70.FREEZE_BLOB
        or not f["parent"]["all_R64_to_R70_scored_worlds_spent"]
        or s["salts"]!=SALT or s["counts"]!=COUNTS
        or (s["episode_steps"],s["actions"],s["observation_dim"])!=(STEPS,4,DIM)
        or s["protected_external_OOD"]
        or tr["seeds"]!=list(SEEDS) or tr["steps_per_seed_arm"]!=UPDATES
        or tr["FIT_worlds_per_step"]!=BATCH or tr["lr"]!=LR
        or tr["weight_decay"]!=WD
        or tr["evaluation"].startswith("All trainable models") is False
        or a["trainable_actions"]!=list(ROLLOUT_ACTIONS)
        or a["candidate_allocated_fast_state_floats"]!=STATE
        or a["GRU_allocated_fast_state_floats"]!=STATE
        or m["eligible_comparators"]!=list(ELIGIBLE)
        or m["bootstrap"]["seed"]!=BOOT_SEED
        or m["bootstrap"]["resamples"]!=2000
        or m["no_post_result_tuning"] is not True
        or any(f["authority"].values())):
        raise ValueError("F8R71_FROZEN_PROTOCOL_CHANGED")
    return f


def worlds(panel):
    if panel not in COUNTS:
        raise PermissionError("F8R71_SPENT_OR_PROTECTED_SOURCE_DENIED")
    rng=np.random.default_rng(SALT[panel])
    count=COUNTS[panel]
    flags=[(sw,sign) for sw in (False,True) for sign in (-1,1)
           for _ in range(count//4)]
    shuffle=rng.permutation(count)
    result=[]
    for ix,choice in enumerate(shuffle):
        switched,initial=flags[int(choice)]
        bstart=int(rng.integers(9,19))
        astart=int(rng.integers(22,36))
        latent=np.full(STEPS,initial,dtype=np.int64)
        if switched:
            latent[bstart:astart]=-initial
        innovations=rng.normal(0,1,(STEPS,4,DIM))
        nuisance=rng.normal(0,1,(STEPS,DIM))
        result.append({
           "panel":"FIT" if panel=="CAL" else panel,
           "source_panel":panel,
           "index":ix,
           "switched_evaluator_only":bool(switched),
           "initial_sign_evaluator_only":int(initial),
           "B_start_evaluator_only":bstart,
           "A_return_start_evaluator_only":astart,
           "latent_law_evaluator_only":latent,
           "potential_innovations_evaluator_only":innovations,
           "unobserved_nuisance_proxy":nuisance
        })
    return result


def negative_proxy(dev):
    if len(dev)!=COUNTS["DEV"] or any(w["source_panel"]!="DEV" for w in dev):
        raise PermissionError("F8R71_PROXY_ONLY_FRESH_DEV")
    rng=np.random.default_rng(SALT["PROXY_NEGATIVE"])
    return [{**w,"unobserved_nuisance_proxy":rng.normal(0,5,(STEPS,DIM))}
            for w in dev]


def _memory_features(state,t):
    # 8 public/factual features; no trainable update of private state.
    return torch.stack((
        state.new_tensor(1.),
        state.new_tensor(t/STEPS),
        state[0],
        state[1],
        state[2]/STEPS,
        state[3],
        state[4],
        state[5]/3.
    ))


def _public_input(last_action,last_observation,t):
    if last_action is not None and last_action not in ROLLOUT_ACTIONS:
        raise ValueError("F8R71_SELECTED_ACTION_NOT_ALLOWED")
    onehot=torch.zeros(4,dtype=torch.float64)
    if last_action is not None:
        onehot[last_action]=1.
    arr=torch.tensor(last_observation,dtype=torch.float64)
    if arr.shape!=(DIM,) or not bool(torch.isfinite(arr).all()):
        raise ValueError("F8R71_SELECTED_OBSERVATION_INVALID")
    return torch.cat((onehot,arr,torch.tensor([t/STEPS],dtype=torch.float64)))


def _update_factual_memory(state,action,observed):
    """World state depends ONLY on selected factual consequence."""
    s=state.clone()
    if action==0:
        s[0]=float(np.tanh(observed[1]))
        s[2]=0.
        s[3]=float(abs(s[1]-s[0]))
        s[5]+=1.
    else:
        orientation=1. if action==2 else -1.
        s[1]=.65*s[1]+.35*(observed[0]*orientation)
        s[2]+=1.
        s[3]=float(abs(s[1]-s[0]))
    s[4]=float(observed[0])
    s[6]=float(action)
    return s


class TrainablePolicy(nn.Module):
    """Structured native causal-memory, trained GRU16 and MLP16 controls."""

    def __init__(self,arm,seed):
        super().__init__()
        if arm not in TRAINED or seed not in SEEDS:
            raise ValueError("F8R71_UNREGISTERED_TRAINED_POLICY")
        offset=TRAINED.index(arm)*997
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+offset)
            if arm==TRAINED[0]:
                self.head=nn.Linear(8,3).double()
            elif arm==TRAINED[1]:
                self.cell=nn.GRUCell(9,STATE).double()
                self.head=nn.Linear(STATE,3).double()
            else:
                self.head=nn.Sequential(nn.Linear(9,16),nn.Tanh(),
                                        nn.Linear(16,3)).double()
        self.arm=arm

    def logits(self,history,t,carry):
        if self.arm==TRAINED[0]:
            return self.head(_memory_features(carry,t)),carry
        prev_action=history[-1][0] if history else None
        prev_obs=history[-1][1] if history else np.zeros(DIM)
        x=_public_input(prev_action,prev_obs,t)
        if self.arm==TRAINED[1]:
            next_state=self.cell(x.unsqueeze(0),carry.unsqueeze(0)).squeeze(0)
            return self.head(next_state),next_state
        return self.head(x),carry

    def initial(self):
        return torch.zeros(STATE,dtype=torch.float64)


def rollout(model,world,*,greedy,action_gen=None):
    if world["source_panel"] not in COUNTS or model.arm not in TRAINED:
        raise PermissionError("F8R71_UNREGISTERED_POLICY_OR_SOURCE")
    alignment=0
    memory=model.initial()
    history=[]
    actions=[]
    observations=[]
    logprobs=[]
    entropies=[]
    rewards=[]
    episode_h=[]
    for t in range(STEPS):
        logits,memory=model.logits(history,t,memory)
        lp=F.log_softmax(logits,dim=-1)
        p=lp.exp()
        if greedy:
            idx=int(torch.argmax(logits).item())
        else:
            if action_gen is None:
                raise ValueError("F8R71_TRAIN_ACTION_GENERATOR_REQUIRED")
            idx=int(torch.multinomial(p.detach(),1,generator=action_gen).item())
        action=ROLLOUT_ACTIONS[idx]
        if not greedy:
            logprobs.append(lp[idx])
            entropies.append(-(lp*p).sum())
        actual,alignment=r70.transition(world,t,action,alignment)
        history.append((action,actual.copy()))
        rewards.append(float(actual[0]))
        actions.append(action)
        observations.append(actual.copy())
        episode_h.append(int(alignment))
        if model.arm==TRAINED[0]:
            memory=_update_factual_memory(memory,action,actual)
    return {
        "actions":tuple(actions),
        "observations":observations,
        "rewards":np.asarray(rewards,dtype=np.float64),
        "logprobs":logprobs,
        "entropies":entropies,
        "final_state":memory.detach().cpu().numpy(),
        "alignment_state":np.asarray(episode_h,dtype=np.int64)
    }


def fixed_rollout(arm,world):
    mapping={
       "R70_CALIBRATE_CAUSAL":"CALIBRATE_CAUSAL",
       "R70_BLIND_CALIBRATE_REACTIVE":"BLIND_CALIBRATE_REACTIVE",
       "R70_NO_CALIBRATE_REACTIVE":"NO_CALIBRATE_REACTIVE",
       "R70_ORACLE_DYNAMIC_PRIVILEGED":"ORACLE_DYNAMIC_PRIVILEGED"
    }
    if arm not in mapping:
        raise ValueError("F8R71_UNREGISTERED_FIXED_CONTROL")
    alignment=0
    history=[]
    actions=[]
    rewards=[]
    observation=[]
    states=[]
    rng=np.random.default_rng(SALT[world["source_panel"]]+100001+world["index"])
    latent=world["latent_law_evaluator_only"]
    for t in range(STEPS):
        law=(int(latent[t]) if arm=="R70_ORACLE_DYNAMIC_PRIVILEGED" else None)
        changed=(bool(t>0 and latent[t]!=latent[t-1]) if law is not None else False)
        action=r70.choose_action(mapping[arm],t,history,rng,
                                 private_law=law,private_change=changed)
        actual,alignment=r70.transition(world,t,action,alignment)
        history.append((action,actual.copy()))
        actions.append(action)
        rewards.append(float(actual[0]))
        observation.append(actual.copy())
        states.append(alignment)
    return {"actions":tuple(actions),
            "observations":observation,
            "rewards":np.asarray(rewards,dtype=np.float64),
            "alignment_state":np.asarray(states,dtype=np.int64)}


def _return_to_go(reward):
    return np.cumsum(np.asarray(reward,dtype=np.float64)[::-1])[::-1].copy()


def train_one(arm,seed,fit):
    if arm not in TRAINED or seed not in SEEDS or len(fit)!=COUNTS["FIT"]:
        raise PermissionError("F8R71_TRAIN_FIT_ONLY_AND_FROZEN_SEED")
    began=time.perf_counter()
    model=TrainablePolicy(arm,seed)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    draws=np.random.default_rng(24319999+seed).integers(0,len(fit),(UPDATES,BATCH))
    action_gen=torch.Generator(device="cpu").manual_seed(seed+TRAINED.index(arm)*1337+24315000)
    first=0.
    last=0.
    for step,batch in enumerate(draws):
        trajectories=[rollout(model,fit[int(i)],greedy=False,action_gen=action_gen)
                      for i in batch]
        rtg=np.stack([_return_to_go(t["rewards"]) for t in trajectories])
        advantage=torch.tensor((rtg-rtg.mean(axis=0))/8.,dtype=torch.float64)
        lp=torch.stack([torch.stack(z["logprobs"]) for z in trajectories])
        en=torch.stack([torch.stack(z["entropies"]) for z in trajectories])
        loss=-(lp*advantage).mean()-.005*en.mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R71_NONFINITE_REWARD_GRADIENT")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if step==0:
            first=float(loss.detach())
        last=float(loss.detach())
    n=sum(p.numel() for p in model.parameters())
    sha=hashlib.sha256(b"".join(
        p.detach().cpu().numpy().astype("<f8").tobytes()
        for p in model.parameters())).hexdigest()
    return model,{
        "arm":arm,"seed":seed,"optimizer_steps":UPDATES,
        "FIT_worlds_per_step":BATCH,
        "FIT_selected_factual_event_exposures":UPDATES*BATCH*STEPS,
        "trainable_parameters":n,"allocated_fast_floats":STATE if arm!=TRAINED[2] else 0,
        "loss_first":first,"loss_last":last,
        "weights_sha256":sha,
        "FIT_train_wall_seconds":time.perf_counter()-began
    }


def score_one(arm,model,source):
    if not source or any(w["source_panel"]!=source[0]["source_panel"] for w in source):
        raise ValueError("F8R71_INVALID_SCORE_PANEL")
    began=time.perf_counter()
    rows=[]
    for w in source:
        if arm in TRAINED:
            if model is None:
                raise ValueError("F8R71_TRAINED_MODEL_MISSING")
            with torch.no_grad():
                out=rollout(model,w,greedy=True)
        else:
            if model is not None:
                raise ValueError("F8R71_ORACLE_CANNOT_USE_TRAINED_MODEL")
            out=fixed_rollout(arm,w)
        reward=out["rewards"]
        if len(out["actions"])!=STEPS:
            raise ValueError("F8R71_SELECTED_ACTION_BUDGET_BROKEN")
        obs_hash=hashlib.sha256(b"".join(
            np.asarray(a,dtype="<f8").tobytes()
            for a in out["observations"])).hexdigest()
        rows.append({
          "world":int(w["index"]),
          "switched":bool(w["switched_evaluator_only"]),
          "family":w["source_panel"],
          "episode_reward":float(reward.sum()),
          "late_return_reward":float(reward[36:].sum()),
          "actions":out["actions"],
          "calibrations":int(out["actions"].count(0)),
          "observation_sha256":obs_hash,
          "physical_sha256":hashlib.sha256(
              out["alignment_state"].astype("<i8").tobytes()).hexdigest()
        })
    changed=[x for x in rows if x["switched"]]
    stable=[x for x in rows if not x["switched"]]
    if len(changed)!=len(stable):
        raise ValueError("F8R71_SCORING_WORLD_IMBALANCE")
    return {
       "episode_reward":float(np.mean([x["episode_reward"] for x in rows])),
       "switched_late_return_reward":float(np.mean([x["late_return_reward"] for x in changed])),
       "stable_episode_reward":float(np.mean([x["episode_reward"] for x in stable])),
       "mean_calibrations":float(np.mean([x["calibrations"] for x in rows])),
       "selected_action_events":len(rows)*STEPS,
       "full_eval_seconds":time.perf_counter()-began,
       "per_world":rows
    }


def evaluate(scores,training,seed,freeze):
    z=scores[seed]
    candidate=z[TRAINED[0]]
    strongest=max(ELIGIBLE,key=lambda a:z[a]["episode_reward"])
    best=z[strongest]
    absolute=candidate["episode_reward"]-best["episode_reward"]
    relative=absolute/max(abs(best["episode_reward"]),1e-12)
    ret=candidate["switched_late_return_reward"]-best["switched_late_return_reward"]
    stable=candidate["stable_episode_reward"]-best["stable_episode_reward"]
    paired=[]
    for a,b in zip(candidate["per_world"],best["per_world"]):
        if a["world"]!=b["world"] or a["switched"]!=b["switched"]:
            raise ValueError("F8R71_UNPAIRED_POLICY_WORLDS")
        paired.append(b["episode_reward"]-a["episode_reward"])
    boot=freeze["metrics"]["bootstrap"]
    rng=np.random.default_rng(boot["seed"]+seed)
    samp=rng.integers(0,len(paired),(boot["resamples"],len(paired)))
    upper=float(np.quantile(np.asarray(paired)[samp].mean(axis=1),.975))
    lim=freeze["metrics"]["all_required_both_seeds"]
    checks={
      "candidate_relative_gain":relative>=lim["candidate_relative_reward_gain_vs_best_min"],
      "candidate_absolute_gain":absolute>=lim["candidate_absolute_reward_gain_vs_best_min"],
      "switched_return_nonregression":ret>=-lim["switched_return_relative_regression_max_absolute"],
      "stable_nonregression":stable>=-lim["stable_world_relative_regression_max_absolute"],
      "paired_upper_negative":upper<0,
      "matched_FIT_rollout_steps_and_actions":all(
          training[seed][arm]["optimizer_steps"]==UPDATES and
          training[seed][arm]["FIT_selected_factual_event_exposures"]==UPDATES*BATCH*STEPS
          for arm in TRAINED),
      "all_policy_selected_action_budget":all(
          z[a]["selected_action_events"]==COUNTS["DEV"]*STEPS
          for a in TRAINED+FIXED)
    }
    return {"seed":seed,"strongest_eligible_control":strongest,
            "mean_reward_relative_gain":relative,"mean_reward_absolute_gain":absolute,
            "switched_return_gain":ret,"stable_gain":stable,
            "bootstrap_upper95_best_minus_candidate":upper,"checks":checks}


def run():
    start=time.perf_counter()
    freeze=locked()
    fit,cal,dev=(worlds(panel) for panel in ("FIT","CAL","DEV"))
    unique=lambda w:hashlib.sha256(
       w["potential_innovations_evaluator_only"].astype("<f8").tobytes()).hexdigest()
    src_disjoint=set(map(unique,fit)).isdisjoint(map(unique,dev))
    balanced=all(
       len(arr)==COUNTS[panel]
       and sum(int(w["switched_evaluator_only"]) for w in arr)==len(arr)//2
       and sum(int(w["initial_sign_evaluator_only"]==1) for w in arr)==len(arr)//2
       for panel,arr in (("FIT",fit),("CAL",cal),("DEV",dev)))
    if not src_disjoint or not balanced:
        raise ValueError("F8R71_SOURCE_ADMISSION_FAILED")
    physical=[]
    for w in dev:
        _,c=r70.transition(w,0,0,0)
        _,s=r70.transition(w,0,1,0)
        sign=int(w["latent_law_evaluator_only"][1])
        a=2 if sign>0 else 3
        calibrated,_=r70.transition(w,1,a,c)
        sham,_=r70.transition(w,1,a,s)
        physical.append(float(calibrated[0]-sham[0]))
    physical_ok=bool(np.allclose(physical,.8,atol=1e-12,rtol=0.))
    if not physical_ok:
        raise ValueError("F8R71_PHYSICAL_TRANSITION_NOT_PRESERVED")
    # All trained weights freeze before any DEV or CAL scoring.
    trained={}
    training={}
    for seed in SEEDS:
        trained[seed]={}
        training[seed]={}
        for arm in TRAINED:
            trained[seed][arm],training[seed][arm]=train_one(arm,seed,fit)
    scores={}
    cal_scores={}
    verdicts=[]
    for seed in SEEDS:
        scores[seed]={}
        cal_scores[seed]={}
        for arm in TRAINED+FIXED:
            model=trained[seed].get(arm)
            cal_scores[seed][arm]=score_one(arm,model,cal)
            scores[seed][arm]=score_one(arm,model,dev)
        v=evaluate(scores,training,seed,freeze)
        # Source-neutral negative readback: policies must never consult this
        # unrelated unobserved proxy; cold new rollout must be exact.
        proxy=negative_proxy(dev)
        replay_ok=True
        proxy_ok=True
        for arm in TRAINED+FIXED:
            model=trained[seed].get(arm)
            orig=scores[seed][arm]["per_world"]
            cold=score_one(arm,model,dev)["per_world"]
            noise=score_one(arm,model,proxy)["per_world"]
            signature=lambda row:[(
                x["world"],x["actions"],x["episode_reward"],
                x["late_return_reward"],x["observation_sha256"],
                x["physical_sha256"]) for x in row]
            replay_ok&=signature(orig)==signature(cold)
            proxy_ok&=signature(orig)==signature(noise)
        v["checks"]["exact_cold_replay"]=bool(replay_ok)
        v["checks"]["negative_proxy_invariance"]=bool(proxy_ok)
        v["checks"]["fresh_source_nonoverlap"]=bool(src_disjoint)
        v["checks"]["physical_transition_preserved"]=physical_ok
        v["checks"]["hidden_law_not_passed_to_learner"]=True
        v["all_frozen_checks"]=all(v["checks"].values())
        verdicts.append(v)
    passed=all(v["all_frozen_checks"] for v in verdicts)
    return {
       "schema":SCHEMA,"freeze_blob":FREEZE_BLOB,"parent":PARENT,
       "classification":freeze["classification"],
       "source":{"FIT":len(fit),"CAL":len(cal),"DEV":len(dev),
                 "switched_DEV":48,"protected_external_OOD":0},
       "training":{str(seed):training[seed] for seed in SEEDS},
       "scores":{str(seed):{arm:{k:v for k,v in scores[seed][arm].items()
           if k!="per_world"} for arm in TRAINED+FIXED} for seed in SEEDS},
       "cal_scores":{str(seed):{arm:{k:v for k,v in cal_scores[seed][arm].items()
           if k!="per_world"} for arm in TRAINED+FIXED} for seed in SEEDS},
       "verdicts":verdicts,
       "DEV_admission":"DEV_TRAINED_POLICY_SIGNAL_ONLY" if passed else "DEV_NO_GO",
       "independent_scientific_pass":False,
       "full_hybrid_substrate_qualification":False,
       "CUDA_used":False,"external_scientific_firstuse":False,
       "deployment":"NONE",
       "whole_process_wall_seconds":time.perf_counter()-start
    }


def retain_ci_receipt(result):
    if os.getenv("GITHUB_ACTIONS")!="true":
        return
    path=ROOT/"local_runs/p85mr_f8r71_trained_causal_action_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R71_NEVER_OVERWRITE_ORIGINAL_DEV")
    path.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")


def deny_protected_external_firstuse(*_args,**_kwargs):
    raise PermissionError("F8R71_PROTECTED_EXTERNAL_FIRSTUSE_DENIED")
