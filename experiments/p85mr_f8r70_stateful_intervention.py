"""R70 prospectively frozen causal action task with persistent physical actuator.

Development-only synthetic task admission. A real selected CALIBRATE action
changes future actuator coupling AND emits factual sign evidence; an equally
costly blind calibration control receives the same physical transition but
may infer the sign only from ordinary selected exploit rewards. An
independently controlled do(CALIBRATE) vs do(SHAM) experiment demonstrates
that future attainable selected reward differs even with identical next
chosen action and identical noise. No policy except explicitly privileged
ORACLE_DYNAMIC receives private signs, regime labels or counterfactual
observations.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r70_stateful_intervention_freeze.json"
FREEZE_BLOB="3e86a43a0a39dd3f5ed7981218e5d544c98c296e"
SCHEMA="ttt-hybrid.f8r70-stateful-intervention-task-admission.v1"
PARENT="efe7081d1a32e5ce96236bdf0a5c9024658d9afb"
SALT={"FIT":24210000,"DEV":24230000,"PROXY_NEGATIVE":24240000}
COUNTS={"FIT":64,"DEV":96}
STEPS=40
ACTIONS=4
DIM=4
PROBES=(0,16,32)
ALPHA=.35
BOOT_SEED=24249999
POLICIES=("CALIBRATE_CAUSAL","BLIND_CALIBRATE_REACTIVE",
          "NO_CALIBRATE_REACTIVE","SHAM_PROXY","RANDOM",
          "ORACLE_DYNAMIC_PRIVILEGED")


def git_blob(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def locked():
    raw=FREEZE.read_bytes()
    if git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("F8R70_ORIGINAL_FREEZE_DRIFT")
    f=json.loads(raw)
    s,p,m=f["source"],f["policies"],f["metrics"]
    if (f["schema"]!=SCHEMA or
        f["stage"]!="PROSPECTIVE_FREEZE_BEFORE_SOURCE_TEST_AND_DEV_SCORING"
        or f["parent"]["R66_SHA"]!=PARENT
        or f["parent"]["R69_result"]!="TASK_ADMISSION_NO_GO"
        or f["parent"]["no_spent_R64_through_R69_scored_cohorts"] is not True
        or s["salts"]!=SALT or s["counts"]!=COUNTS
        or s["episode_steps"]!=STEPS or s["action_count"]!=ACTIONS
        or s["terminal_dims"]!=DIM
        or s["protected_OOD"] or s["external_first_use"]
        or p["fixed_calibration_public_times"]!=list(PROBES)
        or p["reward_reactive_ema_alpha"]!=ALPHA
        or p["same_episode_steps"]!=STEPS
        or p["no_training_or_hidden_oracle_for_nonprivileged"] is not True
        or m["bootstrap"]["seed"]!=BOOT_SEED
        or m["bootstrap"]["resamples"]!=2000
        or m["never_tune_on_scored_DEV"] is not True
        or any(f["authority"].values())):
        raise ValueError("F8R70_FROZEN_CONTRACT_DRIFT")
    return f


def worlds(panel):
    if panel not in COUNTS:
        raise PermissionError("F8R70_OLD_OR_PROTECTED_SOURCE_DENIED")
    rng=np.random.default_rng(SALT[panel])
    count=COUNTS[panel]
    labels=[(sw,sign) for sw in (False,True) for sign in (-1,1)
            for _ in range(count//4)]
    perm=rng.permutation(count)
    out=[]
    for index,which in enumerate(perm):
        switched,initial=labels[int(which)]
        b0=int(rng.integers(12,16))
        a0=int(rng.integers(28,32))
        hidden=np.full(STEPS,initial,dtype=np.int64)
        if switched:
            hidden[b0:a0]=-initial
        innovation=rng.normal(0,1,(STEPS,ACTIONS,DIM))
        unrelated=rng.normal(0,1,(STEPS,DIM))
        out.append({
            "panel":panel,"index":int(index),
            "switched_evaluator_only":bool(switched),
            "initial_sign_evaluator_only":int(initial),
            "B_start_evaluator_only":b0,
            "A_return_start_evaluator_only":a0,
            "latent_law_evaluator_only":hidden,
            "potential_innovations_evaluator_only":innovation,
            "unobserved_nuisance_proxy":unrelated
        })
    return out


def negative_proxy(dev):
    if len(dev)!=COUNTS["DEV"] or any(w["panel"]!="DEV" for w in dev):
        raise PermissionError("F8R70_PROXY_REQUIRES_ORIGINAL_DEV")
    rng=np.random.default_rng(SALT["PROXY_NEGATIVE"])
    return [{**w,"unobserved_nuisance_proxy":rng.normal(0,5,(STEPS,DIM))}
            for w in dev]


def transition(world,t,action,alignment):
    """Physical selected transition; private causal laws exist only inside environment."""
    if world["panel"] not in COUNTS or not 0<=t<STEPS or action not in range(ACTIONS):
        raise ValueError("F8R70_BAD_ENV_ACTION_OR_SOURCE")
    if alignment not in (-1,0,1):
        raise ValueError("F8R70_INVALID_ACTUATOR_STATE")
    m=int(world["latent_law_evaluator_only"][t])
    noise=world["potential_innovations_evaluator_only"][t,action]
    if noise.shape!=(DIM,) or not np.isfinite(noise).all():
        raise ValueError("F8R70_BAD_ENV_NOISE")
    obs=np.empty(DIM,dtype=np.float64)
    nxt=int(alignment)
    if action==0:
        # Causal physical actuator reset, not merely a sign-observation query.
        obs[0]=-.35+.02*noise[0]
        obs[1]=m+.10*noise[1]
        obs[2]=noise[2]
        nxt=m
    elif action==1:
        # Same cost, no repair; spurious shortcut reverses on new DEV worlds.
        obs[0]=-.35+.02*noise[0]
        obs[1]=noise[1]
        obs[2]=(m if world["panel"]=="FIT" else -m)+.10*noise[2]
    else:
        choice=1 if action==2 else -1
        if alignment==m:
            gain,penalty=1.,0.
        elif alignment==0:
            gain,penalty=.35,-.15
        else:
            gain,penalty=.35,-.65
        obs[0]=gain*choice*m+penalty+.70*noise[0]
        obs[1]=noise[1]
        obs[2]=noise[2]
    obs[3]=noise[3]
    return obs,nxt


def choose_action(policy,t,history,rng,*,private_law=None,private_change=False):
    if policy not in POLICIES:
        raise ValueError("F8R70_UNREGISTERED_POLICY")
    if len(history)!=t:
        raise ValueError("F8R70_FACTUAL_HISTORY_LENGTH_MISMATCH")
    if policy=="ORACLE_DYNAMIC_PRIVILEGED":
        if private_law not in (-1,1):
            raise PermissionError("F8R70_ORACLE_REQUIRES_PRIVATE_LAW")
        if t==0 or private_change:
            return 0
        return 2 if private_law>0 else 3
    if private_law is not None or private_change:
        raise PermissionError("F8R70_NONORACLE_PRIVATE_LAW_DENIED")
    if policy=="RANDOM":
        return int(rng.integers(0,ACTIONS))
    if policy=="CALIBRATE_CAUSAL":
        if t in PROBES:
            return 0
        diag=[obs[1] for a,obs in history if a==0]
        if not diag:
            raise ValueError("F8R70_MISSING_SELECTED_CAUSAL_DIAGNOSTIC")
        return 2 if diag[-1]>=0 else 3
    if policy=="SHAM_PROXY":
        if t in PROBES:
            return 1
        proxy=[obs[2] for a,obs in history if a==1]
        if not proxy:
            raise ValueError("F8R70_MISSING_SELECTED_SHAM_PROBE")
        return 2 if proxy[-1]>=0 else 3
    if policy=="BLIND_CALIBRATE_REACTIVE" and t in PROBES:
        return 0
    belief=0.
    for action,obs in history:
        # Reward-only controls ignore all diagnostic and spurious signal
        # coordinates, even after receiving identical physical repairs.
        if action in (2,3):
            orientation=1. if action==2 else -1.
            belief=(1-ALPHA)*belief+ALPHA*orientation*obs[0]
    return 2 if belief>=0 else 3


def run_policy(policy,world):
    if policy not in POLICIES or world["panel"] not in COUNTS:
        raise ValueError("F8R70_INVALID_POLICY_OR_SOURCE")
    latent=world["latent_law_evaluator_only"]
    if latent.shape!=(STEPS,) or world["potential_innovations_evaluator_only"].shape!=(STEPS,ACTIONS,DIM):
        raise ValueError("F8R70_BAD_PRIVATE_WORLD")
    rng=np.random.default_rng(SALT[world["panel"]]+100001+world["index"])
    h=0
    history=[]
    actions=[]
    rewards=[]
    correct=[]
    physical_states=[]
    for t in range(STEPS):
        private=int(latent[t]) if policy=="ORACLE_DYNAMIC_PRIVILEGED" else None
        changed=bool(t and latent[t]!=latent[t-1]) if private is not None else False
        action=choose_action(policy,t,history,rng,private_law=private,private_change=changed)
        selected,h=transition(world,t,action,h)
        history.append((int(action),selected.copy()))
        actions.append(int(action))
        rewards.append(float(selected[0]))
        physical_states.append(int(h))
        if action==0 and t in PROBES:
            correct.append(bool((1 if selected[1]>=0 else -1)==int(latent[t])))
    reward=np.asarray(rewards,dtype=np.float64)
    digest=hashlib.sha256(b"".join(
        np.asarray(obs,dtype="<f8").tobytes() for _,obs in history)).hexdigest()
    return {
       "world":int(world["index"]),
       "switched":bool(world["switched_evaluator_only"]),
       "action_count":len(actions),
       "selected_actions":tuple(actions),
       "calibration_count":int(actions.count(0)),
       "sham_count":int(actions.count(1)),
       "factual_probe_sign_correct":tuple(correct),
       "mean_episode_reward":float(reward.sum()),
       "postreturn_reward":float(reward[32:].sum()),
       "selected_observation_sha256":digest,
       "physical_alignment_after_last_action":int(h),
       "physical_alignment_trajectory_sha256":hashlib.sha256(
           np.asarray(physical_states,dtype="<i8").tobytes()).hexdigest()
    }


def aggregate(policy,world_list):
    started=time.perf_counter()
    rows=[run_policy(policy,w) for w in world_list]
    switched=[x for x in rows if x["switched"]]
    signs=[v for row in switched for v in row["factual_probe_sign_correct"]]
    return {
        "mean_episode_reward":float(np.mean([v["mean_episode_reward"] for v in rows])),
        "mean_switched_postreturn_reward":float(np.mean([v["postreturn_reward"] for v in switched])),
        "causal_sign_accuracy":float(np.mean(signs)) if signs else None,
        "mean_calibration_count":float(np.mean([v["calibration_count"] for v in rows])),
        "mean_sham_count":float(np.mean([v["sham_count"] for v in rows])),
        "total_selected_action_count":int(sum(v["action_count"] for v in rows)),
        "full_process_eval_seconds":time.perf_counter()-started,
        "per_world":rows
    }


def transition_contrast(world_list):
    """Controlled do(action0) vs do(action1): same NEXT action/outcome noise.

    The evaluator alone chooses a correct next exploit to isolate the physical
    actuator state effect. No nonoracle policy sees the private sign.
    """
    contrasts=[]
    for w in world_list:
        _,cal_h=transition(w,0,0,0)
        _,sham_h=transition(w,0,1,0)
        m=int(w["latent_law_evaluator_only"][1])
        same_action=2 if m>0 else 3
        corrected,_=transition(w,1,same_action,cal_h)
        sham,_=transition(w,1,same_action,sham_h)
        contrasts.append(float(corrected[0]-sham[0]))
    return {
        "mean_future_reward_do_calibrate_minus_do_sham":float(np.mean(contrasts)),
        "min_future_reward_contrast":float(min(contrasts)),
        "max_future_reward_contrast":float(max(contrasts)),
        "controlled_worlds":len(contrasts)
    }


def judgment(scores,contrast,freeze,fitdev_ok,balance_ok,phase_ok):
    bounds=freeze["metrics"]["required_all"]
    candidate=scores["CALIBRATE_CAUSAL"]
    blind=scores["BLIND_CALIBRATE_REACTIVE"]
    none=scores["NO_CALIBRATE_REACTIVE"]
    sham=scores["SHAM_PROXY"]
    direct=contrast["mean_future_reward_do_calibrate_minus_do_sham"]
    vs_blind=candidate["mean_episode_reward"]-blind["mean_episode_reward"]
    vs_none=candidate["mean_episode_reward"]-none["mean_episode_reward"]
    vs_sham=candidate["mean_episode_reward"]-sham["mean_episode_reward"]
    post=candidate["mean_switched_postreturn_reward"]-blind["mean_switched_postreturn_reward"]
    def pairs(other):
        p=[]
        for a,b in zip(candidate["per_world"],other["per_world"]):
            if a["world"]!=b["world"] or a["switched"]!=b["switched"]:
                raise ValueError("F8R70_PAIRED_SOURCE_OR_ALIGNMENT_DRIFT")
            p.append(b["mean_episode_reward"]-a["mean_episode_reward"])
        rng=np.random.default_rng(BOOT_SEED)
        sample=rng.integers(0,len(p),(2000,len(p)))
        return float(np.quantile(np.asarray(p)[sample].mean(axis=1),.975))
    up_blind=pairs(blind)
    up_none=pairs(none)
    checks={
       "exact_selected_action_budget":all(v["total_selected_action_count"]==STEPS*96 for v in scores.values()),
       "fresh_source_nonoverlap_balanced":bool(fitdev_ok and balance_ok),
       "calibration_guaranteed_all_three_regimes":bool(phase_ok),
       "causal_probe_sign_accuracy":candidate["causal_sign_accuracy"]>=bounds["causal_probe_sign_accuracy_min"],
       "physical_intervention_future_effect":direct>=bounds["physical_transition_future_reward_gain_min"],
       "causal_net_reward_vs_blind_same_transition":vs_blind>=bounds["causal_episode_reward_vs_blind_same_transition_min"],
       "causal_net_reward_vs_no_calibration":vs_none>=bounds["causal_episode_reward_vs_no_calibrate_reactive_min"],
       "causal_net_reward_vs_sham":vs_sham>=bounds["causal_episode_reward_vs_sham_min"],
       "switched_cold_return_vs_blind":post>=bounds["causal_switched_postreturn_gain_vs_blind_min"],
       "bootstrap_upper_blind_minus_causal_negative":up_blind<0,
       "bootstrap_upper_nocalib_minus_causal_negative":up_none<0
    }
    return {
        "metrics":{
          "physical_future_intervention_reward_delta":direct,
          "causal_minus_blind_episode_reward":vs_blind,
          "causal_minus_no_calibration_episode_reward":vs_none,
          "causal_minus_sham_episode_reward":vs_sham,
          "causal_minus_blind_switched_postreturn_reward":post,
          "causal_actual_selected_sign_accuracy":candidate["causal_sign_accuracy"],
          "paired_bootstrap_upper95_blind_minus_causal":up_blind,
          "paired_bootstrap_upper95_no_calibration_minus_causal":up_none
        },"checks":checks
    }


def run():
    started=time.perf_counter()
    f=locked()
    fit=worlds("FIT")
    dev=worlds("DEV")
    digest=lambda w:hashlib.sha256(w["potential_innovations_evaluator_only"].astype("<f8").tobytes()).hexdigest()
    source_disjoint=set(map(digest,fit)).isdisjoint(map(digest,dev))
    balanced=all(
       len(group)==COUNTS[group[0]["panel"]]
       and sum(int(x["switched_evaluator_only"]) for x in group)==len(group)//2
       and sum(int(x["initial_sign_evaluator_only"]==1) for x in group)==len(group)//2
       and all(sum(int(x["switched_evaluator_only"]==sw and
                       x["initial_sign_evaluator_only"]==sg)
                   for x in group)==len(group)//4
               for sw in (False,True) for sg in (-1,1))
       for group in (fit,dev))
    phase=all(
       12<=x["B_start_evaluator_only"]<=15
       and 28<=x["A_return_start_evaluator_only"]<=31
       and x["latent_law_evaluator_only"][0]==x["initial_sign_evaluator_only"]
       and x["latent_law_evaluator_only"][16]==
          (-x["initial_sign_evaluator_only"] if x["switched_evaluator_only"]
          else x["initial_sign_evaluator_only"])
       and x["latent_law_evaluator_only"][32]==x["initial_sign_evaluator_only"]
       for x in fit+dev)
    if not all((source_disjoint,balanced,phase)):
        raise ValueError("F8R70_SOURCE_ADMISSION_FAILED_BEFORE_SCORE")
    scores={p:aggregate(p,dev) for p in POLICIES}
    contrast=transition_contrast(dev)
    judge=judgment(scores,contrast,f,source_disjoint,balanced,phase)
    changed=negative_proxy(dev)
    replay_ok=True
    proxy_ok=True
    for name in POLICIES:
        original=scores[name]["per_world"]
        copy=aggregate(name,dev)["per_world"]
        proxy=aggregate(name,changed)["per_world"]
        for a,b,c in zip(original,copy,proxy):
            signature=lambda d:(d["world"],d["selected_actions"],d["mean_episode_reward"],
                d["postreturn_reward"],d["selected_observation_sha256"],
                d["physical_alignment_trajectory_sha256"])
            replay_ok &= signature(a)==signature(b)
            proxy_ok &= signature(a)==signature(c)
    judge["checks"]["independent_nuisance_proxy_invariance"]=bool(proxy_ok)
    judge["checks"]["exact_cold_replay"]=bool(replay_ok)
    passed=all(judge["checks"].values())
    return {
       "schema":SCHEMA,"freeze_blob":FREEZE_BLOB,"parent":PARENT,
       "classification":f["classification"],
       "source":{"FIT":len(fit),"DEV":len(dev),"switched_DEV":48,
                 "protected_external_OOD":0},
       "scores":{k:{m:v for m,v in s.items() if m!="per_world"}
                 for k,s in scores.items()},
       "controlled_physical_transition":contrast,
       "judgment":judge,
       "admission":"TASK_ADMISSION_PASS_SYNTHETIC_ONLY" if passed else "TASK_ADMISSION_NO_GO",
       "independent_scientific_pass":False,
       "trained_native_model_gain":False,
       "GPU_firstuse":False,"external_scientific_firstuse":False,
       "deployment":"NONE",
       "whole_process_wall_seconds":time.perf_counter()-started
    }


def retain_ci_receipt(result):
    if os.getenv("GITHUB_ACTIONS")!="true":
        return
    dest=ROOT/"local_runs/p85mr_f8r70_stateful_intervention_dev.json"
    dest.parent.mkdir(exist_ok=True)
    if dest.exists():
        raise FileExistsError("F8R70_REFUSE_TO_OVERWRITE_ORIGINAL_RESULT")
    dest.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")


def deny_protected_firstuse(*_args,**_kwargs):
    raise PermissionError("F8R70_PROTECTED_ORIGINAL_FIRSTUSE_DENIED")
