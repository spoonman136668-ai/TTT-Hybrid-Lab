"""F8-R69: pre-frozen consequential causal-intervention task admission.

This is an author-synthetic, source-neutral TASK admission. It does not train
or qualify a new native Hybrid. All actual actions spend a step, and only
their actually selected outcomes reach agent policies. The evaluator may
retain hidden potential outcomes for *paired scoring only*; no policy other
than the explicitly labelled oracle bound reads latent law/sign.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r69_causal_probe_task_freeze.json"
FREEZE_BLOB="680c7659b51386580fdacb18d049d625da14ec27"
SCHEMA="ttt-hybrid.f8r69-causal-intervention-task-admission.v1"
PARENT="efe7081d1a32e5ce96236bdf0a5c9024658d9afb"
SALT={"FIT":24110000,"DEV":24130000,"PROXY_NEGATIVE":24140000}
COUNTS={"FIT":64,"DEV":96}
STEPS=36
ACTIONS=4
PROBES=(0,14,28)
POLICIES=("ORACLE_UPPER","CAUSAL_PROBE","SPURIOUS_PROBE","REWARD_REACTIVE","RANDOM")
BOOT_SEED=24149999


def git_blob(data):
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()


def locked():
    content=FREEZE.read_bytes()
    if git_blob(content)!=FREEZE_BLOB:
        raise ValueError("F8R69_FROZEN_BLOB_DRIFT")
    f=json.loads(content)
    s,p,m=f["source"],f["policies"],f["metrics"]
    if (f["schema"]!=SCHEMA
        or f["stage"]!="PROSPECTIVE_TASK_ADMISSION_BEFORE_SOURCE_AND_DEV"
        or f["parent"]["R66_SHA"]!=PARENT
        or not f["parent"]["no_spent_R64_to_R68_cohorts"]
        or s["salts"]!=SALT or s["world_counts"]!=COUNTS
        or s["total_steps"]!=STEPS or s["action_count"]!=ACTIONS
        or s["selected_terminal_dimensions"]!=4
        or s["protected_OOD"] or s["external_first_use"]
        or p["constant_probe_times"]!=list(PROBES)
        or p["action_budget_per_policy"]!=STEPS
        or p["causal_and_spurious_probes_each"]!=3
        or p["no_parametric_model_training"] is not True
        or m["world_paired_bootstrap"]["seed"]!=BOOT_SEED
        or m["world_paired_bootstrap"]["resamples"]!=2000
        or m["no_post_result_tuning"] is not True
        or any(f["authority"].values())):
        raise ValueError("F8R69_FROZEN_CONTRACT_DRIFT")
    return f


def worlds(panel):
    """Only FIT/DEV new source; private outcomes are never passed to policies."""
    if panel not in COUNTS:
        raise PermissionError("F8R69_PROTECTED_OR_SPENT_SOURCE_DENIED")
    rng=np.random.default_rng(SALT[panel])
    count=COUNTS[panel]
    kinds=[(switched,sign)
           for switched in (False,True)
           for sign in (-1,1)
           for _ in range(count//4)]
    order=rng.permutation(count)
    result=[]
    for index,chosen in enumerate(order):
        switched,sign=kinds[int(chosen)]
        b0=int(rng.integers(10,14))
        a_return=int(rng.integers(23,27))
        latent=np.full(STEPS,sign,dtype=np.int64)
        if switched:
            latent[b0:a_return]=-sign
        potential=np.empty((STEPS,ACTIONS,4),dtype=np.float64)
        for t in range(STEPS):
            m=int(latent[t])
            # All four potential selected action consequences are generated,
            # but only the chosen row can ever leave evaluator.run_policy.
            for action in range(ACTIONS):
                obs=np.empty(4,dtype=np.float64)
                if action==0:
                    obs[0]=-.18+rng.normal(0,.02)
                    obs[1]=m+rng.normal(0,.10)
                    obs[2]=rng.normal(0,1)
                elif action==1:
                    obs[0]=-.18+rng.normal(0,.02)
                    obs[1]=rng.normal(0,1)
                    obs[2]=(m if panel=="FIT" else -m)+rng.normal(0,.10)
                elif action==2:
                    obs[0]=.70*m+rng.normal(0,.80)
                    obs[1]=rng.normal(0,1)
                    obs[2]=rng.normal(0,1)
                else:
                    obs[0]=-.70*m+rng.normal(0,.80)
                    obs[1]=rng.normal(0,1)
                    obs[2]=rng.normal(0,1)
                obs[3]=rng.normal(0,1)
                potential[t,action]=obs
        # Unchosen irrelevant source, only used by negative invariance test.
        extra_proxy=rng.normal(0,1,(STEPS,4))
        result.append({
            "panel":panel, "index":index, "switched_evaluator_only":bool(switched),
            "initial_sign_evaluator_only":int(sign),
            "B_start_evaluator_only":b0,"A_return_start_evaluator_only":a_return,
            "latent_law_evaluator_only":latent,"potential_evaluator_only":potential,
            "unobserved_proxy":extra_proxy
        })
    return result


def negative_proxy(dev):
    if len(dev)!=COUNTS["DEV"] or any(x["panel"]!="DEV" for x in dev):
        raise PermissionError("F8R69_PROXY_REQUIRES_ORIGINAL_DEV")
    rng=np.random.default_rng(SALT["PROXY_NEGATIVE"])
    return [{**w,"unobserved_proxy":rng.normal(0,5,(STEPS,4))}
            for w in dev]


def choose_action(policy,t,history,rng,*,private_latent=None):
    """All valid non-oracle policies receive only time + selected history."""
    if policy not in POLICIES:
        raise ValueError("F8R69_UNREGISTERED_POLICY")
    if len(history)!=t:
        raise ValueError("F8R69_OBSERVATION_ORDER_VIOLATION")
    if policy=="ORACLE_UPPER":
        if private_latent not in (-1,1):
            raise PermissionError("F8R69_ORACLE_PRIVATE_SIGN_REQUIRED")
        return 2 if private_latent>0 else 3
    if private_latent is not None:
        raise PermissionError("F8R69_POLICY_CANNOT_RECEIVE_PRIVATE_LAW")
    if policy=="RANDOM":
        return int(rng.integers(0,ACTIONS))
    if policy in ("CAUSAL_PROBE","SPURIOUS_PROBE"):
        probe_action=0 if policy=="CAUSAL_PROBE" else 1
        if t in PROBES:
            return probe_action
        observations=[obs for action,obs in history if action==probe_action]
        if not observations:
            raise ValueError("F8R69_MISSING_FACTUAL_PROBE")
        measured=observations[-1][1 if probe_action==0 else 2]
        return 2 if measured>=0 else 3
    # Nonprobing reward-only reactive control. Exactly one observed selected
    # consequence is consumed per update, no other channel or world state.
    belief=0.
    for action,selected in history:
        if action not in (2,3):
            raise ValueError("F8R69_REACTIVE_CONTROL_MUST_NOT_PROBE")
        direction=1. if action==2 else -1.
        belief=.65*belief+.35*selected[0]*direction
    return 2 if belief>=0 else 3


def run_policy(policy,world):
    if policy not in POLICIES:
        raise ValueError("F8R69_POLICY_NOT_REGISTERED")
    if world["panel"] not in COUNTS:
        raise PermissionError("F8R69_PRIVATE_SOURCE_DENIED")
    observed=world["potential_evaluator_only"]
    latent=world["latent_law_evaluator_only"]
    if observed.shape!=(STEPS,ACTIONS,4) or latent.shape!=(STEPS,):
        raise ValueError("F8R69_BAD_WORLD")
    # Independent of private law; only action randomness for random policy.
    rng=np.random.default_rng(SALT[world["panel"]]+100000+world["index"])
    history=[]
    chosen=[]
    rewards=[]
    probe_truth=[]
    for t in range(STEPS):
        action=choose_action(
            policy,t,history,rng,
            private_latent=int(latent[t]) if policy=="ORACLE_UPPER" else None)
        if action not in range(ACTIONS):
            raise ValueError("F8R69_BAD_SELECTED_ACTION")
        result=observed[t,action].copy()
        history.append((action,result))
        chosen.append(action)
        rewards.append(float(result[0]))
        if t in PROBES and action in (0,1):
            coordinate=1 if action==0 else 2
            probe_truth.append(int(np.sign(result[coordinate]))==int(latent[t]))
    selected_digest=hashlib.sha256(
        b"".join(np.asarray(obs,dtype="<f8").tobytes()
                 for _a,obs in history)).hexdigest()
    reward=np.asarray(rewards,dtype=np.float64)
    return {
        "world":int(world["index"]),
        "switched":bool(world["switched_evaluator_only"]),
        "initial_sign":int(world["initial_sign_evaluator_only"]),
        "action_count":len(chosen),
        "selected_actions":tuple(chosen),
        "probe_count":int(sum(x in (0,1) for x in chosen)),
        "probe_correct":tuple(probe_truth),
        "sum_goal_reward":float(np.sum(reward)),
        "cold_return_reward":float(np.sum(reward[28:])),
        "selected_digest":selected_digest
    }


def aggregate(policy,source):
    started=time.perf_counter()
    rows=[run_policy(policy,w) for w in source]
    sum_reward=float(np.mean([v["sum_goal_reward"] for v in rows]))
    cold_reward=float(np.mean([v["cold_return_reward"] for v in rows
                                if v["switched"]]))
    probe_rows=[v for v in rows if v["switched"]]
    probe_accuracy=(float(np.mean([
        ok for row in probe_rows for ok in row["probe_correct"]]))
        if probe_rows and any(row["probe_correct"] for row in probe_rows) else None)
    return {
        "mean_episode_reward":sum_reward,
        "mean_switched_cold_return_reward":cold_reward,
        "switched_probe_sign_accuracy":probe_accuracy,
        "mean_probe_actions":float(np.mean([v["probe_count"] for v in rows])),
        "selected_action_count_total":sum(v["action_count"] for v in rows),
        "wall_seconds":time.perf_counter()-started,
        "per_world":rows
    }


def evaluate(scores,freeze):
    needed=freeze["metrics"]["required_all"]
    v=scores["CAUSAL_PROBE"]
    distract=scores["SPURIOUS_PROBE"]
    reactive=scores["REWARD_REACTIVE"]
    oracle=scores["ORACLE_UPPER"]
    random=scores["RANDOM"]
    gain_random=oracle["mean_episode_reward"]-random["mean_episode_reward"]
    gain_spurious=v["mean_episode_reward"]-distract["mean_episode_reward"]
    gain_reactive=v["mean_episode_reward"]-reactive["mean_episode_reward"]
    cold_gain=(v["mean_switched_cold_return_reward"]-
               distract["mean_switched_cold_return_reward"])
    diffs=[]
    for a,b in zip(v["per_world"],distract["per_world"]):
        if a["world"]!=b["world"] or a["switched"]!=b["switched"]:
            raise ValueError("F8R69_UNPAIRED_OR_UNALIGNED_CAUSAL_WORLDS")
        diffs.append(a["sum_goal_reward"]-b["sum_goal_reward"])
    bootstrap=freeze["metrics"]["world_paired_bootstrap"]
    rng=np.random.default_rng(bootstrap["seed"])
    ids=rng.integers(0,len(diffs),(bootstrap["resamples"],len(diffs)))
    # Predeclared upper 95% of SPURIOUS minus CAUSAL, must be NEGATIVE.
    upper=float(np.quantile(-np.asarray(diffs)[ids].mean(axis=1),.975))
    checks={
        "full_action_budget_and_selected_only":all(
           score["selected_action_count_total"]==len(v["per_world"])*STEPS
           for score in scores.values()),
        "original_fit_dev_nonoverlap":True,
        "balanced_switches_signs":True,
        "actual_hidden_phase_guaranteed_at_probe_times":True,
        "oracle_reward_gap_random":gain_random>=needed["oracle_reward_advantage_over_random_min"],
        "causal_probe_reward_gap_spurious":gain_spurious>=needed["causal_probe_reward_advantage_over_spurious_min"],
        "causal_probe_reward_gap_reactive":gain_reactive>=needed["causal_probe_reward_advantage_over_reward_reactive_min"],
        "causal_cold_return_gap_spurious":cold_gain>=needed["causal_probe_cold_return_advantage_over_spurious_min"],
        "causal_probe_sign_accuracy":v["switched_probe_sign_accuracy"]>=needed["causal_probe_switch_sign_accuracy_min"],
        "paired_bootstrap_negative_upper":upper<0
    }
    return {"metrics":{
        "oracle_minus_random_episode_reward":gain_random,
        "causal_minus_spurious_episode_reward":gain_spurious,
        "causal_minus_reward_reactive_episode_reward":gain_reactive,
        "causal_minus_spurious_switched_cold_reward":cold_gain,
        "causal_probe_switched_sign_accuracy":v["switched_probe_sign_accuracy"],
        "spurious_probe_switched_sign_accuracy":distract["switched_probe_sign_accuracy"],
        "paired_bootstrap_upper95_spurious_minus_causal":upper
    },"checks":checks}


def run():
    started=time.perf_counter()
    f=locked()
    fit=worlds("FIT")
    dev=worlds("DEV")
    all_source=fit+dev
    # These checks are source-admission prerequisites, not optional
    # post-result metrics or privately supplied policy inputs.
    digest=lambda w:hashlib.sha256(w["potential_evaluator_only"].astype("<f8").tobytes()).hexdigest()
    unique=set(map(digest,fit)).isdisjoint(map(digest,dev))
    balances=all(
        (sum(bool(w["switched_evaluator_only"]) for w in group)==len(group)//2
         and sum(w["initial_sign_evaluator_only"]==1 for w in group)==len(group)//2
         and all(sum(w["switched_evaluator_only"] and
                     w["initial_sign_evaluator_only"]==sign for w in group)==len(group)//4
                 for sign in (-1,1)))
        for group in (fit,dev))
    phases=all(
        (10<=w["B_start_evaluator_only"]<=13
         and 23<=w["A_return_start_evaluator_only"]<=26
         and w["latent_law_evaluator_only"][0]==w["initial_sign_evaluator_only"]
         and w["latent_law_evaluator_only"][14] == (
             -w["initial_sign_evaluator_only"] if w["switched_evaluator_only"]
             else w["initial_sign_evaluator_only"])
         and w["latent_law_evaluator_only"][28]==w["initial_sign_evaluator_only"])
        for w in all_source)
    if not unique or not balances or not phases:
        raise ValueError("F8R69_SOURCE_ADMISSION_FAILED")
    scores={p:aggregate(p,dev) for p in POLICIES}
    judgment=evaluate(scores,f)
    judgment["checks"]["original_fit_dev_nonoverlap"]=bool(unique)
    judgment["checks"]["balanced_switches_signs"]=bool(balances)
    judgment["checks"]["actual_hidden_phase_guaranteed_at_probe_times"]=bool(phases)
    negative=negative_proxy(dev)
    replay_ok=True
    proxy_ok=True
    for p in POLICIES:
        original=scores[p]["per_world"]
        same=aggregate(p,dev)["per_world"]
        perturbed=aggregate(p,negative)["per_world"]
        for a,b,c in zip(original,same,perturbed):
            key=lambda z:(z["world"],z["selected_actions"],
                          z["sum_goal_reward"],z["cold_return_reward"],
                          z["selected_digest"])
            replay_ok &= key(a)==key(b)
            proxy_ok &= key(a)==key(c)
    judgment["checks"]["proxy_negative_invariance"]=bool(proxy_ok)
    judgment["checks"]["cold_replay_exact"]=bool(replay_ok)
    admission=all(judgment["checks"].values())
    return {
        "schema":SCHEMA,"freeze_blob":FREEZE_BLOB,"parent":PARENT,
        "classification":f["classification"],
        "source":{"FIT":len(fit),"DEV":len(dev),"switched_DEV":48,
                  "protected_OOD_generated":0},
        "scores":{name:{k:v for k,v in score.items() if k!="per_world"}
                  for name,score in scores.items()},
        "judgment":judgment,
        "admission":"TASK_ADMISSION_PASS_SYNTHETIC_ONLY" if admission
                    else "TASK_ADMISSION_NO_GO",
        "independent_scientific_pass":False,
        "trained_model_improvement":False,
        "GPU_first_use":False,"external_first_use":False,
        "deployment":"NONE",
        "whole_process_wall_seconds":time.perf_counter()-started
    }


def retain_ci_receipt(out):
    if os.getenv("GITHUB_ACTIONS")!="true":
        return
    dest=ROOT/"local_runs/p85mr_f8r69_causal_probe_task_admission.json"
    dest.parent.mkdir(exist_ok=True)
    if dest.exists():
        raise FileExistsError("F8R69_NEVER_OVERWRITE_ORIGINAL_DEV")
    dest.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+"\n")


def deny_external_scientific_first_use(*_args,**_kwargs):
    raise PermissionError("F8R69_ORIGINAL_EXTERNAL_FIRST_USE_DENIED")
