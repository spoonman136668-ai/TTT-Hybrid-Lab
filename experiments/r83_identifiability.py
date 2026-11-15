"""R83: fresh observation-identifiable physical intervention value, before training.

Single prospective CALIBRATION panel ONLY. All worlds have one ACTUAL public
PROBE immediately before the decision, emitting a noisy physical law cue.
An agent receives only its public (13D numeric) factual prefix. Evaluator
realizes and scores matched CALIBRATE versus EXPLOIT branch effects, but
NEVER provides a hidden law, oracle return or an unchosen observation to a
legal policy. Independent sealed QUALIFICATION/DEV/OOD are NOT GENERATED.

No model training, scientific first-use, GPU or integration is performed.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r83_identifiability_freeze.json"
FREEZE_BLOB="2f48f96697849beaa975be1d6984bcc56ecca6aa"
SCHEMA="ttt-hybrid.r83-observable-action-value-identifiability-v1"
PARENT="fc87351eb545d52d04467da29e05c15083fdaba2"
BRANCH="research/r83-observable-intervention-identifiability-20261115"
SEEDS=(999011,999023)
N=512
LENGTH=64
EARLY=16
COLD=48
CALIBRATE=0
EXPLOIT=2


def git_blob(raw):
    clean=raw.replace(b"\r\n",b"\n")
    if b"\r" in clean:
        raise RuntimeError("R83_UNEXPECTED_CARRIAGE_RETURN")
    return hashlib.sha1(b"blob "+str(len(clean)).encode()+b"\0"+clean).hexdigest()


def frozen():
    if git_blob(FREEZE.read_bytes())!=FREEZE_BLOB:
        raise RuntimeError("R83_ORIGINAL_FROZEN_PROTOCOL_GIT_BLOB_DRIFT")
    f=json.loads(FREEZE.read_text(encoding="utf-8"))
    s=f["source"]
    if (f["schema"]!=SCHEMA
        or f["stage"]!="PROSPECTIVE_FREEZE_AFTER_R82_NO_GO_BEFORE_NEW_SOURCE_OR_EXECUTION"
        or f["parent"]["R82_SHA"]!=PARENT
        or f["parent"]["R82_both_seed_result"]!="DEV_ACTIVE_POLICY_NO_GO"
        or s["seeds"]!=list(SEEDS)
        or s["worlds_per_seed"]!=N
        or s["used_now"]!="CALIBRATION ONLY; QUALIFICATION/DEV_SEALED/OOD_SEALED not generated"
        or s["sequence"]!=LENGTH
        or s["early_pivot"]!=EARLY or s["cold_pivot"]!=COLD
        or s["cue_noise_standard_deviation"]!=0.4
        or s["new_salts"]["CALIBRATION"]!=25250111
        or f["gates"]["paired_bootstrap_resamples"]!=2000
        or any(f["authority"].values())):
        raise PermissionError("R83_ORIGINAL_PROTOCOL_SEMANTICS_DRIFT")
    return f


def _law(world,t):
    return -world["initial"] if world["start"]<=t<world["restore"] else world["initial"]


def _public_rollout(world,chosen):
    """Perform a real factual physical rollout; input is PUBLIC only."""
    if chosen not in (CALIBRATE,EXPLOIT):
        raise ValueError("R83_UNREGISTERED_ACTION")
    pivot=world["pivot"]
    x=np.zeros((pivot+1,13),dtype=np.float32)
    state=0
    prior_action=None
    prior_observed=np.zeros(4,dtype=np.float32)
    ret=0.0
    cue=None
    for t in range(pivot+9):
        action=int(world["actions"][t]) if t!=pivot else chosen
        if t<=pivot:
            row=x[t]
            row[action]=1.0
            if prior_action is not None:
                row[4+prior_action]=1.0
            row[8:12]=prior_observed
            row[12]=t/(LENGTH-1)
        law=_law(world,t)
        eps=world["innovations"][t]
        if action==CALIBRATE:
            observed=np.array([-.5,eps[1],eps[2],eps[3]],dtype=np.float32)
            state=law
        elif action==1:  # PROBE has physical cost; cue is PUBLIC only after this step
            observed=np.array([-.1+.01*eps[0],eps[1],
                law+.4*eps[2],eps[3]],dtype=np.float32)
            if t==pivot-1:
                cue=float(observed[2])
        else:
            if action not in (EXPLOIT,3):
                raise RuntimeError("R83_INVALID_PHYSICAL_ACTION")
            direction=1 if action==EXPLOIT else -1
            if state==law:
                amp,penalty=1.0,0.0
            elif state==0:
                amp,penalty=.2,-.1
            else:
                amp,penalty=.2,-.25
            observed=np.array([
                direction*law*amp+penalty+.1*eps[0],
                eps[1],eps[2],eps[3]],dtype=np.float32)
        if t>=pivot:
            ret+=float(observed[0])
        prior_observed=observed
        prior_action=action
    return x,ret,cue


def calibration_world(seed,index):
    """World is generated from NEW salt + seed + index only."""
    frozen()
    if seed not in SEEDS or not isinstance(index,int) or not 0<=index<N:
        raise PermissionError("R83_SEED_OR_SOURCE_COUNT_NOT_PRECOMMITTED")
    salt=frozen()["source"]["new_salts"]["CALIBRATION"]
    rng=np.random.default_rng(np.random.SeedSequence([salt,seed,index]))
    pivot=EARLY if index%2==0 else COLD
    initial=1 if rng.integers(0,2) else -1
    start=int(rng.integers(26,32))
    restore=int(rng.integers(38,44))
    actions=rng.choice([1,2,3],size=LENGTH,p=[.20,.40,.40]).astype(np.int64)
    actions[pivot-1]=1  # guaranteed lawful observation before decision
    actions[pivot+1:pivot+9]=EXPLOIT
    innovations=rng.standard_normal((LENGTH,4))
    world={"initial":initial,"start":start,"restore":restore,
           "pivot":pivot,"actions":actions,"innovations":innovations}
    p0,r0,cue0=_public_rollout(world,CALIBRATE)
    p2,r2,cue2=_public_rollout(world,EXPLOIT)
    if not np.array_equal(p0[:pivot],p2[:pivot]):
        raise RuntimeError("R83_PAIR_PUBLIC_PAST_NOT_IDENTICAL")
    if cue0!=cue2 or not np.isfinite(cue0):
        raise RuntimeError("R83_PAIR_PROBE_NOT_CONSISTENT")
    if not np.array_equal(p0[pivot,4:13],p2[pivot,4:13]):
        raise RuntimeError("R83_PAIR_OBSERVABLES_NOT_IDENTICAL")
    if p0[pivot,0]!=1.0 or p2[pivot,2]!=1.0:
        raise RuntimeError("R83_CURRENT_ACTION_NOT_PUBLIC")
    return {"public_prefix_calibrate":p0,
            "public_prefix_exploit":p2,
            "cue":float(p0[pivot,10]),
            "true_law_private":_law(world,pivot),
            "r0_evaluator_only":r0,
            "r2_evaluator_only":r2,
            "pivot":pivot}


def _shuffled_indices(pivots,seed):
    rng=np.random.default_rng(np.random.SeedSequence([25250555,seed]))
    idx=np.arange(len(pivots))
    new=idx.copy()
    for pivot in (EARLY,COLD):
        where=np.where(pivots==pivot)[0]
        new[where]=rng.permutation(where)
    return new


def _bootstrap_lower(per_world_gain,seed):
    gains=np.asarray(per_world_gain,np.float64)
    if gains.shape!=(N,):
        raise ValueError("R83_PAIRED_BOOTSTRAP_WRONG_WORLD_COUNT")
    rng=np.random.default_rng(np.random.SeedSequence([25250555,seed]))
    selected=rng.integers(0,N,size=(2000,N))
    return float(np.quantile(gains[selected].mean(axis=1),.025))


def diagnostic(seed):
    """Fully predeclared evaluator-only information availability diagnosis."""
    f=frozen()
    if seed not in SEEDS:
        raise ValueError("R83_INVALID_SEED")
    worlds=[calibration_world(seed,i) for i in range(N)]
    cue=np.array([w["cue"] for w in worlds],np.float64)
    sign=np.array([w["true_law_private"] for w in worlds],np.int64)
    pivot=np.array([w["pivot"] for w in worlds],np.int64)
    cal=np.array([w["r0_evaluator_only"] for w in worlds],np.float64)
    exploit=np.array([w["r2_evaluator_only"] for w in worlds],np.float64)
    # Legal public rule: look ONLY at observed previous probe cue in the
    # current legal public prefix. Never inspect sign or branch reward.
    chosen=np.where(cue>0,CALIBRATE,EXPLOIT)
    public_reward=np.where(chosen==CALIBRATE,cal,exploit)
    # Negative: cue distribution and public time preserved but association
    # with the actual current law destroyed by a fixed within-pivot shuffle.
    wrongcue=cue[_shuffled_indices(pivot,seed)]
    negchosen=np.where(wrongcue>0,CALIBRATE,EXPLOIT)
    shuffled_reward=np.where(negchosen==CALIBRATE,cal,exploit)
    # These two baselines and oracle are EVALUATOR-ONLY, never learner inputs.
    best_constant_is_calibrate=bool(cal.mean()>exploit.mean())
    constant_reward=cal if best_constant_is_calibrate else exploit
    oracle=np.maximum(cal,exploit)
    idx={"early":pivot==EARLY,"cold":pivot==COLD}
    if any(v.sum()!=N//2 for v in idx.values()):
        raise RuntimeError("R83_COLD_EARLY_BALANCE_MISMATCH")
    group={}
    for name,mask in idx.items():
        strongest_fixed=max(float(cal[mask].mean()),
                            float(exploit[mask].mean()))
        group[name]={
            "public_mean_return":float(public_reward[mask].mean()),
            "best_fixed_evaluator_return":strongest_fixed,
            "public_minus_best_fixed":float(public_reward[mask].mean()-strongest_fixed),
            "public_sign_accuracy":float(np.mean(np.where(cue[mask]>0,1,-1)==sign[mask]))}
    accuracy=float(np.mean(np.where(cue>0,1,-1)==sign))
    baseline=max(float(cal.mean()),float(exploit.mean()))
    lower=_bootstrap_lower(public_reward-constant_reward,seed)
    checks={
        "legal_probe_accuracy_at_least_0_80":accuracy>=.8,
        "public_vs_best_fixed_reward_at_least_1_50":public_reward.mean()-baseline>=1.5,
        "public_vs_shuffled_reward_at_least_1_50":public_reward.mean()-shuffled_reward.mean()>=1.5,
        "oracle_regret_at_most_0_75":(oracle-public_reward).mean()<=.75,
        "paired_bootstrap_lower95_above_0_50":lower>=.5,
        "early_improvement_at_least_0_80":group["early"]["public_minus_best_fixed"]>=.8,
        "cold_improvement_at_least_0_80":group["cold"]["public_minus_best_fixed"]>=.8}
    return {
        "seed":seed,"worlds":N,
        "CALIBRATION_source_only":True,
        "QUALIFICATION_DEV_OOD_generated":False,
        "probe_public_correct_sign_rate":accuracy,
        "public_policy_mean_realized_return":float(public_reward.mean()),
        "best_fixed_evaluator_mean_return":baseline,
        "shuffled_public_cue_mean_return":float(shuffled_reward.mean()),
        "oracle_evaluator_mean_return":float(oracle.mean()),
        "oracle_minus_public_regret":float((oracle-public_reward).mean()),
        "bootstrap95_lower_public_minus_fixed":lower,
        "world_groups":group,
        "frozen_checks":checks,
        "pass":all(checks.values())}


def admit():
    frozen()
    if (os.getenv("GITHUB_ACTIONS")!="true"
        or os.getenv("GITHUB_EVENT_NAME")!="pull_request"
        or os.getenv("GITHUB_HEAD_REF")!=BRANCH
        or os.getenv("GITHUB_REPOSITORY")!="spoonman136668-ai/TTT-Hybrid-Lab"
        or os.getenv("RUNNER_OS")!="Linux"
        or os.getenv("RUNNER_ENVIRONMENT")!="github-hosted"
        or os.getenv("GITHUB_RUN_ATTEMPT")!="1"
        or "/merge" not in os.getenv("GITHUB_REF","")):
        raise PermissionError("R83_NOT_QUALIFIED_G4_FIRST_ATTEMPT_HOSTED_CPU")
    sha=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,
        capture_output=True,check=False,timeout=10)
    diff=subprocess.run(["git","diff","--quiet","HEAD","--"],cwd=ROOT,
        capture_output=True,check=False,timeout=10)
    expected=os.getenv("EXPECTED_CANDIDATE_SHA","")
    if (len(expected)!=40 or sha.returncode!=0 or sha.stdout.strip()!=expected
        or diff.returncode!=0):
        raise PermissionError("R83_EXACT_SOURCE_CUSTODY_MISMATCH")
    return expected


def original_ci():
    sha=admit()
    path=ROOT/"local_runs/r83_information_identifiability_calibration.json"
    if path.exists():
        raise FileExistsError("R83_DUPLICATE_OR_UNCERTAIN_RESULT")
    started=time.perf_counter()
    results=[diagnostic(seed) for seed in SEEDS]
    passed=all(row["pass"] for row in results)
    receipt={
        "schema":SCHEMA,"freeze_blob":FREEZE_BLOB,
        "candidate_SHA":sha,"parent_R82_SHA":PARENT,
        "worlds_per_seed":N,"all_seeds":list(SEEDS),
        "results":results,
        "status":("ENGINEERING_INFORMATION_IDENTIFIABILITY_CALIBRATION_PASS_ONLY"
            if passed else "ENGINEERING_BENCHMARK_INFORMATION_NO_GO"),
        "no_model_training":True,
        "QUALIFICATION_DEV_OOD_first_use":False,
        "independent_scientific_pass":False,
        "all_process_seconds":time.perf_counter()-started,
        "platform":platform.platform()}
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as h:
        json.dump(receipt,h,indent=2,sort_keys=True,allow_nan=False)
        h.write("\n")
    if not passed:
        raise AssertionError("R83_PREDECLARED_OBSERVABLE_IDENTIFIABILITY_NO_GO")
    return receipt


if __name__=="__main__":
    out=original_ci()
    print(json.dumps({"status":out["status"],"seeds":out["all_seeds"],
        "no_science":not out["independent_scientific_pass"]},sort_keys=True))
