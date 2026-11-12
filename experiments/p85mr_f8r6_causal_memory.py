"""F8-R6 learned, action-conditioned two-regime causal-memory strong CONTROL.

Fits two linear dynamics rules from *only* executed factual toy action/effect
trajectories. No hidden sign is supplied to fitting or inference. At test,
one observable probe action updates an explicit two-rule belief before a
shared two-action goal chooser. No genuine external source or F7 data used.
This is a toy conventional control, never a Hybrid or scientific PASS.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from experiments import p85mr_f8r3_counterpolicy_identifiability as r3

ROOT = Path(__file__).resolve().parents[1]
FREEZE_BLOB = "605c7ca6c000cf9d44aba41bfb270dd20ac454b0"
PARENT_SHA = "90570bd0f7fa13af3898001ad612ed13f255b64e"
FIT_SEED_BASE = 19260000
RNG_SCORE = 19277777
RNG_SHUFFLE = 19279999
RIDGE = 1e-8
FIT_N = 24
HORIZON = 12
SCORE_N = 48
PROBE_S = .2
SEQUEL_ACTIONS = (0,1)


def _blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def locked():
    raw = (ROOT/"experiments/p85mr_f8r6_causal_memory_freeze.json").read_bytes()
    if _blob(raw) != FREEZE_BLOB:
        raise ValueError("F8R6_FROZEN_SCIENCE_CONTRACT_DRIFT")
    f = json.loads(raw)
    w,m,k,a = f["world"],f["models"],f["metrics"],f["authority"]
    if (
        f["schema"]!="ttt-hybrid.f8r6-unsupervised-two-regime-active-probe.v1"
        or f["parent"]["head_sha"]!=PARENT_SHA
        or w["fit_seed_base"]!=FIT_SEED_BASE
        or w["fit_worlds"]!=FIT_N
        or w["fit_actions_per_world"]!=HORIZON
        or w["score_pairs_per_sign"]!=SCORE_N
        or w["score_goals_states_PCG64_seed"]!=RNG_SCORE
        or w["score_probe_state"]!=PROBE_S
        or w["score_actions_per_evaluation"]!=2
        or m["ridge_lambda"]!=RIDGE
        or f["negative_controls"]["wrong_credit_action_labels_shuffled_fit_only_seed"]!=RNG_SHUFFLE
        or k["independent_scientific_pass"] is not False
        or a["external_source_effects"]!=0
        or a["new_F8_CausalGymnasium_firstuse"] is not False
        or a["no_new_workflow_or_scheduler"] is not True
        or a["scientific_pass"] is not False
    ):
        raise ValueError("F8R6_SCOPE_OR_AUTHORITY_MISMATCH")
    r3.locked()
    return f


def policy(s:float) -> int:
    if not isinstance(s,(int,float,np.floating)) or not math.isfinite(float(s)):
        raise ValueError("F8R6_INVALID_STATE")
    return int(float(s)>=0.)


def feature(s:float,a:int) -> np.ndarray:
    if type(a) is not int or a not in SEQUEL_ACTIONS:
        raise ValueError("F8R6_INVALID_EXECUTED_ACTION")
    return np.array([float(s),float(a-policy(s))],dtype=np.float64)


def _effect(s:float,a:int,sign:int) -> float:
    # Controlled mathematical evaluator, NEVER public Causal-Gymnasium.
    return r3.synthetic_effect(float(s),a,sign)


def training_episodes():
    locked()
    logs=[]
    for i in range(FIT_N):
        seed=FIT_SEED_BASE+i
        hidden=1 if i%2==0 else -1
        s=.45*math.sin(seed*.031)+.21*math.cos(seed*.017)
        trajectory=[]
        for step in range(HORIZON):
            # In each episode at least the first action is informative.
            # No FIT hidden sign is returned or passed to the learner.
            off=step==0 or ((seed+7*step)%3==0)
            act=1-policy(s) if off else policy(s)
            observed=_effect(s,act,hidden)
            trajectory.append((float(s),int(act),float(observed)))
            s=observed
        logs.append(tuple(trajectory))
    return tuple(logs)


def _fit(X:np.ndarray,Y:np.ndarray):
    if (X.ndim!=2 or X.shape[1]!=2 or Y.shape!=(X.shape[0],)
        or X.shape[0]<2 or not np.isfinite(X).all()
        or not np.isfinite(Y).all()):
        raise ValueError("F8R6_INVALID_FIT_INFORMATION")
    gram=X.T@X+RIDGE*np.eye(2)
    weights=np.linalg.solve(gram,X.T@Y)
    if not np.isfinite(weights).all():
        raise ValueError("F8R6_UNQUALIFIED_NUMERICAL_REGRESSION")
    return weights


def fit_controls(logs):
    if (type(logs) not in (tuple,list) or len(logs)!=FIT_N
        or any(type(w) not in (tuple,list) or len(w)!=HORIZON for w in logs)):
        raise ValueError("F8R6_UNEXPECTED_FIT_COHORT")
    coeffs=[]
    all_x=[]
    all_y=[]
    for episode in logs:
        X=np.stack([feature(s,a) for s,a,y in episode])
        Y=np.array([float(y) for s,a,y in episode],dtype=np.float64)
        coeffs.append(_fit(X,Y))
        all_x.append(X);all_y.append(Y)
    ranked=sorted(coeffs,key=lambda z:z[1])
    experts=np.stack([
        np.mean(ranked[:FIT_N//2],axis=0),
        np.mean(ranked[FIT_N//2:],axis=0)
    ])
    pooled=_fit(np.concatenate(all_x),np.concatenate(all_y))
    if (experts.shape!=(2,2) or pooled.shape!=(2,)
        or not experts[0,1]<experts[1,1]):
        raise ValueError("F8R6_NO_INDEPENDENT_REGIME_IDENTIFICATION")
    return {"experts":experts,"pooled":pooled,
            "individual_coeffs":np.stack(coeffs)}


def _hash(a):
    return hashlib.sha256(np.asarray(a,dtype="<f8").tobytes()).hexdigest()


def infer(experts,probe_state,selected_action,observed_effect):
    """One visible factual outcome; no counterpolicy target is available."""
    if (np.shape(experts)!=(2,2) or
        not math.isfinite(float(observed_effect))):
        raise ValueError("F8R6_INVALID_COLD_EXPERT_STATE")
    x=feature(probe_state,selected_action)
    errs=np.abs(experts@x-float(observed_effect))
    if not np.isfinite(errs).all():
        raise ValueError("F8R6_INVALID_BELIEF")
    if math.isclose(float(errs[0]),float(errs[1]),rel_tol=0,abs_tol=1e-5):
        return (0.5,0.5)
    return (1.,0.) if errs[0]<errs[1] else (0.,1.)


def plan(experts,belief,s,goal):
    if (len(belief)!=len(experts) or len(experts) not in (1,2)
        or not math.isfinite(float(goal))
        or any(not math.isfinite(float(v)) or v<0 for v in belief)
        or not math.isclose(sum(belief),1.,abs_tol=1e-10)):
        raise ValueError("F8R6_INVALID_COLD_PLANNER_BELIEF")
    return min(SEQUEL_ACTIONS,key=lambda a:(
        (float(np.dot(belief,experts@feature(s,a)))-float(goal))**2,a))


def fixed_pairs():
    rng=np.random.Generator(np.random.PCG64(RNG_SCORE))
    return tuple((float(s),float(g)) for s,g in zip(
        rng.uniform(-.90,.90,size=SCORE_N),
        rng.uniform(-.60,.60,size=SCORE_N)))


def _score_one(mode,fit,s,goal,hidden):
    # The evaluator reveals exactly one factual probe effect per model.
    action=policy(PROBE_S) if mode=="passive" else 1-policy(PROBE_S)
    y=_effect(PROBE_S,action,hidden)
    model=fit["experts"] if mode in ("active","passive") else           fit["pooled"][None,:]
    if mode in ("active","passive"):
        belief=infer(fit["experts"],PROBE_S,action,y)
        selected=plan(fit["experts"],belief,s,goal)
    elif mode=="single":
        selected=plan(model,(1.,),s,goal)
        belief=(1.,)
    elif mode=="fixed":
        selected=0;belief=(1.,)
    else:
        raise ValueError("F8R6_UNKNOWN_CONTROL")
    # Selected action FIRST; oracle sees two outcomes only AFTER selection.
    realized=_effect(s,selected,hidden)
    oracle=min((_effect(s,a,hidden)-goal)**2 for a in SEQUEL_ACTIONS)
    regret=max(0.,(realized-goal)**2-oracle)
    identified=(belief!=(0.5,0.5)) if mode in ("active","passive") else False
    return {"regret":float(regret),"selected":selected,
            "identified":identified,"probe":action,
            "actual_probe":float(y)}


def validate_wrong_credit(logs,fit):
    X=np.stack([feature(s,a) for ep in logs for s,a,y in ep])
    Y=np.array([y for ep in logs for s,a,y in ep])
    true_coef=fit["individual_coeffs"]
    right=np.mean([np.mean((np.stack([feature(s,a) for s,a,y in ep])@coef-
                           np.array([y for s,a,y in ep]))**2)
                   for ep,coef in zip(logs,true_coef)])
    rng=np.random.Generator(np.random.PCG64(RNG_SHUFFLE))
    scrambled=rng.permutation(np.array([a for ep in logs for s,a,y in ep]))
    wcoeff=[]
    pos=0
    for ep in logs:
        acts=scrambled[pos:pos+len(ep)];pos+=len(ep)
        Xbad=np.stack([feature(s,int(a)) for (s,_,y),a in zip(ep,acts)])
        Yep=np.array([y for s,a,y in ep])
        wcoeff.append(_fit(Xbad,Yep))
    bad=np.mean([np.mean((np.stack([feature(s,a) for s,a,y in ep])@coef-
                         np.array([y for s,a,y in ep]))**2)
                 for ep,coef in zip(logs,wcoeff)])
    return {"correct_mse":float(right),"shuffled_mse":float(bad),
            "wrong_to_correct_ratio":float(bad/max(right,1e-12)),
            "action_credit_fixed_seed":RNG_SHUFFLE}


def cold_roundtrip(fit):
    data={"experts":fit["experts"].tolist(),"pooled":fit["pooled"].tolist()}
    encoded=json.dumps(data,sort_keys=True,allow_nan=False,
                       separators=(",",":"))
    decoded=json.loads(encoded)
    cold={k:np.asarray(v,dtype=np.float64) for k,v in decoded.items()}
    if (not np.array_equal(cold["experts"],fit["experts"])
        or not np.array_equal(cold["pooled"],fit["pooled"])):
        raise ValueError("F8R6_COLD_MODEL_RESTORE_DRIFT")
    # Cold retention of causal A sign after different B was visited.
    sig_a=infer(cold["experts"],PROBE_S,0,_effect(PROBE_S,0,-1))
    saved=json.loads(json.dumps(list(sig_a),allow_nan=False))
    sig_b=infer(cold["experts"],PROBE_S,0,_effect(PROBE_S,0,1))
    chosen_a=plan(cold["experts"],sig_a,-.2,.2)
    chosen_again=plan(cold["experts"],saved,-.2,.2)
    if (sig_a==sig_b or chosen_a!=chosen_again):
        raise ValueError("F8R6_COLD_ONE_BIT_RETENTION_OR_CONTEXT_INVALID")
    return {"model_bytes":len(encoded.encode()),"A_to_B_to_A_unchanged":True,
            "cold_experts_hash":_hash(cold["experts"])}


def run():
    locked()
    start=time.perf_counter()
    logs=training_episodes()
    fit=fit_controls(logs)
    fit_time=time.perf_counter()-start
    wrong=validate_wrong_credit(logs,fit)
    cold=cold_roundtrip(fit)
    before=time.perf_counter()
    paired=fixed_pairs()
    rows={}
    modes=("active","single","passive","fixed")
    for mode in modes:
        samples=[_score_one(mode,fit,s,goal,sign)
                 for sign in (-1,1) for s,goal in paired]
        if len(samples)!=SCORE_N*2:
            raise ValueError("F8R6_UNEXPECTED_SOURCE_NEUTRAL_BUDGET")
        rows[mode]={
            "goal_regret_mean":float(math.fsum(v["regret"] for v in samples)/len(samples)),
            "optimal_actions":sum(v["regret"]<=1e-12 for v in samples),
            "inferred_probe_outcomes":sum(v["identified"] for v in samples),
            "probe_action0":sum(v["probe"]==0 for v in samples),
            "probe_action1":sum(v["probe"]==1 for v in samples),
            "factual_selected_actions":len(samples)*2,
            "real_external_actions":0
        }
    predict_time=time.perf_counter()-before
    actual=rows["active"]["goal_regret_mean"]
    others=[rows[k]["goal_regret_mean"] for k in ("single","passive","fixed")]
    pass_dev=(all(v>0 and actual<=.1*v for v in others)
        and rows["active"]["inferred_probe_outcomes"]==2*SCORE_N
        and wrong["wrong_to_correct_ratio"]>=5
        and cold["A_to_B_to_A_unchanged"])
    return {
        "schema":"ttt-hybrid.f8r6-trained-2regime-active-causal-toy-result.v1",
        "status":"F8R6_SOURCE_NEUTRAL_DEV_COMPLETE",
        "F8_R5_original_negative_preserved":True,
        "fresh_toy_fit_sessions":FIT_N,
        "fresh_toy_fit_effects":FIT_N*HORIZON,
        "score_probes_and_selected_effects_per_arm":SCORE_N*2*2,
        "original_external_effects":0,
        "action_budget_matched":True,
        "expert_coeff_sha256":_hash(fit["experts"]),
        "single_coeff_sha256":_hash(fit["pooled"]),
        "expert_coeff":fit["experts"].tolist(),
        "single_coeff":fit["pooled"].tolist(),
        "controls":rows,
        "negative_wrong_action_credit":wrong,
        "cold_restoration":cold,
        "candidate_dev_pass":bool(pass_dev),
        "fit_wall_s":fit_time,
        "evaluation_wall_s":predict_time,
        "full_process_wall_s":time.perf_counter()-start,
        "independent_scientific_pass":False,
        "genuine_source_firstuse":False,
        "unfamiliar_transfer":False,
        "trained_Hybrid_scientific_gain":False,
        "deployment":False
    }


def store_ci_receipt(data):
    if os.environ.get("GITHUB_ACTIONS")!="true":
        return
    folder=ROOT/"local_runs"
    folder.mkdir(exist_ok=True)
    out=folder/"p85mr_f8r6_causal_memory_dev.json"
    if out.exists():
        raise FileExistsError("F8R6_NO_DUPLICATE_CI_RESULT")
    out.write_text(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n",
                   encoding="utf-8")


def refuse_external(*args,**kwargs):
    raise PermissionError("F8R6_REAL_FIRSTUSE_AND_F7_REPLAY_NOT_AUTHORIZED")
