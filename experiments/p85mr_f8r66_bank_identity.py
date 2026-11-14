"""R66 frozen source-neutral causal bank identity falsifier, author-synthetic DEV only.

This independent line branches from qualified R63, NOT failed R64/R65. Its
protocol was committed before this module existed. No new model pretraining,
protected OOD, unchosen consequences, or independent first-use.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT/"experiments/p85mr_f8r66_bank_identity_freeze.json"
FREEZE_BLOB = "4b254705ba61bd4f659e044ec566cc8f7dce7831"
SCHEMA = "ttt-hybrid.f8r66-bank-identity-falsifier-freeze.v1"
PARENT = "fed5b9b1bcd58ae151b62d96b44dfdf0665695cc"
SALT = {"FIT":23810000,"DEV":23820000,"PROXY_NEGATIVE":23830000}
COUNTS = {"FIT":48,"DEV":96}
MODES = ("protected_role_bank","single_allwrite","dual_mixing",
         "surprise_only","global_protected","erase_old_ablation")
CONTROLS = ("single_allwrite","dual_mixing","surprise_only","global_protected")
D = 4
STEPS = 48
GAIN = .65
NOVELTY = .75
CONF = .20
MARGIN = .15
BOOT_SALT = 23849999


def git_blob(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def locked():
    raw = FREEZE.read_bytes()
    if git_blob(raw) != FREEZE_BLOB:
        raise ValueError("F8R66_ORIGINAL_FREEZE_BLOB_DRIFT")
    f = json.loads(raw)
    x=f["source"]
    m=f["memory"]
    if (f["schema"]!=SCHEMA
            or f["stage"]!="PROSPECTIVE_FREEZE_ONLY_BEFORE_R66_SOURCE_OR_DEV_SCORE"
            or f["parent"]["base_R63_SHA"]!=PARENT
            or not f["parent"]["do_not_reuse_spent_scored_cohorts"]
            or (x["FIT_salt"],x["DEV_salt"],x["PROXY_NEGATIVE_salt"])
               != tuple(SALT.values())
            or (x["FIT_worlds"],x["DEV_worlds"])!=(48,96)
            or x["DEV_stable_worlds"]!=48
            or x["train_or_calibration"].startswith("No parametric training") is False
            or (m["learning_gain"],m["novelty_threshold_l2"],
                m["confidence_threshold_l2"],m["return_margin_l2"])
               != (GAIN,NOVELTY,CONF,MARGIN)
            or m["events_per_world"]!=STEPS
            or m["same_capacity_and_action_information"] is not True
            or f["metrics"]["bootstrap_seed"]!=BOOT_SALT
            or f["metrics"]["post_DEV_adjustments_forbidden"] is not True
            or any(f["authority"].values())):
        raise ValueError("F8R66_FROZEN_PROTOCOL_DRIFT")
    return f


def worlds(panel):
    if panel not in COUNTS:
        raise PermissionError("F8R66_PROTECTED_OR_UNREGISTERED_SOURCE_DENIED")
    rng=np.random.default_rng(SALT[panel])
    count=COUNTS[panel]
    switches=np.array([False]*(count//2)+[True]*(count//2))
    rng.shuffle(switches)
    roles=np.arange(STEPS,dtype=np.int64)%D
    out=[]
    for i in range(count):
        law_a=rng.normal(0,.75,(D,D))+np.eye(D)*.25
        law_b=law_a.copy()
        if switches[i]:
            law_b[:2,:2]=np.stack((-law_a[1,:2],law_a[0,:2]),axis=0)
        effects=np.empty((STEPS,D),dtype=np.float64)
        for t,role in enumerate(roles):
            law=law_b if 16<=t<32 else law_a
            effects[t]=law[:,role]+rng.normal(0,.04,D)
        proxy=rng.normal(0,.8,(STEPS,D))
        out.append({"panel":panel,"index":i,"roles":roles.copy(),
                    "selected_effects":effects,"proxy":proxy,
                    "switched_evaluator_only":bool(switches[i]),
                    "private_law_A":law_a,"private_law_B":law_b})
    return out


def proxy_negative(dev):
    if len(dev)!=COUNTS["DEV"] or any(w["panel"]!="DEV" for w in dev):
        raise PermissionError("F8R66_PROXY_SOURCE_MUST_BE_DEV")
    rng=np.random.default_rng(SALT["PROXY_NEGATIVE"])
    return [{**w,"proxy":rng.normal(0,3,(STEPS,D))} for w in dev]


def _validate_factual(roles,effects):
    if (not isinstance(roles,np.ndarray) or roles.shape!=(STEPS,)
            or roles.dtype!=np.int64 or not np.array_equal(roles,np.arange(STEPS)%D)
            or not isinstance(effects,np.ndarray)
            or effects.shape!=(STEPS,D) or not np.isfinite(effects).all()):
        raise ValueError("F8R66_BAD_SELECTED_FACTUAL_EVENTS")


def update_role_bank(mode,roles,effects):
    """No private world identity, regime labels or future selected effects accepted."""
    if mode not in MODES:
        raise ValueError("F8R66_INVALID_MECHANISM")
    _validate_factual(roles,effects)
    w0=np.zeros((D,D),dtype=np.float64)
    w1=np.zeros((D,D),dtype=np.float64)
    role_selector=np.zeros(D,dtype=np.float64)
    errors=np.zeros(STEPS,dtype=np.float64)
    work=0
    switches=0
    snapshots=None
    for t in range(STEPS):
        role=int(roles[t])
        active=int(role_selector[0] if mode=="global_protected" else role_selector[role])
        bank=(w0,w1)
        if mode=="dual_mixing":
            prediction=(w0[:,role]+w1[:,role])*.5
        else:
            prediction=bank[active][:,role].copy()
        observed=effects[t]
        # PRE-observation error is fixed before any consequences arrive.
        errors[t]=float(np.mean((prediction-observed)**2))
        a=active
        b=1-a
        if mode in ("protected_role_bank","erase_old_ablation","global_protected","surprise_only"):
            trained_norm=float(np.linalg.norm(bank[a][:,role]))
            inactive_norm=float(np.linalg.norm(bank[b][:,role]))
            active_err=float(np.linalg.norm(bank[a][:,role]-observed))
            inactive_err=float(np.linalg.norm(bank[b][:,role]-observed))
            fit_other=(inactive_norm>=CONF and
                       inactive_err+MARGIN<active_err)
            novel=(trained_norm>=CONF and active_err>NOVELTY and
                   inactive_norm<CONF)
            choose_new = novel if mode=="surprise_only" else fit_other or novel
            if choose_new:
                a=b
                switches+=1
                if mode=="global_protected":
                    role_selector[0]=float(a)
                else:
                    role_selector[role]=float(a)
        if mode=="dual_mixing":
            w0[:,role]+=GAIN*.5*(observed-w0[:,role])
            w1[:,role]+=GAIN*.5*(observed-w1[:,role])
            work+=2
        else:
            bank[a][:,role]+=GAIN*(observed-bank[a][:,role])
            work+=1
            if mode=="erase_old_ablation":
                bank[1-a][:,role]+=GAIN*(observed-bank[1-a][:,role])
                work+=1
        if t==31:
            snapshots=(w0.copy(),w1.copy())
    assert snapshots is not None
    state=np.concatenate([w0.ravel(),w1.ravel(),role_selector])
    if state.size!=36 or not np.isfinite(state).all():
        raise ValueError("F8R66_FAST_STATE_BUDGET_BROKEN")
    return {"errors":errors,
            "state_hash":hashlib.sha256(state.astype("<f8").tobytes()).hexdigest(),
            "snapshot":snapshots,
            "allocated_fast_floats":36,
            "factual_selected_events":STEPS,
            "matrix_write_operations":work,
            "bank_switch_events":switches}


def _scored_world(mode,world):
    t0=time.perf_counter()
    # The mechanism receives ONLY factual selected roles and observed goal effects.
    o=update_role_bank(mode,world["roles"],world["selected_effects"])
    e=o["errors"]
    switched=world["switched_evaluator_only"]
    # Cold return after one genuinely observed role-0 and role-1 A outcome
    # at t32 and t33; the first unseen return observation is not predictable.
    ret_ix=np.array([36,37,40,41,44,45])
    b_ix=np.array([20,21,24,25,28,29])
    a,b=o["snapshot"]
    truth_a=world["private_law_A"][:,:2]
    truth_b=world["private_law_B"][:,:2]
    truth_A_error=float(np.mean((a[:,:2]-truth_a)**2))
    truth_B_error=float(np.mean((b[:,:2]-truth_b)**2))
    sep=float(np.linalg.norm(a[:,:2]-b[:,:2])/
              max(np.linalg.norm(a[:,:2]),np.linalg.norm(b[:,:2]),1e-9))
    return {"index":world["index"],"switched":switched,
            "B_MSE":float(np.mean(e[b_ix])),
            "return_MSE":float(np.mean(e[ret_ix])),
            "stable_MSE":float(np.mean(e)),
            "A_truth_MSE":truth_A_error,
            "B_truth_MSE":truth_B_error,
            "separation":sep,
            "state_hash":o["state_hash"],
            "matrix_writes":o["matrix_write_operations"],
            "factual_selected_events":o["factual_selected_events"],
            "allocated_fast_floats":o["allocated_fast_floats"],
            "bank_switches":o["bank_switch_events"],
            "wall_seconds":time.perf_counter()-t0}


def aggregate(mode,src):
    rows=[_scored_world(mode,w) for w in src]
    sw=[x for x in rows if x["switched"]]
    st=[x for x in rows if not x["switched"]]
    if len(sw)!=len(st) or len(sw)!=len(src)//2:
        raise ValueError("F8R66_BAD_EVALUATION_BALANCE")
    mean=lambda items,k:float(np.mean([x[k] for x in items]))
    return {"return_MSE":mean(sw,"return_MSE"),
            "B_MSE":mean(sw,"B_MSE"),
            "stable_MSE":mean(st,"stable_MSE"),
            "A_truth_MSE":mean(sw,"A_truth_MSE"),
            "B_truth_MSE":mean(sw,"B_truth_MSE"),
            "separation":mean(sw,"separation"),
            "bank_switches_total":int(sum(x["bank_switches"] for x in rows)),
            "matrix_writes_total":int(sum(x["matrix_writes"] for x in rows)),
            "factual_selected_events":len(rows)*STEPS,
            "fast_floats_per_world":36,
            "per_world":rows}


def judgment(scores):
    frozen=locked()["metrics"]["thresholds"]
    candidate=scores["protected_role_bank"]
    best=min(CONTROLS,key=lambda x:scores[x]["return_MSE"])
    ctrl=scores[best]
    mix=scores["dual_mixing"]
    ablation=scores["erase_old_ablation"]
    relative=lambda x,y: x/max(y,1e-12)-1
    return_gain=1-candidate["return_MSE"]/max(ctrl["return_MSE"],1e-12)
    B_reg=relative(candidate["B_MSE"],ctrl["B_MSE"])
    stable_reg=relative(candidate["stable_MSE"],ctrl["stable_MSE"])
    A_truth_ratio=candidate["A_truth_MSE"]/max(mix["A_truth_MSE"],1e-12)
    ablation_increase=relative(ablation["return_MSE"],candidate["return_MSE"])
    paired=[]
    for c,b in zip(candidate["per_world"],ctrl["per_world"]):
        if c["index"]!=b["index"] or c["switched"]!=b["switched"]:
            raise ValueError("F8R66_UNPAIRED_WORLD")
        if c["switched"]:
            paired.append(c["return_MSE"]-b["return_MSE"])
    rng=np.random.default_rng(BOOT_SALT)
    ix=rng.integers(0,len(paired),(2000,len(paired)))
    upper=float(np.quantile(np.asarray(paired)[ix].mean(axis=1),.975))
    checks={
       "cold_return_gain":return_gain>=frozen["A_return_MSE_relative_reduction_vs_strongest_control_min"],
       "B_nonregression":B_reg<=frozen["B_MSE_relative_regression_max"],
       "stable_nonregression":stable_reg<=frozen["stable_MSE_relative_regression_max"],
       "A_bank_preservation":A_truth_ratio<=frozen["bank_A_truth_MSE_ratio_vs_dual_mixing_max"],
       "distinct_banks":candidate["separation"]>=frozen["distinct_bank_separation_min"],
       "protection_ablation":ablation_increase>=frozen["ablation_return_MSE_relative_increase_min"],
       "paired_bootstrap_upper_negative":upper<0}
    return {"strongest_eligible_control":best,
            "return_relative_gain":return_gain,
            "B_relative_regression":B_reg,
            "stable_relative_regression":stable_reg,
            "A_bank_truth_MSE_ratio_vs_dual":A_truth_ratio,
            "ablation_return_relative_increase":ablation_increase,
            "paired_return_bootstrap_upper_95pct":upper,
            "checks":checks}


def run():
    freeze=locked()
    fit=worlds("FIT")
    dev=worlds("DEV")
    # FIT is source qualification only; no threshold search on FIT or DEV.
    if (len(fit)!=48 or len(dev)!=96
            or sum(x["switched_evaluator_only"] for x in fit)!=24
            or sum(x["switched_evaluator_only"] for x in dev)!=48):
        raise ValueError("F8R66_SOURCE_QUALIFICATION_FAIL")
    t0=time.perf_counter()
    scores={mode:aggregate(mode,dev) for mode in MODES}
    j=judgment(scores)
    negative=proxy_negative(dev)
    proxy_ok=True
    cold_ok=True
    for mode in MODES:
        # independent source manipulation changes only irrelevant proxy;
        # it MUST leave all predictive errors / latent memories identical.
        original=scores[mode]["per_world"]
        rep=aggregate(mode,dev)["per_world"]
        neg=aggregate(mode,negative)["per_world"]
        for a,b,c in zip(original,rep,neg):
            signature=lambda v:(v["return_MSE"],v["B_MSE"],
                                v["stable_MSE"],v["state_hash"])
            cold_ok &= signature(a)==signature(b)
            proxy_ok &= signature(a)==signature(c)
    j["checks"]["proxy_invariance"]=bool(proxy_ok)
    j["checks"]["cold_replay_exact"]=bool(cold_ok)
    j["DEV_mechanism_signal"]=all(j["checks"].values())
    return {"schema":SCHEMA,
            "freeze_blob":FREEZE_BLOB,"parent":PARENT,
            "classification":freeze["classification"],
            "source":{"FIT":len(fit),"DEV":len(dev),
                      "switched_DEV":sum(x["switched_evaluator_only"] for x in dev),
                      "protected_OOD_generated":0},
            "modes":{mode:{k:v for k,v in scores[mode].items() if k!="per_world"}
                     for mode in MODES},
            "judgment":j,
            "verdict":"DEV_MECHANISM_SIGNAL_ONLY" if j["DEV_mechanism_signal"]
                      else "DEV_NO_GO",
            "no_parametric_training":True,
            "independent_scientific_pass":False,
            "GPU_firstuse":False,
            "external_science_firstuse":False,
            "whole_process_wall_seconds":time.perf_counter()-t0}


def retain_ci_receipt(result):
    if os.getenv("GITHUB_ACTIONS")!="true":
        return
    path=ROOT/"local_runs/p85mr_f8r66_bank_identity_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R66_NEVER_OVERWRITE_ORIGINAL_DEV")
    path.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")


def deny_original_firstuse(*_args,**_kwargs):
    raise PermissionError("F8R66_EXTERNAL_ORIGINAL_FIRST_USE_DENIED")
