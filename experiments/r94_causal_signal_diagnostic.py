"""R94: prospective public-causal-factor vs factual-reward-objective test-only diagnostic.

Five identical 4,801-param MLPs, factual selected intervention returns ONLY.
All protected older FIT/DEV/OOD, previous scored TEST worlds and trained model
weights are excluded. New evaluator-only paired DEV/OOD worlds are created
ONLY after every fully trained arm is frozen. This is not native model science.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import r85_delayed_composition as r85
from experiments import r86_native_compositional_value as r86
from experiments import r87_public_composition_diagnostic as r87
from experiments import r93_headcap_exposure_diagnostic as r93

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r94_causal_signal_freeze.json"
FREEZE_BLOB="3158ef1fe0ef7d0d518067a0151f217a600f3611"
PARENT="c7d372504f4f0c0b3695ca1987b3ac01a104b7ff"
SCHEMA="ttt-hybrid.r94-factual-causal-feature-objective-testonly.v1"
BRANCH="research/r94-causal-feature-residual-value-diagnostic-20261115"
# Original source first use was consumed for BOTH seeds in this exact CI run.
# All later commits/runs are ENGINEERING-ONLY and must not reopen DEV/OOD.
ORIGINAL_R94_SPENT_SHA="1ea70bc3b06872e737238cc94bc3efecdd3debe3"
ORIGINAL_R94_SPENT_RUN_ID="43789681024"
SEEDS=(1001111,1001123)
SALT={"FIT":25350111,"FIT_VALIDATION":25350211,"DEV":25350311,
      "OOD_SHIFT":25350411}
SHUFFLE_SALT=25350511
BOOTSTRAP_SALT=25350611
BATCH_SALT=25350711
UNIT_SALT=25359991
COUNT={"FIT":2048,"FIT_VALIDATION":256,"DEV":256,"OOD_SHIFT":256}
ARMS=("MLP64_RAW_DIRECT","MLP64_RAW_RESIDUAL","MLP64_COMPOSED_DIRECT",
      "MLP64_COMPOSED_RESIDUAL","MLP64_COMPOSED_SHUFFLED_RESIDUAL")
RAW_DIRECT,RAW_RES,COMP_DIRECT,COMP_RES,SHUFFLE_RES=ARMS
RESIDUAL={RAW_RES,COMP_RES,SHUFFLE_RES}
COMPOSED={COMP_DIRECT,COMP_RES,SHUFFLE_RES}
STEPS=4096
BATCH=16


def frozen() -> dict:
    if r85.git_blob(FREEZE.read_bytes())!=FREEZE_BLOB:
        raise RuntimeError("R94_ORIGINAL_PROSPECTIVE_PROTOCOL_GIT_BLOB_DRIFT")
    r93.frozen()
    f=json.loads(FREEZE.read_text(encoding="utf-8"))
    p,s,m,t,e,x=f["parent"],f["sources"],f["method"],f["training"],f["evaluation"],f["execution"]
    if (f["schema"]!=SCHEMA
        or f["stage"]!="PROSPECTIVE_FROZEN_BEFORE_R94_IMPLEMENTATION_TESTS_OR_NEW_SOURCE_FIRST_USE"
        or p["R93_engineering_SHA"]!=PARENT
        or p["R93_frozen_blob"]!=r93.FREEZE_BLOB
        or p["R93_original_result"]!="R93_EXPOSURE_HEADCAP_DEV_NO_GO_BOTH_SEEDS"
        or p["older_model_weights_or_scored_cohorts_reused"]
        or s["salts"]!={**SALT,"SHUFFLE":SHUFFLE_SALT,
                         "BOOTSTRAP":BOOTSTRAP_SALT,
                         "TRAIN_BATCHES":BATCH_SALT,"UNIT_ONLY":UNIT_SALT}
        or s["seeds"]!=list(SEEDS) or s["world_counts"]!=COUNT
        or s["FIT_each_of_8_balance_cells"]!=256
        or s["FIT_VALIDATION_DEV_OOD_each_of_8_balance_cells"]!=32
        or s["previous_scored_or_reserved_source_access"]
        or f["arms"]!=list(ARMS)
        or t["all_arms_steps"]!=STEPS or t["batch"]!=BATCH
        or t["total_process_limit_seconds_per_seed"]!=240
        or e["paired_bootstrap_resamples"]!=2000
        or e["feature_checks"]["composed_residual_minus_shuffled_DEV_reward_min"]!=.75
        or e["feature_checks"]["composed_residual_minus_best_fixed_DEV_reward_min"]!=1.25
        or e["feature_checks"]["composed_residual_oracle_DEV_regret_max"]!=.75
        or e["feature_checks"]["composed_residual_legal_rule_DEV_action_agreement_min"]!=.85
        or e["feature_checks"]["composed_residual_minus_shuffled_bootstrap_lower95_min"]!=.10
        or e["feature_checks"]["composed_residual_minus_shuffled_shifted_OOD_reward_min"]!=.50
        or e["feature_checks"]["composed_residual_minus_shuffled_shifted_OOD_cold_reward_min"]!=.50
        or e["objective_checks"]["composed_residual_minus_composed_direct_DEV_reward_min"]!=.15
        or e["objective_checks"]["composed_residual_minus_raw_residual_DEV_reward_min"]!=.15
        or e["objective_checks"]["composed_residual_minus_composed_direct_shifted_OOD_reward_min"]!=.10
        or e["independent_scientific_pass"]
        or x["new_workflow_file"] or x["new_controller"] or x["new_runner"]
        or any(v for k,v in f["authority"].items()
               if k!="cpu_scoped_test_only_development_allowed")
        or not f["authority"]["cpu_scoped_test_only_development_allowed"]):
        raise PermissionError("R94_FROZEN_SOURCE_METHOD_OR_AUTHORITY_MISMATCH")
    new=set(s["salts"].values())
    if len(new)!=len(s["salts"]) or new & set(p["older_spent_salts"]):
        raise RuntimeError("R94_OLD_SPENT_OR_DUPLICATE_SALT_REUSE")
    return f


def _action(seed: int, index: int) -> int:
    if seed not in SEEDS or not isinstance(index,int) or index<0:
        raise PermissionError("R94_UNFROZEN_SEED_OR_WORLD_INDEX")
    return (r85.CALIBRATE if ((index//4)+SEEDS.index(seed))%2==0
            else r85.EXPLOIT_POS)


def balance_preflight()->dict:
    """Index-only, before TRAIN/DEV/OOD physical source first use."""
    result={}
    for seed in SEEDS:
        result[str(seed)]={}
        for panel,n in COUNT.items():
            groups={}
            for index in range(n):
                k=(index%2,((index%r85.N)//2)%2,_action(seed,index))
                groups[k]=groups.get(k,0)+1
            expected=n//8
            if len(groups)!=8 or set(groups.values())!={expected}:
                raise PermissionError("R94_FACTUAL_ACTION_PIVOT_LAW_STRATA_CONFOUNDED")
            result[str(seed)][panel]={"eight_cells":8,"worlds_per_cell":expected}
    return result


def _admit(seed: int) -> str:
    frozen()
    if (seed not in SEEDS
        or os.getenv("GITHUB_ACTIONS")!="true"
        or os.getenv("GITHUB_EVENT_NAME")!="pull_request"
        or os.getenv("GITHUB_REPOSITORY")!="spoonman136668-ai/TTT-Hybrid-Lab"
        or os.getenv("GITHUB_HEAD_REF")!=BRANCH
        or os.getenv("GITHUB_JOB")!="r94-causal-signal"
        or os.getenv("R94_SEED")!=str(seed)
        or os.getenv("RUNNER_OS")!="Linux"
        or os.getenv("RUNNER_ENVIRONMENT")!="github-hosted"
        or os.getenv("GITHUB_RUN_ATTEMPT")!="1"
        or "/merge" not in os.getenv("GITHUB_REF","")):
        raise PermissionError("R94_ORIGINAL_FIRST_ATTEMPT_SCOPED_G4_HOSTED_JOB_REQUIRED")
    sha=os.getenv("EXPECTED_CANDIDATE_SHA","")
    if sha!=ORIGINAL_R94_SPENT_SHA or os.getenv("GITHUB_RUN_ID")!=ORIGINAL_R94_SPENT_RUN_ID:
        raise PermissionError("R94_ALREADY_CONSUMED_FIRST_USE_DENIED")
    head=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
                        capture_output=True,text=True,check=False,timeout=10)
    clean=subprocess.run(["git","diff","--quiet","HEAD","--"],cwd=ROOT,
                         capture_output=True,check=False,timeout=10)
    if len(sha)!=40 or head.returncode or head.stdout.strip()!=sha or clean.returncode:
        raise PermissionError("R94_EXACT_CLEAN_CANDIDATE_SHA_FAILED")
    return sha


def unit_only_world(index:int):
    if not isinstance(index,int) or not 0<=index<4:
        raise PermissionError("R94_ONLY_INDEPENDENT_UNIT_FIXTURES")
    rng=np.random.default_rng(np.random.SeedSequence([UNIT_SALT,index]))
    return r85._make_synthetic_world(index,rng),r85.CALIBRATE if index%2==0 else r85.EXPLOIT_POS


def _world(panel:str,seed:int,index:int,sha:str):
    if panel not in SALT:
        raise PermissionError("R94_PRIOR_SPENT_OR_SEALED_PANEL_DENIED")
    if seed not in SEEDS or not isinstance(index,int) or not 0<=index<COUNT[panel]:
        raise PermissionError("R94_NONFROZEN_PANEL_SEED_OR_INDEX")
    if (not sha or sha!=os.getenv("EXPECTED_CANDIDATE_SHA")
        or os.getenv("GITHUB_ACTIONS")!="true"
        or os.getenv("GITHUB_HEAD_REF")!=BRANCH
        or os.getenv("GITHUB_JOB")!="r94-causal-signal"
        or os.getenv("R94_SEED")!=str(seed)
        or os.getenv("GITHUB_RUN_ATTEMPT")!="1"
        or sha!=ORIGINAL_R94_SPENT_SHA
        or os.getenv("GITHUB_RUN_ID")!=ORIGINAL_R94_SPENT_RUN_ID
        or os.getenv("RUNNER_OS")!="Linux"
        or os.getenv("RUNNER_ENVIRONMENT")!="github-hosted"):
        raise PermissionError("R94_NEW_SOURCE_NOT_ADMITTED_BEFORE_ORIGINAL_CI")
    rng=np.random.default_rng(np.random.SeedSequence([SALT[panel],seed,index]))
    w=r85._make_synthetic_world(index%r85.N,rng)
    if panel=="OOD_SHIFT":
        w=r86._shift_public_world(w,index)
    return w,_action(seed,index)


def _features(prefix:np.ndarray)->tuple[np.ndarray,np.ndarray,float]:
    raw,sign=r87._legal_features(prefix)
    comp=raw.copy()
    comp[:4]=np.asarray([sign,0.,0.,0.],np.float32)
    if raw.shape!=(8,) or raw.dtype!=np.float32 or comp.shape!=(8,):
        raise RuntimeError("R94_NOT_LEGAL_PUBLIC_FEATURE_DIMENSION")
    return raw,comp,float(sign)


def _shuffle_index(pivot:np.ndarray,action:np.ndarray|None,seed:int,panel:str):
    if panel not in COUNT or seed not in SEEDS:
        raise PermissionError("R94_SHUFFLED_PRIOR_OR_SEALED_DATASET_REFUSED")
    if pivot.shape!=(COUNT[panel],):
        raise ValueError("R94_SHUFFLED_PUBLIC_PIVOT_GEOMETRY_MISMATCH")
    if action is not None and action.shape!=pivot.shape:
        raise ValueError("R94_SHUFFLED_FACTUAL_ACTION_GEOMETRY_MISMATCH")
    rng=np.random.default_rng(np.random.SeedSequence([SHUFFLE_SALT,seed,SALT[panel]]))
    source=np.arange(len(pivot))
    groups={}
    for i,timepoint in enumerate(pivot):
        k=(float(timepoint),int(action[i]) if action is not None else -1)
        groups.setdefault(k,[]).append(i)
    for ids in groups.values():
        if len(ids)<4:
            raise RuntimeError("R94_TOO_FEW_SAME_ACTION_PIVOT_SHUFFLE_CONTROL")
        group=np.asarray(ids,dtype=np.int64)
        shuffled=rng.permutation(group)
        # Always move each physical world's causal A evidence if permutation fixed points.
        if np.any(shuffled==group):
            shuffled=np.roll(group,1)
        source[group]=shuffled
    return source


def _factual(panel:str,seed:int,sha:str):
    if panel not in ("FIT","FIT_VALIDATION"):
        raise PermissionError("R94_NO_TRAINING_ON_EVALUATOR_ONLY_PANELS")
    n=COUNT[panel]
    raw=np.zeros((n,8),np.float32)
    comp=np.zeros_like(raw)
    target=np.zeros(n,np.float32)
    chosen=np.zeros(n,np.int8)
    for i in range(n):
        w,a=_world(panel,seed,i,sha)
        x,r=r86._view(w,a)
        f,c,_=_features(x)
        raw[i],comp[i],target[i],chosen[i]=f,c,r,a
    move=_shuffle_index(raw[:,6],chosen,seed,panel)
    shuffled=comp.copy()
    shuffled[:,0]=comp[move,0]
    if not np.array_equal(shuffled[:,4:7],comp[:,4:7]):
        raise RuntimeError("R94_SHUFFLED_CAUSAL_EVIDENCE_CHANGED_PUBLIC_ACTION")
    return {"raw":raw,"composed":comp,"shuffled":shuffled,
            "return":target,"chosen":chosen,"one_factual_realized_branch_only":True}


def _paired(panel:str,seed:int,sha:str):
    if panel not in ("DEV","OOD_SHIFT"):
        raise PermissionError("R94_EVALUATOR_CANNOT_LEARN_ON_NEW_SCORED_PANELS")
    n=COUNT[panel]
    raw0=np.zeros((n,8),np.float32);raw2=np.zeros_like(raw0)
    comp0=np.zeros_like(raw0);comp2=np.zeros_like(raw0)
    returns0=np.zeros(n,np.float64);returns2=np.zeros(n,np.float64)
    legal=np.zeros(n,np.int64)
    last=np.zeros(n,np.int64)
    recent=np.zeros(n,np.int64)
    cold=np.zeros(n,bool)
    for i in range(n):
        w,_=_world(panel,seed,i,sha)
        x0,r0=r86._view(w,r85.CALIBRATE)
        x2,r2=r86._view(w,r85.EXPLOIT_POS)
        if (not np.array_equal(x0[:-1],x2[:-1])
            or not np.array_equal(x0[-1,4:],x2[-1,4:])
            or x0[-1,r85.CALIBRATE]!=1.0 or x2[-1,r85.EXPLOIT_POS]!=1.0):
            raise RuntimeError("R94_LEAKED_UNCHOSEN_ACTION_IN_PUBLIC_PREFIX")
        r0f,c0,sgn0=_features(x0)
        r2f,c2,sgn2=_features(x2)
        if sgn0!=sgn2 or not np.array_equal(r0f[:4],r2f[:4]):
            raise RuntimeError("R94_DIFFERENT_CAUSAL_CUES_BY_EVALUATOR_ACTION")
        raw0[i],raw2[i],comp0[i],comp2[i]=r0f,r2f,c0,c2
        returns0[i],returns2[i]=r0,r2
        legal[i],last[i],recent[i]=(r85._policy(x0),
            r85._policy(x0,"LAST_A"),r85._policy(x0,"RECENT_B"))
        cold[i]=bool(i%2)
    perm=_shuffle_index(comp0[:,6],None,seed,panel)
    shuffled0=comp0.copy();shuffled2=comp2.copy()
    shuffled0[:,0]=comp0[perm,0]
    shuffled2[:,0]=comp0[perm,0]
    if not np.array_equal(shuffled0[:,4:7],comp0[:,4:7]) or not np.array_equal(
            shuffled2[:,4:7],comp2[:,4:7]):
        raise RuntimeError("R94_SHUFFLED_EVALUATOR_CHANGED_PUBLIC_ACTION")
    return {"raw0":raw0,"raw2":raw2,"comp0":comp0,"comp2":comp2,
            "shuffled0":shuffled0,"shuffled2":shuffled2,
            "reward0":returns0,"reward2":returns2,
            "legal":legal,"last":last,"recent":recent,"cold":cold}


def _action_means(train:dict)->dict:
    return {a:float(np.mean(train["return"][train["chosen"]==a]))
            for a in (r85.CALIBRATE,r85.EXPLOIT_POS)}


def _offset(features:np.ndarray,means:dict)->np.ndarray:
    # Feature slot 4 (not slot 0) is the actual selected CALIBRATE action.
    return np.where(features[:,4]>.5,
                    means[r85.CALIBRATE],means[r85.EXPLOIT_POS]).astype(np.float32)


def _arm_input(source:dict,arm:str)->np.ndarray:
    return source["shuffled" if arm==SHUFFLE_RES
                  else "composed" if arm in COMPOSED else "raw"]


def _models(seed:int)->dict:
    out={}
    for arm in ARMS:
        torch.manual_seed(seed+94)
        out[arm]=r87.PublicValueMLP()
    params=[sum(p.numel() for p in model.parameters()) for model in out.values()]
    if len(set(params))!=1 or params[0]!=4801:
        raise RuntimeError("R94_MLP_DIAGNOSTIC_MODEL_CAPACITY_CHANGED")
    first=dict(out[ARMS[0]].named_parameters())
    for arm in ARMS[1:]:
        another=dict(out[arm].named_parameters())
        if first.keys()!=another.keys() or any(
            not torch.equal(first[k],another[k]) for k in first):
            raise RuntimeError("R94_MLP_ARMS_NOT_SAME_INITIAL_WEIGHT_TENSORS")
    return out


def _predict(m,features:np.ndarray)->np.ndarray:
    m.eval()
    outputs=[]
    with torch.no_grad():
        for i in range(0,len(features),BATCH):
            outputs.extend(m(torch.from_numpy(features[i:i+BATCH])).tolist())
    m.train()
    result=np.asarray(outputs,np.float64)
    if len(result)!=len(features) or not np.isfinite(result).all():
        raise RuntimeError("R94_NONFINITE_FACTUAL_VALUE_PREDICTIONS")
    return result


def _mean_square(m,arm,source,means)->float:
    features=_arm_input(source,arm)
    pred=_predict(m,features)
    if arm in RESIDUAL:
        pred=pred+_offset(features,means)
    mse=float(np.mean((pred-source["return"])**2))
    if not math.isfinite(mse):
        raise RuntimeError("R94_NONFINITE_REAL_RETURN_MSE")
    return mse


def _train(m,arm,fit,valid,means,draws):
    features=_arm_input(fit,arm)
    y=fit["return"].copy()
    if arm in RESIDUAL:
        y-=_offset(features,means)
    x=torch.from_numpy(features)
    target=torch.from_numpy(y)
    optimizer=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    first=_mean_square(m,arm,valid,means)
    started=time.perf_counter()
    for k in range(STEPS):
        indices=draws[k]
        estimated=m(x[indices])
        loss=F.mse_loss(estimated,target[indices])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("R94_NONFINITE_ACTUAL_SELECTED_RETURN_LOSS")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(),1.)
        optimizer.step()
    elapsed=time.perf_counter()-started
    train_mse=_mean_square(m,arm,fit,means)
    val_mse=_mean_square(m,arm,valid,means)
    if not all(math.isfinite(z) for z in (train_mse,val_mse,elapsed)):
        raise RuntimeError("R94_NONFINITE_FULLY_TRAINED_FACTUAL_ARM")
    return {"updates":STEPS,"batch":BATCH,"factual_event_exposures":STEPS*BATCH,
            "parameters":sum(p.numel() for p in m.parameters()),
            "initial_FIT_VALIDATION_MSE":first,
            "final_FIT_TRAIN_MSE":train_mse,
            "final_FIT_VALIDATION_MSE":val_mse,
            "train_wall_seconds":elapsed,
            "trained_on_one_selected_factual_return_only":True}


def _score(models:dict,panel:dict,means:dict)->dict:
    rewards={}
    choices={}
    by_model={}
    for arm,m in models.items():
        key=("shuffled" if arm==SHUFFLE_RES
             else "comp" if arm in COMPOSED else "raw")
        v0=_predict(m,panel[key+"0"])
        v2=_predict(m,panel[key+"2"])
        if arm in RESIDUAL:
            v0+=means[r85.CALIBRATE]
            v2+=means[r85.EXPLOIT_POS]
        choices[arm]=np.where(v0>v2,r85.CALIBRATE,r85.EXPLOIT_POS)
    choices["TRAIN_ACTION_MEAN_FIXED"]=np.full(len(panel["reward0"]),
        r85.CALIBRATE if means[r85.CALIBRATE]>means[r85.EXPLOIT_POS]
        else r85.EXPLOIT_POS,dtype=np.int64)
    choices["ALWAYS_CALIBRATE"]=np.full(len(panel["reward0"]),r85.CALIBRATE,dtype=np.int64)
    choices["ALWAYS_EXPLOIT_POS"]=np.full(len(panel["reward0"]),r85.EXPLOIT_POS,dtype=np.int64)
    choices["PUBLIC_A_PARITY_RULE_EVALUATOR_ONLY"]=panel["legal"]
    choices["LAST_A_CUE_EVALUATOR_ONLY"]=panel["last"]
    choices["RECENT_B_RULE_EVALUATOR_ONLY"]=panel["recent"]
    oracle=np.maximum(panel["reward0"],panel["reward2"])
    for name,a in choices.items():
        result=np.where(a==r85.CALIBRATE,panel["reward0"],panel["reward2"])
        rewards[name]=result
        by_model[name]={"realized_mean_reward":float(result.mean()),
                        "cold_mean_reward":float(result[panel["cold"]].mean()),
                        "early_mean_reward":float(result[~panel["cold"]].mean()),
                        "physical_oracle_regret":float((oracle-result).mean()),
                        "legal_rule_action_agreement":float(np.mean(a==panel["legal"]))}
    by_model["PAIRED_PHYSICAL_ORACLE_EVALUATOR_ONLY"]={
        "realized_mean_reward":float(oracle.mean()),
        "cold_mean_reward":float(oracle[panel["cold"]].mean()),
        "early_mean_reward":float(oracle[~panel["cold"]].mean()),
        "physical_oracle_regret":0.0}
    return {"metrics":by_model,"rewards":rewards,
            "policies":choices,"n":len(panel["reward0"])}


def _paired_bootstrap_lower(a:np.ndarray,b:np.ndarray,seed:int)->float:
    if a.shape!=(COUNT["DEV"],) or b.shape!=a.shape:
        raise ValueError("R94_FROZEN_PAIRED_BOOTSTRAP_SOURCE_COUNT")
    rng=np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SALT,seed]))
    indexes=rng.integers(0,len(a),size=(2000,len(a)))
    return float(np.quantile((a-b)[indexes].mean(axis=1),.025))


def original_ci(seed:int)->dict:
    sha=_admit(seed)
    balanced=balance_preflight()
    torch.set_num_threads(1)
    path=ROOT/f"local_runs/r94_causal_signal_seed{seed}.json"
    if path.exists():
        raise FileExistsError("R94_ORIGINAL_SOURCE_FIRST_USE_OR_UNCERTAIN_REPLAY_DENIED")
    started=time.perf_counter()
    out={"schema":SCHEMA,"candidate_SHA":sha,"freeze_blob":FREEZE_BLOB,
         "R93_parent_SHA":PARENT,"seed":seed,"status":"R94_ENGINEERING_OR_RESOURCE_INCOMPLETE_NOT_SCIENTIFIC_NO_GO",
         "platform":platform.platform(),"torch":str(torch.__version__),
         "pre_first_use_balance":balanced[str(seed)],"completed_arms":{},
         "all_five_models_frozen_before_DEV_OOD":False,
         "DEV_OOD_FIRST_USE":False,"older_scored_or_reserved_source_used":False,
         "scientific_pass":False}
    try:
        fit=_factual("FIT",seed,sha)
        valid=_factual("FIT_VALIDATION",seed,sha)
        if not fit["one_factual_realized_branch_only"]:
            raise RuntimeError("R94_DID_NOT_USE_REAL_LOGGED_FACTUAL_RETURN")
        means=_action_means(fit)
        out["TRAIN_action_only_factual_means"]={str(k):v for k,v in means.items()}
        rng=np.random.default_rng(np.random.SeedSequence([BATCH_SALT,seed]))
        draws=rng.integers(0,COUNT["FIT"],size=(STEPS,BATCH))
        models=_models(seed)
        for arm in ARMS:
            out["completed_arms"][arm]=_train(models[arm],arm,fit,valid,means,draws)
            if time.perf_counter()-started>frozen()["training"]["total_process_limit_seconds_per_seed"]:
                raise TimeoutError("R94_FROZEN_TOTAL_TRAINING_CPU_RESOURCE_BOUND")
        if set(out["completed_arms"])!=set(ARMS):
            raise RuntimeError("R94_INCOMPLETE_MATCHED_TRAINING_ARMS")
        out["all_five_models_frozen_before_DEV_OOD"]=True
        # NEW DEV and shifted OOD worlds constructed only after ALL arms freeze.
        out["DEV_OOD_FIRST_USE"]=True
        dev=_paired("DEV",seed,sha)
        ood=_paired("OOD_SHIFT",seed,sha)
        d=_score(models,dev,means)
        o=_score(models,ood,means)
        dm,om=d["metrics"],o["metrics"]
        dp=dm[COMP_RES];ds=dm[SHUFFLE_RES]
        op=om[COMP_RES];os=om[SHUFFLE_RES]
        fixed=max(dm["ALWAYS_CALIBRATE"]["realized_mean_reward"],
                  dm["ALWAYS_EXPLOIT_POS"]["realized_mean_reward"],
                  dm["TRAIN_ACTION_MEAN_FIXED"]["realized_mean_reward"])
        boot=_paired_bootstrap_lower(d["rewards"][COMP_RES],d["rewards"][SHUFFLE_RES],seed)
        measurements={
            "DEV_comp_res_minus_shuffled_mean":dp["realized_mean_reward"]-ds["realized_mean_reward"],
            "DEV_comp_res_minus_best_fixed":dp["realized_mean_reward"]-fixed,
            "DEV_comp_res_oracle_regret":dp["physical_oracle_regret"],
            "DEV_comp_res_legal_action_agreement":dp["legal_rule_action_agreement"],
            "DEV_comp_res_minus_shuffled_paired_lower95":boot,
            "OOD_comp_res_minus_shuffled":op["realized_mean_reward"]-os["realized_mean_reward"],
            "OOD_cold_comp_res_minus_shuffled":op["cold_mean_reward"]-os["cold_mean_reward"],
            "DEV_comp_res_minus_comp_direct":dp["realized_mean_reward"]-dm[COMP_DIRECT]["realized_mean_reward"],
            "DEV_comp_res_minus_raw_res":dp["realized_mean_reward"]-dm[RAW_RES]["realized_mean_reward"],
            "OOD_comp_res_minus_comp_direct":op["realized_mean_reward"]-om[COMP_DIRECT]["realized_mean_reward"]}
        t=frozen()["evaluation"]
        feat=t["feature_checks"];obj=t["objective_checks"]
        fc={
            "DEV_above_shuffled":measurements["DEV_comp_res_minus_shuffled_mean"]>=feat["composed_residual_minus_shuffled_DEV_reward_min"],
            "DEV_above_best_fixed":measurements["DEV_comp_res_minus_best_fixed"]>=feat["composed_residual_minus_best_fixed_DEV_reward_min"],
            "DEV_oracle_regret":dp["physical_oracle_regret"]<=feat["composed_residual_oracle_DEV_regret_max"],
            "DEV_lawful_action_agreement":dp["legal_rule_action_agreement"]>=feat["composed_residual_legal_rule_DEV_action_agreement_min"],
            "DEV_paired_bootstrap":boot>=feat["composed_residual_minus_shuffled_bootstrap_lower95_min"],
            "OOD_shifted_above_shuffled":measurements["OOD_comp_res_minus_shuffled"]>=feat["composed_residual_minus_shuffled_shifted_OOD_reward_min"],
            "OOD_cold_above_shuffled":measurements["OOD_cold_comp_res_minus_shuffled"]>=feat["composed_residual_minus_shuffled_shifted_OOD_cold_reward_min"]}
        oc={
            "DEV_centering_improves_composed":measurements["DEV_comp_res_minus_comp_direct"]>=obj["composed_residual_minus_composed_direct_DEV_reward_min"],
            "DEV_composed_residual_beats_raw_residual":measurements["DEV_comp_res_minus_raw_res"]>=obj["composed_residual_minus_raw_residual_DEV_reward_min"],
            "OOD_centering_improves_composed":measurements["OOD_comp_res_minus_comp_direct"]>=obj["composed_residual_minus_composed_direct_shifted_OOD_reward_min"]}
        out["DEV"]=dm;out["OOD_SHIFT"]=om
        out["comparisons"]=measurements
        out["feature_checks"]={k:bool(v) for k,v in fc.items()}
        out["objective_checks"]={k:bool(v) for k,v in oc.items()}
        out["status"]=(t["status_both_feature_and_objective"]
                       if all(fc.values()) and all(oc.values())
                       else t["status_feature_only"] if all(fc.values())
                       else t["status_no_feature"])
    except BaseException as exc:
        out["error_type"]=type(exc).__name__
        out["status"]="R94_ENGINEERING_OR_RESOURCE_INCOMPLETE_NOT_SCIENTIFIC_NO_GO"
        raise
    finally:
        out["total_process_wall_seconds"]=time.perf_counter()-started
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("x",encoding="utf-8") as file:
            json.dump(out,file,indent=2,sort_keys=True,allow_nan=False)
            file.write("\n")
    return out


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,choices=SEEDS,required=True)
    seed=parser.parse_args().seed
    v=original_ci(seed)
    print(json.dumps({"seed":seed,"status":v["status"],
      "arms":len(v["completed_arms"]),"scored":v["DEV_OOD_FIRST_USE"],
      "scientific_pass":v["scientific_pass"]},sort_keys=True))
