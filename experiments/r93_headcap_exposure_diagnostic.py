"""R93: original source-frozen factual exposure and readout-gradient conditioning test.

All TRAIN/DEV_TEST worlds are NEW, TEST-only and isolated from every scored
R85-R92 cohort. An original exact-clean-SHA first-attempt Linux G4 CI job is
the sole first-use admission; the existing Windows G4 runs only negative and
interface tests. All five-row event values are actual publicly observed rows.
No protected independent science, official fused CUDA, promotion or deployment.
"""
from __future__ import annotations
import gc
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import r74_capability_ceiling as r74
from experiments import r75_triangular_fastweight as r75
from experiments import r85_delayed_composition as r85
from experiments import r88_causal_event_access as r88
from experiments import r89_native_event_access as r89
from experiments import r92_native_diversity_lr_diagnostic as r92

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r93_headcap_exposure_freeze.json"
FREEZE_BLOB="1f2d6f74bb885cd434924396e03e638b7d064b96"
SCHEMA="ttt-hybrid.r93-test-only-balanced-exposure-headgrad-v1"
PARENT="84bf0aa6e3650fc8a9dfc8e0f97becb15b8eda30"
BRANCH="research/r93-native-exposure-head-conditioning-20261115"
SEEDS=(1000811,1000823)
TRAIN_SALT=25340111
DEV_SALT=25340211
SHUFFLE_SALT=25340311
BATCH_SALT=25340411
UNIT_TEST_SALT=25349999
HARD="NATIVE_HARD_GLOBAL1"
HEADCAPPED="NATIVE_HARD_HEADCAP1"
NO_DEPTH="NATIVE_NO_DEPTH_GLOBAL1"
GRU="GRU64_SLOTS5"
SHUFFLED="GRU64_SHUFFLED_A_SLOTS5"
ARMS=(HARD,HEADCAPPED,NO_DEPTH,GRU,SHUFFLED)
NATIVE=ARMS[:3]
LEVELS=((1024,256,1),(32,256,8),(32,256,1),(32,16,8),(32,16,1))
LEVEL_ARMS={0:NATIVE,1:ARMS,2:ARMS,3:ARMS,4:ARMS}
MAX_SECONDS=330
BATCH=2


def frozen()->dict:
    if r85.git_blob(FREEZE.read_bytes())!=FREEZE_BLOB:
        raise RuntimeError("R93_IMMUTABLE_PROSPECTIVE_FREEZE_BLOB_DRIFT")
    r92.frozen()
    f=json.loads(FREEZE.read_text(encoding="utf-8"))
    p,s,d,t,v,e=f["parent"],f["source"],f["design"],f["training"],f["frozen_evaluation"],f["execution"]
    if (f["schema"]!=SCHEMA
        or f["stage"]!="PROSPECTIVE_IMMUTABLE_PROTOCOL_BEFORE_R93_CODE_REGRESSION_OR_TEST_FIRST_USE"
        or p["R92_exact_SHA"]!=PARENT or p["R92_frozen_blob"]!=r92.FREEZE_BLOB
        or p["R92_final"]!="R92_NATIVE_DIVERSITY_LR_NO_GO_DIAGNOSTIC_ONLY_TWO_FRESH_SEEDS"
        or p["R91_spent_dev_salt"]!=25319991
        or p["previous_protected_or_spent_scored_source_reused"]
        or p["no_old_training_weights_or_scored_source_reuse"] is not True
        or s["train_salt"]!=TRAIN_SALT or s["dev_test_salt"]!=DEV_SALT
        or s["shuffle_salt"]!=SHUFFLE_SALT or s["batch_permutation_salt"]!=BATCH_SALT
        or s["unit_test_only_salt"]!=UNIT_TEST_SALT or s["seeds"]!=list(SEEDS)
        or s["train_worlds_max"]!=256 or s["eval_world_count"]!=64
        or s["supersedes_preimplementation_freeze_blob"]!="9f51f6c6850a43cf353d1555505c2360fd49071d"
        or not s["chosen_action_rule"].startswith("For seed_index 0 (1000811) / 1 (1000823)")
        or s["balance_contract"]["train_worlds256_each_cell"]!=32
        or s["balance_contract"]["train_worlds16_each_cell"]!=2
        or s["balance_contract"]["dev_test_worlds64_each_cell"]!=8
        or s["balance_contract"]["must_fail_closed_on_any_imbalance"] is not True
        or d["fullwidth_native_params"]!=16818178
        or [(q["width"],q["worlds"],q["exposures_per_distinct_world"]) for q in d["levels"]]!=list(LEVELS)
        or [q["arms"] for q in d["levels"]]!=[list(x) for x in LEVEL_ARMS.values()]
        or t["global_clip_norm"]!=1 or t["head_grad_clip_norm_when_applicable"]!=1
        or t["batch"]!=BATCH or t["cpu_total_resource_cap_seconds"]!=MAX_SECONDS
        or t["learning_rate"]!=.001 or t["weight_decay"]!=.0001
        or v["mandatory_level"]!="w32_n256_e8"
        or v["thresholds"]["headcap_vs_global_DEV_MSE_relative_gain_min"]!=.08
        or v["thresholds"]["headcap_vs_constant_TRAIN_mean_DEV_MSE_relative_gain_min"]!=.05
        or v["thresholds"]["headcap_vs_global_TRAIN_MSE_relative_gain_min"]!=.10
        or v["thresholds"]["headcap_vs_GRU64_DEV_MSE_relative_gain_min"]!=-.10
        or v["thresholds"]["headcap_head_clip_activation_fraction_min"]!=.25
        or v["thresholds"]["headcap_head_grad_reduction_fraction_min"]!=.20
        or v["independent_scientific_pass"]
        or not e["old_R91_R92_DEV_test_experiment_branch_fenced"]
        or not f["authority"]["CPU_test_only_diagnostic_allowed"]
        or any(q for k,q in f["authority"].items() if k!="CPU_test_only_diagnostic_allowed")):
        raise PermissionError("R93_FROZEN_METHOD_SOURCE_OR_AUTHORITY_DRIFT")
    forbidden=set(p["R92_spent_new_test_salts"])|{25319991,25320991,25330111,25330211}
    if set((TRAIN_SALT,DEV_SALT,SHUFFLE_SALT,BATCH_SALT,UNIT_TEST_SALT)) & forbidden:
        raise RuntimeError("R93_OLD_OR_PROTECTED_SALT_OVERLAP")
    if len(set((TRAIN_SALT,DEV_SALT,SHUFFLE_SALT,BATCH_SALT,UNIT_TEST_SALT)))!=5:
        raise RuntimeError("R93_REPEATED_NEW_SALT")
    return f


def _admit()->str:
    frozen()
    if (os.getenv("GITHUB_ACTIONS")!="true"
        or os.getenv("GITHUB_EVENT_NAME")!="pull_request"
        or os.getenv("GITHUB_REPOSITORY")!="spoonman136668-ai/TTT-Hybrid-Lab"
        or os.getenv("GITHUB_HEAD_REF")!=BRANCH
        or os.getenv("GITHUB_JOB")!="cpu-qualification"
        or os.getenv("RUNNER_OS")!="Linux"
        or os.getenv("RUNNER_ENVIRONMENT")!="github-hosted"
        or os.getenv("GITHUB_RUN_ATTEMPT")!="1"):
        raise PermissionError("R93_ONLY_ORIGINAL_CLEAN_HOSTED_LINUX_G4_FIRST_ATTEMPT")
    sha=os.getenv("EXPECTED_CANDIDATE_SHA","")
    head=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
                        capture_output=True,text=True,check=False,timeout=10)
    dirty=subprocess.run(["git","diff","--quiet","HEAD","--"],cwd=ROOT,
                         capture_output=True,text=False,check=False,timeout=10)
    if len(sha)!=40 or head.returncode or head.stdout.strip()!=sha or dirty.returncode:
        raise PermissionError("R93_ORIGINAL_EXACT_SOURCE_SHA_OR_CLEAN_CUSTODY_FAIL")
    return sha


def _chosen_action(seed:int,index:int)->int:
    """Predeclared index-only action independent of early/cold and causal law.

    R85 physical law stratum is (index % 512 // 2) % 2; pivot timing is
    index % 2. Each block of 4 contains all (pivot, law) strata. Alternating
    actions every four worlds avoids R92's action/timing confound.
    """
    if seed not in SEEDS or not isinstance(index,int) or index<0:
        raise PermissionError("R93_UNREGISTERED_BALANCED_ACTION_SEED_OR_INDEX")
    return (r85.CALIBRATE if ((index//4)+SEEDS.index(seed))%2==0
            else r85.EXPLOIT_POS)


def balance_preflight()->dict:
    """Reject action confounding from indices alone, before ANY source first use."""
    result={}
    for seed in SEEDS:
        result[str(seed)]={}
        for label,indices,expected in (
            ("TRAIN_N256",range(256),32),
            ("TRAIN_N16",range(16),2),
            ("DEV_TEST_N64",range(1000,1064),8)):
            counts={}
            for index in indices:
                timing=index%2
                balance_law=((index%r85.N)//2)%2
                chosen=_chosen_action(seed,index)
                key=(timing,balance_law,chosen)
                counts[key]=counts.get(key,0)+1
            if (len(counts)!=8 or any(v!=expected for v in counts.values())
                or set(k[2] for k in counts)!={r85.CALIBRATE,r85.EXPLOIT_POS}):
                raise PermissionError("R93_ACTION_PIVOT_LAW_STRATA_CONFOUNDED")
            result[str(seed)][label]={"strata_count":len(counts),
                                       "each_stratum_count":expected}
    return result


def _world(panel:str,seed:int,index:int,sha:str):
    if panel not in ("TRAIN","DEV_TEST"):
        raise PermissionError("R93_ALL_SPENT_PROTECTED_AND_OLD_PANELS_DENIED")
    if (seed not in SEEDS or not isinstance(index,int)
        or not (0<=index<256 if panel=="TRAIN" else 1000<=index<1064)):
        raise PermissionError("R93_UNREGISTERED_SEED_OR_WORLD_INDEX")
    if (not sha or sha!=os.getenv("EXPECTED_CANDIDATE_SHA")
        or os.getenv("GITHUB_ACTIONS")!="true"
        or os.getenv("GITHUB_HEAD_REF")!=BRANCH
        or os.getenv("GITHUB_JOB")!="cpu-qualification"
        or os.getenv("RUNNER_OS")!="Linux"
        or os.getenv("RUNNER_ENVIRONMENT")!="github-hosted"
        or os.getenv("GITHUB_RUN_ATTEMPT")!="1"):
        raise PermissionError("R93_NEW_PANEL_SOURCE_BEFORE_EXACT_SHA_ADMISSION_DENIED")
    rng=np.random.default_rng(np.random.SeedSequence([
        TRAIN_SALT if panel=="TRAIN" else DEV_SALT,seed,index]))
    world=r85._make_synthetic_world(index % r85.N,rng)
    action=_chosen_action(seed,index)
    return world,action


def unit_test_world(index:int):
    """Never access any new reserved R93 TRAIN/DEV_TEST data in ordinary tests."""
    if not isinstance(index,int) or not 0<=index<4:
        raise PermissionError("R93_NONFROZEN_UNIT_FIXTURE_INDEX")
    rng=np.random.default_rng(np.random.SeedSequence([UNIT_TEST_SALT,index]))
    world=r85._make_synthetic_world(index % r85.N,rng)
    act=r85.CALIBRATE if index%2==0 else r85.EXPLOIT_POS
    return world,act


def _panel(panel,seed,sha):
    if panel not in ("TRAIN","DEV_TEST"):
        raise PermissionError("R93_NO_UNFROZEN_PANEL")
    xs,ys,acts=[],[],[]
    ids=range(256) if panel=="TRAIN" else range(1000,1064)
    for i in ids:
        world,action=_world(panel,seed,i,sha)
        raw,ret=r89._view(world,action)
        slot=r88._public_slots(raw)
        if raw.shape!=(48,13) or raw.dtype!=np.float32 or slot.shape!=(5,13):
            raise RuntimeError("R93_PUBLIC_PHYSICAL_VIEW_GEOMETRY_DRIFT")
        if not np.array_equal(slot[-1],raw[-1]) or slot[-1,action]!=1.0:
            raise RuntimeError("R93_SELECTED_FACTUAL_ACTION_VIEW_CUSTODY_FAIL")
        if not math.isfinite(ret):
            raise RuntimeError("R93_NONFINITE_REALIZED_FACTUAL_RETURN")
        xs.append(slot);ys.append(float(ret));acts.append(action)
    return {"rows":np.stack(xs).astype(np.float32),
            "target":np.asarray(ys,np.float32),
            "action":np.asarray(acts,np.int64),
            "panel":panel,"one_chosen_factual_branch_only":True}


def _shuffle_A(x:np.ndarray,seed:int,tag:int)->np.ndarray:
    if x.ndim!=3 or x.shape[1:]!=(5,13) or len(x)<4 or len(x)%2:
        raise ValueError("R93_SHUFFLE_EXPECTS_5_PUBLIC_EVENT_ROWS_AND_PAIRED_STRATA")
    strata=x[:,-1,12]
    if len(np.unique(strata))!=2:
        raise RuntimeError("R93_EARLY_COLD_PHYSICAL_STRATA_NOT_FOUND")
    index=np.arange(len(x))
    rng=np.random.default_rng(np.random.SeedSequence([SHUFFLE_SALT,seed,tag]))
    for timepoint in np.unique(strata):
        ids=np.flatnonzero(strata==timepoint)
        if len(ids)<2:
            raise RuntimeError("R93_SHUFFLE_STRATUM_TOO_SMALL_FOR_CAUSAL_NEGATIVE")
        proposed=rng.permutation(ids)
        # A random within-stratum permutation can leave the complete 4-row
        # A history unchanged. A causal-missing negative MUST replace every
        # world's A rows, including 2-world unit fixtures.
        if np.any(proposed==ids):
            proposed=np.roll(ids,1)
        index[ids]=proposed
    out=x.copy()
    out[:,:4]=x[index,:4]
    if not np.array_equal(out[:,-1],x[:,-1]):
        raise RuntimeError("R93_SHUFFLE_CHANGED_OWN_LOGGED_ACTION")
    return out


def balanced_batches(seed:int,n:int,exposures:int)->np.ndarray:
    if seed not in SEEDS or (n,exposures) not in {(x[1],x[2]) for x in LEVELS}:
        raise ValueError("R93_OUTSIDE_FROZEN_EXACT_BALANCED_EXPOSURE_DESIGN")
    if n%BATCH:
        raise RuntimeError("R93_NONPAIRED_BATCH_SIZE")
    rng=np.random.default_rng(np.random.SeedSequence([BATCH_SALT,seed,n]))
    flat=np.concatenate([rng.permutation(n) for _ in range(exposures)])
    b=flat.reshape(-1,BATCH)
    hist=np.bincount(b.ravel(),minlength=n)
    if not (hist==exposures).all():
        raise RuntimeError("R93_PER_WORLD_EXACT_FACTUAL_EXPOSURE_MISMATCH")
    return b.astype(np.int64)


def _model(arm,width,seed):
    if arm not in ARMS or width not in (32,1024) or seed not in SEEDS:
        raise PermissionError("R93_UNREGISTERED_NATIVE_OR_CONVENTIONAL_ARM")
    if arm in (GRU,SHUFFLED):
        torch.manual_seed(seed+64)
        return r89.r81.GRU(64)
    torch.manual_seed(seed)
    m=r74.make_model("HybridPilot_FULL",width)
    old=m.ttt
    ttt=r75.TriangularFastWeightMemory(width,learning_rate=old.eta)
    ttt.load_state_dict(old.state_dict(),strict=True)
    m.ttt=ttt
    if width==1024 and sum(p.numel() for p in m.parameters())!=16818178:
        raise RuntimeError("R93_FULLWIDTH_NATIVE_GEOMETRY_CHANGED")
    if arm==NO_DEPTH:
        m.depth=None
    return m


def _value(model,x):
    result=model(x)
    if result.shape!=(len(x),5,1):
        raise RuntimeError("R93_ACTION_VALUE_READOUT_SHAPE_MISMATCH")
    return result[:,-1,0]


def _mse(model,rows,target):
    model.eval()
    errors=[]
    with torch.no_grad():
        for i in range(0,len(rows),16):
            a=_value(model,torch.from_numpy(rows[i:i+16]))
            b=torch.from_numpy(target[i:i+16])
            errors.extend((a-b).square().tolist())
    model.train()
    error=float(np.mean(errors))
    if not math.isfinite(error):
        raise RuntimeError("R93_NONFINITE_FACTUAL_RETURN_MSE")
    return error


def _norm(parameters):
    total=0.
    for p in parameters:
        if p.grad is not None:
            total+=float(p.grad.detach().float().square().sum())
    return math.sqrt(total)


def _train(arm,width,seed,n,exposures,train):
    m=_model(arm,width,seed)
    x=train["rows"][:n]
    if arm==SHUFFLED:
        x=_shuffle_A(x,seed,n*100+exposures)
    y=train["target"][:n]
    indices=balanced_batches(seed,n,exposures)
    initial=_mse(m,x,y)
    optimizer=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    px=torch.from_numpy(x);py=torch.from_numpy(y)
    head_params=list(m.head.parameters()) if arm in NATIVE else []
    other_params=[p for name,p in m.named_parameters() if not name.startswith("head.")] if arm in NATIVE else []
    head_caps=0
    global_caps=0
    head_reduction_sum=0.
    original_head_shares=[]
    first_grads={}
    started=time.perf_counter()
    for step,idx in enumerate(indices):
        optimizer.zero_grad(set_to_none=True)
        pred=_value(m,px[idx])
        loss=F.mse_loss(pred,py[idx])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("R93_NONFINITE_SELECTED_FACTUAL_RETURN_LOSS")
        loss.backward()
        head_before=_norm(head_params) if head_params else 0.
        rest_before=_norm(other_params) if other_params else 0.
        original_total=math.hypot(head_before,rest_before) if head_params else _norm(m.parameters())
        if step==0:
            pp=dict(m.named_parameters())
            for label,key in (("stem","stem.weight"),("head","head.weight"),
                              ("router","depth.router.weight"),("TTT","ttt.value.weight")):
                v=pp.get(key)
                first_grads[label]=(None if v is None or v.grad is None
                                    else float(v.grad.detach().float().norm()))
            first_grads["preclip_head_norm"]=head_before
            first_grads["preclip_nonhead_norm"]=rest_before
            first_grads["preclip_global_norm"]=original_total
        if arm==HEADCAPPED:
            if head_before>1.0:
                head_caps+=1
            nn.utils.clip_grad_norm_(head_params,max_norm=1.0)
            head_after=_norm(head_params)
            head_reduction_sum+=(head_before-head_after)/max(head_before,1e-12)
        if _norm(m.parameters())>1.0:
            global_caps+=1
        nn.utils.clip_grad_norm_(m.parameters(),max_norm=1.0)
        optimizer.step()
    elapsed=time.perf_counter()-started
    final=_mse(m,x,y)
    if not all(math.isfinite(q) for q in (initial,final,elapsed,head_reduction_sum)):
        raise RuntimeError("R93_NONFINITE_NUMERICAL_DIAGNOSTIC")
    states={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    receipt={"arm":arm,"width":width,"train_worlds":n,"actual_per_world_exposures":exposures,
             "updates":len(indices),"batch":BATCH,
             "factual_record_exposures":len(indices)*BATCH,
             "parameters":sum(p.numel() for p in m.parameters()),
             "first_batch_gradient_norms":first_grads,
             "initial_TRAIN_factual_MSE":initial,"final_TRAIN_factual_MSE":final,
             "global_grad_clip_activation_fraction":global_caps/len(indices),
             "head_grad_clip_activation_fraction":head_caps/len(indices) if arm==HEADCAPPED else None,
             "mean_head_grad_reduction_fraction":head_reduction_sum/len(indices) if arm==HEADCAPPED else None,
             "training_seconds":elapsed}
    del optimizer,m
    gc.collect()
    return receipt,states


def _heldout(arm,width,seed,n,exposures,state,dev):
    m=_model(arm,width,seed)
    m.load_state_dict(state,strict=True)
    rows=dev["rows"]
    if arm==SHUFFLED:
        rows=_shuffle_A(rows,seed,100000+n*100+exposures)
    score=_mse(m,rows,dev["target"])
    del m
    gc.collect()
    return score


def _gain(baseline,candidate):
    return float((baseline-candidate)/max(abs(baseline),1e-12))


def development_ci():
    sha=_admit()
    # Fixed index-only action stratification must be established before
    # constructing a single TRAIN or withheld DEV_TEST physical world.
    balanced=balance_preflight()
    torch.set_num_threads(1)
    target=ROOT/"local_runs/r93_headcap_exposure_linux.json"
    if target.exists():
        raise FileExistsError("R93_ORIGINAL_TEST_ONLY_FIRST_USE_REPLAY_DENIED")
    started=time.perf_counter()
    out={"schema":SCHEMA,"frozen_blob":FREEZE_BLOB,"candidate_SHA":sha,
         "pre_first_use_action_balance":balanced,
         "R92_parent_SHA":PARENT,"platform":platform.platform(),
         "torch":str(torch.__version__),"scientific_pass":False,
         "protected_or_previous_scored_source_used":False,
         "all_arms_frozen_before_DEV_TEST":False,
         "DEV_TEST_original_first_use":False,
         "status":"R93_INCOMPLETE_RESOURCE_NOT_SCIENTIFIC_NO_GO",
         "train_arms":{},"frozen_checks":{}}
    states={}
    train_means={}
    try:
        for seed in SEEDS:
            tr=_panel("TRAIN",seed,sha)
            train_means[seed]={n:float(np.mean(tr["target"][:n])) for _,n,_ in LEVELS}
            for i,(width,n,exposures) in enumerate(LEVELS):
                if time.perf_counter()-started>MAX_SECONDS:
                    out["resource_stop_before"]={"seed":seed,"level":i,"width":width,
                                                "worlds":n,"exposures":exposures}
                    break
                level=f"w{width}_n{n}_e{exposures}"
                out["train_arms"].setdefault(level,{})[str(seed)]={}
                for arm in LEVEL_ARMS[i]:
                    r,state=_train(arm,width,seed,n,exposures,tr)
                    out["train_arms"][level][str(seed)][arm]=r
                    states[(seed,level,arm)]=state
            del tr
            if "resource_stop_before" in out:
                break
        if len(states)!=sum(len(LEVEL_ARMS[i]) for i in range(len(LEVELS)))*len(SEEDS):
            out["status"]="R93_INCOMPLETE_RESOURCE_NOT_SCIENTIFIC_NO_GO"
        elif time.perf_counter()-started>MAX_SECONDS:
            out["resource_stop_after_training_seconds"]=time.perf_counter()-started
            out["status"]="R93_INCOMPLETE_RESOURCE_NOT_SCIENTIFIC_NO_GO"
        else:
            out["all_arms_frozen_before_DEV_TEST"]=True
            out["DEV_TEST_original_first_use"]=True
            # NO train, optimizer, weight or hyperparameter updates after this.
            for seed in SEEDS:
                dv=_panel("DEV_TEST",seed,sha)
                out.setdefault("DEV_TEST",{})[str(seed)]={}
                for width,n,e in LEVELS:
                    level=f"w{width}_n{n}_e{e}"
                    out["DEV_TEST"][str(seed)][level]={
                        "TRAIN_only_constant_DEV_MSE":float(np.mean(
                            (dv["target"]-train_means[seed][n])**2))}
                    for arm in out["train_arms"][level][str(seed)]:
                        state=states.pop((seed,level,arm))
                        out["train_arms"][level][str(seed)][arm]["heldout_DEV_TEST_factual_MSE"]=(
                            _heldout(arm,width,seed,n,e,state,dv))
                del dv
            t=frozen()["frozen_evaluation"]["thresholds"]
            for seed in SEEDS:
                m=out["train_arms"]["w32_n256_e8"][str(seed)]
                original=m[HARD];cap=m[HEADCAPPED];control=m[GRU]
                constant=out["DEV_TEST"][str(seed)]["w32_n256_e8"]["TRAIN_only_constant_DEV_MSE"]
                scores={
                    "headcap_vs_global_DEV_MSE_relative_gain":
                      _gain(original["heldout_DEV_TEST_factual_MSE"],cap["heldout_DEV_TEST_factual_MSE"]),
                    "headcap_vs_trainconstant_DEV_MSE_relative_gain":
                      _gain(constant,cap["heldout_DEV_TEST_factual_MSE"]),
                    "headcap_vs_global_TRAIN_MSE_relative_gain":
                      _gain(original["final_TRAIN_factual_MSE"],cap["final_TRAIN_factual_MSE"]),
                    "headcap_vs_GRU_DEV_MSE_relative_gain":
                      _gain(control["heldout_DEV_TEST_factual_MSE"],cap["heldout_DEV_TEST_factual_MSE"]),
                    "headcap_activation_fraction":cap["head_grad_clip_activation_fraction"],
                    "head_grad_reduction_fraction":cap["mean_head_grad_reduction_fraction"]}
                checks={
                    "improved_over_global_DEV":scores["headcap_vs_global_DEV_MSE_relative_gain"]>=t["headcap_vs_global_DEV_MSE_relative_gain_min"],
                    "beat_constant_on_unseen":scores["headcap_vs_trainconstant_DEV_MSE_relative_gain"]>=t["headcap_vs_constant_TRAIN_mean_DEV_MSE_relative_gain_min"],
                    "improved_global_TRAIN":scores["headcap_vs_global_TRAIN_MSE_relative_gain"]>=t["headcap_vs_global_TRAIN_MSE_relative_gain_min"],
                    "competitive_with_matched_exposure_GRU":scores["headcap_vs_GRU_DEV_MSE_relative_gain"]>=t["headcap_vs_GRU64_DEV_MSE_relative_gain_min"],
                    "headclip_actually_used":scores["headcap_activation_fraction"]>=t["headcap_head_clip_activation_fraction_min"],
                    "headclip_magnitude_reduced":scores["head_grad_reduction_fraction"]>=t["headcap_head_grad_reduction_fraction_min"]}
                out["frozen_checks"][str(seed)]={"scores":scores,
                    "checks":{k:bool(v) for k,v in checks.items()}}
            out["status"]=("R93_HEAD_GRAD_CONDITIONING_DEV_SIGNAL_ONLY"
                           if all(all(v["checks"].values()) for v in out["frozen_checks"].values())
                           else "R93_EXPOSURE_HEADCAP_DEV_NO_GO")
    except BaseException as exc:
        out["error_type"]=type(exc).__name__
        out["status"]="R93_INCOMPLETE_RESOURCE_NOT_SCIENTIFIC_NO_GO"
        raise
    finally:
        out["total_wall_seconds"]=time.perf_counter()-started
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open("x",encoding="utf-8") as f:
            json.dump(out,f,indent=2,sort_keys=True,allow_nan=False)
            f.write("\n")
    return out
