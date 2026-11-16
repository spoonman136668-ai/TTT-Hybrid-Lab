"""R92: frozen high-first native diversity/LR interference diagnosis on TEST-only worlds.

No protected R85-R91 scored source, weights, synthetic old salts, role-labeled
targets, paired oracle or GPU. New factual selected-return TRAIN and DEV_TEST
panels use independent immutable salt and are admitted only for original
branch+SHA+attempt-1 GitHub-hosted Linux G4. All arms train before DEV_TEST
is first instantiated; no earlier R91 DEV_TEST scoring is permitted.
This tests training and heldout factual-return MSE, NOT action policy learning.
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

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import r74_capability_ceiling as r74
from experiments import r75_triangular_fastweight as r75
from experiments import r85_delayed_composition as r85
from experiments import r88_causal_event_access as r88
from experiments import r89_native_event_access as r89
from experiments import r90_native_credit_path_diagnostic as r90
from experiments import r91_native_soft_routing_dev as r91

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r92_native_diversity_lr_freeze.json"
FREEZE_BLOB="149ea5161cd26fef343a60e0b6c93d01b91e620a"
SCHEMA="ttt-hybrid.r92-test-only-diversity-lr-diagnostic.v1"
PARENT="d3bc7eb068ffe14b37adde556765f97f477e7f5c"
BRANCH="research/r92-native-diversity-learning-rate-diagnostic-20261115"
SEEDS=(1000611,1000623)
TRAIN_SALT=25330111
DEV_SALT=25330211
SAMPLE_SALT=25330311
SHUFFLE_SALT=25330411
ARMS=("NATIVE_HARD_LR_1E3","NATIVE_HARD_LR_1E4",
      "NATIVE_SOFT_LR_1E4","GRU64_SLOTS5","GRU64_SHUFFLED_A_SLOTS5")
NATIVE=ARMS[:3]
GRUS=ARMS[3:]
LEVELS=((1024,256),(1024,16),(32,256),(32,64),(32,16),(32,4))
LEARNING_RATE={"NATIVE_HARD_LR_1E3":.001,
               "NATIVE_HARD_LR_1E4":.0001,
               "NATIVE_SOFT_LR_1E4":.0001,
               "GRU64_SLOTS5":.001,"GRU64_SHUFFLED_A_SLOTS5":.001}
STEPS={a:(96 if a in NATIVE else 768) for a in ARMS}
BATCH=2
MAX_SECONDS=360


def frozen() -> dict:
    if r74.git_blob(FREEZE.read_bytes())!=FREEZE_BLOB:
        raise RuntimeError("R92_ORIGINAL_FROZEN_SOURCE_BLOB_DRIFT")
    r91.frozen()
    f=json.loads(FREEZE.read_text(encoding="utf-8"))
    p,s,d,t,a,e=f["parent"],f["source"],f["design"],f["training"],f["analysis"],f["execution"]
    if (f["schema"]!=SCHEMA
        or f["stage"]!="IMMUTABLE_PROSPECTIVE_FREEZE_BEFORE_R92_IMPLEMENTATION_REGRESSION_AND_TEST_FIRST_USE"
        or p["R91_sha"]!=PARENT
        or p["R91_freeze_blob"]!=r91.FREEZE_BLOB
        or p["R91_original_status"]!="R91_SOFT_ROUTING_DEV_NO_GO_BOTH_OS"
        or p["R91_test_only_used_salt"]!=25319991
        or p["previous_protected_or_spent_scored_source_reused"]
        or p["previous_model_weights_reused"]
        or s["train_salt"]!=TRAIN_SALT
        or s["test_salt"]!=DEV_SALT
        or s["sampling_salt"]!=SAMPLE_SALT
        or s["shuffle_salt"]!=SHUFFLE_SALT
        or s["seeds"]!=list(SEEDS)
        or s["train_worlds_max"]!=256
        or s["dev_test_worlds"]!=64
        or d["world_counts_by_width"]!={"1024":[256,16],"32":[256,64,16,4]}
        or d["model_arms"]!=list(ARMS)
        or t["native_updates_per_arm"]!=96
        or t["GRU_updates_per_arm"]!=768
        or t["batch"]!=BATCH
        or t["learning_rates"]!=LEARNING_RATE
        or t["resource_cap_entire_R92_diagnostic_seconds"]!=MAX_SECONDS
        or t["qualification_budget_existing_G4_pytest_seconds"]!=600
        or a["frozen_thresholds"]["width1024_world256_lowLR_hard_vs_highLR_hard_DEV_factual_MSE_relative_gain_min"]!=.05
        or a["frozen_thresholds"]["width1024_world256_lowLR_hard_vs_constant_DEV_factual_MSE_relative_gain_min"]!=.05
        or a["frozen_thresholds"]["width1024_world256_lowLR_hard_train_MSE_relative_gain_min"]!=.10
        or a["frozen_thresholds"]["width1024_world256_softLR_vs_hardLR_DEV_factual_MSE_relative_gain_min"]!=.05
        or a["frozen_thresholds"]["independent_scientific_pass"]
        or not e["R91_original_DEV_TEST_replay_denied_outside_original_R91_branch"]
        or any(v for k,v in f["authority"].items()
               if k!="CPU_test_only_development_allowed")
        or not f["authority"]["CPU_test_only_development_allowed"]):
        raise PermissionError("R92_FROZEN_METHOD_OR_AUTHORITY_DRIFT")
    forbidden={25319991,25320991,25329991,25290111,25300111,25310111}
    if len({TRAIN_SALT,DEV_SALT,SAMPLE_SALT,SHUFFLE_SALT})!=4 or set(
        (TRAIN_SALT,DEV_SALT,SAMPLE_SALT,SHUFFLE_SALT)) & forbidden:
        raise RuntimeError("R92_SPENT_OR_REPEATED_SOURCE_SALT")
    return f


def _admit()->str:
    frozen()
    if (os.getenv("GITHUB_ACTIONS")!="true"
        or os.getenv("GITHUB_EVENT_NAME")!="pull_request"
        or os.getenv("GITHUB_REPOSITORY")!="spoonman136668-ai/TTT-Hybrid-Lab"
        or os.getenv("GITHUB_HEAD_REF")!=BRANCH
        or os.getenv("GITHUB_JOB")!="cpu-qualification"
        or os.getenv("RUNNER_OS")!="Linux"
        or os.getenv("GITHUB_RUN_ATTEMPT")!="1"):
        raise PermissionError("R92_ORIGINAL_EXACT_HEAD_LINUX_G4_FIRST_ATTEMPT_ONLY")
    sha=os.getenv("EXPECTED_CANDIDATE_SHA","")
    observed=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
                            capture_output=True,text=True,check=False,timeout=10)
    dirty=subprocess.run(["git","diff","--quiet","HEAD","--"],cwd=ROOT,
                         capture_output=True,check=False,timeout=10)
    if len(sha)!=40 or observed.returncode or observed.stdout.strip()!=sha or dirty.returncode:
        raise PermissionError("R92_EXACT_SHA_CLEAN_SOURCE_CUSTODY_FAILURE")
    return sha


def _test_only_physics_for_unit_tests(index:int):
    """Test negative without admitting R92 frozen TRAIN or DEV_TEST cohorts."""
    if not 0<=index<4:
        raise PermissionError("R92_NONFROZEN_NONSCIENTIFIC_UNIT_TEST_WORLD")
    rng=np.random.default_rng(np.random.SeedSequence([25339999,index]))
    world=r85._make_synthetic_world(index%r85.N,rng)
    action=r85.CALIBRATE if index%2==0 else r85.EXPLOIT_POS
    return world,action


def _world(panel:str,seed:int,index:int,sha:str):
    if panel not in ("TRAIN","DEV_TEST"):
        raise PermissionError("R92_PROTECTED_OR_PRIOR_TEST_SOURCE_DENIED")
    allowed=((0<=index<256) if panel=="TRAIN" else (1000<=index<1064))
    if seed not in SEEDS or not isinstance(index,int) or not allowed:
        raise PermissionError("R92_NONFROZEN_NEW_TEST_INDEX_OR_SEED")
    if (not sha or sha!=os.getenv("EXPECTED_CANDIDATE_SHA")
        or os.getenv("GITHUB_ACTIONS")!="true"
        or os.getenv("GITHUB_HEAD_REF")!=BRANCH
        or os.getenv("GITHUB_JOB")!="cpu-qualification"
        or os.getenv("RUNNER_OS")!="Linux"
        or os.getenv("GITHUB_RUN_ATTEMPT")!="1"):
        raise PermissionError("R92_ORIGINAL_NEW_TEST_SOURCE_NOT_ADMITTED")
    rng=np.random.default_rng(np.random.SeedSequence([
        TRAIN_SALT if panel=="TRAIN" else DEV_SALT, seed, index]))
    world=r85._make_synthetic_world(index%r85.N,rng)
    action=r85.CALIBRATE if (seed+index)%2==0 else r85.EXPLOIT_POS
    return world,action


def _panel(panel:str,seed:int,sha:str)->dict:
    indices=range(256) if panel=="TRAIN" else range(1000,1064)
    public=[]
    targets=[]
    selected=[]
    for i in indices:
        world,action=_world(panel,seed,i,sha)
        raw,return_value=r89._view(world,action)
        if raw.dtype!=np.float32 or raw.shape!=(48,13):
            raise RuntimeError("R92_NONLEGAL_PUBLIC_48_ROW_SEQUENCE")
        rows=r88._public_slots(raw)
        if not np.array_equal(rows[-1],raw[-1]) or rows[-1,action]!=1:
            raise RuntimeError("R92_SELECTED_ACTION_OR_PRIOR_A_EVENTS_ILLEGAL")
        if not math.isfinite(return_value):
            raise RuntimeError("R92_NONFINITE_PHYSICALLY_REALIZED_FACTUAL_REWARD")
        public.append(rows)
        targets.append(float(return_value))
        selected.append(action)
    return {"slots":np.stack(public).astype(np.float32),
            "target":np.asarray(targets,np.float32),
            "chosen":np.asarray(selected,np.int64),
            "one_true_observed_branch_only":True,
            "panel":panel}


def _shuffle(rows:np.ndarray,seed:int,tag:int)->np.ndarray:
    if rows.ndim!=3 or rows.shape[1:]!=(5,13) or len(rows)<4 or len(rows)%2:
        raise ValueError("R92_SHUFFLE_REQUIRES_PUBLIC_EVENT_INPUT")
    strata=rows[:,-1,12]
    if len(np.unique(strata))!=2:
        raise RuntimeError("R92_EARLY_COLD_STRATUM_MISSING")
    permutation=np.arange(len(rows))
    rng=np.random.default_rng(np.random.SeedSequence([SHUFFLE_SALT,seed,tag]))
    for timepoint in np.unique(strata):
        ids=np.flatnonzero(strata==timepoint)
        permutation[ids]=rng.permutation(ids)
    shuffled=rows.copy()
    shuffled[:,:4]=rows[permutation,:4]
    if not np.array_equal(shuffled[:,-1],rows[:,-1]):
        raise RuntimeError("R92_NEGATIVE_SHUFFLE_CHANGED_OWN_FACTUAL_ACTION")
    return shuffled


def _model(arm:str,width:int,seed:int)->nn.Module:
    if arm not in ARMS or width not in (1024,32) or seed not in SEEDS:
        raise PermissionError("R92_UNREGISTERED_MODEL_ARCH_WIDTH_OR_SEED")
    torch.manual_seed(seed)
    if arm in GRUS:
        torch.manual_seed(seed+64)
        return r89.r81.GRU(64)
    m=r74.make_model("HybridPilot_FULL",width)
    old=m.ttt
    new=r75.TriangularFastWeightMemory(width,learning_rate=old.eta)
    new.load_state_dict(old.state_dict(),strict=True)
    m.ttt=new
    if arm==NATIVE[2]:
        m.depth=r90.SoftDiagnosticRouter(m.depth)
    if width==1024 and sum(p.numel() for p in m.parameters())!=16818178:
        raise RuntimeError("R92_ORIGINAL_NATIVE_PARAMETER_COUNT_CHANGED")
    return m


def _value(m:nn.Module,x:torch.Tensor)->torch.Tensor:
    y=m(x)
    if y.ndim!=3 or y.shape[1:]!=(5,1):
        raise RuntimeError("R92_NATIVE_OR_CONVENTIONAL_READOUT_SHAPE_CHANGED")
    return y[:,-1,0]


def _estimate_mse(m:nn.Module,features:np.ndarray,target:np.ndarray)->float:
    m.eval()
    losses=[]
    with torch.no_grad():
        for i in range(0,len(target),16):
            x=torch.from_numpy(features[i:i+16])
            y=torch.from_numpy(target[i:i+16])
            losses.extend((_value(m,x)-y).square().tolist())
    m.train()
    result=float(np.mean(losses))
    if not math.isfinite(result):
        raise RuntimeError("R92_NONFINITE_OBSERVED_RETURN_ERROR")
    return result


def _train(arm:str,width:int,seed:int,n:int,train:dict):
    m=_model(arm,width,seed)
    source=train["slots"][:n]
    if arm==GRUS[1]:
        source=_shuffle(source,seed,n)
    target=train["target"][:n]
    original_mse=_estimate_mse(m,source,target)
    opt=torch.optim.AdamW(m.parameters(),lr=LEARNING_RATE[arm],weight_decay=.0001)
    sampling=np.random.default_rng(np.random.SeedSequence([SAMPLE_SALT,seed,n]))
    batches=sampling.integers(0,n,size=(max(STEPS.values()),BATCH))
    x=torch.from_numpy(source)
    y=torch.from_numpy(target)
    grads={}
    started=time.perf_counter()
    for update in range(1,STEPS[arm]+1):
        ix=batches[update-1]
        p=_value(m,x[ix])
        loss=F.mse_loss(p,y[ix])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("R92_NONFINITE_FACTUAL_GRADIENT_LOSS")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if update==1:
            names=dict(m.named_parameters())
            for label,key in (("stem","stem.weight"),("readout","head.weight"),
                              ("router","depth.router.weight"),
                              ("ttt_value","ttt.value.weight")):
                grad=names.get(key)
                grads[label]=(None if grad is None or grad.grad is None
                              else float(grad.grad.detach().norm()))
            if any(q is not None and not math.isfinite(q) for q in grads.values()):
                raise RuntimeError("R92_NONFINITE_GROUP_GRADIENT")
        nn.utils.clip_grad_norm_(m.parameters(),1.)
        opt.step()
    final_mse=_estimate_mse(m,source,target)
    elapsed=time.perf_counter()-started
    state={k:p.detach().cpu().clone() for k,p in m.state_dict().items()}
    result={"steps":STEPS[arm],"parameters":sum(p.numel() for p in m.parameters()),
            "chosen_event_exposures":STEPS[arm]*BATCH,
            "initial_factual_TRAIN_MSE":original_mse,
            "final_factual_TRAIN_MSE":final_mse,
            "first_batch_gradient_norm":grads,
            "train_wall_seconds":elapsed,
            "optimizer_lr":LEARNING_RATE[arm],
            "world_count":n,"native_width":width}
    del opt,m
    gc.collect()
    return result,state


def _eval(arm,width,seed,n,state,dev:dict)->float:
    m=_model(arm,width,seed)
    m.load_state_dict(state,strict=True)
    x=dev["slots"]
    if arm==GRUS[1]:
        x=_shuffle(x,seed,10000+n)
    score=_estimate_mse(m,x,dev["target"])
    del m
    gc.collect()
    return score


def _relative_gain(baseline:float,candidate:float)->float:
    return float((baseline-candidate)/max(abs(baseline),1e-12))


def development_ci()->dict:
    """Single original G4 first attempt; source never opened in normal unit tests."""
    sha=_admit()
    torch.set_num_threads(1)
    target=ROOT/"local_runs/r92_native_diversity_lr_linux.json"
    if target.exists():
        raise FileExistsError("R92_ORIGINAL_DEVELOPMENT_FIRST_USE_REPLAY_DENIED")
    began=time.perf_counter()
    result={"schema":SCHEMA,"freeze_blob":FREEZE_BLOB,"parent_R91_SHA":PARENT,
            "candidate_SHA":sha,"scientific_pass":False,
            "protected_or_previous_DEV_OOD_access":False,
            "DEV_TEST_FIRST_USE_AFTER_ALL_TRAINING":False,
            "platform":platform.platform(),"torch":str(torch.__version__),
            "status":"R92_INCOMPLETE_RESOURCE_NOT_SCIENTIFIC_NO_GO",
            "completed":{},"comparison_checks":{}}
    saved={}
    try:
        # Strict high-first: 1024x256 before narrower/fewer-world diagnostics.
        for width,n in LEVELS:
            if time.perf_counter()-began>MAX_SECONDS:
                result["resource_stop_before_level"]={"width":width,"worlds":n}
                break
            key=f"w{width}_n{n}"
            result["completed"][key]={}
            for seed in SEEDS:
                if (seed,"TRAIN") not in saved:
                    saved[(seed,"TRAIN")]=_panel("TRAIN",seed,sha)
                train=saved[(seed,"TRAIN")]
                result["completed"][key][str(seed)]={}
                for arm in ARMS:
                    metrics,state=_train(arm,width,seed,n,train)
                    result["completed"][key][str(seed)][arm]=metrics
                    saved[(seed,width,n,arm)]=state
        # Drop TRAIN panel before any new DEV_TEST is constructed.
        for seed in SEEDS:
            saved.pop((seed,"TRAIN"),None)
        if len(result["completed"])!=len(LEVELS):
            result["status"]="R92_INCOMPLETE_RESOURCE_NOT_SCIENTIFIC_NO_GO"
        else:
            result["all_weights_frozen_before_DEV_TEST"]=True
            # TRAINING IS OVER; never update any weights below this boundary.
            result["DEV_TEST_FIRST_USE_AFTER_ALL_TRAINING"]=True
            for seed in SEEDS:
                dev=_panel("DEV_TEST",seed,sha)
                baseline=float(np.mean((dev["target"]-dev["target"].mean())**2))
                # Constant predictor trained on TRAIN labels only, not DEV mean.
                train=_panel("TRAIN",seed,sha)
                true_constant=float(np.mean((dev["target"]-train["target"].mean())**2))
                result.setdefault("DEV_TEST",{})[str(seed)]={"constant_TRAIN_mean_MSE":true_constant,
                    "DEV_TEST_variance_DIAGNOSTIC_ONLY":baseline,"n":len(dev["target"])}
                for width,n in LEVELS:
                    key=f"w{width}_n{n}"
                    metrics=result["completed"][key][str(seed)]
                    for arm in ARMS:
                        state=saved.pop((seed,width,n,arm))
                        metrics[arm]["heldout_factual_DEV_TEST_MSE"]=_eval(arm,width,seed,n,state,dev)
            t=frozen()["analysis"]["frozen_thresholds"]
            for seed in SEEDS:
                v=result["completed"]["w1024_n256"][str(seed)]
                hi=v[ARMS[0]]
                lo=v[ARMS[1]]
                soft=v[ARMS[2]]
                constant=result["DEV_TEST"][str(seed)]["constant_TRAIN_mean_MSE"]
                comparisons={
                    "lowLR_hard_vs_highLR_hard_DEV_MSE_rel_gain":
                        _relative_gain(hi["heldout_factual_DEV_TEST_MSE"],lo["heldout_factual_DEV_TEST_MSE"]),
                    "lowLR_hard_vs_train_constant_DEV_MSE_rel_gain":
                        _relative_gain(constant,lo["heldout_factual_DEV_TEST_MSE"]),
                    "lowLR_hard_vs_highLR_hard_TRAIN_MSE_rel_gain":
                        _relative_gain(hi["final_factual_TRAIN_MSE"],lo["final_factual_TRAIN_MSE"]),
                    "lowLR_soft_vs_lowLR_hard_DEV_MSE_rel_gain":
                        _relative_gain(lo["heldout_factual_DEV_TEST_MSE"],soft["heldout_factual_DEV_TEST_MSE"])}
                checks={
                    "lowLR_vs_highLR_DEV_improves":
                        comparisons["lowLR_hard_vs_highLR_hard_DEV_MSE_rel_gain"]>=t["width1024_world256_lowLR_hard_vs_highLR_hard_DEV_factual_MSE_relative_gain_min"],
                    "lowLR_vs_constant_DEV_improves":
                        comparisons["lowLR_hard_vs_train_constant_DEV_MSE_rel_gain"]>=t["width1024_world256_lowLR_hard_vs_constant_DEV_factual_MSE_relative_gain_min"],
                    "lowLR_vs_highLR_TRAIN_improves":
                        comparisons["lowLR_hard_vs_highLR_hard_TRAIN_MSE_rel_gain"]>=t["width1024_world256_lowLR_hard_train_MSE_relative_gain_min"],
                    "soft_lowLR_vs_hard_lowLR_DEV_improves":
                        comparisons["lowLR_soft_vs_lowLR_hard_DEV_MSE_rel_gain"]>=t["width1024_world256_softLR_vs_hardLR_DEV_factual_MSE_relative_gain_min"]}
                result["comparison_checks"][str(seed)]={"measurements":comparisons,"checks":checks}
            result["status"]=("R92_NATIVE_DIVERSITY_LR_DEV_SIGNAL_ONLY" if
                all(all(v["checks"].values()) for v in result["comparison_checks"].values())
                else "R92_NATIVE_DIVERSITY_LR_NO_GO_DIAGNOSTIC_ONLY")
    except BaseException as exc:
        result["error_type"]=type(exc).__name__
        result["status"]="R92_INCOMPLETE_RESOURCE_NOT_SCIENTIFIC_NO_GO"
        raise
    finally:
        result["whole_program_wall_seconds"]=time.perf_counter()-began
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open("x",encoding="utf-8") as p:
            json.dump(result,p,indent=2,sort_keys=True,allow_nan=False)
            p.write("\n")
    return result
