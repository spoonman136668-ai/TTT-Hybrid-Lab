"""R95: fresh raw-observation trainable conjunction versus native and conventional controls.

TEST-only: no R85-R94 previous scored/reserved source, weights, private law
teachers or unchosen action labels. All arms trained on legal factual returns.
Heldout DEV and known-law shifted OOD are first realized only after all 8 arms
complete and freeze. A pair-tree is an engineered structural prior, not evidence
of autonomous invention or learning of ordered roles in a commutative parity law.
"""
from __future__ import annotations
import copy
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
from experiments import r81_paired_history as r81
from experiments import r85_delayed_composition as r85
from experiments import r86_native_compositional_value as r86
from experiments import r87_public_composition_diagnostic as r87
from experiments import r88_causal_event_access as r88
from experiments import r94_causal_signal_diagnostic as r94

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r95_learned_relation_freeze.json"
FREEZE_BLOB="e2968738694b67b2557d7c976378b9664dec0860"
PARENT="bc93585f62cb12bd6b04905ab0fa3d79c84372c9"
SCHEMA="ttt-hybrid.r95-raw-causal-conjunction-learning.testonly.v1"
BRANCH="research/r95-learned-causal-conjunction-20261115"
# Original independently verified two-seed TEST-only first use is spent.
ORIGINAL_R95_SPENT_SHA="bc80edd63101cb8f293769a1e6345689acca0705"
ORIGINAL_R95_SPENT_RUN_ID="43792098794"
SEEDS=(1001311,1001323)
SALT={"FIT":25360111,"FIT_VALIDATION":25360211,
      "DEV":25360311,"OOD_SHIFT":25360411}
COUNT={"FIT":2048,"FIT_VALIDATION":256,"DEV":256,"OOD_SHIFT":256}
SHUFFLE_SALT=25360511
BATCH_SALT=25360611
BOOTSTRAP_SALT=25360711
UNIT_SALT=25369991
ARMS=("NATIVE32_RAW_SLOTS5","NATIVE32_FLAT_SIDECAR",
      "NATIVE32_TREE_SIDECAR","MLP64_RAW_CUES","MLP_TREE_RAW_CUES",
      "MLP64_PUBLIC_PARITY_UPPER","MLP_TREE_SHUFFLED_A_NEGATIVE",
      "GRU64_RAW_SLOTS5")
NATIVE_RAW,NATIVE_FLAT,NATIVE_TREE,MLP_RAW,MLP_TREE,UPPER,NEGATIVE,GRU=ARMS
NATIVE=ARMS[:3]
STEPS={a:(1024 if a in NATIVE else 4096) for a in ARMS}
BATCH=16
MAX_SECONDS=240


def frozen()->dict:
    if r85.git_blob(FREEZE.read_bytes())!=FREEZE_BLOB:
        raise RuntimeError("R95_ORIGINAL_PROSPECTIVE_FREEZE_BLOB_DRIFT")
    r94.frozen()
    f=json.loads(FREEZE.read_text(encoding="utf-8"))
    p,s,a,t,e,x=f["parent"],f["source"],f["architecture"],f["training"],f["evaluation"],f["execution"]
    if (f["schema"]!=SCHEMA
        or f["stage"]!="PROSPECTIVE_FROZEN_BEFORE_R95_IMPLEMENTATION_TESTS_CI_OR_NEW_SOURCE_FIRST_USE"
        or p["R94_final_source_sha"]!=PARENT
        or p["R94_freeze_blob"]!=r94.FREEZE_BLOB
        or p["R94_original_result"]!="R94_PUBLIC_CAUSAL_FEATURE_SIGNAL_ONLY"
        or p["previous_scored_training_validation_development_OOD_or_weights_reused"]
        or s["salts"]!={**SALT,"SHUFFLE":SHUFFLE_SALT,"BATCHES":BATCH_SALT,
                         "BOOTSTRAP":BOOTSTRAP_SALT,"UNIT_ONLY":UNIT_SALT}
        or s["seeds"]!=list(SEEDS) or s["world_counts"]!=COUNT
        or s["FIT_worlds_per_joint_8_cell"]!=256
        or s["other_panels_worlds_per_joint_8_cell"]!=32
        or f["arms"]!=list(ARMS) or t["per_arm_steps"]!=STEPS
        or t["batch"]!=BATCH or t["whole_program_CPU_budget_seconds_per_seed"]!=MAX_SECONDS
        or a["full_native_width1024_not_claimed"] is not True
        or a["isolated_experimental_additive_sidecar_only"] is not True
        or a["no_structural_change_to_main_native"] is not True
        or e["bootstrap_resamples"]!=2000
        or e["conventional_checks"]["MLP_tree_minus_raw_DEV_reward_min"]!=.20
        or e["conventional_checks"]["MLP_tree_minus_shuffled_DEV_reward_min"]!=.75
        or e["conventional_checks"]["MLP_tree_minus_shuffled_OOD_cold_reward_min"]!=.50
        or e["conventional_checks"]["MLP_tree_vs_explicit_public_parity_DEV_reward_gap_max"]!=.35
        or e["conventional_checks"]["MLP_tree_legal_DEV_action_agreement_min"]!=.90
        or e["conventional_checks"]["MLP_tree_minus_shuffled_DEV_paired_bootstrap_lower95_min"]!=.10
        or e["native_sidecar_checks"]["native_tree_minus_native_flat_DEV_reward_min"]!=.20
        or e["native_sidecar_checks"]["native_tree_minus_native_raw_DEV_reward_min"]!=.75
        or e["native_sidecar_checks"]["native_tree_minus_native_flat_OOD_cold_reward_min"]!=.20
        or e["native_sidecar_checks"]["native_tree_vs_conventional_MLP_tree_DEV_reward_gap_max"]!=.30
        or e["independent_scientific_pass"]
        or not x["two_new_seed_jobs_in_existing_G4_only"]
        or x["no_new_workflow_file"] is not True
        or not f["authority"]["scoped_cpu_testonly_diagnostic_allowed"]
        or any(bool(z) for k,z in f["authority"].items()
               if k!="scoped_cpu_testonly_diagnostic_allowed")):
        raise PermissionError("R95_FROZEN_PROTOCOL_OR_AUTHORITY_CHANGED")
    new={*SALT.values(),SHUFFLE_SALT,BATCH_SALT,BOOTSTRAP_SALT,UNIT_SALT}
    old=set(r94.frozen()["sources"]["salts"].values())
    if len(new)!=8 or new & old:
        raise RuntimeError("R95_PREVIOUS_SOURCE_SALT_COLLISION")
    return f


def _chosen(seed:int,index:int)->int:
    if seed not in SEEDS or not isinstance(index,int) or index<0:
        raise PermissionError("R95_OUTSIDE_FROZEN_SELECTED_ACTION")
    return r85.CALIBRATE if (index//4+SEEDS.index(seed))%2==0 else r85.EXPLOIT_POS


def balance_preflight():
    out={}
    for seed in SEEDS:
        out[str(seed)]={}
        for panel,n in COUNT.items():
            buckets={}
            for index in range(n):
                cell=(index%2,((index%r85.N)//2)%2,_chosen(seed,index))
                buckets[cell]=buckets.get(cell,0)+1
            if len(buckets)!=8 or set(buckets.values())!={n//8}:
                raise RuntimeError("R95_SELECTED_ACTION_PIVOT_LAW_INDEX_CONFOUND")
            out[str(seed)][panel]={"joint_cells":8,"worlds_per_cell":n//8}
    return out


def _admit(seed:int)->str:
    frozen()
    if (seed not in SEEDS or os.getenv("GITHUB_ACTIONS")!="true"
        or os.getenv("GITHUB_EVENT_NAME")!="pull_request"
        or os.getenv("GITHUB_REPOSITORY")!="spoonman136668-ai/TTT-Hybrid-Lab"
        or os.getenv("GITHUB_HEAD_REF")!=BRANCH
        or os.getenv("GITHUB_JOB")!="r95-learned-relation"
        or os.getenv("R95_SEED")!=str(seed)
        or os.getenv("RUNNER_OS")!="Linux"
        or os.getenv("RUNNER_ENVIRONMENT")!="github-hosted"
        or os.getenv("GITHUB_RUN_ATTEMPT")!="1"
        or "/merge" not in os.getenv("GITHUB_REF","")):
        raise PermissionError("R95_ORIGINAL_HOSTED_FIRST_ATTEMPT_G4_JOB_REQUIRED")
    sha=os.getenv("EXPECTED_CANDIDATE_SHA","")
    if sha!=ORIGINAL_R95_SPENT_SHA or os.getenv("GITHUB_RUN_ID")!=ORIGINAL_R95_SPENT_RUN_ID:
        raise PermissionError("R95_SPENT_TEST_ONLY_FIRST_USE_REPLAY_DENIED")
    actual=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
        capture_output=True,text=True,check=False,timeout=10)
    clean=subprocess.run(["git","diff","--quiet","HEAD","--"],cwd=ROOT,
        capture_output=True,text=True,check=False,timeout=10)
    if len(sha)!=40 or actual.returncode or actual.stdout.strip()!=sha or clean.returncode:
        raise PermissionError("R95_NOT_CLEAN_EXACT_ORIGINAL_SOURCE_SHA")
    return sha


def unit_world(index:int):
    if not isinstance(index,int) or not 0<=index<4:
        raise PermissionError("R95_ONLY_UNIT_NONSCORING_SOURCE")
    rng=np.random.default_rng(np.random.SeedSequence([UNIT_SALT,index]))
    return r85._make_synthetic_world(index,rng),r85.CALIBRATE if index%2==0 else r85.EXPLOIT_POS


def _world(panel:str,seed:int,index:int,sha:str):
    if panel not in SALT:
        raise PermissionError("R95_PREVIOUS_OR_PROTECTED_PANEL_DENIED")
    if seed not in SEEDS or not isinstance(index,int) or not 0<=index<COUNT[panel]:
        raise PermissionError("R95_NONFROZEN_SEED_OR_PANEL_INDEX")
    if (not sha or sha!=os.getenv("EXPECTED_CANDIDATE_SHA")
        or os.getenv("GITHUB_ACTIONS")!="true"
        or os.getenv("GITHUB_HEAD_REF")!=BRANCH
        or os.getenv("GITHUB_JOB")!="r95-learned-relation"
        or os.getenv("R95_SEED")!=str(seed)
        or os.getenv("GITHUB_RUN_ATTEMPT")!="1"
        or sha!=ORIGINAL_R95_SPENT_SHA
        or os.getenv("GITHUB_RUN_ID")!=ORIGINAL_R95_SPENT_RUN_ID
        or os.getenv("RUNNER_OS")!="Linux"
        or os.getenv("RUNNER_ENVIRONMENT")!="github-hosted"):
        raise PermissionError("R95_SOURCE_NOT_ADMITTED_FOR_ORIGINAL_SCIENCE")
    rng=np.random.default_rng(np.random.SeedSequence([SALT[panel],seed,index]))
    w=r85._make_synthetic_world(index%r85.N,rng)
    if panel=="OOD_SHIFT":
        w=r86._shift_public_world(w,index)
    return w,_chosen(seed,index)


def _raw_from_slots(slots:np.ndarray)->np.ndarray:
    if slots.shape!=(5,13) or slots.dtype!=np.float32:
        raise ValueError("R95_INVALID_FIVE_PUBLIC_RAW_ROWS")
    features=np.zeros(8,np.float32)
    if (not np.array_equal(np.rint(slots[:4,9]*4).astype(int),np.arange(1,5))
        or not np.all(slots[:4,5]==1) or not np.all(slots[:4,11]==1)):
        raise RuntimeError("R95_UNLAWFUL_A_PROBE_EVENT_ROLES")
    features[:4]=slots[:4,10]
    features[4]=slots[-1,r85.CALIBRATE]
    features[5]=slots[-1,r85.EXPLOIT_POS]
    features[6]=slots[-1,12]
    if (features[4],features[5]) not in ((1.,0.),(0.,1.)):
        raise RuntimeError("R95_NONFACTUAL_CURRENT_ACTION_FLAGS")
    return features


def _upper(features:np.ndarray)->np.ndarray:
    if features.ndim!=2 or features.shape[1]!=8:
        raise ValueError("R95_PUBLIC_UPPER_FEATURE_DIMENSION")
    out=features.copy()
    out[:,:4]=0.
    out[:,0]=np.where(features[:,:4]>0,1.,-1.).prod(axis=1)
    return out


def _shuffle_index(f:np.ndarray,seed:int,panel:str,action:np.ndarray|None):
    if panel not in COUNT or seed not in SEEDS or f.shape!=(COUNT[panel],8):
        raise PermissionError("R95_SHUFFLE_ONLY_FRESH_FROZEN_PANEL_GEOMETRY")
    if action is not None and action.shape!=(len(f),):
        raise ValueError("R95_SHUFFLED_ACTION_VECTOR_SHAPE")
    groups={}
    for i in range(len(f)):
        key=(float(f[i,6]),int(action[i]) if action is not None else -1)
        groups.setdefault(key,[]).append(i)
    rng=np.random.default_rng(np.random.SeedSequence([SHUFFLE_SALT,seed,SALT[panel]]))
    order=np.arange(len(f))
    for indices in groups.values():
        g=np.asarray(indices,dtype=np.int64)
        if len(g)<4:
            raise RuntimeError("R95_NOT_ENOUGH_PUBLIC_STRATUM_FOR_NEGATIVE")
        candidate=rng.permutation(g)
        if np.any(candidate==g):
            candidate=np.roll(g,1)
        order[g]=candidate
    if np.any(order==np.arange(len(f))):
        raise RuntimeError("R95_SHUFFLE_LEFT_UNCHANGED_CAUSAL_WORLD")
    return order


def _factual(panel:str,seed:int,sha:str)->dict:
    if panel not in ("FIT","FIT_VALIDATION"):
        raise PermissionError("R95_TRAINING_CANNOT_OPEN_SCORED_DEV_OOD")
    n=COUNT[panel]
    slots=np.empty((n,5,13),np.float32)
    raw=np.empty((n,8),np.float32)
    target=np.empty(n,np.float32)
    action=np.empty(n,np.int8)
    for i in range(n):
        w,act=_world(panel,seed,i,sha)
        prefix,ret=r86._view(w,act)
        selected=r88._public_slots(prefix)
        raw[i]=_raw_from_slots(selected)
        slots[i]=selected
        target[i]=ret
        action[i]=act
    perm=_shuffle_index(raw,seed,panel,action)
    shuffled=raw.copy()
    shuffled[:,:4]=raw[perm,:4]
    if not np.array_equal(shuffled[:,4:7],raw[:,4:7]):
        raise RuntimeError("R95_TRAIN_NEGATIVE_CHANGED_REAL_FACTUAL_ACTION")
    return {"slots":slots,"raw":raw,"upper":_upper(raw),"shuffled":shuffled,
            "return":target,"action":action,"only_factual_logged_return":True}


def _paired(panel:str,seed:int,sha:str)->dict:
    if panel not in ("DEV","OOD_SHIFT"):
        raise PermissionError("R95_EVALUATOR_ONLY_FRESH_DEV_OR_SHIFTED_OOD")
    n=COUNT[panel]
    slots0=np.empty((n,5,13),np.float32);slots2=np.empty_like(slots0)
    raw0=np.empty((n,8),np.float32);raw2=np.empty_like(raw0)
    r0=np.empty(n,np.float64);r2=np.empty(n,np.float64)
    lawful=np.empty(n,np.int64);cold=np.empty(n,bool)
    for i in range(n):
        w,_=_world(panel,seed,i,sha)
        x0,v0=r86._view(w,r85.CALIBRATE)
        x2,v2=r86._view(w,r85.EXPLOIT_POS)
        if (not np.array_equal(x0[:-1],x2[:-1])
            or not np.array_equal(x0[-1,4:],x2[-1,4:])
            or x0[-1,0]!=1. or x2[-1,2]!=1.):
            raise RuntimeError("R95_PAIRED_PHYSICAL_EVALUATOR_ACTION_LEAK")
        slots0[i],slots2[i]=r88._public_slots(x0),r88._public_slots(x2)
        raw0[i],raw2[i]=_raw_from_slots(slots0[i]),_raw_from_slots(slots2[i])
        r0[i],r2[i]=v0,v2
        lawful[i]=r85._policy(x0)
        cold[i]=bool(i%2)
    perm=_shuffle_index(raw0,seed,panel,None)
    sh0=raw0.copy();sh2=raw2.copy()
    sh0[:,:4]=raw0[perm,:4];sh2[:,:4]=raw0[perm,:4]
    if (not np.array_equal(sh0[:,4:7],raw0[:,4:7])
        or not np.array_equal(sh2[:,4:7],raw2[:,4:7])):
        raise RuntimeError("R95_NEGATIVE_EVALUATOR_CHANGED_PIVOT_ACTION")
    return {"slots0":slots0,"slots2":slots2,"raw0":raw0,"raw2":raw2,
            "upper0":_upper(raw0),"upper2":_upper(raw2),
            "shuffled0":sh0,"shuffled2":sh2,
            "reward0":r0,"reward2":r2,"lawful":lawful,"cold":cold}


class LearnedPairTree(nn.Module):
    """Trainable pair-to-pair nonlinear merger, NO parity or sign-product input."""
    def __init__(self):
        super().__init__()
        self.pair=nn.Sequential(nn.Linear(2,32),nn.GELU(),
                                nn.Linear(32,16),nn.GELU())
        self.merge=nn.Sequential(nn.Linear(32,32),nn.GELU(),
                                 nn.Linear(32,16),nn.GELU())
        self.readout=nn.Sequential(nn.Linear(19,32),nn.GELU(),nn.Linear(32,1))

    def forward(self,f:torch.Tensor)->torch.Tensor:
        if f.ndim!=2 or f.shape[1]!=8:
            raise RuntimeError("R95_TREE_ONLY_PUBLIC_EIGHT_DIMENSIONAL_FACTORS")
        p0=self.pair(f[:,0:2])
        p1=self.pair(f[:,2:4])
        relation=self.merge(torch.cat((p0,p1),dim=1))
        return self.readout(torch.cat((relation,f[:,4:7]),dim=1))[:,0]


class NativeSidecar(nn.Module):
    """Original 32-wide native core plus separate *experimental* value adapter.

    Direct additive sidecar is NOT proof the native core learns the relation.
    """
    def __init__(self,native:nn.Module,sidecar:nn.Module):
        super().__init__()
        self.native=native
        self.sidecar=sidecar

    def forward(self,slots:torch.Tensor,raw:torch.Tensor)->torch.Tensor:
        return self.native(slots)[:,-1,0]+self.sidecar(raw)


def _native(seed:int)->nn.Module:
    torch.manual_seed(seed)
    m=r74.make_model("HybridPilot_FULL",32)
    prior=m.ttt
    ttt=r75.TriangularFastWeightMemory(32,learning_rate=prior.eta)
    ttt.load_state_dict(prior.state_dict(),strict=True)
    m.ttt=ttt
    return m


def _models(seed:int)->dict:
    if seed not in SEEDS:
        raise PermissionError("R95_UNFROZEN_MODEL_SEED")
    models={}
    models[NATIVE_RAW]=_native(seed)
    torch.manual_seed(seed+301)
    models[NATIVE_FLAT]=NativeSidecar(_native(seed),r87.PublicValueMLP())
    # Reset sidecar independently after original native core init.
    torch.manual_seed(seed+302)
    models[NATIVE_TREE]=NativeSidecar(_native(seed),LearnedPairTree())
    for arm in (MLP_RAW,UPPER):
        torch.manual_seed(seed+303)
        models[arm]=r87.PublicValueMLP()
    for arm in (MLP_TREE,NEGATIVE):
        torch.manual_seed(seed+304)
        models[arm]=LearnedPairTree()
    torch.manual_seed(seed+305)
    models[GRU]=r81.GRU(64)
    originals=[models[NATIVE_RAW],
               models[NATIVE_FLAT].native,models[NATIVE_TREE].native]
    cores=[dict(m.named_parameters()) for m in originals]
    if not (cores[0].keys()==cores[1].keys()==cores[2].keys()):
        raise RuntimeError("R95_ORIGINAL_NATIVE_CORE_MODULE_DRIFT")
    if any(not torch.equal(cores[0][k],cores[j][k])
           for k in cores[0] for j in (1,2)):
        raise RuntimeError("R95_NATIVE_SIDECAR_INITIAL_CORE_MISMATCH")
    return models


def _predict_tensor(model:nn.Module,arm:str,slots:torch.Tensor,raw:torch.Tensor)->torch.Tensor:
    if arm==NATIVE_RAW:
        v=model(slots)[:,-1,0]
    elif arm in (NATIVE_FLAT,NATIVE_TREE):
        v=model(slots,raw)
    elif arm==GRU:
        v=model(slots)[:,-1,0]
    else:
        v=model(raw)
    if v.shape!=(len(slots),):
        raise RuntimeError("R95_FACTUAL_VALUE_SHAPE_CHANGED")
    return v


def _arrays(source:dict,arm:str,action:str="")->tuple[np.ndarray,np.ndarray]:
    suffix=action
    raw_key=("shuffled" if arm==NEGATIVE else
             "upper" if arm==UPPER else "raw")+suffix
    return source["slots"+suffix],source[raw_key]


def _predict_all(model,arm,source,action):
    slots,raw=_arrays(source,arm,action)
    model.eval()
    values=[]
    with torch.no_grad():
        for i in range(0,len(raw),32):
            x=torch.from_numpy(slots[i:i+32])
            f=torch.from_numpy(raw[i:i+32])
            values.extend(_predict_tensor(model,arm,x,f).tolist())
    model.train()
    out=np.asarray(values,np.float64)
    if len(out)!=len(raw) or not np.isfinite(out).all():
        raise RuntimeError("R95_NONFINITE_OR_INCOMPLETE_VALUE_PREDICTION")
    return out


def _mse(model,arm,source):
    pred=_predict_all(model,arm,source,"")
    return float(np.mean((pred-source["return"])**2))


def _train(model,arm,fit,valid,indices):
    slots,raw=_arrays(fit,arm)
    x=torch.from_numpy(slots);f=torch.from_numpy(raw)
    y=torch.from_numpy(fit["return"])
    steps=STEPS[arm]
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    started=time.perf_counter()
    before=_mse(model,arm,valid)
    for k in range(steps):
        ix=indices[k]
        estimate=_predict_tensor(model,arm,x[ix],f[ix])
        loss=F.mse_loss(estimate,y[ix])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("R95_NONFINITE_OBSERVED_FACTUAL_LOSS")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
    elapsed=time.perf_counter()-started
    train_err=_mse(model,arm,fit);valid_err=_mse(model,arm,valid)
    if not all(math.isfinite(q) for q in (before,train_err,valid_err,elapsed)):
        raise RuntimeError("R95_NONFINITE_COMPLETE_OBSERVATIONAL_FIT")
    return {"updates":steps,"batch":BATCH,"observed_training_action_exposures":steps*BATCH,
            "params":sum(v.numel() for v in model.parameters()),
            "initial_validation_return_MSE":before,"final_train_return_MSE":train_err,
            "final_validation_return_MSE":valid_err,
            "fit_wall_seconds":elapsed,"factual_reward_only":True}


def _score(models,panel):
    actions={};metrics={};returns={}
    for arm,m in models.items():
        v0=_predict_all(m,arm,panel,"0")
        v2=_predict_all(m,arm,panel,"2")
        actions[arm]=np.where(v0>v2,r85.CALIBRATE,r85.EXPLOIT_POS)
    n=len(panel["reward0"])
    actions["ALWAYS_CALIBRATE"]=np.full(n,r85.CALIBRATE)
    actions["ALWAYS_EXPLOIT_POS"]=np.full(n,r85.EXPLOIT_POS)
    actions["PUBLIC_PARITY_RULE_EVALUATOR_ONLY"]=panel["lawful"]
    oracle=np.maximum(panel["reward0"],panel["reward2"])
    for name,act in actions.items():
        true=np.where(act==r85.CALIBRATE,panel["reward0"],panel["reward2"])
        returns[name]=true
        metrics[name]={"mean_realized_reward":float(np.mean(true)),
                       "cold_realized_reward":float(np.mean(true[panel["cold"]])),
                       "early_realized_reward":float(np.mean(true[~panel["cold"]])),
                       "oracle_regret":float(np.mean(oracle-true)),
                       "agreement_with_public_parity_rule":float(np.mean(act==panel["lawful"]))}
    metrics["PHYSICAL_PAIRED_ORACLE_EVALUATOR_ONLY"]={
        "mean_realized_reward":float(np.mean(oracle)),
        "cold_realized_reward":float(np.mean(oracle[panel["cold"]])),
        "oracle_regret":0.}
    return {"metrics":metrics,"returns":returns}


def _paired_lower(a,b,seed):
    if a.shape!=(COUNT["DEV"],) or a.shape!=b.shape:
        raise RuntimeError("R95_FROZEN_PAIRED_BOOTSTRAP_SHAPE")
    rng=np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SALT,seed]))
    indices=rng.integers(0,len(a),size=(2000,len(a)))
    return float(np.quantile((a-b)[indices].mean(axis=1),.025))


def original_ci(seed:int)->dict:
    sha=_admit(seed)
    provenance=balance_preflight()
    torch.set_num_threads(1)
    outpath=ROOT/f"local_runs/r95_learned_relation_seed{seed}.json"
    if outpath.exists():
        raise FileExistsError("R95_ORIGINAL_FIRST_USE_OR_UNCERTAIN_REPLAY_REFUSED")
    started=time.perf_counter()
    out={"schema":SCHEMA,"seed":seed,"candidate_SHA":sha,
         "R94_parent_SHA":PARENT,"freeze_blob":FREEZE_BLOB,
         "pre_first_use_joint_balance":provenance[str(seed)],
         "runner_OS":os.getenv("RUNNER_OS"),"platform":platform.platform(),
         "torch":str(torch.__version__),
         "status":"R95_ENGINEERING_OR_RESOURCE_INCOMPLETE_NOT_SCIENTIFIC_NO_GO",
         "trained":{},"all_arms_frozen_before_new_DEV_OOD":False,
         "DEV_OOD_first_use":False,"previous_scored_or_protected_source_used":False,
         "scientific_pass":False}
    try:
        fit=_factual("FIT",seed,sha)
        valid=_factual("FIT_VALIDATION",seed,sha)
        if not fit["only_factual_logged_return"]:
            raise RuntimeError("R95_ILLEGAL_UNCHOSEN_ACTION_FIT_LABEL")
        draws=np.random.default_rng(np.random.SeedSequence(
            [BATCH_SALT,seed])).integers(0,COUNT["FIT"],size=(max(STEPS.values()),BATCH))
        models=_models(seed)
        for arm in ARMS:
            out["trained"][arm]=_train(models[arm],arm,fit,valid,draws)
            if time.perf_counter()-started>MAX_SECONDS:
                raise TimeoutError("R95_FROZEN_TRAINING_RESOURCE_CEILING_EXCEEDED")
        if set(out["trained"])!=set(ARMS):
            raise RuntimeError("R95_ARM_INCOMPLETE_BEFORE_NEW_SCORING")
        out["all_arms_frozen_before_new_DEV_OOD"]=True
        # The DEV/OOD evaluator may first construct paired alternative actions
        # ONLY after the full 8-arm factual FIT training has completed and frozen.
        out["DEV_OOD_first_use"]=True
        dev=_paired("DEV",seed,sha)
        ood=_paired("OOD_SHIFT",seed,sha)
        d=_score(models,dev);o=_score(models,ood)
        dm=d["metrics"];om=o["metrics"]
        cc=frozen()["evaluation"]["conventional_checks"]
        nc=frozen()["evaluation"]["native_sidecar_checks"]
        conv={
           "tree_minus_raw_DEV":dm[MLP_TREE]["mean_realized_reward"]-dm[MLP_RAW]["mean_realized_reward"],
           "tree_minus_shuffled_DEV":dm[MLP_TREE]["mean_realized_reward"]-dm[NEGATIVE]["mean_realized_reward"],
           "tree_minus_shuffled_OOD_cold":om[MLP_TREE]["cold_realized_reward"]-om[NEGATIVE]["cold_realized_reward"],
           "explicit_upper_minus_tree_DEV":dm[UPPER]["mean_realized_reward"]-dm[MLP_TREE]["mean_realized_reward"],
           "tree_lawful_DEV_agreement":dm[MLP_TREE]["agreement_with_public_parity_rule"],
           "tree_minus_shuffled_paired_bootstrap_lower95":
                 _paired_lower(d["returns"][MLP_TREE],d["returns"][NEGATIVE],seed)}
        native={
           "native_tree_minus_native_flat_DEV":
                 dm[NATIVE_TREE]["mean_realized_reward"]-dm[NATIVE_FLAT]["mean_realized_reward"],
           "native_tree_minus_native_raw_DEV":
                 dm[NATIVE_TREE]["mean_realized_reward"]-dm[NATIVE_RAW]["mean_realized_reward"],
           "native_tree_minus_native_flat_OOD_cold":
                 om[NATIVE_TREE]["cold_realized_reward"]-om[NATIVE_FLAT]["cold_realized_reward"],
           "conventional_tree_minus_native_tree_DEV":
                 dm[MLP_TREE]["mean_realized_reward"]-dm[NATIVE_TREE]["mean_realized_reward"]}
        conv_check={
           "beats_strong_RAW_MLP":
               conv["tree_minus_raw_DEV"]>=cc["MLP_tree_minus_raw_DEV_reward_min"],
           "beats_shuffled_input":
               conv["tree_minus_shuffled_DEV"]>=cc["MLP_tree_minus_shuffled_DEV_reward_min"],
           "retains_shifted_cold_against_shuffle":
               conv["tree_minus_shuffled_OOD_cold"]>=cc["MLP_tree_minus_shuffled_OOD_cold_reward_min"],
           "near_explicit_public_parity_upper":
               conv["explicit_upper_minus_tree_DEV"]<=cc["MLP_tree_vs_explicit_public_parity_DEV_reward_gap_max"],
           "learned_lawful_DEV_agreement":
               conv["tree_lawful_DEV_agreement"]>=cc["MLP_tree_legal_DEV_action_agreement_min"],
           "paired_bootstrap_positive":
               conv["tree_minus_shuffled_paired_bootstrap_lower95"]>=cc["MLP_tree_minus_shuffled_DEV_paired_bootstrap_lower95_min"]}
        native_check={
           "native_tree_beats_flat_sidecar":
               native["native_tree_minus_native_flat_DEV"]>=nc["native_tree_minus_native_flat_DEV_reward_min"],
           "native_tree_beats_orig_native":
               native["native_tree_minus_native_raw_DEV"]>=nc["native_tree_minus_native_raw_DEV_reward_min"],
           "native_tree_OOD_cold_beats_flat_sidecar":
               native["native_tree_minus_native_flat_OOD_cold"]>=nc["native_tree_minus_native_flat_OOD_cold_reward_min"],
           "native_tree_near_conventional_tree":
               native["conventional_tree_minus_native_tree_DEV"]<=nc["native_tree_vs_conventional_MLP_tree_DEV_reward_gap_max"]}
        out["DEV"]=dm;out["OOD_SHIFT"]=om
        out["conventional_comparisons"]=conv
        out["native_comparisons"]=native
        out["conventional_checks"]={k:bool(v) for k,v in conv_check.items()}
        out["native_sidecar_checks"]={k:bool(v) for k,v in native_check.items()}
        rule=frozen()["evaluation"]
        out["status"]=(rule["status_both"] if all(conv_check.values()) and all(native_check.values())
                       else rule["status_conventional"] if all(conv_check.values())
                       else rule["status_no_go"])
    except BaseException as exc:
        out["error_type"]=type(exc).__name__
        out["status"]="R95_ENGINEERING_OR_RESOURCE_INCOMPLETE_NOT_SCIENTIFIC_NO_GO"
        raise
    finally:
        out["total_wall_seconds"]=time.perf_counter()-started
        outpath.parent.mkdir(parents=True,exist_ok=True)
        with outpath.open("x",encoding="utf-8") as f:
            json.dump(out,f,indent=2,sort_keys=True,allow_nan=False)
            f.write("\n")
    return out


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,choices=SEEDS,required=True)
    p=parser.parse_args()
    outcome=original_ci(p.seed)
    print(json.dumps({"seed":p.seed,"status":outcome["status"],
       "arms":len(outcome["trained"]),"scored":outcome["DEV_OOD_first_use"],
       "scientific_pass":False},sort_keys=True))
