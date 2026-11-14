"""F8-R63 prospective FRESH four-law synthetic task admission.

No R59/R60 scored data reused. Canonical MLP initial weights are fixed by
precommitted original R62 hashes; conventional GRU and handcoded public
interpolation are independently retained as controls. This is NOT a Native
architecture or independent causal-learning superiority experiment.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments import p85mr_f8r60_crossplatform_precision as r60
from experiments import p85mr_f8r62_canonical_initialization as r62

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r63_fresh_canonical_transfer_freeze.json"
FREEZE_BLOB="120645006a5ca8520056b6e858525f176b01bc81"
SCHEMA="ttt-hybrid.f8r63-fresh-fourlaw-canonical-task-admission-freeze.v1"
PARENT="4f73a80360e526d7bd9332b0ca553013802a1eb6"
SALT={"FIT_train":23510000,"FIT_heldout":23515000,"CAL":23520000,"OOD":23530000}
CASES=dict(r60.CASES)
TARGET_RANGE=dict(r60.TARGET_RANGE)
PROXY_SCALE=dict(r60.PROXY_SCALE)
NOISE_STD=r60.NOISE_STD
SEEDS=r62.SEEDS
ARMS=r60.ARMS
METHODS=tuple((arm,seed) for arm in ARMS for seed in SEEDS)
UPDATES=r60.UPDATES
DRAW_SEED=23549999
LABEL_SEED=23559999
CHECKPOINTS=r60.CHECKPOINTS
PANELS=r60.PANELS
EXPECTED_ACTIONS=61440


def git_blob(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def locked():
    raw=FREEZE.read_bytes()
    if git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("F8R63_FREEZE_BLOB_DRIFT")
    f=json.loads(raw)
    s,m,a=f["source"],f["models"],f["frozen_acceptance"]
    if (f["schema"]!=SCHEMA or
        f["stage"]!="FROZEN_BEFORE_R63_SOURCE_OR_SCORED_RESULTS" or
        f["parent"]["r62_SHA"]!=PARENT or
        f["parent"]["r62_freeze_blob"]!=r62.FREEZE_BLOB or
        s["new_salts"]!=SALT or s["cases"]!=CASES or
        s["target_ranges"]!={p:[-v,v] for p,v in TARGET_RANGE.items()} or
        s["selected_physical_goal_noise_std"]!=NOISE_STD or
        s["noncausal_proxy_SD"]!=PROXY_SCALE or
        m["seeds"]!=list(SEEDS) or m["arms"]!=list(ARMS) or
        m["FIT_steps_per_arm_seed"]!=UPDATES or
        m["total_optimizer_attempts"]!=UPDATES*len(METHODS) or
        m["draw_seed"]!=DRAW_SEED or
        m["tie_epsilon"]!=r60.TIE_EPS or
        m["expected_all_crossplatform_actions"]!=EXPECTED_ACTIONS or
        m["label_permutation_seed"]!=LABEL_SEED or
        m["label_permutation_steps"]!=256 or
        m["expected_MLP_initial_sha256"]!={str(k):v for k,v in r62.EXPECTED.items()} or
        a["exact_all_61440_decisions_OS_parity"] is not True or
        a["independent_scientific_pass"] is not False or
        any(f["authority"].values())):
        raise ValueError("F8R63_FROZEN_PROTOCOL_OR_AUTHORITY_DRIFT")
    r62.locked()
    return f


def _worlds(panel):
    """Fresh source-only worlds with identical R60 structural law *families*.

    The *new* salts and newly generated latent coefficients ensure R63
    training/evaluation are distinct from every spent R59-R62 scored cohort.
    """
    if panel not in PANELS:
        raise ValueError("F8R63_UNREGISTERED_PANEL")
    n=CASES[panel]
    rng=np.random.default_rng(SALT[panel])
    family=rng.permutation(np.arange(n,dtype=np.int64)%4)
    a=rng.normal(0,.5,n)
    bb=rng.normal(0,.8,n)
    c=rng.normal(0,.35,n)
    h=rng.uniform(-.4,.4,n)
    probe_positions=np.array([[-.84,0.,.84]],dtype=np.float64)+rng.uniform(
        -.14,.14,(n,3))
    target_x=rng.uniform(-TARGET_RANGE[panel],TARGET_RANGE[panel],n)
    structural_probe=r60._law_response(family,a,bb,c,h,probe_positions)
    structural_future=r60._law_response(family,a,bb,c,h,target_x[:,None])[:,0]
    nuisance=rng.normal(0,.012,(n,3))
    # Publicly selected action0 and action1 from independent probe resets.
    terminal0=structural_probe+nuisance+rng.normal(0,NOISE_STD,(n,3))
    terminal1=-structural_probe+nuisance+rng.normal(0,NOISE_STD,(n,3))
    # Action-dependent noncausal proxy sensor, also observed in both trials.
    proxy=rng.normal(0,PROXY_SCALE[panel],(n,3))
    proxy0=proxy+nuisance*.5
    proxy1=-proxy+nuisance*.5
    features=np.concatenate((
        probe_positions,(terminal0-terminal1)*.5,
        (proxy0-proxy1)*.5,target_x[:,None]),axis=1).astype(np.float64)
    # Public action features NEVER contain these future target return values.
    label=(structural_future<0).astype(np.int64)
    if features.shape!=(n,10) or not np.isfinite(features).all():
        raise ValueError("F8R63_INVALID_SELECTED_SOURCE")
    return {
        "panel":panel,
        "features":features,
        "labels_evaluator_only":label,
        "law_family_evaluator_only":family,
        "public_probe_positions":probe_positions,
        "public_target_position":target_x,
        "selected_terminal_action0":terminal0,
        "selected_terminal_action1":terminal1,
        "selected_proxy_action0":proxy0,
        "selected_proxy_action1":proxy1,
        "future_goal_effect_evaluator_only":structural_future,
    }


def selected_case(panel,index):
    if (panel not in PANELS or type(index) is not int or
        index<0 or index>=CASES[panel]):
        raise ValueError("F8R63_INVALID_SELECTED_CASE")
    w=_worlds(panel)
    public=[]
    for probe in range(3):
        for action in (0,1):
            for sensor,field in (
                ("terminal_goal","selected_terminal_action"),
                ("noncausal_proxy","selected_proxy_action")):
                public.append({"public_probe":probe,
                    "public_x":float(w["public_probe_positions"][index,probe]),
                    "actually_selected_action":action,
                    "sensor":sensor,
                    "observed_effect":float(w[f"{field}{action}"][index,probe])})
    result={"selected_factual_public_records":public,
            "model_public_feature_vector":w["features"][index].tolist(),
            "public_target_position":float(w["public_target_position"][index]),
            "hidden_return_label_evaluator_only":int(w["labels_evaluator_only"][index])}
    if panel.startswith("FIT_"):
        future=float(w["future_goal_effect_evaluator_only"][index])
        result["FIT_only_actual_selected_target_action_trials_evaluator_only"]=(
            [{"actually_selected_action":0,"selected_return_effect":future},
             {"actually_selected_action":1,"selected_return_effect":-future}])
    return result


def cohort(panel):
    w=_worlds(panel)
    return (torch.from_numpy(w["features"].copy()),
            torch.from_numpy(w["labels_evaluator_only"].copy()))


def source_sanity():
    out={}
    for panel in PANELS:
        w=_worlds(panel)
        n=CASES[panel]
        counts=np.bincount(w["law_family_evaluator_only"],minlength=4).tolist()
        balance=float(np.mean(w["labels_evaluator_only"]==0))
        if (counts!=[n//4]*4 or
            not .4<=balance<=.6 or
            w["features"].dtype!=np.float64):
            raise ValueError("F8R63_SOURCE_DISTRIBUTION_INVALID")
        contrast=(w["selected_terminal_action0"]-
                  w["selected_terminal_action1"])*.5
        if np.max(np.abs(contrast-w["features"][:,3:6]))>1e-12:
            raise ValueError("F8R63_SELECTED_TERMINAL_MISSING")
        out[panel]={"worlds":n,"family_counts":counts,
                    "label0_fraction":balance,
                    "selected_action_factual_records":12}
    return out


def fixed_controls():
    results={}
    for panel in PANELS:
        x,y=cohort(panel)
        a=x.numpy()
        truth=y.numpy()
        probes=a[:,:3]
        effect=a[:,3:6]
        target=a[:,-1]
        pred=np.zeros(len(y),dtype=np.float64)
        for j in range(3):
            others=[k for k in range(3) if k!=j]
            denominator=(probes[:,j]-probes[:,others[0]])*(
                probes[:,j]-probes[:,others[1]])
            pred+=effect[:,j]*(
                (target-probes[:,others[0]])*(target-probes[:,others[1]])/
                denominator)
        choices={
            "fixed_quadratic":(pred<0).astype(np.int64),
            "single_center":(effect[:,1]<0).astype(np.int64),
            "majority":(np.sum(effect<0,axis=1)>=2).astype(np.int64),
            "constant":np.zeros(len(y),dtype=np.int64)}
        results[panel]={k:{"accuracy":float(np.mean(v==truth)),
                           "choices":v.tolist()} for k,v in choices.items()}
    return results


def initialize(arm,seed):
    model=r60.Policy(arm,seed)
    canonical=None
    if arm==ARMS[0]:
        canonical=r62.canonical_init(model,seed)
    return model,canonical


def train_one(arm,seed,cohorts):
    model,initial=initialize(arm,seed)
    opt=torch.optim.AdamW(model.parameters(),lr=r60.LR,weight_decay=0.)
    rng=torch.Generator(device="cpu").manual_seed(DRAW_SEED)
    x,y=cohorts["FIT_train"]
    productive=0
    trace=[]
    started=time.perf_counter()
    for step in range(1,UPDATES+1):
        ids=torch.randint(0,len(y),(r60.BATCH,),generator=rng)
        model.train()
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(x[ids]),y[ids])
        if not torch.isfinite(loss):
            raise ValueError("F8R63_NONFINITE_FIT_LOSS")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in model.parameters()):
            raise ValueError("F8R63_NONFINITE_FIT_GRADIENT")
        productive+=int(any(p.grad is not None and bool(torch.any(p.grad!=0))
                            for p in model.parameters()))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if step in CHECKPOINTS:
            trace.append({"step":step,"loss":float(loss.detach()),
                          "FIT":r60.evaluate(model,*cohorts["FIT_train"])["accuracy"],
                          "FIT_heldout":r60.evaluate(model,*cohorts["FIT_heldout"])["accuracy"]})
    scores={panel:{"active":r60.evaluate(model,*cohorts[panel])}
            for panel in PANELS}
    scores["OOD"].update(r60.negative_ablations(model,cohorts))
    cold,_=initialize(arm,seed)
    cold.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    if any(r60.evaluate(cold,*cohorts[p])["choices"]!=
           scores[p]["active"]["choices"] for p in PANELS):
        raise ValueError("F8R63_COLD_RESTORE_ACTION_MISMATCH")
    families=_worlds("OOD")["law_family_evaluator_only"]
    correct=np.array(scores["OOD"]["active"]["choices"])==(
        cohorts["OOD"][1].numpy())
    return {"arm":arm,"seed":seed,"canonical_MLP_init_sha256":initial,
            "productive_FIT_steps":productive,
            "trainable_parameters":sum(p.numel() for p in model.parameters()),
            "training_seconds":time.perf_counter()-started,
            "FIT_trace":trace,"scores":scores,"cold_restore":True,
            "per_law_OOD_accuracy":{
                str(i):float(np.mean(correct[families==i]))
                for i in range(4)}}


def shuffled_label_control(cohorts):
    x,y=cohorts["FIT_train"]
    model,_=initialize(ARMS[0],SEEDS[0])
    rng=torch.Generator(device="cpu").manual_seed(LABEL_SEED)
    shuffled=y[torch.randperm(len(y),generator=rng)]
    draw=torch.Generator(device="cpu").manual_seed(DRAW_SEED)
    optimizer=torch.optim.AdamW(model.parameters(),lr=r60.LR,weight_decay=0.)
    for _ in range(256):
        idx=torch.randint(0,len(y),(r60.BATCH,),generator=draw)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(x[idx]),shuffled[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
    return {"FIT_heldout_accuracy":r60.evaluate(
                model,*cohorts["FIT_heldout"])["accuracy"],
            "FIT_only":True,"steps":256}


def judge(models,weak,source,negative,freeze):
    limits=freeze["frozen_acceptance"]
    lookup={(m["arm"],m["seed"]):m for m in models}
    seeds=[]
    for seed in SEEDS:
        m=lookup[ARMS[0],seed]["scores"]
        g=lookup[ARMS[1],seed]["scores"]
        expected_hash=lookup[ARMS[0],seed]["canonical_MLP_init_sha256"]
        checks={
            "canonical_init":expected_hash==r62.EXPECTED[seed],
            "MLP_FIT":m["FIT_train"]["active"]["accuracy"]>=limits["MLP_FIT_train_accuracy_min"],
            "MLP_heldout":m["FIT_heldout"]["active"]["accuracy"]>=limits["MLP_FIT_heldout_accuracy_min"],
            "MLP_CAL":m["CAL"]["active"]["accuracy"]>=limits["MLP_CAL_accuracy_min"],
            "MLP_OOD_floor":m["OOD"]["active"]["accuracy"]>=limits["MLP_OOD_accuracy_min"],
            "MLP_OOD_ceiling":m["OOD"]["active"]["accuracy"]<=limits["MLP_OOD_accuracy_max"],
            "GRU_FIT":g["FIT_train"]["active"]["accuracy"]>=limits["GRU_FIT_train_accuracy_min"],
            "GRU_heldout":g["FIT_heldout"]["active"]["accuracy"]>=limits["GRU_FIT_heldout_accuracy_min"],
            "GRU_OOD_floor":g["OOD"]["active"]["accuracy"]>=limits["GRU_OOD_accuracy_min"],
            "GRU_OOD_ceiling":g["OOD"]["active"]["accuracy"]<=limits["GRU_OOD_accuracy_max"],
            "MLP_erase_terminal":m["OOD"]["erase_selected_terminal"]["accuracy"]<=limits["MLP_erased_terminal_OOD_max"],
            "MLP_proxy_independence":m["OOD"]["active"]["accuracy"]-
                m["OOD"]["erase_noncausal_proxy"]["accuracy"]<=limits["MLP_proxy_erase_accuracy_drop_max"]}
        seeds.append({"seed":seed,"per_platform_pass":all(checks.values()),
                      "criteria":checks,
                      "MLP_OOD":m["OOD"]["active"]["accuracy"],
                      "GRU_OOD":g["OOD"]["active"]["accuracy"]})
    f=weak["OOD"]
    wc={
        "fixed_quadratic":limits["fixed_quadratic_OOD_min"]<=f["fixed_quadratic"]["accuracy"]<=limits["fixed_quadratic_OOD_max"],
        "single_center":f["single_center"]["accuracy"]<=limits["single_center_OOD_max"],
        "majority":f["majority"]["accuracy"]<=limits["majority_OOD_max"],
        "constant":f["constant"]["accuracy"]<=limits["constant_OOD_max"],
        "label_shuffle":negative["FIT_heldout_accuracy"]<=limits["FIT_label_shuffle_holdout_accuracy_max"],
        "balanced_source":all(limits["label_balance_min"]<=v["label0_fraction"]<=limits["label_balance_max"] for v in source.values())}
    return {"per_platform_task_admission":all(s["per_platform_pass"] for s in seeds) and all(wc.values()),
            "original_seeds":seeds,"fixed_and_negative_checks":wc,
            "cross_OS_parity":"UNVERIFIED_UNTIL_ORIGINAL_BOTH_OS_ARTIFACTS",
            "final_frozen_admission":"UNVERIFIED_UNTIL_CROSS_OS_COMPARISON",
            "independent_scientific_pass":False,"Native_specific_gain":False}


def run():
    frozen=locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled=False
    source=source_sanity()
    fixed=fixed_controls()
    cohorts={p:cohort(p) for p in PANELS}
    neg=shuffled_label_control(cohorts)
    models=[train_one(arm,seed,cohorts) for arm,seed in METHODS]
    decision=judge(models,fixed,source,neg,frozen)
    count=sum(sum(len(m["scores"][p]["active"]["choices"]) for p in PANELS)+
              sum(len(m["scores"]["OOD"][q]["choices"])
                  for q in ("erase_selected_terminal","erase_noncausal_proxy"))
              for m in models)
    count+=sum(len(fixed[p][q]["choices"]) for p in PANELS
               for q in ("fixed_quadratic","single_center","majority","constant"))
    if count!=EXPECTED_ACTIONS:
        raise ValueError("F8R63_MISSING_ACTION_ARRAYS")
    return {"schema":SCHEMA,"freeze_blob":FREEZE_BLOB,"parent":PARENT,
            "classification":"FRESH_FOUR_LAW_SYNTHETIC_TASK_ADMISSION_ONLY",
            "source_salts":SALT,
            "source_sanity":source,
            "cohort_sizes":CASES,
            "selected_factual_probe_outcomes_per_world":12,
            "fixed_controls":fixed,
            "FIT_only_label_shuffle_negative":neg,
            "models":models,"judgment":decision,
            "optimizer_attempts":UPDATES*len(METHODS),
            "all_categorical_decision_count":count,
            "external_original_firstuse":False,
            "independent_scientific_pass":False,
            "Native_specific_gain":False,
            "deployment":"NONE"}


def retain_ci_receipt(result):
    if os.getenv("GITHUB_ACTIONS")!="true":
        return
    p=ROOT/"local_runs/p85mr_f8r63_fresh_canonical_transfer_dev.json"
    p.parent.mkdir(exist_ok=True)
    if p.exists():
        raise FileExistsError("F8R63_REFUSE_OVERWRITE_ORIGINAL")
    p.write_text(json.dumps(result,sort_keys=True,indent=2,
                            allow_nan=False)+"\n")


def reject_external_original_firstuse(*args,**kwargs):
    raise PermissionError("F8R63_EXTERNAL_ORIGINAL_FIRSTUSE_DENIED")
