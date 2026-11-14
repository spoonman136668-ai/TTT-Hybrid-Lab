"""F8-R60 separately frozen cross-platform numerical reproducibility admission.

New independently salted selected-intervention four-law cohorts, source-equivalent
to R59, but float64 data and model training, one CPU thread, disabled MKLDNN.
Strict original-action parity on Windows and Ubuntu is mandatory. Neither a
model architecture innovation nor an external/native scientific proof.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r60_crossplatform_precision_freeze.json"
FREEZE_BLOB="bfebb073b08c2c8287e836ea60207445f3375455"
SCHEMA="ttt-hybrid.f8r60-fp64-crossplatform-intervention-task-freeze.v1"
PARENT="b5d708bbbbf26e0ffee91d13082742599e793a8f"
SALT={"FIT_train":23410000,"FIT_heldout":23415000,"CAL":23420000,"OOD":23430000}
CASES={"FIT_train":4096,"FIT_heldout":768,"CAL":768,"OOD":1024}
TARGET_RANGE={"FIT_train":.9,"FIT_heldout":.9,"CAL":.9,"OOD":1.3}
PROXY_SCALE={"FIT_train":1.1,"FIT_heldout":1.1,"CAL":1.1,"OOD":1.6}
NOISE_STD=.33941125496954283
SEEDS=(985201,985207)
ARMS=("mlp_selected_goal_effects","gru_selected_goal_effects")
METHODS=tuple((arm,seed) for arm in ARMS for seed in SEEDS)
UPDATES,BATCH,LR=1536,128,.003
DRAW_SEED=23449999
LABEL_SHUFFLE_SEED=23459999
CHECKPOINTS=(1,256,512,1024,1536)
TIE_EPS=.003
PANELS=("FIT_train","FIT_heldout","CAL","OOD")


def git_blob(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def locked():
    raw=FREEZE.read_bytes()
    if git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("F8R60_PROSPECTIVE_FREEZE_DRIFT")
    f=json.loads(raw)
    s,e,a=f["source"],f["execution"],f["frozen_admission"]
    if (f["schema"]!=SCHEMA or
        f["stage"]!="FROZEN_BEFORE_NEW_R60_SOURCE_OR_SCORE" or
        f["parent"]["R59_SHA"]!=PARENT or
        f["parent"]["R59_freeze_blob"]!="e69d0debe514f8695cf0cef57f9dbfbeefc5ba8e" or
        s["salts"]!=SALT or s["cases"]!=CASES or
        s["target_context_range"]!={p:[-x,x] for p,x in TARGET_RANGE.items()} or
        s["selected_terminal_action_observation_noise_std"]!=NOISE_STD or
        s["proxy_sd"]!=PROXY_SCALE or
        e["torch_dtype"]!="float64" or
        e["torch_num_threads"]!=1 or
        e["mkldnn_enabled"] is not False or
        e["model_seeds"]!=list(SEEDS) or
        e["arms"]!=list(ARMS) or
        e["FIT_updates_per_model"]!=UPDATES or
        e["batch_size"]!=BATCH or
        e["learning_rate"]!=LR or
        e["draw_seed"]!=DRAW_SEED or
        e["trainable_total_attempts"]!=UPDATES*len(METHODS) or
        a["every_decision_bitwise_equal_Windows_Ubuntu"] is not True or
        a["independent_scientific_pass"] is not False or
        any(f["authority"].values())):
        raise ValueError("F8R60_PROTOCOL_OR_AUTHORITY_DRIFT")
    return f

def _law_response(family,a,b,c,h,x):
    """Evaluator-private four-law structural source. All shapes (N,1)/(N,3)."""
    family=family[:,None]
    ac=a[:,None]
    bc=b[:,None]
    cc=c[:,None]
    hh=h[:,None]
    return np.select(
        [family==0,family==1,family==2,family==3],
        [ac+bc*x,
         ac+bc*(x*x-.3),
         ac+bc*x+cc*(x*x-.3),
         ac+bc*np.tanh(3*(x-hh))])


def _worlds(panel):
    if panel not in PANELS:
        raise ValueError("F8R60_UNREGISTERED_PANEL")
    n=CASES[panel]
    rng=np.random.default_rng(SALT[panel])
    family=rng.permutation(np.arange(n,dtype=np.int64)%4)
    a=rng.normal(0,.5,n)
    b=rng.normal(0,.8,n)
    c=rng.normal(0,.35,n)
    h=rng.uniform(-.4,.4,n)
    x=np.array([[-.84,0.,.84]],dtype=np.float64)+rng.uniform(
        -.14,.14,(n,3))
    target_x=rng.uniform(-TARGET_RANGE[panel],TARGET_RANGE[panel],n)
    structural_probes=_law_response(family,a,b,c,h,x)
    structural_future=_law_response(family,a,b,c,h,target_x[:,None])[:,0]
    action_invariant=rng.normal(0,.012,(n,3))
    # Two *independently* selected physical action observations at each probe.
    # Their independent noises induce an expected contrast SD of about .24.
    selected0=structural_probes+action_invariant+rng.normal(0,NOISE_STD,(n,3))
    selected1=-structural_probes+action_invariant+rng.normal(0,NOISE_STD,(n,3))
    # Independent action-dependent public proxy effects: selected and observed,
    # deliberately independent of structural causal coefficients and labels.
    proxy=rng.normal(0,PROXY_SCALE[panel],(n,3))
    proxy0=proxy+action_invariant*.5
    proxy1=-proxy+action_invariant*.5
    terminal_contrast=(selected0-selected1)*.5
    proxy_contrast=(proxy0-proxy1)*.5
    vector=np.concatenate((x,terminal_contrast,proxy_contrast,
                           target_x[:,None]),axis=1).astype(np.float64)
    # Actual FIT target return actions: each separately selected on reset.
    # Their consequences are evaluator-owned and NEVER in vector.
    if panel.startswith("FIT_"):
        selected_future0=structural_future.copy()
        selected_future1=-structural_future.copy()
        choice=(selected_future0<selected_future1).astype(np.int64)
    else:
        selected_future0=None
        selected_future1=None
        choice=(structural_future<0).astype(np.int64)
    if vector.shape!=(n,10) or not np.isfinite(vector).all():
        raise ValueError("F8R60_INVALID_PUBLIC_SELECTED_FEATURES")
    if not np.isfinite(structural_future).all():
        raise ValueError("F8R60_NONFINITE_PRIVATE_GOAL")
    return {"panel":panel,"features":vector,"labels":choice,
            "family_evaluator_only":family,
            "probe_positions":x,"target_positions":target_x,
            "selected_terminal_action0":selected0,
            "selected_terminal_action1":selected1,
            "selected_proxy_action0":proxy0,
            "selected_proxy_action1":proxy1,
            "future_target_effect_evaluator_only":structural_future,
            "FIT_selected_future_action0_evaluator_only":selected_future0,
            "FIT_selected_future_action1_evaluator_only":selected_future1}


def selected_case(panel,index):
    if (panel not in PANELS or type(index) is not int or
        not 0<=index<CASES[panel]):
        raise ValueError("F8R60_UNREGISTERED_SELECTED_CASE")
    w=_worlds(panel)
    events=[]
    for source in range(3):
        state=float(w["probe_positions"][index,source])
        for action in (0,1):
            terminal=w[f"selected_terminal_action{action}"][index,source]
            proxy=w[f"selected_proxy_action{action}"][index,source]
            events.extend((
                {"public_probe":source,"public_x":state,
                 "actually_selected_action":action,
                 "sensor":"terminal_goal","observed_effect":float(terminal)},
                {"public_probe":source,"public_x":state,
                 "actually_selected_action":action,
                 "sensor":"noncausal_proxy","observed_effect":float(proxy)}
            ))
    if len(events)!=12:
        raise ValueError("F8R60_MISSING_SELECTED_INTERVENTIONS")
    result={"panel_evaluator_only":panel,
            "selected_public_records":events,
            "public_target_x":float(w["target_positions"][index]),
            "model_feature_vector":w["features"][index].tolist(),
            "future_goal_label_evaluator_only":int(w["labels"][index])}
    if panel.startswith("FIT_"):
        result["FIT_selected_future_return_trials_evaluator_only"]=[
            {"actually_selected_action":0,
             "observed_goal":float(w["FIT_selected_future_action0_evaluator_only"][index])},
            {"actually_selected_action":1,
             "observed_goal":float(w["FIT_selected_future_action1_evaluator_only"][index])}
        ]
    return result


def cohort(panel):
    w=_worlds(panel)
    return (torch.from_numpy(w["features"].copy()),
            torch.from_numpy(w["labels"].copy()))


def source_sanity():
    summary={}
    for panel in PANELS:
        w=_worlds(panel)
        counts=np.bincount(w["family_evaluator_only"],minlength=4).tolist()
        balance=float(np.mean(w["labels"]==0))
        if (counts!=[CASES[panel]//4]*4 or
            not .40<=balance<=.60):
            raise ValueError("F8R60_SOURCE_FAMILY_OR_LABEL_BALANCE_FAILED")
        inputs=w["features"].astype(np.float64)
        reconstructed=(w["selected_terminal_action0"]-
                       w["selected_terminal_action1"])*.5
        if np.max(np.abs(inputs[:,3:6]-reconstructed))>1e-6:
            raise ValueError("F8R60_SELECTED_PHYSICAL_EFFECT_NOT_IN_FEATURES")
        summary[panel]={"worlds":CASES[panel],"family_counts":counts,
                        "label0_fraction":balance,
                        "all_selected_probe_records_per_world":12,
                        "FIT_return_trails_present":panel.startswith("FIT_")}
    return summary


def _pred_from_sign(signal):
    return (signal<0).astype(np.int64)


def weak_controls():
    results={}
    for panel in PANELS:
        x,y=cohort(panel)
        a=x.numpy().astype(np.float64)
        y=y.numpy()
        xs=a[:,:3]
        measured=a[:,3:6]
        target=a[:,-1]
        # Source-only three-point Lagrange polynomial, a strong fixed
        # interpolator using ONLY selected public terminal effects.
        fitted=np.zeros(len(y),dtype=np.float64)
        for j in range(3):
            others=[k for k in range(3) if k!=j]
            interp=((target-xs[:,others[0]])*(target-xs[:,others[1]])/
                ((xs[:,j]-xs[:,others[0]])*(xs[:,j]-xs[:,others[1]])))
            fitted+=interp*measured[:,j]
        choices=_pred_from_sign(fitted)
        single=_pred_from_sign(measured[:,1])
        majority=(np.sum(measured<0,axis=1)>=2).astype(np.int64)
        const=np.zeros_like(y)
        results[panel]={
            "fixed_quadratic_interpolation_accuracy":float(np.mean(choices==y)),
            "single_center_probe_accuracy":float(np.mean(single==y)),
            "majority_three_selected_probes_accuracy":float(np.mean(majority==y)),
            "constant_action0_accuracy":float(np.mean(const==y)),
            "fixed_quadratic_choices":choices.tolist(),
            "single_center_choices":single.tolist(),
            "majority_choices":majority.tolist(),
            "constant_choices":const.tolist()}
    return results


class Policy(nn.Module):
    def __init__(self,arm,seed):
        super().__init__()
        if (arm,seed) not in METHODS:
            raise ValueError("F8R60_UNREGISTERED_ARM_OR_SEED")
        self.arm,self.seed=arm,seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(23479999+seed+(0 if arm==ARMS[0] else 100000))
            if arm==ARMS[0]:
                self.head=nn.Sequential(nn.Linear(10,64),nn.ReLU(),
                    nn.Linear(64,64),nn.ReLU(),nn.Linear(64,2))
            else:
                self.gru=nn.GRU(input_size=4,hidden_size=32,batch_first=True)
                self.readout=nn.Linear(32,2)
        self.double()  # Exact original architecture; all training/forward math float64

    def forward(self,x):
        if x.ndim!=2 or x.shape[1]!=10 or x.dtype!=torch.float64:
            raise ValueError("F8R60_INVALID_OBSERVED_INPUT")
        if self.arm==ARMS[0]:
            return self.head(x)
        probe=torch.stack((
            x[:,:3],x[:,3:6],x[:,6:9],
            x[:,-1,None].expand(-1,3)),dim=-1)
        _,hidden=self.gru(probe)
        return self.readout(hidden[-1])


def evaluate(model,inputs,labels):
    model.eval()
    with torch.no_grad():
        logits=model(inputs)
    if logits.shape!=(len(labels),2) or not bool(torch.isfinite(logits).all()):
        raise ValueError("F8R60_BAD_POLICY_OUTPUT")
    delta=logits[:,1]-logits[:,0]
    # Public, frozen non-informative tie rule avoids inventing evidence
    # from CPU float32 numerical cancellation on tiny logit margins.
    decisions=torch.where(delta.abs()<=TIE_EPS,torch.zeros_like(delta,dtype=torch.long),
                          (delta>0).long())
    return {"accuracy":float((decisions==labels).float().mean()),
            "choices":decisions.tolist(),
            "neutral_ties":int((delta.abs()<=TIE_EPS).sum())}


def negative_ablations(model,cohorts):
    x,y=cohorts["OOD"]
    terminal=x.clone()
    terminal[:,3:6].zero_()
    proxy=x.clone()
    proxy[:,6:9].zero_()
    return {"erase_selected_terminal":evaluate(model,terminal,y),
            "erase_noncausal_proxy":evaluate(model,proxy,y)}


def train_one(arm,seed,cohorts):
    model=Policy(arm,seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.)
    gen=torch.Generator(device="cpu")
    gen.manual_seed(DRAW_SEED)
    x,y=cohorts["FIT_train"]
    productive=0
    trace=[]
    started=time.perf_counter()
    for step in range(1,UPDATES+1):
        ids=torch.randint(0,len(y),(BATCH,),generator=gen)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(x[ids]),y[ids])
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R60_NONFINITE_FIT_LOSS")
        loss.backward()
        params=list(model.parameters())
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
               for p in params):
            raise ValueError("F8R60_NONFINITE_FIT_GRADIENT")
        productive+=int(any(p.grad is not None and bool(torch.any(p.grad!=0))
                            for p in params))
        torch.nn.utils.clip_grad_norm_(params,1.)
        optimizer.step()
        if step in CHECKPOINTS:
            trace.append({"step":step,"train_loss":float(loss.detach()),
                "FIT_accuracy":evaluate(model,*cohorts["FIT_train"])["accuracy"],
                "FIT_heldout_accuracy":evaluate(model,*cohorts["FIT_heldout"])["accuracy"]})
    scores={p:{"active":evaluate(model,*cohorts[p])} for p in PANELS}
    scores["OOD"].update(negative_ablations(model,cohorts))
    cold=Policy(arm,seed)
    cold.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    if any(evaluate(cold,*cohorts[p])["choices"]!=scores[p]["active"]["choices"]
           for p in PANELS):
        raise ValueError("F8R60_COLD_RESTORE_ACTION_MISMATCH")
    # The OOD family type is evaluator-only; never enters model features.
    families=_worlds("OOD")["family_evaluator_only"]
    good=np.array(scores["OOD"]["active"]["choices"])==cohorts["OOD"][1].numpy()
    return {"arm":arm,"seed":seed,"actual_FIT_optimizer_updates":UPDATES,
            "productive_gradient_steps":productive,
            "trainable_parameters":sum(p.numel() for p in model.parameters()),
            "training_wall_seconds":time.perf_counter()-started,
            "trace":trace,"scores":scores,
            "OOD_accuracy_per_law_family":{
                str(i):float(np.mean(good[families==i])) for i in range(4)},
            "cold_restore":True}


def permuted_label_negative(cohorts):
    """Small separate FIT-only negative, not another scored OOD-trained model."""
    model=Policy(ARMS[0],SEEDS[0])
    x,y=cohorts["FIT_train"]
    gen=torch.Generator(device="cpu").manual_seed(LABEL_SHUFFLE_SEED)
    wrong=y[torch.randperm(len(y),generator=gen)]
    draw=torch.Generator(device="cpu").manual_seed(DRAW_SEED)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.)
    steps=256
    for _ in range(steps):
        ids=torch.randint(0,len(y),(BATCH,),generator=draw)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(x[ids]),wrong[ids])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
    score=evaluate(model,*cohorts["FIT_heldout"])
    return {"FIT_only_permuted_label_updates":steps,
            "FIT_heldout_accuracy":score["accuracy"],
            "no_OOD_training":True}


def judge(models,source,fixed,neg,f):
    limits=f["frozen_admission"]
    by={(m["arm"],m["seed"]):m for m in models}
    scored=[]
    for seed in SEEDS:
        mlp=by[ARMS[0],seed]["scores"]
        gru=by[ARMS[1],seed]["scores"]
        controls={
            "MLP_FIT_train":mlp["FIT_train"]["active"]["accuracy"]>=limits["MLP_FIT_train_min"],
            "MLP_FIT_heldout":mlp["FIT_heldout"]["active"]["accuracy"]>=limits["MLP_FIT_holdout_min"],
            "MLP_CAL":mlp["CAL"]["active"]["accuracy"]>=limits["MLP_CAL_min"],
            "MLP_OOD_min":mlp["OOD"]["active"]["accuracy"]>=limits["MLP_OOD_min"],
            "MLP_OOD_max":mlp["OOD"]["active"]["accuracy"]<=limits["MLP_OOD_max"],
            "GRU_FIT_train":gru["FIT_train"]["active"]["accuracy"]>=limits["GRU_FIT_train_min"],
            "GRU_FIT_heldout":gru["FIT_heldout"]["active"]["accuracy"]>=limits["GRU_FIT_holdout_min"],
            "GRU_OOD_min":gru["OOD"]["active"]["accuracy"]>=limits["GRU_OOD_min"],
            "GRU_OOD_max":gru["OOD"]["active"]["accuracy"]<=limits["GRU_OOD_max"],
            "terminal_erasure":mlp["OOD"]["erase_selected_terminal"]["accuracy"]<=limits["terminal_erased_MLP_OOD_max"],
            "proxy_independence":mlp["OOD"]["active"]["accuracy"]-mlp["OOD"]["erase_noncausal_proxy"]["accuracy"]<=limits["proxy_erased_MLP_accuracy_loss_max"]}
        scored.append({"seed":seed,"standalone_numerical_task_pass":all(controls.values()),
                       "criteria":controls,
                       "MLP_OOD":mlp["OOD"]["active"]["accuracy"],
                       "GRU_OOD":gru["OOD"]["active"]["accuracy"]})
    weak=fixed["OOD"]
    common={
        "fixed_quadratic":limits["fixed_quadratic_OOD_min"]<=weak["fixed_quadratic_interpolation_accuracy"]<=limits["fixed_quadratic_OOD_max"],
        "single_probe":weak["single_center_probe_accuracy"]<=limits["single_probe_OOD_max"],
        "majority":weak["majority_three_selected_probes_accuracy"]<=limits["majority_probe_OOD_max"],
        "constant":weak["constant_action0_accuracy"]<=limits["constant_OOD_max"],
        "source_balance":all(.4<=z["label0_fraction"]<=.6 for z in source.values())}
    return {"per_platform_task_admission":all(z["standalone_numerical_task_pass"] for z in scored) and all(common.values()),
            "crossplatform_exact_decision_parity":"UNVERIFIED_UNTIL_ORIGINAL_BOTH_ARTIFACTS",
            "frozen_final_admission":"UNVERIFIED_UNTIL_BOTH_PLATFORM_PARITY",
            "seeds":scored,"source_controls":common,
            "independent_scientific_pass":False,"Native_specific_gain":False}

def run():
    freeze=locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled=False
    source=source_sanity()
    fixed=weak_controls()
    cohorts={panel:cohort(panel) for panel in PANELS}
    negative=permuted_label_negative(cohorts)
    models=[train_one(arm,seed,cohorts) for arm,seed in METHODS]
    verdict=judge(models,source,fixed,negative,freeze)
    return {"schema":SCHEMA,"freeze_blob":FREEZE_BLOB,"parent":PARENT,
            "source_neutral_task_admission_only":True,
            "cohort_sizes":CASES,"source_sanity":source,
            "weak_fixed_controls":fixed,"FIT_label_shuffle_negative":negative,
            "actual_optimizer_attempts":UPDATES*len(METHODS),
            "actual_FIT_only_label_shuffle_updates":256,
            "models":models,"judgment":verdict,
            "independent_scientific_pass":False,
            "novel_model_trained":False,
            "external_original_firstuse":False,"deployment":"NONE"}


def retain_ci_receipt(result):
    if os.getenv("GITHUB_ACTIONS")!="true":
        return
    dest=ROOT/"local_runs/p85mr_f8r60_crossplatform_precision_dev.json"
    dest.parent.mkdir(exist_ok=True)
    if dest.exists():
        raise FileExistsError("F8R60_REFUSE_OVERWRITE")
    dest.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")


def deny_external_original_firstuse(*args,**kwargs):
    raise PermissionError("F8R60_EXTERNAL_ORIGINAL_FIRSTUSE_DENIED")
