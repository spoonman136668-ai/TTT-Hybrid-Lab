"""F8-R62 engineering replay with byte-identical portable MLP initialization.

This is a SEPARATE pre-frozen diagnostic on already spent R60 cohorts.
No scientific rerun, Native architectural claim, GPU first-use, original
R60 result rewrite, or post-score adjustment is permitted.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from experiments import p85mr_f8r60_crossplatform_precision as r60
from experiments import p85mr_f8r61_fp64_first_divergence as r61

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r62_canonical_initialization_freeze.json"
FREEZE_BLOB="5f6d799d2509057096b798a792463d27b48dce23"
SCHEMA="ttt-hybrid.f8r62-canonical-initialization-FIT-only-replay-freeze.v1"
PARENT="0ce501be5ff11a85238d1b065cc326d9339ab760"
SALT="F8R62_CANONICAL_V1"
SEEDS=(985201,985207)
ARM="mlp_selected_goal_effects"
SHAPE={
    "head.0.weight":(64,10),"head.0.bias":(64,),
    "head.2.weight":(64,64),"head.2.bias":(64,),
    "head.4.weight":(2,64),"head.4.bias":(2,),
}
DENOM={
    "head.0.weight":4,"head.0.bias":4,
    "head.2.weight":8,"head.2.bias":8,
    "head.4.weight":8,"head.4.bias":8,
}
EXPECTED={
    985201:"e1dc51db89569357da6b29f6bf98232d8bef1df85a24f68a840757339571bf84",
    985207:"799b72a3a71125241c4527c9292c987746bcde83c7a9da2f7dfcf0ddb5f6bb96",
}
CHECKPOINTS=r61.CHECKPOINTS
SOURCE_HASH="8d11d1b6480b7e0e788f51ac99ec773805ddcbde53aadfa24b32d0585be71775"
PANELS=("FIT_train","FIT_heldout","CAL","OOD")
EXPECTED_ACTIONS=17408


def git_blob(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def locked():
    raw=FREEZE.read_bytes()
    if git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("F8R62_PROSPECTIVE_FREEZE_DRIFT")
    f=json.loads(raw)
    ic=f["canonical_initializer"]
    rp=f["replay"]
    if (f["schema"]!=SCHEMA or
        f["stage"]!="FROZEN_BEFORE_R62_CODE_OR_FIRST_CANONICAL_REPLAY" or
        f["parent"]["r61_sha"]!=PARENT or
        f["parent"]["r61_freeze_blob"]!=r61.FREEZE_BLOB or
        ic["salt"]!=SALT or
        ic["seed_to_exact_initial_hash"]!={str(k):v for k,v in EXPECTED.items()} or
        ic["source_tensor_shapes"]!={k:list(v) for k,v in SHAPE.items()} or
        ic["fan_in_denominator"]!=DENOM or
        rp["seeds"]!=list(SEEDS) or rp["arm"]!=ARM or
        rp["original_R60_source_hash"]!=SOURCE_HASH or
        rp["steps_per_seed"]!=r60.UPDATES or
        rp["batch_size"]!=r60.BATCH or
        rp["draw_seed"]!=r60.DRAW_SEED or
        rp["learning_rate"]!=r60.LR or
        rp["checkpoints"]!=list(CHECKPOINTS) or
        rp["evaluation_splits"]!=list(PANELS) or
        rp["fixed_action_tie_epsilon"]!=r60.TIE_EPS or
        rp["no_evaluator_OOD_data_optimization"] is not True or
        f["qualification"]["independent_scientific_pass"] is not False or
        any(f["authority"].values())):
        raise ValueError("F8R62_PROTOCOL_OR_AUTHORITY_DRIFT")
    r61.locked()
    return f


def portable_value(seed,name,index):
    if seed not in SEEDS or name not in SHAPE or index<0:
        raise ValueError("F8R62_UNREGISTERED_INITIALIZATION")
    raw=f"{SALT}/{seed}/{name}/{index}".encode("utf8")
    u=int.from_bytes(hashlib.sha256(raw).digest()[:2],"little")
    return (u-32768)/(32768*DENOM[name])


def canonical_init(model,seed):
    if seed not in SEEDS:
        raise ValueError("F8R62_UNREGISTERED_SEED")
    if tuple(n for n,_ in model.named_parameters()) != tuple(SHAPE):
        raise ValueError("F8R62_UNREGISTERED_PARAMETER_NAMES")
    with torch.no_grad():
        for name,param in model.named_parameters():
            if tuple(param.shape)!=SHAPE[name] or param.dtype!=torch.float64:
                raise ValueError("F8R62_BAD_PARAMETER_SHAPE_OR_DTYPE")
            values=[portable_value(seed,name,i) for i in range(param.numel())]
            param.copy_(torch.tensor(values,dtype=torch.float64).reshape_as(param))
    digest=r61.weights_digest(model)
    if digest!=EXPECTED[seed]:
        raise ValueError("F8R62_INITIAL_WEIGHT_HASH_MISMATCH")
    return digest


def trace_one(seed,source):
    x,y=source["FIT_train"]
    model=r60.Policy(ARM,seed)
    initial=canonical_init(model,seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=r60.LR,weight_decay=0.)
    rng=torch.Generator(device="cpu").manual_seed(r60.DRAW_SEED)
    started=time.perf_counter()
    records=[]
    checkpoints={"0":{"weight_sha256":initial,
                      "optimizer_sha256":r61.optimizer_state_digest(model,optimizer)}}
    productive=0
    for step in range(1,r60.UPDATES+1):
        ids=torch.randint(0,len(y),(r60.BATCH,),generator=rng)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(x[ids]),y[ids])
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R62_NONFINITE_FIT_LOSS")
        loss.backward()
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
               for p in model.parameters()):
            raise ValueError("F8R62_NONFINITE_FIT_GRADIENT")
        grad_hash=r61.gradient_digest(model)
        gradnorm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        productive+=int(any(p.grad is not None and bool(torch.any(p.grad!=0))
                            for p in model.parameters()))
        optimizer.step()
        weight_hash=r61.weights_digest(model)
        records.append({
            "step":step,"sample_batch_sha256":r61.batch_digest(ids),
            "prebackprop_loss_float64":float(loss.detach()),
            "preclip_gradient_sha256":grad_hash,
            "raw_gradient_l2_norm":gradnorm,
            "poststep_model_weights_sha256":weight_hash})
        if step in CHECKPOINTS:
            checkpoints[str(step)]={
                "weight_sha256":weight_hash,
                "optimizer_sha256":r61.optimizer_state_digest(model,optimizer),
                **r61.fixed_observation(model,x)}
    if len(records)!=r60.UPDATES or productive!=r60.UPDATES:
        raise ValueError("F8R62_INCOMPLETE_OR_ZERO_GRADIENT_UPDATES")
    scores={}
    for panel in PANELS:
        scores[panel]={"active":r60.evaluate(model,*source[panel])}
    scores["OOD"].update(r60.negative_ablations(model,source))
    cold=r60.Policy(ARM,seed)
    canonical_init(cold,seed)
    cold.load_state_dict(model.state_dict(),strict=True)
    if any(r60.evaluate(cold,*source[p])["choices"]!=
           scores[p]["active"]["choices"] for p in PANELS):
        raise ValueError("F8R62_COLD_RESTORE_CATEGORY_DRIFT")
    all_actions=sum(len(scores[p]["active"]["choices"]) for p in PANELS)
    all_actions+=sum(len(scores["OOD"][k]["choices"])
                     for k in ("erase_selected_terminal","erase_noncausal_proxy"))
    if all_actions!=EXPECTED_ACTIONS//len(SEEDS):
        raise ValueError("F8R62_UNEXPECTED_ACTION_CUSTODY")
    return {"arm":ARM,"seed":seed,
            "initial_weight_sha256":initial,
            "training_source_sha256":SOURCE_HASH,
            "original_spent_FIT_only_training":True,
            "optimizer_updates":r60.UPDATES,
            "productive_gradient_updates":productive,
            "trace_every_step":records,
            "milestone_diagnostics":checkpoints,
            "scores":scores,
            "categorical_actions_total":all_actions,
            "cold_restore":True,
            "elapsed_seconds":time.perf_counter()-started}


def run():
    frozen=locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled=False
    source={p:r60.cohort(p) for p in PANELS}
    if r61.data_digest(*source["FIT_train"])!=SOURCE_HASH:
        raise ValueError("F8R62_ORIGINAL_FIT_SOURCE_HASH_MISMATCH")
    trials=[trace_one(seed,source) for seed in SEEDS]
    count=sum(v["categorical_actions_total"] for v in trials)
    if count!=EXPECTED_ACTIONS:
        raise ValueError("F8R62_UNEXPECTED_ACTION_VECTOR_LENGTH")
    return {"schema":SCHEMA,"freeze_blob":FREEZE_BLOB,"parent":PARENT,
            "classification":"ENGINEERING_REPLAY_SPENT_R60_SOURCE_NOT_NEW_SCIENCE",
            "canonical_initialization_method":SALT,
            "canonical_seed_hashes":{str(seed):EXPECTED[seed] for seed in SEEDS},
            "source_FIT_sha256":SOURCE_HASH,
            "reference_previous_R60_science_NO_GO":True,
            "source_scores_OLD_SPENT_FOR_DIAG_ONLY":True,
            "CPU_dtype":"float64","mkldnn_enabled":False,"num_threads":1,
            "original_FIT_optimizer_steps":r60.UPDATES*len(SEEDS),
            "total_scored_categorical_actions":count,
            "trials":trials,
            "crossplatform_initial_hashes":"UNVERIFIED_UNTIL_BOTH_ORIGINAL_OS_ARTIFACTS",
            "crossplatform_training_weights":"UNVERIFIED_UNTIL_BOTH_ORIGINAL_OS_ARTIFACTS",
            "crossplatform_actions":"UNVERIFIED_UNTIL_BOTH_ORIGINAL_OS_ARTIFACTS",
            "independent_scientific_pass":False,
            "Native_specific_gain":False,
            "external_original_firstuse":False,
            "deployment":"NONE"}


def retain_ci_receipt(result):
    if os.getenv("GITHUB_ACTIONS")!="true":
        return
    p=ROOT/"local_runs/p85mr_f8r62_canonical_initialization_dev.json"
    p.parent.mkdir(exist_ok=True)
    if p.exists():
        raise FileExistsError("F8R62_EXISTING_ORIGINAL_REFUSE_OVERWRITE")
    p.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")


def reject_external_original_firstuse(*args,**kwargs):
    raise PermissionError("F8R62_EXTERNAL_ORIGINAL_FIRSTUSE_DENIED")
