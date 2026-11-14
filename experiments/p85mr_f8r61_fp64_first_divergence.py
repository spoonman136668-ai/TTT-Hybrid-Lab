"""F8-R61 exact CPU FP64 optimizer divergence diagnostic.

Engineering only: reuse original R60 FIT_train observations and original MLP
without changing the architecture, training, hyperparameters, source, target,
CUDA, original R60 frozen criteria or R60 OOD results. Record raw every-step
parameter/gradient hashes to locate the first cross-OS difference.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time

import numpy as np
import torch
from torch.nn import functional as F
from experiments import p85mr_f8r60_crossplatform_precision as r60

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r61_fp64_first_divergence_freeze.json"
FREEZE_BLOB="f5907b11cc29b3a38d82a22ff459cb6dda04e932"
SCHEMA="ttt-hybrid.f8r61-fp64-training-divergence-diagnostic-freeze.v1"
PARENT="8fc3045dce2eb76d68936830aec17ac33ba4424a"
SEEDS=(985201,985207)
CHECKPOINTS=(0,1,2,4,16,64,128,256,320,384,448,512,640,768,1024,1536)
ARM="mlp_selected_goal_effects"


def git_blob(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def locked():
    raw=FREEZE.read_bytes()
    if git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("F8R61_ORIGINAL_FROZEN_DIAGNOSTIC_DRIFT")
    f=json.loads(raw)
    a=f["parent"]
    m=f["method"]
    if (f["schema"]!=SCHEMA or
        f["stage"]!="FROZEN_BEFORE_R61_DIAGNOSTIC_CODE_OR_CROSS_PLATFORM_RUN" or
        f["classification"]!="ENGINEERING_DIAGNOSTIC_ONLY_SPENT_R60_FIT_DATA_NOT_NEW_SCIENCE" or
        a["r60_sha"]!=PARENT or
        a["r60_freeze_blob"]!=r60.FREEZE_BLOB or
        m["seeds"]!=list(SEEDS) or
        m["arm"]!=ARM or
        m["updates"]!=r60.UPDATES or
        m["batch_size"]!=r60.BATCH or
        m["draw_seed"]!=r60.DRAW_SEED or
        m["trace_at_checkpoints"]!=list(CHECKPOINTS) or
        m["cpu_num_threads"]!=1 or
        m["mkldnn_enabled"] is not False or
        f["decision"]["scientific_pass"] is not False or
        any(f["authority"].values())):
        raise ValueError("F8R61_PROSPECTIVE_DIAGNOSTIC_CONFIG_DRIFT")
    r60.locked()
    return f


def _tensor_bytes(tensor):
    """IEEE exact dtype little-endian raw bytes, including integer tensors."""
    raw=tensor.detach().cpu().contiguous().numpy()
    if raw.dtype.kind not in ("f","i","u"):
        raise TypeError("F8R61_NON_NUMERIC_TENSOR")
    canonical=np.asarray(raw,dtype=raw.dtype.newbyteorder("<"),order="C")
    return canonical.tobytes(order="C")


def _tagged_update(hashobj,name,t):
    b=name.encode("utf-8")
    hashobj.update(struct.pack("<I",len(b)))
    hashobj.update(b)
    dtype=str(t.dtype).encode("utf-8")
    hashobj.update(struct.pack("<I",len(dtype)))
    hashobj.update(dtype)
    hashobj.update(struct.pack("<I",t.ndim))
    for dim in t.shape:
        hashobj.update(struct.pack("<Q",int(dim)))
    binary=_tensor_bytes(t)
    hashobj.update(struct.pack("<Q",len(binary)))
    hashobj.update(binary)


def hash_named(items):
    h=hashlib.sha256()
    for name,p in sorted(items,key=lambda x:x[0]):
        _tagged_update(h,name,p)
    return h.hexdigest()


def optimizer_state_digest(model,opt):
    pairs=[]
    for name,p in model.named_parameters():
        state=opt.state.get(p,{})
        for key,value in state.items():
            if torch.is_tensor(value):
                pairs.append((name+"/"+str(key),value))
    return hash_named(pairs)


def gradient_digest(model):
    return hash_named((name,p.grad) for name,p in model.named_parameters()
                      if p.grad is not None)


def weights_digest(model):
    return hash_named((name,p) for name,p in model.named_parameters())


def data_digest(x,y):
    return hash_named((("FIT_selected_x",x),("FIT_choice_y",y)))


def batch_digest(ids):
    return hash_named((("original_FIT_selected_batch_indices",ids),))


def fixed_observation(model,x):
    with torch.no_grad():
        first=model.head[0](x[:128])
        second=model.head[2](torch.relu(first))
        logits=model(x[:4])
    return {
        "first_relu_active_fraction":float((first>0).double().mean()),
        "second_relu_active_fraction":float((second>0).double().mean()),
        "first_four_logit_margins":[float(z) for z in logits[:,1]-logits[:,0]],
        "parameter_l2_norm":math.sqrt(sum(float((p*p).sum()) for p in model.parameters()))
    }


def trace_one(seed,x,y):
    if seed not in SEEDS:
        raise ValueError("F8R61_UNREGISTERED_SEED")
    model=r60.Policy(ARM,seed)
    if (x.dtype!=torch.float64 or
        any(p.dtype!=torch.float64 for p in model.parameters())):
        raise ValueError("F8R61_NOT_FP64")
    optimizer=torch.optim.AdamW(model.parameters(),lr=r60.LR,weight_decay=0.)
    draws=torch.Generator(device="cpu").manual_seed(r60.DRAW_SEED)
    begin=time.perf_counter()
    initial=weights_digest(model)
    records=[]
    checkpoints={"0":{"weights_sha256":initial,
                      "optimizer_state_sha256":optimizer_state_digest(model,optimizer),
                      **fixed_observation(model,x)}}
    productive=0
    for step in range(1,r60.UPDATES+1):
        ids=torch.randint(0,len(y),(r60.BATCH,),generator=draws)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(x[ids]),y[ids])
        if not torch.isfinite(loss):
            raise ValueError("F8R61_NONFINITE_FIT_LOSS")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in model.parameters()):
            raise ValueError("F8R61_NONFINITE_FIT_GRADIENT")
        before_grad=gradient_digest(model)
        params=list(model.parameters())
        grad_norm=float(torch.nn.utils.clip_grad_norm_(params,1.))
        productive+=int(any(p.grad is not None and bool(torch.any(p.grad!=0))
                            for p in params))
        optimizer.step()
        records.append({
            "step":step,
            "loss_before_update_float64":float(loss.detach()),
            "selected_batch_sha256":batch_digest(ids),
            "raw_gradient_sha256":before_grad,
            "raw_gradient_global_l2_norm":grad_norm,
            "post_step_weights_sha256":weights_digest(model)
        })
        if step in CHECKPOINTS:
            checkpoints[str(step)]={
                "weights_sha256":records[-1]["post_step_weights_sha256"],
                "optimizer_state_sha256":optimizer_state_digest(model,optimizer),
                **fixed_observation(model,x)}
    if (len(records)!=r60.UPDATES or productive!=r60.UPDATES or
        len(checkpoints)!=len(CHECKPOINTS)):
        raise ValueError("F8R61_INCOMPLETE_DIAGNOSTIC")
    return {"seed":seed,"arm":ARM,
            "trainable_parameters":sum(p.numel() for p in model.parameters()),
            "original_R60_FIT_selected_source_only":True,
            "initial_weights_sha256":initial,
            "actual_FIT_updates":len(records),
            "productive_gradient_updates":productive,
            "trace_every_step":records,
            "milestone_diagnostics":checkpoints,
            "engineering_diagnostic_seconds":time.perf_counter()-begin}


def run():
    locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled=False
    x,y=r60.cohort("FIT_train")
    if x.shape!=(4096,10) or y.shape!=(4096,):
        raise ValueError("F8R61_ORIGINAL_FIT_ONLY_SOURCE_SHAPE_CHANGED")
    source_sha=data_digest(x,y)
    results=[trace_one(seed,x,y) for seed in SEEDS]
    return {"schema":SCHEMA,"parent":PARENT,"freeze_blob":FREEZE_BLOB,
            "classification":"ENGINEERING_ONLY_ORIGINAL_R60_FIT_SOURCE",
            "training_source_sha256":source_sha,
            "source_salt":r60.SALT["FIT_train"],
            "no_heldout_CAL_or_OOD_access":True,
            "CPU_dtype":"float64","mkldnn_enabled":False,"num_threads":1,
            "actual_original_recipe_FIT_updates":2*r60.UPDATES,
            "seeds":results,
            "comparison_status":"UNVERIFIED_UNTIL_ORIGINAL_WINDOWS_AND_UBUNTU_TRACES",
            "original_R60_source_or_science_unchanged":True,
            "independent_scientific_pass":False,
            "no_new_model_scientific_claim":True,
            "CUDA_first_use":False,"deployment":"NONE"}


def retain_ci_receipt(data):
    if os.getenv("GITHUB_ACTIONS")!="true":
        return
    p=ROOT/"local_runs/p85mr_f8r61_fp64_first_divergence_dev.json"
    p.parent.mkdir(exist_ok=True)
    if p.exists():
        raise FileExistsError("F8R61_EXISTING_TRACE_REFUSE_OVERWRITE")
    p.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n")


def reject_original_firstuse(*args,**kwargs):
    raise PermissionError("F8R61_ORIGINAL_EXTERNAL_FIRSTUSE_DENIED")
