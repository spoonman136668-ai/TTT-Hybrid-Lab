"""R119 first native AdamW step boundary trace, only original public FIT.

Every recorded tensor is actually observed from the unchanged R116 forward,
backward and optimizer code. Original frozen R118 shared exact FP32 initial
weights and original public R116 first minibatch are unchanged. No original
R116 UNIT_UNSEEN or protected scientific source is accessed.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import r116_feedback_probe_diagnostic as r116
from experiments import r117_native_weight_portability_diagnostic as r117
from experiments import r118_shared_init_optimizer_diagnostic as r118
from experiments import r119_source_custody as custody

ROOT=Path(__file__).resolve().parents[1]
RECEIPT=ROOT/"local_runs/r119_first_update_FIT_numeric_boundary.json"
SEEDS=custody.SEEDS
STAGES=custody.STAGES
NATIVE=r116.NATIVE


def _pack_one(stage,name,value):
    if isinstance(value,torch.Tensor):
        arr=value.detach().cpu().contiguous().numpy()
    elif isinstance(value,np.ndarray):
        arr=value
    else:
        raise TypeError("R119_UNTRUSTED_NOT_ACTUAL_TENSOR")
    if arr.dtype==np.float32:
        a=arr.astype("<f4",copy=False)
        dtype="float32-le"
    elif arr.dtype==np.int64:
        a=arr.astype("<i8",copy=False)
        dtype="int64-le"
    else:
        raise RuntimeError("R119_NONCANONICAL_CAPTURE_DTYPE")
    if a.dtype.kind=="f" and not np.isfinite(a).all():
        raise RuntimeError("R119_NONFINITE_ACTUAL_BOUNDARY_VALUE")
    raw=a.tobytes(order="C")
    shape=list(a.shape)
    head=(b"R119_CANONICAL_TENSOR_STAGE_V1\0"+stage.encode()+b"\0"
          +name.encode()+b"\0"+dtype.encode()+b"\0"
          +struct.pack("<I",len(shape))
          +b"".join(struct.pack("<Q",int(d)) for d in shape))
    return {
        "name":name,"dtype":dtype,"shape":shape,"n_elements":int(a.size),
        "sha256":hashlib.sha256(head+raw).hexdigest(),
        "raw_le_base64":base64.b64encode(raw).decode("ascii")
    }


def _stage(stage,values):
    if stage not in STAGES or not values:
        raise PermissionError("R119_UNFROZEN_STAGE_OR_EMPTY_OBSERVATION")
    parts=[_pack_one(stage,name,v) for name,v in values]
    h=hashlib.sha256(b"R119_ACTUAL_STAGE_GROUP_V1\0"+stage.encode()+b"\0")
    for item in parts:
        h.update(item["name"].encode()+b"\0")
        h.update(bytes.fromhex(item["sha256"]))
    return {"stage":stage,"sha256":h.hexdigest(),"parts":parts,
            "total_elements":sum(x["n_elements"] for x in parts)}


def _all_parameters(model):
    return [(name,p.detach()) for name,p in model.named_parameters()]


def _all_grads(model):
    out=[]
    for name,p in model.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise RuntimeError("R119_ABSENT_OR_NONFINITE_NATIVE_GRADIENT")
        out.append((name,p.grad.detach()))
    return out


def environment():
    return {
        "system":platform.system(),
        "machine":platform.machine(),
        "python":sys.version.split()[0],
        "torch":torch.__version__,
        "numpy":np.__version__,
        "num_threads":torch.get_num_threads(),
        "mkldnn_enabled":bool(torch.backends.mkldnn.enabled),
        "deterministic_algorithms_enabled":bool(
             torch.are_deterministic_algorithms_enabled()),
        "float32_matmul_precision":torch.get_float32_matmul_precision()
    }


def _make_original_public_fit_first_batch(seed):
    custody.allow_fit("UNIT_FIT")
    if seed not in SEEDS:
        raise PermissionError("R119_FIT_SEED_NOT_FROZEN")
    torch.set_num_threads(1)
    fit,counts=r116.make_fit(seed)
    gen=np.random.default_rng(np.random.SeedSequence([27601119,seed,116]))
    draws=gen.integers(0,len(fit[-1]),size=(512,16))
    if draws.shape!=(512,16) or len(fit[-1])!=256:
        raise RuntimeError("R119_ORIGINAL_PUBLIC_MINIBATCH_DRIFT")
    return fit,draws[0],counts


def trace_first_update(seed):
    if seed not in SEEDS:
        raise PermissionError("R119_UNFROZEN_SEED")
    start=time.perf_counter()
    freeze=custody.audit_freeze()
    fit,ix,counts=_make_original_public_fit_first_batch(seed)
    model=r116.new_models(seed)[NATIVE]
    initial_sha,count=r118.seed_shared_initial_bytes(model,seed,NATIVE)
    if count!=3409:
        raise RuntimeError("R119_NATIVE_ARCHITECTURE_DRIFT")
    # Use unmodified R116 optimizer and model. Hooks are read-only and
    # return None; no manual forward layers or altered autodiff code.
    model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    n,m,g,target=fit
    records=[]
    def add(stage,fields):
        if stage!=STAGES[len(records)]:
            raise RuntimeError("R119_FROZEN_STAGE_ORDER_CHANGED")
        records.append(_stage(stage,fields))
    add("fit_n_first_batch",[("native_public_FIT_batch",n[ix])])
    add("fit_m_first_batch",[("MLP_public_FIT_batch",m[ix])])
    add("fit_g_first_batch",[("GRU_public_FIT_batch",g[ix])])
    add("fit_target_first_batch",[("public_Bayes_FIT_teacher_batch",target[ix])])
    add("fit_minibatch_indices",[("original_16_FIT_indices",ix.astype("<i8"))])
    add("before_first_update_native_raw_FP32_parameters",_all_parameters(model))
    hooks=[]
    hook_names=[
        ("forward.linear1_FP32_output",model.scorer[0]),
        ("forward.gelu1_FP32_output",model.scorer[1]),
        ("forward.linear2_FP32_output",model.scorer[2]),
        ("forward.gelu2_FP32_output",model.scorer[3]),
        ("forward.linear3_FP32_logits",model.scorer[4])]
    for name,layer in hook_names:
        def observe(module,args,output,stage_name=name):
            add(stage_name,[("actual_module_output",output)])
            return None
        hooks.append(layer.register_forward_hook(observe))
    try:
        predicted=model(n[ix],m[ix],g[ix])
    finally:
        for h in hooks:h.remove()
    if predicted.shape!=(16,2):
        raise RuntimeError("R119_ORIGINAL_NATIVE_FORWARD_OUTPUT_DRIFT")
    loss=F.mse_loss(predicted,target[ix])
    add("mse_loss_FP32_scalar",[("original_F_mse_loss",loss)])
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("R119_NONFINITE_ACTUAL_FIT_LOSS")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    add("raw_gradient_native_named_parameters",_all_grads(model))
    original_flat=torch.cat([p.grad.detach().reshape(-1)
                             for p in model.parameters()])
    norm=torch.linalg.vector_norm(original_flat)
    add("raw_gradient_global_norm_FP32",[("preclip_flat_global_L2",norm)])
    original_return=nn.utils.clip_grad_norm_(model.parameters(),1)
    add("clip_returned_preclip_norm_FP32",
        [("original_torch_clip_grad_norm_return",original_return)])
    add("clipped_gradient_native_named_parameters",_all_grads(model))
    if abs(float(norm)-float(original_return))>1e-4:
        raise RuntimeError("R119_GRAD_NORM_INCONSISTENT")
    opt.step()
    add("after_AdamW_step_native_named_parameters",_all_parameters(model))
    for state_key,stage in [
      ("exp_avg","after_AdamW_step_optimizer_exp_avg_by_parameter"),
      ("exp_avg_sq","after_AdamW_step_optimizer_exp_avg_sq_by_parameter"),
      ("step","after_AdamW_step_optimizer_step_by_parameter")]:
        states=[]
        for name,param in model.named_parameters():
            state=opt.state[param]
            if state_key not in state:
                raise RuntimeError("R119_MISSING_ACTUAL_OPTIMIZER_STATE")
            states.append((name,state[state_key].detach()))
        add(stage,states)
    if tuple(x["stage"] for x in records)!=STAGES:
        raise RuntimeError("R119_TRACE_DID_NOT_COMPLETE_FROZEN_BOUNDARIES")
    after=r117.raw_fp32_sha256(model)
    return {
        "seed":seed,
        "initial_native_named_parameters_sha256":initial_sha,
        "initial_native_original_R117_FP32_sha256":records[5]["sha256"],
        "native_raw_FP32_after_one_optimizer_step":after,
        "public_FIT_calibration":counts,
        "FIT_batch_indices":[int(v) for v in ix.tolist()],
        "actual_first_step_trace":records,
        "number_original_FIT_optimizer_updates":1,
        "source_panel":"UNIT_FIT",
        "no_original_scored_or_protected_firstuse":True,
        "process_wall_seconds":time.perf_counter()-start
    }


def original_unhooked_first_update_parameter_hash(seed):
    """Independent negative control: literal R116 first-step semantics."""
    fit,ix,_=_make_original_public_fit_first_batch(seed)
    model=r116.new_models(seed)[NATIVE]
    r118.seed_shared_initial_bytes(model,seed,NATIVE)
    n,m,g,target=fit
    model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    estimate=model(n[ix],m[ix],g[ix])
    loss=F.mse_loss(estimate,target[ix])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(),1)
    opt.step()
    return r117.raw_fp32_sha256(model)


def run_all():
    custody.audit_freeze()
    if (ROOT/"experiments/r119_protected_scientific_firstuse_authorization.json").exists():
        raise PermissionError("R119_PROTECTED_FIRST_USE_DENIED")
    exact_sha=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
              capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    expected=os.getenv("EXPECTED_CANDIDATE_SHA")
    if expected and exact_sha!=expected:
        raise PermissionError("R119_EXACT_SCORING_SOURCE_SHA_DRIFT")
    data={str(seed):trace_first_update(seed) for seed in SEEDS}
    for seed in SEEDS:
        exact=original_unhooked_first_update_parameter_hash(seed)
        if exact!=data[str(seed)]["native_raw_FP32_after_one_optimizer_step"]:
            raise RuntimeError("R119_READ_ONLY_HOOKS_CHANGE_ORIGINAL_NATIVE_STEP")
        data[str(seed)]["unhooked_original_first_step_final_fp32_sha256"]=exact
        data[str(seed)]["unchanged_R116_native_optimizer_first_step"]=True
    return {
        "schema":"ttt-hybrid.r119.actual-FIT-first-native-AdamW-update-numeric-boundary.v1",
        "source_exact_SHA":exact_sha,
        "prospective_freeze_blob":custody.FREEZE_BLOB,
        "status":"R119_TRACE_COMPLETE_CROSS_PLATFORM_ORIGINAL_EVIDENCE_COMPARISON_REQUIRED",
        "environment":environment(),
        "by_seed":data,
        "protected_science_firstuse":False,
        "original_scored_R116_UNSEEN_replayed":False,
        "independent_scientific_pass":False,
        "model_promotion":False,
        "deployment":"HOLD"
    }


def main():
    report=run_all()
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    if RECEIPT.exists():
        raise FileExistsError("R119_ORIGINAL_FIT_TRACE_RECEIPT_ALREADY_EXISTS")
    RECEIPT.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",
                       encoding="utf8")
    print("R119_FIT_TRACE "+json.dumps({
        "sha":report["source_exact_SHA"],
        "status":report["status"],
        "stages":len(STAGES),
        "seeds":{s:{
          "stages_sha256":[p["sha256"] for p in x["actual_first_step_trace"]],
          "unhooked_original_match":x["unchanged_R116_native_optimizer_first_step"],
          "initial_native_sha":x["initial_native_named_parameters_sha256"],
          "after_first_optimizer_step_native_sha":x["native_raw_FP32_after_one_optimizer_step"]}
          for s,x in report["by_seed"].items()},
        "no_original_R116_science_replayed":True,
        "scientific_pass":False
    },sort_keys=True))


if __name__=="__main__":
    main()
