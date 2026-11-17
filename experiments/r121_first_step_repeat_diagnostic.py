"""R121: two independent process repetitions of the same public FIT first step.

Record actual FP32 source, forward activations, raw gradients, clipping and
post-AdamW state for BOTH original torch and R120 canonical clipping.
A first-step repeat has no scientific first use and no UNSEEN access.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
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
from experiments import r120_canonical_clip_portability_diagnostic as r120
from experiments import r121_source_custody as custody

ROOT=Path(__file__).resolve().parents[1]
RECEIPT=ROOT/"local_runs/r121_first_FIT_step_two_process_origins.json"
SEEDS=custody.SEEDS
MODES=custody.MODES
STAGES=custody.STAGES
NATIVE=r116.NATIVE


def _capture(stage,name,tensor):
    if isinstance(tensor,torch.Tensor):
        a=tensor.detach().cpu().contiguous().numpy()
    elif isinstance(tensor,np.ndarray):
        a=np.ascontiguousarray(tensor)
    else:
        raise TypeError("R121_ONLY_ACTUAL_NUMERICAL_TENSORS")
    if a.dtype==np.float32:
        a=a.astype("<f4",copy=False);dtype="float32-le"
    elif a.dtype==np.float64:
        a=a.astype("<f8",copy=False);dtype="float64-le"
    elif a.dtype==np.int64:
        a=a.astype("<i8",copy=False);dtype="int64-le"
    else:
        raise RuntimeError("R121_UNREGISTERED_FLOAT_OR_INTEGER_DTYPE")
    if a.dtype.kind=="f" and not np.isfinite(a).all():
        raise RuntimeError("R121_NONFINITE_ACTUAL_VALUE")
    raw=a.tobytes(order="C")
    shape=list(a.shape)
    head=(b"R121_TENSOR_STAGE_V1\0"+stage.encode()+b"\0"+
          name.encode()+b"\0"+dtype.encode()+b"\0"+
          struct.pack("<I",len(shape))+
          b"".join(struct.pack("<I",n) for n in shape))
    return {"name":name,"shape":shape,"dtype":dtype,
            "sha256":hashlib.sha256(head+raw).hexdigest(),
            "n_elements":int(a.size),
            "raw_le_base64":base64.b64encode(raw).decode("ascii")}


def _stage(stage,items):
    if stage not in STAGES or not items:
        raise PermissionError("R121_NONFROZEN_OR_EMPTY_STAGE")
    parts=[_capture(stage,name,x) for name,x in items]
    h=hashlib.sha256(b"R121_STAGE_V1\0"+stage.encode()+b"\0")
    for p in parts:
        h.update(p["name"].encode()+b"\0")
        h.update(bytes.fromhex(p["sha256"]))
    return {"stage":stage,"sha256":h.hexdigest(),"parts":parts}


def _grads(model):
    result=[]
    for name,p in model.named_parameters():
        if p.grad is None or p.grad.dtype!=torch.float32:
            raise RuntimeError("R121_MISSING_OR_NONFP32_GRADIENT")
        if not bool(torch.isfinite(p.grad).all()):
            raise RuntimeError("R121_NONFINITE_RAW_GRADIENT")
        result.append((name,p.grad.detach()))
    if sum(p.numel() for _,p in result)!=3409:
        raise RuntimeError("R121_NATIVE_GRAD_PARAMETER_LAYOUT")
    return result


def _params(model):
    return [(name,p.detach()) for name,p in model.named_parameters()]


def _original_first_batch(seed):
    custody.allow("UNIT_FIT")
    if seed not in SEEDS:
        raise PermissionError("R121_UNREGISTERED_FIT_SEED")
    torch.set_num_threads(1)
    fit,counts=r116.make_fit(seed)
    rng=np.random.default_rng(np.random.SeedSequence([27601119,seed,116]))
    batches=rng.integers(0,len(fit[-1]),size=(512,16))
    if batches.shape!=(512,16) or len(fit[-1])!=256:
        raise RuntimeError("R121_ORIGINAL_FIT_BATCH_OR_SOURCE_DRIFT")
    return fit,batches[0],counts


def _unhooked_original_first_step(seed,mode,fit,ix):
    model=r116.new_models(seed)[NATIVE]
    r118.seed_shared_initial_bytes(model,seed,NATIVE)
    model.train()
    n,m,g,target=fit
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    prediction=model(n[ix],m[ix],g[ix])
    loss=F.mse_loss(prediction,target[ix])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if mode==MODES[0]:
        nn.utils.clip_grad_norm_(model.parameters(),1.)
    elif mode==MODES[1]:
        r120.canonical_clip_grad_norm_(model)
    else:
        raise PermissionError("R121_NONREGISTERED_CLIPPING_MODE")
    opt.step()
    return r117.raw_fp32_sha256(model)


def trace(seed,mode):
    if seed not in SEEDS or mode not in MODES:
        raise PermissionError("R121_UNREGISTERED_SEED_OR_MODE")
    t0=time.perf_counter()
    fit,ix,counts=_original_first_batch(seed)
    model=r116.new_models(seed)[NATIVE]
    initial,nparams=r118.seed_shared_initial_bytes(model,seed,NATIVE)
    if nparams!=3409:
        raise RuntimeError("R121_INITIAL_NATIVE_3409_LAYOUT_DRIFT")
    n,m,g,target=fit
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    model.train()
    stages=[]
    def add(stage,items):
        if STAGES[len(stages)]!=stage:
            raise RuntimeError("R121_FROZEN_STAGE_CAPTURE_ORDER_CHANGED")
        stages.append(_stage(stage,items))
    add("FIT_first_batch_n",[("n",n[ix])])
    add("FIT_first_batch_m",[("m",m[ix])])
    add("FIT_first_batch_g",[("g",g[ix])])
    add("FIT_first_batch_target",[("target",target[ix])])
    add("FIT_first_batch_indices",[("actual_first_indices",ix.astype("<i8"))])
    add("initial_native_FP32_named_parameters",_params(model))
    layers=[("forward_linear1",model.scorer[0]),("forward_gelu1",model.scorer[1]),
            ("forward_linear2",model.scorer[2]),("forward_gelu2",model.scorer[3]),
            ("forward_linear3_logits",model.scorer[4])]
    hooks=[]
    for stage_name,module in layers:
        def listener(layer,args,output,capture_name=stage_name):
            add(capture_name,[("actual_output",output)])
            return None
        hooks.append(module.register_forward_hook(listener))
    try:
        prediction=model(n[ix],m[ix],g[ix])
    finally:
        for h in hooks:h.remove()
    if prediction.shape!=(16,2):
        raise RuntimeError("R121_R116_NATIVE_PUBLIC_OUTPUT_DRIFT")
    loss=F.mse_loss(prediction,target[ix])
    add("MSE_loss_FP32",[("actual_loss",loss)])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gradients=_grads(model)
    add("all_raw_FP32_named_gradients",gradients)
    floats=(float(x) for _,g in gradients for x in g.cpu().numpy().ravel(order="C"))
    squares=math.fsum(x*x for x in floats)
    stable_norm=math.sqrt(squares)
    add("global_raw_grad_norm_F64_fsum_squared_IEEE754",
        [("F64_fsum_squared",np.asarray(squares,dtype="<f8")),
         ("F64_math_sqrt_norm",np.asarray(stable_norm,dtype="<f8"))])
    if mode==MODES[0]:
        returned=nn.utils.clip_grad_norm_(model.parameters(),1.)
        coefficient=torch.clamp(1.0/(returned+1e-6),max=1.0)
        if coefficient.dtype!=torch.float32:
            raise RuntimeError("R121_TORCH_ORIGINAL_CLIP_COEFFICIENT_DTYPE")
        canonical_result=None
        actual_norm=returned
    else:
        # Measure the original norm using detached CLONES; the true model's
        # raw gradient tensors are never changed by this negative control.
        detached=[]
        for _,p in model.named_parameters():
            clone=torch.nn.Parameter(torch.zeros_like(p))
            clone.grad=p.grad.detach().clone()
            detached.append(clone)
        actual_norm=nn.utils.clip_grad_norm_(detached,1.)
        canonical_result=r120.canonical_clip_grad_norm_(model)
        if (canonical_result["raw_grad_sum_squares_F64_hex"]!=
                struct.pack("<d",squares).hex()):
            raise RuntimeError("R121_ACTUAL_R120_FSUM_RECOMPUTATION_DRIFT")
        coefficient=torch.tensor(canonical_result["coefficient"],dtype=torch.float32)
    add("original_torch_global_raw_gradient_norm_FP32",
        [("original_clip_returned_norm",actual_norm)])
    add("actual_clip_coefficient_FP32",
        [("actual_clip_multiplier",coefficient)])
    add("all_clipped_FP32_named_gradients",_grads(model))
    opt.step()
    add("all_post_AdamW_FP32_named_parameters",_params(model))
    for name,stage in (
      ("exp_avg","optimizer_exp_avg_FP32"),
      ("exp_avg_sq","optimizer_exp_avg_sq_FP32"),
      ("step","optimizer_step_scalar")):
        pieces=[]
        for parameter_name,p in model.named_parameters():
            x=opt.state[p].get(name)
            if x is None:
                raise RuntimeError("R121_ACTUAL_ADAMW_OPTIMIZER_STATE_MISSING")
            pieces.append((parameter_name,x.detach()))
        add(stage,pieces)
    if tuple(x["stage"] for x in stages)!=STAGES:
        raise RuntimeError("R121_MISSING_FROZEN_FIRST_UPDATE_BOUNDARY")
    after=r117.raw_fp32_sha256(model)
    unhooked=_unhooked_original_first_step(seed,mode,fit,ix)
    if unhooked!=after:
        raise RuntimeError("R121_OBSERVER_OR_HOOK_ALTERED_FIRST_ADAMW_STEP")
    return {
      "seed":seed,"mode":mode,"full_FIT_calibration_counts":counts,
      "initial_3409_FP32_SHA256":initial,
      "all_actual_frozen_stages":stages,
      "final_first_step_raw_FP32_SHA256":after,
      "negative_control_unhooked_SHA256":unhooked,
      "negative_control_pass":True,
      "number_AdamW_model_updates":1,
      "source":"UNIT_FIT",
      "no_original_spent_science":True,
      "wall_seconds":time.perf_counter()-t0
    }


def _worker_payload():
    custody.audit_freeze()
    torch.set_num_threads(1)
    results={str(seed):{mode:trace(seed,mode) for mode in MODES}
             for seed in SEEDS}
    return {"seeds":results,
            "environment":{
                "python":sys.version.split()[0],
                "torch":torch.__version__,
                "numpy":np.__version__,
                "platform":platform.system(),
                "machine":platform.machine(),
                "torch_threads":torch.get_num_threads(),
                "MKLDNN":bool(torch.backends.mkldnn.enabled)
            },
            "original_R116_UNIT_UNSEEN_reused":False,
            "number_trace_seeds":len(SEEDS),
            "number_trace_modes":len(MODES)}


def _first_divergence(left,right):
    for seed in SEEDS:
        for mode in MODES:
            a=left["seeds"][str(seed)][mode]["all_actual_frozen_stages"]
            b=right["seeds"][str(seed)][mode]["all_actual_frozen_stages"]
            for i,(x,y) in enumerate(zip(a,b)):
                if x["sha256"]!=y["sha256"]:
                    return {"seed":seed,"mode":mode,"stage_index":i,"stage":x["stage"]}
    return None


def run_all():
    custody.audit_freeze()
    sha=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
        capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    expected=os.getenv("EXPECTED_CANDIDATE_SHA")
    if expected and expected!=sha:
        raise PermissionError("R121_EXACT_SOURCE_SHA_DRIFT")
    if (ROOT/"experiments/r121_protected_science_firstuse_authorization.json").exists():
        raise PermissionError("R121_PROTECTED_SCIENTIFIC_FIRSTUSE_DENIED")
    workers=[]
    for replicate in ("A","B"):
        proc=subprocess.run([sys.executable,"-m",
                             "experiments.r121_first_step_repeat_diagnostic","--worker"],
             cwd=ROOT,capture_output=True,text=True,timeout=120,check=True,
             env={**os.environ,"R121_SINGLE_ORIGINAL_WORKER":replicate})
        try:
            child=json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError("R121_ORIGINAL_CHILD_STDOUT_NOT_CANONICAL_JSON") from e
        if (len(child["seeds"])!=3 or child["number_trace_modes"]!=2
            or child["original_R116_UNIT_UNSEEN_reused"] is not False):
            raise RuntimeError("R121_ORIGINAL_INDEPENDENT_WORKER_INCOMPLETE")
        workers.append({"worker":replicate,**child})
    first=_first_divergence(workers[0],workers[1])
    return {
      "schema":"ttt-hybrid.r121.FIT-first-step-independent-process-actual-source-bytes.v1",
      "exact_SHA":sha,
      "prospective_freeze_blob":custody.FREEZE_BLOB,
      "status":("R121_WITHIN_OS_SOURCE_OR_NUMERIC_DIFFERENCE_DETECTED"
                if first else "R121_WITHIN_OS_TWO_PROCESSES_ALL_STAGES_BITWISE_SAME"),
      "two_independent_workers":workers,
      "within_OS_first_divergence":first,
      "cross_OS_comparison_awaits_original_artifacts":True,
      "original_R116_heldout_replayed":False,
      "scientific_PASS":False,
      "deployment":"HOLD",
      "merge":False}


def main():
    if sys.argv[1:]==["--worker"]:
        print(json.dumps(_worker_payload(),sort_keys=True,separators=(",",":")))
        return
    if len(sys.argv)>1:
        raise PermissionError("R121_UNREGISTERED_WORKER_ARGUMENT")
    receipt=run_all()
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    if RECEIPT.exists():
        raise FileExistsError("R121_ORIGINAL_FIT_DIAGNOSTIC_ARTIFACT_EXISTS")
    RECEIPT.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("R121_FIT_REPEAT "+json.dumps({
        "sha":receipt["exact_SHA"],"status":receipt["status"],
        "first_within_OS_mismatch":receipt["within_OS_first_divergence"],
        "workers":[{"name":worker["worker"],
                    "source_and_initial_gradient_sha":[
                     {"seed":seed,"mode":mode,
                      "source_stage_sha":worker["seeds"][str(seed)][mode]
                        ["all_actual_frozen_stages"][0]["sha256"],
                      "raw_gradient_SHA":worker["seeds"][str(seed)][mode]
                        ["all_actual_frozen_stages"][12]["sha256"],
                      "post_step_sha":worker["seeds"][str(seed)][mode]
                        ["final_first_step_raw_FP32_SHA256"]}
                     for seed in SEEDS for mode in MODES]}
                   for worker in receipt["two_independent_workers"]],
        "protected_sci":False,"heldout_replay":False
    },sort_keys=True))


if __name__=="__main__":
    main()
