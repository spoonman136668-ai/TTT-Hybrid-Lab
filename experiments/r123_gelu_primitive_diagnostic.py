"""R123 first-FIT-source GELU primitive accuracy, forward and derivative bit parity.

Only original public FIT n/m/g first minibatch, 3 frozen seeds, original
canonical native initial weights and ONE forward. No optimizer updates,
validation, heldout, scientific first use, GPU or model mutation.
"""
from __future__ import annotations

import base64
from decimal import Decimal, localcontext
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
from torch.nn import functional as F

from experiments import r116_feedback_probe_diagnostic as r116
from experiments import r118_shared_init_optimizer_diagnostic as r118
from experiments import r122_every_fit_update_gradient_trace as r122
from experiments import r123_source_custody as custody

ROOT=Path(__file__).resolve().parents[1]
RECEIPT=ROOT/"local_runs/r123_original_cpu_gelu_primitive_fit_forward_gradient.json"
SEEDS=custody.SEEDS
CANDIDATES=custody.CANDIDATES
LAYERS=custody.LAYERS


def _tensor_bytes(name,raw):
    if isinstance(raw,torch.Tensor):
        a=raw.detach().cpu().contiguous().numpy()
    elif isinstance(raw,np.ndarray):
        a=np.ascontiguousarray(raw)
    else:
        raise TypeError("R123_NUMERIC_TENSOR_REQUIRED")
    if a.dtype!=np.float32:
        raise RuntimeError("R123_ONLY_ACTUAL_FP32_SOURCE_OR_DERIVATIVE")
    if not np.isfinite(a).all():
        raise RuntimeError("R123_NONFINITE_GELU_INPUT_OUTPUT_OR_GRAD")
    b=a.astype("<f4",copy=False).tobytes(order="C")
    name=name.encode()
    shape=list(a.shape)
    h=hashlib.sha256(b"R123_GELU_ACTUAL_FP32_LE_V1\0"+name+b"\0")
    h.update(struct.pack("<I",len(shape)))
    for dim in shape:h.update(struct.pack("<I",dim))
    h.update(b)
    return {"shape":shape,"n_elements":int(a.size),
            "float32_le_SHA256":h.hexdigest(),
            "float32_le_base64":base64.b64encode(b).decode("ascii")}


def _ordered_fp32(x):
    bit=np.asarray(x,dtype="<f4").ravel().view("<u4").astype(np.uint64)
    return np.where((bit & 0x80000000)!=0,(~bit)&0xffffffff,bit | 0x80000000)


def _error(actual,baseline):
    a=np.asarray(actual,dtype="<f4")
    b=np.asarray(baseline,dtype="<f4")
    if a.shape!=b.shape:
        raise RuntimeError("R123_NOT_SAME_ORIGINAL_ACTIVATION_SHAPE")
    dif=np.abs(a.astype(np.float64)-b.astype(np.float64))
    ulp=np.abs(_ordered_fp32(a).astype(np.int64)-
               _ordered_fp32(b).astype(np.int64))
    return {"n":int(a.size),"different_FP32_elements":int(
                np.count_nonzero(a.view("<u4")!=b.view("<u4"))),
            "max_abs":float(dif.max(initial=0.)),
            "mean_abs":float(dif.mean()),
            "max_ULP":int(ulp.max(initial=0)),
            "first_mismatch_flat_index":(
                int(np.flatnonzero(a.view("<u4")!=b.view("<u4"))[0])
                if np.any(a.view("<u4")!=b.view("<u4")) else None)}


def _erf_decimal_series(x,root2,rootpi,pi):
    z=x/root2
    term=z
    summation=Decimal(0)
    z2=z*z
    for i in range(96):
        summation+=term/Decimal(2*i+1)
        term=term*(-z2)/Decimal(i+1)
    erf=Decimal(2)/rootpi*summation
    phi=(-x*x/Decimal(2)).exp()/(root2*rootpi)
    return x*(Decimal(1)+erf)/Decimal(2),(
            (Decimal(1)+erf)/Decimal(2)+x*phi)


def decimal_forward_gradient(input_array):
    custody.audit_freeze()
    a=np.asarray(input_array,dtype="<f4")
    if not np.isfinite(a).all():
        raise RuntimeError("R123_NONFINITE_DECIMAL_INPUT")
    config=custody.audit_freeze()["capture"]
    if np.max(np.abs(a),initial=0)>config["decimal_input_abs_overflow_limit"]:
        raise RuntimeError("R123_OUTSIDE_PRECOMMITTED_DECIMAL_SERIES_DOMAIN")
    out=np.empty(a.size,dtype="<f4")
    grad=np.empty(a.size,dtype="<f4")
    with localcontext() as ctx:
        ctx.prec=70
        pi=Decimal(custody.PI)
        root2=Decimal(2).sqrt()
        rootpi=pi.sqrt()
        for j,bits in enumerate(a.ravel(order="C")):
            x=Decimal.from_float(float(bits))
            v,dv=_erf_decimal_series(x,root2,rootpi,pi)
            out[j]=np.float32(float(v))
            grad[j]=np.float32(float(dv))
    return out.reshape(a.shape),grad.reshape(a.shape)


def _torch_forward_grad(x,mode):
    t=x.detach().clone().to(dtype=torch.float32,device="cpu").requires_grad_(True)
    if mode=="torch_original_GELU_none":
        y=F.gelu(t,approximate="none")
    elif mode=="torch_GELU_tanh":
        y=F.gelu(t,approximate="tanh")
    elif mode=="torch_explicit_erf":
        y=(0.5*t)*(1.0+torch.erf(t/math.sqrt(2.0)))
    else:
        raise PermissionError("R123_GELU_ALTERNATIVE_NOT_FROZEN")
    (grad,)=torch.autograd.grad(y.sum(),t)
    return y.detach().cpu().numpy().astype("<f4",copy=True),(
        grad.detach().cpu().numpy().astype("<f4",copy=True))


def _original_public_native_inputs(seed):
    custody.allow("UNIT_FIT")
    if seed not in SEEDS:
        raise PermissionError("R123_UNREGISTERED_PUBLIC_FIT_SEED")
    torch.set_num_threads(1)
    fit,batches,source,_=r122._fit_and_batches(seed)
    idx=batches[0]
    n,m,g,target=fit
    model=r116.new_models(seed)[r116.NATIVE]
    canonical_sha,count=r118.seed_shared_initial_bytes(model,seed,r116.NATIVE)
    if count!=3409:
        raise RuntimeError("R123_NATIVE_INITIAL_PARAMETER_COUNT_DRIFT")
    observed={}
    hooks=[]
    for name,layer in ((LAYERS[0],model.scorer[1]),(LAYERS[1],model.scorer[3])):
        def pre(module,inputs,which=name):
            if which in observed:
                raise RuntimeError("R123_GELU_PRE_HOOK_REPEATED")
            observed[which]={"input":inputs[0].detach().clone().cpu()}
        def post(module,inputs,output,which=name):
            observed[which]["actual_output"]=output.detach().clone().cpu()
            return None
        hooks.append(layer.register_forward_pre_hook(pre))
        hooks.append(layer.register_forward_hook(post))
    model.eval()
    try:
        with torch.no_grad():
            prediction=model(n[idx],m[idx],g[idx])
    finally:
        for hook in hooks:hook.remove()
    if set(observed)!=set(LAYERS) or prediction.shape!=(16,2):
        raise RuntimeError("R123_ORIGINAL_NATIVE_HOOK_CAPTURE_INCOMPLETE")
    # Native is evaluated UNCHANGED with no hooks to prevent observer bias.
    with torch.no_grad():
        unhooked=model(n[idx],m[idx],g[idx])
    if not torch.equal(prediction,unhooked):
        raise RuntimeError("R123_READ_ONLY_HOOK_CHANGED_NATIVE_FORWARD")
    total=sum(a["input"].numel() for a in observed.values())
    if total!=2304:
        raise RuntimeError("R123_ORIGINAL_NATIVE_TWO_GELU_SHAPE_DRIFT")
    return observed,source,canonical_sha,idx


def one_seed(seed):
    start=time.perf_counter()
    observed,fit_digest,init_sha,idx=_original_public_native_inputs(seed)
    panels={}
    overall={}
    for lname in LAYERS:
        x=observed[lname]["input"].numpy().astype("<f4",copy=True)
        native=observed[lname]["actual_output"].numpy().astype("<f4",copy=True)
        reference_f,reference_g=_torch_forward_grad(
            torch.from_numpy(x),"torch_original_GELU_none")
        if not np.array_equal(native.view("<u4"),reference_f.view("<u4")):
            raise RuntimeError("R123_ORIGINAL_HOOK_OUTPUT_DIFFERS_FROM_NATIVE_GELU")
        alternatives={}
        for name in CANDIDATES:
            if name=="decimal_erf_series_96":
                f,g=decimal_forward_gradient(x)
            else:
                f,g=_torch_forward_grad(torch.from_numpy(x),name)
            error_f=_error(f,reference_f);error_g=_error(g,reference_g)
            alternatives[name]={
                "forward":_tensor_bytes(lname+"|"+name+"|forward",f),
                "input_gradient":_tensor_bytes(lname+"|"+name+"|gradient",g),
                "forward_error_from_original_GELU":error_f,
                "gradient_error_from_original_GELU":error_g}
        panels[lname]={
            "actual_GELU_input":_tensor_bytes(lname+"|input",x),
            "actual_native_original_forward":_tensor_bytes(lname+"|observed",native),
            "candidates":alternatives
        }
        overall[lname]={
           "decimal_max_abs_forward_error":alternatives["decimal_erf_series_96"][
               "forward_error_from_original_GELU"]["max_abs"],
           "decimal_max_abs_gradient_error":alternatives["decimal_erf_series_96"][
               "gradient_error_from_original_GELU"]["max_abs"],
           "original_GELU_grad_reference_SHA":alternatives[
               "torch_original_GELU_none"]["input_gradient"]["float32_le_SHA256"]
        }
    f=custody.audit_freeze()["capture"]
    fit_fidelity=all(
        v["decimal_max_abs_forward_error"]<=f["pass_max_decimal_forward_abs_error"] and
        v["decimal_max_abs_gradient_error"]<=f["pass_max_decimal_gradient_abs_error"]
        for v in overall.values())
    return {
        "seed":seed,"FIT_source_SHA256":fit_digest,
        "first_minibatch_indices":idx.tolist(),
        "initial_native_FP32_sha256":init_sha,
        "candidate_GELU_primitive_panels":panels,
        "fidelity":overall,
        "decimal_all_fidelity_thresholds_met":fit_fidelity,
        "original_native_model_parameter_updates":0,
        "original_R116_spent_UNSEEN_reused":False,
        "CPU_wall_seconds":time.perf_counter()-start
    }


def worker():
    custody.audit_freeze()
    torch.set_num_threads(1)
    return {
      "seeds":{str(seed):one_seed(seed) for seed in SEEDS},
      "environment":{
        "platform":platform.system(),"machine":platform.machine(),
        "python":sys.version.split()[0],"torch":torch.__version__,
        "numpy":np.__version__,"torch_threads":torch.get_num_threads(),
        "mkldnn":bool(torch.backends.mkldnn.enabled),
        "float32_matmul_precision":torch.get_float32_matmul_precision()
      },
      "heldout_reused":False,"model_updates":0
    }


def _first_worker_difference(a,b):
    for seed in SEEDS:
        left=a["seeds"][str(seed)];right=b["seeds"][str(seed)]
        if left["FIT_source_SHA256"]!=right["FIT_source_SHA256"]:
            return {"seed":seed,"stage":"public_FIT_source"}
        if left["initial_native_FP32_sha256"]!=right["initial_native_FP32_sha256"]:
            return {"seed":seed,"stage":"canonical_native_initial"}
        for layer in LAYERS:
            la=left["candidate_GELU_primitive_panels"][layer]
            rb=right["candidate_GELU_primitive_panels"][layer]
            for k in ("actual_GELU_input","actual_native_original_forward"):
                if la[k]["float32_le_SHA256"]!=rb[k]["float32_le_SHA256"]:
                    return {"seed":seed,"layer":layer,"stage":k}
            for candidate in CANDIDATES:
                for output in ("forward","input_gradient"):
                    if (la["candidates"][candidate][output]["float32_le_SHA256"]!=
                        rb["candidates"][candidate][output]["float32_le_SHA256"]):
                        return {"seed":seed,"layer":layer,
                                "candidate":candidate,"stage":output}
    return None


def run_all():
    custody.audit_freeze()
    sha=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
        capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    required=os.getenv("EXPECTED_CANDIDATE_SHA")
    if required and required!=sha:
        raise PermissionError("R123_NOT_EXACT_G4_CANDIDATE_SOURCE_SHA")
    if (ROOT/"experiments/r123_protected_original_firstuse_authorization.json").exists():
        raise PermissionError("R123_PROTECTED_FIRSTUSE_NOT_AUTHORIZED")
    workers=[]
    for who in custody.WORKERS:
        c=subprocess.run([sys.executable,"-m",
              "experiments.r123_gelu_primitive_diagnostic","--worker"],
              cwd=ROOT,env={**os.environ,"R123_WORKER_ID":who},
              capture_output=True,text=True,check=True,timeout=120)
        obj=json.loads(c.stdout)
        if (len(obj["seeds"])!=3 or obj["heldout_reused"] is not False or
            obj["model_updates"]!=0):
            raise RuntimeError("R123_INDEPENDENT_ORIGINAL_GELU_PROCESS_INCOMPLETE")
        workers.append({"worker_id":who,**obj})
    mismatch=_first_worker_difference(workers[0],workers[1])
    fidelity=all(v["decimal_all_fidelity_thresholds_met"] for worker in workers
                 for v in worker["seeds"].values())
    return {
        "schema":"ttt-hybrid.r123.GELU-source-primitive-original-FIT-CPU.v1",
        "exact_candidate_SHA":sha,
        "prospective_freeze_blob":custody.FREEZE_BLOB,
        "status":("R123_WITHIN_OS_GELU_BITWISE_REPEAT_FAIL" if mismatch else
                  "R123_WITHIN_OS_GELU_PRIMITIVE_REPEAT_PASS_CROSS_OS_REVIEW_REQUIRED"),
        "two_independent_processes":workers,
        "within_OS_first_mismatch":mismatch,
        "decimal_fidelity_to_original_torch_GELU_per_OS":fidelity,
        "cross_platform_primitive_parity_NOT_ASSUMED":True,
        "native_model_training_updates":0,
        "original_scored_R116_UNSEEN_replayed":False,
        "scientific_PASS":False,"merge":False,"promotion":False,"deployment":"HOLD"
    }


def main():
    if sys.argv[1:]==["--worker"]:
        print(json.dumps(worker(),sort_keys=True,separators=(",",":")))
        return
    if len(sys.argv)>1:
        raise PermissionError("R123_UNFROZEN_ORIGINAL_WORKER_ARGUMENT")
    receipt=run_all()
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    if RECEIPT.exists():
        raise FileExistsError("R123_ORIGINAL_GELU_PRIMITIVE_RECEIPT_ALREADY_EXISTS")
    RECEIPT.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n",
                       encoding="utf8")
    print("R123_GELU_ORIGINAL "+json.dumps({
        "source":receipt["exact_candidate_SHA"],
        "status":receipt["status"],
        "within_OS_mismatch":receipt["within_OS_first_mismatch"],
        "decimal_reference_fidelity":receipt["decimal_fidelity_to_original_torch_GELU_per_OS"],
        "seeds":{seed:{
          layer:{
            "original_gelu_forward":data["candidate_GELU_primitive_panels"][layer]
                ["candidates"]["torch_original_GELU_none"]["forward"]["float32_le_SHA256"],
            "decimal_forward":data["candidate_GELU_primitive_panels"][layer]
                ["candidates"]["decimal_erf_series_96"]["forward"]["float32_le_SHA256"],
            "decimal_grad":data["candidate_GELU_primitive_panels"][layer]
                ["candidates"]["decimal_erf_series_96"]["input_gradient"]["float32_le_SHA256"],
            "decimal_fidelity":data["fidelity"][layer]}
          for layer in LAYERS}
          for seed,data in receipt["two_independent_processes"][0]["seeds"].items()},
        "original_spent_sci_replayed":False
    },sort_keys=True))


if __name__=="__main__":
    main()
