"""R122 public-FIT-only native gradient/weight SHA at EVERY original update.

Two independent Python processes per OS, three frozen seeds, original R120
fixed-order clipping only. Full G4 artifacts preserve SHA at 512*17 stages
per seed, without copying large raw tensors or touching spent UNIT_UNSEEN.
"""
from __future__ import annotations

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
from experiments import r117_native_weight_portability_diagnostic as r117
from experiments import r118_shared_init_optimizer_diagnostic as r118
from experiments import r120_canonical_clip_portability_diagnostic as r120
from experiments import r121_first_step_repeat_diagnostic as r121
from experiments import r122_source_custody as custody

ROOT=Path(__file__).resolve().parents[1]
RECEIPT=ROOT/"local_runs/r122_every_512_FIT_step_source_gradient_provenance.json"
SEEDS=custody.SEEDS
UPDATES=custody.UPDATES
STAGES=custody.STAGES
NATIVE=r116.NATIVE


def _hash_actual(stage,items):
    """Actual tensor hash includes stage, original name, shape and LE bytes."""
    if not items:
        raise RuntimeError("R122_EMPTY_ACTUAL_TENSOR_STAGE")
    h=hashlib.sha256(b"R122_ACTUAL_SOURCE_STAGE_V1\0"+stage.encode()+b"\0")
    parts=[]
    for name,value in items:
        if isinstance(value,torch.Tensor):
            arr=value.detach().cpu().contiguous().numpy()
        elif isinstance(value,np.ndarray):
            arr=np.ascontiguousarray(value)
        else:
            raise TypeError("R122_NOT_ACTUAL_NUMERICAL_TENSOR")
        if arr.dtype==np.float32:
            arr=arr.astype("<f4",copy=False);kind="float32"
        elif arr.dtype==np.float64:
            arr=arr.astype("<f8",copy=False);kind="float64"
        elif arr.dtype==np.int64:
            arr=arr.astype("<i8",copy=False);kind="int64"
        else:
            raise RuntimeError("R122_NONFROZEN_CAPTURE_DTYPE")
        if arr.dtype.kind=="f" and not np.isfinite(arr).all():
            raise RuntimeError("R122_NONFINITE_NUMERICAL_OPERATION")
        header=(stage+"\0"+name+"\0"+kind+"\0"+",".join(map(str,arr.shape))).encode()
        digest=hashlib.sha256(header+b"\0"+arr.tobytes(order="C")).hexdigest()
        h.update(name.encode()+b"\0"+bytes.fromhex(digest))
        parts.append({"name":name,"shape":list(arr.shape),"dtype":kind,
                      "sha256":digest,"element_count":int(arr.size)})
    return {"stage":stage,"sha256":h.hexdigest(),"parts":parts}


def _parameters(model):
    result=[(name,p.detach()) for name,p in model.named_parameters()]
    if len(result)!=6 or sum(x.numel() for _,x in result)!=3409:
        raise RuntimeError("R122_NATIVE_NAMED_PARAMETER_LAYOUT_DRIFT")
    return result


def _gradients(model):
    result=[]
    for name,p in model.named_parameters():
        if p.grad is None or p.grad.dtype!=torch.float32:
            raise RuntimeError("R122_MISSING_OR_NON_FP32_GRAD")
        if not bool(torch.isfinite(p.grad).all()):
            raise RuntimeError("R122_NONFINITE_NATIVE_GRADIENT")
        result.append((name,p.grad.detach()))
    if len(result)!=6 or sum(x.numel() for _,x in result)!=3409:
        raise RuntimeError("R122_NATIVE_NAMED_GRADIENT_LAYOUT_DRIFT")
    return result


def _fit_and_batches(seed):
    custody.allow("UNIT_FIT")
    if seed not in SEEDS:
        raise PermissionError("R122_NOT_FROZEN_FIT_SEED")
    torch.set_num_threads(1)
    fit,counts=r116.make_fit(seed)
    rng=np.random.default_rng(np.random.SeedSequence([27601119,seed,116]))
    batches=rng.integers(0,len(fit[-1]),size=(UPDATES,16))
    if batches.shape!=(512,16) or len(fit[-1])!=256:
        raise RuntimeError("R122_FIT_OR_MINIBATCH_SOURCE_DRIFT")
    source={}
    for i,name in enumerate(("n","m","g","public_teacher")):
        source[name]=_hash_actual("full_public_FIT_"+name,[(name,fit[i])])["sha256"]
    source["all_512_minibatch_indices"]=_hash_actual(
        "all_public_FIT_minibatches",[("full_index_schedule",batches.astype("<i8"))])["sha256"]
    return fit,batches,source,counts


def trace_native_full_fit(seed):
    start=time.perf_counter()
    fit,batches,source,counts=_fit_and_batches(seed)
    n,m,g,target=fit
    model=r116.new_models(seed)[NATIVE]
    init_sha,parameter_count=r118.seed_shared_initial_bytes(model,seed,NATIVE)
    if parameter_count!=3409:
        raise RuntimeError("R122_R118_CANONICAL_INITIAL_MODEL_DRIFT")
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    recorded=[]
    current=None
    module_layers=[
        ("forward_linear1",model.scorer[0]),
        ("forward_gelu1",model.scorer[1]),
        ("forward_linear2",model.scorer[2]),
        ("forward_gelu2",model.scorer[3]),
        ("forward_linear3_logits",model.scorer[4])]
    hooks=[]
    for label,layer in module_layers:
        def capture(module,inputs,output,stage_name=label):
            if current is None or len(current)>=len(STAGES) or STAGES[len(current)]!=stage_name:
                raise RuntimeError("R122_MODULE_FORWARD_CAPTURE_STAGE_ORDER_DRIFT")
            current.append(_hash_actual(stage_name,[("native_actual_module_output",output)]))
            return None
        hooks.append(layer.register_forward_hook(capture))
    first_step_final=None
    try:
        for update,idx in enumerate(batches,start=1):
            current=[]
            def record(label,items):
                if STAGES[len(current)]!=label:
                    raise RuntimeError("R122_FROZEN_STAGE_OR_UPDATE_ORDER_DRIFT")
                current.append(_hash_actual(label,items))
            record("minibatch_indices_int64",[("original_FIT_indices",idx.astype("<i8"))])
            record("pre_update_named_FP32_weights",_parameters(model))
            prediction=model(n[idx],m[idx],g[idx])
            if prediction.shape!=(16,2):
                raise RuntimeError("R122_R116_NATIVE_OUTPUT_SHAPE_DRIFT")
            loss=F.mse_loss(prediction,target[idx])
            record("MSE_loss_FP32",[("actual_original_MSE",loss)])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("R122_NONFINITE_ORIGINAL_LOSS")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            record("raw_named_FP32_gradients",_gradients(model))
            clip=r120.canonical_clip_grad_norm_(model)
            squares=np.asarray(
                struct.unpack("<d",bytes.fromhex(clip["raw_grad_sum_squares_F64_hex"]))[0],
                dtype="<f8")
            norm=np.asarray(
                struct.unpack("<d",bytes.fromhex(clip["raw_grad_norm_F64_hex"]))[0],
                dtype="<f8")
            coeff=np.asarray(
                struct.unpack("<f",bytes.fromhex(clip["clipping_multiplier_exact_FP32_hex"]))[0],
                dtype="<f4")
            record("canonical_fsum_squares_F64",[("actual_F64_sumsq",squares)])
            record("canonical_sqrt_norm_F64",[("actual_F64_sqrt",norm)])
            record("canonical_clip_coefficient_FP32",[("actual_F32_coefficient",coeff)])
            record("clipped_named_FP32_gradients",_gradients(model))
            optimizer.step()
            record("post_AdamW_named_FP32_weights",_parameters(model))
            for key,label in (
                ("exp_avg","post_AdamW_exp_avg_FP32"),
                ("exp_avg_sq","post_AdamW_exp_avg_sq_FP32"),
                ("step","post_AdamW_step_FP32")):
                states=[]
                for name,p in model.named_parameters():
                    actual=optimizer.state[p].get(key)
                    if actual is None:
                        raise RuntimeError("R122_ADAMW_STATE_NOT_ACTUAL_OR_MISSING")
                    states.append((name,actual.detach()))
                record(label,states)
            if tuple(x["stage"] for x in current)!=STAGES:
                raise RuntimeError("R122_INCOMPLETE_512_STEP_STAGE_TRACE")
            recorded.append({"update":update,"stages":current})
            if update==1:
                first_step_final=r117.raw_fp32_sha256(model)
            current=None
    finally:
        for hook in hooks:
            hook.remove()
    if len(recorded)!=UPDATES:
        raise RuntimeError("R122_ORIGINAL_512_UPDATE_BUDGET_NOT_COMPLETED")
    unhooked=r121._unhooked_original_first_step(
        seed,r121.MODES[1],fit,batches[0])
    if first_step_final!=unhooked:
        raise RuntimeError("R122_READ_ONLY_HOOKS_CHANGED_UNMODIFIED_ORIGINAL_R120_FIRST_STEP")
    return {
        "seed":seed,"FIT_source_digest":source,
        "FIT_observed_sign_calibration_counts":counts,
        "initial_native_FP32_parameter_SHA256":init_sha,
        "first_step_native_FP32_SHA256":first_step_final,
        "first_step_unhooked_negative_control_SHA256":unhooked,
        "first_step_negative_control_pass":True,
        "last_step_native_FP32_SHA256":r117.raw_fp32_sha256(model),
        "steps":recorded,"number_updates":len(recorded),
        "native_clipping_method":"R120_F64_fsum_clipped_native",
        "original_spent_R116_UNIT_UNSEEN_reused":False,
        "process_wall_seconds":time.perf_counter()-start
    }


def _first_difference(left,right):
    if left["FIT_source_digest"]!=right["FIT_source_digest"]:
        return {"stage":"public_FIT_source","step":0}
    if left["initial_native_FP32_parameter_SHA256"]!=right["initial_native_FP32_parameter_SHA256"]:
        return {"stage":"canonical_initial_model","step":0}
    if len(left["steps"])!=UPDATES or len(right["steps"])!=UPDATES:
        raise RuntimeError("R122_INCOMPLETE_WORKER_UPDATE_COUNT")
    for i,(lstep,rstep) in enumerate(zip(left["steps"],right["steps"]),start=1):
        if lstep["update"]!=i or rstep["update"]!=i:
            raise RuntimeError("R122_UPDATE_SEQUENCE_DRIFT")
        if len(lstep["stages"])!=len(STAGES) or len(rstep["stages"])!=len(STAGES):
            raise RuntimeError("R122_STAGE_SEQUENCE_DRIFT")
        for j,(a,b) in enumerate(zip(lstep["stages"],rstep["stages"])):
            if a["stage"]!=STAGES[j] or b["stage"]!=STAGES[j]:
                raise RuntimeError("R122_STAGE_IDENTITY_DRIFT")
            if a["sha256"]!=b["sha256"]:
                leftparts=a["parts"];rightparts=b["parts"]
                differing=[p["name"] for p,q in zip(leftparts,rightparts)
                           if p["sha256"]!=q["sha256"]]
                return {"step":i,"stage_index":j,"stage":STAGES[j],
                        "differing_parameter_or_tensor_names":differing}
    return None


def _worker():
    custody.audit_freeze()
    torch.set_num_threads(1)
    seeds={str(seed):trace_native_full_fit(seed) for seed in SEEDS}
    return {"by_seed":seeds,"platform":platform.system(),
            "python":sys.version.split()[0],
            "numpy":np.__version__,"torch":torch.__version__,
            "torch_CPU_threads":torch.get_num_threads(),
            "worker_ID":os.getenv("R122_WORKER_ID",""),
            "original_R116_science_replayed":False}


def run_all():
    custody.audit_freeze()
    exact_sha=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
        capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    required=os.getenv("EXPECTED_CANDIDATE_SHA")
    if required and exact_sha!=required:
        raise PermissionError("R122_SOURCE_NOT_EXACT_CANDIDATE_SHA")
    if (ROOT/"experiments/r122_protected_scientific_firstuse_authorization.json").exists():
        raise PermissionError("R122_NOT_AUTHORIZED_PROTECTED_FIRST_USE")
    subprocesses=[]
    for worker in custody.WORKERS:
        result=subprocess.run([sys.executable,"-m",
              "experiments.r122_every_fit_update_gradient_trace","--worker"],
              cwd=ROOT,env={**os.environ,"R122_WORKER_ID":worker},
              capture_output=True,text=True,timeout=180,check=True)
        try:
            child=json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("R122_ORIGINAL_WORKER_NOT_SINGLE_JSON") from exc
        if (child["worker_ID"]!=worker or len(child["by_seed"])!=3 or
            child["original_R116_science_replayed"] is not False or
            any(x["number_updates"]!=UPDATES or
                not x["first_step_negative_control_pass"]
                for x in child["by_seed"].values())):
            raise RuntimeError("R122_TWO_WORKER_SOURCE_OR_TRAINING_INCOMPLETE")
        subprocesses.append(child)
    within={str(seed):_first_difference(
        subprocesses[0]["by_seed"][str(seed)],
        subprocesses[1]["by_seed"][str(seed)]) for seed in SEEDS}
    return {
        "schema":"ttt-hybrid.r122.all-512-FIT-original-gradient-numerical-source.v1",
        "exact_scored_engineering_SHA":exact_sha,
        "prospective_freeze_git_blob":custody.FREEZE_BLOB,
        "status":("R122_WITHIN_OS_DIVERGENCE_DETECTED"
                 if any(x is not None for x in within.values()) else
                 "R122_TWO_WITHIN_OS_WORKERS_ALL_512_STEPS_MATCH"),
        "two_independent_processes":subprocesses,
        "within_OS_first_difference_per_seed":within,
        "cross_platform_reproducibility_NOT_YET_QUALIFIED":True,
        "original_scored_R116_UNIT_UNSEEN_replayed":False,
        "scientific_PASS":False,"GPU_firstuse":False,
        "merge":False,"model_promotion":False,"deployment":"HOLD"}


def main():
    if sys.argv[1:]==["--worker"]:
        print(json.dumps(_worker(),sort_keys=True,separators=(",",":")))
        return
    if len(sys.argv)>1:
        raise PermissionError("R122_NONFROZEN_COMMAND_OR_WORKER_ARG")
    result=run_all()
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    if RECEIPT.exists():
        raise FileExistsError("R122_IMMUTABLE_FIT_DIAGNOSTIC_ALREADY_EXECUTED")
    RECEIPT.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",
                       encoding="utf8")
    print("R122_FIT_FULL_512 "+json.dumps({
        "source":result["exact_scored_engineering_SHA"],
        "status":result["status"],
        "within_OS_first_difference":result["within_OS_first_difference_per_seed"],
        "model_final_hashes":[{str(seed):worker["by_seed"][str(seed)]
                             ["last_step_native_FP32_SHA256"] for seed in SEEDS}
                              for worker in result["two_independent_processes"]],
        "original_spent_science_replayed":False,
        "science_PASS":False
    },sort_keys=True))


if __name__=="__main__":
    main()
