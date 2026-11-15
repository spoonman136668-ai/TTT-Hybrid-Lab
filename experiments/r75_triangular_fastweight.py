"""R75 mathematical delta-rule fast-weight implementation, source-neutral CPU gate.

This is an *implementation-equivalent* batched triangular solve for the exact
original R74 reference TTTLinearMemory online fast-weight state. Original
model.py, native source, R74 freeze, data, and model parameters remain intact.

Proof: W[t]=eta*sum_{i<t} r[i] k[i]^T, r[t]=v[t]-W[t]k[t].
Thus (I+eta*strict_lower(K K^T))R=V. Solve lower triangular R.
Reads[t]=eta*sum_{i<t} r[i]*(q[t] dot k[i]), strictly causal.
Equivalent in real arithmetic; floating-point accumulation and Adam
trajectories need separate qualification and are not bitwise identical.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

from model import Configuration, HybridPilot, TTTLinearMemory
from experiments import r74_capability_ceiling as r74

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r75_triangular_fastweight_freeze.json"
FREEZE_BLOB="2cf720b2214715c7977ffc5b87ea45ed193e3087"
PARENT="e5982735e031a572bd0f111615fc61119450b593"
SCHEMA="ttt-hybrid.r75-triangular-fastweight-engine.v1"
BATCH=2
LARGE_WIDTH=1024
LARGE_LENGTH=128
SEEDS=(997601,997607)


def git_blob(raw):
    raw=raw.replace(b"\r\n",b"\n")
    if b"\r" in raw:
        raise ValueError("R75_UNSUPPORTED_TEXT_CR")
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def frozen():
    spec=json.loads(FREEZE.read_text(encoding="utf-8"))
    if git_blob(FREEZE.read_bytes())!=FREEZE_BLOB:
        raise RuntimeError("R75_ORIGINAL_PROSPECTIVE_FREEZE_DRIFT")
    original=r74.frozen()
    if (spec["schema"]!=SCHEMA
        or spec["stage"]!="FROZEN_AFTER_UNSCORED_LOCAL_EXPLORATORY_PROTOTYPE_BEFORE_GITHUB_IMPLEMENTATION_AND_INDEPENDENT_CI"
        or spec["parent"]["R74_SHA"]!=PARENT
        or spec["parent"]["R74_original_freeze_git_blob"]!=r74.FREEZE_BLOB
        or spec["parent"]["legacy_reference_model_git_blob"]!=r74.MODEL_BLOB
        or spec["qualification"]["large_width"]!=LARGE_WIDTH
        or spec["qualification"]["large_length"]!=LARGE_LENGTH
        or spec["qualification"]["batch"]!=BATCH
        or spec["qualification"]["seeds"]!=list(SEEDS)
        or spec["math"]["strict_lower_triangular"] is not True
        or spec["training_custody"]["new_engine_identity_required"] is not True
        or spec["exploration_disclosure"]["already_observed_before_freeze"] is not True
        or any(spec["authority"].values())):
        raise RuntimeError("R75_ORIGINAL_FROZEN_CONTRACT_DRIFT")
    return spec


class TriangularFastWeightMemory(TTTLinearMemory):
    """Exact original source parameters, alternate causal linear-time matrix algebra."""

    def forward(self,x,update:bool=True):
        if x.ndim!=3 or x.shape[-1]!=self.key.in_features:
            raise ValueError("R75_SHAPE_GUARD")
        _,length,_=x.shape
        z=self.norm(x)
        keys=F.normalize(self.key(z),dim=-1)
        queries=F.normalize(self.query(z),dim=-1)
        values=torch.tanh(self.value(z))
        if update:
            kernel=torch.bmm(keys,keys.transpose(1,2))
            lower=torch.tril(kernel,diagonal=-1)
            identity=torch.eye(length,dtype=x.dtype,device=x.device).unsqueeze(0)
            triangle=identity+self.eta*lower
            residual=torch.linalg.solve_triangular(triangle,values,upper=False)
            read_kernel=torch.bmm(queries,keys.transpose(1,2))
            reads=self.eta*torch.bmm(torch.tril(read_kernel,diagonal=-1),residual)
        else:
            reads=torch.zeros_like(values)
        return x+self.proj(reads)


def make_original_or_optimized(width,optimized,seed):
    if width not in r74.LEVELS and width not in (16,32):
        raise ValueError("R75_UNFROZEN_WIDTH")
    if seed not in SEEDS:
        raise ValueError("R75_UNFROZEN_SEED")
    torch.manual_seed(seed)
    model=HybridPilot(13,1,Configuration(width=width,heads=4))
    if optimized:
        original=model.ttt
        replacement=TriangularFastWeightMemory(width,
                                               learning_rate=original.eta)
        replacement.load_state_dict(original.state_dict(),strict=True)
        model.ttt=replacement
    return model


def _same_seed_forward_backward(width,length,seed,optimized):
    torch.set_num_threads(1)
    model=make_original_or_optimized(width,optimized,seed)
    torch.manual_seed(seed+1)
    inp=torch.randn(BATCH,length,13)
    truth=torch.randn(BATCH,length,1)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    tic=time.perf_counter()
    pred=model(inp)
    objective=(pred-truth).square().mean()
    opt.zero_grad(set_to_none=True)
    objective.backward()
    elapsed=time.perf_counter()-tic
    gradients={n:(p.grad.detach().clone() if p.grad is not None else None)
               for n,p in model.named_parameters()}
    opt.step()
    new={n:p.detach().clone() for n,p in model.named_parameters()}
    count=sum(p.numel() for p in model.parameters())
    del opt,model
    gc.collect()
    return {"pred":pred.detach(),"grad":gradients,"before":before,"after":new,
            "loss":float(objective.detach()),"train_step_wall":elapsed,
            "parameters":count}


def check_math_parity(width=128,length=64,seed=SEEDS[0]):
    frozen()
    if length not in (8,32,64,128):
        raise ValueError("R75_UNREGISTERED_LENGTH")
    a=_same_seed_forward_backward(width,length,seed,False)
    b=_same_seed_forward_backward(width,length,seed,True)
    err=(a["pred"]-b["pred"]).abs().max().item()
    if a["before"].keys()!=b["before"].keys():
        raise RuntimeError("R75_MODEL_PARAMETER_SCHEMA_DRIFT")
    identity=all(torch.equal(a["before"][n],b["before"][n])
                 for n in a["before"])
    grad_diffs=[]
    for n in a["grad"]:
        x,y=a["grad"][n],b["grad"][n]
        if (x is None)!=(y is None):
            raise RuntimeError("R75_CHANGED_AUTOGRAD_REACHABILITY")
        if x is not None:
            grad_diffs.append(float((x-y).abs().max()))
    result={"parameters":a["parameters"],
            "initial_parameter_tensors_identical":bool(identity),
            "original_loss":a["loss"],"triangular_loss":b["loss"],
            "output_max_absolute_error":float(err),
            "gradient_max_absolute_error":max(grad_diffs),
            "original_forward_backward_seconds":a["train_step_wall"],
            "triangular_forward_backward_seconds":b["train_step_wall"],
            "reference_and_triangular_complete_real_AdamW_step":True,
            "width":width,"sequence":length}
    return result


def memory_rss_mib():
    if os.name=="nt":
        import ctypes
        from ctypes import wintypes
        class Counter(ctypes.Structure):
            _fields_=[("cb",wintypes.DWORD),
                      ("PageFaultCount",wintypes.DWORD),
                      ("PeakWorkingSetSize",ctypes.c_size_t),
                      ("WorkingSetSize",ctypes.c_size_t),
                      ("QuotaPeakPagedPoolUsage",ctypes.c_size_t),
                      ("QuotaPagedPoolUsage",ctypes.c_size_t),
                      ("QuotaPeakNonPagedPoolUsage",ctypes.c_size_t),
                      ("QuotaNonPagedPoolUsage",ctypes.c_size_t),
                      ("PagefileUsage",ctypes.c_size_t),
                      ("PeakPagefileUsage",ctypes.c_size_t)]
        info=Counter()
        info.cb=ctypes.sizeof(info)
        k=ctypes.windll.kernel32
        k.GetCurrentProcess.restype=wintypes.HANDLE
        api=ctypes.windll.psapi
        api.GetProcessMemoryInfo.argtypes=(wintypes.HANDLE,
                                          ctypes.POINTER(Counter),wintypes.DWORD)
        api.GetProcessMemoryInfo.restype=wintypes.BOOL
        if api.GetProcessMemoryInfo(k.GetCurrentProcess(),
                                    ctypes.byref(info),info.cb):
            return float(info.PeakWorkingSetSize/(1024*1024))
        return None
    import resource
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(peak/(1024*1024 if sys.platform=="darwin" else 1024))


def worker(optimized,width=LARGE_WIDTH,length=LARGE_LENGTH,seed=SEEDS[0]):
    frozen()
    if width!=LARGE_WIDTH or length!=LARGE_LENGTH:
        raise PermissionError("R75_UNFROZEN_REAL_HOST_PROFILE")
    torch.set_num_threads(1)
    model=make_original_or_optimized(width,optimized,seed)
    torch.manual_seed(seed+1)
    inp=torch.randn(BATCH,length,13)
    truth=torch.randn(BATCH,length,1)
    optim=torch.optim.AdamW(model.parameters(),lr=.001)
    tic=time.perf_counter()
    out=model(inp)
    loss=(out-truth).square().mean()
    optim.zero_grad(set_to_none=True)
    loss.backward()
    num_finite=sum(int(p.grad is not None and
                       bool(torch.isfinite(p.grad).all()))
                   for p in model.parameters())
    optim.step()
    elapsed=time.perf_counter()-tic
    digest=hashlib.sha256("".join(f"{float(v):.11g}," for v in out.detach().reshape(-1)).encode()).hexdigest()
    return {"engine":"TRIANGULAR" if optimized else "REFERENCE_SERIAL",
            "width":width,"sequence":length,
            "batch":BATCH,"parameters":sum(p.numel() for p in model.parameters()),
            "output_digest_float32":digest,
            "finite_grad_tensors":num_finite,
            "loss":float(loss.detach()),"one_full_AdamW_training_step_seconds":elapsed,
            "peak_process_RSS_MiB":memory_rss_mib(),
            "torch":str(torch.__version__),
            "platform":platform.platform(),
            "scientific_training":"UNEXECUTED"}


def profile_two_engines(per_engine_seconds=35):
    spec=frozen()
    if per_engine_seconds<=0 or per_engine_seconds>spec["qualification"]["CPU_wall_budget_per_engine_seconds"]:
        raise PermissionError("R75_RESOURCE_BUDGET_ESCALATION_DENIED")
    attempts=[]
    for optimized in (False,True):
        cmd=[sys.executable,"-m","experiments.r75_triangular_fastweight",
             "--worker","--engine","triangular" if optimized else "serial"]
        started=time.perf_counter()
        try:
            r=subprocess.run(cmd,check=False,text=True,capture_output=True,
                             cwd=str(ROOT),timeout=per_engine_seconds)
            if r.returncode!=0:
                status="PROCESS_FAIL"
                info={"returncode":r.returncode,"stderr_tail":r.stderr[-1600:]}
            else:
                info=json.loads(r.stdout.splitlines()[-1])
                ram=info["peak_process_RSS_MiB"]
                status=("PASS" if ram is not None and
                        ram<=spec["qualification"]["peak_RSS_budget_MiB"]
                        else "RSS_UNVERIFIED_OR_OVER_LIMIT")
        except subprocess.TimeoutExpired:
            status="TIMEOUT"
            info={"timeout_seconds":per_engine_seconds}
        attempts.append({"engine":"TRIANGULAR" if optimized else "REFERENCE_SERIAL",
                         "status":status,
                         "outer_wall_seconds":time.perf_counter()-started,
                         "receipt":info})
    if all(a["status"]=="PASS" for a in attempts):
        old,new=[a["receipt"] for a in attempts]
        if old["parameters"]!=spec["qualification"]["exact_parameter_count"] or \
           new["parameters"]!=old["parameters"]:
            raise ValueError("R75_SOURCE_PARAMETER_COUNT_DRIFT")
        speed=old["one_full_AdamW_training_step_seconds"]/new["one_full_AdamW_training_step_seconds"]
        result=("CPU_ENGINEERING_ACCELERATION_TARGET_MET"
                if speed>=spec["qualification"]["full_CPU_speedup_target_min"]
                else "CPU_ENGINEERING_PARITY_NO_SPEED_QUALIFICATION")
    else:
        speed=None
        result="CPU_ENGINEERING_RESOURCE_OR_EXECUTION_INCOMPLETE"
    return {"schema":SCHEMA,"freeze_blob":FREEZE_BLOB,
            "R74_reference_original_model_blob":r74.MODEL_BLOB,
            "benchmark":"ACTUAL_FULL_HYBRID_TRAIN_STEP_WITHOUT_LEARNING_CURVE",
            "measured_speedup":speed,
            "status":result,
            "trials":attempts,
            "GPU_first_use":False,"external_science_first_use":False,
            "full_65536_training":"NOT_EXECUTED",
            "model_learning_science":"UNVERIFIED",
            "deployment":"NONE"}


def deny_protected_first_use():
    raise PermissionError("R75_OFFICIAL_GPU_OR_EXTERNAL_FIRSTUSE_DENIED")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--worker",action="store_true")
    p.add_argument("--profile",action="store_true")
    p.add_argument("--engine",choices=("serial","triangular"),default="serial")
    a=p.parse_args()
    if a.worker:
        r=worker(a.engine=="triangular")
    elif a.profile:
        r=profile_two_engines()
    else:
        p.error("use --worker or --profile")
    print(json.dumps(r,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
