"""P9 G1: source-qualified existing G7-R3 CUDA runtime, then bounded GPU DEV probe.

No installation, model training, Mamba/TTT kernel use, runner changes or science.
The outer process imports Python standard library only; the admitted child
imports torch from the EXACT existing project G7-R3 venv. Failure is
fail-closed and returns a sanitized immutable receipt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import g7_r5_oneshot as custody

SCHEMA="ttt-hybrid.native-p9.existing-r3-cuda-once.v1"
BASE_SHA="083438e51212c1e41a213001263fdd0d002dd4cc"
FROZEN_MANIFEST="experiments/native_p9_existing_cuda_admission_frozen.json"
MIN_FREE_GIB=8.0
MAX_TORCH_ALLOCATED_MIB=256
MAX_GPU_UTILIZATION_PERCENT=20
MAX_COMPUTE_PROCESSES=0
MAX_CHILD_WALL_SECONDS=130
EXPECTED_TORCH="2.9.0+cu128"
EXPECTED_TRITON="3.5.0"
EXPECTED_CAPABILITY=[12,0]
STEPS=64
BATCH=4
REPETITIONS=3


class AdmissionBlocked(RuntimeError):
    pass


def demand(ok:bool,reason:str):
    if not ok:
        raise AdmissionBlocked(reason)


def checked_git(*args:str)->str:
    try:
        result=subprocess.run(["git","-C",str(ROOT),*args],capture_output=True,
                              text=True,check=False,timeout=12)
        demand(result.returncode==0,"SOURCE_GIT_UNAVAILABLE")
        return result.stdout.strip()
    except (OSError,subprocess.TimeoutExpired):
        raise AdmissionBlocked("SOURCE_GIT_UNAVAILABLE") from None


def environment_root()->Path:
    return Path.home().resolve()/".local"/"share"/"ttt-hybrid"/(
        "g7-r3-"+custody.R3_SOURCE_SHA[:12])


def verify_lineage(expected:str)->dict:
    demand(bool(re.fullmatch(r"[0-9a-f]{40}",expected)),"INVALID_EXPECTED_SHA")
    actual=checked_git("rev-parse","HEAD")
    demand(actual==expected,"EXACT_PROJECT_SHA_MISMATCH")
    demand(not checked_git("status","--porcelain","--untracked-files=all"),
           "DIRTY_PROJECT_SOURCE")
    ancestor=subprocess.run(["git","-C",str(ROOT),"merge-base","--is-ancestor",
                             BASE_SHA,"HEAD"],capture_output=True,timeout=12,
                             check=False)
    demand(ancestor.returncode==0,"P9_SOURCE_ANCESTRY_MISMATCH")
    raw=(ROOT/"source_pins.json").read_bytes()
    demand(hashlib.sha256(raw).hexdigest()==custody.R3_PINS_SHA256,
           "SOURCE_PINS_HASH_DRIFT")
    pinned=json.loads(raw)
    demand(pinned["schema"]=="ttt-hybrid.official-source-pins.v1",
           "SOURCE_PIN_SCHEMA_DRIFT")
    protocol=json.loads((ROOT/FROZEN_MANIFEST).read_text(encoding="utf-8"))
    demand(protocol["science"]["authorized"] is False
           and protocol["science"]["first_use_consumed"] is False
           and protocol["science"]["world_seeds"]=={},
           "SCIENCE_SOURCE_FIRST_USE_PROHIBITED")
    root=environment_root()
    demand(root.is_dir(),"EXISTING_PROJECT_R3_ENV_ABSENT")
    sources={}
    for key in ("mamba2","ttt_linear"):
        try:
            sources[key]=custody.source_custody(root/"sources"/key,pinned["sources"][key])
        except Exception:
            raise AdmissionBlocked("R3_PINNED_"+key.upper()+"_SOURCE_DRIFT") from None
    python=root/"venv"/"bin"/"python"
    demand(python.is_file(),"EXISTING_R3_PYTHON_ABSENT")
    return {
        "candidate_sha":actual,
        "original_g7_r3_sha":custody.R3_SOURCE_SHA,
        "source_pins_sha256":custody.R3_PINS_SHA256,
        "official_sources_verified":{key:{"sha":value["sha"],"clean":value["clean"]}
                                      for key,value in sources.items()},
        "existing_python_present":True
    }


def _smi():
    binary=shutil.which("nvidia-smi")
    if not binary:
        special=Path("/usr/lib/wsl/lib/nvidia-smi")
        binary=str(special) if special.is_file() else None
    demand(binary is not None,"EXISTING_NVIDIA_SMI_UNAVAILABLE")
    try:
        query=subprocess.run(
            [binary,"--query-gpu=name,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True,text=True,check=False,timeout=12)
        demand(query.returncode==0,"NVIDIA_SMI_QUERY_FAILED")
        rows=[x.strip() for x in query.stdout.splitlines() if x.strip()]
        demand(len(rows)==1,"REQUIRE_ONE_GPU")
        fields=[v.strip() for v in rows[0].split(",")]
        demand(len(fields)==4,"GPU_METADATA_INVALID")
        name=fields[0]
        total,free,util=map(int,fields[1:])
        demand("RTX 5070 Ti Laptop GPU" in name,"GPU_DEVICE_DRIFT")
        demand(0<free<=total,"GPU_MEMORY_INVALID")
        demand(free>=MIN_FREE_GIB*1024,"GPU_FREE_VRAM_BELOW_MINIMUM")
        demand(util<=MAX_GPU_UTILIZATION_PERCENT,"GPU_ALREADY_BUSY")
        processes=subprocess.run(
            [binary,"--query-compute-apps=pid","--format=csv,noheader,nounits"],
            capture_output=True,text=True,check=False,timeout=12)
        # nvidia-smi on some WSL builds cannot query compute processes. Fail
        # closed instead of inferring isolation from free memory alone.
        demand(processes.returncode==0,"GPU_OCCUPANCY_UNVERIFIABLE")
        occupied=[line for line in processes.stdout.splitlines()
                  if line.strip() and "no running" not in line.lower()]
        demand(len(occupied)<=MAX_COMPUTE_PROCESSES,"GPU_COMPUTE_OCCUPIED")
        return {"name":name,"total_mib":total,"free_mib":free,
                "gpu_utilization_percent":util,"active_compute_processes":len(occupied)}
    except (ValueError,OSError,subprocess.TimeoutExpired):
        raise AdmissionBlocked("GPU_READONLY_ADMISSION_UNAVAILABLE") from None


def verify_existing_packages(python:Path)->dict:
    code=("import importlib.metadata as m,json,hashlib;"
          "d=sorted((str(x.metadata.get('Name','')).lower().replace('_','-'),"
          "str(x.version)) for x in m.distributions());"
          "print(json.dumps({'sha':hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest(),"
          "'torch':dict(d).get('torch'),'triton':dict(d).get('triton')},sort_keys=True))")
    try:
        output=subprocess.run([str(python),"-I","-B","-c",code],
                              capture_output=True,text=True,timeout=20,check=False)
        demand(output.returncode==0,"EXISTING_R3_PACKAGE_METADATA_FAILED")
        parsed=json.loads(output.stdout)
        demand(parsed["sha"]==custody.R3_MANIFEST_SHA256,
               "EXISTING_R3_PACKAGE_GRAPH_DRIFT")
        demand(parsed["torch"]==EXPECTED_TORCH
               and parsed["triton"]==EXPECTED_TRITON,
               "EXISTING_R3_TORCH_TRITON_DRIFT")
        return {"sha256":parsed["sha"],"torch":parsed["torch"],
                "triton":parsed["triton"]}
    except (ValueError,KeyError,OSError,subprocess.TimeoutExpired):
        raise AdmissionBlocked("EXISTING_R3_PACKAGE_METADATA_UNREADABLE") from None


def child_benchmark(expected_sha:str)->dict:
    demand(sys.version_info[:2]==(3,12),"CHILD_PYTHON_VERSION_DRIFT")
    demand(checked_git("rev-parse","HEAD")==expected_sha,"CHILD_PROJECT_SHA_DRIFT")
    digest,installed=custody.manifest()
    demand(digest==custody.R3_MANIFEST_SHA256
           and installed.get("torch")==EXPECTED_TORCH
           and installed.get("triton")==EXPECTED_TRITON,
           "CHILD_R3_PACKAGE_GRAPH_DRIFT")
    import torch
    demand(torch.__version__==EXPECTED_TORCH
           and torch.version.cuda=="12.8"
           and torch.cuda.is_available(),"CHILD_TORCH_CUDA_UNAVAILABLE")
    demand(torch.cuda.device_count()==1,"CHILD_GPU_COUNT_DRIFT")
    demand(list(torch.cuda.get_device_capability(0))==EXPECTED_CAPABILITY,
           "CHILD_CUDA_CAPABILITY_DRIFT")
    name=torch.cuda.get_device_name(0)
    demand("RTX 5070 Ti Laptop GPU" in name,"CHILD_GPU_NAME_DRIFT")
    free,total=torch.cuda.mem_get_info(0)
    demand(free>=MIN_FREE_GIB*1024**3,"CHILD_INSUFFICIENT_FREE_GPU_MEMORY")
    torch.cuda.set_per_process_memory_fraction(
        min(1.0,MAX_TORCH_ALLOCATED_MIB*1024**2/total),device=0)
    torch.cuda.reset_peak_memory_stats(0)
    from scripts.native_p9_parallel_refinement_dev import run
    result=run("cuda",STEPS,BATCH,REPETITIONS)
    peak=torch.cuda.max_memory_allocated(0)
    demand(peak<=MAX_TORCH_ALLOCATED_MIB*1024**2,"CUDA_PYTORCH_PEAK_OVER_BUDGET")
    demand(result["status"]=="PASS_DEV_INTEGRITY_NOT_SCIENCE"
           and result["gpu_benchmark_verified"] is True
           and len(result["records"])==3
           and result["scientific_first_use"]=="NOT_RUN"
           and result["model_gradient_updates"]==0,
           "P9_GPU_MECHANISM_INTEGRITY_FAIL")
    return {
        "status":"PASS_P9_R3_GPU_DEVELOPMENT_MEASUREMENT",
        "device":name,"gpu_capability":EXPECTED_CAPABILITY,
        "cuda_build":torch.version.cuda,"torch":torch.__version__,
        "peak_torch_allocated_bytes":peak,
        "torch_allocator_ceiling_mib":MAX_TORCH_ALLOCATED_MIB,
        "gpu_free_before_child_bytes":free,
        "measured":result,
        "Mamba2_native_kernel":"NOT_RUN",
        "source_p9_model_modified":False,
        "scientific_first_use":"NOT_RUN",
        "unfamiliar_world_transfer":"NOT_TESTED"
    }


def main()->int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--expected-sha",required=True)
    parser.add_argument("--receipt",type=Path,required=True)
    parser.add_argument("--child",action="store_true")
    args=parser.parse_args()
    receipt=args.receipt.resolve()
    report={
        "schema":SCHEMA,"candidate_sha":args.expected_sha,
        "time_utc":datetime.now(timezone.utc).isoformat(),
        "status":"BLOCKED_NOT_RUN",
        "stage":"CHILD_BENCHMARK" if args.child else "READONLY_ADMISSION",
        "no_installs":True,"no_gpu_training":True,"no_new_runner":True,
        "scientific_first_use":"NOT_RUN","science_seeds_consumed":[],
        "gpu_benchmark":"NOT_RUN","device":"UNVERIFIED",
        "fused_mamba2":"NOT_RUN","native_substrate_unchanged":True
    }
    rc=2
    try:
        demand(sys.platform.startswith("linux") and platform.machine()=="x86_64"
               and os.geteuid()!=0,"WSL_NONROOT_X64_REQUIRED")
        demand("microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower(),
               "REQUIRE_EXISTING_WSL_UBUNTU")
        demand(os.environ.get("NATIVE_P9_DEV_ONLY","1")=="1",
               "SCIENCE_FIRST_USE_DENIED")
        if args.child:
            report.update(child_benchmark(args.expected_sha))
            report["gpu_benchmark"]="COMPLETED_GPU_DEV"
        else:
            provenance=verify_lineage(args.expected_sha)
            report["source"]=provenance
            report["runtime"]=verify_existing_packages(environment_root()/"venv/bin/python")
            report["gpu_inventory"]=_smi()
            report["stage"]="GPU_BENCHMARK_ADMITTED"
            report["status"]="PASS_READONLY_EXACT_CUDA_ENV_ADMISSION"
            # A fresh, bounded child, only after every read-only source/runtime
            # and GPU occupancy check passes. No fallback to CPU.
            try:
                completed=subprocess.run([
                    str(environment_root()/"venv/bin/python"),"-I","-B",
                    str(ROOT/"scripts/native_p9_existing_cuda_dev.py"),
                    "--expected-sha",args.expected_sha,"--receipt",
                    str(receipt.with_name("native_p9_gpu_inner.json")),"--child"],
                    capture_output=True,text=True,timeout=MAX_CHILD_WALL_SECONDS,
                    env={**os.environ,"NATIVE_P9_DEV_ONLY":"1",
                         "CUDA_MODULE_LOADING":"LAZY",
                         "PYTHONDONTWRITEBYTECODE":"1",
                         "EXPECTED_CANDIDATE_SHA":args.expected_sha},
                    check=False)
            except (OSError,subprocess.TimeoutExpired):
                raise AdmissionBlocked("BOUNDED_GPU_CHILD_UNAVAILABLE_OR_TIMEOUT") from None
            child_path=receipt.with_name("native_p9_gpu_inner.json")
            demand(child_path.is_file(),"GPU_CHILD_RECEIPT_MISSING")
            child=json.loads(child_path.read_text())
            report["gpu_child_receipt_sha256"]=hashlib.sha256(child_path.read_bytes()).hexdigest()
            report["gpu_child_status"]=child.get("status")
            report["gpu_benchmark"]=child.get("gpu_benchmark","NOT_RUN")
            if completed.returncode!=0 or child.get("status")!="PASS_P9_R3_GPU_DEVELOPMENT_MEASUREMENT":
                raise AdmissionBlocked("GPU_CHILD_NOT_QUALIFIED")
            report["gpu_result"]=child["measured"]
            report["gpu_peak_torch_allocated_bytes"]=child["peak_torch_allocated_bytes"]
            report["gpu_benchmark"]="COMPLETED_GPU_DEV"
            report["status"]="PASS_P9_EXISTING_R3_CUDA_BENCHMARK_DEV_ONLY"
        rc=0
    except AdmissionBlocked as exc:
        report["status"]="BLOCKED_"+str(exc)
    except Exception as exc:
        report["status"]="FAIL_UNEXPECTED_"+type(exc).__name__
        rc=1
    finally:
        receipt.parent.mkdir(parents=True,exist_ok=True)
        receipt.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
        # Print only allowlisted classifications: no raw environment/errors.
        print("P9_EXISTING_CUDA_GATE="+report["status"],flush=True)
    return rc


if __name__=="__main__":
    raise SystemExit(main())
