"""P9 G2: one-use original R3 environment/GPU admission for controlled DEV measurement.

Standard-library outer process. Reuse exact G1 source custody and occupancy
checks; no new installs, extensions, models, runner, controllers, first-use or
retries. A CUDA-enabled child is forked only after admission.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import native_p9_existing_cuda_dev as g1

SCHEMA="ttt-hybrid.native-p9.g2.existing-r3-gpu-run.v1"
PARENT_SHA="cda3885e15d3ecee91b6a55931849002543ffb15"
FROZEN="experiments/native_p9_gpu_randomized_dev_frozen.json"
CHILD_TIMEOUT_SECONDS=150
MAX_PYTORCH_MIB=256


class Blocked(RuntimeError):
    pass


def guard(ok,reason):
    if not ok:
        raise Blocked(reason)


def frozen_admission()->dict:
    m=json.loads((ROOT/FROZEN).read_text())
    guard(m["schema"]=="ttt-hybrid.native-p9.g2.randomized-cuda-frozen.v1",
          "FROZEN_SCHEMA_DRIFT")
    guard(m["science"]["authorized"] is False and
          m["science"]["first_use_consumed"] is False and
          m["science"]["world_seeds"]=={},"SCIENTIFIC_FIRST_USE_DENIED")
    p=m["design"]
    guard(m["source"]["parent_g1_sha"]==PARENT_SHA and
          p["rounds"]==24 and p["warmups"]==3 and
          p["speed_ratio_max"]==.75 and p["untrained_MSE_ratio_max"]==1.01 and
          p["source_p9_unchanged"] is True and
          p["max_child_seconds"]==CHILD_TIMEOUT_SECONDS,
          "FROZEN_HYPOTHESIS_OR_BUDGET_DRIFT")
    return {"protocol_schema":m["schema"],"rounds":p["rounds"],
            "wall_speed_ratio_goal":p["speed_ratio_max"],
            "mse_ratio_goal":p["untrained_MSE_ratio_max"]}


def _check_root(expected):
    guard(bool(re.fullmatch(r"[0-9a-f]{40}",expected)),"EXPECTED_SHA_INVALID")
    guard(g1.checked_git("rev-parse","HEAD")==expected,"EXACT_SOURCE_DRIFT")
    guard(not g1.checked_git("status","--porcelain","--untracked-files=all"),
          "CHECKOUT_DIRTY")
    try:
        a=subprocess.run(["git","-C",str(ROOT),"merge-base",
                          "--is-ancestor",PARENT_SHA,"HEAD"],
                          capture_output=True,timeout=15,check=False)
        guard(a.returncode==0,"PARENT_G1_ANCESTRY_DRIFT")
    except (OSError,subprocess.TimeoutExpired):
        raise Blocked("SOURCE_ANCESTRY_UNVERIFIED") from None


def cuda_child(expected):
    _check_root(expected)
    frozen_admission()
    import torch
    digest,packages=g1.custody.manifest()
    guard(digest==g1.custody.R3_MANIFEST_SHA256 and
          packages.get("torch")==g1.EXPECTED_TORCH and
          packages.get("triton")==g1.EXPECTED_TRITON,
          "EXISTING_R3_PACKAGE_DRIFT")
    guard(torch.__version__==g1.EXPECTED_TORCH and
          torch.version.cuda=="12.8" and torch.cuda.is_available()
          and torch.cuda.device_count()==1,"EXISTING_CUDA_NOT_QUALIFIED")
    device=torch.device("cuda:0")
    guard(list(torch.cuda.get_device_capability(device))==[12,0]
          and "RTX 5070 Ti Laptop GPU" in torch.cuda.get_device_name(device),
          "GPU_SOURCE_IDENTITY_DRIFT")
    free,total=torch.cuda.mem_get_info(0)
    guard(free>=8*1024**3,"GPU_FREE_MEMORY_NOT_ADMITTED")
    torch.cuda.set_per_process_memory_fraction(
        min(1.,MAX_PYTORCH_MIB*1024**2/total),device=0)
    torch.cuda.reset_peak_memory_stats(device)
    from scripts.native_p9_gpu_randomized_dev import run
    result=run()
    peak=torch.cuda.max_memory_allocated(device)
    guard(peak<=MAX_PYTORCH_MIB*1024**2,"PYTORCH_VRAM_CAP_EXCEEDED")
    guard(result["status"]=="PASS_GPU_DEV_INTEGRITY_NOT_SCIENCE"
          and result["total_measurements"]==288
          and len(result["records"])==3
          and result["scientific_first_use"]=="NOT_RUN"
          and result["scientific_world_seeds_consumed"]==[],
          "GPU_MEASUREMENT_SOURCE_OR_INTEGRITY_FAILED")
    return {"device":torch.cuda.get_device_name(device),
            "torch":torch.__version__,"cuda":torch.version.cuda,
            "capability":[12,0],"initial_free_gpu_bytes":free,
            "peak_torch_allocated_bytes":peak,
            "peak_torch_limit_mib":MAX_PYTORCH_MIB,
            "result":result}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--expected-sha",required=True)
    ap.add_argument("--receipt",required=True,type=Path)
    ap.add_argument("--child",action="store_true")
    a=ap.parse_args()
    path=a.receipt.resolve()
    record={
        "schema":SCHEMA,
        "time_utc":datetime.now(timezone.utc).isoformat(),
        "source_sha":a.expected_sha,"status":"BLOCKED_NOT_RUN",
        "stage":"GPU_CHILD" if a.child else "READONLY_GPU_ADMISSION",
        "no_installation":True,"no_model_training":True,
        "new_controller":False,"source_native_modified":False,
        "scientific_first_use":"NOT_RUN",
        "science_world_seeds_consumed":[],
        "gpu_measurement":"NOT_RUN",
        "fused_Mamba2":"NOT_RUN",
        "MoD_production":"NOT_RUN",
        "FORGE_integrated":False,
        "original_S26_S27_untouched":True,
    }
    code=2
    try:
        guard(sys.platform.startswith("linux") and
              platform.machine()=="x86_64" and os.geteuid()!=0,
              "REQUIRE_EXISTING_NONROOT_WSL_X64")
        guard("microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower(),
              "REQUIRE_EXISTING_UBUNTU_WSL")
        guard(os.environ.get("NATIVE_P9_DEV_ONLY","1")=="1",
              "SCIENCE_SOURCE_PROHIBITED")
        _check_root(a.expected_sha)
        record["frozen"]=frozen_admission()
        if a.child:
            record.update(cuda_child(a.expected_sha))
            record["status"]="PASS_P9_G2_GPU_DEVELOPMENT_INTEGRITY_ONLY"
            record["gpu_measurement"]="COMPLETED_CUDA_EVENTS_AND_WALL"
        else:
            # Reuse complete original G1 source and current package checks.
            record["original_source"]=g1.verify_lineage(a.expected_sha)
            record["existing_cuda_runtime"]=g1.verify_existing_packages(
                g1.environment_root()/"venv"/"bin"/"python")
            record["gpu_inventory_before"]=g1._smi()
            record["stage"]="SOURCE_AND_GPU_OCCUPANCY_ADMITTED"
            inner_path=path.with_name("native_p9_g2_inner.json")
            guard(not inner_path.exists(),"EXISTING_GPU_CHILD_RECEIPT_REPLAY_BLOCKED")
            try:
                proc=subprocess.run([
                    str(g1.environment_root()/"venv"/"bin"/"python"),"-I","-B",
                    str(ROOT/"scripts"/"native_p9_gpu_randomized_once.py"),
                    "--expected-sha",a.expected_sha,
                    "--receipt",str(inner_path),"--child"],
                    capture_output=True,text=True,timeout=CHILD_TIMEOUT_SECONDS,
                    env={**os.environ,"NATIVE_P9_DEV_ONLY":"1",
                         "CUDA_MODULE_LOADING":"LAZY",
                         "PYTHONDONTWRITEBYTECODE":"1",
                         "EXPECTED_CANDIDATE_SHA":a.expected_sha,
                         "OMP_NUM_THREADS":"1"},
                    check=False)
            except (OSError,subprocess.TimeoutExpired):
                raise Blocked("GPU_CHILD_TIMEOUT_OR_UNAVAILABLE") from None
            guard(inner_path.is_file(),"GPU_CHILD_RECEIPT_MISSING")
            raw=inner_path.read_bytes()
            inner=json.loads(raw)
            record["gpu_child_receipt_sha256"]=hashlib.sha256(raw).hexdigest()
            record["gpu_child_status"]=inner.get("status")
            record["gpu_measurement"]=inner.get("gpu_measurement","NOT_RUN")
            guard(proc.returncode==0 and
                  inner.get("status")=="PASS_P9_G2_GPU_DEVELOPMENT_INTEGRITY_ONLY" and
                  inner.get("source_sha")==a.expected_sha,
                  "GPU_CHILD_INTEGRITY_NOT_QUALIFIED")
            record["gpu_result"]=inner["result"]
            record["peak_torch_allocated_bytes"]=inner["peak_torch_allocated_bytes"]
            # Post-run read-only occupancy: important for shared-hardware
            # contention diagnosis; cannot silently dismiss an unknown.
            record["gpu_inventory_after"]=g1._smi()
            record["status"]="PASS_P9_G2_ORIGINAL_R3_CUDA_DEV_ONLY"
            record["gpu_measurement"]="COMPLETED_CUDA_EVENTS_AND_WALL"
        code=0
    except (Blocked,g1.AdmissionBlocked,g1.custody.Blocked) as exc:
        record["status"]="BLOCKED_"+str(exc)
    except Exception as exc:
        record["status"]="FAIL_UNEXPECTED_"+type(exc).__name__
        code=1
    finally:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
        print("P9_G2_EXACT_GPU_STATUS="+record["status"],flush=True)
    return code

if __name__=="__main__":
    raise SystemExit(main())
