"""Run the standalone authentic-source CUDA gate; fail closed before imports."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import traceback
from datetime import datetime, timezone
import torch
from authentic_preflight import evaluate
from authentic_hybrid import AuthenticHybrid, integration_checks, forward_only_checks, load_official_classes

BASE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(mamba_root: Path, ttt_root: Path, *, receipt: Path, forward_only: bool = False):
    preflight = evaluate(mamba_root, ttt_root)
    record = {
        "schema":"ttt-hybrid.authentic-gpu-smoke.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runner_kind":"standalone_manual_or_existing_authorized_runner",
        "source_pins_sha256": sha(BASE / "source_pins.json"),
        "script_sha256": {p:sha(BASE / p) for p in ("authentic_hybrid.py", "run_gate3.py")},
        "environment":{"platform":platform.platform(),
                       "python":platform.python_version(),
                       "torch":torch.__version__,
                       "cuda_available":bool(torch.cuda.is_available()),
                       "torch_cuda_build":torch.version.cuda},
        "preflight":preflight,
        "status":"BLOCKED_SOURCE_RUNTIME_ADMISSION",
        "mode":"FORWARD_ONLY_NO_GRAD" if forward_only else "FULL_ENGINEERING_INTEGRATION",
        "authentic_forward":"NOT_RUN",
        "scientific_qualification":"UNVERIFIED",
        "model_quality_claim":"NONE",
        "production_promotion":"NOT_AUTHORIZED"
    }
    if preflight["status"] != "PASS_SOURCE_RUNTIME_ADMISSION_ONLY":
        receipt.write_text(json.dumps(record,indent=2)+"\n")
        return record, 2
    if not torch.cuda.is_available():
        record["status"]="BLOCKED_CUDA_UNAVAILABLE"
        receipt.write_text(json.dumps(record,indent=2)+"\n")
        return record, 2
    try:
        classes = load_official_classes(mamba_root, ttt_root)
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        record["environment"]["gpu"] = torch.cuda.get_device_name(device)
        record["environment"]["device_memory_bytes"] = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.reset_peak_memory_stats(device)
        model = AuthenticHybrid(official_classes=classes)
        if forward_only:
            record["checks"] = forward_only_checks(model, device=device)
        else:
            record["checks"] = integration_checks(model, device=device)
        torch.cuda.synchronize(device)
        record["environment"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        record["status"]=("PASS_AUTHENTIC_GPU_FORWARD_ONLY"
                          if forward_only else "PASS_AUTHENTIC_GPU_ENGINEERING_SMOKE_ONLY")
        record["authentic_forward"]="PASS"
        return record, 0
    except Exception as exc:
        record["status"]="FAIL_AUTHENTIC_GPU_INTEGRATION"
        record["error_type"]=type(exc).__name__
        record["error_message"]=str(exc)[:500]
        record["traceback_tail"]=traceback.format_exc().splitlines()[-10:]
        return record, 1
    finally:
        receipt.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--forward-only', action='store_true',
                   help='No-grad finite forward, causal/reset controls, no backward or training')
    p.add_argument('--mamba-root',required=True,type=Path)
    p.add_argument('--ttt-root',required=True,type=Path)
    p.add_argument('--receipt',type=Path,default=BASE/'gate3_authentic_receipt.json')
    args=p.parse_args()
    result, code=run(args.mamba_root,args.ttt_root,receipt=args.receipt,
                     forward_only=args.forward_only)
    print(json.dumps({"status":result["status"],"receipt":str(args.receipt)}))
    return code

if __name__=='__main__':
    sys.exit(main())
