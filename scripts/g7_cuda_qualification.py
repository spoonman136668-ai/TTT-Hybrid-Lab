"""G7: exact-source authentic Mamba2+TTTLinear CUDA qualification.

This is an opt-in standalone native GPU engineering runner, NOT a model
benchmark, learning claim, or host-installation tool. It never installs
dependencies or starts competing jobs/controllers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def admission_policy(*, linux: bool, cuda: bool, free_bytes: int,
                     total_bytes: int, min_free_gib: float,
                     max_peak_gib: float) -> dict:
    """Pure, portable negative-control gate; no GPU is touched here."""
    if min_free_gib < 2 or max_peak_gib <= 0 or max_peak_gib > min_free_gib:
        raise ValueError("INVALID_RESOURCE_BOUNDS")
    if not linux:
        reason = "REQUIRES_LINUX_OR_WSL2"
    elif not cuda:
        reason = "CUDA_NOT_AVAILABLE"
    elif free_bytes <= 0 or total_bytes <= 0 or free_bytes > total_bytes:
        reason = "INVALID_GPU_MEMORY_PROBE"
    elif free_bytes < min_free_gib * (1024**3):
        reason = "INSUFFICIENT_FREE_VRAM"
    else:
        reason = "HOST_ADMITTED"
    return {
        "pass": reason == "HOST_ADMITTED", "reason": reason,
        "available_bytes": free_bytes, "total_bytes": total_bytes,
        "min_free_gib": min_free_gib, "max_peak_gib": max_peak_gib,
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            part = f.read(1024 * 1024)
            if not part:
                break
            h.update(part)
    return h.hexdigest()


def source_module_receipt(module: str, expected_path: Path) -> dict:
    imported = importlib.import_module(module)
    observed = Path(inspect.getfile(imported)).resolve()
    expected_path = expected_path.resolve()
    if observed != expected_path:
        raise RuntimeError("UNPINNED_MODULE_ORIGIN:" + module)
    return {
        "module": module, "path": str(observed),
        "sha256": file_sha256(observed), "exact_source_path": True,
    }


def optional_extension(module_name: str) -> dict:
    """Record linked extension custody, never claim it is qualified here."""
    try:
        mod = importlib.import_module(module_name)
        file = getattr(mod, "__file__", None)
        origin = Path(file).resolve() if file else None
        return {"module": module_name, "status": "IMPORTABLE",
                "origin": str(origin) if origin else None,
                "sha256": file_sha256(origin) if origin and origin.is_file() else None}
    except (ImportError, RuntimeError, OSError) as exc:
        return {"module": module_name, "status": "UNAVAILABLE",
                "error_type": type(exc).__name__}


def run(mamba_root: Path, ttt_root: Path, *, receipt: Path,
        min_free_gib: float, max_peak_gib: float) -> tuple[dict, int]:
    started = time.perf_counter()
    mamba_root = mamba_root.resolve()
    ttt_root = ttt_root.resolve()
    receipt = receipt.resolve()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "ttt-hybrid.g7-native-cuda-qualification.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project_sha": None,
        "source_pins_sha256": file_sha256(ROOT / "source_pins.json"),
        "platform": platform.platform(),
        "status": "BLOCKED_NOT_RUN",
        "authentic_integration": "NOT_RUN",
        "scientific_qualification": "UNVERIFIED",
        "deployment": "NOT_AUTHORIZED",
        "resource_limit_gib": max_peak_gib,
        "binary_provenance_qualification": "PARTIAL_UNVERIFIED",
    }
    code = 2
    try:
        ref = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             text=True, capture_output=True, timeout=12, check=False)
        observed = ref.stdout.strip() if ref.returncode == 0 else None
        record["project_sha"] = observed
        if not observed or (len(observed) != 40):
            raise RuntimeError("PROJECT_SHA_UNREADABLE")
        import torch
        record["torch"] = torch.__version__
        record["torch_cuda_build"] = torch.version.cuda
        record["torch_hip_build"] = torch.version.hip
        cuda = torch.cuda.is_available()
        free, total = (torch.cuda.mem_get_info(0) if cuda else (0, 0))
        record["admission"] = admission_policy(
            linux=sys.platform.startswith("linux"), cuda=cuda,
            free_bytes=int(free), total_bytes=int(total),
            min_free_gib=min_free_gib, max_peak_gib=max_peak_gib,
        )
        if not record["admission"]["pass"]:
            record["status"] = "BLOCKED_"+record["admission"]["reason"]
            return record, 2

        from authentic_preflight import evaluate
        preflight = evaluate(mamba_root, ttt_root)
        record["preflight"] = preflight
        if preflight["status"] != "PASS_SOURCE_RUNTIME_ADMISSION_ONLY":
            record["status"] = "BLOCKED_PINNED_SOURCE_ADMISSION"
            return record, 2

        record["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "free_memory_before_bytes": free,
            "torch_arch_list": torch.cuda.get_arch_list(),
        }
        # Resolve Python kernels from the exact source roots, not wheels.
        record["modules"] = [
            source_module_receipt(
                "mamba_ssm.modules.mamba2",
                mamba_root / "mamba_ssm/modules/mamba2.py"),
            source_module_receipt(
                "mamba_ssm.ops.triton.ssd_combined",
                mamba_root / "mamba_ssm/ops/triton/ssd_combined.py"),
            source_module_receipt(
                "mamba_ssm.ops.triton.layernorm_gated",
                mamba_root / "mamba_ssm/ops/triton/layernorm_gated.py"),
            source_module_receipt("ttt", ttt_root / "ttt.py"),
        ]
        record["extensions"] = [
            optional_extension("triton"),
            optional_extension("causal_conv1d_cuda"),
            optional_extension("selective_scan_cuda"),
        ]

        # Force an actual CUDA operation before the integrated smoke.
        torch.cuda.set_device(0)
        a = torch.randn((16, 16), device="cuda", dtype=torch.float32)
        sample = (a @ a).sum()
        torch.cuda.synchronize(0)
        if not torch.isfinite(sample).item():
            raise RuntimeError("CUDA_KERNEL_NONFINITE")

        from run_gate3 import run as native_gate3
        inner = receipt.with_name("g7_native_gate3_subreceipt.json")
        inner_result, inner_code = native_gate3(
            mamba_root, ttt_root, receipt=inner)
        record["native_gate3_status"] = inner_result.get("status")
        record["native_gate3_receipt_sha256"] = (
            file_sha256(inner) if inner.is_file() else None)
        record["native_checks"] = inner_result.get("checks")
        record["authentic_integration"] = inner_result.get("authentic_forward")
        if inner_code:
            record["status"] = (
                "BLOCKED_NATIVE_GATE3" if inner_code == 2
                else "FAIL_NATIVE_GATE3_INTEGRATION"
            )
            return record, inner_code
        peak = torch.cuda.max_memory_allocated(0)
        record["gpu"]["peak_allocated_bytes"] = peak
        if peak > max_peak_gib * (1024**3):
            record["status"] = "FAIL_RESOURCE_CEILING_EXCEEDED"
            return record, 1
        record["status"] = "PASS_NATIVE_CUDA_ENGINEERING_ONLY"
        return record, 0
    except Exception as exc:
        record["status"] = "FAIL_NATIVE_CUDA_INTEGRATION"
        record["error_type"] = type(exc).__name__
        record["error_message"] = str(exc)[:450]
        record["traceback_tail"] = traceback.format_exc().splitlines()[-10:]
        code = 1
        return record, code
    finally:
        record["elapsed_seconds"] = round(time.perf_counter()-started, 3)
        receipt.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mamba-root", type=Path, required=True)
    parser.add_argument("--ttt-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path,
                        default=ROOT/"local_runs"/"g7_native_cuda.json")
    parser.add_argument("--min-free-gib", type=float, default=4.0)
    parser.add_argument("--max-peak-gib", type=float, default=2.0)
    args=parser.parse_args()
    if args.min_free_gib < 2 or args.max_peak_gib <= 0 or args.max_peak_gib > args.min_free_gib:
        parser.error("Resource policy requires 2+ GiB free and 0 < peak <= free budget")
    result, code=run(args.mamba_root,args.ttt_root,
                     receipt=args.receipt,
                     min_free_gib=args.min_free_gib,
                     max_peak_gib=args.max_peak_gib)
    print(json.dumps({"status":result["status"], "project_sha":result["project_sha"],
                      "gpu":result.get("gpu"), "receipt":str(args.receipt),
                      "error":result.get("error_message")}))
    return code


if __name__=="__main__":
    raise SystemExit(main())
