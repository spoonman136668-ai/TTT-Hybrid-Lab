"""Read-only, fail-closed gate for *authentic* upstream Mamba2 and TTT-LM.

No installs, fetching, dispatches, training or source modifications are performed.
A PASS means prerequisite source/runtime admission only; it is NOT scientific
qualification, hybrid integration, or learning/throughput evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parent


def git_read(repo: Path, *args: str) -> tuple[bool, str]:
    try:
        p = subprocess.run(["git", "-C", str(repo), *args], text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=12, check=False)
        return p.returncode == 0, p.stdout.strip() if p.returncode == 0 else p.stderr.strip()
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return False, type(exc).__name__


def source_check(repo: Path, pin: dict) -> dict:
    """Require *exact* source commit, clean tree, and Git blob identity."""
    observed = {"root": str(repo.resolve()), "required_commit": pin["sha"]}
    ok, rev = git_read(repo, "rev-parse", "HEAD")
    observed["actual_commit"] = rev if ok else None
    if not ok or rev != pin["sha"]:
        return {**observed, "pass": False, "reason": "SHA_MISMATCH_OR_UNREADABLE"}
    ok, dirty = git_read(repo, "status", "--porcelain", "--untracked-files=all")
    if not ok or dirty:
        return {**observed, "pass": False, "reason": "DIRTY_OR_UNREADABLE_TREE"}
    for file in pin["required_files"]:
        target = repo / file
        if not target.is_file():
            return {**observed, "pass": False, "reason": "REQUIRED_FILE_MISSING", "file": file}
        ok, blob = git_read(repo, "rev-parse", f"HEAD:{file}")
        expected_blob = pin["observed_blob_sha"].get(file)
        if not ok or blob != expected_blob:
            return {**observed, "pass": False, "reason": "BLOB_MISMATCH", "file": file,
                    "actual_blob": blob if ok else None}
    return {**observed, "pass": True, "reason": "EXACT_SOURCE_CLEAN"}


def module_location_check(module: str, checkout: Path) -> dict:
    """Do not silently accept an unrelated wheel/site-packages installation."""
    try:
        spec = importlib.util.find_spec(module)
        origin = None if spec is None else spec.origin
    except (ImportError, ValueError, AttributeError) as exc:
        origin = None
    if not origin or origin in {"built-in", "frozen"}:
        return {"pass": False, "reason": "MODULE_UNAVAILABLE", "module": module}
    path = Path(origin).resolve()
    root = checkout.resolve()
    return {"pass": root in path.parents, "reason": "MATCHED_CHECKOUT" if root in path.parents else "UNPINNED_INSTALL",
            "origin": str(path), "module": module}


def hardware_check() -> dict:
    result = {"pass": False, "platform": platform.platform(), "python": platform.python_version()}
    if not sys.platform.startswith("linux"):
        result["reason"] = "LINUX_REQUIRED_FOR_UPSTREAM_MAMBA_GPU"
        return result
    try:
        import torch
        result["torch"] = torch.__version__
        result["cuda_available"] = bool(torch.cuda.is_available())
        if not result["cuda_available"]:
            result["reason"] = "CUDA_UNAVAILABLE"
            return result
        result["gpu"] = torch.cuda.get_device_name(0)
        result["gpu_vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
    except (ImportError, RuntimeError, OSError) as exc:
        result["reason"] = "TORCH_CUDA_PROBE_FAILED"
        result["exception_type"] = type(exc).__name__
        return result
    result["pass"] = True
    result["reason"] = "LINUX_CUDA_PRESENT"
    return result


def evaluate(mamba_root: Path, ttt_root: Path, *, hardware=None, module_probe=module_location_check) -> dict:
    pins_path = BASE / "source_pins.json"
    raw = pins_path.read_bytes()
    pins = json.loads(raw)
    if pins.get("schema") != "ttt-hybrid.official-source-pins.v1":
        raise ValueError("Unknown source pin schema")
    results = {}
    for label, root in (("mamba2", mamba_root), ("ttt_linear", ttt_root)):
        source = pins["sources"][label]
        results[label] = {
            "source": source_check(root, source),
            "import_origin": module_probe(source["import_module"], root),
        }
    results["hardware"] = hardware if hardware is not None else hardware_check()
    passed = all(value[part]["pass"] for key, value in results.items()
                 if key != "hardware" for part in ("source", "import_origin")) and results["hardware"]["pass"]
    return {"schema": "ttt-hybrid.authentic-preflight.v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_pins_sha256": hashlib.sha256(raw).hexdigest(),
            "status": "PASS_SOURCE_RUNTIME_ADMISSION_ONLY" if passed else "BLOCKED",
            "scientific_qualification": "UNVERIFIED",
            "authentic_hybrid_forward_test": "NOT_RUN",
            "checks": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mamba-root", type=Path, required=True)
    parser.add_argument("--ttt-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, default=BASE / "authentic_preflight_receipt.json")
    args = parser.parse_args()
    record = evaluate(args.mamba_root, args.ttt_root)
    args.receipt.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0 if record["status"].startswith("PASS") else 2

if __name__ == "__main__":
    sys.exit(main())
