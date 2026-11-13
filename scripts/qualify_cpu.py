"""Evidence-bound CPU engineering qualification for the isolated TTT hybrid pilot.

Runs the repository's complete test suite and asserts authentic CUDA admission
fails closed when official upstream checkouts are deliberately absent.
This runner never installs, trains, dispatches, or promotes models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def run_command(args: list[str], *, timeout: int) -> dict:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            args, cwd=ROOT, capture_output=True, text=True,
            errors="replace", check=False, timeout=timeout,
        )
        return {
            "exit_code": proc.returncode,
            "seconds": round(time.perf_counter() - started, 3),
            "stdout_tail": proc.stdout[-3500:],
            "stderr_tail": proc.stderr[-1500:],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "exit_code": None,
            "seconds": round(time.perf_counter() - started, 3),
            "error_type": type(exc).__name__,
        }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualify(receipt_path: Path) -> dict:
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    revision = run_command(["git", "rev-parse", "HEAD"], timeout=10)
    source_sha = sha256(ROOT / "source_pins.json")
    expected_sha = os.getenv("EXPECTED_CANDIDATE_SHA")
    observed_sha = revision.get("stdout_tail", "").strip() if revision["exit_code"] == 0 else None
    identity_pass = bool(observed_sha) and (
        observed_sha == expected_sha if expected_sha else os.getenv("GITHUB_ACTIONS") != "true"
    )
    record = {
        "schema": "ttt-hybrid.cpu-qualification.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "revision": observed_sha,
        "checkout_identity": {
            "expected_sha": expected_sha,
            "observed_sha": observed_sha,
            "pass": identity_pass,
        },
        "source_pins_sha256": source_sha,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "runner": os.getenv("RUNNER_NAME", "local"),
        "github_run_id": os.getenv("GITHUB_RUN_ID"),
        "qualification_kind": "CPU_ENG_ONLY",
        "scientific_qualification": "UNVERIFIED",
        "authentic_cuda_integration": "NOT_RUN",
    }
    try:
        import torch
        record["torch"] = torch.__version__
        record["torch_cuda_build"] = torch.version.cuda
    except ImportError:
        record["torch"] = None
    record["tests"] = run_command(
        [sys.executable, "-m", "pytest", "-q", "--durations=15", "tests"], timeout=900
    )
    # An intentional negative control. There must be no upstream checkout at
    # either synthetic path, irrespective of whether the host has a GPU.
    sentinel_root = receipt_path.parent / "_qualification_missing_checkouts"
    if sentinel_root.exists():
        record["negative_control"] = {
            "pass": False, "reason": "SENTINEL_PATH_ALREADY_EXISTS"
        }
    else:
        blocked_receipt = receipt_path.parent / "expected_blocked_authentic.json"
        run = run_command([
            sys.executable, "run_gate3.py",
            "--mamba-root", str(sentinel_root / "mamba"),
            "--ttt-root", str(sentinel_root / "ttt"),
            "--receipt", str(blocked_receipt),
        ], timeout=120)
        valid = False
        observed = None
        if blocked_receipt.is_file():
            try:
                observed = json.loads(blocked_receipt.read_text(encoding="utf-8"))
                checks = observed.get("preflight", {}).get("checks", {})
                valid = (
                    run["exit_code"] == 2
                    and observed.get("status") == "BLOCKED_SOURCE_RUNTIME_ADMISSION"
                    and observed.get("authentic_forward") == "NOT_RUN"
                    and all(
                        checks.get(k, {}).get("source", {}).get("reason")
                        == "SHA_MISMATCH_OR_UNREADABLE"
                        for k in ("mamba2", "ttt_linear")
                    )
                )
            except (OSError, ValueError, TypeError, KeyError):
                valid = False
        record["negative_control"] = {
            "pass": valid,
            "probe": run,
            "observed_status": observed.get("status") if isinstance(observed, dict) else None,
        }
    record["status"] = (
        "PASS_CPU_ENGINEERING_ONLY"
        if record["tests"]["exit_code"] == 0
        and record["checkout_identity"]["pass"] is True
        and record["negative_control"].get("pass") is True
        else "FAIL_CPU_QUALIFICATION"
    )
    receipt_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receipt", type=Path,
        default=ROOT / "local_runs" / "cpu_qualification.json",
    )
    args = parser.parse_args()
    result = qualify(args.receipt.resolve())
    print(json.dumps({
        "status": result["status"],
        "revision": result["revision"],
        "tests": result["tests"].get("stdout_tail", "")[-350:],
        "negative_control_pass": result["negative_control"].get("pass"),
        "receipt": str(args.receipt),
    }, indent=2))
    return 0 if result["status"] == "PASS_CPU_ENGINEERING_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
