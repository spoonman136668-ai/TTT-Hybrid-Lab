"""G7-R29: bounded original GPU forward with process-local WSL libcuda link path.
No system mutation, driver/package changes, training or deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g7_r15_compiler_repair_forward as original

ROOT = Path(__file__).resolve().parents[1]
LIB = Path("/usr/lib/wsl/lib")
COMPILER_SHA256 = "1b99826121ae6682a634e5efe09bd3e3df58ce58e0b28f849114ab5b89139c26"

def guard(ok: bool, why: str) -> None:
    if not ok:
        raise original.Blocked(why)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected-sha", required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    args = ap.parse_args()
    report = {
        "schema": "ttt-hybrid.g7-r29.process-local-libcuda-forward.v1",
        "candidate_sha": args.expected_sha,
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "g7_r28_evidence_run": 42361346770,
        "status": "BLOCKED_NOT_STARTED",
        "source_custody": "UNVERIFIED",
        "libcuda_link_path": "/usr/lib/wsl/lib",
        "compiler_sha256": None,
        "link_probe": "NOT_RUN",
        "native_forward": "NOT_RUN",
        "native_status": None,
        "native_gate3_status": None,
        "native_failure_category": None,
        "native_failure_type": None,
        "native_execution_path": None,
        "package_changes": "NONE",
        "training": "NOT_RUN",
        "scientific": "UNVERIFIED",
        "deployment": "NOT_AUTHORIZED",
    }
    result = 2
    try:
        dest = args.receipt.resolve()
        guard(dest.parent == (ROOT / "local_runs").resolve()
              and dest.name == "g7_r29_process_local_forward.json",
              "RECEIPT_PATH_DENIED")
        original.source_custody(args.expected_sha)
        report["source_custody"] = "PASS"
        guard(os.geteuid() == 1000, "UNPRIVILEGED_UID_REQUIRED")
        guard((LIB / "libcuda.so").is_file() and
              (LIB / "libcuda.so.1").is_file(),
              "WSL_LIBCUDA_LINKER_TARGET_DRIFT")
        selected = original.selected_compiler()
        guard(selected is not None and selected[1] == "gcc", "COMPILER_DRIFT")
        compiler = selected[0]
        binary_hash = hashlib.sha256(Path(compiler).read_bytes()).hexdigest()
        report["compiler_sha256"] = binary_hash
        guard(binary_hash == COMPILER_SHA256, "COMPILER_SHA_DRIFT")
        env = dict(os.environ)
        env["LIBRARY_PATH"] = str(LIB)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["CUDA_MODULE_LOADING"] = "LAZY"
        with tempfile.TemporaryDirectory(prefix="ttt-g7-r29-link-") as d:
            target = Path(d) / "g7_cuda_link.so"
            probe = subprocess.run(
                [compiler, "-x", "c", "-", "-shared", "-fPIC",
                 "-o", str(target), "-lcuda"],
                input="int g7_r29_link_probe(void){return 1;}\n",
                text=True, capture_output=True, check=False,
                timeout=30, cwd=ROOT, env=env,
            )
            guard(probe.returncode == 0 and target.is_file()
                  and target.stat().st_size > 0,
                  "PROCESS_LOCAL_LIBCUDA_LINK_FAILED")
        report["link_probe"] = "PASS_PROCESS_LOCAL_GCC_LIBCUDA_LINK"
        outer = ROOT / "local_runs/g7_r29_compiler_forward.json"
        report["native_forward"] = "ONE_ORIGINAL_NONFUSED_FORWARD_STARTED"
        proc = subprocess.run(
            ["/usr/bin/python3", "-I", "-B",
             str(ROOT / "scripts/g7_r15_compiler_repair_forward.py"),
             "--expected-sha", args.expected_sha,
             "--receipt", str(outer)],
            cwd=ROOT, env=env, capture_output=True, text=True,
            timeout=260, check=False,
        )
        guard(outer.is_file(), "NATIVE_RECEIPT_MISSING")
        inner = json.loads(outer.read_text(encoding="utf-8"))
        report["native_status"] = inner.get("status")
        report["native_gate3_status"] = inner.get("native_gate3_status")
        report["native_execution_path"] = inner.get("native_execution_path")
        report["native_forward"] = inner.get("native_forward")
        nested = ROOT / "local_runs/g7_r15_native_forward.json"
        if nested.is_file():
            details = json.loads(nested.read_text(encoding="utf-8"))
            report["native_failure_category"] = details.get("native_failure_category")
            report["native_failure_type"] = details.get("native_failure_type")
        guard(proc.returncode == 0 and inner.get("status") ==
              "PASS_COMPILER_REPAIR_AND_NATIVE_FORWARD",
              "ORIGINAL_NONFUSED_FORWARD_NOT_QUALIFIED")
        report["status"] = "PASS_AUTHENTIC_NONFUSED_FORWARD_WITH_TEMP_CUDA_LINK"
        report["native_forward"] = "PASS_ONE_AUTHENTIC_FORWARD_ONLY"
        result = 0
    except original.Blocked as exc:
        report["status"] = "BLOCKED_" + str(exc)
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["status"] = "BLOCKED_HOST_" + type(exc).__name__
    except Exception as exc:
        report["status"] = "BLOCKED_UNEXPECTED_" + type(exc).__name__
    finally:
        target = args.receipt.resolve()
        if (target.parent == (ROOT / "local_runs").resolve() and
                target.name == "g7_r29_process_local_forward.json"):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                              encoding="utf-8")
        print("G7_R29_STATUS=" + report["status"], flush=True)
        print("G7_R29_LINK=" + report["link_probe"], flush=True)
        print("G7_R29_FORWARD=" + str(report["native_forward"]), flush=True)
    return result

if __name__ == "__main__":
    raise SystemExit(main())
