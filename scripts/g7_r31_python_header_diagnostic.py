"""G7-R31: bounded nonroot confirmation of Python.h in the Triton C helper.

No model execution, CUDA kernels, package changes, network, or persistent writes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g7_r15_compiler_repair_forward as source

ROOT = Path(__file__).resolve().parents[1]
GCC_SHA = "1b99826121ae6682a634e5efe09bd3e3df58ce58e0b28f849114ab5b89139c26"

def check(ok: bool, label: str):
    if not ok:
        raise source.Blocked(label)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    data = {
        "schema": "ttt-hybrid.g7-r31.python312-dev-header-test.v1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "candidate_sha": args.expected_sha,
        "prior_g7_r29": 42361640612,
        "prior_g7_r30": 42361813516,
        "status": "BLOCKED_NOT_STARTED",
        "source_custody": "UNVERIFIED",
        "gcc_sha256": None,
        "headers": {},
        "sysconfig_header_found": False,
        "python_include_c_compile": "NOT_RUN",
        "gcc_return_code": None,
        "gcc_error_tags": [],
        "library_path": "PROCESS_LOCAL_WSL_LIBCUDA",
        "package_changes": "NONE",
        "gpu": "NOT_RUN",
        "training": "NOT_RUN",
        "scientific": "UNVERIFIED",
        "deployment": "NOT_AUTHORIZED",
    }
    result = 2
    try:
        receipt = args.receipt.resolve()
        check(receipt.parent == (ROOT / "local_runs").resolve()
              and receipt.name == "g7_r31_python_header.json",
              "RECEIPT_PATH_DENIED")
        source.source_custody(args.expected_sha)
        data["source_custody"] = "PASS"
        check(os.geteuid() == 1000, "NORMAL_UID_REQUIRED")
        real = source.selected_compiler()
        check(real is not None and real[1] == "gcc", "GCC_UNAVAILABLE")
        compiler = real[0]
        sha = hashlib.sha256(Path(compiler).read_bytes()).hexdigest()
        data["gcc_sha256"] = sha
        check(sha == GCC_SHA, "GCC_SHA_DRIFT")
        headers = {
            "ubuntu_python312": Path("/usr/include/python3.12/Python.h"),
            "local_python312": Path("/usr/local/include/python3.12/Python.h"),
            "venv_python312": source.R3_RUNTIME / "venv/include/python3.12/Python.h",
        }
        data["headers"] = {key: path.is_file() for key, path in headers.items()}
        syscfg = Path(sysconfig.get_paths().get("include") or "/nonexistent") / "Python.h"
        data["sysconfig_header_found"] = syscfg.is_file()
        check(Path("/usr/lib/wsl/lib/libcuda.so").is_file(), "CUDA_LINK_PATH_DRIFT")
        env = {**os.environ, "LIBRARY_PATH": "/usr/lib/wsl/lib",
               "PYTHONDONTWRITEBYTECODE": "1"}
        with tempfile.TemporaryDirectory(prefix="ttt-g7-r31-python-h-") as d:
            target = Path(d) / "python_include_test.so"
            proc = subprocess.run(
                [compiler, "-x", "c", "-", "-shared", "-fPIC",
                 "-I/usr/include/python3.12", "-o", str(target), "-lcuda"],
                input="#include <Python.h>\nint g7_r31_test(void){return 1;}\n",
                text=True, capture_output=True, cwd=ROOT, env=env,
                timeout=30, check=False,
            )
            data["gcc_return_code"] = (proc.returncode
                                        if proc.returncode in range(0,256) else None)
            text = (proc.stderr or "").lower()[:4096]
            tags = {
                "MISSING_PYTHON_H": (
                    "python.h: no such file or directory" in text
                    or "fatal error: python.h" in text
                ),
                "MISSING_LIBCUDA": "cannot find -lcuda" in text,
                "MISSING_CRT": "cannot find crti.o" in text,
                "UNDEFINED_SYMBOL": "undefined reference" in text,
                "PERMISSION": "permission denied" in text,
            }
            data["gcc_error_tags"] = sorted(k for k, v in tags.items() if v)
            success = proc.returncode == 0 and target.is_file() and target.stat().st_size > 0
            data["python_include_c_compile"] = "PASS" if success else "FAIL"
        if success:
            data["status"] = "PASS_PYTHON_HEADER_COMPILE_AVAILABLE"
        elif "MISSING_PYTHON_H" in data["gcc_error_tags"] and not data["headers"]["ubuntu_python312"]:
            data["status"] = "PASS_CONFIRMED_MISSING_PYTHON312_DEV_HEADER"
        else:
            data["status"] = "BLOCKED_C_COMPILER_ERROR_NOT_PYTHON_HEADER"
        result = 0 if data["status"].startswith("PASS_") else 2
    except source.Blocked as exc:
        data["status"] = "BLOCKED_" + str(exc)
    except (OSError, subprocess.TimeoutExpired) as exc:
        data["status"] = "BLOCKED_HOST_" + type(exc).__name__
    except Exception as exc:
        data["status"] = "BLOCKED_UNEXPECTED_" + type(exc).__name__
    finally:
        target = args.receipt.resolve()
        if (target.parent == (ROOT / "local_runs").resolve()
                and target.name == "g7_r31_python_header.json"):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n",
                              encoding="utf-8")
        print("G7_R31_STATUS=" + data["status"], flush=True)
        print("G7_R31_C_COMPILE=" + data["python_include_c_compile"], flush=True)
        print("G7_R31_GPU=NOT_RUN", flush=True)
    return result

if __name__ == "__main__":
    raise SystemExit(main())
