"""G7-R6: one authorized no-grad authentic Mamba-2/TTT CUDA forward.

Isolated existing R3 venv and immutable official source checkouts only.
No installations, downloads, training, optimizer, backward, new runner or controller.
The GitHub artifact contains only the sanitized outer receipt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import g7_r5_oneshot as custody

MIN_FREE_GIB = 4.0
MAX_PEAK_GIB = 1.0
CHILD_TIMEOUT_SECONDS = 175


def valid_native_result(inner: dict, expected_sha: str, child_code: int) -> bool:
    """Pure acceptance check, including negative-control and no-training custody."""
    checks = inner.get("native_forward_checks") or {}
    gpu = inner.get("gpu") or {}
    peak = gpu.get("peak_allocated_bytes")
    mods = inner.get("modules") or []
    return (
        child_code == 0
        and inner.get("project_sha") == expected_sha
        and inner.get("mode") == "AUTHENTIC_FORWARD_ONLY"
        and (inner.get("candidate_identity") or {}).get("pass") is True
        and (inner.get("preflight") or {}).get("status") ==
            "PASS_SOURCE_RUNTIME_ADMISSION_ONLY"
        and (inner.get("admission") or {}).get("pass") is True
        and inner.get("status") == "PASS_NATIVE_CUDA_FORWARD_ONLY"
        and inner.get("native_gate3_status") == "PASS_AUTHENTIC_GPU_FORWARD_ONLY"
        and inner.get("authentic_integration") == "PASS"
        and inner.get("scientific_qualification") == "UNVERIFIED"
        and inner.get("torch") == custody.TORCH
        and inner.get("torch_cuda_build") == "12.8"
        and gpu.get("capability") == [12, 0]
        and type(peak) is int and 0 < peak <= MAX_PEAK_GIB * 1024**3
        and len(mods) == 4
        and all(m.get("exact_source_path") is True for m in mods)
        and checks.get("forward_finite") is True
        and checks.get("shape") == [2, 16, 64]
        and checks.get("episode_reset_deterministic") is True
        and checks.get("prefix_invariance") is True
        and checks.get("suffix_negative_control") is True
        and checks.get("gradient_execution") == "NOT_RUN"
        and checks.get("optimizer_steps") == 0
        and checks.get("training") == "NOT_RUN"
    )


def child(args: argparse.Namespace, env_root: Path) -> int:
    """Entry in already pinned R3 Python, after source and host admission."""
    try:
        package_sha, packages = custody.manifest()
        custody.guard(package_sha == custody.R3_MANIFEST_SHA256,
                      "R3_PACKAGE_MANIFEST_DRIFT")
        custody.guard(packages.get("torch") == custody.TORCH
                      and packages.get("triton") == custody.TRITON,
                      "R3_RUNTIME_PACKAGE_DRIFT")
        sys.path[:0] = [
            str(ROOT),
            str(env_root / "sources" / "mamba2"),
            str(env_root / "sources" / "ttt_linear"),
        ]
        import torch
        custody.guard(torch.__version__ == custody.TORCH
                      and torch.version.cuda == "12.8"
                      and torch.cuda.is_available()
                      and torch.cuda.device_count() == 1
                      and tuple(torch.cuda.get_device_capability(0)) == (12, 0),
                      "R6_CUDA_DEVICE_IDENTITY_DRIFT")
        from scripts.g7_cuda_qualification import run
        _, code = run(
            env_root / "sources" / "mamba2",
            env_root / "sources" / "ttt_linear",
            receipt=Path(args.inner_receipt),
            min_free_gib=MIN_FREE_GIB,
            max_peak_gib=MAX_PEAK_GIB,
            expected_sha=args.expected_sha,
            forward_only=True,
        )
        return code
    except custody.Blocked as exc:
        status, code = "BLOCKED_" + str(exc), 2
    except Exception as exc:
        status, code = "FAIL_CHILD_" + type(exc).__name__, 1
    Path(args.inner_receipt).write_text(
        json.dumps({"status": status, "mode": "AUTHENTIC_FORWARD_ONLY",
                    "scientific_qualification": "UNVERIFIED"},
                   sort_keys=True) + "\n", encoding="utf-8"
    )
    return code


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expected-sha", required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--inner-receipt", type=Path)
    p.add_argument("--child", action="store_true")
    args = p.parse_args()
    env_root = Path.home().resolve() / ".local" / "share" / "ttt-hybrid" / (
        "g7-r3-" + custody.R3_SOURCE_SHA[:12]
    )
    if args.child:
        custody.guard(args.inner_receipt is not None, "MISSING_INNER_RECEIPT")
        return child(args, env_root)

    receipt = args.receipt.resolve()
    info = {
        "schema": "ttt-hybrid.g7-r6-authentic-forward.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_sha": args.expected_sha,
        "status": "BLOCKED_NOT_RUN",
        "mode": "AUTHENTIC_FORWARD_ONLY",
        "min_free_gib": MIN_FREE_GIB,
        "torch_allocator_ceiling_gib": MAX_PEAK_GIB,
        "allocator_scope": "PYTORCH_ONLY",
        "child_timeout_seconds": CHILD_TIMEOUT_SECONDS,
        "r3_baseline_sha": custody.R3_SOURCE_SHA,
        "r3_package_manifest_sha256": custody.R3_MANIFEST_SHA256,
        "source_pins_sha256": custody.R3_PINS_SHA256,
        "scientific": "UNVERIFIED",
        "training": "NOT_RUN",
        "gradient": "NOT_RUN",
        "optimizer": "NOT_RUN",
        "deployment": "NOT_AUTHORIZED",
        "binary_provenance": "PARTIAL_UNVERIFIED",
    }
    code = 2
    try:
        custody.guard(sys.platform == "linux" and platform.machine() == "x86_64",
                      "LINUX_X86_64_REQUIRED")
        custody.guard(sys.version_info[:2] == (3, 12), "PYTHON_312_REQUIRED")
        custody.guard(os.geteuid() != 0, "REFUSE_ROOT")
        custody.guard("microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower(),
                      "WSL2_REQUIRED")
        custody.guard(bool(re.fullmatch(r"[0-9a-f]{40}", args.expected_sha)),
                      "INVALID_CANDIDATE_SHA")
        custody.guard(custody.git(ROOT, "rev-parse", "HEAD") == args.expected_sha,
                      "PROJECT_SHA_DRIFT")
        custody.guard(not custody.git(ROOT, "status", "--porcelain", "--untracked-files=all"),
                      "PROJECT_TREE_DIRTY")
        custody.guard(custody.sha256(ROOT / "source_pins.json") ==
                      custody.R3_PINS_SHA256, "PINNED_MANIFEST_DRIFT")
        pins = json.loads((ROOT / "source_pins.json").read_text(encoding="utf-8"))
        custody.guard(pins.get("schema") == "ttt-hybrid.official-source-pins.v1",
                      "PIN_SCHEMA_MISMATCH")
        for label in ("mamba2", "ttt_linear"):
            info[label] = custody.source_custody(
                env_root / "sources" / label, pins["sources"][label])
        python = env_root / "venv" / "bin" / "python"
        custody.guard(python.is_file(), "R3_ISOLATED_PYTHON_NOT_FOUND")
        inner_receipt = receipt.with_name("g7_r6_native_internal.json")
        try:
            sub = subprocess.run(
                [str(python), "-I", "-B", str(ROOT / "scripts" / "g7_r6_oneshot.py"),
                 "--child", "--expected-sha", args.expected_sha,
                 "--receipt", str(receipt),
                 "--inner-receipt", str(inner_receipt)],
                capture_output=True, text=True, timeout=CHILD_TIMEOUT_SECONDS,
                check=False,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                     "CUDA_MODULE_LOADING": "LAZY",
                     "EXPECTED_CANDIDATE_SHA": args.expected_sha},
            )
        except subprocess.TimeoutExpired:
            raise custody.Blocked("FORWARD_CHILD_TIMEOUT") from None
        except OSError:
            raise custody.Blocked("FORWARD_CHILD_UNAVAILABLE") from None
        custody.guard(inner_receipt.is_file(), "MISSING_NATIVE_RESULT_RECEIPT")
        inner = json.loads(inner_receipt.read_text(encoding="utf-8"))
        info["native_receipt_sha256"] = custody.sha256(inner_receipt)
        info["native_status"] = inner.get("status", "UNVERIFIED")
        info["native_gate3_status"] = inner.get("native_gate3_status")
        info["native_project_sha"] = inner.get("project_sha")
        info["native_mode"] = inner.get("mode")
        info["native_admission_reason"] = (
            (inner.get("admission") or {}).get("reason"))
        gpu = inner.get("gpu") or {}
        info["native_gpu"] = {
            "name": gpu.get("name"),
            "capability": gpu.get("capability"),
            "peak_allocated_bytes": gpu.get("peak_allocated_bytes"),
        }
        checks = inner.get("native_forward_checks") or {}
        info["control_results"] = {
            k: checks.get(k) for k in (
                "forward_finite", "shape", "episode_reset_deterministic",
                "prefix_invariance", "suffix_negative_control",
                "gradient_execution", "optimizer_steps", "training")
        }
        custody.guard(valid_native_result(inner, args.expected_sha, sub.returncode),
                      "NATIVE_FORWARD_NOT_QUALIFIED")
        info["status"] = "PASS_ONE_BOUNDED_NATIVE_FORWARD_ONLY"
        code = 0
    except custody.Blocked as exc:
        info["status"] = "BLOCKED_" + str(exc)
    except Exception as exc:
        info["status"] = "FAIL_UNEXPECTED_" + type(exc).__name__
        code = 1
    finally:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(info, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
        print("G7_R6_STATUS=" + info["status"], flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
