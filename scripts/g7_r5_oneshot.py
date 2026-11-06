"""G7-R5 one-shot bounded CUDA kernel admission on the existing isolated WSL runner.

No installation, downloads, native extensions, model training, integrated forward,
unrelated GPU work, new runner or controller. Outer admission is standard library
only; the inner process uses the already-qualified R3 Python environment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

R3_SOURCE_SHA = "06a01c2c3a980eeb806a7ecb7cca1979a431f80c"
R3_PINS_SHA256 = "44f3e76d2d06ee879fa3222dd3921c7a1f8e08be579b96e03b7545e388a7b768"
R3_MANIFEST_SHA256 = "b0f674fd52f9f514d408840d27b969ce2da19ea119d90a6ce8f59fde05c073df"
TORCH = "2.9.0+cu128"
TRITON = "3.5.0"
MIN_FREE_GIB = 4.0
MAX_PEAK_GIB = 1.0
CHILD_TIMEOUT_SECONDS = 175

ROOT = Path(__file__).resolve().parents[1]


class Blocked(Exception):
    pass


def guard(ok: bool, reason: str) -> None:
    if not ok:
        raise Blocked(reason)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root: Path, *args: str) -> str:
    try:
        p = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True,
            text=True, check=False, timeout=16,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise Blocked("GIT_UNAVAILABLE_OR_TIMEOUT") from None
    guard(p.returncode == 0, "GIT_COMMAND_FAILED")
    return p.stdout.strip()


def source_custody(root: Path, spec: dict) -> dict:
    revision = spec["sha"]
    guard(bool(re.fullmatch(r"[0-9a-f]{40}", revision)), "INVALID_SOURCE_REVISION")
    guard(root.is_dir(), "SOURCE_CHECKOUT_ABSENT")
    guard(git(root, "rev-parse", "HEAD") == revision, "SOURCE_REVISION_DRIFT")
    guard(not git(root, "status", "--porcelain", "--untracked-files=all"),
          "SOURCE_TREE_DIRTY")
    blobs = {}
    for rel in spec["required_files"]:
        guard(bool(re.fullmatch(r"[A-Za-z0-9_./-]{1,180}", rel))
              and ".." not in Path(rel).parts, "UNSAFE_SOURCE_PATH")
        observed = git(root, "hash-object", "--", rel)
        guard(observed == spec["observed_blob_sha"][rel], "SOURCE_BLOB_DRIFT")
        blobs[rel] = observed
    return {"sha": revision, "blobs": blobs, "clean": True}


def manifest() -> tuple[str, dict]:
    distributions = sorted(
        (str(x.metadata.get("Name", "")).lower().replace("_", "-"), str(x.version))
        for x in metadata.distributions()
    )
    h = hashlib.sha256(json.dumps(distributions, sort_keys=True).encode()).hexdigest()
    return h, dict(distributions)


def inner_run(args: argparse.Namespace, env_root: Path) -> int:
    """Only executed by the already-pinned venv Python after outer admission."""
    pins = json.loads((ROOT / "source_pins.json").read_text(encoding="utf-8"))
    sys.path[:0] = [
        str(ROOT),
        str(env_root / "sources" / "mamba2"),
        str(env_root / "sources" / "ttt_linear"),
    ]
    # All optional Mamba kernel modules are skipped by kernel_only=True.
    from scripts.g7_cuda_qualification import run

    _, status = run(
        env_root / "sources" / "mamba2",
        env_root / "sources" / "ttt_linear",
        receipt=Path(args.inner_receipt),
        min_free_gib=MIN_FREE_GIB,
        max_peak_gib=MAX_PEAK_GIB,
        expected_sha=args.expected_sha,
        kernel_only=True,
    )
    return status


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--expected-sha", required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--inner-receipt", type=Path)
    p.add_argument("--child", action="store_true")
    args = p.parse_args()
    env_root = Path.home().resolve() / ".local" / "share" / "ttt-hybrid" / (
        "g7-r3-" + R3_SOURCE_SHA[:12]
    )
    if args.child:
        guard(args.inner_receipt is not None, "MISSING_INNER_RECEIPT")
        return inner_run(args, env_root)

    receipt = args.receipt.resolve()
    info = {
        "schema": "ttt-hybrid.g7-r5-first-native-kernel.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_sha": args.expected_sha,
        "status": "BLOCKED_NOT_STARTED",
        "operation": "one_16x16_float32_cuda_matmul",
        "min_free_gib": MIN_FREE_GIB,
        "torch_allocator_ceiling_gib": MAX_PEAK_GIB,
        "timeout_seconds": CHILD_TIMEOUT_SECONDS,
        "r3_baseline_sha": R3_SOURCE_SHA,
        "r3_package_manifest_sha256": R3_MANIFEST_SHA256,
        "scientific": "UNVERIFIED",
        "native_mamba_forward": "NOT_RUN",
        "native_ttt_forward": "NOT_RUN",
        "training": "NOT_RUN",
        "system_installation": "NOT_RUN",
        "native_extension_build": "NOT_RUN",
        "deployment": "NOT_AUTHORIZED",
    }
    code = 2
    try:
        guard(sys.platform == "linux" and platform.machine() == "x86_64",
              "LINUX_X86_64_REQUIRED")
        guard(sys.version_info[:2] == (3, 12), "PYTHON_312_REQUIRED")
        guard(os.geteuid() != 0, "REFUSE_ROOT")
        guard("microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower(),
              "WSL2_REQUIRED")
        guard(bool(re.fullmatch(r"[0-9a-f]{40}", args.expected_sha)),
              "INVALID_CANDIDATE_SHA")
        guard(git(ROOT, "rev-parse", "HEAD") == args.expected_sha,
              "PROJECT_SHA_DRIFT")
        guard(not git(ROOT, "status", "--porcelain"),
              "PROJECT_TREE_DIRTY")
        pins_raw = (ROOT / "source_pins.json").read_bytes()
        guard(hashlib.sha256(pins_raw).hexdigest() == R3_PINS_SHA256,
              "SOURCE_PINS_DRIFT")
        pins = json.loads(pins_raw)
        guard(pins.get("schema") == "ttt-hybrid.official-source-pins.v1",
              "PIN_SCHEMA_MISMATCH")
        for label in ("mamba2", "ttt_linear"):
            info[label] = source_custody(
                env_root / "sources" / label, pins["sources"][label],
            )
        python = env_root / "venv" / "bin" / "python"
        guard(python.is_file(), "R3_ISOLATED_PYTHON_NOT_FOUND")
        info["r3_environment_present"] = True
        inner_receipt = receipt.with_name("g7_r5_native_internal.json")
        # The child is the ONLY process permitted to import torch/initialize CUDA.
        try:
            sub = subprocess.run(
                [str(python), "-I", "-B", str(ROOT / "scripts" / "g7_r5_oneshot.py"),
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
            raise Blocked("KERNEL_CHILD_TIMEOUT") from None
        except OSError:
            raise Blocked("KERNEL_CHILD_UNAVAILABLE") from None
        guard(inner_receipt.is_file(), "MISSING_NATIVE_RESULT_RECEIPT")
        inner = json.loads(inner_receipt.read_text(encoding="utf-8"))
        info["native_receipt_sha256"] = sha256(inner_receipt)
        info["native_status"] = inner.get("status", "UNVERIFIED")
        info["native_project_sha"] = inner.get("project_sha")
        info["native_mode"] = inner.get("mode")
        info["native_admission_reason"] = inner.get("admission", {}).get("reason")
        info["native_gpu"] = {
            "name": inner.get("gpu", {}).get("name"),
            "capability": inner.get("gpu", {}).get("capability"),
            "peak_allocated_bytes": inner.get("gpu", {}).get("peak_allocated_bytes"),
        }
        info["module_origin_checks"] = [
            {"module": m.get("module"), "exact_source_path": m.get("exact_source_path")}
            for m in inner.get("modules", [])
        ]
        info["native_authentic_integration"] = inner.get("authentic_integration")
        info["native_scientific_qualification"] = inner.get("scientific_qualification")
        valid = (
            sub.returncode == 0
            and inner.get("status") == "PASS_NATIVE_CUDA_KERNEL_ONLY"
            and inner.get("project_sha") == args.expected_sha
            and inner.get("mode") == "CUDA_KERNEL_ONLY"
            and inner.get("candidate_identity", {}).get("pass") is True
            and inner.get("preflight", {}).get("status")
                == "PASS_SOURCE_RUNTIME_ADMISSION_ONLY"
            and inner.get("gpu", {}).get("capability") == [12, 0]
            and isinstance(inner.get("gpu", {}).get("peak_allocated_bytes"), int)
            and 0 < inner["gpu"]["peak_allocated_bytes"] <= MAX_PEAK_GIB * 1024**3
            and inner.get("kernel_only", {}).get("finite_result") is True
            and inner.get("authentic_integration") == "NOT_RUN"
            and inner.get("scientific_qualification") == "UNVERIFIED"
            and len(info["module_origin_checks"]) == 2
            and all(m["exact_source_path"] is True
                    for m in info["module_origin_checks"])
        )
        guard(valid, "NATIVE_GATE_NOT_QUALIFIED")
        info["status"] = "PASS_ONE_BOUNDED_NATIVE_CUDA_MATMUL_ONLY"
        code = 0
    except Blocked as exc:
        info["status"] = "BLOCKED_" + str(exc)
    except Exception as exc:
        # Sanitize type only: never log paths, env, third-party error text.
        info["status"] = "FAIL_UNEXPECTED_" + type(exc).__name__
        code = 1
    finally:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(info, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
        print("G7_R5_STATUS=" + info["status"], flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
