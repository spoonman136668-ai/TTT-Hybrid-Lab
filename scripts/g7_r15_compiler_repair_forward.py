"""G7-R15 one authorized GCC host-helper repair and one authentic bounded forward.

Original G7-R9 runner/WSL route. Never trains, deploys, changes CUDA, installs
torch/triton/models, or retries. Only optional Ubuntu GCC apt provisioning.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PINS_SHA = "44f3e76d2d06ee879fa3222dd3921c7a1f8e08be579b96e03b7545e388a7b768"
R3_SHA = "06a01c2c3a980eeb806a7ecb7cca1979a431f80c"
R3_RUNTIME = Path.home() / ".local/share/ttt-hybrid" / ("g7-r3-" + R3_SHA[:12])


class Blocked(Exception):
    pass


def guard(condition: bool, code: str) -> None:
    if not condition:
        raise Blocked(code)


def command(argv: list[str], *, limit: int, env: dict[str, str] | None = None,
            source: str | None = None) -> tuple[int | None, str]:
    """Never persist stderr or other output; only return bounded stdout for metadata."""
    try:
        p = subprocess.run(
            argv, input=source, text=True, capture_output=True,
            timeout=limit, check=False, cwd=ROOT, env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, ""
    return p.returncode, p.stdout[:250]


def source_custody(expected: str) -> None:
    guard(sys.platform == "linux" and sys.version_info[:2] == (3, 12),
          "UBUNTU_PYTHON_312_REQUIRED")
    guard(os.geteuid() != 0, "REFUSE_ROOT")
    guard("microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower(),
          "WSL_KERNEL_REQUIRED")
    os_release = Path("/etc/os-release").read_text()
    guard(bool(re.search(r'(?m)^ID="?ubuntu"?$', os_release)) and
          bool(re.search(r'(?m)^VERSION_ID="?24[.]04"?$', os_release)),
          "PINNED_UBUNTU_2404_REQUIRED")
    guard(bool(re.fullmatch("[0-9a-f]{40}", expected)), "INVALID_EXPECTED_SHA")
    sha_code, observed = command(["/usr/bin/git", "-C", str(ROOT), "rev-parse", "HEAD"], limit=15)
    guard(sha_code == 0 and observed.strip() == expected, "PROJECT_SOURCE_SHA_DRIFT")
    dirty_code, dirty = command(["/usr/bin/git", "-C", str(ROOT), "status", "--porcelain",
                                 "--untracked-files=all"], limit=15)
    guard(dirty_code == 0 and not dirty.strip(), "DIRTY_PROJECT_CHECKOUT")
    pins = ROOT / "source_pins.json"
    guard(pins.is_file() and hashlib.sha256(pins.read_bytes()).hexdigest() == SOURCE_PINS_SHA,
          "SOURCE_PINS_DRIFT")
    guard((R3_RUNTIME / "venv/bin/python").is_file() and
          (R3_RUNTIME / "sources/mamba2").is_dir() and
          (R3_RUNTIME / "sources/ttt_linear").is_dir(),
          "PINNED_R3_RUNTIME_UNAVAILABLE")


def selected_compiler() -> tuple[str, str] | None:
    """Trust only real Linux distribution compiler executables, not PATH wrappers."""
    for name in ("gcc", "clang", "gcc-13", "gcc-14", "clang-18", "clang-17"):
        for path in ("/usr/bin/" + name, "/bin/" + name):
            p = Path(path)
            if not p.is_file() or not os.access(p, os.X_OK):
                continue
            real = p.resolve()
            if real.is_file() and os.access(real, os.X_OK) and (
                    str(real).startswith("/usr/bin/") or str(real).startswith("/bin/")):
                return (str(real), "gcc" if name.startswith("gcc") else "clang")
    return None


def apt_install_gcc(receipt: dict) -> None:
    sudo = "/usr/bin/sudo"
    apt = "/usr/bin/apt-get"
    guard(Path(sudo).is_file() and Path(apt).is_file(), "GCC_APT_TOOLS_MISSING")
    ok, _ = command([sudo, "-n", "/usr/bin/true"], limit=12)
    guard(ok == 0, "GCC_SUDO_NONINTERACTIVE_DENIED")
    receipt["provisioning"] = "GCC_APT_UPDATE_ATTEMPTED"
    update_code, _ = command([sudo, "-n", apt, "update",
                              "-o", "Acquire::Retries=0"], limit=150)
    guard(update_code == 0, "GCC_APT_UPDATE_FAILED")
    receipt["provisioning"] = "GCC_APT_INSTALL_ATTEMPTED"
    env = dict(os.environ)
    env["DEBIAN_FRONTEND"] = "noninteractive"
    install_code, _ = command(
        [sudo, "-n", apt, "install", "-y", "--no-install-recommends", "gcc"],
        limit=200, env=env,
    )
    guard(install_code == 0, "GCC_APT_INSTALL_FAILED")
    receipt["provisioning"] = "GCC_APT_INSTALL_SUCCEEDED"


def compiler_qualification(executable: str, family: str, receipt: dict) -> None:
    hash_hex = hashlib.sha256(Path(executable).read_bytes()).hexdigest()
    version_code, version = command([executable, "-dumpfullversion"], limit=15)
    if version_code != 0:
        version_code, version = command([executable, "-dumpversion"], limit=15)
    guard(version_code == 0 and
          bool(re.fullmatch(r"[0-9]+(?:[.][0-9]+){0,3}", version.strip())),
          "COMPILER_VERSION_UNVERIFIED")
    # One tiny C-to-shared-library build, not a CUDA kernel or native model build.
    with tempfile.TemporaryDirectory(prefix="ttt-g7-r15-cc-") as temp:
        dest = str(Path(temp) / "host_helper.so")
        ok, _ = command([executable, "-x", "c", "-", "-shared", "-fPIC",
                          "-o", dest], limit=35,
                        source="int g7_r15_compiler_probe(void){return 1;}\n")
        guard(ok == 0 and Path(dest).is_file() and Path(dest).stat().st_size > 0,
              "HOST_C_SHARED_BUILD_FAILED")
    receipt["compiler"] = {
        "family": family,
        "version": version.strip(),
        "binary_sha256": hash_hex,
        "host_c_shared_library_probe": "PASS",
        "effective_cc": "VERIFIED_DISTRIBUTION_COMPILER",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected-sha", required=True)
    ap.add_argument("--receipt", required=True, type=Path)
    args = ap.parse_args()
    report = {
        "schema": "ttt-hybrid.g7-r15.compiler-repair-native-forward.v1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "candidate_sha": args.expected_sha,
        "original_failed_run_id": 42300670454,
        "diagnosed_original_cause": "HOST_C_COMPILER_MISSING",
        "mode": "SINGLE_COMPILER_REPAIR_AND_AUTHENTIC_FORWARD_ONLY",
        "status": "BLOCKED_NOT_STARTED",
        "source_custody": "UNVERIFIED",
        "provisioning": "NOT_ATTEMPTED",
        "compiler": None,
        "native_forward": "NOT_RUN",
        "native_receipt_sha256": None,
        "native_result_status": None,
        "native_gate3_status": None,
        "native_execution_path": None,
        "forward_checks": None,
        "training": "NOT_RUN",
        "gradient": "NOT_RUN",
        "optimizer": "NOT_RUN",
        "scientific": "UNVERIFIED",
        "deployment": "NOT_AUTHORIZED",
        "cuda_runtime_changes": "NONE",
        "torch_or_triton_reinstall": "NONE",
    }
    result_code = 2
    try:
        receipt = args.receipt.resolve()
        guard(receipt.parent == (ROOT / "local_runs").resolve(),
              "UNAUTHORIZED_RECEIPT_DESTINATION")
        source_custody(args.expected_sha)
        report["source_custody"] = "PASS"
        compiler = selected_compiler()
        if compiler is None:
            apt_install_gcc(report)
            compiler = selected_compiler()
            guard(compiler is not None, "COMPILER_NOT_FOUND_AFTER_PROVISION")
        else:
            report["provisioning"] = "SKIPPED_EXISTING_COMPILER"
        executable, family = compiler
        compiler_qualification(executable, family, report)
        # CC only affects this one forward process and its Triton subprocesses.
        # No PATH/system/user shell modifications.
        env = dict(os.environ)
        env["CC"] = executable
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["CUDA_MODULE_LOADING"] = "LAZY"
        report["native_forward"] = "ONE_FORWARD_INVOKED"
        native_receipt = ROOT / "local_runs/g7_r15_native_forward.json"
        forward_code, _ = command(
            ["/usr/bin/python3", "-I", "-B", str(ROOT / "scripts/g7_r6_oneshot.py"),
             "--require-nonfused", "--expected-sha", args.expected_sha,
             "--receipt", str(native_receipt)],
            limit=220, env=env,
        )
        guard(native_receipt.is_file(), "NATIVE_FORWARD_RECEIPT_MISSING")
        raw = native_receipt.read_bytes()
        native = json.loads(raw)
        report["native_receipt_sha256"] = hashlib.sha256(raw).hexdigest()
        report["native_result_status"] = native.get("status")
        report["native_gate3_status"] = native.get("native_gate3_status")
        report["native_execution_path"] = native.get("native_mamba_execution_path")
        chk = native.get("control_results") or {}
        report["forward_checks"] = {
            key: chk.get(key) for key in (
                "forward_finite", "episode_reset_deterministic",
                "prefix_invariance", "suffix_negative_control", "gradient_execution",
                "optimizer_steps", "training"
            )
        }
        guard(forward_code == 0 and native.get("status") ==
              "PASS_ONE_BOUNDED_NATIVE_FORWARD_ONLY",
              "ORIGINAL_BOUNDED_FORWARD_NOT_QUALIFIED")
        report["native_forward"] = "PASS_AUTHENTIC_FORWARD_ONLY"
        report["status"] = "PASS_COMPILER_REPAIR_AND_NATIVE_FORWARD"
        result_code = 0
    except Blocked as exc:
        report["status"] = "BLOCKED_" + str(exc)
    except Exception as exc:
        report["status"] = "FAIL_UNEXPECTED_" + type(exc).__name__
        result_code = 1
    finally:
        # Only one sanitized receipt is exported; do not upload native inner/G3 data.
        out = args.receipt.resolve()
        if out.parent == (ROOT / "local_runs").resolve():
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
        print("G7_R15_STATUS=" + report["status"], flush=True)
        print("G7_R15_PROVISION=" + report["provisioning"], flush=True)
        print("G7_R15_FORWARD=" + report["native_forward"], flush=True)
    return result_code


if __name__ == "__main__":
    raise SystemExit(main())
