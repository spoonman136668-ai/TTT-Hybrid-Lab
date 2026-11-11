"""G7-R17: bounded, unprivileged WSL host-linker diagnosis (not CUDA execution).

One manual, source-bound invocation only. Never installs packages, launches WSL,
compiles project models, opens GPU devices, trains, or schedules a new job.
Preserves the original G7-R16 failure and emits categories, never raw stderr.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
# -I excludes the scripts directory from sys.path; import trusted same-repo code.
sys.path.insert(0, str(ROOT / "scripts"))
import g7_r15_compiler_repair_forward as native

SCHEMA = "ttt-hybrid.g7-r17-unprivileged-host-link-diagnostic.v1"
C_SOURCE = "int g7_r17_fixed_link_probe(void){return 1;}\n"
CHILD_ENV = {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C", "HOME": "/nonexistent"}
MAX_STDERR = 8192


def classify_linker_error(stderr: str) -> str:
    """Categories only; raw compiler output must never escape the process."""
    s = stderr.lower()
    if re.search(r"\b(crti[.]o|crtn[.]o|scrt1[.]o|crt1[.]o)\b", s):
        return "C_RUNTIME_STARTUP_OBJECT_MISSING"
    if "libc.so" in s or re.search(r"cannot find\s+-lc\b", s):
        return "LIBC_LINKER_INPUT_MISSING"
    if "libgcc_s.so" in s or "libgcc.a" in s or "crtbegin" in s:
        return "GCC_RUNTIME_LINKER_INPUT_MISSING"
    if "permission denied" in s:
        return "HOST_BUILD_PERMISSION_DENIED"
    if "no space left on device" in s:
        return "HOST_BUILD_STORAGE_EXHAUSTED"
    if "cannot execute" in s or "exec format error" in s:
        return "HOST_COMPILER_EXECUTION_FAILED"
    return "UNCLASSIFIED_HOST_LINK_FAILURE"


def classify_dev_package(returncode: int | None, stdout: str) -> str:
    """Parse only dpkg's precise status; no inferred installation."""
    if returncode != 0:
        return "NOT_INSTALLED_OR_QUERY_FAILED"
    status = stdout.strip()
    if status == "install ok installed":
        return "INSTALLED"
    if status == "deinstall ok config-files":
        return "REMOVED_CONFIG_ONLY"
    return "UNKNOWN_STATUS"


def fixed_probe(argv: list[str], *, input_text: str | None = None,
                timeout: int = 15) -> subprocess.CompletedProcess[str]:
    """Run only predetermined local command arrays with bounded in-memory output."""
    return subprocess.run(
        argv, input=input_text, text=True, capture_output=True,
        timeout=timeout, check=False, cwd=ROOT, env=CHILD_ENV,
    )


def run(expected_sha: str, receipt: Path) -> tuple[dict, int]:
    report = {
        "schema": SCHEMA,
        "utc": datetime.now(timezone.utc).isoformat(),
        "expected_sha": expected_sha,
        "mode": "UNPRIVILEGED_FIXED_HOST_C_LINK_PROBE_ONLY",
        "status": "BLOCKED_NOT_EXECUTED",
        "libc6_dev": "NOT_QUERIED",
        "compiler": None,
        "compiler_sha256": None,
        "shared_build": "NOT_RUN",
        "link_failure_category": None,
        "gpu": "NOT_ACCESSED",
        "native_mamba": "NOT_RUN",
        "native_ttt": "NOT_RUN",
        "model_training": "NOT_RUN",
        "scientific": "UNVERIFIED",
        "deployment": "NOT_AUTHORIZED",
        "privileged_changes": "NONE",
    }
    code = 2
    try:
        output = receipt.resolve()
        if (output.parent != (ROOT / "local_runs").resolve() or output.suffix != ".json"):
            raise native.Blocked("UNAUTHORIZED_RECEIPT_DESTINATION")
        if output.exists():
            raise native.Blocked("RECEIPT_ALREADY_EXISTS_NO_REPLAY")
        native.source_custody(expected_sha)
        report["status"] = "BLOCKED_NO_COMPILER"
        selected = native.selected_compiler()
        if selected is None:
            return report, 2
        binary, family = selected
        resolved = Path(binary).resolve(strict=True)
        if str(resolved) != binary or not resolved.is_file() or not (
            str(resolved).startswith("/usr/bin/") or str(resolved).startswith("/bin/")
        ):
            raise native.Blocked("UNTRUSTED_COMPILER_PATH")
        report["compiler"] = family
        report["compiler_sha256"] = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if Path("/usr/bin/dpkg-query").is_file():
            try:
                result = fixed_probe(
                    ["/usr/bin/dpkg-query", "--show", "--showformat=${Status}", "libc6-dev"],
                    timeout=12,
                )
                report["libc6_dev"] = classify_dev_package(result.returncode, result.stdout)
            except (OSError, subprocess.TimeoutExpired):
                report["libc6_dev"] = "QUERY_UNAVAILABLE"
        else:
            report["libc6_dev"] = "QUERY_UNAVAILABLE"
        with tempfile.TemporaryDirectory(prefix="ttt-g7-r17-") as tmp:
            dest = str(Path(tmp) / "fixed_probe.so")
            try:
                result = fixed_probe(
                    [binary, "-x", "c", "-", "-shared", "-fPIC", "-o", dest],
                    input_text=C_SOURCE, timeout=35,
                )
                built = Path(dest).is_file() and Path(dest).stat().st_size > 0
                if result.returncode == 0 and built:
                    report["shared_build"] = "PASS"
                    report["status"] = "PASS_HOST_C_SHARED_LINK_ONLY"
                    code = 0
                else:
                    report["shared_build"] = "FAIL"
                    report["link_failure_category"] = classify_linker_error(
                        result.stderr[:MAX_STDERR]
                    )
                    report["status"] = "FAIL_HOST_C_SHARED_LINK"
                    code = 1
            except (OSError, subprocess.TimeoutExpired):
                report["shared_build"] = "FAIL"
                report["link_failure_category"] = "COMPILER_INVOCATION_OR_TIMEOUT"
                report["status"] = "FAIL_HOST_C_SHARED_LINK"
                code = 1
    except native.Blocked as exc:
        report["status"] = "BLOCKED_" + str(exc)
        code = 2
    except Exception as exc:
        report["status"] = "FAIL_INTERNAL_" + type(exc).__name__
        code = 1
    finally:
        # A uniquely named receipt in the project local_runs directory only.
        # No stderr, command logs, host paths, or secrets leave this process.
        output = receipt.resolve()
        if (output.parent == (ROOT / "local_runs").resolve()
                and output.suffix == ".json" and not output.exists()):
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as file:
                json.dump(report, file, sort_keys=True, indent=2)
                file.write("\n")
    return report, code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    result, code = run(args.expected_sha, args.receipt)
    print("G7_R17_STATUS=" + result["status"])
    print("G7_R17_LINK_CATEGORY=" + str(result["link_failure_category"]))
    print("G7_R17_LIBC6_DEV=" + result["libc6_dev"])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
