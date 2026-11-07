"""G7-R23: read-only post-install normalization, audit and host C linker test.

G7-R22 installed five signed package payloads with dpkg exit 0 but stopped on
a receipt comparison that did not normalize native :amd64 package suffixes.
This one-use read-only check must pass before calling the original non-fused GPU
forward. No apt/dpkg writes, root context, or model execution in this file.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g7_r15_compiler_repair_forward as original

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "libc-dev-bin": "2.39-0ubuntu8.8",
    "libc6-dev": "2.39-0ubuntu8.8",
    "libcrypt-dev": "1:4.4.36-4build1",
    "linux-libc-dev": "6.8.0-137.137",
    "rpcsvc-proto": "1.4.2-0ubuntu7",
    "libc6": "2.39-0ubuntu8.8",
    "libc-bin": "2.39-0ubuntu8.8",
    "locales": "2.39-0ubuntu8.8",
}


def run(argv: list[str], timeout: int = 30):
    return subprocess.run(argv, text=True, capture_output=True, timeout=timeout,
                          check=False, cwd=ROOT)


def packages():
    proc = run([
        "/usr/bin/dpkg-query", "-W",
        "-f=" + "$" + "{binary:Package}\t" +
        "$" + "{Status}\t" + "$" + "{Version}\n"
    ], 25)
    original.guard(proc.returncode == 0, "DPKG_READ_FAILED")
    result = {}
    for row in proc.stdout.splitlines():
        fields = row.split("\t")
        if len(fields) != 3 or fields[1] != "install ok installed":
            continue
        name = fields[0]
        if name.endswith(":amd64"):
            name = name[:-6]
        original.guard(name not in result, "DUPLICATE_PACKAGE_AFTER_ARCH_NORMALIZE")
        result[name] = fields[2]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema": "ttt-hybrid.g7-r23.postinstall-readonly-admission.v1",
        "candidate_sha": args.expected_sha,
        "prior_g7_r22_run": 42359793275,
        "prior_r22_zip_sha256": "408d32d71ce12234ed025362d1b2aff3e0de7f7a834f5ea8053c6916667113e6",
        "utc": datetime.now(timezone.utc).isoformat(),
        "status": "BLOCKED_NOT_STARTED",
        "source_custody": "UNVERIFIED",
        "dpkg_audit": "UNVERIFIED",
        "installed_package_count": None,
        "exact_versions": "UNVERIFIED",
        "compiler": None,
        "shared_c_build": "UNVERIFIED",
        "system_changes": "NONE",
        "gpu_forward": "NOT_RUN",
        "training": "NOT_RUN",
        "deployment": "NOT_AUTHORIZED",
    }
    code = 2
    try:
        target = args.receipt.resolve()
        original.guard(target.parent == (ROOT / "local_runs").resolve() and
                       target.name == "g7_r23_postinstall_preflight.json",
                       "RECEIPT_DESTINATION_DENIED")
        original.source_custody(args.expected_sha)
        report["source_custody"] = "PASS"
        audit = run(["/usr/bin/dpkg", "--audit"], 20)
        original.guard(audit.returncode == 0 and not audit.stdout.strip() and
                       not audit.stderr.strip(), "DPKG_AUDIT_NOT_CLEAN")
        report["dpkg_audit"] = "PASS"
        present = packages()
        report["installed_package_count"] = len(present)
        original.guard(len(present) == 550, "INSTALLED_PACKAGE_COUNT_DRIFT")
        for name, version in EXPECTED.items():
            original.guard(present.get(name) == version,
                           "POSTINSTALL_VERSION_MISMATCH_" + name.replace("-", "_"))
        report["exact_versions"] = "PASS_FIVE_NEW_PLUS_THREE_UNCHANGED_RUNTIME"
        compiler = original.selected_compiler()
        original.guard(compiler is not None, "COMPILER_ABSENT")
        binary, family = compiler
        original.compiler_qualification(binary, family, report)
        report["shared_c_build"] = "PASS"
        report["status"] = "PASS_POSTINSTALL_VERSION_AUDIT_AND_C_LINK"
        code = 0
    except original.Blocked as exc:
        report["status"] = "BLOCKED_" + str(exc)
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["status"] = "BLOCKED_HOST_" + type(exc).__name__
    except Exception as exc:
        report["status"] = "BLOCKED_UNEXPECTED_" + type(exc).__name__
    finally:
        out = args.receipt.resolve()
        if (out.parent == (ROOT / "local_runs").resolve() and
                out.name == "g7_r23_postinstall_preflight.json"):
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
        print("G7_R23_PREFLIGHT=" + report["status"], flush=True)
        print("G7_R23_SYSTEM_CHANGES=NONE", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
