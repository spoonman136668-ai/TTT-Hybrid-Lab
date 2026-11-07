"""G7-R19: inspect existing Ubuntu glibc versions and simulate a matching libc6-dev.

Bounded, unprivileged, read-only package queries. Does NOT install, upgrade,
update package indexes, invoke a GPU, train or deploy anything.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g7_r15_compiler_repair_forward as original

ROOT = Path(__file__).resolve().parents[1]
VERSION_RE = re.compile(r"[0-9][A-Za-z0-9.+:~\\-]*\\Z")
PACKAGE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+:-]*\\Z")
PROTECTED = {"libc6", "libc-bin", "locales"}
ALLOWED = {"libc6-dev", "libc-dev-bin", "linux-libc-dev",
           "libcrypt-dev", "rpcsvc-proto", "libnsl-dev", "libtirpc-dev"}

def execute(argv: list[str], seconds: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, text=True, capture_output=True, timeout=seconds,
                          check=False, cwd=ROOT)

def installed(name: str) -> str | None:
    p = execute(["/usr/bin/dpkg-query", "-W", "-f=${Version}", name], seconds=15)
    version = p.stdout.strip()
    return version if p.returncode == 0 and VERSION_RE.fullmatch(version) else None

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema": "ttt-hybrid.g7-r19.readonly-libc-version-match.v1",
        "candidate_sha": args.expected_sha,
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "prior_g7_r18_run": 42357126213,
        "source_custody": "UNVERIFIED",
        "status": "BLOCKED_NOT_STARTED",
        "installed": {},
        "available_matching_libc6_dev": False,
        "available_libc6_dev_versions": [],
        "simulation_status": "NOT_RUN",
        "simulation_exit": None,
        "simulation_inst": [],
        "simulation_remv": [],
        "out_of_scope": [],
        "package_changes": "NONE",
        "gpu_forward": "NOT_RUN",
        "training": "NOT_RUN",
        "deployment": "NOT_AUTHORIZED",
    }
    try:
        dst = args.receipt.resolve()
        if dst.name != "g7_r19_libc_versions.json" or dst.parent != (ROOT / "local_runs").resolve():
            raise original.Blocked("INVALID_RECEIPT_LOCATION")
        original.source_custody(args.expected_sha)
        report["source_custody"] = "PASS"
        for name in ("libc6", "libc-bin", "locales", "libc6-dev", "libc-dev-bin"):
            report["installed"][name] = installed(name)
        libc = report["installed"]["libc6"]
        libc_bin = report["installed"]["libc-bin"]
        if not libc or not libc_bin:
            raise original.Blocked("LIBC_VERSION_UNAVAILABLE")
        if not VERSION_RE.fullmatch(libc):
            raise original.Blocked("INVALID_LIBC_VERSION")
        madison = execute(["/usr/bin/apt-cache", "madison", "libc6-dev"], seconds=20)
        if madison.returncode != 0:
            raise original.Blocked("APT_CANDIDATE_QUERY_FAILED")
        versions = []
        for line in madison.stdout.splitlines():
            match = re.match(r"^\\s*libc6-dev\\s*\\|\\s*([^|\\s]+)\\s*\\|", line)
            if match and VERSION_RE.fullmatch(match.group(1)):
                versions.append(match.group(1))
        report["available_libc6_dev_versions"] = sorted(set(versions))[:24]
        report["available_matching_libc6_dev"] = libc in versions
        if not report["available_matching_libc6_dev"]:
            report["status"] = "NO_EXACT_MATCHING_LIBC6_DEV_IN_APT_INDEX"
            return 0
        simulated = execute(
            ["/usr/bin/apt-get", "-s", "-o", "Debug::NoLocking=1",
             "-o", "Acquire::Retries=0", "install", "-y", "--no-install-recommends",
             "libc6-dev=" + libc],
            seconds=55,
        )
        report["simulation_exit"] = simulated.returncode
        install = []
        remove = []
        for line in simulated.stdout.splitlines():
            for marker, target in (("Inst ", install), ("Remv ", remove)):
                if line.startswith(marker):
                    part = line[len(marker):].split(None, 1)[0]
                    if PACKAGE_RE.fullmatch(part):
                        target.append(part)
        report["simulation_inst"] = sorted(set(install))[:64]
        report["simulation_remv"] = sorted(set(remove))[:64]
        report["out_of_scope"] = [p for p in report["simulation_inst"]
                                  if p not in ALLOWED]
        if simulated.returncode != 0:
            report["status"] = "PINNED_SIMULATION_FAILED_NO_MUTATION"
        elif len(install) > 64 or len(remove) > 64:
            report["status"] = "SIMULATED_SCOPE_TOO_LARGE_NO_MUTATION"
        elif remove:
            report["status"] = "PINNED_SIMULATION_REMOVALS_NO_MUTATION"
        elif any(x in PROTECTED for x in install):
            report["status"] = "PINNED_SIMULATION_CHANGES_SYSTEM_LIBC_NO_MUTATION"
        elif report["out_of_scope"]:
            report["status"] = "PINNED_SIMULATION_OUT_OF_SCOPE_NO_MUTATION"
        elif "libc6-dev" not in install and report["installed"]["libc6-dev"] is None:
            report["status"] = "PINNED_SIMULATION_MISSING_TARGET_NO_MUTATION"
        else:
            report["status"] = "PASS_SAFE_EXACT_VERSION_SIMULATION"
        report["simulation_status"] = report["status"]
        return 0
    except original.Blocked as exc:
        report["status"] = "BLOCKED_" + str(exc)
        return 2
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["status"] = "BLOCKED_" + type(exc).__name__
        return 2
    except Exception as exc:
        report["status"] = "BLOCKED_UNEXPECTED_" + type(exc).__name__
        return 2
    finally:
        dest = args.receipt.resolve()
        if dest.name == "g7_r19_libc_versions.json" and dest.parent == (ROOT / "local_runs").resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(report, sort_keys=True, indent=2) + "\\n",
                            encoding="utf-8")
        print("G7_R19_STATUS=" + report["status"], flush=True)
        print("G7_R19_SOURCE_CUSTODY=" + report["source_custody"], flush=True)
        print("G7_R19_PACKAGE_CHANGES=NONE", flush=True)
        print("G7_R19_GPU=NOT_RUN", flush=True)

if __name__ == "__main__":
    raise SystemExit(main())
