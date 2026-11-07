"""G7-R17 one-shot, unprivileged, sanitized C linker diagnostic.

This file does not install anything or run a model. No secrets, raw stderr,
environment dumps, or arbitrary file paths are recorded in its receipt.
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
import g7_r15_compiler_repair_forward as prior

ROOT = Path(__file__).resolve().parents[1]
RECEIPT_NAMES = {"g7_r17_linker_pre.json", "g7_r17_linker_post.json"}


def execute(argv: list[str], *, input_text: str | None = None, seconds: int = 30):
    return subprocess.run(argv, input=input_text, text=True,
                          capture_output=True, timeout=seconds, check=False,
                          cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "schema": "ttt-hybrid.g7-r17-linker-probe.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_sha": args.expected_sha,
        "status": "BLOCKED_NOT_STARTED",
        "source_custody": "UNVERIFIED",
        "compiler_family": None,
        "compiler_sha256": None,
        "compiler_version": None,
        "libc6_dev_installed": None,
        "crti_present": None,
        "crtn_present": None,
        "stdio_header_present": None,
        "linker_markers": [],
        "shared_library": "NOT_RUN",
        "training": "NOT_RUN",
        "deployment": "NOT_AUTHORIZED",
    }
    result = 2
    try:
        receipt = args.receipt.resolve()
        if receipt.name not in RECEIPT_NAMES or receipt.parent != (ROOT / "local_runs").resolve():
            raise prior.Blocked("RECEIPT_DESTINATION_DENIED")
        prior.source_custody(args.expected_sha)
        report["source_custody"] = "PASS"
        found = prior.selected_compiler()
        if found is None:
            raise prior.Blocked("GCC_COMPILER_ABSENT")
        compiler, family = found
        report["compiler_family"] = family
        report["compiler_sha256"] = hashlib.sha256(Path(compiler).read_bytes()).hexdigest()
        v = execute([compiler, "-dumpfullversion"], seconds=12)
        if v.returncode != 0:
            raise prior.Blocked("COMPILER_VERSION_UNAVAILABLE")
        report["compiler_version"] = v.stdout.strip()[:32]

        pkg = execute(["/usr/bin/dpkg-query", "-W", "-f=${Status}",
                       "libc6-dev"], seconds=12)
        report["libc6_dev_installed"] = (
            pkg.returncode == 0 and pkg.stdout.strip() == "install ok installed"
        )
        report["crti_present"] = Path("/usr/lib/x86_64-linux-gnu/crti.o").is_file()
        report["crtn_present"] = Path("/usr/lib/x86_64-linux-gnu/crtn.o").is_file()
        report["stdio_header_present"] = Path("/usr/include/stdio.h").is_file()

        with tempfile.TemporaryDirectory(prefix="ttt-g7-r17-link-") as temp:
            target = Path(temp) / "host_helper.so"
            p = execute([compiler, "-x", "c", "-", "-shared", "-fPIC",
                         "-o", str(target)],
                        input_text="int g7_r17_compiler_probe(void){return 1;}\n",
                        seconds=35)
            message = (p.stderr or "").lower()[:6000]
            markers = {
                "MISSING_CRTI": "cannot find crti.o" in message,
                "MISSING_CRTN": "cannot find crtn.o" in message,
                "MISSING_LIBC": "cannot find -lc" in message,
                "MISSING_STDIO": "stdio.h: no such file" in message,
                "PERMISSION": "permission denied" in message,
                "DISK_FULL": "no space left on device" in message,
                "LINKER_ERROR": "ld returned" in message,
            }
            report["linker_markers"] = sorted(k for k, v in markers.items() if v)
            if p.returncode == 0 and target.is_file() and target.stat().st_size > 0:
                report["status"] = "PASS_HOST_C_SHARED_LINK"
                report["shared_library"] = "PASS"
                result = 0
            else:
                missing_objects = not report["crti_present"] or not report["crtn_present"]
                missing_libc_marker = (markers["MISSING_CRTI"]
                                       or markers["MISSING_CRTN"]
                                       or markers["MISSING_LIBC"])
                if (not report["libc6_dev_installed"] and missing_objects
                        and missing_libc_marker and not markers["PERMISSION"]
                        and not markers["DISK_FULL"]):
                    report["status"] = "NEEDS_LIBC6_DEV"
                    report["shared_library"] = "FAIL_MISSING_C_RUNTIME_DEV"
                    result = 3
                else:
                    report["status"] = "BLOCKED_UNQUALIFIED_LINK_FAILURE"
                    report["shared_library"] = "FAIL"
    except prior.Blocked as exc:
        report["status"] = "BLOCKED_" + str(exc)
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["status"] = "BLOCKED_HOST_PROBE_" + type(exc).__name__
    except Exception as exc:
        report["status"] = "BLOCKED_UNEXPECTED_" + type(exc).__name__
    finally:
        dest = args.receipt.resolve()
        if dest.name in RECEIPT_NAMES and dest.parent == (ROOT / "local_runs").resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
        print("G7_R17_LINK_STATUS=" + report["status"], flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
