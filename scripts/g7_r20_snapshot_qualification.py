"""G7-R20 one-use read-only Ubuntu signed snapshot qualification.

APT writes package *indexes* only inside a temporary directory as the normal WSL
user. No system apt configuration edits, package installs/upgrades, model loads,
GPU execution, or privilege escalation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g7_r15_compiler_repair_forward as original

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = "20260815T000000Z"
GLIBC_VERSION = "2.39-0ubuntu8.8"
PROTECTED_PACKAGES = {"libc6", "libc-bin", "locales"}
ALLOWED_NEW_PACKAGES = {
    "libc6-dev", "libc-dev-bin", "linux-libc-dev",
    "libcrypt-dev", "rpcsvc-proto", "libnsl-dev", "libtirpc-dev",
}
RECEIPT_NAME = "g7_r20_snapshot_qualification.json"
NAME = re.compile(r"^[a-z0-9][a-z0-9+.-]*(?::[a-z0-9-]+)?$")
HEX = re.compile(r"^[0-9a-f]{64}$")


def command(args: list[str], *, timeout: int):
    return subprocess.run(
        args, text=True, capture_output=True, timeout=timeout,
        cwd=ROOT, check=False,
    )


def installed(name: str) -> str | None:
    r = command(
        ["/usr/bin/dpkg-query", "-W", "-f=" + "${Status} ${Version}", name],
        timeout=10,
    )
    out = r.stdout.strip()
    if r.returncode == 0 and out.startswith("install ok installed "):
        return out.removeprefix("install ok installed ").strip()
    return None


def metadata_from_signed_apt(cache_cmd: list[str], name: str, version: str):
    r = command(["/usr/bin/apt-cache", *cache_cmd, "show", name], timeout=30)
    if r.returncode != 0:
        return None
    for stanza in r.stdout.split("\n\n"):
        entries = {}
        for ln in stanza.splitlines():
            if ": " in ln and not ln.startswith(" "):
                k, value = ln.split(": ", 1)
                entries[k] = value.strip()
        if (entries.get("Package") == name
                and entries.get("Version") == version
                and entries.get("Architecture") == "amd64"
                and HEX.fullmatch(entries.get("SHA256", ""))):
            filename = entries.get("Filename", "")
            if not filename.startswith("pool/main/g/glibc/"):
                continue
            return {
                "package": name,
                "version": version,
                "architecture": "amd64",
                "index_sha256": entries["SHA256"],
                "filename": filename,
                "size_bytes": int(entries.get("Size", "0")),
            }
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected-sha", required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    args = ap.parse_args()
    report = {
        "schema": "ttt-hybrid.g7-r20.signed-snapshot-readonly.v1",
        "candidate_sha": args.expected_sha,
        "snapshot_utc": SNAPSHOT,
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BLOCKED_NOT_STARTED",
        "source_custody": "UNVERIFIED",
        "normal_user": "UNVERIFIED",
        "installed_glibc": {},
        "signed_snapshot_update": "NOT_RUN",
        "signed_release_count": 0,
        "index_sha256": {},
        "apt_simulation": "NOT_RUN",
        "apt_simulation_exit": None,
        "simulated_installs": [],
        "simulated_removals": [],
        "changes_to_existing_packages": [],
        "out_of_scope": [],
        "actual_system_changes": "NONE",
        "package_downloads": "NONE",
        "gpu_forward": "NOT_RUN",
        "scientific": "UNVERIFIED",
        "deployment": "NOT_AUTHORIZED",
    }
    try:
        dest = args.receipt.resolve()
        if (dest.parent != (ROOT / "local_runs").resolve()
                or dest.name != RECEIPT_NAME):
            raise original.Blocked("INVALID_RECEIPT_DESTINATION")
        original.source_custody(args.expected_sha)
        report["source_custody"] = "PASS"
        if os.geteuid() != 1000:
            raise original.Blocked("EXPECTED_NONROOT_UID_1000")
        report["normal_user"] = "UID_1000"
        for package in ("libc6", "libc-bin", "locales", "libc6-dev", "libc-dev-bin"):
            report["installed_glibc"][package] = installed(package)
        if (report["installed_glibc"]["libc6"] != GLIBC_VERSION
                or report["installed_glibc"]["libc-bin"] != GLIBC_VERSION):
            raise original.Blocked("EXISTING_GLIBC_VERSION_DRIFT")
        keyring = Path("/usr/share/keyrings/ubuntu-archive-keyring.gpg")
        if not keyring.is_file():
            raise original.Blocked("UBUNTU_SIGNING_KEYRING_MISSING")

        with tempfile.TemporaryDirectory(prefix="ttt-g7-r20-apt-") as d:
            base = Path(d)
            sources = base / "sources.list.d"
            sources.mkdir()
            lists = base / "lists"
            lists.mkdir()
            (lists / "partial").mkdir()
            archives = base / "archives"
            archives.mkdir()
            (archives / "partial").mkdir()
            snapshot_contents = (
                "Types: deb\n"
                "URIs: http://archive.ubuntu.com/ubuntu\n"
                "Suites: noble noble-updates\n"
                "Components: main\n"
                "Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg\n"
                f"Snapshot: {SNAPSHOT}\n\n"
                "Types: deb\n"
                "URIs: http://security.ubuntu.com/ubuntu\n"
                "Suites: noble-security\n"
                "Components: main\n"
                "Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg\n"
                f"Snapshot: {SNAPSHOT}\n"
            )
            (sources / "ubuntu.sources").write_text(snapshot_contents, encoding="utf-8")
            opts = [
                "-o", "Dir::Etc::sourcelist=/dev/null",
                "-o", "Dir::Etc::sourceparts=" + str(sources),
                "-o", "Dir::State::lists=" + str(lists),
                "-o", "Dir::Cache::archives=" + str(archives),
                "-o", "APT::Sandbox::User=",
                "-o", "Acquire::Retries=0",
                "-o", "Acquire::Languages=none",
                "-o", "Acquire::AllowInsecureRepositories=false",
                "-o", "Acquire::AllowDowngradeToInsecureRepositories=false",
                "-o", "APT::Get::AllowUnauthenticated=false",
                "-o", "APT::Get::List-Cleanup=0",
            ]
            apt_update = command(["/usr/bin/apt-get", *opts, "update"], timeout=155)
            signed_files = sorted(lists.glob("*_InRelease"))
            report["signed_release_count"] = len(signed_files)
            if apt_update.returncode != 0 or len(signed_files) < 3:
                report["status"] = "BLOCKED_SNAPSHOT_SIGNED_INDEX_UNAVAILABLE"
                return 2
            report["signed_snapshot_update"] = "PASS_SIGNED_APT_INDEX"
            for package in ("libc6-dev", "libc-dev-bin"):
                meta = metadata_from_signed_apt(opts, package, GLIBC_VERSION)
                if meta is None:
                    report["status"] = "BLOCKED_MATCHING_GLIBC_DEV_INDEX_ABSENT"
                    return 2
                report["index_sha256"][package] = meta
            # This is only a simulation against the currently installed dpkg status.
            simulate = command([
                "/usr/bin/apt-get", *opts,
                "-s", "install", "-y", "--no-install-recommends",
                "libc6-dev=" + GLIBC_VERSION,
                "libc-dev-bin=" + GLIBC_VERSION,
            ], timeout=60)
            report["apt_simulation_exit"] = simulate.returncode
            installs, removes = set(), set()
            for line in simulate.stdout.splitlines():
                for marker, destset in (("Inst ", installs), ("Remv ", removes)):
                    if line.startswith(marker):
                        candidate = line[len(marker):].split(maxsplit=1)[0]
                        if NAME.fullmatch(candidate):
                            destset.add(candidate)
            report["simulated_installs"] = sorted(installs)
            report["simulated_removals"] = sorted(removes)
            report["out_of_scope"] = sorted(installs - ALLOWED_NEW_PACKAGES)
            report["changes_to_existing_packages"] = sorted(
                name for name in installs if installed(name) is not None
            )
            if simulate.returncode != 0:
                report["status"] = "BLOCKED_SNAPSHOT_PINNED_SIMULATION_FAILED"
            elif removes:
                report["status"] = "BLOCKED_SNAPSHOT_REQUIRES_REMOVALS"
            elif report["out_of_scope"]:
                report["status"] = "BLOCKED_SNAPSHOT_OUT_OF_SCOPE"
            elif report["changes_to_existing_packages"]:
                report["status"] = "BLOCKED_SNAPSHOT_CHANGES_EXISTING_PACKAGES"
            elif "libc6-dev" not in installs:
                report["status"] = "BLOCKED_SNAPSHOT_MISSING_LIBC6_DEV_TARGET"
            else:
                report["status"] = "PASS_SIGNED_SNAPSHOT_EXACT_VERSION_SIMULATION"
                report["apt_simulation"] = "PASS_READONLY"
                return 0
            report["apt_simulation"] = report["status"]
            return 2
    except original.Blocked as exc:
        report["status"] = "BLOCKED_" + str(exc)
        return 2
    except (subprocess.TimeoutExpired, OSError) as exc:
        report["status"] = "BLOCKED_HOST_" + type(exc).__name__
        return 2
    except Exception as exc:
        report["status"] = "BLOCKED_UNEXPECTED_" + type(exc).__name__
        return 2
    finally:
        path = args.receipt.resolve()
        if path.parent == (ROOT / "local_runs").resolve() and path.name == RECEIPT_NAME:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                            encoding="utf-8")
        print("G7_R20_STATUS=" + report["status"], flush=True)
        print("G7_R20_CHANGES=NONE", flush=True)
        print("G7_R20_GPU=NOT_RUN", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
