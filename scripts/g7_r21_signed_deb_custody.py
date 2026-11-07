"""G7-R21: one-use read-only package-byte custody against a signed Ubuntu snapshot.

Requires original project source custody and pinned Ubuntu 24.04 / WSL normal
user. Downloads only five .deb files into an ephemeral private directory and
hashes each against signed Ubuntu archive metadata. No dpkg install, system
apt configuration change, sudo/root, GPU work, or deployment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g7_r15_compiler_repair_forward as previous

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = "20260815T000000Z"
GLIBC = "2.39-0ubuntu8.8"
EXPECTED_NEW = {
    "libc-dev-bin", "libc6-dev", "libcrypt-dev",
    "linux-libc-dev", "rpcsvc-proto",
}
PRIOR_R20_HASHES = {
    "libc6-dev": "bb8741966e7c1d2e2c0b84bb311717a0908fb563d9b2247b7212710e3cd88b94",
    "libc-dev-bin": "c894e5a5f137429657d09e853fbbb19d53fc164c60804c396cd43873b0b4f734",
}
VERSION = re.compile(r"^[0-9][a-zA-Z0-9.:+~_-]*$")
PKG = re.compile(r"^[a-z0-9][a-z0-9+.-]*$")
SHA = re.compile(r"^[0-9a-f]{64}$")
SOURCE_FILE = re.compile(r"^pool/main/[a-z0-9+.-]+/[a-z0-9+._/-]+[.]deb$")
RECEIPT = "g7_r21_signed_package_payloads.json"


def command(argv: list[str], *, timeout: int, cwd: Path = ROOT):
    return subprocess.run(argv, cwd=cwd, capture_output=True,
                          text=True, timeout=timeout, check=False)


def installed(package: str) -> str | None:
    result = command(
        ["/usr/bin/dpkg-query", "-W",
         "-f=" + "$" + "{Status} " + "$" + "{Version}", package],
        timeout=12,
    )
    prefix = "install ok installed "
    raw = result.stdout.strip()
    if result.returncode == 0 and raw.startswith(prefix):
        value = raw[len(prefix):]
        return value if VERSION.fullmatch(value) else None
    return None


def meta(apt_opts: list[str], package: str, version: str):
    response = command(
        ["/usr/bin/apt-cache", *apt_opts, "show", package], timeout=25
    )
    if response.returncode != 0:
        return None
    for stanza in response.stdout.split("\n\n"):
        fields = {}
        for line in stanza.splitlines():
            if ": " in line and not line.startswith(" "):
                key, val = line.split(": ", 1)
                fields[key] = val.strip()
        path = fields.get("Filename", "")
        checksum = fields.get("SHA256", "")
        size = fields.get("Size", "")
        if (fields.get("Package") == package
                and fields.get("Version") == version
                and fields.get("Architecture") == "amd64"
                and SHA.fullmatch(checksum)
                and SOURCE_FILE.fullmatch(path)
                and ".." not in path
                and size.isdecimal() and 0 < int(size) < 32 * 1024 * 1024):
            return {
                "package": package,
                "version": version,
                "filename": path,
                "signed_index_sha256": checksum,
                "size_bytes": int(size),
            }
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected-sha", required=True)
    ap.add_argument("--receipt", required=True, type=Path)
    args = ap.parse_args()
    report = {
        "schema": "ttt-hybrid.g7-r21.signed-deb-byte-custody.v1",
        "candidate_sha": args.expected_sha,
        "prior_g7_r20_run": 42358456909,
        "snapshot": SNAPSHOT,
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BLOCKED_NOT_STARTED",
        "source_custody": "UNVERIFIED",
        "user": "UNVERIFIED",
        "signed_snapshot": "NOT_RUN",
        "inrelease_count": 0,
        "simulation": "NOT_RUN",
        "simulated_versions": {},
        "verified_packages": [],
        "failed_package": None,
        "downloaded_bytes_total": 0,
        "package_downloads_persisted": "NONE",
        "system_changes": "NONE",
        "gpu_forward": "NOT_RUN",
        "scientific": "UNVERIFIED",
        "deployment": "NOT_AUTHORIZED",
    }
    try:
        dst = args.receipt.resolve()
        if dst.parent != (ROOT / "local_runs").resolve() or dst.name != RECEIPT:
            raise previous.Blocked("UNAUTHORIZED_RECEIPT_PATH")
        previous.source_custody(args.expected_sha)
        report["source_custody"] = "PASS"
        if os.geteuid() != 1000:
            raise previous.Blocked("NONROOT_UID_1000_REQUIRED")
        report["user"] = "UID_1000"
        for package in ("libc6", "libc-bin", "locales"):
            if installed(package) != GLIBC:
                raise previous.Blocked("INSTALLED_GLIBC_DRIFT")
        if installed("libc6-dev") is not None or installed("libc-dev-bin") is not None:
            raise previous.Blocked("DEV_PACKAGE_ALREADY_INSTALLED")
        if not Path("/usr/share/keyrings/ubuntu-archive-keyring.gpg").is_file():
            raise previous.Blocked("SIGNING_KEYRING_MISSING")
        with tempfile.TemporaryDirectory(prefix="ttt-g7-r21-snapshot-") as directory:
            root = Path(directory)
            source_dir = root / "sources"
            source_dir.mkdir()
            lists = root / "lists"
            lists.mkdir()
            (lists / "partial").mkdir()
            archives = root / "archives"
            archives.mkdir()
            (archives / "partial").mkdir()
            downloads = root / "downloads"
            downloads.mkdir()
            contents = (
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
            (source_dir / "snapshot.sources").write_text(contents, encoding="utf-8")
            opts = [
                "-o", "Dir::Etc::sourcelist=/dev/null",
                "-o", "Dir::Etc::sourceparts=" + str(source_dir),
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
            updated = command(["/usr/bin/apt-get", *opts, "update"], timeout=160)
            report["inrelease_count"] = len(list(lists.glob("*_InRelease")))
            if updated.returncode != 0 or report["inrelease_count"] < 3:
                raise previous.Blocked("SIGNED_SNAPSHOT_INDEX_FAILED")
            report["signed_snapshot"] = "PASS_SIGNED_INDEX"
            simulated = command(
                ["/usr/bin/apt-get", *opts, "-s", "install",
                 "-y", "--no-install-recommends",
                 "libc6-dev=" + GLIBC, "libc-dev-bin=" + GLIBC],
                timeout=55,
            )
            if simulated.returncode != 0:
                raise previous.Blocked("SNAPSHOT_APT_SIMULATION_FAILED")
            parsed: dict[str, str] = {}
            removals = []
            for line in simulated.stdout.splitlines():
                if line.startswith("Remv "):
                    removals.append(line)
                hit = re.match(
                    r"^Inst\s+([^\s]+)(?:\s+\[[^\]]+\])?\s+\((\S+)", line
                )
                if hit:
                    package, version = hit.groups()
                    if (not PKG.fullmatch(package) or
                            not VERSION.fullmatch(version) or
                            package in parsed):
                        raise previous.Blocked("MALFORMED_PACKAGE_SIMULATION")
                    parsed[package] = version
            if removals or set(parsed) != EXPECTED_NEW:
                raise previous.Blocked("SIMULATED_PACKAGE_SCOPE_DRIFT")
            for package in sorted(parsed):
                if installed(package) is not None:
                    raise previous.Blocked("SIMULATION_CHANGES_EXISTING_PACKAGE")
            if parsed["libc6-dev"] != GLIBC or parsed["libc-dev-bin"] != GLIBC:
                raise previous.Blocked("GLIBC_PACKAGE_VERSION_DRIFT")
            report["simulation"] = "PASS_FIVE_NEW_NO_UPGRADES_OR_REMOVALS"
            report["simulated_versions"] = dict(sorted(parsed.items()))
            total = 0
            for package, version in sorted(parsed.items()):
                report["failed_package"] = package
                record = meta(opts, package, version)
                if record is None:
                    raise previous.Blocked("MISSING_SIGNED_PACKAGE_HASH")
                if (package in PRIOR_R20_HASHES
                        and record["signed_index_sha256"] != PRIOR_R20_HASHES[package]):
                    raise previous.Blocked("PRIOR_R20_INDEX_HASH_DRIFT")
                if total + record["size_bytes"] > 64 * 1024 * 1024:
                    raise previous.Blocked("MAX_DOWNLOAD_SIZE_EXCEEDED")
                before = {x.name for x in downloads.glob("*.deb")}
                got = command(
                    ["/usr/bin/apt-get", *opts, "download",
                     package + "=" + version],
                    timeout=85, cwd=downloads,
                )
                after = {x.name for x in downloads.glob("*.deb")}
                new = after - before
                if got.returncode != 0 or len(new) != 1:
                    raise previous.Blocked("EXACT_VERSION_DEB_DOWNLOAD_FAILED")
                candidate = downloads / next(iter(new))
                if not candidate.is_file() or candidate.is_symlink():
                    raise previous.Blocked("INVALID_DOWNLOADED_FILE")
                raw = candidate.read_bytes()
                actual = hashlib.sha256(raw).hexdigest()
                if (len(raw) != record["size_bytes"]
                        or actual != record["signed_index_sha256"]):
                    raise previous.Blocked("SIGNED_INDEX_PAYLOAD_MISMATCH")
                report["verified_packages"].append({
                    "package": package,
                    "version": version,
                    "index_sha256": record["signed_index_sha256"],
                    "actual_sha256": actual,
                    "verified_bytes": len(raw),
                })
                total += len(raw)
                report["downloaded_bytes_total"] = total
                candidate.unlink()
            report["failed_package"] = None
            if len(report["verified_packages"]) != 5:
                raise previous.Blocked("INCOMPLETE_CUSTODY")
            report["status"] = "PASS_FIVE_SIGNED_DEB_BYTES_EXACT_MATCH"
            return 0
    except previous.Blocked as exc:
        report["status"] = "BLOCKED_" + str(exc)
        return 2
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["status"] = "BLOCKED_HOST_" + type(exc).__name__
        return 2
    except Exception as exc:
        report["status"] = "BLOCKED_UNEXPECTED_" + type(exc).__name__
        return 2
    finally:
        receipt = args.receipt.resolve()
        if receipt.name == RECEIPT and receipt.parent == (ROOT / "local_runs").resolve():
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
        print("G7_R21_STATUS=" + report["status"], flush=True)
        print("G7_R21_SYSTEM_CHANGES=NONE", flush=True)
        print("G7_R21_GPU=NOT_RUN", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
