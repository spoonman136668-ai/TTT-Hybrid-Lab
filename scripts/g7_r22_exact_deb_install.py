"""G7-R22: authorized exact-five signed Ubuntu .deb staging and narrow install.

Normal user stages five verified .deb files from a signed snapshot in a bounded
temporary directory. Root mode can install exactly those five files after
independent SHA256/metadata checks. It never upgrades an existing package.
The original R15 -> R6 nonfused GPU forward is a separate normal-user gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g7_r15_compiler_repair_forward as original
import g7_r21_signed_deb_custody as r21

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "libc-dev-bin": ("2.39-0ubuntu8.8", "c894e5a5f137429657d09e853fbbb19d53fc164c60804c396cd43873b0b4f734"),
    "libc6-dev": ("2.39-0ubuntu8.8", "bb8741966e7c1d2e2c0b84bb311717a0908fb563d9b2247b7212710e3cd88b94"),
    "libcrypt-dev": ("1:4.4.36-4build1", "2edff420ef80b4a3f3751e65c33423ef30e563122a58b759e4854ea8d84ba1b1"),
    "linux-libc-dev": ("6.8.0-137.137", "2b2cb11434e5f72703e7133219dc00a59cf2a966d8c4ff0ff0133992369da3f0"),
    "rpcsvc-proto": ("1.4.2-0ubuntu7", "7eb710fe148d224c159ddec1ceb0ba53ead52a80a6793dcdae1474acf20d8f71"),
}
RUN_ID_RE = re.compile(r"^[1-9][0-9]{7,14}$")
RECEIPTS = {"g7_r22_stage.json", "g7_r22_install.json"}


def ensure(condition: bool, code: str) -> None:
    if not condition:
        raise original.Blocked(code)


def call(argv: list[str], *, limit: int, cwd: Path = ROOT):
    return subprocess.run(argv, text=True, capture_output=True, check=False,
                          timeout=limit, cwd=cwd)


def installed_versions():
    values = {}
    for package in ("libc6", "libc-bin", "locales", *sorted(EXPECTED)):
        values[package] = r21.installed(package)
    return values


def full_installed():
    proc = call(
        ["/usr/bin/dpkg-query", "-W",
         "-f=" + "$" + "{binary:Package}" + "\t" +
         "$" + "{Status}" + "\t" + "$" + "{Version}" + "\n"],
        limit=30,
    )
    ensure(proc.returncode == 0, "DPKG_QUERY_FAILED")
    records = {}
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[1] == "install ok installed":
            ensure(parts[0] not in records, "DUPLICATE_DPKG_PACKAGE")
            records[parts[0]] = parts[2]
    return records


def exact_pkg_status(values) -> None:
    for name in ("libc6", "libc-bin", "locales"):
        ensure(values.get(name) == r21.GLIBC, "SYSTEM_GLIBC_VERSION_DRIFT")
    for name in EXPECTED:
        ensure(values.get(name) is None, "NEW_PACKAGE_ALREADY_PRESENT")


def guard_stage_path(run_id: str):
    ensure(RUN_ID_RE.fullmatch(run_id) is not None, "INVALID_RUN_ID")
    directory = Path("/tmp") / ("ttt-g7-r22-stage-" + run_id)
    ensure(directory.parent == Path("/tmp") and not directory.is_symlink(),
           "BAD_STAGE_DIRECTORY")
    return directory


def run_stage(report: dict, directory: Path, expected_sha: str):
    original.source_custody(expected_sha)
    report["source_custody"] = "PASS"
    ensure(os.geteuid() == 1000, "NORMAL_WSL_UID_REQUIRED")
    ensure(not directory.exists(), "STAGING_PATH_ALREADY_EXISTS")
    exact_pkg_status(installed_versions())
    ensure(Path("/usr/share/keyrings/ubuntu-archive-keyring.gpg").is_file(),
           "UBUNTU_KEYRING_MISSING")
    directory.mkdir(mode=0o700)
    report["staging_created"] = True
    sources = directory / "sources"
    sources.mkdir(mode=0o700)
    lists = directory / "lists"
    lists.mkdir(mode=0o700)
    (lists / "partial").mkdir()
    archives = directory / "archives"
    archives.mkdir(mode=0o700)
    (archives / "partial").mkdir()
    downloads = directory / "downloads"
    downloads.mkdir(mode=0o700)
    (sources / "snapshot.sources").write_text(
        "Types: deb\n"
        "URIs: http://archive.ubuntu.com/ubuntu\n"
        "Suites: noble noble-updates\n"
        "Components: main\n"
        "Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg\n"
        f"Snapshot: {r21.SNAPSHOT}\n\n"
        "Types: deb\n"
        "URIs: http://security.ubuntu.com/ubuntu\n"
        "Suites: noble-security\n"
        "Components: main\n"
        "Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg\n"
        f"Snapshot: {r21.SNAPSHOT}\n",
        encoding="utf-8",
    )
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
    update = call(["/usr/bin/apt-get", *opts, "update"], limit=165)
    signed_count = len(list(lists.glob("*_InRelease")))
    ensure(update.returncode == 0 and signed_count >= 3,
           "SIGNED_APT_SNAPSHOT_UNAVAILABLE")
    report["signed_snapshot"] = "PASS"
    report["inrelease_count"] = signed_count
    simulated = call(
        ["/usr/bin/apt-get", *opts, "-s", "install", "-y",
         "--no-install-recommends", "libc6-dev=" + r21.GLIBC,
         "libc-dev-bin=" + r21.GLIBC], limit=60,
    )
    ensure(simulated.returncode == 0, "APT_SIMULATION_FAILED")
    parsed, removed = {}, []
    for line in simulated.stdout.splitlines():
        if line.startswith("Remv "):
            removed.append(line)
        match = re.match(r"^Inst\s+([^\s]+)(?:\s+\[[^\]]+\])?\s+\((\S+)", line)
        if match:
            name, version = match.groups()
            ensure(r21.PKG.fullmatch(name) and r21.VERSION.fullmatch(version) and
                   name not in parsed, "MALFORMED_APT_SIMULATION")
            parsed[name] = version
    ensure(not removed and set(parsed) == set(EXPECTED), "APT_PACKAGE_SCOPE_DRIFT")
    ensure(all(parsed[p] == EXPECTED[p][0] for p in EXPECTED),
           "APT_PACKAGE_VERSION_DRIFT")
    ensure(all(r21.installed(p) is None for p in EXPECTED),
           "APT_NEW_PACKAGE_ALREADY_PRESENT")
    report["apt_simulation"] = "PASS_FIVE_NEW_ZERO_REMOVALS"
    total = 0
    for name, (version, digest) in sorted(EXPECTED.items()):
        report["failed_package"] = name
        record = r21.meta(opts, name, version)
        ensure(record is not None and record["signed_index_sha256"] == digest,
               "SIGNED_INDEX_HASH_DRIFT")
        ensure(total + record["size_bytes"] <= 8 * 1024 * 1024,
               "DOWNLOAD_SIZE_LIMIT")
        pre = set(downloads.glob("*.deb"))
        proc = call(["/usr/bin/apt-get", *opts, "download", f"{name}={version}"],
                    limit=90, cwd=downloads)
        new = set(downloads.glob("*.deb")) - pre
        ensure(proc.returncode == 0 and len(new) == 1,
               "SIGNED_DEB_DOWNLOAD_FAILED")
        payload = next(iter(new))
        ensure(payload.is_file() and not payload.is_symlink(),
               "UNSAFE_DEB_FILE")
        raw = payload.read_bytes()
        ensure(len(raw) == record["size_bytes"] and
               hashlib.sha256(raw).hexdigest() == digest,
               "PAYLOAD_HASH_MISMATCH")
        staged = directory / (name + ".deb")
        payload.rename(staged)
        report["verified_packages"].append({
            "name": name, "version": version,
            "sha256": digest, "bytes": len(raw),
        })
        total += len(raw)
    report["failed_package"] = None
    report["verified_bytes"] = total
    ensure(len(report["verified_packages"]) == 5, "INCOMPLETE_STAGING")
    report["status"] = "PASS_FIVE_SIGNED_DEBS_STAGED"


def verify_root_payloads(directory: Path):
    ensure(os.geteuid() == 0, "ROOT_REQUIRED")
    ensure(directory.is_dir() and not directory.is_symlink(),
           "STAGING_MISSING_OR_SYMLINK")
    existing = {p.name for p in directory.glob("*.deb")}
    ensure(existing == {name + ".deb" for name in EXPECTED},
           "STAGED_PACKAGE_NAMES_DRIFT")
    for name, (version, checksum) in sorted(EXPECTED.items()):
        path = directory / (name + ".deb")
        ensure(path.is_file() and not path.is_symlink() and
               hashlib.sha256(path.read_bytes()).hexdigest() == checksum,
               "ROOT_RECHECK_PACKAGE_HASH_FAILED")
        for field, desired in (("Package", name), ("Version", version),
                               ("Architecture", "amd64")):
            value = call(["/usr/bin/dpkg-deb", "--field", str(path), field], limit=20)
            ensure(value.returncode == 0 and value.stdout.strip() == desired,
                   "DEB_CONTROL_METADATA_MISMATCH")


def run_install(report: dict, directory: Path):
    verify_root_payloads(directory)
    exact_pkg_status(installed_versions())
    audit = call(["/usr/bin/dpkg", "--audit"], limit=20)
    ensure(audit.returncode == 0 and not audit.stdout.strip(), "DPKG_PREAUDIT_FAILED")
    before = full_installed()
    report["preexisting_count"] = len(before)
    names = sorted(EXPECTED)
    report["install_attempted"] = True
    command = ["/usr/bin/dpkg", "--install",
               *[str(directory / (name + ".deb")) for name in names]]
    proc = call(command, limit=150)
    report["dpkg_exit"] = proc.returncode
    after = full_installed()
    report["postexisting_count"] = len(after)
    changed = sorted(
        name for name in set(before) | set(after) if before.get(name) != after.get(name)
    )
    report["changed_packages"] = changed[:30]
    ensure(proc.returncode == 0, "DPKG_INSTALL_FAILED_OR_PARTIAL")
    ensure(changed == names, "UNEXPECTED_EXISTING_PACKAGE_CHANGE")
    for name, (version, _) in EXPECTED.items():
        ensure(r21.installed(name) == version,
               "NEW_PACKAGE_INSTALLED_VERSION_DRIFT")
    for name in ("libc6", "libc-bin", "locales"):
        ensure(r21.installed(name) == r21.GLIBC,
               "SYSTEM_RUNTIME_MODIFIED")
    audit = call(["/usr/bin/dpkg", "--audit"], limit=20)
    ensure(audit.returncode == 0 and not audit.stdout.strip(),
           "DPKG_POSTAUDIT_FAILED")
    report["status"] = "PASS_ONLY_FIVE_NEW_DEVELOPMENT_PACKAGES_INSTALLED"


def run_cleanup(report: dict, directory: Path):
    ensure(os.geteuid() == 1000, "NORMAL_UID_REQUIRED_FOR_CLEANUP")
    ensure(directory.is_dir() and not directory.is_symlink(),
           "STAGE_PATH_UNAVAILABLE_FOR_CLEANUP")
    ensure(directory.stat().st_uid == 1000, "STAGE_OWNER_DRIFT")
    shutil.rmtree(directory)
    report["status"] = "PASS_STAGE_CLEANUP"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("stage", "install", "cleanup"), required=True)
    ap.add_argument("--expected-sha", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    args = ap.parse_args()
    report = {
        "schema": "ttt-hybrid.g7-r22.five-deb-one-shot.v1",
        "phase": args.phase,
        "expected_sha": args.expected_sha,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BLOCKED_NOT_STARTED",
        "source_custody": "UNVERIFIED",
        "signed_snapshot": "NOT_RUN",
        "apt_simulation": "NOT_RUN",
        "verified_packages": [],
        "verified_bytes": 0,
        "failed_package": None,
        "install_attempted": False,
        "dpkg_exit": None,
        "changed_packages": [],
        "system_runtime_expected": r21.GLIBC,
        "gpu_forward": "NOT_RUN",
        "training": "NOT_RUN",
        "deployment": "NOT_AUTHORIZED",
    }
    code = 2
    try:
        directory = guard_stage_path(args.run_id)
        ensure(re.fullmatch(r"[0-9a-f]{40}", args.expected_sha) is not None,
               "INVALID_CANDIDATE_SHA")
        target_name = "g7_r22_" + args.phase + ".json"
        ensure(target_name in ("g7_r22_stage.json", "g7_r22_install.json",
                                "g7_r22_cleanup.json"),
               "UNAUTHORIZED_RECEIPT_NAME")
        target = args.receipt.resolve()
        ensure(target.name == target_name and target.parent ==
               (ROOT / "local_runs").resolve(), "UNAUTHORIZED_RECEIPT_PATH")
        if args.phase == "stage":
            run_stage(report, directory, args.expected_sha)
        elif args.phase == "install":
            run_install(report, directory)
        else:
            run_cleanup(report, directory)
        code = 0
    except original.Blocked as exc:
        report["status"] = "BLOCKED_" + str(exc)
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["status"] = "BLOCKED_HOST_" + type(exc).__name__
    except Exception as exc:
        report["status"] = "BLOCKED_UNEXPECTED_" + type(exc).__name__
    finally:
        target = args.receipt.resolve()
        if target.parent == (ROOT / "local_runs").resolve() and
           target.name in ("g7_r22_stage.json", "g7_r22_install.json",
                           "g7_r22_cleanup.json"):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                              encoding="utf-8")
        print("G7_R22_" + args.phase.upper() + "=" + report["status"], flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
