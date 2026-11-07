"""G7-S8 signed temporary Python headers and equal-episode three-arm native scientific comparison.

Read-only system: APT indexes/downloads/unpacking live only in an ephemeral
normal-user directory. Only if source custody, signed payload hash, exact ABI
package version, and negative/positive Python.h C builds pass will one original
bounded nonfused GPU forward run with local CPATH and LIBRARY_PATH.
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
SNAPSHOT = "20260901T000000Z"
GCC_SHA = "1b99826121ae6682a634e5efe09bd3e3df58ce58e0b28f849114ab5b89139c26"
HEX = re.compile(r"^[0-9a-f]{64}$")
VERSION = re.compile(r"^[0-9][A-Za-z0-9.:+~_-]{1,100}$")


def guard(condition: bool, reason: str):
    if not condition:
        raise previous.Blocked(reason)


def execute(argv, *, timeout=30, cwd=ROOT, env=None, source=None):
    return subprocess.run(argv, input=source, text=True, capture_output=True,
                          check=False, cwd=cwd, env=env, timeout=timeout)


def version_installed(package):
    fmt = "-f=" + chr(36) + "{Status} " + chr(36) + "{Version}"
    r = execute(["/usr/bin/dpkg-query", "-W", fmt, package], timeout=12)
    prefix = "install ok installed "
    s = r.stdout.strip()
    return s[len(prefix):] if r.returncode == 0 and s.startswith(prefix) else None


def dpkg_hash():
    fmt = ("-f=" + chr(36) + "{binary:Package}\t" +
           chr(36) + "{Status}\t" + chr(36) + "{Version}\n")
    p = execute(["/usr/bin/dpkg-query", "-W", fmt], timeout=25)
    guard(p.returncode == 0, "DPKG_READ_FAILED")
    return hashlib.sha256(p.stdout.encode("utf-8")).hexdigest()


def signed_package(opts, package, exact_version):
    p = execute(["/usr/bin/apt-cache", *opts, "show", package], timeout=25)
    guard(p.returncode == 0, "SIGNATURE_INDEX_PACKAGE_UNAVAILABLE")
    versions = set()
    found = None
    for paragraph in p.stdout.split("\n\n"):
        fields = {}
        for line in paragraph.splitlines():
            if ": " in line and not line.startswith(" "):
                key, value = line.split(": ", 1)
                fields[key] = value.strip()
        if fields.get("Package") != package or fields.get("Architecture") != "amd64":
            continue
        version = fields.get("Version", "")
        if VERSION.fullmatch(version):
            versions.add(version)
        if version != exact_version:
            continue
        digest = fields.get("SHA256", "")
        filename = fields.get("Filename", "")
        size = fields.get("Size", "")
        guard(HEX.fullmatch(digest) is not None, "SIGNED_INDEX_SHA_MISSING")
        guard(filename.startswith("pool/main/p/python3.12/")
              and filename.endswith(".deb") and ".." not in filename,
              "SIGNED_INDEX_FILENAME_DENIED")
        guard(size.isdecimal() and 0 < int(size) < 16 * 1024 * 1024,
              "SIGNED_INDEX_SIZE_DENIED")
        found = {"sha256": digest, "bytes": int(size),
                 "version": version}
        break
    return found, sorted(versions)[:16]


def c_header_link(cc, env, output):
    test = ("#include <Python.h>\n"
            "int g7_r32_probe(void){return PY_MAJOR_VERSION*100+PY_MINOR_VERSION;}\n")
    p = execute([cc, "-x", "c", "-", "-shared", "-fPIC",
                 "-o", str(output), "-lcuda"],
                source=test, env=env, timeout=30)
    error = (p.stderr or "").lower()[:4096]
    errors = {
        "MISSING_PYTHON_H": ("python.h: no such file" in error
                             or "fatal error: python.h" in error),
        "MISSING_LIBCUDA": "cannot find -lcuda" in error,
        "MISSING_CRT": "cannot find crti.o" in error,
        "UNDEFINED_SYMBOL": "undefined reference" in error,
        "MISSING_MULTIARCH_PYCONFIG": ("pyconfig.h" in error and "no such file" in error),
    }
    return {
        "pass": p.returncode == 0 and output.is_file() and output.stat().st_size > 0,
        "exit": p.returncode if 0 <= p.returncode <= 255 else None,
        "errors": sorted(k for k, v in errors.items() if v),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    result = {
        "schema": "ttt-hybrid.g7-s8.signed-ephemeral-equal-exposure-admission.v1",
        "candidate_sha": args.expected_sha,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "previous_r31_run": 42362001058,
        "previous_r34_run": 42364329017,
        "qualifying_r35_run": 42364516609,
        "scientific": "UNVERIFIED",
        "science_status": "NOT_RUN",
        "science_metric": None,
        "science_receipt_sha256": None,
        "status": "BLOCKED_NOT_STARTED",
        "qualifying_s6_run": 42368703057,
        "s5_science_null_run": 42367320764,
        "s8_native_arms": 2,
        "s8_conventional_arms": 1,
        "s8_positive_control_run": 42368703057,
        "source_custody": "UNVERIFIED",
        "python_abi_debian_version": None,
        "snapshot": SNAPSHOT,
        "signed_releases": 0,
        "available_libpython3_12_dev_versions": [],
        "exact_signed_header_metadata": "UNVERIFIED",
        "package_sha256": None,
        "actual_package_sha256": None,
        "extracted_python_h": False,
        "negative_control": None,
        "positive_control": None,
        "native_forward": "NOT_RUN",
        "native_status": None,
        "native_gate3_status": None,
        "native_failure_category": None,
        "native_failure_type": None,
        "native_receipt_sha256": None,
        "dpkg_before_sha256": None,
        "dpkg_after_sha256": None,
        "installed_packages_changed": "UNVERIFIED",
        "training": "NOT_RUN",
        "science": "UNVERIFIED",
        "deployment": "NOT_AUTHORIZED",
    }
    exit_code = 2
    try:
        dest = args.receipt.resolve()
        guard(dest.parent == (ROOT / "local_runs").resolve()
              and dest.name == "g7_s8_ephemeral_runtime_admission.json",
              "RECEIPT_DESTINATION_DENIED")
        previous.source_custody(args.expected_sha)
        result["source_custody"] = "PASS"
        guard(os.geteuid() == 1000 and sys.version_info[:2] == (3, 12),
              "PYTHON_UID_DRIFT")
        guard(not Path("/usr/include/python3.12/Python.h").is_file(),
              "SYSTEM_PYTHON_HEADERS_ALREADY_PRESENT")
        python_version = version_installed("python3.12-minimal")
        stdlib_version = version_installed("libpython3.12-stdlib")
        result["python_abi_debian_version"] = python_version
        guard(python_version is not None and VERSION.fullmatch(python_version),
              "PYTHON_ABI_PACKAGE_VERSION_UNAVAILABLE")
        guard(python_version == stdlib_version,
              "PYTHON_STDLIB_ABI_MISMATCH")
        guard(version_installed("libpython3.12-dev") is None,
              "LIBPYTHON_DEV_ALREADY_INSTALLED")
        compiler = previous.selected_compiler()
        guard(compiler is not None and compiler[1] == "gcc",
              "PINNED_GCC_UNAVAILABLE")
        binary = compiler[0]
        guard(hashlib.sha256(Path(binary).read_bytes()).hexdigest() == GCC_SHA,
              "PINNED_GCC_SHA_MISMATCH")
        guard(Path("/usr/lib/wsl/lib/libcuda.so").is_file(),
              "WSL_LIBCUDA_FILE_ABSENT")
        guard(Path("/usr/share/keyrings/ubuntu-archive-keyring.gpg").is_file(),
              "UBUNTU_SIGNING_KEYRING_MISSING")
        result["dpkg_before_sha256"] = dpkg_hash()
        with tempfile.TemporaryDirectory(prefix="ttt-g7-r32-ephemeral-pydev-") as d:
            base = Path(d)
            sources = base / "sources"
            sources.mkdir()
            lists = base / "lists"
            lists.mkdir()
            (lists / "partial").mkdir()
            archives = base / "archives"
            archives.mkdir()
            (archives / "partial").mkdir()
            downloads = base / "downloads"
            downloads.mkdir()
            extraction = base / "extract"
            extraction.mkdir()
            (sources / "ubuntu.sources").write_text(
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
                f"Snapshot: {SNAPSHOT}\n",
                encoding="utf-8",
            )
            options = [
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
            p = execute(["/usr/bin/apt-get", *options, "update"], timeout=160)
            result["signed_releases"] = len(list(lists.glob("*_InRelease")))
            guard(p.returncode == 0 and result["signed_releases"] >= 3,
                  "SIGNED_UBUNTU_INDEX_UNAVAILABLE")
            package, versions = signed_package(
                options, "libpython3.12-dev", python_version)
            result["available_libpython3_12_dev_versions"] = versions
            guard(package is not None, "MATCHING_SIGNED_PYTHON_DEV_NOT_IN_SNAPSHOT")
            guard(package["sha256"] == "864360533639b45256258475c7843ac7c56f9bf556b356753f21f7e3012fea67",
                  "PREVIOUS_R34_SIGNED_PACKAGE_SHA_DRIFT")
            result["exact_signed_header_metadata"] = "PASS_MATCH_INSTALLED_ABI"
            result["package_sha256"] = package["sha256"]
            dl = execute(["/usr/bin/apt-get", *options, "download",
                          "libpython3.12-dev=" + python_version],
                         timeout=90, cwd=downloads)
            found = list(downloads.glob("*.deb"))
            guard(dl.returncode == 0 and len(found) == 1,
                  "SIGNED_PYTHON_HEADER_DOWNLOAD_FAILED")
            pth = found[0]
            guard(pth.is_file() and not pth.is_symlink(), "UNSAFE_HEADER_PAYLOAD")
            contents = pth.read_bytes()
            calculated = hashlib.sha256(contents).hexdigest()
            result["actual_package_sha256"] = calculated
            guard(len(contents) == package["bytes"]
                  and calculated == package["sha256"],
                  "SIGNED_PYTHON_HEADER_PAYLOAD_DRIFT")
            unpack = execute(["/usr/bin/dpkg-deb", "--extract",
                              str(pth), str(extraction)], timeout=75)
            guard(unpack.returncode == 0, "TEMP_HEADER_UNPACK_FAILED")
            include = extraction / "usr/include/python3.12"
            guard((include / "Python.h").is_file()
                  and (include / "pyconfig.h").is_file()
                  and (extraction / "usr/include/x86_64-linux-gnu/python3.12/pyconfig.h").is_file(),
                  "TEMP_PYTHON_HEADER_CONTENTS_INCOMPLETE")
            result["extracted_python_h"] = True
            env = dict(os.environ)
            env["LIBRARY_PATH"] = "/usr/lib/wsl/lib"
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            negative = c_header_link(binary, env, base / "negative.so")
            result["negative_control"] = negative
            guard(not negative["pass"]
                  and "MISSING_PYTHON_H" in negative["errors"],
                  "NEGATIVE_CONTROL_INVALID")
            scoped = dict(env)
            scoped["CPATH"] = str(include) + os.pathsep + str(extraction / "usr/include")
            # Torch deterministic CUDA GEMMs require this process-local
            # cuBLAS workspace setting. No global or system environment write.
            scoped["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            positive = c_header_link(binary, scoped, base / "positive.so")
            result["positive_control"] = positive
            guard(positive["pass"], "TEMP_PYTHON_HEADER_COMPILE_FAILED")
            # SCIENCE pilot reuses the already-qualified signed package and
            # dual multiarch CPATH. Original official sources are imported
            # only inside the pinned R3 environment, with its manifest checks.
            science_receipt = ROOT / "local_runs/g7_s8_equal_episodes_evidence.json"
            venv = previous.R3_RUNTIME / "venv/bin/python"
            guard(venv.is_file(), "PINNED_SCIENCE_PYTHON_MISSING")
            result["training"] = "BOUNDED_EXPERIMENT_ONLY"
            result["native_forward"] = "ONE_BOUNDED_SCIENCE_PILOT_STARTED"
            child = execute(
                [str(venv), "-I", "-B",
                 str(ROOT / "scripts/g7_s8_equal_exposure_three_arms.py"),
                 "--expected-sha", args.expected_sha,
                 "--protocol", str(ROOT / "experiments/g7_s8_frozen_episode_matched.json"),
                 "--receipt", str(science_receipt)],
                env=scoped, timeout=1080,
            )
            guard(science_receipt.is_file(), "SCIENCE_RESULT_RECEIPT_MISSING")
            result_bytes = science_receipt.read_bytes()
            evidence = json.loads(result_bytes)
            result["science_receipt_sha256"] = hashlib.sha256(result_bytes).hexdigest()
            result["science_status"] = evidence.get("status")
            result["support_causal_status"] = (
                "ASSAY_VALID" if evidence.get("oracle_controls_pass") else
                "ASSAY_INVALID")
            result["support_oracle_controls_pass"] = evidence.get(
                "oracle_controls_pass")
            result["conventional_positive_control_pass"] = evidence.get(
                "conventional_positive_control_pass")
            result["matched_training_episodes_per_seed"] = evidence.get(
                "training_episodes_per_seed_all_arms")
            result["science_metric"] = evidence.get("online_vs_static_relative_mse_reduction")
            result["native_forward"] = "SCIENCE_PILOT_COMPLETE"
            guard(child.returncode == 0
                  and evidence.get("status") in (
                      "PRELIMINARY_ASSOCIATIVE_LEARNING_SIGNAL_REQUIRES_INDEPENDENT_REPLICATION",
                      "ONLINE_TTT_INCREMENTAL_ADVANTAGE_NOT_DEMONSTRATED")
                  and evidence.get("oracle_controls_pass") is True
                  and evidence.get("conventional_positive_control_pass") is True
                  and evidence.get("training_episodes_per_seed_all_arms") == 32768,
                  "S8_EQUAL_EPISODE_NATIVE_COMPARISON_UNQUALIFIED")
            result["status"] = "PASS_SIGNED_EPHEMERAL_HEADERS_AND_S8_THREE_ARM_MEASURED"
            exit_code = 0
    except previous.Blocked as exc:
        result["status"] = "BLOCKED_" + str(exc)
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["status"] = "BLOCKED_HOST_" + type(exc).__name__
    except Exception as exc:
        result["status"] = "BLOCKED_UNEXPECTED_" + type(exc).__name__
    finally:
        if result["dpkg_before_sha256"]:
            try:
                result["dpkg_after_sha256"] = dpkg_hash()
                result["installed_packages_changed"] = (
                    "NONE" if result["dpkg_before_sha256"] ==
                    result["dpkg_after_sha256"] else "DRIFT"
                )
                if result["installed_packages_changed"] != "NONE":
                    result["status"] = "BLOCKED_DPKG_INVENTORY_DRIFT"
                    exit_code = 2
            except Exception:
                result["status"] = "BLOCKED_DPKG_POSTRUN_UNAVAILABLE"
                exit_code = 2
        target = args.receipt.resolve()
        if (target.parent == (ROOT / "local_runs").resolve()
                and target.name == "g7_s8_ephemeral_runtime_admission.json"):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
        print("G7_S8_STATUS=" + result["status"], flush=True)
        print("G7_S8_HEADER=" + result["exact_signed_header_metadata"], flush=True)
        print("G7_S8_C_POSITIVE=" +
              str(result["positive_control"]["pass"] if result["positive_control"] else "NOT_RUN"),
              flush=True)
        print("G7_S8_FORWARD=" + result["native_forward"], flush=True)
        print("G7_S8_DPKG=" + result["installed_packages_changed"], flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
