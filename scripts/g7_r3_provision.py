"""One-time WSL-only isolated CUDA package staging with exact source custody.

Never imports torch or triton, starts GPU kernels, compiles native extensions,
trains models, installs system packages, or changes drivers/controllers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from urllib.request import urlopen

UV_VERSION = "0.9.9"
UV_ARCHIVE = "uv-x86_64-unknown-linux-gnu.tar.gz"
UV_SHA256 = "9ec303873e00deed44d1b2b52b85ab7aa55d849588d7242298748390eaea07ef"
UV_URL = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{UV_ARCHIVE}"
TORCH_VERSION = "2.9.0"
TRITON_VERSION = "3.5.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
PYPI_INDEX = "https://pypi.org/simple"
SUPPORT_PACKAGES = (
    "transformers==4.44.2", "numpy==1.26.4", "einops==0.8.1",
    "packaging==24.2", "ninja==1.13.0", "setuptools==80.9.0",
    "wheel==0.45.1",
)
MIN_FREE_GIB = 16
MAX_DOWNLOAD_BYTES = 40 * 1024 * 1024


class StagingBlocked(RuntimeError):
    pass


def fail(reason):
    raise StagingBlocked(reason)


def run(label, args, *, env=None, timeout=300):
    # Do not print command arguments, environment variables, or remote error text.
    try:
        result = subprocess.run(
            args, check=False, env=env, capture_output=True,
            text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        fail(label + "_TIMEOUT")
    except OSError:
        fail(label + "_COMMAND_UNAVAILABLE")
    if result.returncode != 0:
        fail(label + "_EXIT_" + str(result.returncode))
    print("G7_R3_STEP_PASS=" + label, flush=True)
    return result.stdout.strip()


def download_verified_uv(target):
    archive = target / "uv.tar.gz"
    h = hashlib.sha256()
    n = 0
    try:
        with urlopen(UV_URL, timeout=70) as response, archive.open("wb") as output:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                n += len(block)
                if n > MAX_DOWNLOAD_BYTES:
                    fail("UV_ARCHIVE_OVERSIZE")
                h.update(block)
                output.write(block)
    except (OSError, ValueError):
        fail("UV_DOWNLOAD_FAILED")
    if h.hexdigest() != UV_SHA256:
        fail("UV_ARCHIVE_SHA256_MISMATCH")
    with tarfile.open(archive, "r:gz") as tar:
        items = [m for m in tar.getmembers() if m.isfile() and m.name.endswith("/uv")]
        if len(items) != 1 or items[0].size > 90 * 1024 * 1024:
            fail("UV_ARCHIVE_MEMBER_INVALID")
        binary = target / "uv"
        src = tar.extractfile(items[0])
        if src is None:
            fail("UV_ARCHIVE_MEMBER_UNREADABLE")
        with binary.open("wb") as dest:
            shutil.copyfileobj(src, dest)
        binary.chmod(0o700)
    archive.unlink()
    version = run("UV_VERSION", [str(binary), "--version"], timeout=15)
    if version != "uv " + UV_VERSION:
        fail("UV_VERSION_MISMATCH")
    return binary


def clone_source(root, name, spec):
    url, sha = spec["url"], spec["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        fail("SOURCE_PIN_FORMAT_" + name)
    dest = root / "sources" / name
    dest.mkdir(parents=True, exist_ok=False)
    run("GIT_INIT_" + name, ["git", "init", "-q", str(dest)])
    run("GIT_FETCH_" + name, ["git", "-C", str(dest), "fetch", "--no-tags", "--depth=1", url, sha], timeout=300)
    run("GIT_CHECKOUT_" + name, ["git", "-C", str(dest), "checkout", "--detach", "-q", "FETCH_HEAD"])
    actual = run("GIT_SHA_" + name, ["git", "-C", str(dest), "rev-parse", "HEAD"])
    if actual != sha:
        fail("SOURCE_COMMIT_MISMATCH_" + name)
    result = {"sha": sha, "required_blobs": {}}
    for rel in spec["required_files"]:
        if not re.fullmatch(r"[A-Za-z0-9_./-]{1,180}", rel) or ".." in Path(rel).parts:
            fail("UNSAFE_SOURCE_PATH_" + name)
        expected = spec["observed_blob_sha"][rel]
        actual_blob = run("GIT_BLOB_" + name, ["git", "-C", str(dest), "hash-object", "--", rel])
        if actual_blob != expected:
            fail("SOURCE_BLOB_MISMATCH_" + name)
        result["required_blobs"][rel] = actual_blob
    return result


def packages_metadata(python):
    # Metadata introspection only: no torch/triton or CUDA module imports.
    code = (
        "import importlib.metadata as m,json;"
        "a=sorted((str(x.metadata.get('Name','')).lower().replace('_','-'),"
        "str(x.version)) for x in m.distributions());"
        "print(json.dumps(a,separators=(',',':')))"
    )
    out = run("PACKAGE_METADATA", [str(python), "-I", "-c", code], timeout=35)
    return json.loads(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expected-sha", required=True)
    p.add_argument("--source-pins", required=True, type=Path)
    p.add_argument("--receipt", required=True, type=Path)
    args = p.parse_args()
    receipt = args.receipt.resolve()
    report = {
        "schema": "ttt-hybrid.g7-r3-isolated-provision.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_sha": args.expected_sha,
        "status": "BLOCKED_UNVERIFIED",
        "scope": "ISOLATED_ENVIRONMENT_AND_EXACT_SOURCE_PROVENANCE_ONLY",
        "cuda_kernels": "NOT_RUN", "cuda_runtime": "NOT_PROBED",
        "native_extensions": "NOT_BUILT", "training": "NOT_RUN",
        "system_packages": "NOT_INSTALLED", "scientific": "UNVERIFIED",
        "uv_version": UV_VERSION, "uv_archive_sha256": UV_SHA256,
        "torch_target": TORCH_VERSION + "+cu128", "triton_target": TRITON_VERSION,
        "runtime_imports": "NOT_RUN", "source_imports": "NOT_RUN",
    }
    exit_code = 2
    try:
        if sys.platform != "linux" or platform.machine() != "x86_64":
            fail("UNSUPPORTED_RUNTIME_PLATFORM")
        if sys.version_info[:2] != (3, 12):
            fail("UNSUPPORTED_PYTHON_VERSION")
        if not re.fullmatch(r"[0-9a-f]{40}", args.expected_sha):
            fail("INVALID_CANDIDATE_SHA")
        if os.geteuid() == 0:
            fail("REFUSE_ROOT_INSTALL")
        info = Path("/etc/os-release").read_text(encoding="utf-8")
        if 'ID=ubuntu' not in info and 'ID="ubuntu"' not in info:
            fail("NOT_UBUNTU")
        if "24.04" not in info:
            fail("WRONG_UBUNTU_VERSION")
        if not Path("/proc/sys/kernel/osrelease").is_file():
            fail("NO_WSL_KERNEL_METADATA")
        if "microsoft" not in Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8").lower():
            fail("NOT_WSL_ENVIRONMENT")
        pins_data = args.source_pins.read_bytes()
        pins = json.loads(pins_data)
        report["source_pins_sha256"] = hashlib.sha256(pins_data).hexdigest()
        if pins.get("schema") != "ttt-hybrid.official-source-pins.v1":
            fail("SOURCE_PINS_SCHEMA_DRIFT")
        home = Path.home().resolve()
        if not home.is_dir():
            fail("HOME_UNAVAILABLE")
        free_bytes = shutil.disk_usage(home).free
        report["initial_free_gib"] = round(free_bytes / 1024**3, 2)
        if free_bytes < MIN_FREE_GIB * 1024**3:
            fail("INSUFFICIENT_DISK_FOR_CUDA_WHEELS")
        root = home / ".local" / "share" / "ttt-hybrid" / ("g7-r3-" + args.expected_sha[:12])
        if root.exists():
            fail("EXISTING_G7_R3_ENV_REQUIRES_RECONCILIATION")
        root.mkdir(parents=True, exist_ok=False)
        uv = download_verified_uv(root)
        venv = root / "venv"
        env = os.environ.copy()
        env.update({"UV_CACHE_DIR": str(root / "uv-cache"),
                    "UV_NO_PROGRESS": "1", "UV_CONCURRENT_DOWNLOADS": "2",
                    "UV_HTTP_TIMEOUT": "90", "UV_LINK_MODE": "copy"})
        run("CREATE_ISOLATED_VENV", [str(uv), "venv", "--no-project", "--python", "/usr/bin/python3", str(venv)], env=env)
        python = venv / "bin" / "python"
        run("INSTALL_TORCH_CU128", [str(uv), "pip", "install", "--python", str(python), "--index-url", TORCH_INDEX,
                                   "torch==" + TORCH_VERSION], env=env, timeout=1850)
        run("INSTALL_SUPPORT_PACKAGES", [str(uv), "pip", "install", "--python", str(python), "--index-url", PYPI_INDEX,
                                         *SUPPORT_PACKAGES], env=env, timeout=950)
        run("VALIDATE_PACKAGE_GRAPH", [str(uv), "pip", "check", "--python", str(python)], env=env, timeout=120)
        pairs = packages_metadata(python)
        observed = dict(pairs)
        torch = observed.get("torch", "")
        triton = observed.get("triton", "")
        if not torch.startswith(TORCH_VERSION + "+cu128") or triton != TRITON_VERSION:
            fail("TORCH_CUDA_OR_TRITON_MISMATCH")
        for req in SUPPORT_PACKAGES:
            key, value = req.split("==")
            if observed.get(key) != value:
                fail("SUPPORT_DEPENDENCY_MISMATCH")
        report["installed_package_versions"] = {k: observed.get(k) for k in
            ("torch", "triton", "numpy", "transformers", "einops", "packaging", "ninja", "setuptools", "wheel")}
        report["package_manifest_sha256"] = hashlib.sha256(json.dumps(pairs, sort_keys=True).encode()).hexdigest()
        report["package_count"] = len(pairs)
        report["sources"] = {
            "mamba2": clone_source(root, "mamba2", pins["sources"]["mamba2"]),
            "ttt_linear": clone_source(root, "ttt_linear", pins["sources"]["ttt_linear"]),
        }
        report["final_free_gib"] = round(shutil.disk_usage(home).free / 1024**3, 2)
        report["status"] = "PASS_ISOLATED_PACKAGE_AND_SOURCE_PROVENANCE_ONLY"
        exit_code = 0
    except StagingBlocked as exc:
        report["status"] = "BLOCKED_" + str(exc)
    except Exception as exc:
        report["status"] = "FAIL_UNEXPECTED_" + type(exc).__name__
    finally:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print("G7_R3_STATUS=" + report["status"], flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
