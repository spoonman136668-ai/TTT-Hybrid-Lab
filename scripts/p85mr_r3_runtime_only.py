"""P85MR-R3: bounded WSL Python runtime repair; never executes scientific source."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys

PINS = {
    "numpy": "1.26.4", "gymnasium": "0.29.1", "networkx": "3.4.2",
    "pydash": "8.0.5", "toposort": "1.10", "cloudpickle": "3.0.0",
    "farama-notifications": "0.0.4", "typing-extensions": "4.12.2",
}
REPO = "spoonman136668-ai/TTT-Hybrid-Lab"
BRANCH = "research/p85mr-r3-wsl-runtime-only-20261111"
ROOT_NAME = "ttt-hybrid"
PUBLIC_INDEX = "https://pypi.org/simple"
BASE = "g7-s25-cpu311-r1/venv/bin/python"
PREEXISTING = (
    BASE, "p85mr-causalgym-cpu311/venv/bin/python",
    "causalgym-cpu311/venv/bin/python",
)
NEW = "p85mr-causalgym-cpu311-r3/venv"


def run(argv: list[str], seconds: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=seconds,
                          check=False, env={"PATH": "/usr/bin:/bin",
                                            "HOME": os.environ["HOME"],
                                            "LANG": "C.UTF-8",
                                            "PIP_CONFIG_FILE": "/dev/null"})


def qualified(python: Path) -> bool:
    if not python.is_file():
        return False
    code = (
        "import importlib,importlib.metadata as m,json,sys;"
        "p=" + repr(PINS) + ";"
        "assert sys.version_info[:2]==(3,11);"
        "assert all(m.version(k)==v for k,v in p.items());"
        "[importlib.import_module(k) for k in"
        " ('numpy','gymnasium','networkx','pydash','toposort')];"
        "print('P85MR_R3_PINNED_IMPORT_PASS')"
    )
    result = run([str(python), "-I", "-B", "-c", code], 35)
    return result.returncode == 0 and "P85MR_R3_PINNED_IMPORT_PASS" in result.stdout


def admission(project: Path) -> None:
    required = {
        "GITHUB_REPOSITORY": REPO,
        "GITHUB_ACTOR": "spoonman136668-ai",
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/" + BRANCH,
        "GITHUB_RUN_NUMBER": "1",
        "GITHUB_RUN_ATTEMPT": "1",
        "RUNNER_NAME": "TTT-HYBRID-GPU-WIN",
        "P85MR_R3_RUNTIME_ONLY": "1",
    }
    if any(os.getenv(k) != v for k, v in required.items()):
        raise RuntimeError("R3_OWNER_ROUTE_DENIED")
    sha = os.getenv("GITHUB_SHA", "")
    if not re.fullmatch("[0-9a-f]{40}", sha):
        raise RuntimeError("R3_SOURCE_SHA_MISSING")
    if not (project / "experiments/p85mr_r3_runtime_only_frozen.json").is_file():
        raise RuntimeError("R3_FREEZE_ABSENT")
    protocol = json.loads((project / "experiments/p85mr_r3_runtime_only_frozen.json").read_text())
    if (protocol.get("schema") != "ttt-hybrid.p85mr-r3-wsl-runtime-only.v1"
            or protocol.get("parent_sha") != "55faf21d05de4a6b62af52666130517e3aeb260e"
            or protocol.get("pinned_distributions") != PINS
            or protocol.get("new_runtime_relative") != NEW):
        raise RuntimeError("R3_FROZEN_PROTOCOL_DRIFT")
    check = run(["/usr/bin/git", "-C", str(project), "rev-parse", "HEAD"], 15)
    if check.returncode != 0 or check.stdout.strip() != sha:
        raise RuntimeError("R3_PROJECT_SHA_DRIFT")
    origin = run(["/usr/bin/git", "-C", str(project), "remote", "get-url", "origin"], 15)
    if origin.returncode or origin.stdout.strip() not in (
        "https://github.com/" + REPO, "https://github.com/" + REPO + ".git"
    ):
        raise RuntimeError("R3_GIT_ORIGIN_DRIFT")
    if run(["/usr/bin/git", "-C", str(project), "status", "--porcelain"], 15).stdout.strip():
        raise RuntimeError("R3_DIRTY_PROJECT")
    if sys.version_info[:2] != (3, 11) or os.getuid() != 1000:
        raise RuntimeError("R3_WSL_UNPRIVILEGED_PY311_REQUIRED")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    report: dict[str, object] = {
        "schema": "ttt-hybrid.p85mr-r3-runtime-only-result.v1",
        "status": "PRE_EFFECT_BLOCKED", "science_started": False,
        "simulator_reset": False, "source_first_use": False,
        "reserved_data_access": False, "training_started": False,
        "installed_new_runtime": False,
    }
    try:
        admission(args.project.resolve())
        home = Path.home().resolve()
        if not re.fullmatch(r"/home/[A-Za-z_][A-Za-z0-9_-]{0,32}", str(home)):
            raise RuntimeError("R3_UNEXPECTED_HOME")
        shared = home / ".local/share" / ROOT_NAME
        for rel in PREEXISTING:
            if qualified(shared / rel):
                report["status"] = "PASS_EXISTING_PINNED_WSL_RUNTIME"
                report["runtime_alias"] = rel.split("/")[0]
                break
        else:
            existing = shared / BASE
            if not existing.is_file():
                raise RuntimeError("R3_EXISTING_PROVEN_BASE_PYTHON_MISSING")
            target = shared / NEW
            if target.parent.exists():
                raise RuntimeError("R3_TARGET_ALREADY_EXISTS_NO_OVERWRITE")
            target.parent.mkdir(parents=True, exist_ok=False)
            make = run([str(existing), "-I", "-B", "-m", "venv", str(target)], 90)
            if make.returncode:
                raise RuntimeError("R3_ISOLATED_VENV_CREATION_FAILED")
            py = target / "bin/python"
            args_pip = [
                str(py), "-I", "-B", "-m", "pip", "--isolated",
                "install", "--disable-pip-version-check", "--no-input",
                "--no-cache-dir", "--only-binary=:all:", "--index-url", PUBLIC_INDEX,
                "--report", str(target.parent / "pip-install-report.json"),
            ] + [f"{name}=={version}" for name, version in sorted(PINS.items())]
            install = run(args_pip, 480)
            if install.returncode:
                raise RuntimeError("R3_PINNED_BINARY_WHEEL_INSTALL_FAILED")
            if not qualified(py):
                raise RuntimeError("R3_POSTINSTALL_PACKAGE_IMPORT_OR_VERSION_FAIL")
            install_report = json.loads((target.parent / "pip-install-report.json").read_text())
            wheels = []
            for item in install_report.get("install", []):
                md = item.get("metadata", {})
                archive = item.get("download_info", {}).get("archive_info", {})
                hashes = archive.get("hashes", {})
                digest = hashes.get("sha256", "")
                if not re.fullmatch("[0-9a-f]{64}", digest):
                    raise RuntimeError("R3_WHEEL_SHA256_MISSING")
                wheels.append({"name": md.get("name"), "version": md.get("version"),
                               "wheel_sha256": digest})
            if not wheels:
                raise RuntimeError("R3_WHEEL_RECEIPT_EMPTY")
            (target.parent / "pip-install-report.json").unlink()
            report["wheel_records"] = sorted(wheels, key=lambda z: str(z["name"]).lower())
            report["runtime_alias"] = NEW.split("/")[0]
            report["installed_new_runtime"] = True
            report["status"] = "PASS_NEW_ISOLATED_PINNED_WSL_RUNTIME"
        report["pinned_packages"] = PINS
        report["python_minor"] = "3.11"
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired,
            json.JSONDecodeError) as exc:
        # Never print user paths, pip error text, environment variables or credentials.
        report["failure_class"] = (
            str(exc) if isinstance(exc, RuntimeError) and str(exc).startswith("R3_")
            else "R3_UNEXPECTED_RUNTIME_FAILURE"
        )
    finally:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("P85MR_R3_STATUS=" + str(report["status"]))
    return 0 if str(report["status"]).startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
