"""G7-R4: exact-custody PyTorch CUDA metadata and official module import gate.

Read-only GPU runtime admission: imports and CUDA device queries only.
No model constructors, forward passes, CUDA kernels, compilation, installation,
training, controllers, or activation of execution workflows.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

R3_QUALIFIED_SHA = "06a01c2c3a980eeb806a7ecb7cca1979a431f80c"
R3_SOURCE_PINS_SHA256 = "44f3e76d2d06ee879fa3222dd3921c7a1f8e08be579b96e03b7545e388a7b768"
R3_PACKAGE_MANIFEST_SHA256 = "b0f674fd52f9f514d408840d27b969ce2da19ea119d90a6ce8f59fde05c073df"
R3_TORCH = "2.9.0+cu128"
R3_TRITON = "3.5.0"
RUNTIME_TIMEOUT = 130


class AdmissionBlocked(Exception):
    pass


def blocked(reason: str) -> None:
    raise AdmissionBlocked(reason)


def command(*argv: str, timeout: int = 30) -> str:
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        blocked("SOURCE_COMMAND_UNAVAILABLE_OR_TIMEOUT")
    if p.returncode:
        blocked("SOURCE_COMMAND_FAILED")
    return p.stdout.strip()


def verify_source(root: Path, spec: dict) -> dict:
    expected_sha = spec["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", expected_sha):
        blocked("INVALID_SOURCE_SHA")
    observed = command("git", "-C", str(root), "rev-parse", "HEAD")
    if observed != expected_sha:
        blocked("SOURCE_REVISION_DRIFT")
    if command("git", "-C", str(root), "status", "--porcelain"):
        blocked("SOURCE_TREE_DIRTY")
    checked = {}
    for rel in spec["required_files"]:
        if ".." in Path(rel).parts or not re.fullmatch(r"[A-Za-z0-9_./-]{1,180}", rel):
            blocked("INVALID_REQUIRED_SOURCE_PATH")
        expected_blob = spec["observed_blob_sha"][rel]
        observed_blob = command("git", "-C", str(root), "hash-object", "--", rel)
        if observed_blob != expected_blob:
            blocked("SOURCE_BLOB_DRIFT")
        checked[rel] = observed_blob
    return {"sha": expected_sha, "blobs": checked}


def child(python: Path, source: str, arguments: list[str], label: str, *, timeout=RUNTIME_TIMEOUT) -> dict:
    """Isolated child with bounded output and no user path disclosures."""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUDA_MODULE_LOADING"] = "LAZY"
    try:
        proc = subprocess.run([str(python), "-I", "-B", "-c", source, *arguments],
                              capture_output=True, text=True, timeout=timeout, env=env,
                              check=False)
    except subprocess.TimeoutExpired:
        return {"status": "BLOCKED_TIMEOUT", "step": label}
    except OSError:
        return {"status": "BLOCKED_PYTHON_NOT_FOUND", "step": label}
    # Third-party output/warnings are not persisted. Only our strict sentinel is parsed.
    matches = [line[13:] for line in proc.stdout.splitlines() if line.startswith("G7R4_RESULT::")]
    if not matches or len(matches) != 1 or len(matches[0]) > 4500:
        return {"status": "FAIL_MISSING_BOUNDED_RESULT", "step": label,
                "child_exit_ok": proc.returncode == 0}
    try:
        report = json.loads(matches[0])
    except json.JSONDecodeError:
        return {"status": "FAIL_INVALID_JSON_RESULT", "step": label}
    if not isinstance(report, dict) or not isinstance(report.get("status"), str):
        return {"status": "FAIL_INVALID_RESULT_SCHEMA", "step": label}
    if proc.returncode != 0 and report["status"] == "PASS":
        return {"status": "FAIL_EXIT_CONFLICT", "step": label}
    return report


CHECK_PACKAGES_AND_CUDA = r'''
import importlib.metadata as meta
import hashlib
import json
try:
    packages = sorted((str(d.metadata.get('Name', '')).lower().replace('_','-'),
                       str(d.version)) for d in meta.distributions())
    manifest_hash = hashlib.sha256(json.dumps(packages, sort_keys=True).encode()).hexdigest()
    observed = dict(packages)
    result = {'manifest_sha256': manifest_hash,
              'torch_metadata': observed.get('torch'),
              'triton_metadata': observed.get('triton')}
    if manifest_hash != __import__('sys').argv[1]:
        result['status'] = 'BLOCKED_PACKAGE_MANIFEST_DRIFT'
    elif observed.get('torch') != '2.9.0+cu128' or observed.get('triton') != '3.5.0':
        result['status'] = 'BLOCKED_PACKAGE_VERSION_DRIFT'
    else:
        import torch
        import triton
        result['torch_runtime'] = str(torch.__version__)
        result['triton_runtime'] = str(triton.__version__)
        result['torch_cuda_build'] = str(torch.version.cuda)
        if torch.__version__ != '2.9.0+cu128' or triton.__version__ != '3.5.0':
            result['status'] = 'BLOCKED_IMPORT_VERSION_DRIFT'
        elif not torch.cuda.is_available():
            result['status'] = 'BLOCKED_CUDA_NOT_AVAILABLE'
        elif torch.cuda.device_count() != 1:
            result['status'] = 'BLOCKED_CUDA_DEVICE_COUNT_UNEXPECTED'
        else:
            props = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info(0)
            archs = list(torch.cuda.get_arch_list())
            cc = [int(props.major), int(props.minor)]
            arch = 'sm_' + str(cc[0]) + str(cc[1])
            result['gpu'] = {'name': str(props.name)[:90],
                             'compute_capability': cc,
                             'total_vram_mib': int(total // (1024**2)),
                             'free_vram_mib': int(free // (1024**2)),
                             'torch_arch_list': archs[:40],
                             'computed_arch': arch}
            if free < 4 * 1024**3 or total <= 0 or free > total:
                result['status'] = 'BLOCKED_GPU_VRAM_ADMISSION'
            elif arch not in archs:
                result['status'] = 'BLOCKED_CUDA_ARCH_UNSUPPORTED'
            else:
                result['status'] = 'PASS'
except Exception as exc:
    result = {'status': 'FAIL_RUNTIME_EXCEPTION', 'error_type': type(exc).__name__}
print('G7R4_RESULT::' + json.dumps(result, separators=(',', ':'), sort_keys=True))
'''

CHECK_SOURCE_IMPORT = r'''
import importlib
import inspect
import json
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
module = sys.argv[2]
symbol = sys.argv[3]
rel = sys.argv[4]
result = {}
try:
    sys.path.insert(0, str(root))
    imported = importlib.import_module(module)
    obj = getattr(imported, symbol)
    observed = Path(inspect.getfile(obj)).resolve()
    expected = (root / rel).resolve()
    result['status'] = 'PASS' if observed == expected else 'BLOCKED_IMPORT_ORIGIN_DRIFT'
    result['symbol'] = symbol
    result['origin_matches_pin'] = (observed == expected)
except ModuleNotFoundError as exc:
    result['status'] = 'BLOCKED_MISSING_DEPENDENCY'
    name = exc.name or ''
    result['missing_module'] = name[:90] if name.replace('_','').replace('.','').isalnum() else 'REDACTED'
except (ImportError, RuntimeError, OSError, AttributeError) as exc:
    result['status'] = 'BLOCKED_IMPORT_COMPATIBILITY'
    result['error_type'] = type(exc).__name__
except Exception as exc:
    result['status'] = 'FAIL_UNEXPECTED_IMPORT_ERROR'
    result['error_type'] = type(exc).__name__
print('G7R4_RESULT::' + json.dumps(result, separators=(',', ':'), sort_keys=True))
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected-sha", required=True)
    ap.add_argument("--source-pins", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    args = ap.parse_args()
    receipt = args.receipt.resolve()
    record = {
        "schema": "ttt-hybrid.g7-r4-cuda-runtime-admission.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_sha": args.expected_sha,
        "r3_source_sha": R3_QUALIFIED_SHA,
        "status": "BLOCKED_UNVERIFIED",
        "scope": "CUDA_DEVICE_METADATA_AND_PINNED_MODULE_IMPORTS_ONLY",
        "kernel_execution": "NOT_RUN", "model_construction": "NOT_RUN",
        "gpu_tensor_allocation": "NOT_REQUESTED", "training": "NOT_RUN",
        "native_extension_build": "NOT_RUN", "system_installation": "NOT_RUN",
        "scientific": "UNVERIFIED",
    }
    code = 2
    try:
        if sys.platform != "linux" or platform.machine() != "x86_64":
            blocked("WRONG_PLATFORM")
        if sys.version_info[:2] != (3, 12):
            blocked("WRONG_PYTHON_VERSION")
        kernel = Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8").lower()
        if "microsoft" not in kernel:
            blocked("NOT_WSL")
        if not re.fullmatch(r"[0-9a-f]{40}", args.expected_sha):
            blocked("INVALID_CANDIDATE_SHA")
        if os.geteuid() == 0:
            blocked("REFUSE_ROOT")
        pins_raw = args.source_pins.read_bytes()
        sha256 = hashlib.sha256(pins_raw).hexdigest()
        if sha256 != R3_SOURCE_PINS_SHA256:
            blocked("R3_SOURCE_PINS_DRIFT")
        pins = json.loads(pins_raw)
        if pins.get("schema") != "ttt-hybrid.official-source-pins.v1":
            blocked("SOURCE_PINS_SCHEMA_MISMATCH")
        root = Path.home().resolve() / ".local" / "share" / "ttt-hybrid" / ("g7-r3-" + R3_QUALIFIED_SHA[:12])
        python = root / "venv" / "bin" / "python"
        if not python.is_file():
            blocked("R3_ENVIRONMENT_ABSENT")
        record["sources"] = {
            "mamba2": verify_source(root / "sources" / "mamba2", pins["sources"]["mamba2"]),
            "ttt_linear": verify_source(root / "sources" / "ttt_linear", pins["sources"]["ttt_linear"]),
        }
        record["source_pins_sha256"] = sha256
        record["runtime"] = child(python, CHECK_PACKAGES_AND_CUDA,
                                  [R3_PACKAGE_MANIFEST_SHA256], "CUDA_DEVICE")
        runtime_state = record["runtime"].get("status")
        # Never import third-party source when the immutable package baseline
        # has drifted or the base imports themselves failed.
        if runtime_state not in (
            "PASS", "BLOCKED_CUDA_NOT_AVAILABLE",
            "BLOCKED_CUDA_DEVICE_COUNT_UNEXPECTED", "BLOCKED_GPU_VRAM_ADMISSION",
            "BLOCKED_CUDA_ARCH_UNSUPPORTED",
        ):
            blocked("CUDA_RUNTIME_GATE_FAILED")
        # Source imports are useful independent negative controls even when
        # CUDA device admission blocks. Neither constructs a model or runs it.
        for label, module, symbol, rel in (
            ("mamba2", "mamba_ssm.modules.mamba2", "Mamba2", "mamba_ssm/modules/mamba2.py"),
            ("ttt_linear", "ttt", "TTTForCausalLM", "ttt.py"),
        ):
            source_root = root / "sources" / label
            report = child(python, CHECK_SOURCE_IMPORT,
                           [str(source_root), module, symbol, rel], "IMPORT_" + label)
            record.setdefault("source_imports", {})[label] = report
        if runtime_state != "PASS":
            blocked("CUDA_RUNTIME_GATE_FAILED")
        if any(v.get("status") != "PASS" for v in record["source_imports"].values()):
            blocked("PINNED_SOURCE_IMPORT_GATE_FAILED")
        record["status"] = "PASS_CUDA_DEVICE_AND_PINNED_IMPORTS_NO_KERNELS"
        code = 0
    except AdmissionBlocked as exc:
        record["status"] = "BLOCKED_" + str(exc)
    except Exception as exc:
        record["status"] = "FAIL_UNEXPECTED_" + type(exc).__name__
    finally:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("G7_R4_STATUS=" + record["status"], flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
