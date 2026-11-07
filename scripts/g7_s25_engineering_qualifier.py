"""G7-S25 exact-source engineering-only qualification via established R3 WSL venv.

No model fitting, frozen data, scientific first-use, package installation, GPU,
runner provisioning or deployment. The official TTT-LM/G5 code remains separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import g7_r5_oneshot as custody

SOURCE_BASE = "7d1f8bb1547d4d39df7412091a68dabb03f0ac3f"
S20_BLOB = "ac3f9ddd5b879e53e62030707fb6419adaf732ca"
REQUIRED = (
    "experiments/g7_s25_frozen_contradiction_ttt.json",
    "scripts/g7_s25_contradiction_gate.py",
    "tests/test_g7_s25_gate_contract.py",
    ".github/workflows/g7-s25-gate-design-qualification.yml",
    "scripts/g7_s25_numeric_ttt.py",
    "tests/test_g7_s25_numeric_ttt.py",
    "scripts/g7_s25_engineering_qualifier.py",
    ".github/workflows/g7-s25-numeric-engineering.yml",
    "scripts/g7_s25_paired_interventions.py",
    "tests/test_g7_s25_paired_interventions.py",
    ".github/workflows/g7-s25-paired-physics-engineering.yml",
    "scripts/g7_s25_cost_ledger.py",
    "tests/test_g7_s25_cost_ledger.py",
    ".github/workflows/g7-s25-cost-ledger-qualification.yml",
    ".github/workflows/g7-s25-cost-ledger-qualification-r2.yml",
    "scripts/g7_s25_baseline_admission.py",
    "tests/test_g7_s25_baseline_admission.py",
    ".github/workflows/g7-s25-baseline-contract-once.yml",
    "scripts/g7_s25_causal_baseline_stream.py",
    "tests/test_g7_s25_causal_baseline_stream.py",
    ".github/workflows/g7-s25-causal-stream-once.yml",
    ".github/workflows/g7-s25-full-regression-existing-runtime.yml",
    "tests/test_g7_s25_realized_primary.py",
    ".github/workflows/g7-s25-primary-residual-once.yml",
    "scripts/g7_s25_gate_diagnostic.py",
    "tests/test_g7_s25_gate_diagnostic.py",
    ".github/workflows/g7-s25-mechanism-diagnostic-once.yml",
)

def ensure(ok: bool, cause: str) -> None:
    if not ok:
        raise RuntimeError(cause)

def command(argv: list[str], timeout: int, env: dict[str, str]) -> dict:
    start = time.perf_counter()
    result = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True,
                            timeout=timeout, check=False, env=env)
    return {"exit_code": result.returncode,
            "wall_seconds": round(time.perf_counter() - start, 3),
            "stdout_tail": result.stdout[-300:]}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sha", required=True)
    args = p.parse_args()
    outfile = ROOT / "local_runs/g7_s25_numeric_engineering.json"
    status = {
        "schema": "ttt-hybrid.g7-s25.numeric-engineering.v1",
        "sha": args.sha,
        "status": "BLOCKED_NOT_STARTED",
        "source": "UNVERIFIED",
        "pinned_environment": "UNVERIFIED",
        "scientific_first_use": "NOT_RUN",
        "training": "NOT_RUN",
        "calibration": "NOT_RUN",
        "heldout": "NOT_RUN",
        "gpu": "NOT_USED",
        "packages_changed": False,
        "full_required_platform_regression": "UNVERIFIED",
        "deployment": "NOT_AUTHORIZED",
    }
    rc = 2
    try:
        ensure(os.geteuid() == 1000 and
               "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower(),
               "NORMAL_USER_WSL_REQUIRED")
        ensure(custody.git(ROOT, "rev-parse", "HEAD") == args.sha,
               "EXACT_SHA_DRIFT")
        ensure(not custody.git(ROOT, "status", "--porcelain", "--untracked-files=all"),
               "DIRTY_TREE")
        files = custody.git(ROOT, "diff", "--name-only", SOURCE_BASE, "HEAD").splitlines()
        ensure(len(files) == len(REQUIRED) and set(files) == set(REQUIRED),
               "S25_FILE_SCOPE_DRIFT")
        ensure(custody.git(ROOT, "hash-object", "--", "scripts/g7_s20_hidden_multicycle.py")
               == S20_BLOB, "S20_PHYSICS_DRIFT")
        ensure(custody.sha256(ROOT / "source_pins.json") == custody.R3_PINS_SHA256,
               "SOURCE_PINS_DRIFT")
        protocol = json.loads((ROOT / REQUIRED[0]).read_text())
        ensure(protocol["schema"] ==
               "ttt-hybrid.g7-s25.contradiction-gated-continuous-ttt.v1"
               and protocol["authority"]["scientific_first_use_authorized"] is False
               and protocol["authority"]["no_training_execution"] is True
               and protocol["seeds"]["engineering_only"] == [3101, 3109, 3119],
               "S25_FROZEN_PROTOCOL_DRIFT")
        ensure(custody.git(ROOT, "hash-object", "--", REQUIRED[0]) ==
               "31633b341403729b7ffbb7d73f3a04098ad44380",
               "S25_FROZEN_PROTOCOL_BLOB_DRIFT")
        status["source"] = "PASS_EXACT_TWENTYSEVEN_FILES_FROZEN_S25_AND_S20_BLOBS"
        venv = (Path.home().resolve() / ".local/share/ttt-hybrid"
                / ("g7-r3-" + custody.R3_SOURCE_SHA[:12])
                / "venv/bin/python")
        ensure(venv.is_file(), "PINNED_R3_PYTHON_ABSENT")
        env = dict(os.environ)
        env.update({"CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1",
                    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        fingerprint = command(
            [str(venv), "-I", "-B", "-c",
             "import sys;sys.path.insert(0,sys.argv[1]);import g7_r5_oneshot as c;print(c.manifest()[0])",
             str(ROOT / "scripts")], 30, env)
        ensure(fingerprint["exit_code"] == 0 and
               fingerprint["stdout_tail"].strip() == custody.R3_MANIFEST_SHA256,
               "PINNED_R3_PACKAGE_MANIFEST_DRIFT")
        status["pinned_environment"] = "PASS_EXISTING_EXACT_R3_MANIFEST"
        gate = command([sys.executable, "-I", "-B",
                        str(ROOT / "tests/test_g7_s25_gate_contract.py")], 45, env)
        status["gate_contract"] = gate
        ensure(gate["exit_code"] == 0 and "Ran 9 tests" in gate["stdout_tail"]
               and "OK" in gate["stdout_tail"], "PREREGISTERED_GATE_TEST_FAIL")
        model = command([str(venv), "-I", "-B",
                         str(ROOT / "tests/test_g7_s25_numeric_ttt.py")], 180, env)
        status["numeric_contract"] = model
        ensure(model["exit_code"] == 0 and "Ran 10 tests" in model["stdout_tail"]
               and "OK" in model["stdout_tail"], "NUMERIC_TTT_CONTRACT_FAIL")
        physics = command([str(venv), "-I", "-B",
                           str(ROOT / "tests/test_g7_s25_paired_interventions.py")],
                          180, env)
        status["paired_physics_contract"] = physics
        ensure(physics["exit_code"] == 0 and "Ran 9 tests" in physics["stdout_tail"]
               and "OK" in physics["stdout_tail"], "PAIRED_PHYSICS_CONTRACT_FAIL")
        ledger = command([str(venv), "-I", "-B",
                          str(ROOT / "tests/test_g7_s25_cost_ledger.py")],
                         220, env)
        status["cost_ledger_contract"] = ledger
        ensure(ledger["exit_code"] == 0
               and "Ran 10 tests" in ledger["stdout_tail"]
               and "OK" in ledger["stdout_tail"],
               "G7_S25_COST_LEDGER_TEST_FAIL")
        # This stdlib test fabricates synthetic *receipts only*, not worlds.
        # It exercises strict data splits, plateau logic and negative controls.
        baseline = command([sys.executable, "-I", "-B",
                            str(ROOT / "tests/test_g7_s25_baseline_admission.py")],
                           60, env)
        status["baseline_receipt_admission_contract"] = baseline
        ensure(baseline["exit_code"] == 0
               and "Ran 16 tests" in baseline["stdout_tail"]
               and "OK" in baseline["stdout_tail"],
               "BASELINE_RECEIPT_CONTRACT_FAIL")
        # Same numeric 32-frame observed prefix must reach both trained
        # controls and online TTT without current target leakage.
        stream = command([str(venv), "-I", "-B",
                          str(ROOT / "tests/test_g7_s25_causal_baseline_stream.py")],
                         150, env)
        status["causal_baseline_stream_contract"] = stream
        ensure(stream["exit_code"] == 0 and
               "Ran 10 tests" in stream["stdout_tail"] and
               "OK" in stream["stdout_tail"],
               "CAUSAL_BASELINE_STREAM_TEST_FAIL")
        # Critically, surprise gating must score residual of the action the
        # simulator actually executed, never the arbitrary plus-action probe.
        primary = command([str(venv), "-I", "-B",
                           str(ROOT / "tests/test_g7_s25_realized_primary.py")],
                          150, env)
        status["realized_primary_contract"] = primary
        ensure(primary["exit_code"] == 0
               and "Ran 6 tests" in primary["stdout_tail"]
               and "OK" in primary["stdout_tail"],
               "S25_REALIZED_PRIMARY_RESIDUAL_TEST_FAIL")
        # The untrained mechanism diagnostic must account for true selected
        # contradiction triggers separately from fallback budget-forcing.
        mechanism = command([str(venv), "-I", "-B",
                             str(ROOT / "tests/test_g7_s25_gate_diagnostic.py")],
                            150, env)
        status["gate_mechanism_contract"] = mechanism
        ensure(mechanism["exit_code"] == 0
               and "Ran 8 tests" in mechanism["stdout_tail"]
               and "OK" in mechanism["stdout_tail"],
               "S25_MECHANISM_DIAGNOSTIC_CONTRACT_FAIL")
        # Full pytest regression on pinned R3 cannot be inferred from focused tests.
        check = command([str(venv), "-I", "-B", "-c",
                         "import importlib.util;print('PYTEST_PRESENT' if importlib.util.find_spec('pytest') else 'PYTEST_ABSENT')"],
                        20, env)
        status["full_regression_dependency"] = check["stdout_tail"].strip()
        status["status"] = "PASS_78_FOCUSED_ENGINEERING_ONLY"
        rc = 0
    except Exception as exc:
        status["status"] = "BLOCKED_" + type(exc).__name__
        status["blocked_reason"] = str(exc)[:128]
    finally:
        outfile.parent.mkdir(parents=True, exist_ok=True)
        outfile.write_text(json.dumps(status, sort_keys=True, indent=2) + "\n")
        print("G7_S25_NUMERIC_ENGINEERING=" + status["status"], flush=True)
        print("G7_S25_FULL_REGRESSION=" + status["full_required_platform_regression"], flush=True)
        print("G7_S25_SCIENCE=NOT_RUN", flush=True)
    return rc

if __name__ == "__main__":
    raise SystemExit(main())
