"""G7-S24 Phase 0 engineering-only proof; no scientific training or held-out evaluation.

Runs on the existing non-root isolated WSL runner. The current frozen S24
study's development seeds alone may be used. Full pytest regression is
attempted only if available in the already-pinned R3 environment: no installs.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import g7_r5_oneshot as custody

S24_BASE = "7d1f8bb1547d4d39df7412091a68dabb03f0ac3f"
S24_FROZEN_BLOB = "4d5ad298560a169d1fe95feaa90ffc9a89a2e0e5"
S20_SOURCE_BLOB = "ac3f9ddd5b879e53e62030707fb6419adaf732ca"
DEV_SEEDS = (5101, 5113, 5129)
REGIMES = ("source_train", "sine_train", "rational_train",
           "source_ood", "sine_ood", "rational_ood")


def demand(ok, label):
    if not ok:
        raise RuntimeError(label)


def package_fingerprint():
    fingerprint, distributions = custody.manifest()
    demand(fingerprint == custody.R3_MANIFEST_SHA256, "PINNED_PACKAGE_DRIFT")
    demand(distributions.get("torch") == custody.TORCH
           and distributions.get("triton") == custody.TRITON,
           "PINNED_TORCH_OR_TRITON_DRIFT")
    return fingerprint


def validate_source(sha):
    demand(os.geteuid() == 1000, "NORMAL_USER_REQUIRED")
    demand("microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower(),
           "ISOLATED_WSL_REQUIRED")
    demand(custody.git(ROOT, "rev-parse", "HEAD") == sha, "HEAD_DRIFT")
    demand(not custody.git(ROOT, "status", "--porcelain", "--untracked-files=all"),
           "DIRTY_SOURCE")
    demand(custody.sha256(ROOT / "source_pins.json") == custody.R3_PINS_SHA256,
           "PINNED_SOURCE_CONFIG_DRIFT")
    demand(custody.git(ROOT, "hash-object", "--",
                       "experiments/g7_s24_frozen_good_views.json") ==
           S24_FROZEN_BLOB, "FROZEN_S24_PROTOCOL_DRIFT")
    demand(custody.git(ROOT, "hash-object", "--",
                       "scripts/g7_s20_hidden_multicycle.py") ==
           S20_SOURCE_BLOB, "S20_SIMULATOR_DRIFT")
    frozen = json.loads((ROOT / "experiments/g7_s24_frozen_good_views.json")
                        .read_text(encoding="utf-8"))
    demand(frozen.get("schema") ==
           "ttt-hybrid.g7-s24.good-view-native-continual-world-model.v1",
           "S24_SCHEMA_DRIFT")
    demand(tuple(frozen["train"]["development_source_seeds"]) == DEV_SEEDS,
           "DEVELOPMENT_SEEDS_DRIFT")
    demand(frozen["qualifications"]["current_status"] ==
           "PROTOCOL_REGISTERED_NOT_EXECUTED", "SCIENCE_STATUS_DRIFT")
    demand("NO NEW SCIENTIFIC FIRST USE" in
           frozen["authority_and_isolation"]["execution"],
           "FIRST_USE_NOT_ISOLATED")


def direct_contract():
    """Focused native numerical contract: no inference, optimization or fitting."""
    import torch
    import g7_s24_views as views

    torch.set_num_threads(1)
    results = []
    started = time.perf_counter()

    def check(name, condition):
        demand(bool(condition), "FAIL_" + name)
        results.append(name)

    for regime in REGIMES:
        for seed in DEV_SEEDS:
            check("s20_byte_replay_" + regime + "_" + str(seed),
                  views.reference_equivalent_trace(seed, 3, regime))

    for regime in REGIMES:
        seed = DEV_SEEDS[0]
        good = views.independent_views(seed, 4, regime)
        repeat = views.independent_views(seed, 4, regime)
        bad = views.same_trace_bad_views(seed, 4, regime)
        check("numeric_shapes_" + regime,
              good.sensor_sequence.shape == (4, 32, 4)
              and good.event_nodes.shape == (4, 32, 6)
              and good.event_edges.shape == (2, 62)
              and good.sensor_sequence.dtype.is_floating_point
              and good.event_nodes.dtype.is_floating_point)
        check("view_determinism_" + regime,
              all(torch.equal(getattr(good, f), getattr(repeat, f))
                  for f in good.__dataclass_fields__))
        check("no_label_or_world_key_fields_" + regime,
              set(good.__dataclass_fields__) ==
              {"sensor_sequence", "event_nodes", "event_edges"})
        order = torch.argsort(good.event_nodes[0, :, 5])
        bad_order = torch.argsort(bad.event_nodes[0, :, 5])
        check("independent_action_schedule_" + regime,
              not torch.equal(good.sensor_sequence[:, :, 1:3],
                              good.event_nodes[:, order, 1:3]))
        check("independent_observation_noise_" + regime,
              not torch.equal(good.sensor_sequence[:, :, 0:1],
                              good.event_nodes[:, order, 0:1]))
        check("same_trace_negative_" + regime,
              torch.equal(bad.sensor_sequence,
                          bad.event_nodes[:, bad_order, :4]))
        src, dst = good.event_edges
        delta = (good.event_nodes[0, src, 5]
                 - good.event_nodes[0, dst, 5]).abs()
        check("graph_chronology_" + regime,
              torch.allclose(delta, torch.full((62,), 1 / 31),
                             atol=1e-6))
        check("observations_finite_" + regime,
              torch.isfinite(good.sensor_sequence).all()
              and torch.isfinite(good.event_nodes).all())

    try:
        views.independent_views(DEV_SEEDS[0], 3, "not_a_family")
    except ValueError as exc:
        check("unknown_regime_rejected",
              str(exc) == "UNREGISTERED_S24_REGIME")
    else:
        demand(False, "INVALID_REGIME_ACCEPTED")
    return {"status": "PASS", "count": len(results),
            "checks": results,
            "elapsed_seconds": round(time.perf_counter() - started, 3)}


def inner(sha: str, receipt: Path) -> int:
    record = {
        "schema": "ttt-hybrid.g7-s24.phase0-engineering.v1",
        "sha": sha, "status": "UNVERIFIED",
        "science": "NOT_RUN", "training": "NOT_RUN",
        "evaluation": "NOT_RUN", "heldout_seeds_used": False,
        "qualifier": "NORMAL_USER_R3_PINNED_ENV",
        "packages_installed": False, "cuda_used": False,
        "full_regression": "NOT_RUN",
    }
    outcome = 1
    try:
        validate_source(sha)
        record["pinned_manifest_sha256"] = package_fingerprint()
        record["focused"] = direct_contract()
        if importlib.util.find_spec("pytest") is None:
            record["full_regression"] = "UNVERIFIED_PYTEST_NOT_IN_PINNED_ENV"
        else:
            env = dict(os.environ)
            env.update({"CUDA_VISIBLE_DEVICES": "-1",
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONPATH": str(ROOT) + os.pathsep + str(ROOT / "scripts"),
                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            try:
                result = subprocess.run(
                    [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
                    cwd=ROOT, env=env, text=True, capture_output=True,
                    check=False, timeout=270)
                record["regression_exit_code"] = result.returncode
                # Short bounded test summary only; no full output of host files.
                record["regression_summary"] = result.stdout[-900:]
                record["full_regression"] = (
                    "PASS" if result.returncode == 0 else "FAIL")
            except subprocess.TimeoutExpired:
                record["full_regression"] = "TIMEOUT"
        record["status"] = ("PASS_FOCUSED_ENGINEERING_ONLY" if
                            record["focused"]["status"] == "PASS" else
                            "FAIL_FOCUSED_ENGINEERING")
        outcome = 0
    except Exception as exc:
        record["status"] = "BLOCKED_" + type(exc).__name__
        record["reason"] = str(exc)[:120]
    finally:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
        print("G7_S24_PHASE0=" + record["status"], flush=True)
        print("G7_S24_FOCUSED_COUNT=" +
              str(record.get("focused", {}).get("count", 0)), flush=True)
        print("G7_S24_FULL_REGRESSION=" + record["full_regression"], flush=True)
        print("G7_S24_SCIENCE=NOT_RUN", flush=True)
    return outcome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected-sha", required=True)
    ap.add_argument("--inner", action="store_true")
    args = ap.parse_args()
    if args.inner:
        return inner(args.expected_sha,
                     ROOT / "local_runs" / "g7_s24_phase0_engineering.json")
    try:
        validate_source(args.expected_sha)
        venv = (Path.home().resolve() / ".local/share/ttt-hybrid"
                / ("g7-r3-" + custody.R3_SOURCE_SHA[:12]) / "venv/bin/python")
        demand(venv.is_file(), "PINNED_R3_ENV_MISSING")
        env = dict(os.environ)
        env.update({"CUDA_VISIBLE_DEVICES": "-1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        proc = subprocess.run(
            [str(venv), "-I", "-B", __file__, "--expected-sha",
             args.expected_sha, "--inner"], cwd=ROOT, env=env,
            text=True, capture_output=True, timeout=320)
        print(proc.stdout[-1500:], end="", flush=True)
        return proc.returncode
    except Exception as exc:
        print("G7_S24_PHASE0=BLOCKED_" + type(exc).__name__, flush=True)
        print("G7_S24_SCIENCE=NOT_RUN", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
