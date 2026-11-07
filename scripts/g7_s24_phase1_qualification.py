"""G7-S24 engineering-only phase1 proof of native encoders and gradient isolation.

No optimizer steps, heldout data, scientific baseline, network access, package
installation, CUDA or promotion. Source-bound to an existing pinned R3 WSL venv.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import g7_s24_phase0_engineering as phase0
import g7_r5_oneshot as custody


def require(value, name):
    if not bool(value):
        raise RuntimeError(name)


def focused_encoder_contract():
    import torch
    import g7_s24_encoders as enc
    torch.set_num_threads(1)
    torch.manual_seed(191)
    outcomes = []
    def check(name, value):
        require(value, "FAIL_" + name)
        outcomes.append(name)

    for seed, family in ((5101, "source_train"), (5113, "sine_train"),
                         (5129, "rational_train")):
        p = enc.development_modality_pairs(seed, 20, family)
        again = enc.development_modality_pairs(seed, 20, family)
        check("paired_repeatability_" + family,
              all(torch.equal(getattr(p, key), getattr(again, key))
                  for key in p.__dataclass_fields__))
        check("numeric_no_label_fields_" + family,
              set(p.__dataclass_fields__) ==
              {"sensor_a", "sensor_b", "graph_a_nodes", "graph_a_edges",
               "graph_b_nodes", "graph_b_edges"})
        check("independent_sensor_views_" + family,
              p.sensor_a.shape == p.sensor_b.shape == (20, 32, 4)
              and not torch.equal(p.sensor_a[:, :, 1:3], p.sensor_b[:, :, 1:3]))
        check("independent_graph_views_" + family,
              p.graph_a_nodes.shape == p.graph_b_nodes.shape == (20, 32, 6)
              and not torch.equal(p.graph_a_nodes, p.graph_b_nodes))
        sensor = enc.SensorGRU()
        graph = enc.EventMessagePassing()
        sensor.zero_grad(set_to_none=True)
        graph.zero_grad(set_to_none=True)
        sloss, gloss = enc.independent_losses(sensor, graph, p)
        check("sensor_finite_" + family,
              all(bool(torch.isfinite(v).all()) for v in sloss.values()))
        check("graph_finite_" + family,
              all(bool(torch.isfinite(v).all()) for v in gloss.values()))
        sloss["loss"].backward()
        check("sensor_gradient_only_" + family,
              any(v.grad is not None and torch.isfinite(v.grad).all()
                  for v in sensor.parameters())
              and all(v.grad is None for v in graph.parameters()))
        sensor.zero_grad(set_to_none=True)
        gloss["loss"].backward()
        check("graph_gradient_only_" + family,
              any(v.grad is not None and torch.isfinite(v.grad).all()
                  for v in graph.parameters())
              and all(v.grad is None for v in sensor.parameters()))
        graph.eval()
        perm = torch.randperm(32)
        inv = torch.argsort(perm)
        with torch.no_grad():
            a = graph(p.graph_a_nodes, p.graph_a_edges)
            b = graph(p.graph_a_nodes[:, perm], inv[p.graph_a_edges])
        check("graph_permutation_invariance_" + family,
              torch.allclose(a, b, atol=1e-6, rtol=1e-6))
    try:
        enc.development_modality_pairs(5201, 20, "source_train")
        require(False, "HELDOUT_LIKE_SEED_NOT_REJECTED")
    except ValueError as e:
        check("non_dev_seed_rejected", str(e) == "S24_PHASE1_DEVELOPMENT_SEED_ONLY")
    zeros = torch.zeros(20, 16)
    collapsed = enc.within_modality_jepa(zeros, zeros)
    check("collapse_penalty", float(collapsed["variance"]) > 0
          and bool(torch.isfinite(collapsed["loss"])))
    return {"status": "PASS", "count": len(outcomes), "names": outcomes}


def regression_if_available() -> dict:
    """The full suite is executed only in the existing verified interpreter."""
    if importlib.util.find_spec("pytest") is None:
        return {"status": "UNVERIFIED_PYTEST_NOT_INSTALLED_IN_PINNED_R3"}
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(ROOT) + os.pathsep + str(ROOT / "scripts")})
    try:
        done = subprocess.run(
            [sys.executable, "-B", "-m", "pytest", "-q",
             "-p", "no:cacheprovider", "tests"],
            cwd=ROOT, text=True, capture_output=True, timeout=270,
            check=False, env=env)
        return {"status": "PASS" if done.returncode == 0 else "FAIL",
                "exit_code": done.returncode,
                "summary": done.stdout[-650:]}
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT"}


def inner(sha: str):
    receipt = ROOT / "local_runs" / "g7_s24_phase1_engineering.json"
    out = {"schema": "ttt-hybrid.g7-s24.phase1-engineering.v1",
           "sha": sha, "status": "BLOCKED_NOT_RUN",
           "scientific_first_use": "NOT_RUN",
           "training": "NOT_RUN", "heldout_used": False,
           "cuda": "NOT_USED", "packages_installed": False,
           "full_regression": "NOT_RUN"}
    rc = 2
    try:
        phase0.validate_source(sha)
        out["pinned_manifest_sha256"] = phase0.package_fingerprint()
        t0 = time.perf_counter()
        out["phase0"] = phase0.direct_contract()
        out["phase1"] = focused_encoder_contract()
        require(out["phase0"]["count"] == 67 and
                out["phase1"]["count"] == 29, "FOCUSED_TEST_COUNT_DRIFT")
        out["focused_elapsed_seconds"] = round(time.perf_counter() - t0, 3)
        out["full_regression"] = regression_if_available()
        out["status"] = "PASS_PHASE0_PHASE1_FOCUSED_ENGINEERING"
        rc = 0
    except Exception as e:
        out["status"] = "BLOCKED_" + type(e).__name__
        out["reason"] = str(e)[:100]
    finally:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
        print("G7_S24_PHASE1=" + out["status"], flush=True)
        print("G7_S24_FULL_REGRESSION=" +
              (out["full_regression"].get("status", "NOT_RUN")
               if isinstance(out["full_regression"], dict)
               else out["full_regression"]), flush=True)
        print("G7_S24_SCIENCE=NOT_RUN", flush=True)
    return rc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sha", required=True)
    p.add_argument("--inner", action="store_true")
    args = p.parse_args()
    if args.inner:
        return inner(args.sha)
    try:
        phase0.validate_source(args.sha)
        venv = (Path.home().resolve() / ".local/share/ttt-hybrid"
                / ("g7-r3-" + custody.R3_SOURCE_SHA[:12])
                / "venv/bin/python")
        require(venv.is_file(), "PINNED_ENV_MISSING")
        env = dict(os.environ)
        env.update({"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
        child = subprocess.run(
            [str(venv), "-I", "-B", __file__, "--sha", args.sha, "--inner"],
            cwd=ROOT, capture_output=True, text=True, timeout=330,
            check=False, env=env)
        print(child.stdout[-1100:], end="", flush=True)
        return child.returncode
    except Exception as e:
        print("G7_S24_PHASE1=BLOCKED_" + type(e).__name__, flush=True)
        print("G7_S24_SCIENCE=NOT_RUN", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
