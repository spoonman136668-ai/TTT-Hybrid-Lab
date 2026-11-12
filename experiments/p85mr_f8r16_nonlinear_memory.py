"""F8-R16: spent-F7 nonlinear dynamics vs previous factual residual memory.

Only consumes previously captured, immutable owner F7 ZIP archives. This is
retrospective research and never imports the external environment or issues a
source action, model-selected step, or protected first-use request.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments import p85mr_f8r2_linear_f7_retrospective as r2
from experiments import p85mr_f8r14_residual_memory as r14

HERE = Path(__file__).resolve().parent
FREEZE = HERE / "p85mr_f8r16_nonlinear_memory_freeze.json"
FREEZE_BLOB = "80a4d90e3530724cfd247368468e6fafea6e6966"
R14_RESULT = HERE / "p85mr_f8r14_residual_memory_result.json"
R14_RESULT_BLOB = "3ff04988f51d07ed2c58f3dcdc8335f86beacf17"
R2_SRC = HERE / "p85mr_f8r2_linear_f7_retrospective.py"
R2_SRC_BLOB = "7b22ceac71709e9b06512a405265d7f81545346d"
ARMS = ("linear7", "quadratic35")
LAMBDAS = tuple(r2.GRID)
SEED = 1825990128
BOOTSTRAPS = 10000
WORLDS_TEST = 32
SCHEMA = "ttt-hybrid.f8r16-spent-f7-nonlinear-vs-residual-memory.v1"


def _blob(raw: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
    ).hexdigest()


def locked() -> dict:
    raw = FREEZE.read_bytes()
    if _blob(raw) != FREEZE_BLOB:
        raise ValueError("F8R16_FROZEN_PROTOCOL_BLOB_DRIFT")
    f = json.loads(raw)
    r14.locked()
    if (_blob(R14_RESULT.read_bytes()) != R14_RESULT_BLOB
            or _blob(R2_SRC.read_bytes()) != R2_SRC_BLOB
            or f.get("schema") != SCHEMA
            or f.get("stage") != "FROZEN_BEFORE_R16_ACTUAL_NUMERICAL_RESULT"
            or f["source"]["source_archive_sha256"] != r2.F7_ZIP
            or f["source"]["training_archive_sha256"] != r2.F7_TRAIN_ZIP
            or f["source"]["capture_sha256"] != r2.F7_CAPTURE
            or f["parent"]["r2_source_blob"] != R2_SRC_BLOB
            or f["parent"]["r14_result_blob"] != R14_RESULT_BLOB
            or f["split"]["training_worlds"] != 80
            or f["split"]["lambda_selector_worlds"] != 16
            or f["split"]["old_evaluated_test_worlds"] != WORLDS_TEST
            or f["fitting"]["ridge_grid"] != list(LAMBDAS)
            or f["thresholds"]["primary"] !=
                "relative MSE gain for quadratic35+lag1 vs quadratic35 at least 10%"
            or f["thresholds"]["bootstrap_seed"] != SEED
            or f["controls"]["independent_scientific_pass"] is not False
            or f["native_owner_requirements"]["firstuse"] is not False):
        raise ValueError("F8R16_SOURCE_OR_FIRSTUSE_SCOPE_DRIFT")
    return f


def _score_one(x: np.ndarray, y: np.ndarray, arm: str) -> dict:
    """FIT80 selector16 refit96, then score only old previously seen worlds."""
    a = r2.features(x, arm)
    if a.shape != (128, 95, 7 if arm == "linear7" else 35):
        raise ValueError("F8R16_BAD_ORIGINAL_FEATURES")
    xfit, yfit = a[:80], y[:80]
    xsel, ysel = a[80:96], y[80:96]
    baseline_grid = []
    for lam in LAMBDAS:
        w = r2.fit(xfit, yfit, lam)
        baseline_grid.append(float(r2.errors(xsel, ysel, w).mean()))
    base_lam = LAMBDAS[int(np.argmin(baseline_grid))]
    w80 = r2.fit(xfit, yfit, base_lam)
    rfit = yfit - xfit @ w80
    rsel = ysel - xsel @ w80
    pastfit = r14.previous_factual_residual(rfit)
    pastsel = r14.previous_factual_residual(rsel)
    memory_grid = []
    for lam in LAMBDAS:
        corr = r2.fit(pastfit[:, 1:], rfit[:, 1:], lam)
        memory_grid.append(float(np.square(rsel - pastsel @ corr).mean()))
    mem_lam = LAMBDAS[int(np.argmin(memory_grid))]
    base, corr = r14.train_lagged(
        a[:96], y[:96], baseline_lambda=base_lam,
        memory_lambda=mem_lam)
    eb = r14.error_per_world(a[96:], y[96:], base, np.zeros_like(corr))
    em = r14.error_per_world(a[96:], y[96:], base, corr)
    en = r14.error_per_world(a[96:], y[96:], base, corr, delay=7)
    if (eb.shape != (WORLDS_TEST,) or em.shape != (WORLDS_TEST,)
            or en.shape != (WORLDS_TEST,)
            or any(not np.isfinite(z).all() for z in (eb, em, en))):
        raise ValueError("F8R16_INVALID_RETROSPECTIVE_METRICS")
    gain = eb - em
    cluster_draws = np.random.default_rng(SEED).integers(
        0, WORLDS_TEST, size=(BOOTSTRAPS, WORLDS_TEST), dtype=np.int32)
    draws = gain[cluster_draws].mean(axis=1)
    lower = float(np.percentile(draws, 2.5))
    upper = float(np.percentile(draws, 97.5))
    b = float(eb.mean())
    m = float(em.mean())
    n = float(en.mean())
    improvement = (b-m)/b if b > 0 else -1.0
    def _hex(a):
        return hashlib.sha256(np.asarray(a, dtype="<f8").tobytes()).hexdigest()
    return {
        "arm": arm,
        "base_parameters": int(base.size),
        "memory_additional_parameters": int(corr.size),
        "baseline_lambda": base_lam, "memory_lambda": mem_lam,
        "selector_baseline_mse": baseline_grid,
        "selector_memory_mse": memory_grid,
        "baseline_coefficient_sha256": _hex(base),
        "memory_coefficient_sha256": _hex(corr),
        "baseline_old_F7_mse": b,
        "memory_old_F7_mse": m,
        "lag8_negative_old_F7_mse": n,
        "relative_memory_improvement": improvement,
        "bootstrap_lower_absolute_gain": lower,
        "bootstrap_upper_absolute_gain": upper,
        "frozen_criterion_pass": bool(
            improvement >= .10 and lower > 0 and m < n),
        "perworld_baseline_mse": eb.tolist(),
        "perworld_memory_mse": em.tolist(),
        "perworld_delay8_mse": en.tolist(),
        "perworld_baseline_sha256": _hex(eb),
        "perworld_memory_sha256": _hex(em),
        "perworld_delay8_sha256": _hex(en),
    }


def evaluate_spent_arrays(x: np.ndarray, y: np.ndarray) -> dict:
    locked()
    r14.validate(x, y)
    started = time.perf_counter()
    a = {name: _score_one(x, y, name) for name in ARMS}
    prior = json.loads(R14_RESULT.read_bytes())
    if (abs(a["linear7"]["baseline_old_F7_mse"]
                - prior["baseline_heldout_mse"]) > 1e-13
            or abs(a["linear7"]["memory_old_F7_mse"]
                - prior["memory_heldout_mse"]) > 1e-13
            or a["quadratic35"]["baseline_coefficient_sha256"] !=
                "053f574d94d76654d0cb22114aab7b5986eecad66108f2c96514d6ef73137976"
            or abs(a["quadratic35"]["baseline_old_F7_mse"]
                   - 2.7441133476852674e-6) > 1e-13):
        raise ValueError("F8R16_R2_OR_R14_ORIGINAL_NUMERICAL_PARITY_DRIFT")
    return {
        "schema": SCHEMA,
        "classification": "ALREADY_SPENT_F7_RETROSPECTIVE_ONLY",
        "already_consumed_worlds": 128, "fit_worlds": 80,
        "selector_worlds": 16, "already_inspected_test_worlds": WORLDS_TEST,
        "original_factual_test_effects": WORLDS_TEST * 95,
        "new_original_source_effects": 0,
        "original_source_firstuse_authorized": False,
        "independent_scientific_pass": False,
        "arms": a,
        "source_zip_sha256": r2.F7_ZIP,
        "training_zip_sha256": r2.F7_TRAIN_ZIP,
        "bootstrap_worlds": WORLDS_TEST,
        "bootstrap_resamples": BOOTSTRAPS,
        "bootstrap_seed": SEED,
        "elapsed_dev_cpu_seconds": time.perf_counter() - started,
    }


def evaluate_spent_archives(f7_zip: Path, trained_zip: Path) -> dict:
    # Reuse R2 exact trusted original GitHub artifact and receipt reader.
    x, y, _ = r2.read_spent(f7_zip, trained_zip)
    return evaluate_spent_arrays(x.astype(np.float64), y.astype(np.float64))


def refuse_original_firstuse(*_args, **_kwargs):
    raise PermissionError("F8R16_NEW_ORIGINAL_FIRSTUSE_NOT_AUTHORIZED")
