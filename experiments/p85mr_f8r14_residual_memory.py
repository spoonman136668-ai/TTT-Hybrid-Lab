"""F8-R14: consumed F7 causal last-residual memory vs adequate linear baseline.

NO simulator import, source reset/do, new data generation, owner execution, or
sealed first-use. The 32 historical F7 FIT-holdout worlds have been inspected
in earlier research and are DEV-only retrospective observations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from experiments import p85mr_f8r2_linear_f7_retrospective as r2

HERE = Path(__file__).resolve().parent
FREEZE = HERE / "p85mr_f8r14_residual_memory_freeze.json"
FREEZE_BLOB = "3245b989c2230df9960293f5ab82e8c7e6875732"
R2_CODE = HERE / "p85mr_f8r2_linear_f7_retrospective.py"
R2_CODE_BLOB = "7b22ceac71709e9b06512a405265d7f81545346d"
SCHEMA = "ttt-hybrid.f8r14-retrospective-residual-memory-qualification.v1"
RIDGE = (0., 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.)
N_BOOT = 10000
BOOT_SEED = 1825990027
FIT, SELECT, HELDOUT, STEPS, D, CHANNELS = 80, 16, 32, 95, 7, 4


def _blob(raw: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
    ).hexdigest()


def locked() -> dict:
    raw = FREEZE.read_bytes()
    if _blob(raw) != FREEZE_BLOB:
        raise ValueError("F8R14_PREREGISTERED_FREEZE_BLOB_DRIFT")
    f = json.loads(raw)
    r2.locked()
    if (_blob(R2_CODE.read_bytes()) != R2_CODE_BLOB
            or f["schema"] != SCHEMA
            or f["stage"] != "FROZEN_BEFORE_F8R14_RETROSPECTIVE_NUMERICAL_RESULT"
            or f["data"]["cohort"] !=
                "80 FIT, 16 FIT-selector, 32 old FIT-holdout (already used, NOT fresh generalization)"
            or f["scope"]["f8r2_source_code_blob"] != R2_CODE_BLOB
            or f["scope"]["original_F7_archive_sha256"] != r2.F7_ZIP
            or f["scope"]["original_capture_sha256"] != r2.F7_CAPTURE
            or f["fitting"]["ridge_lambdas"] != list(RIDGE)
            or f["fitting"]["baseline_params"] != 28
            or f["fitting"]["memory_params"] != 16
            or f["fitting"]["combined_params"] != 44
            or f["fitting"]["bootstrap"]["resamples"] != N_BOOT
            or f["fitting"]["bootstrap"]["seed"] != BOOT_SEED
            or f["authority"]["new_CartPoleWind_source_actions"] is not False
            or f["authority"]["original_scientific_pass"] is not False):
        raise ValueError("F8R14_SOURCE_OR_AUTHORITY_DRIFT")
    return f


def validate(x: np.ndarray, y: np.ndarray) -> None:
    if (not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray)
            or x.shape != (FIT+SELECT+HELDOUT, STEPS, D)
            or y.shape != (FIT+SELECT+HELDOUT, STEPS, CHANNELS)
            or x.dtype != np.float64 or y.dtype != np.float64
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.allclose(np.linalg.norm(x,axis=-1), 1.,atol=1e-6)):
        raise ValueError("F8R14_UNTRUSTED_OR_INCOMPLETE_F7_FACTS")


def previous_factual_residual(residuals: np.ndarray, extra_delay=0) -> np.ndarray:
    """At event t, use only observed errors <= t-1-extra_delay."""
    if (not isinstance(residuals,np.ndarray)
            or residuals.ndim != 3 or residuals.shape[-1] != CHANNELS
            or residuals.dtype != np.float64
            or not np.isfinite(residuals).all()
            or type(extra_delay) is not int or extra_delay not in (0,7)):
        raise ValueError("F8R14_INVALID_LAG_OR_FUTURE_LABEL")
    out=np.zeros_like(residuals)
    gap=1+extra_delay
    if residuals.shape[1]>gap:
        out[:,gap:]=residuals[:,:-gap]
    return out


def train_lagged(x: np.ndarray, y: np.ndarray, *,
                 baseline_lambda:float, memory_lambda:float) -> tuple[np.ndarray,np.ndarray]:
    if baseline_lambda not in RIDGE or memory_lambda not in RIDGE:
        raise ValueError("F8R14_UNFROZEN_RIDGE")
    base=r2.fit(x,y,baseline_lambda)
    residual=y-x@base
    lag=previous_factual_residual(residual)
    correction=r2.fit(lag[:,1:],residual[:,1:],memory_lambda)
    return base,correction


def error_per_world(x: np.ndarray,y: np.ndarray,base: np.ndarray,
                    correction: np.ndarray, *,delay=0) -> np.ndarray:
    residual=y-x@base
    lag=previous_factual_residual(residual,delay)
    return np.square(residual-lag@correction).mean(axis=(1,2))


def evaluate_arrays(x:np.ndarray,y:np.ndarray) -> dict:
    """Pure retrospective FIT80/selector16/old-heldout32; no new source."""
    locked()
    validate(x,y)
    t0=time.perf_counter()
    xfit,yfit=x[:FIT],y[:FIT]
    xselect,yselect=x[FIT:FIT+SELECT],y[FIT:FIT+SELECT]
    xtest,ytest=x[FIT+SELECT:],y[FIT+SELECT:]
    baseline_grid=[]
    for lam in RIDGE:
        w=r2.fit(xfit,yfit,lam)
        err=float(r2.errors(xselect,yselect,w).mean())
        baseline_grid.append({"lambda":lam,"selector_mse":err})
    base_lambda=min(baseline_grid,key=lambda z:(z["selector_mse"],RIDGE.index(z["lambda"])))["lambda"]
    fitbase=r2.fit(xfit,yfit,base_lambda)
    fitres=yfit-xfit@fitbase
    selectres=yselect-xselect@fitbase
    train_prev=previous_factual_residual(fitres)
    select_prev=previous_factual_residual(selectres)
    memory_grid=[]
    for lam in RIDGE:
        w=r2.fit(train_prev[:,1:],fitres[:,1:],lam)
        score=float(np.square(selectres-select_prev@w).mean())
        memory_grid.append({"lambda":lam,"selector_mse":score})
    mem_lambda=min(memory_grid,key=lambda z:(z["selector_mse"],RIDGE.index(z["lambda"])))["lambda"]
    base,corr=train_lagged(x[:FIT+SELECT],y[:FIT+SELECT],
                          baseline_lambda=base_lambda,memory_lambda=mem_lambda)
    base_errors=error_per_world(xtest,ytest,base,np.zeros_like(corr))
    mem_errors=error_per_world(xtest,ytest,base,corr)
    delayed_errors=error_per_world(xtest,ytest,base,corr,delay=7)
    gain=base_errors-mem_errors
    idx=np.random.default_rng(BOOT_SEED).integers(
        0,HELDOUT,size=(N_BOOT,HELDOUT),dtype=np.int32)
    draws=gain[idx].mean(axis=1)
    lower=float(np.percentile(draws,2.5))
    upper=float(np.percentile(draws,97.5))
    b=float(base_errors.mean())
    m=float(mem_errors.mean())
    delayed=float(delayed_errors.mean())
    pct=(b-m)/b if b>0 else float("-inf")
    passes=(pct>=.10 and lower>0 and m<delayed)
    return {
        "schema":SCHEMA,
        "classification":"ALREADY_CONSUMED_F7_FIT_RETROSPECTIVE_DEV_ONLY",
        "fit_worlds":FIT,"selector_worlds":SELECT,
        "old_holdout_worlds_NOT_NEW":HELDOUT,
        "factual_events_per_world":STEPS,"old_holdout_factual_events":HELDOUT*STEPS,
        "baseline_parameters":28,"additional_memory_parameters":16,
        "original_f7_zip_sha256":r2.F7_ZIP,"original_capture_sha256":r2.F7_CAPTURE,
        "baseline_lambda":base_lambda,"memory_lambda":mem_lambda,
        "baseline_selection_grid":baseline_grid,
        "memory_selection_grid":memory_grid,
        "baseline_original_linear7_perworld_mse":base_errors.tolist(),
        "residual_memory_perworld_mse":mem_errors.tolist(),
        "delay7_negative_perworld_mse":delayed_errors.tolist(),
        "baseline_heldout_mse":b,"memory_heldout_mse":m,
        "delay7_heldout_mse":delayed,
        "relative_memory_improvement":pct,
        "paired_world_bootstrap_2p5_absolute_gain":lower,
        "paired_world_bootstrap_97p5_absolute_gain":upper,
        "paired_world_bootstrap_resamples":N_BOOT,
        "paired_world_bootstrap_seed":BOOT_SEED,
        "passed_frozen_retrospective_criterion":bool(passes),
        "independent_scientific_pass":False,
        "real_original_source_actions":0,"original_F7_firstuse_replayed":False,
        "new_unseen_cal_or_ood":False,"new_model_selected_source_actions":0,
        "wall_seconds":time.perf_counter()-t0,
    }


def evaluate_spent_archives(f7_archive:Path,trained_archive:Path)->dict:
    # Original F8-R2 verifies both immutable ZIP digests, canonical capture,
    # source IDs and all 12,288 real factual receipts before any fitting.
    x,y,_=r2.read_spent(f7_archive,trained_archive)
    return evaluate_arrays(x.astype(np.float64),y.astype(np.float64))


def refuse_original_firstuse(*_args,**_kwargs):
    raise PermissionError("F8R14_ORIGINAL_SOURCE_FIRSTUSE_NOT_AUTHORIZED")
