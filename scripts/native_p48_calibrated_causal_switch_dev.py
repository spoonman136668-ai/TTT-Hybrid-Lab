"""P48 frozen DEV: prospective null-calibrated P47 causal-change evidence.

This is a conventional, source-free-of-external-data comparison; NOT a trained
neural model, independent scientific first use, or original-source acceptance.
The calibration cohort is stable-only and cannot contain evaluation worlds.
No P47 source or historic consumed P47 cohorts are modified or replayed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import time
from pathlib import Path

import numpy as np

from scripts import native_p47_change_vs_nuisance_dev as p47

SCHEMA = 'ttt-hybrid.p48.calibrated-causal-change-dev-receipt.v1'
COHORTS = {
    770001: (4096, (.35, 2.0), 'calibration'),
    770101: (1536, (.35, .90), 'evaluation'),
    770103: (1536, (.95, 1.35), 'evaluation'),
    770107: (1536, (1.40, 2.0), 'evaluation'),
}
QUANTILE = .99


def fresh_source(seed: int) -> dict:
    """Generate *actual* observations only from a frozen independent DEV seed."""
    if type(seed) is not int or seed not in COHORTS:
        raise ValueError('P48_UNREGISTERED_SOURCE_FIRST_USE_REFUSED')
    n, band, role = COHORTS[seed]
    g = np.random.default_rng(seed)
    if role == 'calibration':
        kind = np.full(n, 2, dtype=np.int64)
    else:
        kind = np.repeat(np.array([0, 1, 2], dtype=np.int64), [768, 384, 384])
        g.shuffle(kind)
    k = g.choice(np.array(p47.SPLITS, dtype=np.int64), n)
    initial = g.choice(np.array([-1., 1.]), n)
    a = initial[:, None] * np.power(-1., np.arange(p47.N))[None, :]
    u = lambda lo, hi: g.uniform(lo, hi, (n, 1))
    b0, b1, b2 = u(-.2, .2), u(-.4, .4), u(-.1, .1)
    amp = u(0, .08)
    phase = u(-math.pi, math.pi)
    freq = u(*band)
    beta = g.choice(np.array([-1., 1.]), (n, 1)) * u(.25, .45)
    changed = (kind[:, None] == 0) & (np.arange(p47.N)[None, :] >= k[:, None])
    beta_seq = beta * np.where(changed, -1., 1.)
    # Correct the P47 manifest typo: EXACT preregistered jump absolute
    # amplitude is 0.40 to 0.70 (not 0.40 to 7).
    jump = g.choice(np.array([-1., 1.]), (n, 1)) * u(.40, .70)
    shock = (kind[:, None] == 1) & (np.arange(p47.N)[None, :] >= k[:, None])
    nuisance = (b0 + b1 * p47.T + b2 * p47.T**2
                + amp * np.sin(2 * math.pi * freq * p47.T + phase) + jump * shock)
    realized_y = nuisance + beta_seq * a + g.normal(0, .02, (n, p47.N))
    physical_effect = beta_seq[:, -1].copy()
    trace = hashlib.sha256(b''.join(z.tobytes() for z in
        (a, realized_y, physical_effect, kind, k))).hexdigest()
    return dict(a=a, y=realized_y, kind=kind, k=k,
                beta=physical_effect, sha=trace, seed=seed, role=role)


def fitted_hypotheses(a: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """P47 original designs, fixed alternatives and BIC, no new search tuning."""
    # Original source's validator rejects non-alternating or fabricated actions.
    p47.estimate(a, y)
    n = len(a)
    best_null = np.full(n, np.inf)
    best_null_effect = np.zeros(n)
    best_null_kind = np.full(n, -1, dtype=np.int64)
    best_null_k = np.full(n, -1, dtype=np.int64)
    best_causal = np.full(n, np.inf)
    best_causal_effect = np.zeros(n)
    best_causal_k = np.full(n, -1, dtype=np.int64)
    for sign in (-1., 1.):
        ids = np.flatnonzero(a[:, 0] == sign)
        if not len(ids):
            continue
        full_eff, full_bic = p47.fit(y[ids], sign)
        best_null[ids] = full_bic
        best_null_effect[ids] = full_eff
        for k in p47.SPLITS:
            null_effect, null_bic = p47.fit(y[ids], sign, 'nuisance', k)
            chosen = null_bic < best_null[ids]
            idx = ids[chosen]
            best_null[idx] = null_bic[chosen]
            best_null_effect[idx] = null_effect[chosen]
            best_null_kind[idx] = 1
            best_null_k[idx] = k
            causal_effect, causal_bic = p47.fit(y[ids], sign, 'causal', k)
            chosen = causal_bic < best_causal[ids]
            idx = ids[chosen]
            best_causal[idx] = causal_bic[chosen]
            best_causal_effect[idx] = causal_effect[chosen]
            best_causal_k[idx] = k
    return dict(margin=best_null - best_causal,
                null_effect=best_null_effect, null_kind=best_null_kind,
                null_k=best_null_k,
                causal_effect=best_causal_effect, causal_k=best_causal_k)


def calibrated_threshold(calibration: dict) -> tuple[float, str]:
    if calibration['seed'] != 770001 or calibration['role'] != 'calibration':
        raise ValueError('P48_CALIBRATION_SOURCE_CUSTODY_REFUSED')
    if not np.all(calibration['kind'] == 2):
        raise ValueError('P48_CALIBRATION_CONTAINS_NONSTABLE_WORLD')
    history = fitted_hypotheses(calibration['a'], calibration['y'])
    margin = history['margin']
    if not np.isfinite(margin).all():
        raise ValueError('P48_NONFINITE_CALIBRATION_MARGIN')
    # Fixed prospective 0.99 quantile with higher interpolation.
    threshold = float(np.quantile(margin, QUANTILE, method='higher'))
    return threshold, hashlib.sha256(margin.tobytes()).hexdigest()


def predict(a: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    if not np.isfinite(threshold):
        raise ValueError('P48_INVALID_FROZEN_THRESHOLD')
    p47_result = p47.estimate(a, y)
    fits = fitted_hypotheses(a, y)
    # Never select causal unless it beat both the unchanged and nuisance models
    # and its *prospectively calibrated* improvement clears stable null.
    accepted = (fits['margin'] > max(0.0, threshold))
    result = {k: v.copy() for k, v in p47_result.items()}
    result['candidate'] = np.where(accepted, fits['causal_effect'], fits['null_effect'])
    result['model_kind'] = np.where(accepted, 0, fits['null_kind'])
    result['chosen_k'] = np.where(accepted, fits['causal_k'], fits['null_k'])
    return result


def study() -> dict:
    started = time.perf_counter()
    calibration = fresh_source(770001)
    threshold, margin_sha = calibrated_threshold(calibration)
    reports = []
    for seed in (770101, 770103, 770107):
        data = fresh_source(seed)
        reference = p47.estimate(data['a'], data['y'])
        original = p47.metrics(data, reference)
        proposed = predict(data['a'], data['y'], threshold)
        candidate = p47.metrics(data, proposed)
        # Independently reload factual actions and observations (not scorer labels)
        # to check exact cold deterministic replay, using the *same* pre-fit margin.
        buf = io.BytesIO()
        np.savez_compressed(buf, a=data['a'], y=data['y'])
        buf.seek(0)
        with np.load(buf, allow_pickle=False) as restored:
            cold = predict(restored['a'], restored['y'], threshold)
        exact = all(np.array_equal(proposed[k], cold[k]) for k in proposed)
        reports.append(dict(seed=seed, source_trace_sha256=data['sha'],
                            original=original, calibrated=candidate, cold_exact=exact,
                            margin_pass_count=int((proposed['model_kind']==0).sum())))
    return dict(schema=SCHEMA,
                scope='AUTHORED_DEVELOPMENT_NO_EXTERNAL_SCIENCE_SOURCE',
                original_P47_sha='4d61a660b057f73b3f907c59a37dafa49b4f38fb',
                P47_source_git_blob='9be334264131db3bae79f19bed3271e060a1a903',
                calibration_seed=770001, calibration_source_sha256=calibration['sha'],
                calibration_margin_sha256=margin_sha,
                quantile=QUANTILE, threshold=threshold,
                reports=reports,
                all_frozen_gates=all(row['calibrated']['frozen_dev_pass']
                                     and row['cold_exact'] for row in reports),
                elapsed_seconds=time.perf_counter()-started,
                original_owner_scientific_first_use='NOT_RUN',
                original_exact_Windows_Linux_full_regression='NOT_RUN',
                strong_trained_GRU_MLP_controls='NOT_RUN',
                FORGE_controller_or_dispatch='UNTOUCHED')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--receipt', type=Path, required=True)
    args=p.parse_args()
    result=study()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    for row in result['reports']:
        a=row['original']; b=row['calibrated']
        print(json.dumps(dict(seed=row['seed'], source_sha=row['source_trace_sha256'],
            original_detection=a['detection'], calibrated_detection=b['detection'],
            original_gate=a['gates'], candidate_gate=b['gates'],
            original_stable_MAE=a['results']['candidate']['stable']['mae'],
            calibrated_stable_MAE=b['results']['candidate']['stable']['mae'],
            candidate_causal_MAE=b['results']['candidate']['causal']['mae'],
            cold_exact=row['cold_exact']),sort_keys=True))
    print(json.dumps(dict(threshold=result['threshold'],
         calibration_sha256=result['calibration_source_sha256'],
         all_frozen_gates=result['all_frozen_gates'],
         elapsed_seconds=result['elapsed_seconds']),sort_keys=True))


if __name__=='__main__':
    main()
