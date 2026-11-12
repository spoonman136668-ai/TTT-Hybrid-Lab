"""Post-outcome coefficient audit of already-consumed P78-MR DEV evidence.

This is a descriptive, source-neutral sensitivity calculation, NOT a new
experiment, holdout, training run, root-cause proof, or SCIENTIFIC_PASS.
Only the frozen repository summary and P77 update equation are read.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULT = HERE / "p78mr_fitcal_val_dev_result.json"
REFERENCE = HERE / "p77_write_policy.py"
RESULT_BLOB = "c180bfa8dab735000360ba2e87f51dbe42ec3d89"
REFERENCE_BLOB = "01051609f9af5b7d6932b5de7d30cfe7526d4e02"
SEEDS = (978201, 978203, 978207)
EPS = 1e-6


def git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def read_locked(path: Path, expected_sha: str) -> bytes:
    raw = path.read_bytes()
    if git_blob(raw) != expected_sha:
        raise ValueError("P78MR_POSTHOC_SOURCE_DRIFT")
    return raw


def logit(theta: tuple[float, float, float], error_mse: float,
          predicted_energy: float) -> float:
    """Exact non-directional3 logit for given positive feature values."""
    if (len(theta) != 3 or not all(math.isfinite(v) for v in theta)
            or not math.isfinite(error_mse) or error_mse < 0
            or not math.isfinite(predicted_energy) or predicted_energy < 0):
        raise ValueError("P78MR_POSTHOC_INVALID_FEATURE")
    return (theta[0] + theta[1] * math.log(error_mse + EPS)
            + theta[2] * math.log(predicted_energy + EPS))


def gain(theta: tuple[float, float, float], error_mse: float,
         predicted_energy: float) -> float:
    z = logit(theta, error_mse, predicted_energy)
    # Stable sigmoid, including extreme but valid counterfactual features.
    return math.exp(z)/(1.0+math.exp(z)) if z < 0 else 1.0/(1.0+math.exp(-z))


def audit() -> dict:
    read_locked(REFERENCE, REFERENCE_BLOB)
    saved = json.loads(read_locked(RESULT, RESULT_BLOB))
    rows = saved["experiment"]["seeds"]
    if tuple(x["seed"] for x in rows) != SEEDS:
        raise ValueError("P78MR_POSTHOC_SEED_DRIFT")
    out = []
    for row in rows:
        t = tuple(row["no_direction"]["selected_theta"])
        if len(t) != 3 or not all(math.isfinite(x) for x in t):
            raise ValueError("P78MR_POSTHOC_THETA_DRIFT")
        # Evaluator-chosen illustrative feature values; not sourced data.
        curve = [gain(t, m, 0.1) for m in (0.01, 0.1, 1.0)]
        out.append({
            "seed": row["seed"],
            "theta": list(t),
            "error_log_coefficient": t[1],
            "prediction_log_coefficient": t[2],
            "illustrative_gains_error_mse_0p01_0p1_1": curve,
            "inverse_error_slope_at_fixed_pred_energy": t[1] < 0,
        })
    return {
        "classification": "POSTHOC_EQUATION_FINGERPRINT_ONLY",
        "results_git_blob": RESULT_BLOB,
        "equation_git_blob": REFERENCE_BLOB,
        "fixed_predicted_energy_fixture": 0.1,
        "illustrative_error_mse_values": [0.01, 0.1, 1.0],
        "rows": out,
        "interpretation": ("At fixed predicted energy, the learned gain is "
                           "monotonically DECREASING with error size across "
                           "these saved coefficients; this is not proof the "
                           "effect caused heldout error improvement."),
        "source_birth": False,
        "training": False,
        "heldout_scored": False,
        "scientific_pass": False,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), sort_keys=True, indent=2))
