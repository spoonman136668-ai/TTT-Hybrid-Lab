"""F8-R2: read-only, retrospective linear dynamics controls on spent F7 FIT.

No simulator, no external effect generation, no owner/runners/training
workflow, no new OOD or independent scientific acceptance. CLI explicitly
requires TWO previously produced exact GitHub artifact ZIP files.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/p85mr_f8r2_f7_linear_baseline_freeze.json"
FREEZE_BLOB = "057e6744c1e147afe9e5bafd1f27153ec849ddf4"
F7_ZIP = "5480c0a8d2e518f9c0c9e3e011e3dfa9d3019fb720d10641c02bb36b5023f0ef"
F7_CAPTURE = "7df8e2a2dc276cb90fbcac91619083dec7911453d5af10d2ccbdcfd45e5007d4"
F7_TRAIN_ZIP = "9ef1eb672f965a88bb6098e1fbbc44490d42c8440a69abe2a840a19a1d160d38"
SCALES = np.array([2.4, 3.0, .2095, 3.5], dtype=np.float64)
GRID = (0., 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.)
ARMS = ("state5", "action3", "linear7", "quadratic35")
SSM_MSE = (3.5338950e-6, 4.9913427e-6, 3.3410481e-6)
SHUFFLE_SEED = 1823999


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def locked():
    raw = FREEZE.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if blob != FREEZE_BLOB:
        raise ValueError("F8R2_PRECOMMITTED_FREEZE_DRIFT")
    f = json.loads(raw)
    if (f["schema"] != "ttt-hybrid.f8r2-retrospective-f7-factual-linear-dynamics.v1"
            or f["source"]["zip_sha256"] != F7_ZIP
            or f["source"]["trained_archive_sha256"] != F7_TRAIN_ZIP
            or f["split"]["fit_train_worlds"] != "indices 0..79 (80 worlds)"
            or f["training"]["ridge_grid"] != list(GRID)
            or not f["authority"]["only_spent_F7_source_readonly"]
            or f["authority"]["new_source_first_use"]
            or f["authority"]["unseen_F8_CAL_OOD"]):
        raise ValueError("F8R2_PROTOCOL_DRIFT")
    return f


def read_spent(f7_archive: Path, trained_archive: Path):
    locked()
    if (sha256(f7_archive.read_bytes()) != F7_ZIP
            or sha256(trained_archive.read_bytes()) != F7_TRAIN_ZIP):
        raise ValueError("F8R2_ORIGINAL_ARTIFACT_DRIFT")
    with zipfile.ZipFile(f7_archive) as z:
        raw = z.read("p85mr_f7_fit_capture.json")
    if sha256(raw) != F7_CAPTURE:
        raise ValueError("F8R2_CAPTURE_DRIFT")
    capture = json.loads(raw)
    with zipfile.ZipFile(trained_archive) as z:
        trained = json.loads(z.read("p85mr_f7_recovery_fit_result.json"))
    if (capture.get("schema") != "ttt-hybrid.p80mr.cartpole-do-capture.v1"
            or capture.get("source_id") != "p85mr-f7-fit-43195349254"
            or capture.get("custody_ref") != "ghrun:43195349254"
            or capture.get("upstream_commit") != "c94e9a4cd3641588b6b8507e71ccee5c27032bdf"
            or len(capture.get("worlds", ())) != 128
            or len(trained.get("results", ())) != 9
            or trained["FIT_bypass_wins_seeds"] != 0
            or trained["scientific_pass"] is not False):
        raise ValueError("F8R2_SOURCE_OR_PRIOR_SCIENCE_DRIFT")
    xworld, yworld, observed_refs = [], [], set()
    for wi, world in enumerate(capture["worlds"]):
        trans = world["transitions"]
        if len(trans) != 96 or world["probe_event_ids"] != list(range(2, 97)):
            raise ValueError("F8R2_WORLD_PROBES_DRIFT")
        features, targets = [], []
        last = None
        for t, event in enumerate(trans, 1):
            ref = f"ghrun:43195349254:F7-fit:{wi}:{t}"
            act = event["executed_action"]
            if (event["event_id"] != t or event["mode"] != "do"
                    or event["receipt_ref"] != ref or ref in observed_refs
                    or type(act) is not int or act not in (0, 1)
                    or event["reward"] != 1. or event["terminated"]
                    or event["truncated"] != (t == 96)):
                raise ValueError("F8R2_ACTION_RECEIPT_DRIFT")
            observed_refs.add(ref)
            before = np.asarray(event["state_before"], dtype=np.float64)
            after = np.asarray(event["state_after"], dtype=np.float64)
            if (before.shape != (4,) or after.shape != (4,)
                    or not np.isfinite(before).all() or not np.isfinite(after).all()
                    or (last is not None and np.max(np.abs(before-last)) > 1e-5)):
                raise ValueError("F8R2_INVALID_PUBLIC_PHYSICS")
            last = after
            if t > 1:
                key = np.asarray([1.0, *np.tanh(before/SCALES),
                                  float(act == 0), float(act == 1)])
                key /= np.linalg.norm(key)
                features.append(key)
                targets.append(np.tanh((after-before)/SCALES))
        xworld.append(np.stack(features))
        yworld.append(np.stack(targets))
    if len(observed_refs) != 12288:
        raise ValueError("F8R2_ORIGINAL_FIRSTUSE_COUNT_DRIFT")
    return np.stack(xworld), np.stack(yworld), trained


def features(x: np.ndarray, arm: str):
    if arm == "state5":
        return x[..., :5]
    if arm == "action3":
        return x[..., [0, 5, 6]]
    if arm == "linear7":
        return x
    if arm == "quadratic35":
        squared = np.stack([x[..., i]*x[..., j]
                            for i in range(7) for j in range(i, 7)], -1)
        return np.concatenate([x, squared], -1)
    raise ValueError("F8R2_UNREGISTERED_ARM")


def fit(x, y, lam):
    if (x.ndim != 3 or y.shape != (*x.shape[:2], 4)
            or x.shape[1] < 1 or lam not in GRID
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("F8R2_INVALID_FIT")
    worlds, steps, width = x.shape
    a = x.reshape(-1, width) / math.sqrt(steps)
    b = y.reshape(-1, 4) / math.sqrt(steps)
    gram = a.T @ a + lam*np.eye(width)
    rhs = a.T @ b
    if lam == 0:
        return np.linalg.lstsq(gram, rhs, rcond=1e-12)[0]
    return np.linalg.solve(gram, rhs)


def errors(x, y, weights):
    return np.mean((x @ weights - y)**2, axis=(1, 2))


def evaluate(f7_archive: Path, trained_archive: Path):
    t0 = time.perf_counter()
    x, y, prior = read_spent(f7_archive, trained_archive)
    if x.shape != (128, 95, 7) or y.shape != (128, 95, 4):
        raise ValueError("F8R2_SOURCE_SHAPE_DRIFT")
    result = {
        "schema": "ttt-hybrid.f8r2-linear-retrospective-result.v1",
        "classification": "ALREADY_SPENT_F7_FIT_DEVELOPMENT_ONLY",
        "freeze_git_blob": FREEZE_BLOB,
        "f7_artifact_sha256": F7_ZIP,
        "f7_training_artifact_sha256": F7_TRAIN_ZIP,
        "f7_capture_sha256": F7_CAPTURE,
        "train_worlds": 80, "fitdev_worlds": 16, "spent_holdout_worlds": 32,
        "train_events": 7600, "spent_holdout_events": 3040,
        "prior_f7_ssm_mse": [{"seed": z["model_seed"], "mse": z["FIT_holdout_delta_MSE"]}
            for z in prior["results"] if z["arm"] == "original_256state_SSM"],
        "strongest_prior_f7_ssm_mse": min(SSM_MSE),
        "arms": {}, "scientific_pass": False,
        "new_effects": 0, "firstuse_replayed": False, "ood_opened": False,
    }
    for arm in ARMS:
        a = features(x, arm)
        fits = []
        for lam in GRID:
            w = fit(a[:80], y[:80], lam)
            fits.append({"lambda": lam, "fitdev_mse": float(errors(a[80:96], y[80:96], w).mean())})
        chosen = min(fits, key=lambda z: (z["fitdev_mse"], GRID.index(z["lambda"])))
        w = fit(a[:96], y[:96], chosen["lambda"])
        result["arms"][arm] = {
            "fitdev_grid": fits, "selected_lambda": chosen["lambda"],
            "weights": int(w.size),
            "coefficients": w.tolist(),
            "coeff_sha256": sha256(np.asarray(w, dtype="<f8").tobytes()),
            "old_f7_holdout_mse": float(errors(a[96:], y[96:], w).mean()),
            "per_world_mse": errors(a[96:], y[96:], w).tolist(),
        }
    # Train-only destroyed action credit; keep actual physical state and effect.
    rng = np.random.default_rng(SHUFFLE_SEED)
    shuffled = x[:96].copy()
    values = shuffled[:, :, [5, 6]].reshape(-1, 2)
    shuffled[:, :, 5:7] = values[rng.permutation(len(values))].reshape(96, 95, 2)
    shuffled /= np.linalg.norm(shuffled, axis=-1, keepdims=True)
    w_bad = fit(shuffled, y[:96], result["arms"]["linear7"]["selected_lambda"])
    bad_mse = float(errors(x[96:], y[96:], w_bad).mean())
    good_mse = result["arms"]["linear7"]["old_f7_holdout_mse"]
    result["negative_shuffled_actions"] = {
        "fixed_seed": SHUFFLE_SEED, "same_linear7_lambda": result["arms"]["linear7"]["selected_lambda"],
        "heldout_mse": bad_mse, "ratio_vs_correct": bad_mse/good_mse,
        "unchosen_effects_observed": False, "true_counterfactual_identified": False,
    }
    best = min(result["arms"][z]["old_f7_holdout_mse"] for z in ("linear7", "quadratic35"))
    result["retrospective_baseline_beats_ssm_by_10pct"] = best <= .9*min(SSM_MSE)
    result["action_predictive_10pct"] = good_mse <= .9*result["arms"]["state5"]["old_f7_holdout_mse"]
    result["zero_baseline_mse"] = float((y[96:]**2).mean())
    result["local_elapsed_seconds"] = time.perf_counter()-t0
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-f7-fit-zip", type=Path, required=True)
    p.add_argument("--original-f7-trained-zip", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = evaluate(args.original_f7_fit_zip, args.original_f7_trained_zip)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({"linear7_mse": result["arms"]["linear7"]["old_f7_holdout_mse"],
                      "quadratic35_mse": result["arms"]["quadratic35"]["old_f7_holdout_mse"],
                      "strongest_ssm": result["strongest_prior_f7_ssm_mse"],
                      "retrospective_only": True, "scientific_pass": False}))


if __name__ == "__main__":
    main()
