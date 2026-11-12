"""P78-MR R5: retrospective FIT training of a 3-weight residual controller.

Only previously consumed owner F3 FIT/CAL and frozen F4 SSM checkpoint.
No simulator, new source, paid inference, OOD, owner session or model promotion.
All scientific decisions remain retrospective DEV, never SCIENTIFIC_PASS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import zipfile

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from p78mr_gain_posthoc_audit import read_locked
from p78mr_r3_f3_transfer import (
    F3_ROOT, F3_ZIP, FIT_SHA, CAL_SHA, factual_event, read_prior_cal,
)
from p78mr_r4_ssm_residual import frozen_ssm

FREEZE = HERE / "p78mr_r5_consumed_fit_training_freeze.json"
FREEZE_BLOB = "51429265c6ec40d757709bac7b6e9d53f78c9b8d"
SEEDS = (979501, 979503, 979507)
GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
SAMPLE_SEED = 979599
STEPS = 240
BATCH = 16
CHECK = 40
ORIGINAL_CAL = 1.2177126073709132e-5


def locked() -> dict:
    d = json.loads(read_locked(FREEZE, FREEZE_BLOB))
    if (d["schema"] != "ttt-hybrid.p78mr-r5-consumed-F3-FIT-trainable-residual-v1"
            or d["source"]["zip_sha256"] != F3_ZIP
            or d["source"]["fit_capture_sha256"] != FIT_SHA
            or d["source"]["cal_capture_sha256"] != CAL_SHA
            or d["base_model"]["cal_delta_mse"] != ORIGINAL_CAL
            or d["frozen_fit"]["seeds"] != list(SEEDS)
            or d["frozen_fit"]["sampler"] !=
                "torch CPU Generator seed=979599; randint(96) minibatch 16 for 240 updates; same indices all arms/seeds"
            or d["frozen_fit"]["updates"] != STEPS
            or d["frozen_fit"]["batch"] != BATCH
            or d["frozen_fit"]["eval_interval"] != CHECK
            or d["controls"]["fixed_grid"] != list(GRID)
            or d["authority"]["new_science_source"] is not False
            or d["authority"]["owner_first_use"] is not False
            or d["authority"]["protected_ood"] is not False):
        raise ValueError("P78MR_R5_FROZEN_PROTOCOL_DRIFT")
    return d


def read_consumed_fit(archive: Path) -> list[dict]:
    """No generator call; check exact prior owner ZIP and 128-world FIT cohort."""
    if (not archive.is_file() or
            hashlib.sha256(archive.read_bytes()).hexdigest() != F3_ZIP):
        raise ValueError("P78MR_R5_PRIOR_F3_ARCHIVE_DRIFT")
    with zipfile.ZipFile(archive) as z:
        raw_fit = z.read(F3_ROOT + "fit.json")
        raw_cal = z.read(F3_ROOT + "cal.json")
        owner = json.loads(z.read(F3_ROOT + "RESULT.json"))
    if (hashlib.sha256(raw_fit).hexdigest() != FIT_SHA
            or hashlib.sha256(raw_cal).hexdigest() != CAL_SHA
            or owner.get("status") != "SOURCE_COLLECTED_INDEPENDENT_WITNESS_PENDING"
            or owner.get("executed_actions") != 4447
            or owner.get("ood_created") is not False
            or owner.get("science_pass") is not False
            or owner.get("owner", {}).get("run_id") != "43073165842"):
        raise ValueError("P78MR_R5_SPENT_SOURCE_CUSTODY_DRIFT")
    capture = json.loads(raw_fit)
    if (capture.get("schema") != "ttt-hybrid.p80mr.cartpole-do-capture.v1"
            or capture.get("source_id") != "p85mr-f3-fit-43073165842"
            or capture.get("producer_id") != "ttt-hybrid-f3-owner"
            or capture.get("custody_ref") != "ghrun:43073165842"
            or capture.get("upstream_repo") != "CausalAILab/Causal-Gymnasium"
            or capture.get("upstream_commit") !=
                "c94e9a4cd3641588b6b8507e71ccee5c27032bdf"
            or len(capture.get("worlds", ())) != 128):
        raise ValueError("P78MR_R5_UNQUALIFIED_FIT_COHORT")
    seen = set()
    count = 0
    for wi, world in enumerate(capture["worlds"]):
        records = world["transitions"]
        if (len(records) < 2 or world["probe_event_ids"] !=
                list(range(2, len(records) + 1))):
            raise ValueError("P78MR_R5_FIT_PROBES_DRIFT")
        last = None
        for i, step in enumerate(records, 1):
            receipt = f"ghrun:43073165842:fit:{wi}:{i}"
            if (step["event_id"] != i or step["mode"] != "do"
                    or step["receipt_ref"] != receipt
                    or receipt in seen or step["executed_action"] not in (0, 1)):
                raise ValueError("P78MR_R5_FIT_EVENT_CUSTODY_DRIFT")
            a, b = step["state_before"], step["state_after"]
            if (len(a) != 4 or len(b) != 4
                    or not all(type(x) in (int, float) and math.isfinite(x)
                               for x in [*a, *b, step["reward"]])
                    or (last is not None and any(abs(x-y) > 1e-5
                                                 for x, y in zip(last, a)))
                    or type(step["terminated"]) is not bool
                    or type(step["truncated"]) is not bool
                    or (step["terminated"] or step["truncated"]) and i != len(records)):
                raise ValueError("P78MR_R5_INVALID_FIT_FACTS")
            last = b
            seen.add(receipt)
            count += 1
    if count != 2923 or len(seen) != 2923:
        raise ValueError("P78MR_R5_FIT_EVENT_COUNT_DRIFT")
    return capture["worlds"]


def cache_original_ssm(base, worlds: list[dict]) -> list[tuple]:
    """The original trained SSM always sees y[t] AFTER making its p[t]."""
    episodes = []
    with torch.no_grad():
        for world in worlds:
            state = torch.zeros((1, 256), dtype=torch.float32)
            a_series, residual_series, base_series = [], [], []
            for item in world["transitions"]:
                a, y = factual_event(item)
                a = a[None]
                y = y[None]
                prior = base.read(state, a)
                state = base.encode({"action": a[:, None], "effect": y[:, None]},
                                    carry=state)
                a_series.append(a[0])
                residual_series.append((y - prior)[0])
                base_series.append(prior[0])
            episodes.append((torch.stack(a_series), torch.stack(residual_series),
                             torch.stack(base_series)))
    return episodes


def pack(episodes: list[tuple]) -> tuple:
    if not episodes or any(not isinstance(e, tuple) or len(e) != 3
                           or e[0].ndim != 2 or e[0].shape[1] != 16
                           or e[0].shape != e[1].shape
                           or e[1].shape != e[2].shape
                           or len(e[0]) < 2 for e in episodes):
        raise ValueError("P78MR_R5_BAD_EPISODE")
    total, long = len(episodes), max(len(e[0]) for e in episodes)
    a = torch.zeros((total, long, 16), dtype=torch.float32)
    residual = torch.zeros_like(a)
    valid = torch.zeros((total, long), dtype=torch.bool)
    for i, (actions, errors, _) in enumerate(episodes):
        length = len(actions)
        if (not bool(torch.isfinite(actions).all())
                or not bool(torch.isfinite(errors).all())
                or not bool(((actions.norm(dim=-1)-1).abs()<1e-4).all())):
            raise ValueError("P78MR_R5_BAD_FACTS_OR_UNITS")
        a[i, :length] = actions
        residual[i, :length] = errors
        valid[i, :length] = True
    return a, residual, valid


def residual_loss(theta: torch.Tensor | None, batch: tuple,
                  *, fixed_gain: float | None = None) -> torch.Tensor:
    """Differentiable factual writes. Per-world mean then equal-world mean."""
    actions, targets, valid = batch
    if (actions.ndim != 3 or actions.shape[-1] != 16
            or targets.shape != actions.shape or valid.shape != actions.shape[:2]
            or actions.dtype != torch.float32 or targets.dtype != torch.float32
            or valid.dtype != torch.bool or
            not bool((valid[:, 0]).all()) or
            not bool((valid.sum(dim=1) >= 2).all())
            or not bool(torch.isfinite(actions).all())
            or not bool(torch.isfinite(targets).all())):
        raise ValueError("P78MR_R5_BAD_BATCH")
    if (fixed_gain is None and
            (theta is None or theta.shape != (3,) or theta.dtype != torch.float32
             or not bool(torch.isfinite(theta).all()))):
        raise ValueError("P78MR_R5_MISSING_OR_INVALID_THETA")
    if (fixed_gain is not None and
            (theta is not None or fixed_gain not in (0., *GRID))):
        raise ValueError("P78MR_R5_INVALID_FIXED_GAIN")
    b, steps, _ = actions.shape
    w = actions.new_zeros((b, 16, 16))
    per_world = actions.new_zeros((b,))
    for t in range(steps):
        a = actions[:, t]
        target = targets[:, t]
        pred = torch.bmm(w, a.unsqueeze(-1)).squeeze(-1)
        delta = target - pred
        if fixed_gain is not None:
            gain = a.new_full((b,), fixed_gain)
        else:
            features = torch.stack((
                torch.ones_like(delta[:, 0]),
                torch.log(delta.square().mean(dim=-1) + 1e-6),
                torch.log(pred.square().mean(dim=-1) + 1e-6),
            ), dim=-1)
            gain = torch.sigmoid(features @ theta)
        w = w + (gain * valid[:, t])[:, None, None] * (
            delta[:, :, None] * a[:, None, :])
        if t > 0:
            per_world = per_world + (
                pred[:, :4] - target[:, :4]).square().mean(dim=-1) * valid[:, t]
    loss = (per_world / valid[:, 1:].sum(dim=1)).mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError("P78MR_R5_NONFINITE_LOSS")
    return loss


def evaluate(episodes: list[tuple], *, theta=None, eta=None) -> float:
    with torch.no_grad():
        return float(residual_loss(theta, pack(episodes), fixed_gain=eta))


def fit_only(train: list[tuple], fitdev: list[tuple]) -> dict:
    if len(train) != 96 or len(fitdev) != 32:
        raise ValueError("P78MR_R5_FIT_SPLIT_DRIFT")
    fixed = {str(rate): evaluate(fitdev, eta=rate) for rate in GRID}
    selected_rate = min(GRID, key=lambda x: (fixed[str(x)], GRID.index(x)))
    indices = torch.randint(0, 96, (STEPS, BATCH),
        generator=torch.Generator(device="cpu").manual_seed(SAMPLE_SEED))
    samples_sha = hashlib.sha256(indices.numpy().tobytes()).hexdigest()
    trained = []
    for seed in SEEDS:
        rng = torch.Generator(device="cpu").manual_seed(seed)
        theta = torch.nn.Parameter((torch.rand(3, generator=rng) - .5) * .1)
        optimizer = torch.optim.AdamW([theta], lr=.01, weight_decay=.01)
        best, best_step, best_theta = float("inf"), 0, None
        trace = []
        start = time.perf_counter()
        for step in range(1, STEPS + 1):
            batch = [train[int(i)] for i in indices[step-1]]
            optimizer.zero_grad(set_to_none=True)
            loss = residual_loss(theta, pack(batch))
            loss.backward()
            if theta.grad is None or not bool(torch.isfinite(theta.grad).all()):
                raise ValueError("P78MR_R5_NONFINITE_GRADIENT")
            gn = float(torch.nn.utils.clip_grad_norm_([theta], 1.))
            optimizer.step()
            if step % CHECK == 0:
                mse = evaluate(fitdev, theta=theta)
                trace.append({"step": step, "fitdev_mse": mse,
                              "train_mse": float(loss.detach()),
                              "theta": theta.detach().tolist(),
                              "gradient_preclip_l2": gn})
                if mse < best:
                    best, best_step, best_theta = mse, step, theta.detach().clone()
        trained.append({
            "seed": seed, "selected_step": best_step,
            "selected_theta": best_theta.tolist(),
            "fitdev_mse": best, "trace": trace,
            "train_wall_seconds_local": time.perf_counter() - start,
        })
    return {"selected_eta": selected_rate, "fixed_fitdev_mses": fixed,
            "sampler_sha256": samples_sha, "models": trained}


def retrospective(f3_existing: Path, f4_existing: Path) -> dict:
    locked()
    if torch.cuda.is_initialized():
        raise PermissionError("P78MR_R5_CPU_ONLY")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    base = frozen_ssm(f4_existing)
    assert not any(p.requires_grad for p in base.parameters())
    fit = cache_original_ssm(base, read_consumed_fit(f3_existing))
    if len(fit) != 128:
        raise ValueError("P78MR_R5_FIT_COUNT_DRIFT")
    selected = fit_only(fit[:96], fit[96:])
    # This is the first CAL access in R5; it is historically consumed.
    cal = cache_original_ssm(base, read_prior_cal(f3_existing))
    base_mse = evaluate(cal, eta=0.)
    if abs(base_mse - ORIGINAL_CAL) > 1e-8:
        raise ValueError("P78MR_R5_ORIGINAL_TRAINED_SSM_MISMATCH")
    fixed_mse = evaluate(cal, eta=selected["selected_eta"])
    old_fixed = evaluate(cal, eta=0.4)
    for model in selected["models"]:
        theta = torch.tensor(model["selected_theta"], dtype=torch.float32)
        mse = evaluate(cal, theta=theta)
        model["cal_mse"] = mse
        model["ratio_vs_original_ssm"] = mse / base_mse
        model["ratio_vs_fit_selected_fixed"] = mse / fixed_mse
        model["frozen_primary_dev_pass"] = (
            mse <= .95 * base_mse and mse <= .95 * fixed_mse)
    return {
        "schema": "ttt-hybrid.p78mr-r5-consumed-fit-training-result.v1",
        "classification": "RETROSPECTIVE_DEV_ALREADY_CONSUMED_F3_SOURCE",
        "freeze_blob": FREEZE_BLOB,
        "f3_archive_sha256": F3_ZIP, "fit_worlds": 128,
        "fit_train_worlds": 96, "fitdev_worlds": 32,
        "cal_worlds": 64, "cal_observed_events": 1524,
        "cal_scored_event_2plus": 1460,
        "original_ssm_cal_mse": base_mse,
        "fit_selected_fixed_eta": selected["selected_eta"],
        "fixed_fitdev_mses": selected["fixed_fitdev_mses"],
        "fit_selected_fixed_cal_mse": fixed_mse,
        "old_fixed_0p4_cal_mse": old_fixed,
        "sampler_sha256": selected["sampler_sha256"],
        "models": selected["models"],
        "all_three_frozen_primary_pass": all(
            m["frozen_primary_dev_pass"] for m in selected["models"]),
        "total_local_wall_seconds": time.perf_counter() - start,
        "new_external_actions": 0, "source_first_use_replayed": False,
        "ood_opened": False, "scientific_pass": False,
        "note": "F3 CAL was already used by F4/F5. Source retained in owner ZIP. "
                "Retrospective DEV; 512-float overlays not capacity/compute "
                "matched to new fully trained 512-state conventional models.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f3-existing-archive", type=Path, required=True)
    parser.add_argument("--f4-existing-controls", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result = retrospective(args.f3_existing_archive, args.f4_existing_controls)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True,
        allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"classification": result["classification"],
        "fit_selected_fixed_eta": result["fit_selected_fixed_eta"],
        "models": [{"seed": m["seed"], "mse": m["cal_mse"],
                    "pass": m["frozen_primary_dev_pass"]}
                   for m in result["models"]],
        "scientific_pass": False}, sort_keys=True))


if __name__ == "__main__":
    main()
