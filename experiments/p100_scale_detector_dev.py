"""P100: fresh-source normalized causal innovation detector (diagnostic only).

Pre-source protocol frozen at 0d85bf542c425c02e40a1035123a94f4247a527c.
Prediction always remains original P86 W256. Detection is not causal recovery.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional as F

from experiments import p86_portable_causal_preflight as p86
from experiments import p97_recurrent_context_dev as p97
from experiments import p99_event_checkpoint_dev as p99

FREEZE = "0d85bf542c425c02e40a1035123a94f4247a527c"
FREEZE_BLOB = "9cd86c9bd3d16b223d86b976754d139246d96818"
D = 16
SEEDS = {"A_FIT": 1000111, "A_CAL": 1000113,
         "B_FIT": 1000211, "B_CAL": 1000213}
RESERVED = {"OOD": 1000117, "NEGATIVE": 1000119}
N = {"FIT": 128, "CAL": 64}
THRESHOLDS = (1.5, 2., 3.)
PERSIST = (2, 3)
BLOBS = {
    "model.py": "5d9ea57b827aa1796548447868fa5ee14c251b49",
    "experiments/p86_portable_causal_preflight.py":
        "008492665ee9090db5791963f2e2336ea283ee62",
    "experiments/p97_recurrent_context_dev.py":
        "74030c3d7d2f5f51781c740493be21d3c769792d",
    "experiments/p99_event_checkpoint_dev.py":
        "23279971d2f7531716c349991d6fd56095656c9e",
}


def source(key: str) -> dict:
    if key not in SEEDS:
        raise ValueError("P100_PROTECTED_SOURCE_DENIED")
    fam, split = key.split("_")
    seed, n = SEEDS[key], N[split]
    switches = set(sorted(range(n), key=lambda w: p86.digest_int(
        seed, w, "p100-regime", 0, 0, 2**32))[:n // 2])
    histories, queries, effects, flags = [], [], [], []
    for w in range(n):
        u0, v0 = p86.basis(seed, w, "p100-u0"), p86.basis(seed, w, "p100-v0")
        u1, v1 = p86.basis(seed, w, "p100-u1"), p86.basis(seed, w, "p100-v1")
        h, q, y = [], [], []
        for t in range(72):
            a = [p86.digest_int(seed, w, "p100-action", t, j, 193) - 96
                 for j in range(D)]
            if fam == "B":
                a = [z // (2 ** (j % 4)) for j, z in enumerate(a)]
            u, v = (u1, v1) if (w in switches and 24 <= t < 48) else (u0, v0)
            h.append([z / 128 for z in a] + p86.causal_effect(a, u, v))
        for i in range(3):
            a = [p86.digest_int(seed, w, "p100-probe", i, j, 193) - 96
                 for j in range(D)]
            if fam == "B":
                a = [z // (2 ** (j % 4)) for j, z in enumerate(a)]
            u, v = (u1, v1) if (w in switches and i == 1) else (u0, v0)
            q.append([z / 128 for z in a])
            y.append(p86.causal_effect(a, u, v))
        histories.append(h)
        queries.append(q)
        effects.append(y)
        flags.append(w in switches)
    result = {
        "history": torch.tensor(histories, dtype=torch.float32),
        "probes": torch.tensor(queries, dtype=torch.float32),
        "targets": torch.tensor(effects, dtype=torch.float32),
        "changed_world": torch.tensor(flags, dtype=torch.bool),
    }
    if not p97.audit(result):
        raise ValueError("P100_SOURCE_AUDIT_FAILED")
    return result


@dataclass(frozen=True)
class DetectorState:
    W: torch.Tensor                 # 256 float32 factual weights/world
    energy: torch.Tensor            # 1 float32 reference effect energy/world
    count: torch.Tensor             # int64 confirmed observation count
    consecutive: torch.Tensor       # int64 above-threshold consecutive evidence
    cooldown: torch.Tensor          # int64 refractory steps

    def snapshot(self) -> "DetectorState":
        return DetectorState(*(getattr(self, k).detach().clone()
                               for k in self.__dataclass_fields__))


def initial(history: torch.Tensor) -> DetectorState:
    if history.ndim != 3 or history.shape[-1] != 2 * D or history.dtype != torch.float32:
        raise ValueError("P100_INVALID_INITIAL")
    b = len(history)
    z = torch.zeros(b, dtype=torch.int64, device=history.device)
    return DetectorState(history.new_zeros((b, D, D)), history.new_zeros(b),
                         z.clone(), z.clone(), z.clone())


def stream(history: torch.Tensor, threshold: float, persistence: int, *,
           adaptive: bool = True,
           confirmed: torch.Tensor | None = None,
           state: DetectorState | None = None):
    """One factual W256 write per confirmed observation; alarms are event edges."""
    if (not isinstance(history, torch.Tensor) or history.ndim != 3
        or history.shape[-1] != 2 * D or history.dtype != torch.float32
        or threshold not in THRESHOLDS or persistence not in PERSIST):
        raise ValueError("P100_INVALID_INPUT")
    b, steps, _ = history.shape
    if confirmed is None:
        confirmed = torch.ones((b, steps), dtype=torch.bool, device=history.device)
    if (confirmed.shape != (b, steps) or confirmed.dtype != torch.bool
        or confirmed.device != history.device):
        raise ValueError("P100_INVALID_CONFIRMATION")
    if state is None:
        state = initial(history)
    if (not isinstance(state, DetectorState)
        or state.W.shape != (b, D, D) or state.energy.shape != (b,)
        or any(getattr(state, k).shape != (b,)
               for k in ("count", "consecutive", "cooldown"))
        or state.W.dtype != torch.float32 or state.energy.dtype != torch.float32
        or any(getattr(state, k).dtype != torch.int64
               for k in ("count", "consecutive", "cooldown"))
        or any(getattr(state, k).device != history.device
               for k in state.__dataclass_fields__)
        or not bool(torch.isfinite(state.W).all())
        or not bool(torch.isfinite(state.energy).all())
        or not bool((state.count >= 0).all())):
        raise ValueError("P100_INVALID_STATE")
    x, y = history[:, :, :D], history[:, :, D:]
    if not bool(torch.isfinite(x).all()):
        raise ValueError("P100_NONFINITE_ACTION")
    safe_y = torch.where(confirmed.unsqueeze(-1), y, torch.zeros_like(y))
    if not bool(torch.isfinite(safe_y).all()):
        raise ValueError("P100_NONFINITE_CONFIRMED_EFFECT")
    W, energy = state.W, state.energy
    count, consecutive, cooldown = state.count, state.consecutive, state.cooldown
    alarms, predictions = [], []
    keys = F.normalize(x, dim=-1)
    for t in range(steps):
        ok = confirmed[:, t]
        k = keys[:, t]
        pred = torch.bmm(W, k.unsqueeze(-1)).squeeze(-1)
        predictions.append(pred)
        innovation = safe_y[:, t] - pred
        err_sq = innovation.square().mean(-1)
        effect_sq = safe_y[:, t].square().mean(-1)
        # Both arms use a dimensionless ratio. Static arm freezes its observed
        # pre-change calibration after 16 facts; adaptive continues its EMA.
        ratio = err_sq / energy.clamp_min(1e-6)
        high = (ratio > threshold) & (count >= 16)
        tentative = torch.where(high, consecutive + 1, torch.zeros_like(consecutive))
        firing = ok & (cooldown == 0) & (tentative >= persistence)
        alarms.append(firing)
        new_consecutive = torch.where(firing, torch.zeros_like(tentative), tentative)
        new_cooldown = torch.where(firing, torch.full_like(cooldown, 8),
                                   torch.clamp(cooldown - 1, min=0))
        update = .99 * W + innovation.unsqueeze(-1) * k.unsqueeze(1)
        W = torch.where(ok[:, None, None], update, W)
        # Initialize calibration using only previously confirmed events.
        new_energy = .90 * energy + .10 * effect_sq
        scale_mask = ok & (torch.ones_like(ok) if adaptive else count < 16)
        energy = torch.where(scale_mask, new_energy, energy)
        count = count + ok.to(torch.int64)
        consecutive = torch.where(ok, new_consecutive, consecutive)
        cooldown = torch.where(ok, new_cooldown, cooldown)
    final = DetectorState(W, energy, count, consecutive, cooldown)
    return final, torch.stack(alarms, dim=1), torch.stack(predictions, dim=1)


def fixed_w256(history: torch.Tensor):
    """Exactly the P86 W256 recurrence; no detector or extra memory."""
    W = history.new_zeros((len(history), D, D))
    keys = F.normalize(history[:, :, :D], dim=-1)
    for t in range(history.shape[1]):
        k = keys[:, t]
        pred = torch.bmm(W, k.unsqueeze(-1)).squeeze(-1)
        error = history[:, t, D:] - pred
        W = .99 * W + error.unsqueeze(-1) * k.unsqueeze(1)
    return W


def detection(alarms: torch.Tensor, world: dict) -> dict:
    changed = world["changed_world"]
    stable = ~changed
    fp = alarms[stable, 16:].any(-1)
    at_B = alarms[changed, 24:36].any(-1)
    at_A = alarms[changed, 48:60].any(-1)
    return {
        "stable_false_alarm_count": int(fp.sum()),
        "stable_false_alarm_rate": float(fp.float().mean()),
        "middle_entry_count": int(at_B.sum()),
        "middle_entry_rate": float(at_B.float().mean()),
        "return_entry_count": int(at_A.sum()),
        "return_entry_rate": float(at_A.float().mean()),
        "total_alarm_events": int(alarms.sum()),
    }


def advantage(row: dict) -> float:
    return (row["middle_entry_rate"] + row["return_entry_rate"]
            - 2 * row["stable_false_alarm_rate"])


def choose(fit: dict, adaptive: bool):
    grid = []
    with torch.no_grad():
        for threshold in THRESHOLDS:
            for persist in PERSIST:
                _, alarms, _ = stream(
                    fit["history"], threshold, persist, adaptive=adaptive)
                m = detection(alarms, fit)
                grid.append({
                    "threshold": threshold, "persistence": persist,
                    "fit_score": advantage(m), "FIT": m,
                })
    selected = max(grid, key=lambda r: r["fit_score"])
    return (selected["threshold"], selected["persistence"]), grid


def negative(model: tuple, world: dict):
    threshold, persistence = model
    w = {k: v[:8].clone() for k, v in world.items()}
    h = w["history"]
    with torch.no_grad():
        whole, alarms, pred = stream(h, threshold, persistence)
        part = initial(h)
        parts = []
        a = []
        for index in range(3):
            part, al, predictions = stream(
                h[:, index*24:(index+1)*24], threshold, persistence,
                state=part.snapshot())
            a.append(al)
            parts.append(predictions)
        cold = (all(torch.equal(getattr(whole, k), getattr(part, k))
                    for k in DetectorState.__dataclass_fields__)
                and torch.equal(torch.cat(a, 1), alarms)
                and torch.equal(torch.cat(parts, 1), pred))
        mask = torch.ones((8, 72), dtype=torch.bool)
        mask[:, 1::3] = False
        altered = h.clone()
        altered[:, 1::3, D:] = float("nan")
        correct, alarms_true, pred_true = stream(
            h, threshold, persistence, confirmed=mask)
        tainted, alarms_fake, pred_fake = stream(
            altered, threshold, persistence, confirmed=mask)
        inert = (all(torch.equal(getattr(correct, k), getattr(tainted, k))
                     for k in DetectorState.__dataclass_fields__)
                 and torch.equal(alarms_true, alarms_fake)
                 and torch.equal(pred_true, pred_fake))
    return {"bitwise_cold_resume": bool(cold), "unconfirmed_nan_inert": bool(inert)}


def median_ms(fn):
    with torch.no_grad():
        for _ in range(2):
            fn()
        data = []
        for _ in range(7):
            t0 = time.perf_counter()
            fn()
            data.append((time.perf_counter() - t0)*1000)
    return statistics.median(data)


def run() -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path(__file__).resolve().parents[1]
    frozen_file = root / "experiments/p100_scale_detector_freeze.json"
    if p86.blob(frozen_file) != FREEZE_BLOB:
        raise ValueError("P100_FREEZE_BLOB_DRIFT")
    f = json.loads(frozen_file.read_text(encoding="utf-8"))
    if (f["schema"] != "ttt-hybrid.p100-scale-normalized-causal-change-detector-freeze.v1"
        or f["source"]["seeds"] != {**SEEDS, "OOD_RESERVED":1000117,
                                    "NEGATIVE_RESERVED":1000119}
        or f["method"]["ratio_grid"] != list(THRESHOLDS)
        or f["method"]["persistence_grid"] != list(PERSIST)
        or f["method"]["warmup"] != 16 or f["method"]["refractory"] != 8):
        raise ValueError("P100_PROTOCOL_DRIFT")
    for path, blob in BLOBS.items():
        if p86.blob(root / path) != blob:
            raise ValueError("P100_ORIGINAL_BLOB_DRIFT")
    worlds = {key: source(key) for key in SEEDS}
    hashes = {key: p97.fingerprint(w) for key, w in worlds.items()}
    if len(set(hashes.values())) != 4:
        raise ValueError("P100_SOURCE_COLLISION")
    out = {
        "schema": "ttt-hybrid.p100-scale-change-detector-dev-result.v1",
        "freeze_commit": FREEZE, "freeze_blob": FREEZE_BLOB,
        "source_sha256": hashes, "status": "NOT_RUN",
        "science_pass": False, "detector_only": True,
        "causal_recovery_proven": False, "neural_training": False,
        "ood_generated": False, "negative_generated": False,
        "external_real_source_used": False, "original_models_modified": False,
        "persistent_float_state": 257, "families": {},
    }
    for fam in ("A", "B"):
        fit, cal = worlds[fam+"_FIT"], worlds[fam+"_CAL"]
        norm_setting, norm_grid = choose(fit, adaptive=True)
        raw_setting, raw_grid = choose(fit, adaptive=False)
        p99_setting, p99_grid = p99.select_fit(fit)
        with torch.no_grad():
            model, normalized, pred = stream(
                cal["history"], *norm_setting, adaptive=True)
            reference, static, _ = stream(
                cal["history"], *raw_setting, adaptive=False)
            prior_model = p99.EventCheckpointMemory(*p99_setting)
            _, _, prior_route = prior_model.episode(
                cal["history"], cal["probes"], trace=True)
            baseline = fixed_w256(cal["history"])
            if (not torch.equal(baseline, model.W)
                or not torch.equal(baseline, reference.W)
                or model.W.numel() != len(cal["history"])*256):
                raise ValueError("P100_CAUSAL_BASELINE_PARITY")
            norm_stats = detection(normalized, cal)
            raw_stats = detection(static, cal)
            historical = p99.diagnostic(prior_route, cal)
            negatives = negative(norm_setting, cal)
            wrong = cal["history"].clone()
            wrong[:, :, D:] = wrong[:, :, D:].roll(13, dims=1)
            _, _, wrong_pred = stream(wrong, *norm_setting, adaptive=True)
            # Wrong history must predict TRUE unshifted factual effects.
            clean_error = (pred[:, 16:] - cal["history"][:, 16:, D:]).square().mean()
            wrong_error = (wrong_pred[:, 16:] - cal["history"][:, 16:, D:]).square().mean()
            wrong_ratio = float(wrong_error / clean_error.clamp_min(1e-12))
            negatives["wrong_credit_degrades"] = wrong_ratio >= 1.05
            if not all(negatives.values()):
                raise ValueError("P100_REQUIRED_NEGATIVE_FAILED")
        sample = cal["history"][:8]
        cost = {
            "W256_ms": median_ms(lambda: fixed_w256(sample)),
            "detector_ms": median_ms(lambda: stream(sample, *norm_setting)),
        }
        cost["ratio"] = cost["detector_ms"] / cost["W256_ms"]
        performance = {
            "stable_false": norm_stats["stable_false_alarm_rate"] <= .10,
            "B_entry": norm_stats["middle_entry_rate"] >= .75,
            "A_return": norm_stats["return_entry_rate"] >= .75,
            "better_than_static": (
                (norm_stats["middle_entry_rate"] + norm_stats["return_entry_rate"]
                 - raw_stats["middle_entry_rate"] - raw_stats["return_entry_rate"]) / 2
                >= .10),
            "whole_episode_cost": cost["ratio"] <= 2.5,
        }
        out["families"][fam] = {
            "FIT_selected_normalized": {
                "threshold": norm_setting[0], "persistence": norm_setting[1]},
            "FIT_selected_static": {
                "threshold": raw_setting[0], "persistence": raw_setting[1]},
            "FIT_selected_p99": {
                "shock": p99_setting[0], "persistence": p99_setting[1]},
            "FIT_grid_normalized": norm_grid,
            "FIT_grid_static": raw_grid,
            "FIT_grid_p99": p99_grid,
            "CAL_normalized": norm_stats,
            "CAL_static": raw_stats,
            "CAL_previous_P99": historical,
            "wrong_credit_ratio": wrong_ratio,
            "negative_controls": negatives,
            "performance_gates": performance,
            "CPU": cost,
            "family_pass": all(performance.values()),
            "stable_world_count": int((~cal["changed_world"]).sum()),
            "changed_world_count": int(cal["changed_world"].sum()),
        }
    out["status"] = ("DEV_DETECTOR_ONLY_PASS_NEEDS_RECOVERY_AND_EXTERNAL_FIRST_USE"
                     if all(z["family_pass"] for z in out["families"].values())
                     else "DEV_FAIL_CHANGE_DETECTION_OR_COST")
    return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--receipt", required=True, type=Path)
    args = p.parse_args()
    result = run()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "science_pass": False}))


if __name__ == "__main__":
    main()
