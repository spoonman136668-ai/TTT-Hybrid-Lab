"""P97: recurrent causal context memory (A->B->A), author-synthetic DEV only.

Frozen before source birth: 45373e14dc586aa2c3d518700c67dba546919378.
No protected source, deployment, or original Hybrid modification.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional as F

from experiments import p86_portable_causal_preflight as p86

D = 16
FREEZE = "45373e14dc586aa2c3d518700c67dba546919378"
SEEDS = {"A_FIT": 970111, "A_CAL": 970113,
         "B_FIT": 970211, "B_CAL": 970213}
RESERVED = {"OOD": 970117, "NEGATIVE": 970119}
N = {"FIT": 256, "CAL": 128}
BLOBS = {
    "model.py": "5d9ea57b827aa1796548447868fa5ee14c251b49",
    "experiments/p86_portable_causal_preflight.py":
        "008492665ee9090db5791963f2e2336ea283ee62",
    "native_p94_confirmed_causal_memory.py":
        "378167499362173023b02a44e1b9ffb7a1eeba42",
}
RATES = (.5, 1., 1.5)
DECAYS = (.95, .99, 1.)
SURPRISE = (2., 3.)
ALTERNATIVE = (.5, .7)


def source(key: str) -> dict:
    if key not in SEEDS:
        raise ValueError("P97_PROTECTED_SOURCE_DENIED")
    family, split = key.split("_")
    seed, n = SEEDS[key], N[split]
    changed = set(sorted(range(n), key=lambda w: p86.digest_int(
        seed, w, "regime", 0, 0, 2**32))[:n // 2])
    histories, probes, targets, flags = [], [], [], []
    for world in range(n):
        u0, v0 = p86.basis(seed, world, "u0"), p86.basis(seed, world, "v0")
        u1, v1 = p86.basis(seed, world, "u1"), p86.basis(seed, world, "v1")
        h, q, y = [], [], []
        for t in range(72):
            a = [p86.digest_int(seed, world, "action", t, j, 193) - 96
                 for j in range(D)]
            if family == "B":
                a = [z // (2 ** (j % 4)) for j, z in enumerate(a)]
            ub, vb = (u1, v1) if (world in changed and 24 <= t < 48) else (u0, v0)
            h.append([z / 128 for z in a] + p86.causal_effect(a, ub, vb))
        for phase in range(3):
            a = [p86.digest_int(seed, world, "probe", phase, j, 193) - 96
                 for j in range(D)]
            if family == "B":
                a = [z // (2 ** (j % 4)) for j, z in enumerate(a)]
            ub, vb = (u1, v1) if (world in changed and phase == 1) else (u0, v0)
            q.append([z / 128 for z in a])
            y.append(p86.causal_effect(a, ub, vb))
        histories.append(h)
        probes.append(q)
        targets.append(y)
        flags.append(world in changed)
    return {
        "history": torch.tensor(histories, dtype=torch.float32),
        "probes": torch.tensor(probes, dtype=torch.float32),
        "targets": torch.tensor(targets, dtype=torch.float32),
        "changed_world": torch.tensor(flags, dtype=torch.bool),
    }


def fingerprint(world: dict) -> str:
    return hashlib.sha256(b"".join(
        world[k].contiguous().cpu().numpy().tobytes()
        for k in ("history", "probes", "targets", "changed_world")
    )).hexdigest()


def audit(world: dict) -> bool:
    b = len(world["history"])
    return bool(
        world["history"].shape == (b, 72, 32)
        and world["probes"].shape == (b, 3, D)
        and world["targets"].shape == (b, 3, D)
        and world["changed_world"].shape == (b,)
        and int(world["changed_world"].sum()) == b // 2
        and all(bool(torch.isfinite(world[k]).all())
                for k in ("history", "probes", "targets"))
        and bool((world["probes"].norm(dim=-1) > 0).all())
    )


def mse(pred: torch.Tensor, world: dict) -> dict[str, float]:
    e = (pred - world["targets"]).square().mean(-1)
    changed = world["changed_world"]
    return {
        "all": float(e.mean()),
        "initial": float(e[:, 0].mean()),
        "mid_changed": float(e[changed, 1].mean()),
        "mid_stable": float(e[~changed, 1].mean()),
        "return_changed": float(e[changed, 2].mean()),
        "stable_final": float(e[~changed, 2].mean()),
    }


def baseline(history: torch.Tensor, probes: torch.Tensor,
             eta: float = 1., decay: float = .99) -> torch.Tensor:
    if history.ndim != 3 or history.shape[1:] != (72, 2 * D):
        raise ValueError("P97_BASELINE_HISTORY")
    b = len(history)
    W = history.new_zeros((b, D, D))
    k = F.normalize(history[:, :, :D], dim=-1)
    q = F.normalize(probes, dim=-1)
    predictions = []
    for t in range(72):
        error = history[:, t, D:] - torch.bmm(W, k[:, t, :, None]).squeeze(-1)
        W = decay * W + eta * error[:, :, None] * k[:, t, None, :]
        if t in (23, 47, 71):
            i = (t + 1) // 24 - 1
            predictions.append(torch.bmm(W, q[:, i, :, None]).squeeze(-1))
    return torch.stack(predictions, dim=1)


@dataclass(frozen=True)
class ContextState:
    weights: torch.Tensor  # [batch,2,16,16], 512 float32
    innovation: torch.Tensor  # [batch,2], 2 float32
    visits: torch.Tensor  # [batch,2], int64 control
    active: torch.Tensor  # [batch], int64 control
    shock: torch.Tensor  # [batch], int64 control
    better: torch.Tensor  # [batch], int64 control
    confirmed_count: torch.Tensor  # [batch], int64 control

    def snapshot(self) -> "ContextState":
        return ContextState(*(value.detach().clone() for value in (
            self.weights, self.innovation, self.visits, self.active,
            self.shock, self.better, self.confirmed_count)))


class RecurrentContextMemory:
    """Two causally fitted models, no oracle phase indicators, no offline weights."""

    def __init__(self, surprise: float, alternative: float):
        if surprise not in SURPRISE or alternative not in ALTERNATIVE:
            raise ValueError("P97_UNFROZEN_PARAMETERS")
        self.surprise = surprise
        self.alternative = alternative

    @staticmethod
    def validate_actions(x: torch.Tensor, ndim: int) -> None:
        if (not isinstance(x, torch.Tensor) or x.ndim != ndim
            or x.shape[-1] != D or x.dtype != torch.float32
            or not bool(torch.isfinite(x).all())):
            raise ValueError("P97_INVALID_ACTIONS")

    def initial(self, query: torch.Tensor) -> ContextState:
        self.validate_actions(query, 2)
        b = len(query)
        z = torch.zeros(b, dtype=torch.int64, device=query.device)
        return ContextState(
            query.new_zeros((b, 2, D, D)), query.new_zeros((b, 2)),
            torch.zeros((b, 2), dtype=torch.int64, device=query.device),
            z.clone(), z.clone(), z.clone(), z.clone())

    def check(self, state: ContextState, action: torch.Tensor) -> None:
        b = len(action)
        if (not isinstance(state, ContextState)
            or state.weights.shape != (b, 2, D, D)
            or state.innovation.shape != (b, 2)
            or state.visits.shape != (b, 2)
            or any(x.shape != (b,) for x in (
                state.active, state.shock, state.better, state.confirmed_count))
            or any(x.dtype != torch.float32 for x in (state.weights, state.innovation))
            or any(x.dtype != torch.int64 for x in (
                state.visits, state.active, state.shock, state.better, state.confirmed_count))
            or any(x.device != action.device for x in (
                state.weights, state.innovation, state.visits, state.active,
                state.shock, state.better, state.confirmed_count))
            or not bool(torch.isfinite(state.weights).all())
            or not bool(torch.isfinite(state.innovation).all())
            or not bool((state.innovation >= 0).all())
            or not bool((state.visits >= 0).all())
            or not bool(((state.active == 0) | (state.active == 1)).all())):
            raise ValueError("P97_INVALID_STATE")

    def read_only(self, query: torch.Tensor, state: ContextState) -> torch.Tensor:
        self.validate_actions(query, 2)
        self.check(state, query)
        rows = torch.arange(len(query), device=query.device)
        current = state.weights[rows, state.active]
        return torch.bmm(current, F.normalize(query, dim=-1).unsqueeze(-1)).squeeze(-1)

    def ingest(self, actions: torch.Tensor, effects: torch.Tensor,
               confirmed: torch.Tensor, state: ContextState) -> ContextState:
        self.validate_actions(actions, 3)
        self.check(state, actions)
        if (not isinstance(effects, torch.Tensor) or effects.shape != actions.shape
            or effects.dtype != torch.float32 or effects.device != actions.device
            or not isinstance(confirmed, torch.Tensor)
            or confirmed.shape != actions.shape[:2] or confirmed.dtype != torch.bool
            or confirmed.device != actions.device):
            raise ValueError("P97_INVALID_CONFIRMATION")
        safe = torch.where(confirmed.unsqueeze(-1), effects,
                           torch.zeros_like(effects))
        if not bool(torch.isfinite(safe).all()):
            raise ValueError("P97_NONFINITE_CONFIRMED_EFFECT")
        b = len(actions)
        rows = torch.arange(b, device=actions.device)
        W, energy = state.weights, state.innovation
        visits = state.visits
        active, shock, better = state.active, state.shock, state.better
        count = state.confirmed_count
        keys = F.normalize(actions, dim=-1)
        for t in range(actions.shape[1]):
            ok = confirmed[:, t]
            k = keys[:, t]
            prediction = torch.einsum("boij,bj->boi", W, k)
            square = (prediction - safe[:, t, None, :]).square().mean(-1)
            alternate = 1 - active
            ac = visits[rows, active]
            bc = visits[rows, alternate]
            err_a = square[rows, active]
            err_b = square[rows, alternate]
            ema_a = energy[rows, active]
            is_shock = (ac >= 8) & (bc < 8) & (err_a > self.surprise * ema_a.clamp_min(1e-7))
            is_better = (ac >= 8) & (bc >= 8) & (err_b < self.alternative * err_a)
            next_shock = torch.where(is_shock, shock + 1, torch.zeros_like(shock))
            next_better = torch.where(is_better, better + 1, torch.zeros_like(better))
            flip = ((bc == 0) & (next_shock >= 2)) | ((bc >= 8) & (next_better >= 2))
            flip = ok & flip
            chosen = torch.where(flip, alternate, active)
            prediction_chosen = prediction[rows, chosen]
            error = safe[:, t] - prediction_chosen
            current = W[rows, chosen]
            proposal = .99 * current + error[:, :, None] * k[:, None, :]
            # Model selection and all factual writes are gated by real confirmation.
            selected = torch.nn.functional.one_hot(chosen, 2).to(dtype=torch.bool)
            W = torch.where((ok[:, None] & selected)[:, :, None, None],
                            proposal[:, None, :, :], W)
            innov_new = .90 * energy[rows, chosen] + .10 * error.square().mean(-1)
            energy = torch.where(ok[:, None] & selected, innov_new[:, None], energy)
            visits = visits + (ok[:, None] & selected).to(torch.int64)
            active = torch.where(ok, chosen, active)
            shock = torch.where(ok & ~flip, next_shock,
                                torch.where(ok, torch.zeros_like(shock), shock))
            better = torch.where(ok & ~flip, next_better,
                                 torch.where(ok, torch.zeros_like(better), better))
            count = count + ok.to(torch.int64)
        return ContextState(W, energy, visits, active, shock, better, count)

    def episode(self, history: torch.Tensor, probes: torch.Tensor,
                confirmed: torch.Tensor | None = None,
                initial: ContextState | None = None) -> tuple[torch.Tensor, ContextState]:
        if (not isinstance(history, torch.Tensor) or history.ndim != 3
            or history.shape[1:] != (72, 2 * D)
            or history.dtype != torch.float32):
            raise ValueError("P97_INVALID_HISTORY")
        self.validate_actions(probes, 3)
        if probes.shape != (len(history), 3, D) or probes.device != history.device:
            raise ValueError("P97_INVALID_PROBES")
        if confirmed is None:
            confirmed = torch.ones(history.shape[:2], dtype=torch.bool, device=history.device)
        state = self.initial(probes[:, 0]) if initial is None else initial
        predictions = []
        for j in range(3):
            h = history[:, j * 24:(j + 1) * 24]
            state = self.ingest(h[:, :, :D], h[:, :, D:],
                                confirmed[:, j * 24:(j + 1) * 24], state)
            predictions.append(self.read_only(probes[:, j], state))
        return torch.stack(predictions, 1), state


def select_control(fit: dict) -> tuple[tuple[float, float], list]:
    rows = []
    for rate in RATES:
        for decay in DECAYS:
            scores = mse(baseline(fit["history"], fit["probes"], rate, decay), fit)
            rows.append({"rate": rate, "decay": decay, "fit": scores})
    selected = min(rows, key=lambda x: x["fit"]["all"])
    return (selected["rate"], selected["decay"]), rows


def select_candidate(fit: dict) -> tuple[tuple[float, float], list]:
    rows = []
    for surprise in SURPRISE:
        for alternative in ALTERNATIVE:
            model = RecurrentContextMemory(surprise, alternative)
            predictions, _ = model.episode(fit["history"], fit["probes"])
            rows.append({"surprise": surprise, "alternative": alternative,
                         "fit": mse(predictions, fit)})
    selected = min(rows, key=lambda x: x["fit"]["all"])
    return (selected["surprise"], selected["alternative"]), rows


def cold_and_negative(model: RecurrentContextMemory, world: dict) -> dict:
    h, q = world["history"][:8].clone(), world["probes"][:8].clone()
    full, state = model.episode(h, q)
    a = model.initial(q[:, 0])
    for block in range(3):
        part = h[:, block * 24:(block + 1) * 24]
        a = model.ingest(part[:, :, :D], part[:, :, D:],
                         torch.ones((8, 24), dtype=torch.bool), a.snapshot())
    cold = all(torch.equal(getattr(a, field), getattr(state, field))
               for field in ContextState.__dataclass_fields__)
    cold = cold and torch.equal(model.read_only(q[:, 2], a), full[:, 2])
    mask = torch.ones((8, 72), dtype=torch.bool)
    mask[:, 1::3] = False
    tamper = h.clone()
    tamper[:, 1::3, D:] = float("nan")
    p1, s1 = model.episode(h, q, mask)
    p2, s2 = model.episode(tamper, q, mask)
    masked = torch.equal(p1, p2) and all(
        torch.equal(getattr(s1, field), getattr(s2, field))
        for field in ContextState.__dataclass_fields__)
    return {"cold_bitwise": bool(cold), "masked_nan_inert": bool(masked)}


def median_ms(fn) -> float:
    with torch.no_grad():
        for _ in range(2):
            fn()
        durations = []
        for _ in range(7):
            start = time.perf_counter()
            fn()
            durations.append((time.perf_counter() - start) * 1000)
    return statistics.median(durations)


def run() -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path(__file__).resolve().parents[1]
    out = {"schema": "ttt-hybrid.p97-recurrent-context-dev-result.v1",
           "freeze_commit": FREEZE, "status": "NOT_RUN",
           "scientific_pass": False, "protected_ood_generated": False,
           "protected_negative_generated": False, "original_model_modified": False,
           "predictive_float_state": 514, "control_int_tensors": 5,
           "model_training": "ONLINE_FACTS_ONLY"}
    out["source_custody"] = {path: p86.blob(root / path) for path in BLOBS}
    if out["source_custody"] != BLOBS:
        out["status"] = "DEV_FAIL_SOURCE_CUSTODY"
        return out
    worlds = {key: source(key) for key in SEEDS}
    out["sources"] = {key: {"sha256": fingerprint(w), "audit_pass": audit(w)}
                      for key, w in worlds.items()}
    if not all(row["audit_pass"] for row in out["sources"].values()):
        out["status"] = "DEV_FAIL_SOURCE_AUDIT"
        return out
    out["families"] = {}
    for family in ("A", "B"):
        fit, cal = worlds[family + "_FIT"], worlds[family + "_CAL"]
        control, grid = select_control(fit)
        setting, candidate_grid = select_candidate(fit)
        model = RecurrentContextMemory(*setting)
        pred, state = model.episode(cal["history"], cal["probes"])
        ref = baseline(cal["history"], cal["probes"], *control)
        fixed = baseline(cal["history"], cal["probes"])
        zero = torch.zeros_like(pred)
        wrong = cal["history"].clone()
        wrong[:, :, D:] = wrong[:, :, D:].roll(13, dims=1)
        wrong_pred, _ = model.episode(wrong, cal["probes"])
        numbers = {"candidate": mse(pred, cal), "FIT_selected_W256": mse(ref, cal),
                   "fixed_W256": mse(fixed, cal), "zero": mse(zero, cal),
                   "wrong_credit": mse(wrong_pred, cal)}
        # A matched-state conventional W256 with 258 additional inert floats
        # has identical outputs, not an unfair smaller claimed state.
        inert = cal["history"].new_zeros((len(cal["history"]), 258))
        if (inert.numel() != len(cal["history"]) * 258
            or state.weights.numel() != len(cal["history"]) * 512
            or state.innovation.numel() != len(cal["history"]) * 2
            or not torch.equal(ref, baseline(cal["history"], cal["probes"], *control))):
            raise ValueError("P97_CAPACITY_OR_PARITY")
        negative = cold_and_negative(model, cal)
        negative["wrong_credit"] = (
            numbers["wrong_credit"]["all"] >= 1.05 * numbers["candidate"]["all"])
        sample = {k: v[:8] for k, v in cal.items()}
        cost = {"W256_ms": median_ms(lambda: baseline(sample["history"], sample["probes"], *control)),
                "context_ms": median_ms(lambda: model.episode(sample["history"], sample["probes"]))}
        cost["ratio"] = cost["context_ms"] / cost["W256_ms"]
        cand, base = numbers["candidate"], numbers["FIT_selected_W256"]
        quality = {
            "mid_changed": cand["mid_changed"] <= .90 * base["mid_changed"],
            "return_changed": cand["return_changed"] <= .85 * base["return_changed"],
            "stable_final": cand["stable_final"] <= 1.05 * base["stable_final"],
            "combined": cand["all"] <= .95 * base["all"],
        }
        passed = all(quality.values()) and all(negative.values()) and cost["ratio"] <= 3.
        out["families"][family] = {
            "FIT_selected_W256": {"eta": control[0], "decay": control[1]},
            "FIT_control_grid": grid,
            "FIT_selected_context": {"surprise": setting[0], "alternative": setting[1]},
            "FIT_context_grid": candidate_grid,
            "CAL": numbers, "quality": quality, "negative": negative,
            "cpu": cost, "family_pass": bool(passed),
            "matched_inert_514_control_parity": True,
            "switched_worlds": int(cal["changed_world"].sum()),
            "source_split": family + "_CAL",
        }
    out["status"] = ("DEV_CAL_ONLY_PASS_PROTECTED_FIRST_USE_REQUIRED"
                     if all(x["family_pass"] for x in out["families"].values())
                     else "DEV_FAIL_QUALITY_OR_COST")
    return out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "family_pass": {k: v["family_pass"] for k, v in report.get("families", {}).items()},
                      "scientific_pass": report["scientific_pass"]}))


if __name__ == "__main__":
    main()
