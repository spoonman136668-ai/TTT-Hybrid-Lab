"""P96: preregistered online feature-precision causal memory DEV falsifier.

Protocol was committed BEFORE source generation at
3750d547626f048711e24dd6cdadb2d101d75283.
No reserved/real sources, protected first use, original-model changes or promotion.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional as F

from experiments import p86_portable_causal_preflight as p86

D = 16
FREEZE = "3750d547626f048711e24dd6cdadb2d101d75283"
SEEDS = {"A_FIT": 960111, "A_CAL": 960113,
         "B_FIT": 960211, "B_CAL": 960213}
RESERVED = {"OOD": 960117, "NEGATIVE": 960119}
NUM_WORLDS = {"FIT": 256, "CAL": 128}
ORIGINAL_BLOBS = {
    "model.py": "5d9ea57b827aa1796548447868fa5ee14c251b49",
    "experiments/p86_portable_causal_preflight.py": "008492665ee9090db5791963f2e2336ea283ee62",
    "native_p94_confirmed_causal_memory.py": "378167499362173023b02a44e1b9ffb7a1eeba42",
}
RATES = (0.5, 1.0, 1.5)
DECAYS = (0.95, 0.99, 1.0)


def source(key: str) -> dict[str, torch.Tensor]:
    """Two new author-generated dyadic families. Protected seeds are inaccessible."""
    if key not in SEEDS:
        raise ValueError("P96_PROTECTED_OR_UNKNOWN_SOURCE")
    family, split = key.split("_")
    seed = SEEDS[key]
    n = NUM_WORLDS[split]
    changed_ids = set(sorted(
        range(n), key=lambda w: p86.digest_int(seed, w, "regime", 0, 0, 2**32)
    )[:n // 2])
    histories, queries, effects, regimes = [], [], [], []
    for world in range(n):
        u0, v0 = p86.basis(seed, world, "u0"), p86.basis(seed, world, "v0")
        u1, v1 = p86.basis(seed, world, "u1"), p86.basis(seed, world, "v1")
        changed = world in changed_ids
        past = []
        for t in range(49):
            ai = [p86.digest_int(seed, world, "action", t, j, 193) - 96
                  for j in range(D)]
            if family == "B":
                ai = [a // (2 ** (j % 4)) for j, a in enumerate(ai)]
            action = [a / 128 for a in ai]
            u, v = (u0, v0) if t < 24 or not changed else (u1, v1)
            target = p86.causal_effect(ai, u, v)
            if t < 48:
                past.append(action + target)
            else:
                queries.append(action + [0.] * D)
                effects.append(target)
        histories.append(past)
        regimes.append(changed)
    return {
        "history": torch.tensor(histories, dtype=torch.float32),
        "query": torch.tensor(queries, dtype=torch.float32),
        "effect": torch.tensor(effects, dtype=torch.float32),
        "changed_world": torch.tensor(regimes, dtype=torch.bool),
    }


@dataclass(frozen=True)
class PrecisionState:
    weights: torch.Tensor
    precision: torch.Tensor
    confirmed_count: torch.Tensor

    def snapshot(self) -> "PrecisionState":
        return PrecisionState(self.weights.detach().clone(),
                              self.precision.detach().clone(),
                              self.confirmed_count.clone())


class DiagonalPrecisionCausalMemory:
    """256 factual weights + 16 learned diagonal energy floats; no offline weights."""

    def initial(self, query: torch.Tensor) -> PrecisionState:
        self._actions(query, 2)
        return PrecisionState(query.new_zeros((len(query), D, D)),
                              query.new_zeros((len(query), D)),
                              torch.zeros(len(query), dtype=torch.int64, device=query.device))

    @staticmethod
    def _actions(action: torch.Tensor, rank: int) -> None:
        if (not isinstance(action, torch.Tensor) or action.ndim != rank or
            action.shape[-1] != D or action.dtype != torch.float32 or
            not bool(torch.isfinite(action).all())):
            raise ValueError("P96_INVALID_ACTION")

    def _state(self, state: PrecisionState, action: torch.Tensor) -> None:
        if (not isinstance(state, PrecisionState) or
            state.weights.shape != (len(action), D, D) or
            state.precision.shape != (len(action), D) or
            state.confirmed_count.shape != (len(action),) or
            state.weights.dtype != torch.float32 or
            state.precision.dtype != torch.float32 or
            state.confirmed_count.dtype != torch.int64 or
            any(z.device != action.device for z in
                (state.weights, state.precision, state.confirmed_count)) or
            not bool(torch.isfinite(state.weights).all()) or
            not bool(torch.isfinite(state.precision).all()) or
            not bool((state.precision >= 0).all()) or
            not bool((state.confirmed_count >= 0).all())):
            raise ValueError("P96_INVALID_STATE")

    def ingest(self, action: torch.Tensor, effect: torch.Tensor,
               confirmed: torch.Tensor, state: PrecisionState) -> PrecisionState:
        self._actions(action, 3)
        self._state(state, action)
        if (not isinstance(effect, torch.Tensor) or effect.shape != action.shape or
            effect.dtype != action.dtype or effect.device != action.device or
            not isinstance(confirmed, torch.Tensor) or
            confirmed.shape != action.shape[:2] or
            confirmed.dtype != torch.bool or confirmed.device != action.device):
            raise ValueError("P96_INVALID_CONFIRMATION")
        safe_effect = torch.where(confirmed.unsqueeze(-1), effect,
                                  torch.zeros_like(effect))
        if not bool(torch.isfinite(safe_effect).all()):
            raise ValueError("P96_NONFINITE_CONFIRMED_EFFECT")
        W, c, count = state.weights, state.precision, state.confirmed_count
        for t in range(action.shape[1]):
            k = F.normalize(action[:, t], dim=-1)
            predicted = torch.bmm(W, k.unsqueeze(-1)).squeeze(-1)
            err = safe_effect[:, t] - predicted
            next_c = .99 * c + k.square()
            pre = k * torch.rsqrt(next_c + 1e-4)
            normalizer = (k * pre).sum(-1, keepdim=True).clamp_min(1e-12)
            g = pre / normalizer
            candidate = .99 * W + err.unsqueeze(-1) * g.unsqueeze(1)
            selected = confirmed[:, t]
            W = torch.where(selected[:, None, None], candidate, W)
            c = torch.where(selected[:, None], next_c, c)
            count = count + selected.to(torch.int64)
        return PrecisionState(W, c, count)

    def read_only(self, query: torch.Tensor, state: PrecisionState) -> torch.Tensor:
        self._actions(query, 2)
        self._state(state, query)
        return torch.bmm(state.weights,
                         F.normalize(query, dim=-1).unsqueeze(-1)).squeeze(-1)

    def episode(self, world: dict[str, torch.Tensor],
                confirmed: torch.Tensor | None = None,
                initial: PrecisionState | None = None
                ) -> tuple[torch.Tensor, PrecisionState]:
        history, query = world["history"], world["query"][:, :D]
        if (history.ndim != 3 or history.shape[1:] != (48, D * 2) or
            history.dtype != torch.float32 or history.device != query.device):
            raise ValueError("P96_INVALID_HISTORY")
        if confirmed is None:
            confirmed = torch.ones(history.shape[:2], dtype=torch.bool,
                                   device=history.device)
        state = self.initial(query) if initial is None else initial
        updated = self.ingest(history[:, :, :D], history[:, :, D:],
                              confirmed, state)
        return self.read_only(query, updated), updated


def score(prediction: torch.Tensor, world: dict) -> dict[str, float]:
    return p86.error(prediction, world)


def source_receipt(key: str, world: dict) -> dict:
    audit = p86.audit(world)
    if not audit["pass"]:
        raise ValueError("P96_SOURCE_AUDIT_FAILURE")
    return {"sha256": p86.world_sha(world), "audit": audit,
            "family": key.split("_")[0], "split": key.split("_")[1]}


def select_baseline(fit: dict) -> tuple[tuple[float, float], list]:
    rows = []
    for eta in RATES:
        for decay in DECAYS:
            pred, state = p86.online(fit, eta, decay)
            if state.shape[1:] != (D, D):
                raise ValueError("P96_BASELINE_STATE_SIZE")
            m = score(pred, fit)
            rows.append({"eta": eta, "decay": decay, "fit_mse": m})
    best = min(rows, key=lambda r: r["fit_mse"]["combined"])
    return (best["eta"], best["decay"]), rows


def cold_restore(mem: DiagonalPrecisionCausalMemory, world: dict) -> bool:
    w = {key: value[:8] for key, value in world.items()}
    q = w["query"][:, :D]
    full, full_state = mem.episode(w)
    initial = mem.initial(q)
    first = mem.ingest(w["history"][:, :24, :D], w["history"][:, :24, D:],
                       torch.ones((8, 24), dtype=torch.bool), initial)
    restored = first.snapshot()
    second = mem.ingest(w["history"][:, 24:, :D], w["history"][:, 24:, D:],
                        torch.ones((8, 24), dtype=torch.bool), restored)
    return bool(torch.equal(second.weights, full_state.weights) and
                torch.equal(second.precision, full_state.precision) and
                torch.equal(second.confirmed_count, full_state.confirmed_count) and
                torch.equal(full, mem.read_only(q, second)))


def confirmation_negative(mem: DiagonalPrecisionCausalMemory,
                          world: dict) -> bool:
    w = {key: value[:8].clone() for key, value in world.items()}
    mask = torch.ones((8, 48), dtype=torch.bool)
    mask[:, 1::3] = False
    tamper = {key: value.clone() for key, value in w.items()}
    tamper["history"][:, 1::3, D:] = float("nan")
    a, sa = mem.episode(w, mask)
    b, sb = mem.episode(tamper, mask)
    return bool(torch.equal(a, b) and torch.equal(sa.weights, sb.weights) and
                torch.equal(sa.precision, sb.precision) and
                torch.equal(sa.confirmed_count, sb.confirmed_count))


def median_ms(fn) -> float:
    for _ in range(3):
        fn()
    times = []
    for _ in range(10):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def run() -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path(__file__).resolve().parents[1]
    receipt = {
        "schema": "ttt-hybrid.p96-diagonal-precision-dev-result.v1",
        "freeze_commit": FREEZE,
        "status": "NOT_RUN",
        "scientific_pass": False,
        "protected_ood_generated": False,
        "protected_negative_generated": False,
        "original_model_modified": False,
        "learned_memory_floats": 272,
        "model_training": "ONLINE_ONLY",
    }
    receipt["source_custody"] = {k: p86.blob(root / k) for k in ORIGINAL_BLOBS}
    if receipt["source_custody"] != ORIGINAL_BLOBS:
        receipt["status"] = "DEV_FAIL_SOURCE_CUSTODY"
        return receipt
    worlds = {name: source(name) for name in SEEDS}
    receipt["sources"] = {key: source_receipt(key, world)
                          for key, world in worlds.items()}
    mem = DiagonalPrecisionCausalMemory()
    receipt["families"] = {}
    for family in ("A", "B"):
        fit, cal = worlds[family + "_FIT"], worlds[family + "_CAL"]
        selected, fit_grid = select_baseline(fit)
        fixed, fixed_state = p86.online(cal, 1., .99)
        chosen, chosen_state = p86.online(cal, *selected)
        candidate, candidate_state = mem.episode(cal)
        if (chosen_state.shape[1:] != (D, D) or
            candidate_state.weights.numel() != len(cal["effect"]) * 256 or
            candidate_state.precision.numel() != len(cal["effect"]) * 16 or
            not bool(torch.equal(fixed, p86.online(cal, 1., .99)[0]))):
            raise ValueError("P96_BASELINE_PARITY_OR_MEMORY_BUDGET")
        changed = cal["changed_world"]
        # A conventional W256 with 16 initialized but inert extra floats is bitwise identical.
        matched = torch.zeros((len(cal["effect"]), 16), dtype=torch.float32)
        if (matched.numel() != len(cal["effect"]) * 16 or
            not torch.equal(chosen, p86.online(cal, *selected)[0])):
            raise ValueError("P96_MATCHED_272_CONTROL_PARITY")
        wrong = {k: v.clone() for k, v in cal.items()}
        wrong["history"][:, :, D:] = wrong["history"][:, :, D:].roll(11, dims=1)
        wrong_pred, _ = mem.episode(wrong)
        measured = {"fixed_W256": score(fixed, cal),
                    "fit_selected_W256": score(chosen, cal),
                    "matched_272_inert_W256": score(chosen, cal),
                    "diagonal_precision": score(candidate, cal),
                    "wrong_credit": score(wrong_pred, cal),
                    "zero": score(torch.zeros_like(candidate), cal)}
        negative = {
            "wrong_credit_pass":
                measured["wrong_credit"]["combined"] >=
                1.05 * measured["diagonal_precision"]["combined"],
            "bitwise_cold_restore": cold_restore(mem, cal),
            "unconfirmed_nan_mask": confirmation_negative(mem, cal),
        }
        slice8 = {k: v[:8] for k, v in cal.items()}
        times = {"W256_ms": median_ms(lambda: p86.online(slice8, 1., .99)),
                 "precision_ms": median_ms(lambda: mem.episode(slice8))}
        times["CPU_ratio"] = times["precision_ms"] / times["W256_ms"]
        conventional = measured["fit_selected_W256"]
        proposed = measured["diagonal_precision"]
        quality_gates = {
            "combined": proposed["combined"] <=
                (1.02 if family == "A" else .90) * conventional["combined"],
            "stable": proposed["stable"] <=
                (1.05 if family == "A" else .95) * conventional["stable"],
        }
        if family == "B":
            quality_gates["changed"] = (
                proposed["changed"] <= .90 * conventional["changed"])
        family_pass = (all(quality_gates.values()) and
                       all(negative.values()) and
                       times["CPU_ratio"] <= 3.)
        receipt["families"][family] = {
            "FIT_selected": {"eta": selected[0], "decay": selected[1]},
            "FIT_grid": fit_grid,
            "CAL": measured,
            "controls": negative,
            "quality_gates": quality_gates,
            "cost": times,
            "family_pass": bool(family_pass),
            "matched_memory_floats": 272,
            "hidden_switch_world_count": int(changed.sum()),
        }
    receipt["status"] = (
        "DEV_CAL_ONLY_PASS_PROTECTED_FIRST_USE_REQUIRED"
        if all(v["family_pass"] for v in receipt["families"].values())
        else "DEV_FAIL_QUALITY_OR_COST_OR_CONTROLS"
    )
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", type=Path, required=True)
    a = ap.parse_args()
    receipt = run()
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    a.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "family_pass": {
        k: v["family_pass"] for k, v in receipt.get("families", {}).items()},
        "scientific_pass": receipt["scientific_pass"]}))


if __name__ == "__main__":
    main()
