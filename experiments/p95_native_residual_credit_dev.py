"""P95 author-synthetic DEV falsifier: fully train native Hybrid residual over confirmed W256.

Protocol was frozen first at Git commit 5ca8e19948c9ff90b6dd0c4e9f7e2a2eb36ef545.
Protected OOD/negative external or reserved worlds are never available here.
Do not treat engineering success or this synthetic CAL experiment as SCIENTIFIC_PASS.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from experiments import p86_portable_causal_preflight as p86
from native_p94_confirmed_causal_memory import AnchoredHybridPilot, ConfirmedCausalMemory

D = 16
SEEDS = {"FIT": 950111, "CAL": 950113}
N = {"FIT": 384, "CAL": 128}
RESERVED = {"OOD": 950117, "NEGATIVE": 950119}
EXPECTED_SHA = {
    "FIT": "fd20fadcb23fee33958b6c5950c6c9e5310f5c96447ff954341abf0ccb5d9fcb",
    "CAL": "e5fa7731f41e378f8d2f098b7867d1a2892973c34c2fe75cc6c25dbe89f65f19",
}
ORIGINAL_BLOBS = {
    "model.py": "5d9ea57b827aa1796548447868fa5ee14c251b49",
    "experiments/p86_portable_causal_preflight.py": "008492665ee9090db5791963f2e2336ea283ee62",
    "native_p94_confirmed_causal_memory.py": "378167499362173023b02a44e1b9ffb7a1eeba42",
}
MODEL_SEEDS = (950201, 950203, 950207)
STEPS, BATCH, CHECK = 300, 16, 30
FREEZE = "5ca8e19948c9ff90b6dd0c4e9f7e2a2eb36ef545"


def source(split: str):
    """Use exact P86 integer/dyadic rules with new P95 seeds, never reserved sources."""
    if split not in SEEDS:
        raise ValueError("P95_RESERVED_SOURCE_DENIED")
    seed, n = SEEDS[split], N[split]
    changed_ids = set(sorted(range(n), key=lambda w: p86.digest_int(
        seed, w, "regime", 0, 0, 2**32))[:n // 2])
    histories, queries, effects, regimes = [], [], [], []
    for world in range(n):
        u0, v0 = p86.basis(seed, world, "u0"), p86.basis(seed, world, "v0")
        u1, v1 = p86.basis(seed, world, "u1"), p86.basis(seed, world, "v1")
        shifted = world in changed_ids
        past = []
        for t in range(49):
            ai = [p86.digest_int(seed, world, "action", t, j, 193) - 96 for j in range(D)]
            action = [a / 128 for a in ai]
            u, v = (u0, v0) if t < 24 or not shifted else (u1, v1)
            target = p86.causal_effect(ai, u, v)
            if t < 48:
                past.append(action + target)
            else:
                queries.append(action + [0.] * D)
                effects.append(target)
        histories.append(past)
        regimes.append(shifted)
    return {
        "history": torch.tensor(histories, dtype=torch.float32),
        "query": torch.tensor(queries, dtype=torch.float32),
        "effect": torch.tensor(effects, dtype=torch.float32),
        "changed_world": torch.tensor(regimes, dtype=torch.bool),
    }


def check_cohort(split, world):
    sha = p86.world_sha(world)
    audit = p86.audit(world)
    if sha != EXPECTED_SHA[split] or not audit["pass"]:
        raise ValueError("P95_SOURCE_HASH_OR_AUDIT_MISMATCH")
    return {"worlds": N[split], "sha256": sha, "audit": audit}


def score(p, world):
    return p86.error(p, world)


def model_prediction(model, world):
    # Exactly 48 historical confirmed effect writes; final query effect is withheld.
    return model(world["history"], world["query"][:, :D], use_residual=True)


def fingerprint(model):
    return hashlib.sha256(b"".join(p.detach().cpu().contiguous().numpy().tobytes()
                         for p in model.state_dict().values())).hexdigest()


def train(seed, fit, cal, ix):
    torch.manual_seed(seed)
    model = AnchoredHybridPilot().train()
    init_hash = fingerprint(model)
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=.002, weight_decay=.01)
    start = time.perf_counter()
    best, best_step, best_state = float("inf"), 0, None
    trace = []
    for step in range(1, STEPS + 1):
        batch = ix[step - 1]
        opt.zero_grad(set_to_none=True)
        prediction = model_prediction(model, {k: v[batch] for k, v in fit.items()})
        loss = (prediction - fit["effect"][batch]).square().mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError("P95_NONFINITE_LOSS")
        loss.backward()
        grad = nn.utils.clip_grad_norm_(params, 1.)
        if not bool(torch.isfinite(grad)):
            raise ValueError("P95_NONFINITE_GRAD")
        opt.step()
        if step % CHECK == 0:
            model.eval()
            with torch.no_grad():
                mse = score(model_prediction(model, cal), cal)
            if not all(math.isfinite(z) for z in mse.values()):
                raise ValueError("P95_NONFINITE_CAL")
            trace.append({"step": step, "cal": mse})
            if mse["combined"] < best:
                best, best_step, best_state = mse["combined"], step, copy.deepcopy(model.state_dict())
            model.train()
    if best_state is None:
        raise AssertionError("P95_NO_CHECKPOINT")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model_prediction(model, cal)
    return model, {
        "seed": seed,
        "start_model_sha256": init_hash,
        "selected_model_sha256": fingerprint(model),
        "selected_checkpoint": best_step,
        "cal": score(prediction, cal),
        "train_seconds": time.perf_counter() - start,
        "updates": STEPS,
        "parameter_count": sum(p.numel() for p in params),
        "trace": trace,
    }


def cold_restore(world):
    w = world["history"][:8]
    q = world["query"][:8, :D]
    mem = ConfirmedCausalMemory()
    yes = torch.ones((8, 48), dtype=torch.bool)
    first = mem.ingest_confirmed(w[:, :24, :D], w[:, :24, D:],
                                 yes[:, :24], mem.initial_state(q))
    state = first.snapshot()
    tail = mem.ingest_confirmed(w[:, 24:, :D], w[:, 24:, D:], yes[:, 24:], state)
    full = mem.forward_episode(w, q)[1]
    return (torch.equal(tail.fast_weights, full.fast_weights) and
            torch.equal(tail.confirmed_count, full.confirmed_count) and
            torch.equal(mem.read_only(q, tail), mem.read_only(q, full)))


def median_episode_ms(fn):
    for _ in range(3):
        with torch.no_grad():
            fn()
    times = []
    for _ in range(10):
        start = time.perf_counter()
        with torch.no_grad():
            fn()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    receipt = {
        "schema": "ttt-hybrid.p95-native-residual-author-dev-result.v1",
        "freeze_commit": FREEZE, "scientific_pass": False,
        "protected_OOD_generated": False, "protected_NEGATIVE_generated": False,
        "original_model_modified": False, "status": "NOT_RUN",
    }
    root = Path(__file__).resolve().parents[1]
    receipt["source_custody"] = {path: p86.blob(root / path) for path in ORIGINAL_BLOBS}
    if receipt["source_custody"] != ORIGINAL_BLOBS:
        receipt["status"] = "DEV_FAIL_SOURCE_CUSTODY"
        return receipt
    fit, cal = source("FIT"), source("CAL")
    receipt["cohorts"] = {
        "FIT": check_cohort("FIT", fit), "CAL": check_cohort("CAL", cal),
    }
    if not cold_restore(cal):
        receipt["status"] = "DEV_FAIL_COLD_RESTORE"
        return receipt
    frozen, _ = p86.online(cal, 1., .99)
    original, _ = p86.p78_original(cal)
    receipt["controls"] = {
        "W256_CAL_frozen": score(frozen, cal),
        "P78_original": score(original, cal),
        "zero": score(torch.zeros_like(frozen), cal),
    }
    if not all(receipt["controls"]["W256_CAL_frozen"][z] < receipt["controls"]["zero"][z]
               for z in ("combined", "stable", "changed")):
        receipt["status"] = "DEV_FAIL_STRONG_W256_CONTROL"
        return receipt
    # Same frozen indices for all three trainable model seeds.
    ix = torch.randint(len(fit["effect"]), (STEPS, BATCH),
                       generator=torch.Generator().manual_seed(950299))
    receipt["sampler_sha256"] = hashlib.sha256(ix.numpy().tobytes()).hexdigest()
    rows = []
    for seed in MODEL_SEEDS:
        model, info = train(seed, fit, cal, ix)
        # Wrong-credit negative is diagnostic on this already-consumed CAL, never new OOD.
        wrong = {k: v.clone() for k, v in cal.items()}
        wrong["history"][:, :, D:] = wrong["history"][:, :, D:].roll(11, dims=1)
        with torch.no_grad():
            wrong_predict = model_prediction(model, wrong)
        info["wrong_credit_mse"] = score(wrong_predict, cal)
        rows.append((model, info))
    receipt["models"] = [info for _, info in rows]
    sample = {k: v[:8] for k, v in cal.items()}
    base = lambda: p86.online(sample, 1., .99)
    residual = lambda: model_prediction(rows[0][0], sample)
    receipt["cpu_ms"] = {
        "W256": median_episode_ms(base),
        "anchored_hybrid": median_episode_ms(residual),
    }
    receipt["cpu_ratio"] = receipt["cpu_ms"]["anchored_hybrid"] / receipt["cpu_ms"]["W256"]
    conventional = receipt["controls"]["W256_CAL_frozen"]
    individual = []
    for _, info in rows:
        e = info["cal"]
        individual.append({
            "seed": info["seed"],
            "quality_combined_pass": e["combined"] <= .90 * conventional["combined"],
            "quality_stable_pass": e["stable"] <= .95 * conventional["stable"],
            "quality_changed_pass": e["changed"] <= .90 * conventional["changed"],
            "wrong_credit_pass": info["wrong_credit_mse"]["combined"] >= 1.05 * e["combined"],
        })
    receipt["gates"] = individual
    receipt["cost_pass"] = receipt["cpu_ratio"] <= 3.
    receipt["status"] = ("DEV_CAL_ONLY_PASS_PROTECTED_FIRST_USE_REQUIRED"
                         if receipt["cost_pass"] and all(all(v for k, v in row.items()
                             if k.endswith("_pass")) for row in individual)
                         else "DEV_FAIL_QUALITY_OR_COST")
    return receipt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", type=Path, required=True)
    a = ap.parse_args()
    out = run()
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    a.receipt.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": out["status"], "cost_ratio": out.get("cpu_ratio"),
        "gates": out.get("gates"),
        "protected_OOD_generated": out["protected_OOD_generated"],
        "scientific_pass": out["scientific_pass"],
    }))


if __name__ == "__main__":
    main()
