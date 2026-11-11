"""P98: blind context-switch diagnostic on NEW A-B-A causal-dev cohorts.

No P97 prior CAL retuning. The learner never sees hidden phase/regime labels.
Only author-synthetic FIT/CAL; never authentic F3 source, OOD or deployment.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import torch

from experiments import p86_portable_causal_preflight as p86
from experiments import p97_recurrent_context_dev as p97

FREEZE_COMMIT = "ec045d0eddc4ba226803ff0bea30ef7a309e5cbf"
FREEZE_BLOB = "dcc2420278d0e8e937bd701c5d4029e5e8c67f58"
SEEDS = {"A_FIT": 980111, "A_CAL": 980113,
         "B_FIT": 980211, "B_CAL": 980213}
RESERVED = {"OOD_RESERVED": 980117, "NEGATIVE_RESERVED": 980119}
COUNTS = {"FIT": 128, "CAL": 64}
BLOBS = {
    "model.py": "5d9ea57b827aa1796548447868fa5ee14c251b49",
    "experiments/p86_portable_causal_preflight.py":
        "008492665ee9090db5791963f2e2336ea283ee62",
    "experiments/p97_recurrent_context_dev.py":
        "74030c3d7d2f5f51781c740493be21d3c769792d",
}
D = 16


def frozen_protocol():
    root = Path(__file__).resolve().parents[1]
    raw = (root / "experiments/p98_blind_switch_diagnostics_freeze.json").read_bytes()
    got = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if got != FREEZE_BLOB:
        raise ValueError("P98_FREEZE_BLOB_DRIFT")
    f = json.loads(raw)
    if (
        f["schema"] != "ttt-hybrid.p98-blinded-context-switch-diagnostics-dev-freeze.v1"
        or f["identity"]["parent_sha"] != "94e3370f4869869effd19a6d849cc93a722f2bde"
        or f["source"]["seeds"] != {**SEEDS, **RESERVED}
        or f["source"]["worlds"] != COUNTS
        or f["identity"]["p97_source_sha"] != BLOBS["experiments/p97_recurrent_context_dev.py"]
        or f["candidate"]["threshold_grid"]["surprise"] != list(p97.SURPRISE)
        or f["candidate"]["threshold_grid"]["alternative"] != list(p97.ALTERNATIVE)
        or f["controls"]["W256_GRID"]["eta"] != list(p97.RATES)
        or f["controls"]["W256_GRID"]["decay"] != list(p97.DECAYS)
    ):
        raise ValueError("P98_FROZEN_PROTOCOL_DRIFT")
    for path, expected in BLOBS.items():
        if p86.blob(root / path) != expected:
            raise ValueError("P98_ORIGINAL_SOURCE_BLOB_DRIFT")
    return f


def source(key: str) -> dict:
    if key not in SEEDS:
        raise ValueError("P98_PROTECTED_OR_UNKNOWN_SOURCE_DENIED")
    family, split = key.split("_")
    seed, n = SEEDS[key], COUNTS[split]
    switched = set(sorted(range(n), key=lambda w: p86.digest_int(
        seed, w, "p98-regime", 0, 0, 2**32))[:n // 2])
    histories, probes, targets, flags = [], [], [], []
    for world in range(n):
        u0, v0 = p86.basis(seed, world, "p98-u0"), p86.basis(seed, world, "p98-v0")
        u1, v1 = p86.basis(seed, world, "p98-u1"), p86.basis(seed, world, "p98-v1")
        h, q, y = [], [], []
        for t in range(72):
            a = [p86.digest_int(seed, world, "p98-action", t, j, 193)-96
                 for j in range(D)]
            if family == "B":
                a = [z // (2 ** (j % 4)) for j, z in enumerate(a)]
            u, v = (u1, v1) if (world in switched and 24 <= t < 48) else (u0, v0)
            h.append([z / 128 for z in a] + p86.causal_effect(a, u, v))
        for phase in range(3):
            a = [p86.digest_int(seed, world, "p98-probe", phase, j, 193)-96
                 for j in range(D)]
            if family == "B":
                a = [z // (2 ** (j % 4)) for j, z in enumerate(a)]
            u, v = (u1, v1) if (world in switched and phase == 1) else (u0, v0)
            q.append([z / 128 for z in a])
            y.append(p86.causal_effect(a, u, v))
        histories.append(h)
        probes.append(q)
        targets.append(y)
        flags.append(world in switched)
    out = {"history": torch.tensor(histories, dtype=torch.float32),
           "probes": torch.tensor(probes, dtype=torch.float32),
           "targets": torch.tensor(targets, dtype=torch.float32),
           "changed_world": torch.tensor(flags, dtype=torch.bool)}
    if not p97.audit(out):
        raise ValueError("P98_FRESH_SOURCE_AUDIT_FAIL")
    return out


def route_trace(model, world: dict) -> tuple[torch.Tensor, torch.Tensor, p97.ContextState]:
    """Observe causal route decisions without passing ground-truth flags to memory."""
    h, q = world["history"], world["probes"]
    b = len(h)
    state = model.initial(q[:, 0])
    actions, effects = h[:, :, :D], h[:, :, D:]
    active = torch.empty((b, 72), dtype=torch.int64)
    predictions = []
    with torch.no_grad():
        for t in range(72):
            state = model.ingest(
                actions[:, t:t+1], effects[:, t:t+1],
                torch.ones((b, 1), dtype=torch.bool), state)
            active[:, t] = state.active
            if t in (23, 47, 71):
                predictions.append(model.read_only(q[:, (t+1)//24-1], state))
        ref_pred, ref_state = model.episode(h, q)
    got = torch.stack(predictions, dim=1)
    # Bitwise route audit is an independent negative, not a changed model.
    if not torch.equal(got, ref_pred) or any(
        not torch.equal(getattr(state, name), getattr(ref_state, name))
        for name in p97.ContextState.__dataclass_fields__
    ):
        raise ValueError("P98_PER_EVENT_VS_BATCH_ROUTE_PARITY_FAIL")
    return got, active, state


def route_diagnostics(pred, active, world):
    changed = world["changed_world"]  # EVALUATOR-ONLY oracle, not passed to learner.
    stable = ~changed
    a0 = active[:, 23]
    # Switch events never use hidden labels. Only the evaluator assigns true/false.
    flips = torch.zeros_like(active, dtype=torch.bool)
    flips[:, 1:] = active[:, 1:] != active[:, :-1]
    stable_flip = flips[stable, 8:].any(dim=1)
    changed_flip = flips[changed, 24:36].any(dim=1) & (
        (active[changed, 24:36] != a0[changed, None]).any(dim=1))
    return_back = (active[changed, 47] != a0[changed]) & (
        (active[changed, 48:60] == a0[changed, None]).any(dim=1))
    e = (pred[:, 2] - world["targets"][:, 2]).square().mean(-1)
    e_yes = e[stable][stable_flip]
    e_no = e[stable][~stable_flip]
    ratio = None if len(e_yes) == 0 or len(e_no) == 0 else (
        float(e_yes.mean()) / max(float(e_no.mean()), 1e-12))
    return {
        "stable_worlds": int(stable.sum()), "changed_worlds": int(changed.sum()),
        "stable_false_switch_worlds": int(stable_flip.sum()),
        "stable_false_switch_rate": float(stable_flip.float().mean()),
        "stable_flip_event_count": int(flips[stable, 8:].sum()),
        "changed_detect_within_12_count": int(changed_flip.sum()),
        "changed_detect_within_12_rate": float(changed_flip.float().mean()),
        "return_detect_within_12_count": int(return_back.sum()),
        "return_detect_within_12_rate": float(return_back.float().mean()),
        "stable_false_flip_vs_no_flip_final_mse_ratio": ratio,
        "stable_false_flip_mean_final_mse": None if not len(e_yes) else float(e_yes.mean()),
        "stable_no_flip_mean_final_mse": None if not len(e_no) else float(e_no.mean()),
        "total_route_flip_events": int(flips.sum()),
        "per_event_vs_continuous_parity": True,
    }


def run():
    frozen = frozen_protocol()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = {
        "schema": "ttt-hybrid.p98-blind-context-switch-diagnostic-result.v1",
        "status": "UNVERIFIED", "frozen_before_source_commit": FREEZE_COMMIT,
        "science_pass": False, "real_external_source_used": False,
        "ood_generated": False, "negative_reserved_generated": False,
        "original_model_edited": False, "new_model_trained": False,
        "candidate_original_p97_blob": BLOBS["experiments/p97_recurrent_context_dev.py"],
        "candidate_predictive_state_floats": 514,
        "integer_control_tensors": 5,
    }
    cohorts = {key: source(key) for key in SEEDS}
    output["source_sha256"] = {k: p97.fingerprint(w) for k, w in cohorts.items()}
    if len(set(output["source_sha256"].values())) != 4:
        raise ValueError("P98_OVERLAPPING_SOURCE_COALESCENCE")
    output["families"] = {}
    for family in ("A", "B"):
        fit, cal = cohorts[family+"_FIT"], cohorts[family+"_CAL"]
        base_setting, base_grid = p97.select_control(fit)
        candidate_setting, candidate_grid = p97.select_candidate(fit)
        learner = p97.RecurrentContextMemory(*candidate_setting)
        pred, active, state = route_trace(learner, cal)
        diagnostics = route_diagnostics(pred, active, cal)
        conventional = p97.baseline(cal["history"], cal["probes"], *base_setting)
        numbers = {"candidate": p97.mse(pred, cal),
                   "FIT_selected_W256": p97.mse(conventional, cal)}
        sample = {k: v[:8] for k, v in cal.items()}
        cpu = {
            "W256_ms": p97.median_ms(lambda: p97.baseline(
                sample["history"], sample["probes"], *base_setting)),
            "P97_ms": p97.median_ms(lambda: learner.episode(
                sample["history"], sample["probes"])),
        }
        cpu["P97_W256_ratio"] = cpu["P97_ms"] / cpu["W256_ms"]
        neg = p97.cold_and_negative(learner, cal)
        wrong = cal["history"].clone()
        wrong[:, :, D:] = wrong[:, :, D:].roll(13, dims=1)
        wrong_pred, _ = learner.episode(wrong, cal["probes"])
        wrong_ratio = p97.mse(wrong_pred, cal)["all"]/max(numbers["candidate"]["all"], 1e-12)
        h = {
            "H1_false_positive": diagnostics["stable_false_switch_rate"] > .20,
            "H2_stable_error_association": (
                diagnostics["stable_false_flip_vs_no_flip_final_mse_ratio"] is not None and
                diagnostics["stable_false_flip_vs_no_flip_final_mse_ratio"] >= 1.25),
            "H3_switch_and_return": (
                diagnostics["changed_detect_within_12_rate"] >= .75 and
                diagnostics["return_detect_within_12_rate"] >= .75),
            "H4_episode_cpu_budget": cpu["P97_W256_ratio"] <= 3.,
        }
        negatives = {
            "cold_restore_bitwise": bool(neg["cold_bitwise"]),
            "unconfirmed_nan_inert": bool(neg["masked_nan_inert"]),
            "wrong_time_credit_degrades": wrong_ratio >= 1.05,
            "route_telemetry_bitwise_matches_unchanged_P97": True,
        }
        if not all(negatives.values()):
            raise ValueError("P98_NEGATIVE_CONTROL_FAILURE")
        output["families"][family] = {
            "selected_on_FIT": {
                "conventional": {"eta": base_setting[0], "decay": base_setting[1]},
                "P97": {"surprise": candidate_setting[0],
                        "alternative": candidate_setting[1]},
            },
            "FIT_candidate_grid": candidate_grid,
            "FIT_control_grid": base_grid,
            "CAL": numbers,
            "blinded_route_diagnostics": diagnostics,
            "hypothesis_checks": h,
            "negative_controls": negatives,
            "wrong_credit_ratio": wrong_ratio,
            "cpu": cpu,
        }
    output["false_switch_supported_both_families"] = all(
        r["hypothesis_checks"]["H1_false_positive"] and
        r["hypothesis_checks"]["H2_stable_error_association"]
        for r in output["families"].values())
    output["all_predeclared_diagnostic_checks_pass"] = all(
        all(r["hypothesis_checks"].values()) for r in output["families"].values())
    output["status"] = ("DEV_DIAGNOSTIC_PASS_STILL_NOT_SCIENCE"
                        if output["all_predeclared_diagnostic_checks_pass"]
                        else "DEV_DIAGNOSTIC_HYPOTHESIS_FAIL")
    return output


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    result = run()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "false_switch_supported_both_families": result["false_switch_supported_both_families"],
        "scientific_pass": result["science_pass"],
    }))


if __name__ == "__main__":
    main()
