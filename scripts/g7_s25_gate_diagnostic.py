"""G7-S25 development-only mechanism diagnostic: observable trigger vs forced budget.

Measures whether the *existing, UNTRAINED* numeric contradiction gate responds
to signed action-contrast conflicts in engineering worlds. No optimization,
baseline fit, official TTT, scientific fit/validation/heldout seeds, threshold
search, CUDA, installed packages, scientific first-use, or performance claims.

This is a falsification-oriented pre-science sanity diagnostic only.
Untrained random slow features are not representative of a competent trained
baseline. A null here cannot establish the scientific null after training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from collections import defaultdict

import torch

import g7_s25_contradiction_gate as gate
import g7_s25_cost_ledger as ledger
import g7_s25_numeric_ttt as numeric
import g7_s25_paired_interventions as physics

DEV_SEEDS = (3101, 3109, 3119)
DEV_REGIMES = ("source_stable", "source_signflip",
               "sine_signflip", "rational_signflip")
COMPARE_MODES = ("periodic", "random", "surprise", "contradiction")
DIAGNOSTIC_BATCH = 4
DIAGNOSTIC_STEPS = 32
SWITCH = 20
MAX_DIAGNOSTIC_SECONDS = 170
RUN_SCHEMA = "ttt-hybrid.g7-s25.untrained-gate-mechanism-diagnostic.v1"


def check(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def summarize(rows: list[dict]) -> dict:
    """Return a descriptive pre-first-use null without changing thresholds."""
    check(rows and all(r.get("mode") in COMPARE_MODES for r in rows),
          "DIAGNOSTIC_EMPTY_OR_UNKNOWN_MODE")
    by_mode = defaultdict(list)
    for row in rows:
        by_mode[row["mode"]].append(row)
    for mode in COMPARE_MODES:
        check(len(by_mode[mode]) == len(DEV_SEEDS) * len(DEV_REGIMES),
              "DIAGNOSTIC_MISSING_SEED_OR_REGIME")
    contradiction = by_mode["contradiction"]
    nonforced = sum(row["selected_trigger_steps"] for row in contradiction)
    post_switch = sum(
        row["post_switch_selected_trigger_steps"]
        for row in contradiction if row["regime"].endswith("signflip")
    )
    # This is not a post-hoc science threshold; the diagnostic reports the
    # raw evidence rather than tuning or accepting a mechanism.
    return {
        "schema": RUN_SCHEMA,
        "status": (
            "UNTRAINED_DIAGNOSTIC_TRIGGER_NULL"
            if nonforced == 0 or post_switch == 0
            else "UNTRAINED_DIAGNOSTIC_TRIGGER_EXERCISED_NOT_SCIENCE"
        ),
        "untrained_contradiction_selected_trigger_world_steps": nonforced,
        "untrained_signflip_post_switch_selected_trigger_world_steps": post_switch,
        "contradiction_forced_budget_world_steps": sum(
            row["forced_budget_updates"] for row in contradiction),
        "total_equal_budget_contradiction_world_updates": sum(
            row["real_fast_weight_update_worlds"] for row in contradiction),
        "matched_worlds_per_seed_regime": DIAGNOSTIC_BATCH,
        "diagnostic_seed_regime_pairs": len(DEV_SEEDS) * len(DEV_REGIMES),
        "training_or_hyperparameter_search": "NOT_RUN",
        "scientific_first_use": "NOT_RUN",
        "science": "NOT_SCIENTIFIC_PASS",
    }


@torch.no_grad()
def run_diagnostic(*, batch: int = DIAGNOSTIC_BATCH,
                   seeds: tuple[int, ...] = DEV_SEEDS,
                   regimes: tuple[str, ...] = DEV_REGIMES,
                   modes: tuple[str, ...] = COMPARE_MODES,
                   max_seconds: int = MAX_DIAGNOSTIC_SECONDS) -> dict:
    check(type(batch) is int and 1 <= batch <= DIAGNOSTIC_BATCH,
          "DEVELOPMENT_BATCH_CAP")
    check(seeds and len(set(seeds)) == len(seeds) and
          set(seeds).issubset(DEV_SEEDS), "DEVELOPMENT_SEEDS_ONLY")
    check(regimes and len(set(regimes)) == len(regimes) and
          set(regimes).issubset(DEV_REGIMES), "DEVELOPMENT_REGIMES_ONLY")
    check(modes and len(set(modes)) == len(modes) and
          set(modes).issubset(COMPARE_MODES), "DEVELOPMENT_MODES_ONLY")
    check(1 <= max_seconds <= MAX_DIAGNOSTIC_SECONDS,
          "DIAGNOSTIC_TIME_LIMIT")
    torch.set_num_threads(1)
    torch.manual_seed(3101)
    network = numeric.NumericFastWeightTTT(
        numeric.ConventionalGRU()).eval()
    check(all(not t.is_cuda for t in network.parameters()),
          "DIAGNOSTIC_CPU_REQUIRED")
    started = time.monotonic()
    rows = []
    keyed_streams = {}
    keyed_checkpoints = {}
    for seed in seeds:
        for regime in regimes:
            for mode in modes:
                check(time.monotonic() - started < max_seconds,
                      "DIAGNOSTIC_TIMEOUT")
                environment = physics.PairedS20World(seed,batch,regime)
                episode = numeric.PrequentialTTTEpisode(
                    network, mode=mode, seed=seed, batch=batch,
                    reference=torch.zeros(batch,2))
                record = ledger.EpisodeWorkLedger(
                    episode,source_seed=seed)
                shadows = [
                    gate.PrequentialGate(
                        mode,engineering_seed=seed+101*j)
                    for j in range(batch)]
                triggered = 0
                selected_trigger = 0
                forced = 0
                exhausted = 0
                post_selected = 0
                post_triggered = 0
                post_forced = 0
                observed_contradictions = 0
                observed_post_contradictions = 0
                selected_by_step = []
                for t in range(DIAGNOSTIC_STEPS):
                    sample = environment.emit()
                    predictions = record.step(sample)
                    mask = episode.update_masks[-1]
                    for j in range(batch):
                        check(time.monotonic() - started < max_seconds,
                              "DIAGNOSTIC_TIMEOUT")
                        actual_action = int(
                            sample.realized_action_index[j].item())
                        residual = float((
                            sample.observed_pair[j,actual_action] -
                            predictions[j,actual_action]).item())
                        decision = shadows[j].observe(
                            t,
                            predicted_minus=float(
                                predictions[j,0].item()),
                            predicted_plus=float(
                                predictions[j,1].item()),
                            observed_minus=float(
                                sample.observed_pair[j,0].item()),
                            observed_plus=float(
                                sample.observed_pair[j,1].item()),
                            primary_residual=residual)
                        check(decision.update == bool(mask[j]),
                              "DIAGNOSTIC_SHADOW_GATE_DIVERGENCE")
                        check(decision.real_updates_used ==
                              episode.gates[j].updates_used,
                              "DIAGNOSTIC_UPDATE_COUNT_DRIFT")
                        trigger = int(decision.trigger)
                        selected = int(
                            decision.reason == "trigger")
                        force = int(
                            decision.reason == "forced_budget")
                        exhaustion = int(
                            decision.reason == "budget_exhausted")
                        triggered += trigger
                        selected_trigger += selected
                        forced += force
                        exhausted += exhaustion
                        observed_contradictions += int(
                            decision.contradiction_evidence_count >= 2)
                        if t >= SWITCH:
                            post_triggered += trigger
                            post_selected += selected
                            post_forced += force
                            observed_post_contradictions += int(
                                decision.contradiction_evidence_count >= 2)
                    selected_by_step.append(
                        int(mask.sum().item()))
                for shadow in shadows:
                    check(shadow.close() == gate.MATCHED_UPDATES,
                          "DIAGNOSTIC_SHADOW_BUDGET_DRIFT")
                counts = record.close()
                world_counts = environment.close()
                check(counts["real_fast_weight_update_worlds"] ==
                      batch * gate.MATCHED_UPDATES,
                      "DIAGNOSTIC_REAL_UPDATE_BUDGET_MISMATCH")
                check(world_counts["paired_observations"] ==
                      batch * DIAGNOSTIC_STEPS * 2,
                      "DIAGNOSTIC_PAIRED_OBSERVATION_MISMATCH")
                check(triggered >= selected_trigger and
                      sum(selected_by_step) ==
                      selected_trigger + forced,
                      "DIAGNOSTIC_TRIGGER_ACCOUNTING_MISMATCH")
                key=(seed,regime)
                old_stream = keyed_streams.setdefault(
                    key,counts["input_stream_sha256"])
                old_checkpoint = keyed_checkpoints.setdefault(
                    key,counts["model_checkpoint_sha256"])
                check(old_stream == counts["input_stream_sha256"],
                      "DIAGNOSTIC_UNMATCHED_INFORMATION")
                check(old_checkpoint == counts["model_checkpoint_sha256"],
                      "DIAGNOSTIC_UNMATCHED_SLOW_CHECKPOINT")
                rows.append({
                    "seed": seed, "regime": regime, "mode": mode,
                    "worlds":batch,
                    "stream_sha256":counts["input_stream_sha256"],
                    "slow_checkpoint_sha256":
                        counts["model_checkpoint_sha256"],
                    "observed_action_pairs":
                        counts["paired_intervention_observations"],
                    "real_fast_weight_update_worlds":
                        counts["real_fast_weight_update_worlds"],
                    "trigger_world_steps":triggered,
                    "selected_trigger_steps":selected_trigger,
                    "forced_budget_updates":forced,
                    "budget_exhausted_triggers":exhausted,
                    "post_switch_trigger_world_steps":post_triggered,
                    "post_switch_selected_trigger_steps":post_selected,
                    "post_switch_forced_budget_steps":post_forced,
                    "persistent_contradiction_world_steps":
                        observed_contradictions,
                    "post_switch_persistent_contradiction_world_steps":
                        observed_post_contradictions,
                    "updates_by_time":selected_by_step,
                })
    partial = (len(seeds) != len(DEV_SEEDS) or
               len(regimes) != len(DEV_REGIMES) or
               len(modes) != len(COMPARE_MODES) or
               batch != DIAGNOSTIC_BATCH)
    if partial:
        summary={
            "status":"PASS_PARTIAL_DEVELOPMENT_MECHANICS_ONLY",
            "scientific_first_use":"NOT_RUN",
            "scientific_interpretation":"NOT_ALLOWED",
        }
    else:
        summary=summarize(rows)
    return {
        "schema":RUN_SCHEMA,
        "status":"PASS_DEVELOPMENT_MECHANISM_DIAGNOSTIC_EXECUTED",
        "summary":summary,
        "rows":rows,
        "untrained_slow_weights":True,
        "all_policy_arms_same_observed_stream":True,
        "source_seeds_used":list(seeds),
        "modes":list(modes),
        "regimes":list(regimes),
        "frozen_scientific_seeds_used":False,
        "baseline_fitting":"NOT_RUN",
        "scientific_first_use":"NOT_RUN",
        "heldout":"NOT_RUN",
        "gpu":"NOT_USED",
        "full_cross_platform_regression":"UNVERIFIED",
        "scientific_qualification":"NOT_SCIENTIFIC_PASS",
        "elapsed_wall_seconds":round(time.monotonic()-started,3),
    }


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--receipt",type=Path,required=True)
    parser.add_argument("--expected-sha",required=True)
    args=parser.parse_args()
    receipt=args.receipt.resolve()
    if (len(args.expected_sha)!=40 or
            any(c not in "0123456789abcdef" for c in args.expected_sha)):
        raise SystemExit("DIAGNOSTIC_INVALID_EXACT_SHA")
    # Stdlib source custody. The authorized workflow separately pins the
    # exact HEAD and file scope before invoking this read-only diagnostic.
    import subprocess
    root=Path(__file__).resolve().parents[1]
    p=subprocess.run(
        ["git","-C",str(root),"rev-parse","HEAD"],
        capture_output=True,text=True,timeout=12,check=False)
    if p.returncode or p.stdout.strip()!=args.expected_sha:
        raise SystemExit("DIAGNOSTIC_SOURCE_SHA_DRIFT")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("DIAGNOSTIC_CUDA_ISOLATION_REQUIRED")
    value=run_diagnostic()
    value["candidate_sha"]=args.expected_sha
    receipt.parent.mkdir(parents=True,exist_ok=True)
    receipt.write_text(
        json.dumps(value,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("G7_S25_MECHANISM_DIAGNOSTIC="+value["status"])
    print("G7_S25_GATE_MECHANISM="+value["summary"]["status"])
    print("G7_S25_SCIENCE=NOT_RUN")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
