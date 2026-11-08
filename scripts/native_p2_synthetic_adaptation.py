"""P2 bounded synthetic fast-memory adaptation diagnostic (NOT scientific first-use).

Every trial starts with the same measured analog sensor, and action effects
reverse after 40 of 80 independent trials. This deliberately easy analytic
fixture has no S20/S26/S27 world generator, scientific seeds or heldout data.
Its purpose is to falsify basic online correction, reversal recovery and
a deliberately wrong-feedback negative without claiming generalization.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from native_continuous_state import NativeContinuousHybrid, ContinuousState
from scripts.native_prequential_gate import NativeCausalEpisode

MODEL_INITIALIZATIONS = tuple(range(8))  # synthetic model initialization, NOT world or science seeds
TRIALS = 80
REVERSE_AT = 40
POST_EARLY = (40, 48)
POST_LATE = (56, 80)
EFFECT_GAIN = 0.45
SENSOR_OFFSET = 0.04
SIGN_THRESHOLD = 0.05
FAST_RATE = 0.25
EXPECTED_SHA_ANCESTOR = "706658b5038c11642829868b0f292ac2fe4452e0"
ARMS = ("factual_fast_memory", "no_feedback_same_model", "sign_inverted_feedback_negative")


def analytic_consequence(action: torch.Tensor, trial: int) -> torch.Tensor:
    """A predetermined algebraic observation, not a generated science world."""
    if not 0 <= trial < TRIALS:
        raise ValueError("P2_TRIAL_OUT_OF_RANGE")
    if not isinstance(action, torch.Tensor) or action.shape != (1, 1):
        raise ValueError("P2_INVALID_ACTION_SHAPE")
    gain = EFFECT_GAIN if trial < REVERSE_AT else -EFFECT_GAIN
    return gain * action + SENSOR_OFFSET


def _copy_state(state: ContinuousState) -> ContinuousState:
    return ContinuousState(state.recurrent.detach().clone(),
                           state.fast_weights.detach().clone(),
                           state.history.detach().clone())


def _cold_restore_exact(model: NativeContinuousHybrid, state: ContinuousState) -> bool:
    data = io.BytesIO()
    torch.save({"model": model.state_dict(), "state": {
        "recurrent": state.recurrent, "fast_weights": state.fast_weights,
        "history": state.history}}, data)
    data.seek(0)
    payload = torch.load(data, map_location="cpu", weights_only=True)
    restored = NativeContinuousHybrid(1, 1, width=12, history_limit=4,
                                      fast_rate=FAST_RATE).eval()
    restored.load_state_dict(payload["model"], strict=True)
    state_copy = ContinuousState(**payload["state"])
    sensor = torch.zeros((1, 1), dtype=torch.float32)
    action = torch.ones((1, 1), dtype=torch.float32)
    dt = torch.tensor([[0.2]], dtype=torch.float32)
    expected = model.step(sensor, action, dt, _copy_state(state))[0]
    observed = restored.step(sensor, action, dt, state_copy)[0]
    return bool(torch.equal(expected, observed))


def run_one(initialization: int) -> dict:
    if type(initialization) is not int or initialization not in MODEL_INITIALIZATIONS:
        raise ValueError("P2_UNREGISTERED_ENGINEERING_INITIALIZATION")
    torch.manual_seed(initialization)
    model = NativeContinuousHybrid(1, 1, width=12, history_limit=4,
                                   fast_rate=FAST_RATE).eval()
    factual = NativeCausalEpisode(model)
    controls: dict[str, ContinuousState | None] = {
        "no_feedback_same_model": None,
        "sign_inverted_feedback_negative": None,
    }
    sq_errors = {arm: [] for arm in ARMS}
    paired_effects = {arm: [] for arm in ARMS}
    cold_restored = False
    with torch.no_grad():
        for t in range(TRIALS):
            # Independent trials with a reset measured input. Memory remains
            # intentionally continuous across the same recurring instrument.
            action = torch.tensor([[1.0 if t % 2 else -1.0]], dtype=torch.float32)
            alternate = -action
            sensor = torch.zeros((1, 1), dtype=torch.float32)
            dt = torch.tensor([[0.2]], dtype=torch.float32)
            outcome = analytic_consequence(action, t)
            ticket, counterfactual = factual.predict_pair(sensor, action, alternate, dt)
            predictions = {ARMS[0]: ticket.prediction}
            effects = {
                ARMS[0]: ((ticket.prediction - counterfactual) if action.item() > 0
                          else (counterfactual - ticket.prediction)).item()
            }
            new_states = {}
            for arm in controls:
                prior = controls[arm]
                pred, after_read = model.step(sensor, action, dt, prior)
                alt, _ = model.step(sensor, alternate, dt, prior)
                predictions[arm] = pred
                effects[arm] = ((pred - alt) if action.item() > 0
                                else (alt - pred)).item()
                new_states[arm] = after_read
            # Score ALL pre-outcome predictions before any memory change.
            for arm in ARMS:
                sq_errors[arm].append(float((predictions[arm] - outcome).square().mean()))
                paired_effects[arm].append(effects[arm])

            # Only true observed factual consequence enters the P0 episode.
            factual.observe(ticket, outcome, factual=True)
            controls["no_feedback_same_model"] = new_states["no_feedback_same_model"]
            # Deliberately WRONG-feedback control: not reported as factual,
            # never passed into the factual-only episode API.
            controls["sign_inverted_feedback_negative"] = model.learn_outcome(
                new_states["sign_inverted_feedback_negative"], action,
                predictions["sign_inverted_feedback_negative"], -outcome)
            if t == 56:
                cold_restored = _cold_restore_exact(model, factual.snapshot())

    def mean_window(arm: str, start: int, end: int) -> float:
        return sum(sq_errors[arm][start:end]) / (end-start)

    def sign_stats(arm: str, start: int, end: int) -> dict:
        effects = paired_effects[arm][start:end]
        eligible = [e for e in effects if abs(e) >= SIGN_THRESHOLD]
        correct = sum(1 for e in eligible if e < 0)  # post-reversal true effect = -0.9
        return {"correct": correct, "eligible": len(eligible),
                "coverage": len(eligible)/len(effects),
                "accuracy": correct/len(eligible) if eligible else 0.0}

    late = {arm: mean_window(arm, *POST_LATE) for arm in ARMS}
    early = {arm: mean_window(arm, *POST_EARLY) for arm in ARMS}
    effect_stats = sign_stats(ARMS[0], *POST_LATE)
    checks = {
        "late_factual_better_than_no_feedback_by_90pct": late[ARMS[0]] <= .1 * late[ARMS[1]],
        "late_factual_better_than_wrong_feedback_by_90pct": late[ARMS[0]] <= .1 * late[ARMS[2]],
        "late_factual_action_effect_sign_accuracy_at_least_95pct": effect_stats["accuracy"] >= .95,
        "late_factual_action_effect_coverage_at_least_95pct": effect_stats["coverage"] >= .95,
        "early_reversal_error_spike_is_preserved": early[ARMS[0]] > early[ARMS[1]],
        "fast_memory_cold_restore_bit_identical": cold_restored,
    }
    return {
        "model_initialization": initialization,
        "status": "PASS_SYNTHETIC_ADAPTATION_API_ONLY" if all(checks.values())
                  else "FAIL_SYNTHETIC_ADAPTATION_API_ONLY",
        "controls_same_model_weights": True,
        "prequential_prediction_before_factual_observation": True,
        "alternate_consequence_not_used_for_learning": True,
        "after_reversal_early_mse": early,
        "after_reversal_late_mse": late,
        "post_late_paired_action_effect": effect_stats,
        "checks": checks,
    }


def run() -> dict:
    if os.getenv("NATIVE_P2_SYNTHETIC_ONLY", "1") != "1":
        raise RuntimeError("P2_SCIENCE_FIRSTUSE_FORBIDDEN")
    torch.set_num_threads(1)
    start = time.perf_counter()
    records = [run_one(i) for i in MODEL_INITIALIZATIONS]
    complete = len(records) == len(MODEL_INITIALIZATIONS)
    success = complete and all(r["status"] == "PASS_SYNTHETIC_ADAPTATION_API_ONLY"
                               for r in records)
    fixture = {
        "role": "REPEATED_ANALYTIC_INSTRUMENT_NO_SCIENCE_SEED",
        "n_trials": TRIALS, "reversal_trial": REVERSE_AT,
        "gain_before": EFFECT_GAIN, "gain_after": -EFFECT_GAIN,
        "sensor_offset": SENSOR_OFFSET,
        "action_alternating": [-1.0, 1.0],
        "fixed_measured_sensor": 0.0,
        "dt": 0.2,
        "after_reversal_early": list(POST_EARLY),
        "after_reversal_late": list(POST_LATE),
        "same_continuous_model_all_arms": True
    }
    return {
        "schema": "ttt-hybrid.native-p2.synthetic-factual-reversal.v1",
        "status": "PASS_SYNTHETIC_ENGINEERING_ONLY" if success
                  else "FAIL_SYNTHETIC_ENGINEERING_ONLY",
        "qualification": "ENGINEERING_NOT_SCIENCE",
        "science_first_use": "NOT_RUN",
        "scientific_seeds_consumed": [],
        "worlds_constructed": False,
        "model_fit_updates": 0,
        "scientific_transfer": "NOT_TESTED",
        "frozen_model_weights": True,
        "online_feedback": "FACTUAL_ONLY_IN_P0_ARM",
        "limitation": "Synthetic algebraic reset-trials, tuned from prior local DEV; not prospective unseen environments, real-world chronology or trained conventional controls.",
        "torch_version": torch.__version__,
        "fixture": fixture,
        "records": records,
        "whole_process_wall_seconds": time.perf_counter() - start
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "record_count": len(report["records"]),
        "all_post_reversal_late_mse": [
            {"init": r["model_initialization"],
             "late": r["after_reversal_late_mse"],
             "sign": r["post_late_paired_action_effect"]}
            for r in report["records"]],
        "whole_process_wall_seconds": report["whole_process_wall_seconds"],
        "science_first_use": report["science_first_use"]
    }, sort_keys=True))
    return 0 if report["status"] == "PASS_SYNTHETIC_ENGINEERING_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
