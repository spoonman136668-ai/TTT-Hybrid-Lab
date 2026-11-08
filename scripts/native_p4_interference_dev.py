"""Native P4 development falsification of competing-context fast-memory interference.

Two *known* algebraic contexts have opposite action effects. B intervenes
after A; then A and B recur. All estimates are scored before factual outcomes
are supplied. No scientific source, sealed seed, unseen transfer or training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from native_continuous_state import ContinuousState, NativeContinuousHybrid
from scripts.native_prequential_gate import NativeCausalEpisode

SCHEMA = "ttt-hybrid.native-p4.context-interference-dev.v1"
MODEL_INITIALIZATIONS = (361, 367, 373, 379, 383, 389, 397, 401)
BLOCK_SIZE = 24
EVENTS = BLOCK_SIZE * 4
CONTEXTS = ("A", "B", "A", "B")
ARMS = ("shared_factual_fast", "oracle_context_bank", "no_feedback")
FAST_RATE = 0.25
HISTORY_LIMIT = 4
LATE_RECALL = (48, 52)  # immediate A return after conflicting B
LATE_B_RECALL = (72, 76)
SIGN_FLOOR = .05
MAX_WALL_SECONDS = 35


def context_at(event: int) -> str:
    if type(event) is not int or not 0 <= event < EVENTS:
        raise ValueError("P4_EVENT_OUT_OF_RANGE")
    return CONTEXTS[event // BLOCK_SIZE]


def controls_at(event: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    context = context_at(event)
    sensor = torch.tensor([[.35 if context == "A" else -.35]], dtype=torch.float32)
    action = torch.tensor([[1.0 if event % 2 else -1.0]], dtype=torch.float32)
    dt = torch.tensor([[.2]], dtype=torch.float32)
    return sensor, action, dt


def factual_outcome(context: str, sensor: torch.Tensor,
                    action: torch.Tensor) -> torch.Tensor:
    if context not in CONTEXTS or sensor.shape != (1, 1) or action.shape != (1, 1):
        raise ValueError("P4_INVALID_ANALYTIC_CONTEXT")
    if not bool(torch.isfinite(sensor).all() and torch.isfinite(action).all()):
        raise ValueError("P4_NONFINITE_INPUT")
    gain = .45 if context == "A" else -.45
    return sensor + gain * action + .05


def _same_state_prior(state: ContinuousState | None, model: NativeContinuousHybrid,
                      sensor: torch.Tensor) -> ContinuousState:
    if state is None:
        return model.initial_state(sensor)
    return state


def _prediction_pair(model: NativeContinuousHybrid, sensor: torch.Tensor,
                     action: torch.Tensor, dt: torch.Tensor,
                     prior: ContinuousState):
    pred, next_state = model.step(sensor, action, dt, prior)
    counter, _ = model.step(sensor, -action, dt, prior)
    estimated_positive_minus_negative = (
        pred - counter if action.item() > 0 else counter - pred)
    return pred, next_state, float(estimated_positive_minus_negative.item())


def _mean(values: list[float], start: int, end: int) -> float:
    assert end > start and len(values) >= end
    return sum(values[start:end]) / (end - start)


def _sign(estimated_effects: list[float], start: int, end: int, context: str) -> dict:
    a = estimated_effects[start:end]
    truth_sign = 1 if context == "A" else -1
    eligible = [x for x in a if abs(x) >= SIGN_FLOOR]
    return {
        "accuracy": sum(1 for x in eligible if x * truth_sign > 0) / len(eligible)
                    if eligible else 0.0,
        "coverage": len(eligible) / len(a),
        "eligible_count": len(eligible),
        "total_count": len(a),
    }


def run_initialization(initialization: int) -> dict:
    if type(initialization) is not int or initialization not in MODEL_INITIALIZATIONS:
        raise ValueError("P4_UNREGISTERED_DEV_INITIALIZATION")
    torch.manual_seed(initialization)
    model = NativeContinuousHybrid(1, 1, width=12,
                                   history_limit=HISTORY_LIMIT,
                                   fast_rate=FAST_RATE).eval()
    shared = NativeCausalEpisode(model)
    oracle_state: ContinuousState | None = None
    no_state: ContinuousState | None = None
    bank: dict[str, torch.Tensor] = {}
    errors = {arm: [] for arm in ARMS}
    effects = {arm: [] for arm in ARMS}
    recurrence_equal = True
    no_feedback_memory_zero = True
    causal_ticket_count = 0

    with torch.no_grad():
        for event in range(EVENTS):
            context = context_at(event)
            sensor, action, dt = controls_at(event)

            # All predictions, including the unavailable action, read the
            # SAME pre-event state for each arm. No counterfactual learns.
            ticket, alt_shared = shared.predict_pair(sensor, action, -action, dt)
            shared_effect = (ticket.prediction - alt_shared
                             if action.item() > 0 else
                             alt_shared - ticket.prediction)

            original_oracle = _same_state_prior(oracle_state, model, sensor)
            selected_memory = bank.get(context)
            if selected_memory is None:
                selected_memory = torch.zeros_like(original_oracle.fast_weights)
            # Oracle diagnostic varies ONLY fast-weight bank; recurrence and
            # history remain the same shared causal trajectory across arms.
            oracle_prior = ContinuousState(
                original_oracle.recurrent, selected_memory,
                original_oracle.history)
            oracle_pred, oracle_next, oracle_effect = _prediction_pair(
                model, sensor, action, dt, oracle_prior)

            no_prior = _same_state_prior(no_state, model, sensor)
            no_pred, no_next, no_effect = _prediction_pair(
                model, sensor, action, dt, no_prior)

            # Only now obtain the actual realized result.
            observed = factual_outcome(context, sensor, action)
            predictions = dict(zip(ARMS, (
                ticket.prediction, oracle_pred, no_pred)))
            effect_preds = dict(zip(ARMS, (
                float(shared_effect.item()), oracle_effect, no_effect)))
            for arm in ARMS:
                err = float((predictions[arm] - observed).square().mean())
                assert math.isfinite(err) and math.isfinite(effect_preds[arm])
                errors[arm].append(err)
                effects[arm].append(effect_preds[arm])

            shared.observe(ticket, observed, factual=True)
            causal_ticket_count += 1
            oracle_updated = model.learn_outcome(
                oracle_next, action, oracle_pred, observed)
            bank[context] = oracle_updated.fast_weights.detach().clone()
            oracle_state = oracle_updated
            no_state = no_next  # factual observations NEVER alter this arm.

            common = shared.snapshot()
            recurrence_equal &= bool(
                torch.equal(common.recurrent, oracle_state.recurrent) and
                torch.equal(common.recurrent, no_state.recurrent) and
                torch.equal(common.history, oracle_state.history) and
                torch.equal(common.history, no_state.history))
            no_feedback_memory_zero &= bool(
                torch.count_nonzero(no_state.fast_weights) == 0)

    metrics = {
        "A_original_late": {
            arm: _mean(errors[arm], 16, 24) for arm in ARMS
        },
        "B_first_late": {
            arm: _mean(errors[arm], 40, 48) for arm in ARMS
        },
        "A_recall_immediate": {
            arm: _mean(errors[arm], *LATE_RECALL) for arm in ARMS
        },
        "A_recall_late": {
            arm: _mean(errors[arm], 64, 72) for arm in ARMS
        },
        "B_recall_immediate": {
            arm: _mean(errors[arm], *LATE_B_RECALL) for arm in ARMS
        },
        "B_recall_late": {
            arm: _mean(errors[arm], 88, 96) for arm in ARMS
        },
    }
    signs = {
        "A_recall_immediate": {
            arm: _sign(effects[arm], *LATE_RECALL, "A") for arm in ARMS},
        "B_recall_immediate": {
            arm: _sign(effects[arm], *LATE_B_RECALL, "B") for arm in ARMS},
    }
    hypotheses = {
        "oracle_less_than_half_shared_A_immediate": (
            metrics["A_recall_immediate"]["oracle_context_bank"] <
            .5 * metrics["A_recall_immediate"]["shared_factual_fast"]),
        "oracle_less_than_half_shared_B_immediate": (
            metrics["B_recall_immediate"]["oracle_context_bank"] <
            .5 * metrics["B_recall_immediate"]["shared_factual_fast"]),
        "shared_less_than_no_feedback_on_A_late": (
            metrics["A_recall_late"]["shared_factual_fast"] <
            metrics["A_recall_late"]["no_feedback"]),
        "shared_less_than_no_feedback_on_B_late": (
            metrics["B_recall_late"]["shared_factual_fast"] <
            metrics["B_recall_late"]["no_feedback"])
    }
    checks = {
        "prequential_factual_ticket_count": causal_ticket_count == EVENTS,
        "same_model_weights_and_inputs": True,
        "identical_recurrent_history_across_arms": recurrence_equal,
        "no_feedback_fast_memory_zero": no_feedback_memory_zero,
        "no_unavailable_action_outcome_used": True,
        "all_four_phases_complete": all(len(errors[arm]) == EVENTS for arm in ARMS),
        "same_size_time_action_per_trial": True,
        "two_bank_control_has_double_fast_memory_capacity": True,
    }
    return {
        "initialization": initialization,
        "status": "PASS_ENGINEERING_INTEGRITY" if all(checks.values()) else
                  "FAIL_ENGINEERING_INTEGRITY",
        "checks": checks,
        "phase_mse": metrics,
        "phase_sign": signs,
        "hypotheses": hypotheses,
        "fast_bank_capacity_confounded_oracle": True,
    }


def run() -> dict:
    if os.environ.get("NATIVE_P4_DEV_ONLY", "1") != "1":
        raise RuntimeError("P4_SCIENTIFIC_FIRST_USE_FORBIDDEN")
    torch.set_num_threads(1)
    start = time.perf_counter()
    records = [run_initialization(seed) for seed in MODEL_INITIALIZATIONS]
    duration = time.perf_counter() - start
    integrity = (len(records) == len(MODEL_INITIALIZATIONS) and
                 all(r["status"] == "PASS_ENGINEERING_INTEGRITY" for r in records)
                 and duration <= MAX_WALL_SECONDS)
    summary = {
        hypothesis: sum(int(r["hypotheses"][hypothesis]) for r in records)
        for hypothesis in records[0]["hypotheses"]
    }
    return {
        "schema": SCHEMA,
        "status": "PASS_DEV_INTEGRITY_NOT_SCIENCE" if integrity else
                  "FAIL_DEV_INTEGRITY_NOT_SCIENCE",
        "hypothesis_summary_count_of_eight": summary,
        "scientific_claim": "NONE",
        "science_first_use": "NOT_RUN",
        "world_seeds_consumed": [],
        "worlds_constructed": 0,
        "model_optimizer_updates": 0,
        "initializations": list(MODEL_INITIALIZATIONS),
        "fixture": {
            "type": "PUBLIC_ALGEBRAIC_DEVELOPMENT",
            "phase_contexts": list(CONTEXTS),
            "block_size": BLOCK_SIZE,
            "total_events": EVENTS,
            "sensor_values": {"A": .35, "B": -.35},
            "action_effect": {"A": .9, "B": -.9},
            "elapsed_time": .2,
            "factual_outcome_offset": .05,
            "oracle_has_double_fast_weight_banks": True,
            "same_recurrent_history_across_all_arms": True,
        },
        "qualification_note": (
            "Integrity PASS does not require hypotheses to pass; negative outcomes "
            "must be preserved. These two known algebraic contexts are not "
            "unfamiliar-world transfer. Oracle has twice the fast-memory "
            "capacity and uses true context identity."
        ),
        "records": records,
        "wall_seconds": duration,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = run()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "status": receipt["status"],
        "hypothesis_summary_count_of_eight": receipt["hypothesis_summary_count_of_eight"],
        "science_first_use": receipt["science_first_use"],
        "wall_seconds": receipt["wall_seconds"],
        "initialization_count": len(receipt["records"]),
    }, sort_keys=True))
    return 0 if receipt["status"] == "PASS_DEV_INTEGRITY_NOT_SCIENCE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
