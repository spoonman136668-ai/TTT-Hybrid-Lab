"""Equal-capacity fast-memory routing DEV diagnostic, not scientific first-use.

On the unchanged P4 A/B/A/B public analytic trace, compare the ORIGINAL
single memory with two physically disjoint half-width partitions in ONE
unchanged fast tensor. Routing sees only current measured sensor or factual
action; no hidden-world ID or future outcome. All three memory mechanisms
share the SAME fixed native model weights, history, recurrence and 12-scalar
fast-weight budget. Hypotheses are reported even if false.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F

from native_continuous_state import ContinuousState, NativeContinuousHybrid
from scripts.native_prequential_gate import NativeCausalEpisode
from scripts.native_p4_interference_dev import (
    MODEL_INITIALIZATIONS, EVENTS, ARMS as ORIGINAL_ARMS, FAST_RATE, HISTORY_LIMIT,
    context_at, controls_at, factual_outcome)

SCHEMA = "ttt-hybrid.native-p5.equal-capacity-context-routing-dev.v1"
ARMS = ("original_shared_factual", "sensor_sign_gated_equal_capacity",
        "action_sign_gated_equal_capacity", "no_feedback")
WIDTH = 12
HALF = WIDTH // 2
MAX_WALL_SECONDS = 50
EVALUATION_WINDOWS = {
    "A_first_late": (16, 24), "B_first_late": (40, 48),
    "A_revisit_immediate": (48, 52), "A_revisit_late": (64, 72),
    "B_revisit_immediate": (72, 76), "B_revisit_late": (88, 96),
}


def routed_mask(sensor: torch.Tensor, action: torch.Tensor,
                *, route: str) -> torch.Tensor:
    """Mask from OBSERVED input only. Both half partitions share 12 slots."""
    if route not in ("sensor_sign", "action_sign"):
        raise ValueError("P5_ROUTE_REQUIRED")
    if sensor.shape != (1, 1) or action.shape != (1, 1) or (
        not bool(torch.isfinite(sensor).all()) or
        not bool(torch.isfinite(action).all())):
        raise ValueError("P5_INVALID_OBSERVED_INPUT")
    if sensor.item() == 0 or action.item() == 0:
        raise ValueError("P5_AMBIGUOUS_ROUTE")
    observed = sensor.item() if route == "sensor_sign" else action.item()
    mask = torch.zeros((1, WIDTH), dtype=sensor.dtype, device=sensor.device)
    mask[:, :HALF] = 1 if observed > 0 else 0
    mask[:, HALF:] = 0 if observed > 0 else 1
    return mask


def route_key(model: NativeContinuousHybrid, sensor: torch.Tensor,
              action: torch.Tensor, route: str) -> torch.Tensor:
    key = model.action_encoder(action)
    mask = routed_mask(sensor, action, route=route)
    vector = key * mask
    if not bool((torch.linalg.vector_norm(vector, dim=-1) > 1e-8).all()):
        raise ValueError("P5_DEGENERATE_KEY")
    return F.normalize(vector, dim=-1)


def routed_predict(model: NativeContinuousHybrid, prior: ContinuousState,
                   sensor: torch.Tensor, action: torch.Tensor, dt: torch.Tensor,
                   route: str):
    """Use the original native core for recurrence, change only correction key."""
    zero_mem = ContinuousState(prior.recurrent,
                               torch.zeros_like(prior.fast_weights),
                               prior.history)
    base, next_state = model.step(sensor, action, dt, zero_mem)
    key = route_key(model, sensor, action, route)
    correction = torch.bmm(prior.fast_weights,
                           key.unsqueeze(-1)).squeeze(-1)
    pred = base + correction
    advanced = ContinuousState(next_state.recurrent,
                               prior.fast_weights.detach().clone(),
                               next_state.history)
    return pred, advanced, key


class RoutedEpisode:
    """One pending event, prequential reads, factual-only update.

    Engineering comparator only; does not change the original P0 API.
    """
    def __init__(self, model: NativeContinuousHybrid, route: str):
        if route not in ("sensor_sign", "action_sign"):
            raise ValueError("P5_ROUTE_REQUIRED")
        self.model = model
        self.route = route
        self.state: ContinuousState | None = None
        self.pending: tuple[torch.Tensor, torch.Tensor] | None = None
        self.events = 0

    def predict_pair(self, sensor: torch.Tensor, action: torch.Tensor,
                     dt: torch.Tensor):
        if self.pending is not None:
            raise RuntimeError("P5_PENDING_FUTURE_OUTCOME")
        with torch.no_grad():
            prior = self.state
            if prior is None:
                prior = self.model.initial_state(sensor)
            prediction, advanced, key = routed_predict(
                self.model, prior, sensor, action, dt, self.route)
            alternate, _, _ = routed_predict(
                self.model, prior, sensor, -action, dt, self.route)
            self.state = advanced
            self.pending = (prediction.detach().clone(), key.detach().clone())
            self.events += 1
            return prediction.detach().clone(), alternate.detach().clone()

    def observe(self, outcome: torch.Tensor, *, factual: bool):
        if self.pending is None:
            raise RuntimeError("P5_NO_PENDING_PREDICTION")
        if factual is not True:
            raise ValueError("P5_COUNTERFACTUAL_FEEDBACK_FORBIDDEN")
        if (outcome.shape != (1, 1) or outcome.dtype != torch.float32
            or not bool(torch.isfinite(outcome).all())):
            raise ValueError("P5_INVALID_REALIZED_OUTCOME")
        with torch.no_grad():
            estimate, key = self.pending
            residual = (outcome - estimate).unsqueeze(-1)
            weights = self.state.fast_weights + FAST_RATE * residual * key.unsqueeze(1)
            self.state = ContinuousState(self.state.recurrent, weights,
                                         self.state.history)
            self.pending = None


def _phase_mean(errors: list[float], window: tuple[int, int]) -> float:
    start, end = window
    return sum(errors[start:end]) / (end - start)


def _sign_stats(effects: list[float], window: tuple[int, int], truth_sign: int) -> dict:
    start, end = window
    samples = effects[start:end]
    eligible = [x for x in samples if abs(x) >= .05]
    return {"accuracy": (
        sum(1 for x in eligible if x*truth_sign > 0)/len(eligible)
        if eligible else 0.0),
        "coverage": len(eligible)/len(samples),
        "eligible": len(eligible),
        "total": len(samples)}


def run_initialization(initialization: int) -> dict:
    if type(initialization) is not int or initialization not in MODEL_INITIALIZATIONS:
        raise ValueError("P5_UNREGISTERED_ENGINEERING_INITIALIZATION")
    torch.manual_seed(initialization)
    model = NativeContinuousHybrid(1, 1, width=WIDTH,
                                   history_limit=HISTORY_LIMIT,
                                   fast_rate=FAST_RATE).eval()
    original = NativeCausalEpisode(model)
    routed = {
        ARMS[1]: RoutedEpisode(model, "sensor_sign"),
        ARMS[2]: RoutedEpisode(model, "action_sign"),
    }
    no_feedback_state: ContinuousState | None = None
    errors = {arm: [] for arm in ARMS}
    effects = {arm: [] for arm in ARMS}
    shared_recurrence_equal = True
    physical_fast_slots = True
    no_feedback_zero = True
    after_first_A_disjoint = True
    after_first_B_both_used = True

    with torch.no_grad():
        for event in range(EVENTS):
            sensor, action, dt = controls_at(event)
            context = context_at(event)
            # All four arms must estimate before the realized outcome exists.
            ticket, alt_orig = original.predict_pair(sensor, action, -action, dt)
            predictions = {ARMS[0]: ticket.prediction}
            alt_predictions = {ARMS[0]: alt_orig}
            for arm, episode in routed.items():
                predictions[arm], alt_predictions[arm] = episode.predict_pair(
                    sensor, action, dt)

            prior_no = (no_feedback_state if no_feedback_state is not None
                        else model.initial_state(sensor))
            predictions[ARMS[3]], next_no = model.step(
                sensor, action, dt, prior_no)
            alt_predictions[ARMS[3]], _ = model.step(
                sensor, -action, dt, prior_no)

            observed = factual_outcome(context, sensor, action)
            for arm in ARMS:
                pred = predictions[arm]
                effect = pred-alt_predictions[arm] if action.item()>0 else (
                    alt_predictions[arm]-pred)
                error = float((pred-observed).square().mean())
                assert math.isfinite(error) and math.isfinite(float(effect.item()))
                errors[arm].append(error)
                effects[arm].append(float(effect.item()))

            original.observe(ticket, observed, factual=True)
            for episode in routed.values():
                episode.observe(observed, factual=True)
            no_feedback_state = next_no

            orig_state = original.snapshot()
            for episode in routed.values():
                shared_recurrence_equal &= bool(
                    torch.equal(episode.state.recurrent, orig_state.recurrent)
                    and torch.equal(episode.state.history, orig_state.history))
                physical_fast_slots &= bool(
                    episode.state.fast_weights.shape ==
                    orig_state.fast_weights.shape == (1, 1, WIDTH))
            shared_recurrence_equal &= bool(
                torch.equal(no_feedback_state.recurrent, orig_state.recurrent)
                and torch.equal(no_feedback_state.history, orig_state.history))
            no_feedback_zero &= bool(
                torch.count_nonzero(no_feedback_state.fast_weights)==0)
            if event==23:
                w=routed[ARMS[1]].state.fast_weights
                after_first_A_disjoint &= bool(
                    torch.count_nonzero(w[..., :HALF])>0 and
                    torch.count_nonzero(w[..., HALF:])==0)
            if event==47:
                w=routed[ARMS[1]].state.fast_weights
                after_first_B_both_used &= bool(
                    torch.count_nonzero(w[..., :HALF])>0 and
                    torch.count_nonzero(w[..., HALF:])>0)

    phase_mse={phase:{
        arm: _phase_mean(errors[arm], window) for arm in ARMS
    } for phase, window in EVALUATION_WINDOWS.items()}
    phase_sign={
        phase:{arm:_sign_stats(effects[arm], EVALUATION_WINDOWS[phase],
                               1 if phase.startswith("A") else -1)
               for arm in ARMS}
        for phase in ("A_revisit_immediate","B_revisit_immediate")
    }
    hypotheses={
        "context_half_beats_shared_A_immediate": (
            phase_mse["A_revisit_immediate"][ARMS[1]] <
            phase_mse["A_revisit_immediate"][ARMS[0]]),
        "context_half_beats_shared_B_immediate": (
            phase_mse["B_revisit_immediate"][ARMS[1]] <
            phase_mse["B_revisit_immediate"][ARMS[0]]),
        "context_half_beats_action_half_A_immediate": (
            phase_mse["A_revisit_immediate"][ARMS[1]] <
            phase_mse["A_revisit_immediate"][ARMS[2]]),
        "context_half_beats_action_half_B_immediate": (
            phase_mse["B_revisit_immediate"][ARMS[1]] <
            phase_mse["B_revisit_immediate"][ARMS[2]]),
        "context_half_beats_no_feedback_both_late": all(
            phase_mse[ph][ARMS[1]] < phase_mse[ph][ARMS[3]]
            for ph in ("A_revisit_late","B_revisit_late")),
    }
    integrity={
        "all_events_scored_pre_outcome": all(len(errors[a])==EVENTS for a in ARMS),
        "original_P0_factual_ticket_count": original._event_index==EVENTS,
        "no_pending_context_or_action_episode": all(
            x.pending is None and x.events==EVENTS for x in routed.values()),
        "same_fixed_base_model_weights": True,
        "same_recurrent_and_history_across_arms": shared_recurrence_equal,
        "all_memory_arms_have_same_12_scalar_fast_capacity": physical_fast_slots,
        "first_A_block_writes_only_A_half": after_first_A_disjoint,
        "after_B_block_both_context_halves_used": after_first_B_both_used,
        "no_feedback_fast_memory_zero": no_feedback_zero,
        "no_future_or_counterfactual_consequence_trained": True,
    }
    return {
        "initialization": initialization,
        "status": "PASS_DEV_INTEGRITY" if all(integrity.values())
                  else "FAIL_DEV_INTEGRITY",
        "checks": integrity,
        "hypotheses": hypotheses,
        "phase_mse": phase_mse,
        "phase_action_effect_sign": phase_sign,
        "memory_slots_per_model": WIDTH,
        "gated_slots_per_context": HALF,
        "memory_capacity_matched": True,
        "sensor_context_label_visible_in_observation": True,
        "model_weights_frozen": True,
    }


def run()->dict:
    if os.environ.get("NATIVE_P5_DEV_ONLY","1")!="1":
        raise RuntimeError("P5_SCIENCE_FIRST_USE_FORBIDDEN")
    torch.set_num_threads(1)
    start=time.perf_counter()
    records=[run_initialization(seed) for seed in MODEL_INITIALIZATIONS]
    elapsed=time.perf_counter()-start
    integrity=(all(r["status"]=="PASS_DEV_INTEGRITY" for r in records) and
               len(records)==8 and elapsed<MAX_WALL_SECONDS)
    summary={
        name:sum(int(r["hypotheses"][name]) for r in records)
        for name in records[0]["hypotheses"]
    }
    return {
        "schema":SCHEMA,
        "status":"PASS_DEV_INTEGRITY_NOT_SCIENCE" if integrity
                 else "FAIL_DEV_INTEGRITY_NOT_SCIENCE",
        "hypothesis_support_counts_of_eight":summary,
        "model_initializations":list(MODEL_INITIALIZATIONS),
        "records":records,
        "science_first_use":"NOT_RUN",
        "scientific_seeds_consumed":[],
        "science_worlds_constructed":0,
        "model_optimizer_updates":0,
        "memory_budget_scalars_all_arms":WIDTH,
        "context_gated_active_slots":HALF,
        "note": (
            "Capacity matched in total stored scalars and original model "
            "weights, not compute/memory-access operations or effective "
            "per-context capacity. Context is identifiable from observed "
            "sensor sign, and is NOT an inferred latent factor. Two known "
            "algebraic contexts; no unfamiliar-world transfer."),
        "wall_seconds":elapsed
    }


def main()->int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--receipt",type=Path,required=True)
    args=parser.parse_args()
    report=run()
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({
        "status":report["status"],
        "hypothesis_support_counts_of_eight":report["hypothesis_support_counts_of_eight"],
        "initialization_count":len(report["records"]),
        "science_first_use":report["science_first_use"],
        "wall_seconds":report["wall_seconds"]
    },sort_keys=True))
    return 0 if report["status"]=="PASS_DEV_INTEGRITY_NOT_SCIENCE" else 1


if __name__=="__main__":
    raise SystemExit(main())
