"""C7 DEV: history-derived choice between two intact *saved* native-memory states.

This bridges C6's recovered old fast state with the exact hand-coded P6
FactualSlopeEvidence inference mechanism. Neither current hidden regime,
future outcome nor event schedule is passed to non-oracle selectors.
The frozen step80 model and original saved native states never learn/write.
This is NOT a trained or independently qualified autonomous memory router.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from native_continuous_state import ContinuousState
from native_transactional_resume import (
    _rng_snapshot, _rng_restore, prepare_isolated_candidate,
)
from native_training_checkpoint import validate_chain
from scripts.native_prequential_gate import NativeCausalEpisode
from scripts import native_p3_continuity as factual
from scripts.native_c6_frozen_memory_factorial import graft_fast, _clone
from scripts.native_c7_p6_factual_evidence import FactualSlopeEvidence, SLOPE_SIGN_FLOOR

SCHEMA = "ttt-hybrid.c7.history-derived-saved-native-state-routing-dev.v1"
FROZEN = ROOT / "experiments" / "native_c7_observed_history_archive_dev.json"
ARMS = (
    "inferred_p6", "oracle_upper_bound", "always_recent",
    "always_old", "action_sign", "anti_action_evidence",
)


def _require(ok, reason):
    if not ok:
        raise AssertionError("C7_" + reason)


def _canonical(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode()


def _spec():
    spec = json.loads(FROZEN.read_text("utf-8"))
    d = spec["eval"]
    _require(spec["schema"] == "ttt-hybrid.c7.observed-history-archive-selection-dev.v1"
             and spec["source_parent_sha"] == "3d601a803e0d72d4bb173400d7045a55e0f80ae3"
             and spec["train"]["model_seeds"] == list(factual.MODEL_INITIALIZATIONS)
             and spec["train"]["optimizer_updates"] == factual.TOTAL_EVENTS == 80
             and d["arms"] == list(ARMS)
             and d["events_per_block"] == 16
             and d["events_per_record"] == 64
             and d["response_effect_threshold"] == SLOPE_SIGN_FLOOR
             and d["no_new_fast_memory_writes"] is True
             and d["no_optimizer_updates_after_checkpoints"] is True
             and spec["science"]["original_first_use"] is False
             and spec["science"]["production_owner_promotion"] is False,
             "FROZEN_PROTOCOL_DRIFT")
    return spec


def route_from_history_only(arm: str, action: torch.Tensor,
                            inferred: FactualSlopeEvidence,
                            anti: FactualSlopeEvidence) -> int:
    """No hidden context, event index, labels, targets, or current outcome."""
    if arm == "inferred_p6":
        return inferred.choice
    if arm == "anti_action_evidence":
        return anti.choice
    if arm == "always_recent":
        return -1
    if arm == "always_old":
        return 1
    if arm == "action_sign":
        return 1 if float(action.item()) > 0 else -1
    raise ValueError("C7_NONORACLE_ARM_REQUIRED")


def predict_with_archived_fast(model, current_native, old_native,
                               action: torch.Tensor, choice: int) -> torch.Tensor:
    """Never advance/write archived model or memory in an evaluation query."""
    if choice not in (-1, 1) or action.shape != (1, 1):
        raise ValueError("C7_CHOICE_OR_ACTION_INVALID")
    native = graft_fast(old_native, current_native) if choice == 1 else _clone(current_native)
    state = ContinuousState(
        native["recurrent"], native["fast_weights"], native["history"])
    sensor = torch.zeros((1, 1), dtype=torch.float32)
    elapsed = torch.full((1, 1), .2, dtype=torch.float32)
    with torch.no_grad():
        prediction, _ = model.step(sensor, action, elapsed, state)
    return prediction.detach().clone()


def evaluate_order(model, recent_native, old_native,
                   order: tuple[str, ...], spec: dict) -> dict:
    if (list(order) not in spec["eval"]["context_orders"]
            or len(order) != 4 or set(order) != {"old", "recent"}):
        raise ValueError("C7_UNREGISTERED_ORDER")
    inferred = FactualSlopeEvidence()
    anti = FactualSlopeEvidence()
    errors = {arm: [] for arm in ARMS}
    routes = {arm: [] for arm in ARMS}
    records = []
    block_size = spec["eval"]["events_per_block"]
    with torch.no_grad():
        for i in range(block_size * len(order)):
            label = order[i // block_size]  # evaluator only, never route input
            action = torch.tensor([[-1. if i % 2 == 0 else 1.]], dtype=torch.float32)
            chosen = {}
            scored = {}
            # All non-oracle choices happen BEFORE the current consequence exists.
            for arm in ARMS:
                if arm == "oracle_upper_bound":
                    choice = 1 if label == "old" else -1
                else:
                    choice = route_from_history_only(arm, action, inferred, anti)
                chosen[arm] = choice
                scored[arm] = predict_with_archived_fast(
                    model, recent_native, old_native, action, choice)
            t = 0 if label == "old" else 40
            observed = factual.analytic_consequence(action, t)
            one = {}
            for arm in ARMS:
                err = float((scored[arm] - observed).square().item())
                _require(math.isfinite(err) and err >= 0, "NONFINITE_PREDICTION")
                errors[arm].append(err)
                routes[arm].append(chosen[arm])
                one[arm] = err
            records.append({
                "index": i, "current_action": float(action.item()),
                "ground_truth_label_evaluator_only": label,
                "pre_outcome_choices": dict(chosen),
                "squared_errors": one,
                "factual_observed_after_all_predictions": float(observed.item()),
            })
            # P6 evidence is updated ONLY after all arm predictions are scored.
            inferred.observe(action, observed)
            # Same observations/cost; action labels deliberately misassociated.
            anti.observe(-action, observed)
    _require(inferred.observations == anti.observations == len(records) == 64,
             "FACTUAL_EVIDENCE_COUNT")
    _require(all(len(errors[a]) == 64 for a in ARMS), "EVENT_COUNT")
    comparisons = {}
    late_indices = []
    immediate_indices = []
    for block in range(len(order)):
        start = block * block_size
        immediate_indices.extend(range(start, start + 4))
        late_indices.extend(range(start + 4, start + block_size))
    for arm in ARMS:
        comparisons[arm] = {
            "late_mse": sum(errors[arm][i] for i in late_indices) / len(late_indices),
            "immediate_mse": sum(errors[arm][i] for i in immediate_indices) / len(immediate_indices),
            "late_route_accuracy": sum(
                routes[arm][i] == (1 if order[i // block_size] == "old" else -1)
                for i in late_indices) / len(late_indices),
            "all_64_squared_errors": errors[arm],
            "pre_outcome_route_sequence": routes[arm],
        }
    h = spec["eval"]["hypotheses"]
    i = comparisons["inferred_p6"]
    hypotheses = {
        "inferred_beats_always_recent_late_by_0p10":
            comparisons["always_recent"]["late_mse"] - i["late_mse"] >= h[
                "inferred_late_mse_better_than_always_recent_by_at_least"],
        "inferred_beats_action_sign_late_by_0p10":
            comparisons["action_sign"]["late_mse"] - i["late_mse"] >= h[
                "inferred_late_mse_better_than_action_sign_by_at_least"],
        "inferred_beats_anti_action_late_by_0p10":
            comparisons["anti_action_evidence"]["late_mse"] - i["late_mse"] >= h[
                "inferred_late_mse_better_than_anti_action_evidence_by_at_least"],
        "inferred_late_route_accuracy_at_least_0p875":
            i["late_route_accuracy"] >= h[
                "inferred_late_route_accuracy_at_least"],
        "oracle_late_no_worse_than_inferred":
            comparisons["oracle_upper_bound"]["late_mse"] <= i["late_mse"] + h[
                "oracle_late_mse_no_worse_than_inferred_tolerance"],
    }
    return {
        "order": list(order),
        "context_selector": "HAND_CODED_P6_FACTUAL_SLOPE_SIGN",
        "prediction_order": "PREDICTIONS_BEFORE_CURRENT_FACTUAL_OBSERVATION",
        "measurements": comparisons, "hypotheses": hypotheses,
        "inferred_changes_at_observation_counts": list(inferred._change_events),
        "oracle_only_uses_evaluator_context": True,
        "used_hidden_label_for_inference": False,
        "used_current_outcome_for_selection": False,
        "new_fast_memory_writes": 0, "postcheckpoint_optimizer_updates": 0,
        "raw_event_ledger_sha256": hashlib.sha256(_canonical(records)).hexdigest(),
        "raw_event_ledger": records,
    }


def measure_seed(seed: int, scratch: Path, spec: dict) -> dict:
    _require(seed in factual.MODEL_INITIALIZATIONS
             and scratch.is_dir() and not any(scratch.iterdir()),
             "SEED_OR_SCRATCH_DRIFT")
    identity = factual.identity()
    live, optimizer = factual.new_model_and_optimizer(seed)
    episode = NativeCausalEpisode(live)
    root = factual.save_completed(scratch, identity, live, optimizer, episode, 0, None)
    for t in range(40):
        factual.transact(live, optimizer, episode, t)
    old_tip = factual.save_completed(scratch, identity, live, optimizer, episode, 40, root)
    for t in range(40, 80):
        factual.transact(live, optimizer, episode, t)
    new_tip = factual.save_completed(scratch, identity, live, optimizer, episode, 80, old_tip)
    _require(validate_chain(scratch, new_tip, identity)["parent_manifest_sha256"] == old_tip,
             "ANCESTRY_DRIFT")
    old = prepare_isolated_candidate(
        scratch, tip_sha256=old_tip, identity=identity,
        make_fresh_pair=lambda: factual.new_model_and_optimizer(seed),
        expected_position=40, expected_parent_sha256=root, minimum_steps=40,
        minimum_consumed_examples=40, protected_model=live,
        protected_optimizer=optimizer)
    recent = prepare_isolated_candidate(
        scratch, tip_sha256=new_tip, identity=identity,
        make_fresh_pair=lambda: factual.new_model_and_optimizer(seed),
        expected_position=80, expected_parent_sha256=old_tip, minimum_steps=80,
        minimum_consumed_examples=80, protected_model=old.model,
        protected_optimizer=old.optimizer)
    _require(old.sampler["position"] == 40 and recent.sampler["position"] == 80,
             "TRAINING_CURSOR_DRIFT")
    model_before = deepcopy(recent.model.state_dict())
    opt_before = deepcopy(recent.optimizer.state_dict())
    old_before = _clone(old.native_state)
    recent_before = _clone(recent.native_state)
    results = [
        evaluate_order(recent.model, recent.native_state, old.native_state, tuple(order), spec)
        for order in spec["eval"]["context_orders"]
    ]
    _require(factual._exact_equal(model_before, recent.model.state_dict())
             and factual._exact_equal(opt_before, recent.optimizer.state_dict())
             and factual._exact_equal(old_before, old.native_state)
             and factual._exact_equal(recent_before, recent.native_state),
             "LEARNED_MODEL_OR_ARCHIVE_MUTATED")
    _require(not (scratch / "accepted").exists() and not (scratch / "latest").exists(),
             "UNAUTHORIZED_POINTER_CHANGE")
    return {
        "seed": seed, "root_sha256": root,
        "old_checkpoint_sha256": old_tip, "current_checkpoint_sha256": new_tip,
        "factual_train_steps_before_evaluation": 80, "evaluation_orders": results,
        "checkpoint_and_native_state_unchanged": True,
        "no_new_model_or_fast_weight_learning": True,
        "science_first_use": "NOT_RUN", "model_promoted": False,
    }


def run() -> dict:
    if (os.environ.get("NATIVE_C7_ENGINEERING_ONLY", "1") != "1"
            or os.environ.get("NATIVE_P3_ENGINEERING_ONLY", "1") != "1"):
        raise RuntimeError("C7_SCIENCE_FIRST_USE_FORBIDDEN")
    start = time.perf_counter()
    torch.set_num_threads(1)
    host_rng = _rng_snapshot()
    try:
        spec = _spec()
        with tempfile.TemporaryDirectory(prefix="hybrid-c7-saved-state-routing-dev-") as td:
            records = []
            for seed in spec["train"]["model_seeds"]:
                scratch = Path(td) / str(seed)
                scratch.mkdir()
                records.append(measure_seed(seed, scratch, spec))
    finally:
        _rng_restore(host_rng)
    seconds = round(time.perf_counter() - start, 3)
    _require(seconds <= spec["resources"]["max_wall_seconds"],
             "TOTAL_RUNTIME_BUDGET_EXCEEDED")
    counts = {}
    names = list(records[0]["evaluation_orders"][0]["hypotheses"])
    for name in names:
        counts[name] = sum(int(order["hypotheses"][name])
                           for rec in records for order in rec["evaluation_orders"])
    return {
        "schema": SCHEMA, "status": "PASS_CAUSAL_HISTORY_SNAPSHOT_SELECTION_INTEGRITY_DEV",
        "source_sha": factual.source_sha(),
        "frozen_protocol_sha256": hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        "record_count": len(records),
        "orders_per_seed": 2, "hypothesis_support_of_6": counts,
        "science_first_use": "NOT_RUN", "scientifically_accepted": False,
        "new_training_loop_activated": False, "model_promoted": False,
        "trained_context_router": False, "independent_heldout_qualified": False,
        "wall_seconds": seconds, "records": records,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, required=True)
    args = ap.parse_args()
    result = run()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"], "source_sha": result["source_sha"],
        "record_count": result["record_count"],
        "hypothesis_support_of_6": result["hypothesis_support_of_6"],
        "per_order": [
            {"seed": rec["seed"], "order": o["order"],
             "inferred_late_mse": o["measurements"]["inferred_p6"]["late_mse"],
             "oracle_late_mse": o["measurements"]["oracle_upper_bound"]["late_mse"],
             "always_recent_late_mse": o["measurements"]["always_recent"]["late_mse"],
             "anti_action_late_mse": o["measurements"]["anti_action_evidence"]["late_mse"],
             "inferred_late_route_accuracy": o["measurements"]["inferred_p6"]["late_route_accuracy"]}
            for rec in result["records"] for o in rec["evaluation_orders"]
        ],
        "scientific_first_use": result["science_first_use"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
