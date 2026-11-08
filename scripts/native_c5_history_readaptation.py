"""C5 DEV: factual-prefix readaptation of a genuinely trained C4 checkpoint.

A previous C4 zero-context probe found old-regime regression in all three
development seeds. This is a mechanistic follow-up, NOT original science:
give the exact trained 80-step native candidate only past, realized factual
outcomes from a familiar regime, and compare matched opposite-regime factual
prefixes and no-feedback trajectories.

Outcome after prediction only. Fast-memory-only change, no gradient updates.
Any regained old performance is rapid *readaptation*, not proof of retrieval.
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
from native_transactional_resume import _rng_snapshot, _rng_restore, prepare_isolated_candidate
from scripts.native_prequential_gate import NativeCausalEpisode
from scripts import native_p3_continuity as factual
from scripts import native_c4_measured_retention as c4

SCHEMA = "ttt-hybrid.c5.history-prefix-readaptation-dev.v1"
FROZEN = ROOT / "experiments" / "native_c5_history_readaptation_dev_frozen.json"
ARMS = ("old_factual", "recent_factual", "no_feedback")


def _require(ok, reason):
    if not ok:
        raise AssertionError("C5_" + reason)


def _canonical(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode()


def _config():
    spec = json.loads(FROZEN.read_text(encoding="utf-8"))
    assay = spec["assay"]
    _require(
        spec["schema"] == "ttt-hybrid.c5.observed-history-readaptation-dev.v1"
        and spec["source_parent_sha"] == "3efda04fc609a6b722ff4f4c679c91652e297438"
        and spec["seeds"] == list(factual.MODEL_INITIALIZATIONS)
        and spec["train"]["total_updates"] == factual.TOTAL_EVENTS == 80
        and spec["train"]["checkpoint_at"] == factual.CHECKPOINT_AT == 40
        and assay["prefix_lengths"] == [0, 2, 8]
        and assay["prefix_arms"] == list(ARMS)
        and assay["prequential_evaluation"].startswith("same C4")
        and spec["science_first_use"] is False
        and spec["promotion_allowed"] is False,
        "CONFIG_DRIFT")
    return spec


def _copy_native(native):
    return {k: v.detach().clone() for k, v in native.items()}


def _state(native):
    return ContinuousState(native["recurrent"].detach().clone(),
                           native["fast_weights"].detach().clone(),
                           native["history"].detach().clone())


def _asdict(state):
    return {"recurrent": state.recurrent.detach().clone(),
            "fast_weights": state.fast_weights.detach().clone(),
            "history": state.history.detach().clone()}


def _prefix(model, native, arm: str, length: int) -> tuple[dict, list[float]]:
    """Observe only completed factual transitions; do not train model weights."""
    if arm not in ARMS or length not in (0, 2, 8):
        raise ValueError("C5_UNREGISTERED_ARM_OR_PREFIX_LENGTH")
    episode = NativeCausalEpisode(model)
    episode._state = NativeCausalEpisode._copy_state(_state(native))
    episode._event_index = factual.TOTAL_EVENTS
    no_feedback_state = _state(native)
    errors = []
    with torch.no_grad():
        for i in range(length):
            action = torch.tensor([[-1.0 if i % 2 == 0 else 1.0]])
            sensor = torch.zeros((1, 1))
            elapsed = torch.full((1, 1), 0.2)
            # No context label or factual target is passed to step()/predict().
            if arm == "no_feedback":
                pred, no_feedback_state = model.step(
                    sensor, action, elapsed, no_feedback_state)
            else:
                ticket = episode.predict(sensor, action, elapsed)
                pred = ticket.prediction
            # The outcome is revealed only after this event's prediction.
            # t selects an already-known analytic DEV regime for the EVALUATOR.
            # The model, episode and router never receive this hidden t or tag.
            t = i if arm != "recent_factual" else factual.CHECKPOINT_AT + i
            outcome = factual.analytic_consequence(action, t)
            error = float((pred - outcome).square().item())
            _require(math.isfinite(error), "NONFINITE_PREFIX_MEASUREMENT")
            errors.append(error)
            if arm != "no_feedback":
                episode.observe(ticket, outcome, factual=True)
    final = no_feedback_state if arm == "no_feedback" else episode.snapshot()
    return _asdict(final), errors


def _mse(model, state, regime):
    vector = c4._measure(model, state, regime, 16)
    _require(len(vector) == 16 and all(math.isfinite(x) for x in vector),
             "EVALUATION_VECTOR_INVALID")
    return sum(vector) / len(vector), vector


def measure_one(initialization: int, checkpoint_dir: Path) -> dict:
    """Use existing original P3 optimizer/fast-memory training; never change its code."""
    _require(initialization in factual.MODEL_INITIALIZATIONS
             and checkpoint_dir.is_dir()
             and not any(checkpoint_dir.iterdir()), "SEED_OR_SCRATCH_DRIFT")
    started = time.perf_counter()
    identity = factual.identity()
    model, opt = factual.new_model_and_optimizer(initialization)
    episode = NativeCausalEpisode(model)
    root = factual.save_completed(checkpoint_dir, identity, model, opt, episode, 0, None)
    for t in range(40):
        factual.transact(model, opt, episode, t)
    champion_tip = factual.save_completed(
        checkpoint_dir, identity, model, opt, episode, 40, root)
    for t in range(40, 80):
        factual.transact(model, opt, episode, t)
    challenger_tip = factual.save_completed(
        checkpoint_dir, identity, model, opt, episode, 80, champion_tip)

    champion = prepare_isolated_candidate(
        checkpoint_dir, tip_sha256=champion_tip, identity=identity,
        make_fresh_pair=lambda: factual.new_model_and_optimizer(initialization),
        expected_position=40, minimum_steps=40, minimum_consumed_examples=40,
        expected_parent_sha256=root,
        protected_model=model, protected_optimizer=opt)
    challenger = prepare_isolated_candidate(
        checkpoint_dir, tip_sha256=challenger_tip, identity=identity,
        make_fresh_pair=lambda: factual.new_model_and_optimizer(initialization),
        expected_position=80, minimum_steps=80, minimum_consumed_examples=80,
        expected_parent_sha256=champion_tip,
        protected_model=champion.model, protected_optimizer=champion.optimizer)
    _require(champion.sampler["position"] == 40
             and challenger.sampler["position"] == 80, "COLD_CURSOR_RESET")

    saved_model = deepcopy(challenger.model.state_dict())
    saved_opt = deepcopy(challenger.optimizer.state_dict())
    saved_native = _copy_native(challenger.native_state)

    champ_old, _ = _mse(champion.model, champion.native_state, "old_regime_mse")
    champ_recent, _ = _mse(champion.model, champion.native_state, "postreversal_mse")
    base_old, _ = _mse(challenger.model, challenger.native_state, "old_regime_mse")
    base_recent, _ = _mse(challenger.model, challenger.native_state, "postreversal_mse")

    metrics = {}
    states = {}
    for n in (0, 2, 8):
        for arm in ARMS:
            updated, prefix_errors = _prefix(challenger.model, challenger.native_state, arm, n)
            old_mse, old_vec = _mse(challenger.model, updated, "old_regime_mse")
            recent_mse, recent_vec = _mse(challenger.model, updated, "postreversal_mse")
            norm = float(torch.linalg.vector_norm(
                updated["fast_weights"] - challenger.native_state["fast_weights"]).item())
            evidence = {"seed": initialization, "prefix_length": n, "arm": arm,
                        "parent_checkpoint": challenger_tip,
                        "old_sq_errors": old_vec, "recent_sq_errors": recent_vec,
                        "prefix_prequential_sq_errors": prefix_errors}
            metrics[f"{arm}_{n}"] = {
                "old_mse": old_mse,
                "recent_mse": recent_mse,
                "mean_prefix_prequential_mse":
                    sum(prefix_errors) / len(prefix_errors) if prefix_errors else None,
                "fast_memory_l2_delta": norm,
                "samples_old": len(old_vec), "samples_recent": len(recent_vec),
                "factual_prefix_samples": len(prefix_errors),
                "evidence_sha256": hashlib.sha256(_canonical(evidence)).hexdigest(),
                "squared_error_vectors": evidence,
            }
            states[(n, arm)] = updated
    for n in (0, 2, 8):
        old, recent, zero = (states[(n, name)] for name in ARMS)
        _require(torch.equal(old["recurrent"], recent["recurrent"])
                 and torch.equal(old["history"], recent["history"])
                 and torch.equal(old["recurrent"], zero["recurrent"])
                 and torch.equal(old["history"], zero["history"]),
                 "PREFIX_MISMATCHED_RECURRENT_HISTORY")
        _require(torch.equal(zero["fast_weights"], saved_native["fast_weights"]),
                 "NO_FEEDBACK_CHANGED_FAST_MEMORY")
    _require(factual._exact_equal(challenger.model.state_dict(), saved_model)
             and factual._exact_equal(challenger.optimizer.state_dict(), saved_opt)
             and factual._exact_equal(challenger.native_state, saved_native)
             and not (checkpoint_dir / "accepted").exists()
             and not (checkpoint_dir / "latest").exists(),
             "UNAUTHORIZED_MODEL_OR_CHECKPOINT_MUTATION")

    diag = _config()["frozen_diagnostics"]
    at8 = metrics["old_factual_8"]
    recent8 = metrics["recent_factual_8"]
    negative8 = metrics["no_feedback_8"]
    hypotheses = {
        "old_evidence_recovers_at_least_0p10":
            base_old - at8["old_mse"] >= diag["old_recovery_mse_gain_at_8_at_least"],
        "old_evidence_beats_matched_recent_factual_by_0p10":
            recent8["old_mse"] - at8["old_mse"] >= diag["old_prefix_beats_recent_prefix_old_mse_at_8_by"],
        "old_evidence_beats_no_feedback_by_0p10":
            negative8["old_mse"] - at8["old_mse"] >= diag["old_prefix_beats_no_feedback_old_mse_at_8_by"],
        "old_evidence_within_step40_reference_by_0p05":
            at8["old_mse"] - champ_old <= diag["old_8_prefix_within_old_champion_mse_absolute"],
    }
    _require(time.perf_counter() - started <= 120, "SINGLE_SEED_BUDGET_EXCEEDED")
    return {
        "seed": initialization, "status": "PASS_PREQUENTIAL_PREFIX_ASSAY_INTEGRITY_DEV",
        "checkpoint_lineage": {
            "root": root, "step40": champion_tip, "step80": challenger_tip,
            "champion_position": 40, "challenger_position": 80},
        "baseline": {
            "step40_old_mse": champ_old, "step40_recent_mse": champ_recent,
            "step80_old_mse": base_old, "step80_recent_mse": base_recent},
        "arms": metrics, "hypotheses": hypotheses,
        "model_and_optimizer_frozen_after_step80": True,
        "fast_memory_update_only_during_factual_prefix": True,
        "prefix_does_not_consume_future_or_counterfactual_outcomes": True,
        "independent_scientific_acceptance": False,
        "scientific_first_use": "NOT_RUN", "model_promoted": False,
        "elapsed_seconds": round(time.perf_counter()-started, 3),
    }


def run() -> dict:
    if (os.environ.get("NATIVE_P3_ENGINEERING_ONLY", "1") != "1"
            or os.environ.get("NATIVE_C5_ENGINEERING_ONLY", "1") != "1"):
        raise RuntimeError("C5_SCIENCE_FIRST_USE_FORBIDDEN")
    started = time.perf_counter()
    torch.set_num_threads(1)
    original_rng = _rng_snapshot()
    try:
        cfg = _config()
        with tempfile.TemporaryDirectory(prefix="hybrid-c5-context-evidence-dev-") as td:
            records = []
            for init in cfg["seeds"]:
                root = Path(td) / str(init)
                root.mkdir()
                records.append(measure_one(init, root))
    finally:
        _rng_restore(original_rng)
    wall = time.perf_counter()-started
    _require(wall <= cfg["budget"]["max_wall_seconds_total"],
             "TOTAL_RUNTIME_BUDGET_EXCEEDED")
    return {
        "schema": SCHEMA,
        "status": "PASS_PREQUENTIAL_CAUSAL_HISTORY_INTEGRITY_DEV",
        "source_sha": factual.source_sha(),
        "frozen_protocol_sha256": hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        "record_count": len(records),
        "per_seed_hypotheses": [r["hypotheses"] for r in records],
        "scientific_first_use": "NOT_RUN",
        "real_causal_worlds": 0,
        "trained_conventional_controls": "NOT_RUN_IN_THIS_C5_DIAGNOSTIC",
        "independent_holdout": False,
        "latent_context_label_given_to_predictor": False,
        "whole_process_cost_qualified": False,
        "model_promoted": False,
        "active_training_activated": False,
        "wall_seconds": round(wall, 3),
        "records": records,
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipt", type=Path, required=True)
    args = p.parse_args()
    result = run()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "sha": result["source_sha"],
        "record_count": result["record_count"],
        "hypotheses": result["per_seed_hypotheses"],
        "summary": [{
            "seed": r["seed"], "base_old": r["baseline"]["step80_old_mse"],
            "old_after_8_old_facts": r["arms"]["old_factual_8"]["old_mse"],
            "old_after_8_recent_facts": r["arms"]["recent_factual_8"]["old_mse"],
            "old_after_8_no_feedback": r["arms"]["no_feedback_8"]["old_mse"],
        } for r in result["records"]],
        "science_first_use": result["scientific_first_use"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
