"""C3 DEV: champion-safe factual native training across an actual cold checkpoint.

Reuses the unchanged P3 factual-feedback learner from PR #54 and isolated
candidate recovery from C2. CPU-only public analytic events; science first-use
NOT RUN. No accepted-pointer write, owner loop activation or scientific claims.
"""
from __future__ import annotations

from copy import deepcopy
import json
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
    prepare_isolated_candidate, _rng_snapshot, _rng_restore,
)
from native_training_checkpoint import validate_chain
from scripts.native_prequential_gate import NativeCausalEpisode
from scripts import native_p3_continuity as factual

SCHEMA = "ttt-hybrid.c3-factual-champion-safe-resume-dev.v1"
MAX_WALL_SECONDS = 120


def _episode(model, native, cursor):
    state = ContinuousState(
        native["recurrent"], native["fast_weights"], native["history"])
    model._validate_state(state, torch.zeros((1, 1), dtype=torch.float32))
    ep = NativeCausalEpisode(model)
    ep._state = NativeCausalEpisode._copy_state(state)
    ep._event_index = cursor
    if ep._pending is not None:
        raise ValueError("C3_UNEXPECTED_PENDING_TICKET")
    return ep


def _require(condition, reason):
    if not condition:
        raise AssertionError("C3_" + reason)


def verify_one(initialization: int, directory: Path) -> dict:
    """40 factual updates -> isolated restore -> 40 more, exact reference."""
    identity = factual.identity()
    model, opt = factual.new_model_and_optimizer(initialization)
    episode = NativeCausalEpisode(model)
    root = factual.save_completed(directory, identity, model, opt, episode,
                                  0, None)
    for step in range(factual.CHECKPOINT_AT):
        factual.transact(model, opt, episode, step)
    original_native = deepcopy(factual.state_dict(episode, model))
    tip = factual.save_completed(
        directory, identity, model, opt, episode,
        factual.CHECKPOINT_AT, root)
    _require(validate_chain(directory, tip, identity)["steps"]
             == factual.CHECKPOINT_AT, "PARENT_LINEAGE")
    _require(bool(torch.count_nonzero(original_native["fast_weights"])),
             "FASTM_STATE_NOT_LEARNED")

    # Retained model acts as the protected reference. Do NOT load into it.
    uninterrupted = [factual.transact(model, opt, episode, step)
                     for step in range(factual.CHECKPOINT_AT,
                                       factual.TOTAL_EVENTS)]
    reference_model = deepcopy(model.state_dict())
    reference_optimizer = deepcopy(opt.state_dict())
    reference_native = deepcopy(factual.state_dict(episode, model))
    reference_rng_after = torch.get_rng_state().clone()

    candidate = prepare_isolated_candidate(
        directory, tip_sha256=tip, identity=identity,
        make_fresh_pair=lambda: factual.new_model_and_optimizer(initialization),
        expected_position=factual.CHECKPOINT_AT,
        expected_parent_sha256=root,
        minimum_steps=factual.CHECKPOINT_AT,
        minimum_consumed_examples=factual.CHECKPOINT_AT,
        protected_model=model, protected_optimizer=opt)
    _require(candidate.sampler == {
        "position": factual.CHECKPOINT_AT,
        "event_index": factual.CHECKPOINT_AT,
        "pending": False}, "CANDIDATE_PENDING_OR_CURSOR")
    _require(factual._exact_equal(candidate.native_state, original_native),
             "CANDIDATE_FAST_MEMORY_NOT_RETAINED")
    _require(factual._exact_equal(model.state_dict(), reference_model)
             and factual._exact_equal(opt.state_dict(), reference_optimizer),
             "CHAMPION_CHANGED_DURING_RESTORE")

    cold_episode = _episode(candidate.model, candidate.native_state,
                            factual.CHECKPOINT_AT)
    # C2 intentionally does not install saved RNG in the host process.
    # This bounded engineering-only continuation explicitly restores it
    # inside a saved ambient RNG scope; never activate an owner implicitly.
    ambient = _rng_snapshot()
    try:
        _rng_restore(candidate.restored_rng)
        resumed = [
            factual.transact(candidate.model, candidate.optimizer,
                             cold_episode, step)
            for step in range(factual.CHECKPOINT_AT, factual.TOTAL_EVENTS)
        ]
        _require(torch.equal(torch.get_rng_state(), reference_rng_after),
                 "POSTRESUME_RNG_DRIFT")
        child = factual.save_completed(
            directory, identity, candidate.model, candidate.optimizer,
            cold_episode, factual.TOTAL_EVENTS, tip)
    finally:
        _rng_restore(ambient)

    verified = validate_chain(directory, child, identity)
    _require(verified["steps"] == factual.TOTAL_EVENTS
             and verified["consumed_examples"] == factual.TOTAL_EVENTS
             and verified["parent_manifest_sha256"] == tip,
             "CHILD_NONADVANCING_LINEAGE")
    checks = {
        "source_identical_checkpoint_lineage":
            root != tip and tip != child and child != root,
        "factual_only_prequential_sequence":
            len(uninterrupted) == len(resumed) == factual.TOTAL_EVENTS - factual.CHECKPOINT_AT,
        "postcheckpoint_factual_predictions_bit_exact":
            uninterrupted == resumed,
        "postcheckpoint_model_parameters_bit_exact":
            factual._exact_equal(reference_model, candidate.model.state_dict()),
        "postcheckpoint_optimizer_moments_bit_exact":
            factual._exact_equal(reference_optimizer,
                                 candidate.optimizer.state_dict()),
        "postcheckpoint_native_memory_bit_exact":
            factual._exact_equal(reference_native,
                                 factual.state_dict(cold_episode,
                                                    candidate.model)),
        "protected_reference_model_unchanged":
            factual._exact_equal(reference_model, model.state_dict())
            and factual._exact_equal(reference_optimizer, opt.state_dict()),
        "full_factual_progress_80_no_replay":
            child != tip and verified["consumed_examples"] == 80,
        "retained_champion_accepted_pointer_not_touched":
            not (directory / "accepted").exists()
            and not (directory / "latest").exists(),
    }
    _require(all(checks.values()), "FINAL_EXACT_CONTINUITY_FAIL")
    final = prepare_isolated_candidate(
        directory, tip_sha256=child, identity=identity,
        make_fresh_pair=lambda: factual.new_model_and_optimizer(initialization),
        expected_position=factual.TOTAL_EVENTS,
        expected_parent_sha256=tip,
        minimum_steps=factual.TOTAL_EVENTS,
        minimum_consumed_examples=factual.TOTAL_EVENTS,
        protected_model=model, protected_optimizer=opt)
    _require(factual._exact_equal(final.model.state_dict(), reference_model)
             and factual._exact_equal(final.optimizer.state_dict(),
                                      reference_optimizer)
             and factual._exact_equal(final.native_state, reference_native),
             "FINAL_COLD_RESTORE_REGRESSION")
    return {
        "initialization": initialization,
        "status": "PASS_FACTUAL_CANDIDATE_ISOLATED_RESUME_DEV",
        "checks": checks,
        "root_manifest_sha256": root,
        "parent_manifest_sha256": tip,
        "child_manifest_sha256": child,
        "checkpoint_at_steps": factual.CHECKPOINT_AT,
        "final_steps": factual.TOTAL_EVENTS,
        "final_consumed_examples": verified["consumed_examples"],
        "protected_reference_unchanged": True,
        "science_first_use": "NOT_RUN",
        "training_activated": False,
        "model_promoted": False,
    }


def run() -> dict:
    if (os.environ.get("NATIVE_P3_ENGINEERING_ONLY", "1") != "1"
            or os.environ.get("NATIVE_C3_ENGINEERING_ONLY", "1") != "1"):
        raise RuntimeError("C3_SCIENCE_FIRST_USE_FORBIDDEN")
    start = time.perf_counter()
    torch.set_num_threads(1)
    host_rng = _rng_snapshot()
    try:
        with tempfile.TemporaryDirectory(prefix="hybrid-c3-factual-dev-") as td:
            records = []
            for init in factual.MODEL_INITIALIZATIONS:
                directory = Path(td) / str(init)
                directory.mkdir()
                records.append(verify_one(init, directory))
    finally:
        _rng_restore(host_rng)
    seconds = round(time.perf_counter() - start, 3)
    if seconds > MAX_WALL_SECONDS:
        raise RuntimeError("C3_RUNTIME_BUDGET_EXCEEDED")
    return {
        "schema": SCHEMA,
        "status": "PASS_FACTUAL_TRAINING_ISOLATION_AND_EXACT_CONTINUITY_DEV",
        "exact_source_sha": factual.source_sha(),
        "record_count": len(records),
        "factual_dev_seeds": list(factual.MODEL_INITIALIZATIONS),
        "total_factual_gradient_updates_per_seed": factual.TOTAL_EVENTS,
        "protected_champion_not_mutated": True,
        "original_science_worlds": 0,
        "science_first_use": "NOT_RUN",
        "scientific_cumulative_improvement": "UNVERIFIED",
        "production_training": "INACTIVE",
        "model_promoted": False,
        "wall_seconds": seconds,
        "records": records,
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "source": result["exact_source_sha"],
        "records": result["record_count"],
        "checks_pass": all(all(r["checks"].values())
                           for r in result["records"]),
        "wall_seconds": result["wall_seconds"],
        "science_first_use": result["science_first_use"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
