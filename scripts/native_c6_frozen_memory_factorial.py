"""C6 DEV: factorial ablation of archived native memory vs trained model weights.

Unlike C5's eight old-factual events, this never re-exposes outcomes to a
training API or calls learn_outcome. It uses original C2-isolated step40 and
step80 *saved* state. The evaluator chooses an old snapshot with oracle
knowledge of the target regime: selection is NOT an inferred context router.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from native_transactional_resume import (
    _rng_snapshot, _rng_restore, prepare_isolated_candidate,
)
from native_training_checkpoint import validate_chain
from scripts.native_prequential_gate import NativeCausalEpisode
from scripts import native_p3_continuity as factual
from scripts import native_c4_measured_retention as c4

SCHEMA = "ttt-hybrid.c6.frozen-memory-factorial-result-dev.v1"
FROZEN = ROOT / "experiments" / "native_c6_frozen_memory_factorial_dev.json"
ARMS = (
    "step40_model_step40_state",
    "step80_model_step80_state",
    "step80_model_old_fast_only",
    "step80_model_old_full_native",
    "step40_model_step80_state",
    "step80_model_old_fast_rolled_sham",
    "step80_model_zero_fast",
)


def _require(ok: bool, label: str):
    if not ok:
        raise AssertionError("C6_" + label)


def _canonical(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode()


def _spec():
    spec = json.loads(FROZEN.read_text("utf-8"))
    e = spec["experiments"]
    _require(
        spec["schema"] == "ttt-hybrid.c6.frozen-memory-factorial-dev.v1"
        and spec["parent_sha"] == "23d35015df39a91a4cb0e26e4e82d4c22f6fea67"
        and spec["seeds"] == list(factual.MODEL_INITIALIZATIONS)
        and spec["model_training_updates"] == factual.TOTAL_EVENTS == 80
        and e["arms"] == list(ARMS)
        and e["samples_per_regime"] == 16
        and e["optimizer_updates_during_evaluation"] == 0
        and e["fast_memory_writes_during_evaluation"] == 0
        and spec["science_first_use"] is False
        and spec["budget"]["original_science_worlds"] == 0,
        "FROZEN_PROTOCOL_DRIFT")
    return spec


def _clone(native):
    return {k: v.detach().clone() for k, v in native.items()}


def graft_fast(old_native: dict, current_native: dict, *, rolled: bool = False):
    """Copies old *saved* fast memory, retains current recurrent/history.

    No alias with the archived saved-state object; neither memory bank is
    modified. Width-roll is a same-capacity/norm sham, not a fair learned
    routing or equivalent-compute scientific comparator.
    """
    _require(set(old_native) == set(current_native)
             == {"fast_weights", "recurrent", "history"}, "NATIVE_FIELDS")
    _require(old_native["fast_weights"].shape
             == current_native["fast_weights"].shape, "FAST_SHAPE")
    updated = _clone(current_native)
    weight = old_native["fast_weights"].detach().clone()
    if rolled:
        weight = torch.roll(weight, shifts=1, dims=-1)
    updated["fast_weights"] = weight
    return updated


def _variants(champion, challenger):
    old = _clone(champion.native_state)
    recent = _clone(challenger.native_state)
    _require(torch.count_nonzero(old["fast_weights"]).item() > 0,
             "OLD_FAST_MEMORY_EMPTY")
    _require(torch.count_nonzero(recent["fast_weights"]).item() > 0,
             "RECENT_FAST_MEMORY_EMPTY")
    rolled = graft_fast(old, recent, rolled=True)
    _require(torch.equal(
        torch.sort(rolled["fast_weights"].reshape(-1)).values,
        torch.sort(old["fast_weights"].reshape(-1)).values),
        "SHAM_FAST_MULTISUBSET_CHANGED")
    zero = _clone(recent)
    zero["fast_weights"].zero_()
    return {
        ARMS[0]: (champion.model, old),
        ARMS[1]: (challenger.model, recent),
        ARMS[2]: (challenger.model, graft_fast(old, recent)),
        ARMS[3]: (challenger.model, _clone(old)),
        ARMS[4]: (champion.model, _clone(recent)),
        ARMS[5]: (challenger.model, rolled),
        ARMS[6]: (challenger.model, zero),
    }


def measure_one(seed: int, scratch: Path) -> dict:
    spec = _spec()
    _require(seed in spec["seeds"] and scratch.is_dir()
             and not any(scratch.iterdir()), "SEED_OR_SCRATCH_DRIFT")
    started = time.perf_counter()
    identity = factual.identity()
    live, opt = factual.new_model_and_optimizer(seed)
    ep = NativeCausalEpisode(live)
    root = factual.save_completed(scratch, identity, live, opt, ep, 0, None)
    for t in range(40):
        factual.transact(live, opt, ep, t)
    old_tip = factual.save_completed(scratch, identity, live, opt, ep, 40, root)
    for t in range(40, 80):
        factual.transact(live, opt, ep, t)
    new_tip = factual.save_completed(scratch, identity, live, opt, ep, 80, old_tip)
    _require(validate_chain(scratch, new_tip, identity)["parent_manifest_sha256"]
             == old_tip, "LINEAGE_REGRESSION")

    champion = prepare_isolated_candidate(
        scratch, tip_sha256=old_tip, identity=identity,
        make_fresh_pair=lambda: factual.new_model_and_optimizer(seed),
        expected_position=40, expected_parent_sha256=root,
        minimum_steps=40, minimum_consumed_examples=40,
        protected_model=live, protected_optimizer=opt)
    challenger = prepare_isolated_candidate(
        scratch, tip_sha256=new_tip, identity=identity,
        make_fresh_pair=lambda: factual.new_model_and_optimizer(seed),
        expected_position=80, expected_parent_sha256=old_tip,
        minimum_steps=80, minimum_consumed_examples=80,
        protected_model=champion.model, protected_optimizer=champion.optimizer)
    _require(champion.sampler["position"] == 40
             and challenger.sampler["position"] == 80, "SAMPLER_RESET")

    frozen = {
        "champion_model": deepcopy(champion.model.state_dict()),
        "challenger_model": deepcopy(challenger.model.state_dict()),
        "champion_opt": deepcopy(champion.optimizer.state_dict()),
        "challenger_opt": deepcopy(challenger.optimizer.state_dict()),
        "champion_native": _clone(champion.native_state),
        "challenger_native": _clone(challenger.native_state),
    }
    variants = _variants(champion, challenger)
    records = {}
    for name in ARMS:
        test_model, chosen_state = variants[name]
        vectors = {}
        means = {}
        for regime in spec["experiments"]["regimes"]:
            errors = c4._measure(test_model, chosen_state, regime, 16)
            vectors[regime] = errors
            means[regime] = sum(errors) / len(errors)
        evidence = {"arm": name, "seed": seed, "old_checkpoint_sha": old_tip,
                    "new_checkpoint_sha": new_tip, "vectors": vectors}
        records[name] = {
            "old_mse": means["old_regime_mse"],
            "recent_mse": means["postreversal_mse"],
            "samples_per_regime": 16,
            "read_only": True,
            "evidence_sha256": hashlib.sha256(_canonical(evidence)).hexdigest(),
            "raw_evidence": evidence,
        }
    _require(factual._exact_equal(champion.model.state_dict(), frozen["champion_model"])
             and factual._exact_equal(challenger.model.state_dict(), frozen["challenger_model"])
             and factual._exact_equal(champion.optimizer.state_dict(), frozen["champion_opt"])
             and factual._exact_equal(challenger.optimizer.state_dict(), frozen["challenger_opt"])
             and factual._exact_equal(champion.native_state, frozen["champion_native"])
             and factual._exact_equal(challenger.native_state, frozen["challenger_native"]),
             "SAVED_CHECKPOINT_MUTATED")
    _require(not (scratch / "accepted").exists()
             and not (scratch / "latest").exists(), "UNAUTHORIZED_PROMOTION")
    diagnostics = spec["hypotheses_diagnostic_only"]
    old = lambda arm: records[arm]["old_mse"]
    tests = {
        "fast_only_recovers_old_by_0p10":
            old(ARMS[1]) - old(ARMS[2])
            >= diagnostics["fast_only_old_mse_gain_vs_step80_at_least"],
        "full_old_native_recovers_old_by_0p10":
            old(ARMS[1]) - old(ARMS[3])
            >= diagnostics["full_old_native_old_mse_gain_vs_step80_at_least"],
        "old_model_new_state_recovers_old_by_0p10":
            old(ARMS[1]) - old(ARMS[4])
            >= diagnostics["old_model_with_new_state_old_mse_gain_vs_step80_at_least"],
        "authentic_old_fast_beats_width_roll_by_0p10":
            old(ARMS[5]) - old(ARMS[2])
            >= diagnostics["fast_only_beats_rolled_sham_old_mse_by_at_least"],
        "old_fast_within_old_champion_0p05":
            old(ARMS[2]) - old(ARMS[0])
            <= diagnostics["fast_only_old_mse_within_step40_by"],
    }
    _require(time.perf_counter()-started < spec["budget"]["max_total_wall_seconds"],
             "SEED_RUNTIME_BUDGET")
    return {
        "seed": seed, "status": "PASS_NO_FEEDBACK_FACTORIAL_EVIDENCE_INTEGRITY_DEV",
        "root_sha256": root,
        "step40_sha256": old_tip, "step80_sha256": new_tip,
        "variants": records,
        "diagnostic_hypotheses": tests,
        "all_models_optimizers_and_saved_memory_unchanged": True,
        "fast_memory_writes_during_evaluation": 0,
        "optimizer_updates_during_evaluation": 0,
        "actual_context_router": "NOT_PRESENT_ORACLE_EVALUATOR_SELECTED_SNAPSHOT",
        "scientific_first_use": "NOT_RUN",
        "scientifically_accepted": False,
        "model_promoted": False,
        "wall_seconds": round(time.perf_counter()-started, 3),
    }


def run() -> dict:
    if (os.environ.get("NATIVE_C6_ENGINEERING_ONLY", "1") != "1"
            or os.environ.get("NATIVE_P3_ENGINEERING_ONLY", "1") != "1"):
        raise RuntimeError("C6_SCIENCE_FIRST_USE_FORBIDDEN")
    started = time.perf_counter()
    torch.set_num_threads(1)
    previous_rng = _rng_snapshot()
    try:
        spec = _spec()
        with tempfile.TemporaryDirectory(prefix="hybrid-c6-frozen-state-dev-") as d:
            records = []
            for seed in spec["seeds"]:
                trial = Path(d) / str(seed)
                trial.mkdir()
                records.append(measure_one(seed, trial))
    finally:
        _rng_restore(previous_rng)
    wall = time.perf_counter()-started
    _require(wall < spec["budget"]["max_total_wall_seconds"],
             "TOTAL_WALL_BUDGET")
    return {
        "schema": SCHEMA,
        "status": "PASS_NO_FEEDBACK_MEMORY_WEIGHT_FACTORIAL_DEV",
        "source_sha": factual.source_sha(),
        "frozen_protocol_sha256": hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        "record_count": len(records),
        "diagnostic_hypotheses": [r["diagnostic_hypotheses"] for r in records],
        "science_first_use": "NOT_RUN",
        "scientifically_accepted": False,
        "model_promoted": False,
        "trainer_activated": False,
        "wall_seconds": round(wall, 3),
        "records": records,
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, sort_keys=True, indent=2)+"\n",
                            encoding="utf-8")
    print(json.dumps({
        "status": result["status"], "source_sha": result["source_sha"],
        "seed_count": result["record_count"],
        "results": [{
            "seed": r["seed"], "hypotheses": r["diagnostic_hypotheses"],
            "old_mse": {k: v["old_mse"] for k, v in r["variants"].items()},
            "recent_mse": {k: v["recent_mse"] for k, v in r["variants"].items()},
        } for r in result["records"]],
        "scientific_first_use": result["science_first_use"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
