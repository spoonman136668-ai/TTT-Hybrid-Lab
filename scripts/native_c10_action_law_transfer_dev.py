"""C10: unchanged learned P7 vs trained MLP vs past-only ridge on shifted laws.

Pre-registered synthetic DEV, not independent heldout scientific source.
The trained native model and both archived native fast memories are frozen.
Each selector predicts from prior actually completed factual observations;
current context, consequence, and noise never enter nonoracle route inputs.
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
import numpy as np
import torch

from native_training_checkpoint import validate_chain
from native_transactional_resume import (
    _rng_snapshot, _rng_restore, prepare_isolated_candidate,
)
from scripts.native_prequential_gate import NativeCausalEpisode
from scripts import native_p3_continuity as factual
from scripts import native_p7_learned_router_dev as p7
from scripts import native_c8_trained_archive_selection as c8
from scripts import native_c7_history_archive_selection as c7

SCHEMA = "ttt-hybrid.c10.action-law-transfer-results-dev.v1"
FROZEN = ROOT / "experiments" / "native_c10_action_law_transfer_frozen.json"
ARMS = (
    "trained_p7", "trained_mlp", "online_ridge", "always_recent",
    "oracle_evaluator_only", "p7_zero_history",
)


def _require(ok, reason):
    if not ok:
        raise AssertionError("C10_" + reason)


def _canon(x):
    return (json.dumps(x, sort_keys=True, separators=(",", ":"),
                       allow_nan=False, ensure_ascii=True) + "\n").encode()


def _frozen():
    d = json.loads(FROZEN.read_text(encoding="utf-8"))
    tr, x = d["training"], d["transfer"]
    _require(
        d["schema"] == "ttt-hybrid.c10.action-law-transfer-dev.v1"
        and d["parent_sha"] == "14fc30a0507439733f6bdeb17e439bfedaef21f2"
        and d["original_p7_source_blob"] == c8.P7_ORIGINAL_BLOB
        and d["original_science_first_use_authorized"] is False
        and tr["native_seeds"] == list(factual.MODEL_INITIALIZATIONS)
        and tr["router_seeds"] == list(p7.INIT_SEEDS)
        and tr["original_p7_adamw_updates"] == p7.UPDATES == 600
        and tr["original_p7_batch"] == p7.BATCH == 256
        and tr["native_pretrain_events"] == 80
        and tr["old_checkpoint_events"] == 40
        and x["dev_stream_seeds"] == [123041, 123059]
        and x["conditions"] == ["linear_magnitude", "saturating", "deadzone"]
        and x["actions"] == [-1., -.65, -.25, .25, .65, 1.]
        and x["events_per_trace"] == 96
        and x["arms"] == list(ARMS)
        and x["ridge_control"]["ridge_lambda"] == .05
        and x["ridge_control"]["min_samples"] == 2
        and x["ridge_control"]["min_effect_magnitude"] == .04
        and x["competence_gate"]["oracle_mature_mse_max"] == .20
        and x["maturity_min_events_since_latent_switch"] == 8
        and d["budget"]["gpu_seconds"] == 0,
        "FROZEN_PROTOCOL_SOURCE_DRIFT")
    return d


def _law_outcome(condition: str, action: float, regime: str,
                 noise: float) -> float:
    """Evaluator's new public DEV law: *not* passed to any selector."""
    if not math.isfinite(action) or not math.isfinite(noise):
        raise ValueError("C10_NONFINITE_ACTION_OR_NOISE")
    if condition == "linear_magnitude":
        value = action
    elif condition == "saturating":
        value = math.tanh(2. * action) / math.tanh(2.)
    elif condition == "deadzone":
        value = math.copysign(max(abs(action) - .30, 0.) / .70, action)
    else:
        raise ValueError("C10_UNKNOWN_CONSTITUTIVE_LAW")
    if regime not in ("old", "recent"):
        raise ValueError("C10_UNKNOWN_EVALUATOR_CONTEXT")
    return .04 + (1. if regime == "old" else -1.) * .45 * value + noise


def make_trace(seed: int, native_seed: int, condition: str, cfg: dict) -> list[dict]:
    x = cfg["transfer"]
    if (seed not in x["dev_stream_seeds"] or
            native_seed not in cfg["training"]["native_seeds"] or
            condition not in x["conditions"]):
        raise ValueError("C10_UNREGISTERED_DEV_CONTEXT")
    rng = np.random.default_rng(
        seed + native_seed * 71 + x["conditions"].index(condition) * 2341)
    latent = 1 if rng.integers(0, 2) == 1 else -1
    changed_at = 0
    out = []
    for i in range(x["events_per_trace"]):
        if i and rng.random() < x["latent_flip_probability"]:
            latent = -latent
            changed_at = i
        a = float(x["actions"][int(rng.integers(0, len(x["actions"])))])
        noise = float(rng.normal(0., x["observation_noise_sigma"]))
        out.append({
            "index": i, "action": a,
            "true_context_evaluator_only": "old" if latent == 1 else "recent",
            "age_since_switch_evaluator_only": i - changed_at,
            "noise_evaluator_only": noise,
        })
    _require(len(out) == 96, "UNEXPECTED_TRACE_LENGTH")
    return out


def past_only_ridge_choice(history: tuple[tuple[float, float], ...],
                           *, ridge: float = .05, floor: float = .04) -> int:
    """Two-feature regularized linear fit, only past (action,outcome) pairs.

    No latent label, future outcome, or event time. Intercept is unpenalized,
    action coefficient is stabilized by frozen ridge lambda.
    """
    if not isinstance(history, tuple):
        raise ValueError("C10_HISTORY_TUPLE_REQUIRED")
    recent = history[-8:]
    if len(recent) < 2:
        return 1
    actions = [float(p[0]) for p in recent]
    facts = [float(p[1]) for p in recent]
    if any(not math.isfinite(v) for v in actions + facts):
        raise ValueError("C10_HISTORY_NONFINITE")
    mx = sum(actions)/len(actions)
    my = sum(facts)/len(facts)
    cov = sum((a-mx)*(f-my) for a,f in zip(actions,facts))
    denom = sum((a-mx)**2 for a in actions) + ridge
    slope = cov/denom
    if abs(slope) < floor:
        return 1
    return 1 if slope > 0 else -1


def _eval_trace(step80_model, old_native, new_native, trained, conventional,
                plan: list[dict], cfg: dict) -> dict:
    if len(plan) != cfg["transfer"]["events_per_trace"]:
        raise ValueError("C10_TRACE_LENGTH_DRIFT")
    past: list[tuple[float,float]] = []
    errs = {a: [] for a in ARMS}
    choices = {a: [] for a in ARMS}
    ledger = []
    for event in plan:
        action = torch.tensor([[event["action"]]], dtype=torch.float32)
        h = tuple(past)
        # Apart from the explicitly labeled ORACLE, no arm can access the
        # regime label, event index, current outcome, or phase age.
        routes = {
            "trained_p7": c8.signed_pair_route(trained, h),
            "trained_mlp": c8.signed_pair_route(conventional, h),
            "online_ridge": past_only_ridge_choice(h),
            "always_recent": -1,
            "oracle_evaluator_only": 1 if event[
                "true_context_evaluator_only"] == "old" else -1,
            "p7_zero_history": c8.signed_pair_route(
                trained, h, history_mode="zero"),
        }
        predictions = {
            arm: c7.predict_with_archived_fast(
                step80_model, new_native, old_native, action, route)
            for arm, route in routes.items()
        }
        # The evaluator materializes this consequence *after* all choices.
        target = _law_outcome(
            cfg["condition"], float(event["action"]),
            event["true_context_evaluator_only"], event["noise_evaluator_only"])
        observed = torch.tensor([[target]], dtype=torch.float32)
        scores = {}
        for arm in ARMS:
            score = float((predictions[arm]-observed).square().item())
            _require(math.isfinite(score) and score >= 0., "INVALID_SCORE")
            scores[arm] = score
            errs[arm].append(score)
            choices[arm].append(routes[arm])
        ledger.append({
            "index": event["index"], "action": event["action"],
            "context_evaluator_only": event["true_context_evaluator_only"],
            "age_evaluator_only": event["age_since_switch_evaluator_only"],
            "number_of_completed_facts_before_route": len(h),
            "route_chosen_before_current_result": routes,
            "current_factual_result_after_all_predictions": float(observed.item()),
            "squared_errors": scores,
        })
        # Only completed actual factual observations enter later histories.
        past.append((float(event["action"]), float(observed.item())))
    mature = [i for i, event in enumerate(ledger)
              if event["age_evaluator_only"] >= cfg["transfer"][
                  "maturity_min_events_since_latent_switch"]]
    early = [i for i in range(len(ledger)) if i not in set(mature)]
    _require(len(mature) >= 8 and len(early) >= 8, "UNQUALIFIABLE_MATURITY")
    results = {}
    for arm in ARMS:
        e = errs[arm]
        results[arm] = {
            "full_mse": sum(e)/len(e),
            "mature_mse": sum(e[i] for i in mature)/len(mature),
            "early_mse": sum(e[i] for i in early)/len(early),
            "mature_route_accuracy": sum(
                choices[arm][i] ==
                (1 if ledger[i]["context_evaluator_only"] == "old" else -1)
                for i in mature)/len(mature),
            "squared_errors": e,
            "routes_before_current_outcome": choices[arm],
        }
    thresholds = cfg["transfer"]["preregistered_diagnostics"]
    own = results["trained_p7"]["mature_mse"]
    hypotheses = {
        "trained_context_accuracy_at_least_0p75":
            results["trained_p7"]["mature_route_accuracy"] >=
            thresholds["trained_p7_mature_context_accuracy_min"],
        "trained_better_than_trained_mlp_by_0p01":
            results["trained_mlp"]["mature_mse"]-own >=
            thresholds["trained_p7_better_than_mlp_mature_mse_by"],
        "trained_better_than_online_ridge_by_0p01":
            results["online_ridge"]["mature_mse"]-own >=
            thresholds["trained_p7_better_than_ridge_mature_mse_by"],
        "trained_better_than_always_recent_by_0p05":
            results["always_recent"]["mature_mse"]-own >=
            thresholds["trained_p7_better_than_always_recent_mature_mse_by"],
        "trained_better_than_zero_history_by_0p05":
            results["p7_zero_history"]["mature_mse"]-own >=
            thresholds["trained_p7_better_than_zero_history_mature_mse_by"],
        "archived_memory_oracle_competent_mse_at_most_0p20":
            results["oracle_evaluator_only"]["mature_mse"] <=
            thresholds["oracle_mature_mse_at_most"],
    }
    return {
        "condition": cfg["condition"],
        "source_dev_seed": cfg["stream_seed"],
        "events": len(ledger), "mature_events": len(mature),
        "all_arms_same_factual_actions": True,
        "revealed_after_all_predictions": True,
        "selector_has_no_current_latent_label": True,
        "native_optimizer_and_fast_memory_updates": 0,
        "metrics": results,
        "diagnostic_hypotheses": hypotheses,
        "ledger_sha256": hashlib.sha256(_canon(ledger)).hexdigest(),
        "trace_sha256": hashlib.sha256(_canon(plan)).hexdigest(),
        "raw_prequential_ledger": ledger,
    }


def _native_checkpoint_evals(seed: int, fitted, cfg: dict):
    identity = factual.identity()
    with tempfile.TemporaryDirectory(prefix="hybrid-c10-immutable-dev-") as d:
        path = Path(d)
        model, optimizer = factual.new_model_and_optimizer(seed)
        episode = NativeCausalEpisode(model)
        root = factual.save_completed(path, identity, model, optimizer, episode, 0, None)
        for i in range(40):
            factual.transact(model, optimizer, episode, i)
        old_sha = factual.save_completed(path, identity, model, optimizer, episode, 40, root)
        for i in range(40,80):
            factual.transact(model, optimizer, episode, i)
        new_sha = factual.save_completed(path, identity, model, optimizer, episode, 80, old_sha)
        _require(validate_chain(path, new_sha, identity)["parent_manifest_sha256"]
                 == old_sha, "CHECKPOINT_ANCESTRY")
        old = prepare_isolated_candidate(
            path, tip_sha256=old_sha, identity=identity,
            expected_position=40, expected_parent_sha256=root,
            minimum_steps=40, minimum_consumed_examples=40,
            make_fresh_pair=lambda: factual.new_model_and_optimizer(seed),
            protected_model=model, protected_optimizer=optimizer)
        recent = prepare_isolated_candidate(
            path, tip_sha256=new_sha, identity=identity,
            expected_position=80, expected_parent_sha256=old_sha,
            minimum_steps=80, minimum_consumed_examples=80,
            make_fresh_pair=lambda: factual.new_model_and_optimizer(seed),
            protected_model=old.model, protected_optimizer=old.optimizer)
        before = (
            deepcopy(recent.model.state_dict()),
            deepcopy(recent.optimizer.state_dict()),
            deepcopy(old.native_state),
            deepcopy(recent.native_state),
            c8._weights_sha(fitted["learned"]),
            c8._weights_sha(fitted["conventional"]),
        )
        observations = []
        for condition in cfg["transfer"]["conditions"]:
            for stream in cfg["transfer"]["dev_stream_seeds"]:
                plan = make_trace(stream, seed, condition, cfg)
                local = dict(cfg)
                local["condition"] = condition
                local["stream_seed"] = stream
                observations.append(_eval_trace(
                    recent.model, old.native_state, recent.native_state,
                    fitted["learned"], fitted["conventional"], plan, local))
        _require(
            factual._exact_equal(before[0],recent.model.state_dict())
            and factual._exact_equal(before[1],recent.optimizer.state_dict())
            and factual._exact_equal(before[2],old.native_state)
            and factual._exact_equal(before[3],recent.native_state)
            and c8._weights_sha(fitted["learned"]) == before[4]
            and c8._weights_sha(fitted["conventional"]) == before[5]
            and not (path/"latest").exists()
            and not (path/"accepted").exists(),
            "STORED_MODEL_OR_MEMORY_CHANGED")
        return {
            "native_seed": seed, "root_checkpoint_sha256": root,
            "archived_old_sha256": old_sha,
            "current_new_sha256": new_sha,
            "checkpoint_weights_optimizer_and_native_memory_unchanged": True,
            "assays": observations,
        }


def run():
    if any(os.environ.get(k, "1") != "1" for k in (
        "NATIVE_C10_ENGINEERING_ONLY", "NATIVE_P3_ENGINEERING_ONLY",
        "NATIVE_P7_DEV_ONLY")):
        raise RuntimeError("C10_FIRST_SCIENTIFIC_USE_FORBIDDEN")
    start = time.perf_counter()
    torch.set_num_threads(1)
    host_rng = _rng_snapshot()
    try:
        cfg = _frozen()
        train = p7.synthetic_dev(p7.TRAIN_DEV_SEED, p7.TRAIN_EPISODES)
        valid = p7.synthetic_dev(p7.VALIDATION_DEV_SEED, p7.VALID_EPISODES)
        _require(train[0].shape == (6000,16)
                 and valid[0].shape == (1920,16), "TRAINING_SOURCE_DRIFT")
        all_results = []
        for native_seed, router_seed in zip(
            cfg["training"]["native_seeds"], cfg["training"]["router_seeds"],
            strict=True):
            models = c8._fit_trainable_arms(router_seed, train, valid)
            result = _native_checkpoint_evals(native_seed, models, cfg)
            result["p7_seed"] = router_seed
            result["trained_model_fit_receipts"] = models["source_training"]
            all_results.append(result)
    finally:
        _rng_restore(host_rng)
    duration = time.perf_counter()-start
    _require(duration <= cfg["budget"]["max_wall_seconds"], "BUDGET_EXCEEDED")
    counts = {}
    for condition in cfg["transfer"]["conditions"]:
        matches = [a for rec in all_results for a in rec["assays"]
                   if a["condition"] == condition]
        _require(len(matches) == 6, "CONDITION_INCOMPLETE")
        counts[condition] = {
            "trial_count": 6,
            "diagnostic_support_of_6": {
                name: sum(int(a["diagnostic_hypotheses"][name]) for a in matches)
                for name in matches[0]["diagnostic_hypotheses"]
            },
            "mean_mature_mse": {
                arm: sum(a["metrics"][arm]["mature_mse"] for a in matches)/6
                for arm in ARMS
            },
            "mean_mature_route_accuracy": {
                arm: sum(a["metrics"][arm]["mature_route_accuracy"]
                         for a in matches)/6
                for arm in ARMS
            },
            "mean_mature_coverage": sum(a["mature_events"] for a in matches)/576,
        }
    return {
        "schema": SCHEMA, "status": "PASS_FROZEN_ACTION_LAW_TRANSFER_DEV_INTEGRITY",
        "full_records": all_results, "native_seed_count": len(all_results),
        "distinct_dev_trace_count": 18,
        "condition_diagnostics": counts,
        "frozen_protocol_sha256": hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        "source_sha": factual.source_sha(),
        "original_p7_source_blob": c8.P7_ORIGINAL_BLOB,
        "independently_qualified_scientific_worlds": 0,
        "original_science_first_use": "NOT_RUN",
        "matched_flops_or_verified_convergence": False,
        "accepted_model_promoted": False,
        "running_trainer_modified": False,
        "whole_process_wall_seconds": round(duration,3),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    a = run()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(a, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({
        "status": a["status"], "distinct_dev_trace_count": a["distinct_dev_trace_count"],
        "condition_diagnostics": a["condition_diagnostics"],
        "original_science_first_use": a["original_science_first_use"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
