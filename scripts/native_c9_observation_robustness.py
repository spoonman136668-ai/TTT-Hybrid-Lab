"""C9 DEV: genuinely trained P7/MLP archived-memory routing under observation stress.

Only public development variants of the familiar P3 signed-gain family.
Intervention actions and factual feedback masks vary. A selector receives
ONLY past *revealed*, completed (action, consequence) pairs, never a latent
context, withheld consequence, future action, or current target. Native
checkpoints are read-only; no extra optimizer or fast-memory updates.
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

from native_transactional_resume import prepare_isolated_candidate, _rng_snapshot, _rng_restore
from native_training_checkpoint import validate_chain
from scripts.native_prequential_gate import NativeCausalEpisode
from scripts.native_c7_p6_factual_evidence import FactualSlopeEvidence
from scripts import native_c7_history_archive_selection as c7
from scripts import native_c8_trained_archive_selection as c8
from scripts import native_p7_learned_router_dev as p7
from scripts import native_p3_continuity as factual

SCHEMA = "ttt-hybrid.c9.stress-result-dev.v1"
FROZEN = ROOT / "experiments" / "native_c9_observation_stress_dev_frozen.json"
ARMS = (
    "p7_trained_router", "trained_conventional_mlp", "p6_handcoded",
    "always_recent", "oracle_evaluator_only", "p7_zero_history",
)


def _require(ok, reason):
    if not ok:
        raise AssertionError("C9_" + reason)


def _canonical(data):
    return (json.dumps(data, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode()


def _frozen():
    c = json.loads(FROZEN.read_text("utf-8"))
    t, e, s = c["training"], c["stress"], c["scientific"] if "scientific" in c else c["qualification"]
    _require(
        c["schema"] == "ttt-hybrid.c9.frozen-dev-observation-stress.v1"
        and c["parent_c8_sha"] == "dc9288552d91c1f56051b0d348ca8ca8e98e2bb4"
        and c["p7_original_source_blob"] == c8.P7_ORIGINAL_BLOB
        and t["p7_train_seeds"] == list(p7.INIT_SEEDS) == [303, 307, 311]
        and t["native_model_seeds"] == list(factual.MODEL_INITIALIZATIONS)
        and t["p7_real_adamw_updates"] == p7.UPDATES == 600
        and t["p7_minibatch"] == p7.BATCH == 256
        and t["p7_train_examples"] == 6000
        and t["p7_validation_examples"] == 1920
        and s["all_18_traces_required"] is True
        and s["hidden_scientific_first_use"] is False
        and e["stream_dev_seeds"] == [71931, 71947]
        and e["duration_events"] == 96
        and e["switch_probability"] == .06
        and e["conditions"] == [
            {"id": "balanced", "action_plus_probability": .5,
             "observed_feedback_probability": 1.},
            {"id": "action_imbalanced", "action_plus_probability": .9,
             "observed_feedback_probability": 1.},
            {"id": "feedback_sparse", "action_plus_probability": .5,
             "observed_feedback_probability": .45},
        ]
        and e["arms"] == list(ARMS)
        and e["native_updates_during_evaluation"] == 0
        and e["trainable_router_gradient_updates_during_evaluation"] == 0,
        "FROZEN_SOURCE_OR_PROTOCOL_DRIFT")
    return c


def make_trace(seed: int, native_seed: int, condition: dict, cfg: dict):
    """Factual latent context exists only inside evaluator's immutable plan."""
    if seed not in cfg["stress"]["stream_dev_seeds"]:
        raise ValueError("C9_UNREGISTERED_DEV_STREAM_SEED")
    if native_seed not in cfg["training"]["native_model_seeds"]:
        raise ValueError("C9_UNREGISTERED_NATIVE_SEED")
    conditions = cfg["stress"]["conditions"]
    if condition not in conditions:
        raise ValueError("C9_UNREGISTERED_OBSERVATION_CONDITION")
    index = conditions.index(condition)
    rng = np.random.default_rng(seed + native_seed * 37 + index * 1031)
    latent = 1 if rng.random() < .5 else -1
    last_switch_index = 0
    rows = []
    for i in range(cfg["stress"]["duration_events"]):
        if i > 0 and rng.random() < cfg["stress"]["switch_probability"]:
            latent = -latent
            last_switch_index = i
        a = 1 if rng.random() < condition["action_plus_probability"] else -1
        noise = float(rng.normal(0., cfg["stress"]["observational_gaussian_noise_sigma"]))
        reveal = bool(rng.random() < condition["observed_feedback_probability"])
        rows.append({
            "index": i, "action": a,
            "latent_regime_evaluator_only": "old" if latent == 1 else "recent",
            "since_context_switch_evaluator_only": i-last_switch_index,
            "noise_evaluator_only": noise,
            "reveal_to_all_after_prediction": reveal,
        })
    _require(len(rows) == 96 and all(math.isfinite(x["noise_evaluator_only"])
                                     for x in rows), "TRACE_CONSTRUCTION")
    return rows


def _condition_metrics(arm, ledger, choices, errors, times, cfg):
    n = len(ledger)
    _require(n == len(errors) == len(choices) == 96, "TRACE_LENGTH")
    mature = [i for i, e in enumerate(ledger)
              if e["since_context_switch_evaluator_only"] >= 8]
    immediate = [i for i, e in enumerate(ledger)
                 if e["since_context_switch_evaluator_only"] < 8]
    _require(bool(mature) and bool(immediate), "EMPTY_MATURE_OR_IMMEDIATE_PROBE")
    accuracy = lambda ii: sum(
        choices[i] == (1 if ledger[i]["latent_regime_evaluator_only"] == "old" else -1)
        for i in ii) / len(ii)
    return {
        "all_mse": sum(errors)/n,
        "mature_mse": sum(errors[i] for i in mature)/len(mature),
        "immediate_mse": sum(errors[i] for i in immediate)/len(immediate),
        "mature_route_accuracy": accuracy(mature),
        "mature_samples": len(mature),
        "immediate_samples": len(immediate),
        "inference_wall_seconds": times,
        "raw_squared_errors": errors,
        "pre_outcome_choices": choices,
    }


def evaluate_trace(model, old_native, recent_native, trained_router,
                   trained_mlp, trace, cfg):
    """Causal-order online stress over identical per-arm intervention outcomes."""
    _require(len(trace) == cfg["stress"]["duration_events"] == 96, "TRACE_LENGTH")
    history = []
    p6rule = FactualSlopeEvidence()
    squared = {a: [] for a in ARMS}
    choices = {a: [] for a in ARMS}
    arm_times = {a: 0. for a in ARMS}
    record = []
    for step in trace:
        action = torch.tensor([[float(step["action"])]], dtype=torch.float32)
        prior = tuple(history)
        ctx = step["latent_regime_evaluator_only"]  # hidden evaluator-only
        # All nonoracle routes depend on a PRIOR factual record or fixed action.
        route = {
            "p7_trained_router": c8.signed_pair_route(trained_router, prior),
            "trained_conventional_mlp": c8.signed_pair_route(trained_mlp, prior),
            "p6_handcoded": p6rule.choice,
            "always_recent": -1,
            "oracle_evaluator_only": 1 if ctx == "old" else -1,
            "p7_zero_history": c8.signed_pair_route(
                trained_router, prior, history_mode="zero"),
        }
        predictions = {}
        for arm, choice in route.items():
            t0 = time.perf_counter()
            pred = c7.predict_with_archived_fast(
                model, recent_native, old_native, action, choice)
            arm_times[arm] += time.perf_counter()-t0
            predictions[arm] = pred
        # Consequence computed AFTER ALL routing/prediction functions return.
        with torch.no_grad():
            target = factual.analytic_consequence(
                action, 0 if ctx == "old" else 40)
            target = target + step["noise_evaluator_only"]
        observed = float(target.item())
        realized_errors = {}
        for arm in ARMS:
            err = float((predictions[arm]-target).square().item())
            _require(math.isfinite(err) and err >= 0., "NONFINITE_SCORE")
            squared[arm].append(err)
            choices[arm].append(route[arm])
            realized_errors[arm] = err
        reveal = bool(step["reveal_to_all_after_prediction"])
        record.append({
            "index": step["index"], "action": step["action"],
            "latent_regime_evaluator_only": ctx,
            "since_context_switch_evaluator_only":
                step["since_context_switch_evaluator_only"],
            "factual_history_count_before_prediction": len(prior),
            "selected_archives_before_outcome": dict(route),
            "factual_consequence_evaluator_only": observed,
            "outcome_revealed_to_selectors_after_pred": reveal,
            "per_arm_squared_error": realized_errors,
        })
        if reveal:
            history.append((float(step["action"]), observed))
            p6rule.observe(action, target)
    total_reveals = sum(int(e["outcome_revealed_to_selectors_after_pred"])
                        for e in record)
    _require(p6rule.observations == total_reveals
             and len(record) == 96, "OBSERVATION_MASK_PARITY")
    metrics = {
        arm: _condition_metrics(arm, record, choices[arm], squared[arm],
                                arm_times[arm], cfg)
        for arm in ARMS
    }
    t = cfg["stress"]["hypotheses_diagnostic_only"]
    learned = metrics["p7_trained_router"]
    def improvement(ref):
        return metrics[ref]["mature_mse"] - learned["mature_mse"]
    hypotheses = {
        "trained_route_accuracy_at_least_0p75":
            learned["mature_route_accuracy"] >= t[
                "p7_mature_route_accuracy_at_least"],
        "trained_beats_always_recent_by_0p05":
            improvement("always_recent") >= t[
                "p7_mature_mse_better_than_always_recent_by_at_least"],
        "trained_beats_trained_mlp_by_0p01":
            improvement("trained_conventional_mlp") >= t[
                "p7_mature_mse_better_than_conventional_by_at_least"],
        "trained_beats_p6_fixed_by_0p02":
            improvement("p6_handcoded") >= t[
                "p7_mature_mse_better_than_p6_by_at_least"],
        "trained_beats_zero_history_by_0p05":
            improvement("p7_zero_history") >= t[
                "p7_mature_mse_better_than_zero_history_by_at_least"],
    }
    return {
        "status": "PASS_OBSERVATIONAL_TRACE_INTEGRITY_DEV",
        "events": 96,
        "context_switches": sum(
            1 for e in record[1:] if e["since_context_switch_evaluator_only"] == 0),
        "revealed_factual_samples": total_reveals,
        "feedback_fraction": total_reveals/96,
        "measurements": metrics,
        "hypotheses": hypotheses,
        "ledger_sha256": hashlib.sha256(_canonical(record)).hexdigest(),
        "ledger": record,
        "model_and_native_fast_writes": 0,
        "trained_model_gradient_updates_during_eval": 0,
        "oracle_evaluator_only": True,
        "current_outcome_seen_by_router": False,
        "unrevealed_factual_outcomes_seen_by_router": False,
    }


def _one_native_seed(seed, trained, cfg):
    identity = factual.identity()
    with tempfile.TemporaryDirectory(prefix="hybrid-c9-source-locked-") as tdir:
        rootdir = Path(tdir)
        model, optimizer = factual.new_model_and_optimizer(seed)
        ep = NativeCausalEpisode(model)
        root = factual.save_completed(rootdir, identity, model, optimizer, ep, 0, None)
        for i in range(40):
            factual.transact(model, optimizer, ep, i)
        old_tip = factual.save_completed(
            rootdir, identity, model, optimizer, ep, 40, root)
        for i in range(40,80):
            factual.transact(model, optimizer, ep, i)
        new_tip = factual.save_completed(
            rootdir, identity, model, optimizer, ep, 80, old_tip)
        _require(validate_chain(rootdir, new_tip, identity)["parent_manifest_sha256"]
                 == old_tip, "CHECKPOINT_CHAIN")
        old = prepare_isolated_candidate(
            rootdir, tip_sha256=old_tip, identity=identity,
            make_fresh_pair=lambda: factual.new_model_and_optimizer(seed),
            expected_position=40, expected_parent_sha256=root,
            minimum_steps=40, minimum_consumed_examples=40,
            protected_model=model, protected_optimizer=optimizer)
        current = prepare_isolated_candidate(
            rootdir, tip_sha256=new_tip, identity=identity,
            make_fresh_pair=lambda: factual.new_model_and_optimizer(seed),
            expected_position=80, expected_parent_sha256=old_tip,
            minimum_steps=80, minimum_consumed_examples=80,
            protected_model=old.model, protected_optimizer=old.optimizer)
        source = (
            deepcopy(current.model.state_dict()),
            deepcopy(current.optimizer.state_dict()),
            deepcopy(old.native_state),
            deepcopy(current.native_state),
            c8._weights_sha(trained["learned"]),
            c8._weights_sha(trained["conventional"]),
        )
        traces = []
        for condition in cfg["stress"]["conditions"]:
            for dev_seed in cfg["stress"]["stream_dev_seeds"]:
                plan = make_trace(dev_seed, seed, condition, cfg)
                result = evaluate_trace(
                    current.model, old.native_state, current.native_state,
                    trained["learned"], trained["conventional"], plan, cfg)
                result["condition"] = condition["id"]
                result["stream_dev_seed"] = dev_seed
                result["native_init_seed"] = seed
                result["generated_trace_sha256"] = hashlib.sha256(
                    _canonical(plan)).hexdigest()
                traces.append(result)
        _require(
            factual._exact_equal(current.model.state_dict(), source[0])
            and factual._exact_equal(current.optimizer.state_dict(), source[1])
            and factual._exact_equal(old.native_state, source[2])
            and factual._exact_equal(current.native_state, source[3])
            and c8._weights_sha(trained["learned"]) == source[4]
            and c8._weights_sha(trained["conventional"]) == source[5]
            and not (rootdir/"latest").exists()
            and not (rootdir/"accepted").exists(),
            "MODEL_OPTIMIZER_OR_MEMORY_MUTATION")
    return {
        "native_init_seed": seed,
        "checkpoint_root_sha256": root,
        "checkpoint_old_sha256": old_tip,
        "checkpoint_current_sha256": new_tip,
        "all_sources_models_and_native_memory_unchanged": True,
        "traces": traces,
    }


def run():
    if any(os.environ.get(flag, "1") != "1" for flag in (
        "NATIVE_C9_ENGINEERING_ONLY", "NATIVE_P3_ENGINEERING_ONLY",
        "NATIVE_P7_DEV_ONLY",
    )):
        raise RuntimeError("C9_ORIGINAL_SCIENCE_FIRST_USE_FORBIDDEN")
    started = time.perf_counter()
    torch.set_num_threads(1)
    host = _rng_snapshot()
    try:
        cfg = _frozen()
        training = p7.synthetic_dev(p7.TRAIN_DEV_SEED, p7.TRAIN_EPISODES)
        validation = p7.synthetic_dev(p7.VALIDATION_DEV_SEED, p7.VALID_EPISODES)
        _require(training[0].shape == (6000,16)
                 and validation[0].shape == (1920,16), "P7_SOURCE_FIT_DATA")
        records = []
        for native_seed, p7seed in zip(
            cfg["training"]["native_model_seeds"],
            cfg["training"]["p7_train_seeds"], strict=True):
            trained = c8._fit_trainable_arms(p7seed, training, validation)
            rec = _one_native_seed(native_seed, trained, cfg)
            rec["original_p7_model_init_seed"] = p7seed
            rec["full_trained_control_provenance"] = trained["source_training"]
            records.append(rec)
    finally:
        _rng_restore(host)
    elapsed = time.perf_counter()-started
    _require(elapsed <= cfg["resource"]["max_total_wall_seconds"],
             "TOTAL_DEV_WALL_BUDGET")
    _require(len(records) == 3
             and all(len(r["traces"]) == 6 for r in records),
             "INCOMPLETE_18_TRACE_DESIGN")
    per_condition = {}
    for cond in cfg["stress"]["conditions"]:
        cid = cond["id"]
        six = [trace for rec in records for trace in rec["traces"]
               if trace["condition"] == cid]
        _require(len(six) == 6, "INCOMPLETE_CONDITION")
        names = list(six[0]["hypotheses"])
        per_condition[cid] = {
            "trace_count": 6,
            "hypothesis_support_of_6": {
                k: sum(int(x["hypotheses"][k]) for x in six) for k in names
            },
            "mean_mature_mse": {
                arm: sum(x["measurements"][arm]["mature_mse"] for x in six)/6
                for arm in ARMS
            },
            "mean_mature_route_accuracy": {
                arm: sum(x["measurements"][arm]["mature_route_accuracy"]
                         for x in six)/6
                for arm in ARMS
            },
            "mean_feedback_revealed_fraction":
                sum(x["feedback_fraction"] for x in six)/6,
            "mean_context_switches": sum(x["context_switches"] for x in six)/6,
        }
    return {
        "schema": SCHEMA,
        "status": "PASS_TRAINED_MEMORY_ROUTING_DEV_STRESS_INTEGRITY",
        "source_sha": factual.source_sha(),
        "p7_source_blob": c8.P7_ORIGINAL_BLOB,
        "frozen_protocol_sha256": hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        "native_seed_count": len(records), "trace_count": 18,
        "condition_diagnostics": per_condition,
        "records": records,
        "scientific_first_use": "NOT_RUN",
        "unfamiliar_scientific_transfer": "NOT_TESTED",
        "flops_and_total_compute_matched": False,
        "model_promoted": False, "active_training_activated": False,
        "native_archived_memory_unchanged": True,
        "whole_process_wall_seconds": round(elapsed, 3),
    }


def main():
    import argparse
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--receipt", type=Path, required=True)
    args = pa.parse_args()
    d = run()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(d, sort_keys=True, indent=2)+"\n",
                            encoding="utf-8")
    print(json.dumps({
        "status": d["status"],
        "trace_count": d["trace_count"],
        "native_seed_count": d["native_seed_count"],
        "condition_diagnostics": d["condition_diagnostics"],
        "scientific_first_use": d["scientific_first_use"],
        "whole_process_wall_seconds": d["whole_process_wall_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
