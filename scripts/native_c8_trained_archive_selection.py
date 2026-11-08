"""C8 DEV: trained P7 factual-history archive selector versus trained conventional MLP.

Bridges the ORIGINAL P7 gradient-trained source from independent PR #62 to
C7's genuine source-bound P3 step40/step80 saved native fast memories. BOTH
trainable selectors train on the same P7 factual DEV source, minibatch RNG,
600 AdamW updates and near-matched parameter counts. This is not matched
FLOPs, new science, a trained native world model, or production controller.
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

from native_transactional_resume import (
    prepare_isolated_candidate, _rng_snapshot, _rng_restore,
)
from native_training_checkpoint import validate_chain
from scripts.native_prequential_gate import NativeCausalEpisode
from scripts.native_c7_p6_factual_evidence import FactualSlopeEvidence
from scripts import native_c7_history_archive_selection as c7
from scripts import native_p7_learned_router_dev as p7
from scripts import native_p3_continuity as factual

SCHEMA = "ttt-hybrid.c8.trained-history-archive-selection-result-dev.v1"
FROZEN = ROOT / "experiments" / "native_c8_trained_history_archive_dev.json"
P7_ORIGINAL_BLOB = "7db1fe42695f16bf2afe730f4348210156ac3c24"
ARMS = (
    "p7_trained_router", "trained_conventional_mlp", "p6_handcoded",
    "oracle_upper_bound", "always_recent", "p7_reversed_history",
    "p7_zero_history",
)


def _require(ok, why):
    if not ok:
        raise AssertionError("C8_" + why)


def _canonical(x):
    return (json.dumps(x, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode()


def _frozen():
    cfg = json.loads(FROZEN.read_text("utf-8"))
    d, t, e = cfg["dataset"], cfg["training"], cfg["evaluation"]
    _require(
        cfg["schema"] == "ttt-hybrid.c8.trained-history-archive-selection-dev.v1"
        and cfg["source_parent_c7_sha"] == "b7f0b53ca94d00d63bea1ae83e4093d4201d20fb"
        and cfg["p7_original"]["script_git_blob"] == P7_ORIGINAL_BLOB
        and t["trained_seeds"] == list(p7.INIT_SEEDS) == [303, 307, 311]
        and t["native_learner_seeds"] == list(factual.MODEL_INITIALIZATIONS)
        and t["adamw_updates"] == p7.UPDATES == 600
        and t["minibatch"] == p7.BATCH == 256
        and t["parameters"]["LearnedRouter"] == 828
        and t["parameters"]["ConventionalMLP"] == 849
        and d["history_pairs"] == p7.HISTORY == 8
        and d["train_seed"] == p7.TRAIN_DEV_SEED
        and d["validation_seed"] == p7.VALIDATION_DEV_SEED
        and d["train_examples"] == p7.TRAIN_EPISODES*(p7.EVENTS_PER_EPISODE-p7.HISTORY)
        and d["validation_examples"] == p7.VALID_EPISODES*(p7.EVENTS_PER_EPISODE-p7.HISTORY)
        and e["arms"] == list(ARMS)
        and e["events_per_order"] == 64
        and e["orders"] == c7._spec()["eval"]["context_orders"]
        and cfg["native"]["new_native_fast_writes_after_training"] == 0
        and cfg["native"]["new_native_optimizer_updates_after_training"] == 0
        and cfg["scientific"]["first_use_authorized"] is False
        and cfg["scientific"]["model_promotion"] is False,
        "FROZEN_SOURCE_OR_PROTOCOL_DRIFT")
    return cfg


def _weights_sha(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        a = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(a.shape)).encode())
        digest.update(str(a.dtype).encode())
        digest.update(a.numpy().tobytes())
    return digest.hexdigest()


def _sampler_sha(seed: int, count: int) -> str:
    """Same source P7 train_one seed+9999 minibatch generator, both arms."""
    gen = torch.Generator().manual_seed(seed + 9999)
    digest = hashlib.sha256()
    for _ in range(p7.UPDATES):
        indices = torch.randint(count, (p7.BATCH,), generator=gen)
        digest.update(indices.numpy().tobytes())
    return digest.hexdigest()


def signed_pair_route(model: torch.nn.Module,
                      completed_history: tuple[tuple[float, float], ...],
                      *, history_mode: str = "chronological") -> int:
    """Predict signed action effect from previous factual pairs only.

    Compute BOTH hypothetical action predictions using the same history.
    This supplies equal two-forward inference calls to the trained P7
    mixture and conventional MLP. Hypothetical *predictions* are okay;
    no hypothetical *outcomes* are ever observed or learned from.
    """
    if type(completed_history) is not tuple or history_mode not in (
        "chronological", "reversed", "zero"):
        raise ValueError("C8_HISTORY_ARGUMENT_INVALID")
    if len(completed_history) < p7.HISTORY:
        return 1  # identical cold-start tie break; never access future data
    last = completed_history[-p7.HISTORY:]
    if history_mode == "reversed":
        last = tuple(reversed(last))
    if history_mode == "zero":
        flat = [0.] * (p7.HISTORY * 2)
    else:
        flat = [float(x) for pair in last for x in pair]
    _require(len(flat) == 16 and all(math.isfinite(x) for x in flat),
             "NONFINITE_HISTORY")
    h = torch.tensor([flat], dtype=torch.float32)
    with torch.no_grad():
        y_plus, _ = model(h, torch.tensor([[1.]], dtype=torch.float32))
        y_minus, _ = model(h, torch.tensor([[-1.]], dtype=torch.float32))
    slope = .5 * (float(y_plus.item()) - float(y_minus.item()))
    _require(math.isfinite(slope), "NONFINITE_TRAINED_ROUTER")
    return 1 if slope >= 0. else -1


def _fit_trainable_arms(seed: int, train, validation):
    learned, lr = p7.train_one(p7.LearnedRouter, seed, train, validation)
    conventional, mlp = p7.train_one(p7.ConventionalMLP, seed, train, validation)
    _require(lr["training_updates"] == mlp["training_updates"] == p7.UPDATES
             and lr["distinct_training_examples"] == mlp["distinct_training_examples"]
             == train[0].shape[0] == 6000
             and lr["parameters"] == 828 and mlp["parameters"] == 849
             and lr["model_weights_changed_by_real_gradients"]
             and mlp["model_weights_changed_by_real_gradients"],
             "TRAINED_CONTROL_PARITY")
    return {
        "learned": learned.eval(), "conventional": conventional.eval(),
        "source_training": {
            "learned": lr, "conventional": mlp,
            "sampler_indices_sha256_both_arms": _sampler_sha(seed, len(train[0])),
            "trained_router_sha256": _weights_sha(learned),
            "trained_mlp_sha256": _weights_sha(conventional),
            "information_equal": True,
            "minibatch_order_equal": True,
            "updates_equal": True,
            "parameters_within_5pct": abs(828-849)/849 < .05,
            "flops_matched": False,
            "context_labels_used_in_train_loss": False,
        },
    }


def _read_only_route_eval(step80_model, old_native, recent_native, learned,
                          conventional, order: tuple[str, ...], cfg) -> dict:
    if list(order) not in cfg["evaluation"]["orders"]:
        raise ValueError("C8_UNREGISTERED_CONTEXT_ORDER")
    history: list[tuple[float, float]] = []
    rule = FactualSlopeEvidence()
    errors = {arm: [] for arm in ARMS}
    choices = {arm: [] for arm in ARMS}
    event_ledger = []
    with torch.no_grad():
        for i in range(64):
            regime = order[i // 16]  # evaluator only, never trained router input
            action = torch.tensor([[-1. if i % 2 == 0 else 1.]],
                                  dtype=torch.float32)
            prior = tuple(history)
            route = {
                "p7_trained_router": signed_pair_route(learned, prior),
                "trained_conventional_mlp": signed_pair_route(conventional, prior),
                "p6_handcoded": rule.choice,
                "oracle_upper_bound": 1 if regime == "old" else -1,
                "always_recent": -1,
                "p7_reversed_history": signed_pair_route(
                    learned, prior, history_mode="reversed"),
                "p7_zero_history": signed_pair_route(
                    learned, prior, history_mode="zero"),
            }
            preds = {arm: c7.predict_with_archived_fast(
                step80_model, recent_native, old_native, action, choice)
                for arm, choice in route.items()}
            observed = factual.analytic_consequence(
                action, 0 if regime == "old" else 40)
            per_event_errors = {}
            for arm in ARMS:
                err = float((preds[arm]-observed).square().item())
                _require(math.isfinite(err), "NONFINITE_EVALUATION")
                errors[arm].append(err)
                choices[arm].append(route[arm])
                per_event_errors[arm] = err
            event_ledger.append({
                "event": i, "action": float(action.item()),
                "ground_truth_regime_evaluator_only": regime,
                "pre_outcome_route": dict(route),
                "prequential_squared_errors": per_event_errors,
                "factual_consequence_revealed_after_predictions":
                    float(observed.item()),
                "history_count_before_prediction": len(prior),
            })
            history.append((float(action.item()), float(observed.item())))
            # Source P6 detector updates only AFTER event's predictions.
            rule.observe(action, observed)
    _require(len(history) == rule.observations == 64
             and all(len(errors[a]) == len(choices[a]) == 64 for a in ARMS),
             "HISTORY_LEAKAGE_OR_LENGTH")
    late = [i for block in range(4) for i in range(block*16+4, block*16+16)]
    immediate = [i for block in range(4) for i in range(block*16, block*16+4)]
    metrics = {}
    for arm in ARMS:
        metrics[arm] = {
            "late_mse": sum(errors[arm][i] for i in late)/len(late),
            "immediate_mse": sum(errors[arm][i] for i in immediate)/len(immediate),
            "late_context_accuracy": sum(
                choices[arm][i] == (1 if order[i // 16] == "old" else -1)
                for i in late)/len(late),
            "raw_squared_errors": errors[arm],
            "pre_outcome_route_choices": choices[arm],
        }
    # Reuse original C7 source unchanged as an independent integration
    # oracle for fixed P6, true-label upper bound and always-current controls.
    c7_baseline = c7.evaluate_order(
        step80_model, recent_native, old_native, order, c7._spec())
    for a, c7arm in (
        ("p6_handcoded", "inferred_p6"),
        ("oracle_upper_bound", "oracle_upper_bound"),
        ("always_recent", "always_recent"),
    ):
        _require(metrics[a]["raw_squared_errors"]
                 == c7_baseline["measurements"][c7arm]["all_64_squared_errors"]
                 and metrics[a]["pre_outcome_route_choices"]
                 == c7_baseline["measurements"][c7arm]["pre_outcome_route_sequence"],
                 "C7_CONTROL_SOURCE_PARITY_" + a)
    thresholds = cfg["evaluation"]["hypotheses_diagnostic_only"]
    earned = metrics["p7_trained_router"]
    def gap(a):
        return metrics[a]["late_mse"] - earned["late_mse"]
    hypotheses = {
        "trained_late_context_accuracy_at_least_0p75":
            earned["late_context_accuracy"] >= thresholds["trained_router_late_accuracy_at_least"],
        "trained_beats_always_recent_late_by_0p10":
            gap("always_recent") >= thresholds["trained_router_beats_always_recent_late_mse_by"],
        "trained_beats_trained_mlp_late_by_0p01":
            gap("trained_conventional_mlp") >= thresholds["trained_router_beats_trained_mlp_late_mse_by"],
        "trained_beats_shuffled_history_late_by_0p05":
            gap("p7_reversed_history") >= thresholds["trained_router_beats_reversed_history_late_mse_by"],
        "trained_beats_zero_history_late_by_0p10":
            gap("p7_zero_history") >= thresholds["trained_router_beats_zero_history_late_mse_by"],
        "trained_near_oracle_late_within_0p05":
            earned["late_mse"] - metrics["oracle_upper_bound"]["late_mse"]
            <= thresholds["trained_router_near_oracle_late_mse_within"],
    }
    return {
        "order": list(order),
        "events": 64,
        "measurements": metrics,
        "hypotheses": hypotheses,
        "p7_and_mlp_trained_with_actual_factual_gradients": True,
        "truly_learned_native_context_router": False,
        "no_regime_label_in_training_or_prediction": True,
        "current_factual_result_revealed_after_all_routes": True,
        "p6_controls_exact_source_parity": True,
        "native_fast_memory_writes": 0, "native_optimizer_updates": 0,
        "ledger_sha256": hashlib.sha256(_canonical(event_ledger)).hexdigest(),
        "raw_event_ledger": event_ledger,
    }


def _one_native_seed(native_seed: int, trained, cfg) -> dict:
    identity = factual.identity()
    with tempfile.TemporaryDirectory(prefix="c8-native-saved-state-") as td:
        directory = Path(td)
        model, opt = factual.new_model_and_optimizer(native_seed)
        ep = NativeCausalEpisode(model)
        root = factual.save_completed(directory, identity, model, opt, ep, 0, None)
        for t in range(40):
            factual.transact(model, opt, ep, t)
        tip40 = factual.save_completed(directory, identity, model, opt, ep, 40, root)
        for t in range(40,80):
            factual.transact(model, opt, ep, t)
        tip80 = factual.save_completed(directory, identity, model, opt, ep, 80, tip40)
        _require(validate_chain(directory, tip80, identity)["parent_manifest_sha256"]
                 == tip40, "NATIVE_CHECKPOINT_LINEAGE")
        old = prepare_isolated_candidate(
            directory, tip_sha256=tip40, identity=identity,
            make_fresh_pair=lambda: factual.new_model_and_optimizer(native_seed),
            expected_position=40, expected_parent_sha256=root, minimum_steps=40,
            minimum_consumed_examples=40, protected_model=model,
            protected_optimizer=opt)
        recent = prepare_isolated_candidate(
            directory, tip_sha256=tip80, identity=identity,
            make_fresh_pair=lambda: factual.new_model_and_optimizer(native_seed),
            expected_position=80, expected_parent_sha256=tip40, minimum_steps=80,
            minimum_consumed_examples=80, protected_model=old.model,
            protected_optimizer=old.optimizer)
        frozen_model = deepcopy(recent.model.state_dict())
        frozen_opt = deepcopy(recent.optimizer.state_dict())
        old_native = deepcopy(old.native_state)
        recent_native = deepcopy(recent.native_state)
        learned_before = _weights_sha(trained["learned"])
        conventional_before = _weights_sha(trained["conventional"])
        orders = [
            _read_only_route_eval(
                recent.model, old.native_state, recent.native_state,
                trained["learned"], trained["conventional"], tuple(order), cfg)
            for order in cfg["evaluation"]["orders"]
        ]
        _require(factual._exact_equal(recent.model.state_dict(), frozen_model)
                 and factual._exact_equal(recent.optimizer.state_dict(), frozen_opt)
                 and factual._exact_equal(old.native_state, old_native)
                 and factual._exact_equal(recent.native_state, recent_native)
                 and _weights_sha(trained["learned"]) == learned_before
                 and _weights_sha(trained["conventional"]) == conventional_before,
                 "TRAINED_WEIGHTS_OR_SAVED_MEMORY_MUTATED")
        _require(not (directory/"accepted").exists()
                 and not (directory/"latest").exists(),
                 "UNAUTHORIZED_ACCEPTED_MODEL")
    return {
        "native_seed": native_seed,
        "checkpoint_root_sha256": root,
        "checkpoint_step40_sha256": tip40,
        "checkpoint_step80_sha256": tip80,
        "orders": orders,
        "source_trained_models_unchanged": True,
        "source_native_model_optimizer_and_archives_unchanged": True,
        "science_first_use": "NOT_RUN",
        "model_promoted": False,
    }


def run() -> dict:
    if (os.environ.get("NATIVE_C8_ENGINEERING_ONLY", "1") != "1"
            or os.environ.get("NATIVE_P3_ENGINEERING_ONLY", "1") != "1"
            or os.environ.get("NATIVE_P7_DEV_ONLY", "1") != "1"):
        raise RuntimeError("C8_SCIENCE_FIRST_USE_FORBIDDEN")
    started = time.perf_counter()
    torch.set_num_threads(1)
    host_rng = _rng_snapshot()
    try:
        cfg = _frozen()
        train = p7.synthetic_dev(p7.TRAIN_DEV_SEED, p7.TRAIN_EPISODES)
        validate = p7.synthetic_dev(p7.VALIDATION_DEV_SEED, p7.VALID_EPISODES)
        _require(train[0].shape == (6000,16)
                 and validate[0].shape == (1920,16),
                 "P7_DATA_SOURCE_DRIFT")
        results = []
        for native_seed, p7seed in zip(
            cfg["training"]["native_learner_seeds"],
            cfg["training"]["trained_seeds"],
            strict=True,
        ):
            trained = _fit_trainable_arms(p7seed, train, validate)
            rec = _one_native_seed(native_seed, trained, cfg)
            rec["trained_p7_init_seed"] = p7seed
            rec["matched_training"] = trained["source_training"]
            results.append(rec)
    finally:
        _rng_restore(host_rng)
    elapsed = time.perf_counter()-started
    _require(elapsed <= cfg["resources"]["max_wall_seconds"],
             "CPU_DEV_WALL_BUDGET")
    names = list(results[0]["orders"][0]["hypotheses"])
    counts = {
        n: sum(int(order["hypotheses"][n])
               for rec in results for order in rec["orders"])
        for n in names
    }
    return {
        "schema": SCHEMA,
        "status": "PASS_TRAINED_P7_NATIVE_ARCHIVE_BRIDGE_INTEGRITY_DEV",
        "original_p7_source_blob": P7_ORIGINAL_BLOB,
        "source_sha": factual.source_sha(),
        "frozen_protocol_sha256": hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        "records": results, "record_count": len(results),
        "orders_per_record": 2, "hypothesis_support_out_of_6": counts,
        "scientific_first_use": "NOT_RUN",
        "trainable_p7_and_conventional_mlp_actually_fit": True,
        "equal_information_updates_minibatch_indices": True,
        "flops_or_full_compute_matched": False,
        "original_scientific_holdout": False, "scientifically_accepted": False,
        "active_trainer_activated": False, "model_promoted": False,
        "whole_process_wall_seconds": round(elapsed, 3),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", type=Path, required=True)
    args = ap.parse_args()
    result = run()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "record_count": result["record_count"],
        "hypothesis_support_out_of_6": result["hypothesis_support_out_of_6"],
        "records": [
            {"native_seed": r["native_seed"],
             "p7_seed": r["trained_p7_init_seed"],
             "validation_mse_p7": r["matched_training"]["learned"]["final_validation_mse"],
             "validation_mse_mlp": r["matched_training"]["conventional"]["final_validation_mse"],
             "orders": [
                 {"order": x["order"],
                  "late_mse": {k: v["late_mse"] for k,v in x["measurements"].items()},
                  "late_route_accuracy": {k: v["late_context_accuracy"]
                                          for k,v in x["measurements"].items()}}
                 for x in r["orders"]]}
            for r in result["records"]],
        "scientific_first_use": result["scientific_first_use"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
