"""G7-S25 conventional-control provenance and competence *pre-admission*.

Pure standard-library checks. There is NO training, calibration, model or
simulator import, frozen data read, candidate acceptance or scientific result.
Receipts may be fabricated; this validates logical contract consistency ONLY.
Actual source/run custody, baseline competence and scientific qualification
must be established separately on a registered first-use run.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

FROZEN_PROTOCOL_GIT_BLOB = "31633b341403729b7ffbb7d73f3a04098ad44380"
S20_SOURCE_GIT_BLOB = "ac3f9ddd5b879e53e62030707fb6419adaf732ca"
REQUIRED_ARMS = ("GRU_fully_trained", "RBF_KRR")
TRAIN_FIT_SEEDS = (6203, 6211, 6221)
VALIDATION_SEEDS = (6301, 6311, 6323)
DEVELOPMENT_SEEDS = (3101, 3109, 3119)
MIN_GRU_UPDATES = 640
MAX_GRU_UPDATES = 1600
VAL_INTERVAL = 64
VAL_WINDOW = 5
VAL_PLATEAU_MAX_RELATIVE_RANGE = 0.001
MAX_RSS_BYTES = 2_147_483_648


class BaselineAdmissionBlocked(ValueError):
    pass


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise BaselineAdmissionBlocked(reason)


def positive_finite(number: Any, reason: str) -> float:
    require(type(number) in (int, float) and math.isfinite(number) and number > 0,
            reason)
    return float(number)


def sha256_digest(value: Any, reason: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value), reason)
    return value


def validate_frozen_protocol(path: Path) -> dict:
    """Read and check registration only, NEVER execute its seed allocations."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload["schema"] ==
            "ttt-hybrid.g7-s25.contradiction-gated-continuous-ttt.v1",
            "FROZEN_PROTOCOL_SCHEMA_DRIFT")
    require(payload["provenance"]["s20_world_git_blob"] == S20_SOURCE_GIT_BLOB,
            "FROZEN_S20_SOURCE_DRIFT")
    seeds = payload["seeds"]
    require(tuple(seeds["fit"]) == TRAIN_FIT_SEEDS and
            tuple(seeds["validation"]) == VALIDATION_SEEDS and
            tuple(seeds["engineering_only"]) == DEVELOPMENT_SEEDS,
            "FROZEN_SPLIT_DRIFT")
    disjoint_roles = ("fit", "validation", "heldout_signflip",
                      "heldout_transfer", "negative", "engineering_only")
    groups = [seeds[k] for k in disjoint_roles]
    require(len({v for group in groups for v in group}) ==
            sum(len(group) for group in groups),
            "CROSS_SPLIT_SEED_COLLISION")
    authority = payload["authority"]
    require(authority["scientific_first_use_authorized"] is False and
            authority["no_training_execution"] is True and
            authority["no_merge_or_promotion"] is True,
            "FIRST_USE_GATE_CHANGED")
    require(payload["budgets"]["training_source_min_updates"] == MIN_GRU_UPDATES
            and payload["budgets"]["batch_worlds"] == 32
            and payload["budgets"]["max_rss_bytes"] == MAX_RSS_BYTES,
            "FROZEN_TRAINING_BUDGET_DRIFT")
    require(payload["arms"]["GRU_fully_trained"] and
            payload["arms"]["RBF_KRR"],
            "FROZEN_STRONG_BASELINES_MISSING")
    return payload


def _record_check(record: dict, arm: str, fit_seed: int, val_seed: int,
                  expected_candidate_sha: str) -> None:
    require(type(record) is dict, "BASELINE_RECEIPT_NOT_MAPPING")
    require(record.get("arm") == arm, "BASELINE_ARM_SUBSTITUTION")
    require(record.get("fit_seed") == fit_seed and
            record.get("validation_seed") == val_seed,
            "BASELINE_TRAIN_VALIDATION_SPLIT_DRIFT")
    require(record.get("candidate_sha") == expected_candidate_sha,
            "BASELINE_CANDIDATE_SHA_DRIFT")
    require(record.get("frozen_protocol_git_blob") == FROZEN_PROTOCOL_GIT_BLOB
            and record.get("s20_world_git_blob") == S20_SOURCE_GIT_BLOB,
            "BASELINE_SOURCE_PROVENANCE_DRIFT")
    for key in ("training_stream_sha256", "validation_stream_sha256",
                "checkpoint_sha256"):
        sha256_digest(record.get(key), "BASELINE_MISSING_" + key.upper())
    require(record["training_stream_sha256"] !=
            record["validation_stream_sha256"], "BASELINE_DATA_LEAKAGE")
    for key in record:
        require(not any(s in key.lower() for s in
                        ("heldout", "oracle", "test_seed", "future_target")),
                "BASELINE_FUTURE_INFORMATION_FIELD")
    for key in ("training_cpu_seconds", "validation_cpu_seconds",
                "checkpoint_bytes"):
        positive_finite(record.get(key), "BASELINE_INVALID_" + key.upper())
    rss = positive_finite(record.get("peak_rss_bytes"),
                          "BASELINE_INVALID_PEAK_RSS")
    require(rss <= MAX_RSS_BYTES, "BASELINE_RSS_CAP_EXCEEDED")
    val_mse = positive_finite(record.get("validation_mse"),
                              "BASELINE_INVALID_VALIDATION_MSE")
    naive_mse = positive_finite(record.get("static_no_dynamics_mse"),
                                "BASELINE_INVALID_NAIVE_MSE")
    require(val_mse < naive_mse,
            "BASELINE_HAS_NOT_BEAT_NAIVE_CONVENTIONAL_CONTROL")
    require(record.get("training_source_only") is True
            and record.get("validation_only_for_selection") is True,
            "BASELINE_PROVENANCE_NOT_DECLARED")


def _gru_logical_gate(record: dict) -> None:
    updates = record.get("updates")
    require(type(updates) is int and MIN_GRU_UPDATES <= updates <= MAX_GRU_UPDATES
            and updates % VAL_INTERVAL == 0,
            "GRU_UNDERTRAINED_OR_BUDGET_VIOLATION")
    require(record.get("batch_worlds") == 32 and
            record.get("train_world_exposures") == updates * 32,
            "GRU_SOURCE_WORLD_EXPOSURE_DRIFT")
    require(record.get("converged") is True and
            record.get("trainer_status") == "BASELINE_VALIDATION_PLATEAU",
            "GRU_NO_DECLARED_CONVERGENCE")
    require(record.get("width") == 32 and
            record.get("optimizer") == "AdamW_lr0.001_wd0.01",
            "GRU_PARAMETER_OR_TRAINER_DRIFT")
    values = record.get("validation_history")
    require(type(values) is list and len(values) >= VAL_WINDOW,
            "GRU_MISSING_FIVE_SPACED_VALIDATIONS")
    end_steps = [record["step"] for record in values]
    require(all(type(x) is int for x in end_steps) and
            all(a + VAL_INTERVAL == b for a, b in zip(end_steps, end_steps[1:]))
            and end_steps[0] >= MIN_GRU_UPDATES
            and end_steps[-1] == updates,
            "GRU_VALIDATION_CHECKPOINT_INTERVAL_DRIFT")
    mse = [positive_finite(row.get("mse"),
                           "GRU_INVALID_VALIDATION_WINDOW_MSE")
           for row in values]
    last = mse[-VAL_WINDOW:]
    require((max(last) - min(last)) / min(last) < VAL_PLATEAU_MAX_RELATIVE_RANGE,
            "GRU_VALIDATION_PLATEAU_NOT_ESTABLISHED")
    require(abs(mse[-1] - record["validation_mse"])
            <= 1e-10 * max(mse[-1], 1.0),
            "GRU_FINAL_VALIDATION_RECEIPT_DRIFT")


def _krr_logical_gate(record: dict) -> None:
    gamma = record.get("gamma")
    ridge = record.get("ridge")
    require(type(gamma) in (int, float) and gamma in (0.5, 2.0, 6.0)
            and type(ridge) in (int, float) and ridge in (0.01, 0.08),
            "KRR_NOT_FROZEN_CALIBRATION_GRID")
    require(type(record.get("train_query_samples")) is int and
            record["train_query_samples"] >= 16,
            "KRR_INSUFFICIENT_TRAIN_ONLY_EXPOSURE")
    require(record.get("normalization_fitted_on_train_only") is True
            and record.get("hyperparameter_selected_on_validation_only") is True,
            "KRR_ORACLE_OR_HELDOUT_CALIBRATION_NOT_EXCLUDED")
    require(record.get("history_width") == 32 and
            record.get("numeric_query_width") == 3,
            "KRR_NUMERIC_INPUT_DRIFT")


def validate_receipt_shapes_only(
    protocol: dict,
    candidate_sha: str,
    records: list[dict],
) -> dict:
    """Fail closed on malformed/missing claimed conventional controls.

    DO NOT interpret returned status as authenticated evidence, scientific
    competence or a fully trained baseline. This is a shape/logic check only.
    """
    sha256_digest(candidate_sha, "INVALID_CANDIDATE_SHA")
    require(protocol["authority"]["scientific_first_use_authorized"] is False,
            "FIRST_USE_GATE_CHANGED")
    require(type(records) is list and
            len(records) == len(TRAIN_FIT_SEEDS) * len(REQUIRED_ARMS),
            "MISSING_CONVENTIONAL_CONTROL_OR_SEED")
    mapping = {}
    for item in records:
        require(type(item) is dict, "BASELINE_RECEIPT_NOT_MAPPING")
        key = (item.get("arm"), item.get("fit_seed"))
        require(key not in mapping, "DUPLICATE_CONTROL_OR_SEED")
        mapping[key] = item
    for i, (train_seed, valid_seed) in enumerate(zip(
            TRAIN_FIT_SEEDS, VALIDATION_SEEDS)):
        r_gru = mapping.get(("GRU_fully_trained", train_seed))
        r_krr = mapping.get(("RBF_KRR", train_seed))
        require(r_gru is not None and r_krr is not None,
                "BASELINE_MISSING_MATCHED_CONVENTIONAL_CONTROL")
        _record_check(r_gru, "GRU_fully_trained", train_seed,
                      valid_seed, candidate_sha)
        _record_check(r_krr, "RBF_KRR", train_seed,
                      valid_seed, candidate_sha)
        require(r_gru["training_stream_sha256"] ==
                r_krr["training_stream_sha256"] and
                r_gru["validation_stream_sha256"] ==
                r_krr["validation_stream_sha256"],
                "CONVENTIONAL_CONTROLS_NOT_MATCHED_ON_OBSERVATIONS")
        require(r_gru["checkpoint_sha256"] != r_krr["checkpoint_sha256"],
                "BASELINE_CHECKPOINT_ALIASING")
        _gru_logical_gate(r_gru)
        _krr_logical_gate(r_krr)
    return {
        "schema": "ttt-hybrid.g7-s25.baseline-pre-admission.v1",
        "status": "PASS_RECEIPT_STRUCTURE_ONLY_NEVER_SCIENTIFIC_PASS",
        "controls": list(REQUIRED_ARMS),
        "fit_seed_count": len(TRAIN_FIT_SEEDS),
        "record_count": len(records),
        "source_fingerprint_claims_checked": True,
        "trained_baseline_execution_verified": False,
        "receipt_authenticity_verified": False,
        "information_and_training_cost_matched_scientifically": False,
        "full_regression": "UNVERIFIED",
        "scientific_first_use": "NOT_RUN",
        "deployment": "NOT_AUTHORIZED",
    }
