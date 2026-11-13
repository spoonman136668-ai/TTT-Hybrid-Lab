"""F8-R10: frozen paired-world DEV acceptance logic, NOT scientific first-use.

All measurements here must be explicitly FABRICATED_DEV_ONLY. This module
cannot import, read or admit source-owner F8 observations. A future authorized
native owner must separately qualify signed effects, freezes and thresholds.
The unit of resampling is one of 16 physical world seeds, not 352 sessions.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
FREEZE = HERE / "p85mr_f8r10_paired_science_freeze.json"
FREEZE_BLOB = "b473cc9021f7cd8662ef62edc4609e405e800bfd"
R5_PROPOSAL = HERE / "p85mr_f8r5_owner_science_trial_proposal.json"
R5_BLOB = "50cf2ff33f62d6ea6a25a7c789921838826d0094"
R6_BUDGET = HERE / "p85mr_f8r6_full_action_budget_freeze.json"
R6_BLOB = "59475472a4c316fa04a3510e4138a872abc88a91"
TRAINED_ARMS = (
    "original_hard_depth_HybridPilot",
    "F8_surrogate_hard_forward_native",
    "28_coefficient_linear7_ridge",
    "140_coefficient_quadratic35_ridge",
    "fully_trained_256_state_SSM",
    "capacity_matched_conventional_SSM",
    "trained_conventional_behavior_policy",
)
CANDIDATES = TRAINED_ARMS[:2]
BASELINES = TRAINED_ARMS[2:] + ("frozen_handwritten_feedback_policy",)
SEEDS = (985201, 985203, 985207)
WORLD_COUNT = 16
WORLD_SEED_BASE = 1824300000
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 1824990000
HORIZON = 96
ROWS_EXPECTED = 352
MAX_SOURCE_EFFECTS = 33792
MAX_MODEL_QUERIES = 67584
EXAMPLE_ONLY = "FABRICATED_DEV_ONLY"


def git_blob(raw: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
    ).hexdigest()


def _load_exact(path: Path, sha: str) -> dict:
    raw = path.read_bytes()
    if git_blob(raw) != sha:
        raise ValueError("F8R10_IMMUTABLE_BLOB_DRIFT")
    return json.loads(raw)


def locked() -> dict:
    f = _load_exact(FREEZE, FREEZE_BLOB)
    r5 = _load_exact(R5_PROPOSAL, R5_BLOB)
    r6 = _load_exact(R6_BUDGET, R6_BLOB)
    p = f["test_plan"]
    b = f["frozen_primary"]
    if (f["schema"] !=
            "ttt-hybrid.f8r10-paired-world-scientific-power-admission-dev.v1"
            or f["stage"] !=
                "FROZEN_BEFORE_IMPLEMENTATION_OR_F8_SCIENTIFIC_SOURCE_FIRST_USE"
            or tuple(p["trained_arms"]) != TRAINED_ARMS
            or tuple(p["model_seeds"]) != SEEDS
            or p["paired_worlds"] != WORLD_COUNT
            or p["models_per_world"] != 22
            or p["physical_trial_sessions"] != ROWS_EXPECTED
            or p["world_seed_base"] != WORLD_SEED_BASE
            or p["max_trial_selected_actions"] != MAX_SOURCE_EFFECTS
            or p["max_trial_model_queries"] != MAX_MODEL_QUERIES
            or tuple(b["candidate_arms"]) != CANDIDATES
            or tuple(b["baseline_pool"]) != BASELINES
            or b["bootstrap"]["resamples"] != BOOTSTRAP_N
            or b["bootstrap"]["seed"] != BOOTSTRAP_SEED
            or f["permissions"]["owner_firstuse"] is not False
            or f["permissions"]["scientific_pass"] is not False
            or r5["authority"]["model_selected_action_FirstUse"] is not False
            or r6["phase_first_use"]["sealed_model_selected_test"]["approved"]
                is not False
            or r6["test"]["max_sessions"] != ROWS_EXPECTED):
        raise ValueError("F8R10_OWNER_OR_FROZEN_METRIC_DRIFT")
    return f


def policy_instances() -> tuple[tuple[str, int | str], ...]:
    return (tuple((arm, seed) for arm in TRAINED_ARMS for seed in SEEDS)
            + (("frozen_handwritten_feedback_policy", "fixed"),))


def session_id(arm: str, seed: int | str, world: int) -> str:
    return f"dev:f8r10:{arm}:{seed}:{world}"


def audit_fabricated_panel(records: list[dict]) -> dict:
    """Prohibit real owner data; all declared records must be synthetic."""
    locked()
    if type(records) is not list or len(records) != ROWS_EXPECTED:
        raise ValueError("F8R10_INCOMPLETE_OR_INVALID_PANEL")
    identities = policy_instances()
    expected = {(arm, seed, w) for arm, seed in identities
                for w in range(WORLD_COUNT)}
    actual: dict[tuple[str, int | str, int], dict] = {}
    keys = {"classification", "session_id", "policy_arm", "model_seed",
            "world_index", "world_seed", "mean_stability_cost",
            "survival_steps", "selected_actions", "hypothetical_queries"}
    for r in records:
        if type(r) is not dict or set(r) != keys:
            raise ValueError("F8R10_RECORD_SHAPE_OR_SOURCE_UNTRUSTED")
        arm, seed, w = r["policy_arm"], r["model_seed"], r["world_index"]
        if (r["classification"] != EXAMPLE_ONLY
                or type(arm) is not str or
                type(seed) not in (int, str)
                or type(w) is not int or w not in range(WORLD_COUNT)
                or (arm, seed, w) not in expected
                or (arm, seed, w) in actual):
            raise ValueError("F8R10_UNTRUSTED_OR_DUPLICATED_POLICY_WORLD")
        if (r["session_id"] != session_id(arm, seed, w)
                or type(r["world_seed"]) is not int
                or r["world_seed"] != WORLD_SEED_BASE + w
                or type(r["mean_stability_cost"]) not in (int, float)
                or not math.isfinite(r["mean_stability_cost"])
                or r["mean_stability_cost"] < 0
                or type(r["survival_steps"]) is not int
                or type(r["selected_actions"]) is not int
                or not 1 <= r["survival_steps"] <= HORIZON
                or r["survival_steps"] != r["selected_actions"]
                or type(r["hypothetical_queries"]) is not int
                or r["hypothetical_queries"] !=
                    (0 if seed == "fixed" else 2 * r["selected_actions"])):
            raise ValueError("F8R10_UNTRUSTED_EFFECT_OR_QUERY_ACCOUNTING")
        actual[(arm, seed, w)] = r
    if expected != set(actual):
        raise ValueError("F8R10_MISSING_PAIRED_WORLDS")
    effects = sum(r["selected_actions"] for r in actual.values())
    queries = sum(r["hypothetical_queries"] for r in actual.values())
    if effects > MAX_SOURCE_EFFECTS or queries > MAX_MODEL_QUERIES:
        raise ValueError("F8R10_GLOBAL_TRIAL_CAP_EXCEEDED")
    return {
        "schema": "ttt-hybrid.f8r10-fabricated-panel-accounting.v1",
        "classification": EXAMPLE_ONLY,
        "independent_world_clusters": WORLD_COUNT,
        "policy_sessions": len(actual),
        "actual_source_effects": 0,
        "fictional_selected_effects": effects,
        "fictional_hypothetical_queries": queries,
        "owner_source_firstuse_authorized": False,
        "scientific_pass": False,
    }


def _pct(values: list[float], fraction: float) -> float:
    """Fixed-index descriptive percentile with no data-dependent tuning."""
    sorted_values = sorted(values)
    return sorted_values[int((len(sorted_values)-1) * fraction)]


def evaluate_fabricated_panel(records: list[dict]) -> dict:
    """Descriptive *hypothetical* precommitted gate, never SCIENTIFIC_PASS."""
    tally = audit_fabricated_panel(records)
    measurements = {(r["policy_arm"], r["model_seed"]):
                    [None] * WORLD_COUNT for r in records}
    for r in records:
        measurements[(r["policy_arm"], r["model_seed"])][r["world_index"]] = (
            float(r["mean_stability_cost"]), r["survival_steps"])
    baselines = [(arm, seed) for arm, seed in policy_instances()
                 if arm in BASELINES]
    def average(key: tuple, indices: list[int], pos: int) -> float:
        return sum(measurements[key][w][pos] for w in indices) / len(indices)
    full = list(range(WORLD_COUNT))
    best = min(baselines, key=lambda k: (average(k, full, 0), str(k)))
    best_cost = average(best, full, 0)
    if best_cost <= 0:
        # A baseline already achieves zero aggregate cost; cannot establish
        # a 10% strict relative gain from nonnegative costs.
        raise ValueError("F8R10_STRONGEST_CONTROL_ZERO_COST")
    best_survival = average(best, full, 1)
    rng = random.Random(BOOTSTRAP_SEED)
    draws = [[rng.randrange(WORLD_COUNT) for _ in range(WORLD_COUNT)]
             for _ in range(BOOTSTRAP_N)]
    results = {}
    for arm in CANDIDATES:
        model_seeds = []
        for seed in SEEDS:
            key = (arm, seed)
            cost = average(key, full, 0)
            survival = average(key, full, 1)
            difference = best_cost - cost
            gains = []
            for worlds in draws:
                # Re-select from the SAME FROZEN 16 conventional instances on
                # each paired cluster bootstrap, never choosing model on an
                # unregistered world or treating model seeds as new clusters.
                b = min(average(control, worlds, 0) for control in baselines)
                gains.append(b - average(key, worlds, 0))
            lower = _pct(gains, .025)
            upper = _pct(gains, .975)
            margin = (best_cost - cost) / best_cost
            seed_pass = (margin >= .10 and lower > 0
                         and survival >= best_survival)
            model_seeds.append({
                "seed": seed, "mean_cost": cost,
                "mean_survival_steps": survival,
                "relative_gain_vs_strongest": margin,
                "paired_world_mean_difference": difference,
                "paired_bootstrap_2p5_difference": lower,
                "paired_bootstrap_97p5_difference": upper,
                "hypothetical_seed_gate_pass": seed_pass,
            })
        results[arm] = {
            "seeds": model_seeds,
            "hypothetical_2_of_3_gate_pass": (
                sum(z["hypothetical_seed_gate_pass"] for z in model_seeds) >= 2),
        }
    return {**tally,
        "frozen_candidate_source": "UNAPPROVED_NOT_RUN",
        "baseline_selection": "predeclared strongest of 16 conventional instances",
        "strongest_baseline": {"arm": best[0], "seed": best[1],
                              "mean_cost": best_cost,
                              "mean_survival_steps": best_survival},
        "bootstrap": {"resamples": BOOTSTRAP_N,
                      "cluster_count": WORLD_COUNT,
                      "seed": BOOTSTRAP_SEED,
                      "independent_policy_sessions": False},
        "candidate_results": results,
        "prospective_scientific_acceptance": "BLOCKED_OWNER_NOT_AUTHORIZED",
        "scientific_pass": False,
    }


def refuse_real_firstuse(*_args, **_kwargs):
    raise PermissionError("F8R10_F8_OWNER_FIRSTUSE_NOT_AUTHORIZED")
