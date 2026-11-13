"""F8-R11: recompute the F8-R10 physical stability cost from DEV-only traces.

This adapter only accepts expressly FABRICATED_DEV_ONLY inputs, not real
CartPoleWind source or owner receipts. Even a successful synthetic panel is
never evidence of scientific pass, actor authority or observed first-use.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from experiments import p85mr_f8r10_paired_science as r10

HERE = Path(__file__).resolve().parent
FREEZE = HERE / "p85mr_f8r11_trace_scoring_freeze.json"
FREEZE_BLOB = "3aad48e4562395d6ddee21e8b054465fc38a3221"
R10_CODE = HERE / "p85mr_f8r10_paired_science.py"
R10_CODE_BLOB = "2f6be7a5d75f8ae5ddbf7e340a342c6d413b2095"
R5_CONTRACT = HERE / "p85mr_f8r5_owner_science_trial_proposal.json"
R5_CONTRACT_BLOB = "50cf2ff33f62d6ea6a25a7c789921838826d0094"

SCHEMA = "ttt-hybrid.f8r11-factual-trace-scoring-dev.v1"
CLASSIFICATION = "FABRICATED_DEV_ONLY"
EPISODE_KEYS = frozenset((
    "classification", "session_id", "policy_arm", "model_seed",
    "world_index", "world_seed", "events",
))
EVENT_KEYS = frozenset((
    "event_id", "state_before", "intent_selected_action",
    "selected_action", "state_after",
    "reward", "terminated", "truncated", "hypothetical_action_queries",
))
HORIZON = 96
ANGLE_SCALE = 0.2095
ANGULAR_VEL_SCALE = 3.5
ANGULAR_VEL_COEFF = 0.1
MISSING_STEP_PENALTY = 2.0
CONTINUITY_ABS_TOL = 1e-9


def git_blob(raw: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
    ).hexdigest()


def locked() -> dict:
    raw = FREEZE.read_bytes()
    if git_blob(raw) != FREEZE_BLOB:
        raise ValueError("F8R11_FROZEN_TRACE_METRIC_DRIFT")
    f = json.loads(raw)
    r10.locked()
    if (git_blob(R10_CODE.read_bytes()) != R10_CODE_BLOB
            or git_blob(R5_CONTRACT.read_bytes()) != R5_CONTRACT_BLOB
            or f.get("schema") != SCHEMA
            or f.get("stage") !=
                "FROZEN_BEFORE_R11_IMPLEMENTATION_TESTS_OR_ORIGINAL_SOURCE_ACCESS"
            or f["classification"] != CLASSIFICATION
            or f["parent"]["r10_evaluator_blob"] != R10_CODE_BLOB
            or f["parent"]["r5_owner_proposal_blob"] != R5_CONTRACT_BLOB
            or f["metric"]["horizon_steps"] != HORIZON
            or f["metric"]["angle_index"] != 2
            or f["metric"]["angular_velocity_index"] != 3
            or f["metric"]["angle_scale"] != ANGLE_SCALE
            or f["metric"]["angular_velocity_scale"] != ANGULAR_VEL_SCALE
            or f["metric"]["angular_velocity_coefficient"] != ANGULAR_VEL_COEFF
            or f["metric"]["missing_step_penalty"] != MISSING_STEP_PENALTY
            or f["metric"]["state_continuity_abs"] != CONTINUITY_ABS_TOL
            or f["authority"]["external_effects"] is not False
            or f["authority"]["owner_first_use"] is not False
            or f["authority"]["scientific_pass"] is not False):
        raise ValueError("F8R11_ORIGINAL_SCOPE_OR_AUTHORITY_DRIFT")
    return f


def _finite_real(value: object) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("F8R11_NONFINITE_OR_INVALID_PHYSICAL_VALUE")
    return float(value)


def _state(value: object) -> tuple[float, float, float, float]:
    if type(value) is not list or len(value) != 4:
        raise ValueError("F8R11_INVALID_PHYSICAL_STATE_SHAPE")
    return tuple(_finite_real(v) for v in value)


def score_fabricated_trace(trace: dict) -> dict:
    """Build an original R10 DEV summary; never accept caller-supplied cost."""
    locked()
    if type(trace) is not dict or frozenset(trace) != EPISODE_KEYS:
        raise ValueError("F8R11_INVALID_OR_UNTRUSTED_TRACE_SHAPE")
    arm, seed, world = (trace["policy_arm"], trace["model_seed"],
                        trace["world_index"])
    if (trace["classification"] != CLASSIFICATION
            or type(arm) is not str
            or type(seed) not in (int, str)
            or type(world) is not int
            or world not in range(r10.WORLD_COUNT)
            or (arm, seed) not in r10.policy_instances()
            or type(trace["world_seed"]) is not int
            or trace["world_seed"] != r10.WORLD_SEED_BASE + world
            or trace["session_id"] != r10.session_id(arm, seed, world)):
        raise ValueError("F8R11_UNREGISTERED_POLICY_OR_WORLD")
    events = trace["events"]
    if type(events) is not list or not 1 <= len(events) <= HORIZON:
        raise ValueError("F8R11_INCOMPLETE_OR_UNBOUNDED_EFFECT_TRACE")
    previous_after = None
    costs = []
    query_total = 0
    for step, event in enumerate(events, start=1):
        if type(event) is not dict or frozenset(event) != EVENT_KEYS:
            raise ValueError("F8R11_UNKNOWN_OR_MISSING_EFFECT_FIELDS")
        if type(event["event_id"]) is not int or event["event_id"] != step:
            raise ValueError("F8R11_EVENT_GAP_REPLAY_OR_WRONG_ORDER")
        before, after = (_state(event["state_before"]),
                         _state(event["state_after"]))
        if (previous_after is not None and
                any(abs(x-y) > CONTINUITY_ABS_TOL
                    for x, y in zip(previous_after, before))):
            raise ValueError("F8R11_BROKEN_PHYSICAL_TRAJECTORY")
        previous_after = after
        action = event["selected_action"]
        intent = event["intent_selected_action"]
        if (type(intent) is not int or intent not in (0, 1)
                or type(action) is not int or action not in (0, 1)
                or intent != action):
            raise ValueError("F8R11_INTENT_SELECTED_ACTION_CREDIT_MISMATCH")
        if (type(event["terminated"]) is not bool
                or type(event["truncated"]) is not bool):
            raise ValueError("F8R11_INVALID_TERMINAL_FLAGS")
        queries = event["hypothetical_action_queries"]
        expected = [] if seed == "fixed" else [0, 1]
        if type(queries) is not list or queries != expected:
            raise ValueError("F8R11_WRONG_OR_DUPLICATED_HYPOTHETICAL_QUERY")
        query_total += len(queries)
        _finite_real(event["reward"])
        if step < len(events) and (
                event["terminated"] or event["truncated"]):
            raise ValueError("F8R11_RECORDED_EFFECT_AFTER_TERMINATION")
        if step == len(events):
            if step < HORIZON and (
                    event["terminated"] is not True
                    or event["truncated"] is not False):
                raise ValueError("F8R11_EARLY_STOP_FLAGS_OR_MISSING_EFFECT")
            if step == HORIZON and event["truncated"] is not True:
                raise ValueError("F8R11_HORIZON_TRUNCATION_MISSING")
        step_cost = (abs(after[2]) / ANGLE_SCALE
                     + ANGULAR_VEL_COEFF * abs(after[3]) / ANGULAR_VEL_SCALE)
        if not math.isfinite(step_cost):
            raise ValueError("F8R11_UNBOUNDED_PHYSICAL_COST")
        costs.append(step_cost)
    n = len(events)
    mean_cost = math.fsum(
        [*costs, (HORIZON - n) * MISSING_STEP_PENALTY]) / HORIZON
    if not math.isfinite(mean_cost):
        raise ValueError("F8R11_UNBOUNDED_PHYSICAL_COST")
    return {
        "classification": CLASSIFICATION,
        "session_id": trace["session_id"],
        "policy_arm": arm,
        "model_seed": seed,
        "world_index": world,
        "world_seed": trace["world_seed"],
        "mean_stability_cost": mean_cost,
        "survival_steps": n,
        "selected_actions": n,
        "hypothetical_queries": query_total,
    }


def compile_fabricated_panel(traces: list[dict]) -> tuple[list[dict], dict]:
    """Full 352-trace evidence accounting, reusing *original R10* admission."""
    locked()
    if type(traces) is not list or len(traces) != r10.ROWS_EXPECTED:
        raise ValueError("F8R11_INCOMPLETE_FABRICATED_PAIRED_PANEL")
    panel = [score_fabricated_trace(trace) for trace in traces]
    counts = r10.audit_fabricated_panel(panel)
    return panel, {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "input_episodes": len(traces),
        "independent_paired_worlds": counts["independent_world_clusters"],
        "fictional_selected_actions": counts["fictional_selected_effects"],
        "fictional_model_queries": counts["fictional_hypothetical_queries"],
        "physically_executed_external_actions": 0,
        "owner_source_first_use": False,
        "scientific_pass": False,
    }


def evaluate_fabricated_trajectories(traces: list[dict]) -> dict:
    panel, counts = compile_fabricated_panel(traces)
    result = r10.evaluate_fabricated_panel(panel)
    return {**result, "trace_recomputation": counts,
            "scientific_pass": False,
            "owner_source_first_use": False}


def refuse_original_source_receipt(*_args, **_kwargs):
    raise PermissionError("F8R11_REAL_SOURCE_OR_OWNER_FIRST_USE_NOT_AUTHORIZED")
