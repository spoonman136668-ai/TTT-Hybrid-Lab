"""F8-R12 DEV-only pre-action decision falsifier for the original R5 planner.

Checks that a declared model-controlled selected action is the argmin of
TWO declared hypothetical predictions on the SAME observed pre-action state.
This DOES NOT prove predictions were produced by real model weights, ahead
of actual source effects, or by the authorized owner. No first-use access.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from experiments import p85mr_f8r11_trace_scoring as r11
from experiments import p85mr_f8r10_paired_science as r10

HERE = Path(__file__).resolve().parent
FREEZE = HERE / "p85mr_f8r12_preaction_decision_freeze.json"
FREEZE_BLOB = "6458bfbbe5f79a37d226d36d2fff38652141080f"
R11_SRC = HERE / "p85mr_f8r11_trace_scoring.py"
R11_BLOB = "84aff1fec69bbd068b02c1e89ed3c891256163ce"
R5_FREEZE = HERE / "p85mr_f8r5_owner_science_trial_proposal.json"
R5_BLOB = "50cf2ff33f62d6ea6a25a7c789921838826d0094"
TRACE_EXTENSION = "pre_action_predictions"
PREDICTION_FIELDS = frozenset(
    ("hypothetical_action", "query_state_before", "predicted_state_after"))
PROTOCOL = "ttt-hybrid.f8r12-preaction-control-decision-dev.v1"
DEV = "FABRICATED_DEV_ONLY"


def git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def locked() -> dict:
    raw = FREEZE.read_bytes()
    if git_blob(raw) != FREEZE_BLOB:
        raise ValueError("F8R12_FROZEN_PROTOCOL_DRIFT")
    f = json.loads(raw)
    r11.locked()
    if (git_blob(R11_SRC.read_bytes()) != R11_BLOB
            or git_blob(R5_FREEZE.read_bytes()) != R5_BLOB
            or f["schema"] != PROTOCOL
            or f["stage"] !=
                "FROZEN_BEFORE_IMPLEMENTATION_OR_F8_NEW_EFFECTS"
            or f["classification"] != DEV
            or f["parent"]["r11_trace_evaluator_blob"] != R11_BLOB
            or f["parent"]["r5_owner_proposal_blob"] != R5_BLOB
            or f["decision"]["trace_event_extension"] !=
                "exactly one extra event field 'pre_action_predictions', list of two exact dictionaries for trained policies, or empty list for fixed scripted policy"
            or f["decision"]["action_order"] != [0, 1]
            or f["authority"]["owner_firstuse"] is not False
            or f["authority"]["science_pass"] is not False
            or f["authority"]["new_source_effects"] is not False):
        raise ValueError("F8R12_SCOPE_OR_FIRSTUSE_AUTHORITY_DRIFT")
    return f


def _state(v: object) -> tuple[float, float, float, float]:
    if type(v) is not list or len(v) != 4:
        raise ValueError("F8R12_INVALID_PREDICTION_STATE")
    if any(type(n) not in (int, float) or not math.isfinite(n) for n in v):
        raise ValueError("F8R12_INVALID_PREDICTION_STATE")
    return tuple(float(n) for n in v)


def _predicted_step_cost(v: object) -> float:
    s = _state(v)
    cost = abs(s[2]) / r11.ANGLE_SCALE + (
        r11.ANGULAR_VEL_COEFF * abs(s[3]) / r11.ANGULAR_VEL_SCALE)
    if not math.isfinite(cost):
        raise ValueError("F8R12_UNBOUNDED_HYPOTHETICAL_COST")
    return cost


def check_and_strip_fabricated_trace(trace: dict) -> tuple[dict, dict]:
    """Validate synthetic model argmin; output original R11 trace schema."""
    locked()
    if type(trace) is not dict or set(trace) != set(r11.EPISODE_KEYS):
        raise ValueError("F8R12_UNTRUSTED_TRACE_SHAPE")
    if trace["classification"] != DEV:
        raise ValueError("F8R12_REAL_OWNER_RECEIPT_REFUSED")
    events = trace["events"]
    if type(events) is not list or not 1 <= len(events) <= r11.HORIZON:
        raise ValueError("F8R12_INVALID_EPISODE_LENGTH")
    scripted = (trace["policy_arm"] == "frozen_handwritten_feedback_policy"
                and trace["model_seed"] == "fixed")
    checked = []
    decision_count = 0
    for index, event in enumerate(events, 1):
        if (type(event) is not dict or
                set(event) != set(r11.EVENT_KEYS) | {TRACE_EXTENSION}):
            raise ValueError("F8R12_UNKNOWN_DECISION_FIELDS")
        predictions = event[TRACE_EXTENSION]
        if type(predictions) is not list:
            raise ValueError("F8R12_INVALID_PREACTION_PREDICTIONS")
        expected_pre = _state(event["state_before"])
        if scripted:
            if predictions:
                raise ValueError("F8R12_SCRIPTED_POLICY_CANNOT_CLAIM_MODEL")
        else:
            if len(predictions) != 2:
                raise ValueError("F8R12_MISSING_OR_EXTRA_MODEL_QUERY")
            costs = []
            for action, prediction in enumerate(predictions):
                if (type(prediction) is not dict or
                        frozenset(prediction) != PREDICTION_FIELDS
                        or type(prediction["hypothetical_action"]) is not int
                        or prediction["hypothetical_action"] != action
                        or _state(prediction["query_state_before"]) !=
                            expected_pre):
                    raise ValueError("F8R12_WRONG_HYPOTHETICAL_ACTION_OR_INPUT")
                costs.append(_predicted_step_cost(
                    prediction["predicted_state_after"]))
            # R5 fixes action0 on a tie. This is a check against self-reported
            # predictions, not independent authentication of model inference.
            decision = 0 if costs[0] <= costs[1] else 1
            if (type(event["intent_selected_action"]) is not int
                    or type(event["selected_action"]) is not int
                    or event["intent_selected_action"] != decision
                    or event["selected_action"] != decision):
                raise ValueError("F8R12_SELECTED_ACTION_NOT_MODEL_ARGMIN")
            decision_count += 1
        # Never pass imaginary predicted counterfactuals to the original
        # R11 physical-effects and termination evaluator.
        checked.append({key: value for key, value in event.items()
                        if key != TRACE_EXTENSION})
    stripped = dict(trace, events=checked)
    # R11 authenticates *shape and internal consistency of fabricated values*
    # only, not owner provenance.
    summary = r11.score_fabricated_trace(stripped)
    return stripped, {
        "classification": DEV,
        "session_id": summary["session_id"],
        "decision_steps_checked": decision_count,
        "scripted_control": scripted,
        "owner_verified_inference": False,
        "real_source_effects": 0,
        "scientific_pass": False,
    }


def evaluate_fabricated_decision_panel(traces: list[dict]) -> dict:
    locked()
    if type(traces) is not list or len(traces) != r10.ROWS_EXPECTED:
        raise ValueError("F8R12_INCOMPLETE_PAIRED_DECISION_PANEL")
    verified = [check_and_strip_fabricated_trace(t) for t in traces]
    # Original R11 re-derives factual post-state score and early-stop penalty.
    result = r11.evaluate_fabricated_trajectories([v[0] for v in verified])
    return {
        **result,
        "preaction_decision_verification": {
            "classification": DEV,
            "decision_steps_checked": sum(v[1]["decision_steps_checked"]
                                          for v in verified),
            "scripted_sessions": sum(v[1]["scripted_control"]
                                     for v in verified),
            "independent_world_clusters": r10.WORLD_COUNT,
            "signed_original_owner_inference": False,
            "real_source_firstuse": False,
        },
        "scientific_pass": False,
    }


def refuse_original_model_or_source(*_args, **_kwargs):
    raise PermissionError("F8R12_ORIGINAL_FIRSTUSE_AND_WEIGHT_INFERENCE_NOT_AUTHORIZED")
