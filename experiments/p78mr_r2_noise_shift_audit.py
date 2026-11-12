"""P78-MR R2: immutable-weight toy noise vs persistent reversal diagnostic.

No source generation, fit/cal/val selectors, environment reset/do, training,
network access, model checkpoint editing, or scientific first use.
The original P78 selected weights are read-only.  The outcome is a fixed,
hand-authored microfixture sensitivity probe, NOT independent generalization.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
# Original P77 used sibling absolute imports, so expose exactly this directory.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from p77_write_policy import WritePolicy  # noqa: E402
from p78mr_adaptive_controls import new_learned  # noqa: E402
from p78mr_gain_posthoc_audit import (  # noqa: E402
    RESULT, RESULT_BLOB, REFERENCE, REFERENCE_BLOB,
    SEEDS, read_locked, audit as coefficient_audit,
)

FREEZE = HERE / "p78mr_r2_noise_shift_freeze.json"
FREEZE_BLOB = "5dfb7058e7aefd0bdfdc41ac886505248efe8854"
FIXED_GAIN = 0.4
PREFIX = 32
FOLLOW = 12
TOTAL = PREFIX + 1 + FOLLOW
D = 16
CHECKPOINTS = (0, 1, 2, 4, 8, 12)
SCENARIOS = ("outlier", "reversal", "stable")


def _load_frozen() -> tuple[dict, list[dict]]:
    f = json.loads(read_locked(FREEZE, FREEZE_BLOB))
    if (f.get("schema") != "ttt-hybrid.p78mr-r2-fixed-weights-noise-vs-reversal.v1"
            or f["frozen_fixture"]["precise_episode_lengths"]["total"] != TOTAL
            or not f["source_class"].startswith("HAND_AUTHORED")
            or f["authority"]["training"] is not False
            or f["authority"]["new_factual_first_use"] is not False):
        raise ValueError("P78MR_R2_FREEZE_MISMATCH")
    read_locked(REFERENCE, REFERENCE_BLOB)
    saved = json.loads(read_locked(RESULT, RESULT_BLOB))
    if saved["experiment"]["selected_fixed"]["gain"] != FIXED_GAIN:
        raise ValueError("P78MR_R2_CAL_CONTROL_DRIFT")
    prior = coefficient_audit()
    if tuple(row["seed"] for row in prior["rows"]) != SEEDS:
        raise ValueError("P78MR_R2_P78_WEIGHTS_DRIFT")
    return f, prior["rows"]


def toy_facts(case: str) -> dict[str, torch.Tensor]:
    """Only manually specified e0 sequences. Never calls a source generator."""
    if case not in SCENARIOS:
        raise ValueError("P78MR_R2_UNREGISTERED_CASE")
    action = torch.zeros((1, TOTAL, D), dtype=torch.float32)
    action[:, :, 0] = 1.0
    effect = torch.zeros_like(action)
    effect[:, :PREFIX, 0] = 2.0
    effect[:, PREFIX, 0] = -2.0 if case != "stable" else 2.0
    effect[:, PREFIX + 1:, 0] = -2.0 if case == "reversal" else 2.0
    return {"action": action, "effect": effect}


def frozen_model(seed: int, theta: tuple[float, ...]) -> WritePolicy:
    if seed not in SEEDS or len(theta) != 3 or not all(math.isfinite(x) for x in theta):
        raise ValueError("P78MR_R2_UNAUTHORIZED_THETA")
    model = new_learned("non_directional3", seed)
    with torch.no_grad():
        model.theta.copy_(torch.tensor(theta, dtype=torch.float32))
    model.eval()
    return model


def simulate(model: WritePolicy, case: str) -> dict:
    """Each prediction precedes its observed effect write; no future outcome."""
    facts = toy_facts(case)
    state = torch.zeros((1, D, D), dtype=torch.float32)
    predictions, squared_errors, carries = [], [], []
    with torch.inference_mode():
        for t in range(TOTAL):
            actual_action = facts["action"][:, t]
            predicted = float(model.read(state, actual_action)[0, 0])
            observed = float(facts["effect"][0, t, 0])
            if not math.isfinite(predicted):
                raise ValueError("P78MR_R2_NONFINITE_PREDICTION")
            predictions.append(predicted)
            squared_errors.append((predicted - observed) ** 2)
            one_confirmed_event = {
                "action": facts["action"][:, t:t+1],
                "effect": facts["effect"][:, t:t+1],
            }
            state = model.encode(one_confirmed_event, carry=state)
            if not bool(torch.isfinite(state).all()):
                raise ValueError("P78MR_R2_NONFINITE_STATE")
            carries.append(state.clone())
        query = facts["action"][:, -1]
        terminal_prediction = float(model.read(state, query)[0, 0])
    target_after_disruption = -2.0 if case == "reversal" else 2.0
    after = {
        str(n): float(carries[PREFIX + n][0, 0, 0]) for n in CHECKPOINTS
    }
    return {
        "case": case,
        "read_before_write": True,
        "events": TOTAL,
        "prefix_prediction_before_disruption": predictions[PREFIX],
        "disruption_prewrite_mse": squared_errors[PREFIX],
        "after_disruption_and_followup_predictions": after,
        "first_followup_mse": squared_errors[PREFIX + 1],
        "first_four_followup_prewrite_mse": sum(
            squared_errors[PREFIX+1:PREFIX+5]
        ) / 4.0,
        "end12_mse": (terminal_prediction - target_after_disruption) ** 2,
        "post12_prediction": terminal_prediction,
        "trace_prewrite_predictions": predictions,
        "trace_prewrite_errors": squared_errors,
    }


def audit() -> dict:
    freeze, stored_rows = _load_frozen()
    fixed = WritePolicy("fixed_eta", eta=FIXED_GAIN)
    fixed_results = {case: simulate(fixed, case) for case in SCENARIOS}
    cohort = []
    for row in stored_rows:
        model = frozen_model(row["seed"], tuple(row["theta"]))
        learned = {case: simulate(model, case) for case in SCENARIOS}
        f_noise, m_noise = fixed_results["outlier"], learned["outlier"]
        f_shift, m_shift = fixed_results["reversal"], learned["reversal"]
        h_noise = m_noise["first_followup_mse"] <= f_noise["first_followup_mse"]
        h_shift = (
            m_shift["first_four_followup_prewrite_mse"] <=
            f_shift["first_four_followup_prewrite_mse"]
            and m_shift["end12_mse"] <= f_shift["end12_mse"]
        )
        if abs(learned["outlier"]["prefix_prediction_before_disruption"]
               - learned["reversal"]["prefix_prediction_before_disruption"]) > 0:
            raise AssertionError("P78MR_R2_FUTURE_EFFECT_LEAKAGE")
        cohort.append({
            "seed": row["seed"], "theta": row["theta"],
            "frozen_learned": learned,
            "h_noise": h_noise, "h_shift": h_shift,
            "joint_pass": h_noise and h_shift,
        })
    return {
        "schema": "ttt-hybrid.p78mr-r2-hand-authored-noise-vs-reversal.v1",
        "classification": "TOY_POSTHOC_FIXED_WEIGHTS_ONLY",
        "scientific_pass": False,
        "source_birth": False,
        "model_training": False,
        "holdout_scoring": False,
        "freeze_git_blob": FREEZE_BLOB,
        "original_result_blob": RESULT_BLOB,
        "original_policy_blob": REFERENCE_BLOB,
        "frozen_fixture": freeze["frozen_fixture"],
        "fixed_cal_selected_gain": FIXED_GAIN,
        "fixed_controls": fixed_results,
        "seeds": cohort,
        "joint_all_seed_pass": all(x["joint_pass"] for x in cohort),
        "decision": ("TOY_JOINT_PASS" if all(x["joint_pass"] for x in cohort)
                     else "TOY_JOINT_FAIL"),
        "note": ("All values use a manufactured e0 trace; frozen theta was "
                 "not fitted or selected against this trace. Do not infer "
                 "actual-world causal or OOD performance."),
    }


if __name__ == "__main__":
    print(json.dumps(audit(), sort_keys=True, indent=2))
