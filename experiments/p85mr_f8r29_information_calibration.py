"""F8-R29: source-neutral calibration of actual fifth-action information value.

Only author-generated deterministic mathematical effects. No native model is
trained here: the positive analytic diagnostic is a TASK calibration and cannot
be promoted to a Hybrid learning, science, or deployment claim.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/p85mr_f8r29_information_calibration_freeze.json"
FREEZE_BLOB = "4666d7afd6b418d8637faaa880644d9465ac2f9e"
SCHEMA = "ttt-hybrid.f8r29-active-information-calibration.v1"
PANELS = ("dev", "ood")
POLICIES = ("active_information", "fixed_probe0", "fixed_probe1", "seeded_random")
EARLY = ((-.2, 0), (.2, 0), (-.2, 1), (.2, 1))
Q_ABS = {"dev": 1.0, "ood": .9}
RETURN_STATES = {"dev": .7, "ood": .65}
B_STEPS = {"dev": 8, "ood": 16}
GROUPS = 12
SEED = 20459999
DRAWS = 10000
PROBE_STATE = .4
GOAL = 1.0


def git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def locked() -> dict:
    raw = FREEZE.read_bytes()
    if git_blob(raw) != FREEZE_BLOB:
        raise ValueError("F8R29_FROZEN_BLOB_DRIFT")
    f = json.loads(raw)
    d, s, a = f["design"], f["score"], f["authority"]
    if (f["schema"] != SCHEMA
        or f["stage"] != "FROZEN_BEFORE_IMPLEMENTATION_OR_NUMERIC_RESULT"
        or f["classification"] != "SOURCE_NEUTRAL_ANALYTIC_CAUSAL_INFORMATION_DEV_CALIBRATION_ONLY"
        or f["parent"]["head"] != "98fead8094f1bdacae418ccc4d54800c561bf827"
        or f["parent"]["r28_source_blob"] != "07ca35d7eebff3f5a3df94f8b00e72ead60467f7"
        or f["parent"]["r28_freeze_blob"] != "0ab949d433a4405435aa72612a17f28859ce2315"
        or d["early_factual_probes"] != [list(z) for z in EARLY]
        or d["fifth_probe_state"] != PROBE_STATE
        or d["q_abs_by_panel"] != Q_ABS
        or d["return_state_by_panel"] != RETURN_STATES
        or d["B_steps_by_panel"] != B_STEPS
        or d["groups_per_panel"] != GROUPS or d["worlds_per_panel"] != 48
        or s["draws"] != DRAWS or s["seed"] != SEED
        or f["policies"]["variants"] != list(POLICIES)
        or s["independent_scientific_pass"] is not False
        or any(v is not False for v in a.values())):
        raise ValueError("F8R29_PROTOCOL_OR_AUTHORITY_DRIFT")
    return f


def after(state: float, action: int, context: int, q: float) -> float:
    if action not in (0, 1) or context not in (0, 1):
        raise ValueError("INVALID_ACTION_CONTEXT")
    if not all(math.isfinite(x) for x in (state, q)):
        raise ValueError("INVALID_PHYSICAL_INPUT")
    # In exact mathematics s=+/-0.2 has zero curvature. Suppress IEEE-754
    # cancellation residue to prevent hidden q leaking through early facts.
    curvature = state * state - .04
    if abs(curvature) < 1e-14:
        curvature = 0.0
    return .4 * state - .03 + ((5 / 3) * q * curvature
                              if action == context else 0.0)


def _b_history(panel: str, group: int) -> tuple:
    # Independent nuisance stream; no q, context, fifth action or return label.
    n = B_STEPS[panel]
    return tuple((t % 2, round((group - 5) * .011 + (t % 5) * .019, 8))
                 for t in range(n))


def world(panel: str, group: int, context: int, sign: int) -> dict:
    if panel not in PANELS or type(group) is not int or not 0 <= group < GROUPS \
            or context not in (0, 1) or sign not in (-1, 1):
        raise ValueError("F8R29_UNREGISTERED_WORLD")
    q = sign * Q_ABS[panel]
    return {
        "panel": panel, "group": group, "public_context": context,
        "hidden_q_evaluator_only": q, "b_history": _b_history(panel, group),
        "early_actual": tuple((s, a, after(s, a, context, q)) for s, a in EARLY),
        "return_state": RETURN_STATES[panel], "goal": GOAL,
    }


def worlds(panel: str) -> list[dict]:
    if panel not in PANELS:
        raise ValueError("F8R29_UNREGISTERED_PANEL")
    return [world(panel, g, c, z) for g in range(GROUPS)
            for c in (0, 1) for z in (-1, 1)]


def probe_action(policy: str, panel: str, group: int, public_context: int) -> int:
    if policy not in POLICIES or panel not in PANELS \
            or type(group) is not int or not 0 <= group < GROUPS \
            or public_context not in (0, 1):
        raise ValueError("F8R29_INVALID_PUBLIC_PROBE_INPUT")
    if policy == "active_information":
        return public_context
    if policy == "fixed_probe0":
        return 0
    if policy == "fixed_probe1":
        return 1
    raw = f"f8r29/{panel}/{group}/{public_context}".encode("ascii")
    return hashlib.sha256(raw).digest()[0] & 1


def return_choice(public_context: int, panel: str, fifth_action: int,
                  fifth_observed_effect: float | None) -> int:
    # This policy receives only public context, selected factual fifth action
    # and its *actual observed* effect. Hidden evaluator q is never passed.
    if public_context not in (0, 1) or fifth_action not in (0, 1) or panel not in PANELS:
        raise ValueError("F8R29_INVALID_MODEL_INPUT")
    if fifth_observed_effect is not None and not math.isfinite(fifth_observed_effect):
        raise ValueError("F8R29_INVALID_FACTUAL_EFFECT")
    state = RETURN_STATES[panel]
    safe = .4 * state - .03
    safe_loss = (safe - GOAL) ** 2
    if fifth_action == public_context and fifth_observed_effect is not None:
        denom = (5 / 3) * (PROBE_STATE ** 2 - .04)
        inferred = (fifth_observed_effect - (.4 * PROBE_STATE - .03)) / denom
        if not math.isclose(abs(inferred), Q_ABS[panel], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("F8R29_INCONSISTENT_FACTUAL_OBSERVATION")
        candidate = (inferred,)
    else:
        candidate = (-Q_ABS[panel], Q_ABS[panel])
    risk_loss = sum((safe + (5 / 3) * q * (state * state - .04) - GOAL) ** 2
                    for q in candidate) / len(candidate)
    if risk_loss < safe_loss:
        return public_context
    # Safe action is the other one; tie is risk-denying, fixed before outcomes.
    return 1 - public_context


def actual_regret(w: dict, action: int) -> float:
    if action not in (0, 1):
        raise ValueError("F8R29_INVALID_RETURN_ACTION")
    state, c, q = w["return_state"], w["public_context"], w["hidden_q_evaluator_only"]
    effects = [after(state, a, c, q) for a in (0, 1)]
    loss = [(x - w["goal"]) ** 2 for x in effects]
    return max(0.0, loss[action] - min(loss))


def physical_plan(w: dict, policy: str, erase: bool = False) -> dict:
    # Never calculate an unselected fifth effect. The oracle is scorer-only.
    c = w["public_context"]
    selected_probe = probe_action(policy, w["panel"], w["group"], c)
    selected_effect = after(PROBE_STATE, selected_probe, c,
                            w["hidden_q_evaluator_only"])
    selected_return = return_choice(c, w["panel"], selected_probe,
                                    None if erase else selected_effect)
    return {
        "group": w["group"], "public_context": c,
        "fifth_action": selected_probe, "selected_fifth_effect": selected_effect,
        "selected_return_action": selected_return,
        "factual_early_count": len(w["early_actual"]),
        "factual_fifth_count": 1, "B_count": len(w["b_history"]),
        "return_regret": actual_regret(w, selected_return),
    }


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _lower_active_gain(active: list[float], fixed0: list[float],
                       fixed1: list[float]) -> float:
    rng = random.Random(SEED)
    gains = []
    for _ in range(DRAWS):
        ix = [rng.randrange(GROUPS) for __ in range(GROUPS)]
        gains.append(min(_mean([fixed0[i] for i in ix]),
                         _mean([fixed1[i] for i in ix]))
                     - _mean([active[i] for i in ix]))
    gains.sort()
    return gains[int(.025 * DRAWS)]


def panel_score(panel: str) -> dict:
    eps = worlds(panel)
    panels = {p: [physical_plan(w, p) for w in eps] for p in POLICIES}
    groups = {p: [_mean([r["return_regret"] for r in rows if r["group"] == g])
                  for g in range(GROUPS)] for p, rows in panels.items()}
    scores = {p: _mean(groups[p]) for p in POLICIES}
    erasure = _mean([physical_plan(w, "active_information", True)["return_regret"]
                     for w in eps])
    lower = _lower_active_gain(groups["active_information"],
                               groups["fixed_probe0"], groups["fixed_probe1"])
    fixed = min(scores["fixed_probe0"], scores["fixed_probe1"])
    s = locked()["score"]
    passed = (scores["active_information"] <= s["active_mean_regret_max"]
              and fixed - scores["active_information"] >=
              s["active_vs_best_fixed_absolute_gain_min"]
              and lower > s["active_vs_best_fixed_bootstrap_lower_2p5_gt"]
              and erasure - scores["active_information"] >=
              s["fifth_effect_erasure_regret_increase_min"])
    return {
        "panel": panel, "world_count": len(eps), "groups": GROUPS,
        "policy_mean_regret": scores, "erased_active_regret": erasure,
        "paired_bootstrap_lower_gain": lower, "best_fixed_gain": fixed - scores["active_information"],
        "task_calibration_pass": passed, "model_training_updates": 0,
        "factual_selected_fifth_effects_each_policy": len(eps),
        "factual_early_effects_each_policy": len(eps) * len(EARLY),
        "B_effects_each_policy": len(eps) * B_STEPS[panel],
    }


def run() -> dict:
    locked()
    panels = {panel: panel_score(panel) for panel in PANELS}
    return {
        "schema": SCHEMA, "classification": "SOURCE_NEUTRAL_ANALYTIC_TASK_CALIBRATION_ONLY",
        "freeze_blob": FREEZE_BLOB, "parent_head": "98fead8094f1bdacae418ccc4d54800c561bf827",
        "panels": panels, "analytic_task_calibration_pass":
            all(x["task_calibration_pass"] for x in panels.values()),
        "native_model_training_updates": 0,
        "independent_scientific_pass": False, "original_external_effects": 0,
        "spent_F7_replay": False, "deployment": "NONE",
    }


def retain_ci_receipt(data: dict) -> None:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    path = ROOT / "local_runs/p85mr_f8r29_information_calibration_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R29_REFUSE_RECEIPT_OVERWRITE")
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def refuse_original_external_firstuse(*_args, **_kwargs):
    raise PermissionError("F8R29_EXTERNAL_FIRSTUSE_NOT_AUTHORIZED")
