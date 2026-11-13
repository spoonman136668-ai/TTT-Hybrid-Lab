"""F8-R4: one-action synthetic causal identification under matched evidence budget.

Exact mathematical two-world counterexample from F8-R3, NOT empirical F8:
one observed informative action resolves the latent mechanism; one equally
budgeted behavior-policy action cannot. No Hybrid, training, simulator or
external source effects are used.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from experiments.p85mr_f8r3_counterpolicy_identifiability import (
    behavior, synthetic_effect,
)

FREEZE = Path(__file__).with_name("p85mr_f8r4_minimal_intervention_freeze.json")
FREEZE_BLOB = "be252064e283ed038e31336fb170fcb2c4e0a473"
SCHEMA = "ttt-hybrid.f8r4.minimal-intervention-identification-dev.v1"
BELIEF_SCHEMA = "ttt-hybrid.f8r4.cold-sign-bit.v1"
QUERY_STATE = 0.2
TRANSFER_STATE = -0.2
TRANSFER_GOAL = 0.2


def locked():
    raw = FREEZE.read_bytes()
    git_blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if git_blob != FREEZE_BLOB:
        raise ValueError("F8R4_PRECOMMITTED_FREEZE_DRIFT")
    f = json.loads(raw)
    p, b = f["conditions"], f["boundaries"]
    if (f["schema"] != SCHEMA or
            f["source_parent"] != "3cf6f1b18e727a340ce07e927231fc62ee6accbb"
            or f["classification"] !=
            "PRECOMMITTED_SYNTHETIC_CAUSAL_INFORMATION_VALUE_TEST_NOT_EXTERNAL_F8"
            or p["world_signs"] != [-1, 1]
            or p["query_budget_per_arm"] != 1
            or p["query_state"] != QUERY_STATE
            or p["query_action"] != 0 or p["passive_control_action"] != 1
            or p["transfer_state"] != TRANSFER_STATE
            or p["transfer_goal"] != TRANSFER_GOAL
            or b["no_new_external_actions"] is not True
            or b["no_original_F7_firstuse_or_any_replay"] is not True
            or b["no_sci_first_use_or_merge_approval"] is not True):
        raise ValueError("F8R4_FROZEN_PROTOCOL_OR_AUTHORITY_DRIFT")
    return f


def _real(x, name):
    if type(x) not in (int, float) or not math.isfinite(x):
        raise ValueError("F8R4_INVALID_" + name)
    return float(x)


class OneBitBelief:
    """World is initially UNKNOWN; identification uses factual effect only."""

    def __init__(self):
        self.sign = None
        self.observations = 0

    def observe(self, state: float, action: int, realized_effect: float):
        if self.observations != 0:
            raise ValueError("F8R4_ACTION_EVIDENCE_BUDGET_EXHAUSTED")
        state = _real(state, "STATE")
        value = _real(realized_effect, "EFFECT")
        if type(action) is not int or action not in (0, 1):
            raise ValueError("F8R4_INVALID_ACTION")
        possible = [sign for sign in (-1, 1)
                    if math.isclose(synthetic_effect(state, action, sign),
                                    value, rel_tol=0, abs_tol=1e-12)]
        if not possible:
            raise ValueError("F8R4_OUTCOME_INCOMPATIBLE_WITH_BOTH_WORLDS")
        self.observations = 1
        if len(possible) == 1:
            self.sign = possible[0]

    def choose(self, state: float, goal: float):
        state, goal = _real(state, "STATE"), _real(goal, "GOAL")
        signs = (-1, 1) if self.sign is None else (self.sign,)
        def loss(action):
            return sum((synthetic_effect(state, action, sign) - goal)**2
                       for sign in signs) / len(signs)
        return min((0, 1), key=lambda a: (loss(a), a))

    def cold_export(self):
        if self.sign not in (-1, 1) or self.observations != 1:
            raise ValueError("F8R4_NO_IDENTIFIED_WORLD_TO_PERSIST")
        return {"schema": BELIEF_SCHEMA,
                "world_bit": 1 if self.sign == 1 else 0}

    @classmethod
    def cold_restore(cls, receipt):
        if (type(receipt) is not dict or set(receipt) != {"schema", "world_bit"}
                or receipt.get("schema") != BELIEF_SCHEMA
                or type(receipt.get("world_bit")) is not int
                or receipt["world_bit"] not in (0, 1)):
            raise ValueError("F8R4_INVALID_COLD_BIT_RECEIPT")
        b = cls()
        b.sign = 1 if receipt["world_bit"] == 1 else -1
        b.observations = 1
        return b


def evaluate():
    frozen = locked()
    rows = []
    for hidden_sign in (-1, 1):
        active, passive = OneBitBelief(), OneBitBelief()
        # The agent receives an actual outcome of its CHOSEN synthetic action,
        # not the hidden_sign; equal evidence budgets (1 action per arm).
        active_action, passive_action = 0, 1
        active.observe(QUERY_STATE, active_action,
                       synthetic_effect(QUERY_STATE, active_action, hidden_sign))
        passive.observe(QUERY_STATE, passive_action,
                        synthetic_effect(QUERY_STATE, passive_action, hidden_sign))
        # One-bit cold export/import BEFORE any transfer action is selected.
        after_cold = OneBitBelief.cold_restore(active.cold_export())
        active_choice = after_cold.choose(TRANSFER_STATE, TRANSFER_GOAL)
        passive_choice = passive.choose(TRANSFER_STATE, TRANSFER_GOAL)
        # Oracle used ONLY by evaluator to score a selected action, never sent
        # to the learner at transfer or counted as training evidence.
        oracle_choice = min((0, 1), key=lambda a: (
            (synthetic_effect(TRANSFER_STATE, a, hidden_sign)-TRANSFER_GOAL)**2,
            a))
        rows.append({
            "hidden_world_for_evaluator_only": hidden_sign,
            "actual_query_actions_per_arm": 1,
            "active_identified_mechanism_from_outcome": active.sign == hidden_sign,
            "passive_mechanism_unidentified": passive.sign is None,
            "active_cold_recovered_action": active_choice,
            "passive_action": passive_choice,
            "offline_oracle_best_action": oracle_choice,
            "active_matches_oracle": active_choice == oracle_choice,
            "passive_matches_oracle": passive_choice == oracle_choice,
        })
    active_accuracy = sum(x["active_matches_oracle"] for x in rows)
    passive_accuracy = sum(x["passive_matches_oracle"] for x in rows)
    if (active_accuracy != 2 or passive_accuracy != 1
            or not all(x["active_identified_mechanism_from_outcome"]
                       and x["passive_mechanism_unidentified"] for x in rows)):
        raise AssertionError("F8R4_FROZEN_NEGATIVE_CONTROL_NO_GO")
    return {
        "schema": SCHEMA,
        "classification": "EXACT_SYNTHETIC_EXPLANATION_NOT_EMPIRICAL",
        "frozen_blob": FREEZE_BLOB,
        "synthetic_worlds": 2,
        "actual_synthetic_evidence_actions_per_arm_per_world": 1,
        "original_F7_or_external_source_actions": 0,
        "active_transfer_action_accuracy": "2/2",
        "passive_transfer_action_accuracy": "1/2",
        "one_bit_cold_restore_verified": True,
        "observationally_indistinguishable_until_query": True,
        "evaluation_rows": rows,
        "trained_Hybrid_improved": False,
        "unknown_family_transfer_established": False,
        "scientific_pass": False,
        "deployment": "NONE",
    }


def refuse_external_first_use(*_args, **_kwargs):
    raise PermissionError("F8R4_EXTERNAL_SCIENTIFIC_FIRST_USE_NOT_AUTHORIZED")
