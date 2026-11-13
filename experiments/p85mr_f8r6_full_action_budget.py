"""F8-R6 source-neutral global physical effect BUDGET, NOT first-use authority.

Compiles an exact proposed 480-session/46,080-effect worst-case ledger. All
entries are future session identities, not claims or permission to instantiate
the real simulator. Tests operate on fabricated counters only. No source import,
task, workflow, runner, scheduling, net, or owner mutations are possible here.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/p85mr_f8r6_full_action_budget_freeze.json"
SOURCE_PROPOSAL = ROOT / "experiments/p85mr_f8r5_owner_science_trial_proposal.json"
FREEZE_BLOB = "59475472a4c316fa04a3510e4138a872abc88a91"
SCHEMA = "ttt-hybrid.f8r6-full-original-physical-source-budget.v1"
HORIZON = 96


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") +
                        b"\0" + data).hexdigest()


def locked():
    raw = FREEZE.read_bytes()
    if git_blob(raw) != FREEZE_BLOB:
        raise ValueError("F8R6_FROZEN_BUDGET_BLOB_DRIFT")
    f = json.loads(raw)
    proposal = json.loads(SOURCE_PROPOSAL.read_text(encoding="utf-8"))
    p = f["predecessor"]
    if (f["schema"] != SCHEMA
        or f["status"] != "FROZEN_PROPOSAL_NOT_APPROVED_FOR_ANY_ORIGINAL_SOURCE_FIRSTUSE"
        or p["sha"] != "90570bd0f7fa13af3898001ad612ed13f255b64e"
        or p["original_F7_effects_consumed"] != 12288
        or git_blob(SOURCE_PROPOSAL.read_bytes()) !=
            p["original_F8_f8r5_owner_proposal_blob"]
        or proposal["authority"]["source_FirstUse"] is not False
        or f["original_source"]["sha"] != "c94e9a4cd3641588b6b8507e71ccee5c27032bdf"
        or f["original_source"]["cartpole_blob"] !=
            "d65a6a31941709dc9201259d80dd4ebbcf48b4fa"
        or f["original_source"]["max_steps_per_session"] != HORIZON):
        raise ValueError("F8R6_SOURCE_OR_CUSTODY_DRIFT")
    if any(item["approved"] is not False
           for item in f["phase_first_use"].values()):
        raise PermissionError("F8R6_NEW_ORIGINAL_SOURCE_FIRSTUSE_NOT_APPROVED")
    if (f["denials"]["science_pass"] is not False or any(
            v is not True for k, v in f["denials"].items()
            if k != "science_pass")):
        raise PermissionError("F8R6_OWNER_SOURCE_DENIAL_DRIFT")
    t = f["test"]
    if (t["worlds"] != 16 or len(t["trained_arms"]) != 7
        or len(t["training_seeds"]) != 3
        or len(set(t["trained_arms"])) != 7
        or len(set(t["training_seeds"])) != 3
        or t["total_policy_instances"] != 22
        or t["trained_arm_instances"] != 21
        or t["scripted_control_instances"] != 1
        or t["statistical_unit"].startswith("16 paired physical world-seed clusters") is False
        or f["global"]["max_actual_selected_action_effects"] != 46080
        or f["global"]["max_original_source_resets"] != 480
        or f["global"]["max_original_source_constructor_calls"] != 480):
        raise ValueError("F8R6_BUDGET_OR_PAIRED_UNIT_DRIFT")
    return f


def planned_sessions():
    """Return *proposed* identities only; each would need an owner claim.

    A repeated test world seed is deliberate paired evaluation, not a replay
    of an effect: each policy's fresh session must make real decisions and
    record its own action and outcome if separately authorized later.
    """
    f = locked()
    source_sha = f["original_source"]["sha"]
    sessions = []
    def append(phase, arm, learner_seed, world, seed):
        identity = f"{phase}:{arm}:{learner_seed}:{world}:{source_sha}"
        sessions.append({"identity": identity,
                         "phase": phase, "arm": arm,
                         "learner_seed": learner_seed,
                         "world_index": world,
                         "source_seed": seed,
                         "maximum_factual_actions": HORIZON})
    c = f["calibration"]
    for profile in range(c["profiles"]):
        for policy in range(c["policies"]):
            for world in range(c["worlds_each"]):
                seed = c["seed_base"] + 1000*profile + 100*policy + world
                append("calibration", f"profile{profile}-policy{policy}",
                       "fixed", world, seed)
    for phase, key in (("fit", "common_fit"), ("selector", "fit_selector")):
        spec = f[key]
        for world in range(spec["worlds"]):
            append(phase, "frozen-exploratory", "fixed", world,
                   spec["seed_base"]+world)
    t = f["test"]
    for arm in t["trained_arms"]:
        for learner_seed in t["training_seeds"]:
            for world in range(t["worlds"]):
                append("test", arm, learner_seed, world,
                       t["seed_base"]+world)
    for world in range(t["worlds"]):
        append("test", t["scripted_control"], "fixed", world,
               t["seed_base"]+world)
    expected = {"calibration": 48, "fit": 64, "selector": 16, "test": 352}
    actual = Counter(s["phase"] for s in sessions)
    identities = [s["identity"] for s in sessions]
    phase_effects = {
        "calibration": c["max_selected_action_effects"],
        "fit": f["common_fit"]["max_selected_action_effects"],
        "selector": f["fit_selector"]["max_selected_action_effects"],
        "test": t["max_selected_action_effects"],
    }
    if (actual != expected or len(identities) != len(set(identities))
        or sum(s["maximum_factual_actions"] for s in sessions) != 46080
        or any(actual[k]*HORIZON != phase_effects[k] for k in expected)
        or len({s["source_seed"] for s in sessions}) !=
            f["global"]["max_reported_world_seed_groups"]
        or any(1821000000 <= s["source_seed"] < 1821000128
               for s in sessions)
        or len({s["source_seed"] for s in sessions
                if s["phase"] == "test"}) != 16
        or any(sum(s["source_seed"] == seed and s["phase"] == "test"
                   for s in sessions) != 22
               for seed in range(t["seed_base"], t["seed_base"]+16))):
        raise ValueError("F8R6_SESSION_OR_SOURCE_COLLISION")
    return sessions


def proposed_receipt():
    f = locked()
    sessions = planned_sessions()
    counts = Counter(s["phase"] for s in sessions)
    return {
        "schema": SCHEMA,
        "source_kind": "UNAPPROVED_PROSPECTIVE_ONLY",
        "session_counts": dict(counts),
        "sessions_proposed": len(sessions),
        "physical_factual_actions_worst_case": sum(
            s["maximum_factual_actions"] for s in sessions),
        "original_factual_actions_executed_now": 0,
        "paired_evaluation_worlds": 16,
        "paired_evaluation_policy_instances": 22,
        "independent_sealed_worlds_NOT_352": True,
        "future_CAL_OOD_effects_authorized": False,
        "scientific_pass": False,
        "deployment": "NONE"
    }


def check_fabricated_counter_panel(rows):
    """Adversarial DEV check only; does NOT validate real source custody."""
    sessions = {s["identity"]: s for s in planned_sessions()}
    if not isinstance(rows, (tuple, list)):
        raise ValueError("F8R6_MALFORMED_TEST_COUNTER_PANEL")
    seen, total = set(), 0
    for row in rows:
        if (type(row) is not dict or
            set(row) != {"classification", "identity", "attempted_factual_actions"}
            or row["classification"] != "FABRICATED_DEV_COUNTER_ONLY"
            or type(row["identity"]) is not str
            or row["identity"] not in sessions
            or row["identity"] in seen
            or type(row["attempted_factual_actions"]) is not int
            or not 0 <= row["attempted_factual_actions"] <= HORIZON):
            raise ValueError("F8R6_DUPLICATE_OR_EXCESS_UNQUALIFIED_COUNTER")
        seen.add(row["identity"])
        total += row["attempted_factual_actions"]
    if total > 46080 or len(seen) > 480:
        raise ValueError("F8R6_GLOBAL_SOURCE_ACTION_CEILING")
    return {"classification": "FABRICATED_DEV_ONLY",
            "sessions_counted": len(seen),
            "factual_actions_counted": total,
            "genuine_owner_custody_verified": False,
            "source_first_use_authorized": False}


def refuse_original_source_first_use(*_args, **_kwargs):
    raise PermissionError("F8R6_ORIGINAL_SOURCE_OWNER_FIRSTUSE_NOT_AUTHORIZED")
