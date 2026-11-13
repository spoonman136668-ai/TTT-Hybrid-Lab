"""F8-R5: owner-gated real causal-action experiment proposal (source-neutral).

Only verifies a prospectively committed scope and fabricated calibration
fixtures. Never imports the simulator, calls a runner or certifies owner source
first-use. A real owner must separately authorize and implement each phase.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

FREEZE = Path(__file__).with_name("p85mr_f8r5_owner_science_trial_proposal.json")
FREEZE_BLOB = "50cf2ff33f62d6ea6a25a7c789921838826d0094"
SCHEMA = "ttt-hybrid.f8r5-owner-science-trial-proposal.v1"
F7_SPENT_SEEDS = range(1821000000, 1821000128)
CALIBRATION_POLICIES = ("frozen_scripted_feedback", "fixed_seed_random")
CALIBRATION_COUNT = 48


def locked():
    raw = FREEZE.read_bytes()
    git_blob = hashlib.sha1(
        b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
    if git_blob != FREEZE_BLOB:
        raise ValueError("F8R5_PRECOMMITTED_FREEZE_DRIFT")
    proposal = json.loads(raw)
    a = proposal["authority"]
    c = proposal["difficulty_calibration"]
    if (proposal["schema"] != SCHEMA
        or proposal["stage"] != "UNAPPROVED_SOURCE_NEUTRAL_PROSPECTIVE_CONTRACT_ONLY"
        or proposal["parent"]["source_sha"] !=
           "d1b24351ed8f3c9c8fd233d75dc0ad120a3c91cb"
        or proposal["original_immutable_source"]["commit"] !=
           "c94e9a4cd3641588b6b8507e71ccee5c27032bdf"
        or proposal["original_immutable_source"]["cartpole_wind_blob"] !=
           "d65a6a31941709dc9201259d80dd4ebbcf48b4fa"
        or c["seed_base"] != 1824000000
        or c["worlds_per_candidate_policy"] != 8
        or c["max_episode_actions"] != 96
        or c["max_physical_actions"] != 4608
        or c["total_max_sessions"] != CALIBRATION_COUNT
        or len(c["candidate_wind_std_init_theta_std"]) != 3
        or c["policies"] != list(CALIBRATION_POLICIES)
        or proposal["proposed_sealed_worlds"]["never_tune_on_test"]["seed_base"] != 1824300000
        or proposal["proposed_sealed_worlds"]["unfamiliar_family_reserved_not_defined"]["worlds"] != 0
        or any(value is not False for k, value in a.items()
               if k != "allowed_now")
        or proposal["historical_exclusion"]["reuse_any_F7_world"] is not False
        or proposal["historical_exclusion"]["prior_F7_native_science_pass"] is not False):
        raise ValueError("F8R5_SOURCE_OR_OWNER_SCOPE_DRIFT")
    return proposal


def session_seed(partition: str, index: int, *, profile_index=None, policy=None) -> int:
    """Compute prospective unique seeds; does not claim or execute a session."""
    f = locked()
    if type(index) is not int:
        raise ValueError("F8R5_UNREGISTERED_SESSION")
    if partition == "difficulty_calibration":
        c = f["difficulty_calibration"]
        if (type(profile_index) is not int
            or not 0 <= profile_index < len(c["candidate_wind_std_init_theta_std"])
            or policy not in CALIBRATION_POLICIES
            or not 0 <= index < c["worlds_per_candidate_policy"]):
            raise ValueError("F8R5_UNREGISTERED_CALIBRATION_SCOPE")
        return (c["seed_base"] + 1000*profile_index +
                100*CALIBRATION_POLICIES.index(policy) + index)
    worlds = f["proposed_sealed_worlds"].get(partition)
    if (partition not in ("fit", "fit_selector", "never_tune_on_test")
        or profile_index is not None or policy is not None
        or not isinstance(worlds, dict)
        or not 0 <= index < worlds["worlds"]):
        raise ValueError("F8R5_UNREGISTERED_OR_UNAPPROVED_COHORT")
    return worlds["seed_base"] + index



F8R14_FREEZE = Path(__file__).with_name("p85mr_f8r14_action_rng_freeze.json")
F8R14_FREEZE_BLOB = "0b1bf0b35b044bc69a1548269ad6f319c40d9546"
F8R14_DOMAIN = "ttt-hybrid:f8r14:fixed_seed_random:action:v1"


def fixed_seed_random_contract() -> dict:
    """Immutable source-neutral action-RNG contract; no owner execution grant."""
    raw = F8R14_FREEZE.read_bytes()
    actual = hashlib.sha1(
        b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
    ).hexdigest()
    if actual != F8R14_FREEZE_BLOB:
        raise ValueError("F8R14_FROZEN_ACTION_RNG_DRIFT")
    f = json.loads(raw)
    scope = f["scope"]
    authority = f["authority"]
    if (f["schema"] != "ttt-hybrid.f8r14-calibration-action-rng-isolation.v1"
        or f["stage"] != "FROZEN_BEFORE_R14_IMPLEMENTATION_AND_TEST"
        or f["classification"] != "SOURCE_NEUTRAL_DEV_ONLY"
        or f["parent"]["parent_head_sha"] != "208931e399020f7176e3c09d1552ccfe2a35b8ff"
        or f["parent"]["original_f8r5_science_proposal_blob"] != FREEZE_BLOB
        or scope["policy"] != "fixed_seed_random"
        or scope["action_domain"] != F8R14_DOMAIN
        or scope["calibration_profile_count"] != 3
        or scope["episodes_per_profile"] != 8
        or scope["max_step_count"] != 96
        or scope["environment_rng_independent"] is not True
        or scope["prefix_stateless"] is not True
        or scope["call_order_independent"] is not True
        or any(value is not False for value in authority.values())):
        raise ValueError("F8R14_ACTION_RNG_SCOPE_OR_OWNER_DRIFT")
    return f


def fixed_seed_random_action(profile_index: int, episode_index: int,
                             step_index: int) -> int:
    """Pure, stateless proposed calibration action; never reads simulator RNG.

    Index is ZERO-BASED and is the actual factual selected-action step.
    Calling this for a hypothetical action does not reveal its outcome.
    """
    fixed_seed_random_contract()
    if type(step_index) is not int or not 0 <= step_index < 96:
        raise ValueError("F8R14_UNREGISTERED_ACTION_STEP")
    seed = session_seed("difficulty_calibration", episode_index,
                        profile_index=profile_index, policy="fixed_seed_random")
    message = (F8R14_DOMAIN.encode("ascii") + b"\0"
               + seed.to_bytes(8, "big") + step_index.to_bytes(2, "big"))
    return hashlib.sha256(message).digest()[0] & 1



F8R15_FREEZE = Path(__file__).with_name("p85mr_f8r15_scripted_feedback_freeze.json")
F8R15_FREEZE_BLOB = "532d76b4402eafc67c099feca26aa5decc0a6072"


def frozen_scripted_feedback_contract() -> dict:
    """Prospective source-neutral scripted policy; owner source effects denied."""
    raw = F8R15_FREEZE.read_bytes()
    blob = hashlib.sha1(
        b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
    ).hexdigest()
    if blob != F8R15_FREEZE_BLOB:
        raise ValueError("F8R15_FROZEN_SCRIPTED_POLICY_DRIFT")
    f = json.loads(raw)
    p = f["frozen_policy"]
    if (f["schema"] != "ttt-hybrid.f8r15-scripted-feedback-dev.v1"
            or f["stage"] != "FROZEN_BEFORE_F8R15_IMPLEMENTATION_AND_TEST"
            or f["classification"] != "SOURCE_NEUTRAL_DEV_ONLY"
            or f["parent"]["sha"] != "eb60c6803db725d4b941ca95937cee08db949bae"
            or f["parent"]["r5_proposal_blob"] != FREEZE_BLOB
            or f["parent"]["random_policy_freeze_blob"] != F8R14_FREEZE_BLOB
            or p["name"] != CALIBRATION_POLICIES[0]
            or p["deterministic_rule"] !=
                "return int(pole_angle + 0.5 * pole_ang_vel > 0.0)"
            or p["action_zero_on_tie"] is not True
            or p["step_index_zero_based_max"] != 95
            or p["direct_source_policy_adapter"] is not False
            or any(value is not False for value in f["authority"].values())):
        raise ValueError("F8R15_SCRIPTED_SCOPE_OR_AUTHORITY_DRIFT")
    return f


def frozen_scripted_feedback_action(profile_index: int, episode_index: int,
                                    step_index: int, state: list | tuple) -> int:
    """Select the F7 balancing *sign* from current public 4D state only.

    F7's XOR exploratory flips are intentionally excluded; the prospective
    F8 calibration scripted control is a deterministic feedback comparator.
    Accept Python four-vector of public observables, not source wind.
    """
    frozen_scripted_feedback_contract()
    session_seed("difficulty_calibration", episode_index,
                 profile_index=profile_index, policy="frozen_scripted_feedback")
    if type(step_index) is not int or not 0 <= step_index < 96:
        raise ValueError("F8R15_UNREGISTERED_SCRIPTED_STEP")
    if (type(state) not in (tuple, list) or len(state) != 4
            or any(type(v) not in (float, int) or not math.isfinite(v)
                   for v in state)):
        raise ValueError("F8R15_INVALID_PUBLIC_4D_STATE")
    return int(state[2] + 0.5 * state[3] > 0.0)


def proposed_seed_receipt():
    f = locked()
    calibrated = [
        session_seed("difficulty_calibration", i, profile_index=p, policy=policy)
        for p in range(3) for policy in CALIBRATION_POLICIES for i in range(8)]
    other = [
        session_seed(part, i)
        for part in ("fit", "fit_selector", "never_tune_on_test")
        for i in range(f["proposed_sealed_worlds"][part]["worlds"])]
    all_seeds = calibrated + other
    if (len(calibrated) != CALIBRATION_COUNT
        or len(all_seeds) != len(set(all_seeds))
        or any(seed in F7_SPENT_SEEDS for seed in all_seeds)
        or len(other) != 96):
        raise ValueError("F8R5_COHORT_COLLISION")
    return {
        "schema": SCHEMA, "calibration_sessions_proposed": len(calibrated),
        "fit_and_selection_and_test_worlds_proposed": len(other),
        "max_calibration_effects_PROPOSED_NOT_APPROVED": 4608,
        "unfamiliar_worlds_approved": 0,
        "new_effects_executed": 0,
        "science_firstuse_authorized": False,
        "scientific_pass": False,
    }


def diagnostic_on_fabricated_calibration(rows):
    """Validate admission logic only; never accepts an original-source receipt.

    Every row MUST be identified as FABRICATED_DEV_ONLY. This result cannot
    qualify an original task or grant source first-use/owner action authority.
    """
    f = locked()
    c = f["difficulty_calibration"]
    limits = c["minimum_admission"]
    if type(rows) not in (list, tuple) or len(rows) != CALIBRATION_COUNT:
        raise ValueError("F8R5_INCOMPLETE_FABRICATED_PANEL")
    groups, seen = {}, set()
    expected_keys = {
        "classification", "profile_index", "policy", "episode_index",
        "prospective_seed", "steps", "terminated", "truncated"}
    for row in rows:
        if type(row) is not dict or set(row) != expected_keys:
            raise ValueError("F8R5_INVALID_FABRICATED_ROW")
        p, policy, index = row["profile_index"], row["policy"], row["episode_index"]
        try:
            seed = session_seed("difficulty_calibration", index,
                                profile_index=p, policy=policy)
        except (TypeError, ValueError) as error:
            raise ValueError("F8R5_INVALID_FABRICATED_ROW") from error
        steps = row["steps"]
        if (row["classification"] != "FABRICATED_DEV_ONLY"
            or type(steps) is not int or not 1 <= steps <= 96
            or row["prospective_seed"] != seed
            or type(row["terminated"]) is not bool
            or type(row["truncated"]) is not bool
            # CartPoleWindSCM.step forcibly sets truncated=True at its
            # horizon, even when physics independently terminates there.
            # A simultaneous terminal+truncation at step 96 is legitimate.
            or (steps < 96 and (not row["terminated"] or row["truncated"]))
            or (steps == 96 and not row["truncated"])
            or seed in seen):
            raise ValueError("F8R5_INVALID_FABRICATED_ROW")
        seen.add(seed)
        groups.setdefault((p, policy), []).append(steps)
    summaries = []
    selected = None
    for p in range(len(c["candidate_wind_std_init_theta_std"])):
        s = groups.get((p, "frozen_scripted_feedback"), [])
        rnd = groups.get((p, "fixed_seed_random"), [])
        if len(s) != 8 or len(rnd) != 8:
            raise ValueError("F8R5_FABRICATED_PROFILE_MISSING")
        s_term = sum(v < 96 for v in s)
        rnd_term = sum(v < 96 for v in rnd)
        mean_gap = statistics.mean(s)-statistics.mean(rnd)
        s_stdev = statistics.stdev(s)
        admitted = (
            limits["scripted_early_terminated_min"] <= s_term <=
                limits["scripted_early_terminated_max"]
            and rnd_term >= limits["random_early_terminated_min"]
            and mean_gap >= limits["scripted_mean_survival_advantage_over_random_steps"]
            and s_stdev >= limits["scripted_survival_sample_std_min_steps"])
        summaries.append({
            "profile_index": p,
            "scripted_early_terminations": s_term,
            "random_early_terminations": rnd_term,
            "scripted_vs_random_survival_gap": mean_gap,
            "scripted_sample_stdev": s_stdev,
            "fabricated_admission_result": admitted})
        if admitted and selected is None:
            selected = p
    return {
        "schema": SCHEMA,
        "classification": "FABRICATED_DEV_ONLY_NOT_ORIGINAL_OWNER_EVIDENCE",
        "first_qualifying_profile_index": selected,
        "dev_stop_code": "NO_FABRICATED_PROFILE_PASSED" if selected is None else None,
        "profile_diagnostics": summaries,
        "original_task_qualified": False,
        "original_source_actions": 0,
        "firstuse_authorized": False,
        "scientific_pass": False,
    }


def require_real_source_firstuse(*_args, **_kwargs):
    raise PermissionError("F8R5_OWNER_SCOPED_FIRSTUSE_NOT_AUTHORIZED")
