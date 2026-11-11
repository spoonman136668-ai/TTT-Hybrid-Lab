"""P105 pure, read-only feasibility fence for previously consumed F3 FIT logs.

NO simulator imports, source access, owner dispatch, action selection, model update,
GPU inference, network, archive download, or first-use functionality. The F3
empirical support summary was observed BEFORE this protocol and is diagnostic,
not a prospective scientific outcome.

This module only checks the shape/coverage of an in-memory factual capture
supplied by an authorized reader, and refuses claims about unobserved effects.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

FREEZE_COMMIT = "2b687182b90a844a5af210c39502a9a2f4bcc9ae"
FREEZE_BLOB = "cd1bb40fd74df9d693067798fa5360a0cb205cbb"
UPSTREAM_REPO = "CausalAILab/Causal-Gymnasium"
UPSTREAM_SHA = "c94e9a4cd3641588b6b8507e71ccee5c27032bdf"
UPSTREAM_PATH = "causal_gym/envs/cartpole_wind.py"
UPSTREAM_BLOB = "d65a6a31941709dc9201259d80dd4ebbcf48b4fa"
CAPTURE_SCHEMA = "ttt-hybrid.p80mr.cartpole-do-capture.v1"
CAPTURE_FIELDS = frozenset((
    "schema", "upstream_repo", "upstream_commit", "upstream_path",
    "upstream_blob", "source_id", "producer_id", "custody_ref", "worlds",
))
WORLD_FIELDS = frozenset(("world_id", "episode_seed", "transitions", "probe_event_ids"))
STEP_FIELDS = frozenset((
    "event_id", "state_before", "executed_action", "state_after",
    "reward", "terminated", "truncated", "mode", "receipt_ref",
))


def _git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") +
                        b"\0" + raw).hexdigest()


def locked_protocol() -> dict:
    """Read only the checked-in local protocol. Never inspect F3 FIT/CAL here."""
    path = Path(__file__).with_name("p105_authentic_action_support_freeze.json")
    raw = path.read_bytes()
    if _git_blob(raw) != FREEZE_BLOB:
        raise ValueError("P105_PROTOCOL_GIT_BLOB_DRIFT")
    x = json.loads(raw)
    if (x.get("schema") !=
        "ttt-hybrid.p105-f3-support-feasibility-and-f6-design-preflight.v1"
        or x.get("stage") != "FROZEN_NON_EFFECTFUL_FEASIBILITY_ONLY_NOT_F6_FIRST_USE"
        or x["exact_parent"]["sha"] != "4d5c49be278077fe3b0be80080d39128a153aa77"
        or x["exact_parent"]["f3_original_artifact_zip_sha256"] !=
            "a04697bdfba1eb93548244329f4f76fb807489aa7766f4a343de897a56caceb5"
        or x["exact_parent"]["f3_fit_capture_sha256"] !=
            "b495d6c41fd54c6955fd83ead7569282146af45bc68b3a3dba2a63808e1d6f7b"
        or x["authority"]["new_owner_effect_execution_authorized"] is not False
        or x["proposed_F6_future_protocol_only"]["not_yet_authorized_or_executed"] is not True):
        raise ValueError("P105_PROTOCOL_CONTRACT_DRIFT")
    return x


def _finite_state(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 4
        or any(type(x) not in (int,float) or not math.isfinite(x) for x in value)):
        raise ValueError("P105_INVALID_F3_STATE")
    return tuple(float(x) for x in value)


def _finite_reward(value):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError("P105_INVALID_F3_REWARD")
    return float(value)


def observational_support(capture: dict, *, expected_raw_sha256: str|None=None,
                          original_bytes: bytes|None=None) -> dict:
    """Descriptive necessary support only; NEVER a causal off-policy score.

    This accepts an in-memory, exact-schema F3 FIT capture, nothing else.
    No code path can execute or create real-world actions.
    """
    locked_protocol()
    if not isinstance(capture,dict) or set(capture) != CAPTURE_FIELDS:
        raise ValueError("P105_BAD_CAPTURE_FIELDS")
    if (capture["schema"] != CAPTURE_SCHEMA or
        (capture["upstream_repo"],capture["upstream_commit"],
         capture["upstream_path"],capture["upstream_blob"]) !=
        (UPSTREAM_REPO,UPSTREAM_SHA,UPSTREAM_PATH,UPSTREAM_BLOB)):
        raise ValueError("P105_WRONG_EXTERNAL_SOURCE")
    if (capture["source_id"] != "p85mr-f3-fit-43073165842" or
        capture["producer_id"] != "ttt-hybrid-f3-owner" or
        capture["custody_ref"] != "ghrun:43073165842"):
        raise ValueError("P105_ONLY_CONSUMED_FIT_PERMITTED")
    original_fit_sha = locked_protocol()["exact_parent"]["f3_fit_capture_sha256"]
    original_authenticated = False
    if expected_raw_sha256 is not None:
        if (expected_raw_sha256 != original_fit_sha or
            not isinstance(original_bytes,bytes) or
            hashlib.sha256(original_bytes).hexdigest()!=original_fit_sha or
            json.loads(original_bytes)!=capture):
            raise ValueError("P105_CAPTURE_CONTENT_SHA_DRIFT")
        original_authenticated = True
    elif original_bytes is not None:
        raise ValueError("P105_UNBOUND_RAW_CAPTURE")
    worlds=capture["worlds"]
    if not isinstance(worlds,list) or not 1<=len(worlds)<=256:
        raise ValueError("P105_INVALID_WORLD_COUNT")
    actions=Counter()
    exact_states=defaultdict(set)
    n=0
    terminals=0
    truncations=0
    total_receipts=set()
    for w in worlds:
        if not isinstance(w,dict) or set(w)!=WORLD_FIELDS:
            raise ValueError("P105_BAD_WORLD_FIELDS")
        if (not isinstance(w["world_id"],str) or not w["world_id"].startswith("fit-")
            or type(w["episode_seed"]) is not int
            or w["episode_seed"]<0):
            raise ValueError("P105_NOT_FIT_WORLD")
        transitions=w["transitions"]
        if not isinstance(transitions,list) or not 2<=len(transitions)<=4096:
            raise ValueError("P105_BAD_TRANSITIONS")
        if w["probe_event_ids"]!=list(range(2,len(transitions)+1)):
            raise ValueError("P105_PROBE_RECEIPT_DRIFT")
        for idx,e in enumerate(transitions,start=1):
            if not isinstance(e,dict) or set(e)!=STEP_FIELDS:
                raise ValueError("P105_BAD_EVENT_FIELDS")
            action=e["executed_action"]
            if (type(action) is not int or action not in (0,1)
                or type(e["event_id"]) is not int or e["event_id"]!=idx
                or e["mode"]!="do" or type(e["terminated"]) is not bool
                or type(e["truncated"]) is not bool
                or not isinstance(e["receipt_ref"],str) or not e["receipt_ref"]):
                raise ValueError("P105_NOT_CONFIRMED_BINARY_DO")
            before=_finite_state(e["state_before"])
            _finite_state(e["state_after"])
            _finite_reward(e["reward"])
            key=(w["world_id"],idx,e["receipt_ref"])
            if key in total_receipts:
                raise ValueError("P105_DUPLICATE_FACTUAL_RECEIPT")
            total_receipts.add(key)
            actions[action]+=1
            exact_states[before].add(action)
            terminals+=int(e["terminated"])
            truncations+=int(e["truncated"])
            n+=1
    paired=sum(1 for action_set in exact_states.values() if action_set=={0,1})
    return {
        "schema":"ttt-hybrid.p105-historical-fit-action-support-audit.v1",
        "qualification":"DESCRIPTIVE_FIT_SUPPORT_ONLY",
        "worlds":len(worlds),"confirmed_effects":n,
        "action_0":actions[0],"action_1":actions[1],
        "unique_exact_states":len(exact_states),
        "repeated_exact_state_events":n-len(exact_states),
        "exact_state_with_both_actions":paired,
        "terminated":terminals,"truncated":truncations,
        "action_probability_field_logged":False,
        "source_pinned_to_F3":original_authenticated,
        "original_F3_fit_capture_authenticated":original_authenticated,
        "source_claims_F3_header_identity":True,
        "direct_offpolicy_counterfactual_effect_qualified":False,
        "native_hybrid_or_Aopt_gain_proven":False,
        "science_pass":False,
        "authentic_source_effects_executed":False,
        "only_retrospective_fit_evidence":True,
        "new_F6_source_authorized_or_generated":False,
    }


def refuse_unobserved_action_effect(*_args, **_kwargs):
    """No deterministic model guess is allowed to masquerade as factual do()."""
    raise PermissionError("P105_UNOBSERVED_COUNTERFACTUAL_NO_DIRECT_EFFECT")


def validate_historical_summary(measured: dict) -> bool:
    """Check a supplied result against frozen historical FIT metadata only."""
    f=locked_protocol()
    x=f["f3_fit_observation_only"]
    requirements={
        "worlds":x["exact_worlds"],
        "confirmed_effects":x["confirmed_transition_count"],
        "action_0":x["action_0"],
        "action_1":x["action_1"],
        "unique_exact_states":x["unique_exact_4D_state_before"],
        "repeated_exact_state_events":x["duplicate_exact_state_count"],
        "exact_state_with_both_actions":x["exact_state_with_both_actions_count"],
        "terminated":x["source_terminal_count"],
        "truncated":x["truncated_count"],
        "action_probability_field_logged":False,
        "direct_offpolicy_counterfactual_effect_qualified":False,
        "new_F6_source_authorized_or_generated":False,
    }
    if not isinstance(measured,dict) or any(measured.get(k)!=v
                                            for k,v in requirements.items()):
        raise ValueError("P105_HISTORICAL_FIT_SUPPORT_MISMATCH")
    return True
