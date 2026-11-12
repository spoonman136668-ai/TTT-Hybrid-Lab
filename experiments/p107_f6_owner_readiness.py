"""P107: immutable *fixture-only* receipt contract for future F6 owner first-use.

The module contains no environment, network, or owner-run capability.
The F6 owner first-use, native filesystem durability and conventional SSM
competence are NOT qualified by these tests. All writes use user-supplied
temporary fixture directories and are never real CartPoleWind effects.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from experiments import p106_f6_source_neutral_policies as p106

FREEZE_COMMIT="fb74b8d245e7c9d1e13403cc35ee36487f0a781b"
FREEZE_BLOB="95c5d4fe840287c1c86aacd54b6eadd60f2e0d0f"
PARENT_SHA="9368dd4015b7e9151ff525f88d5a64e831f1713b"
P106_BLOB="e7899e53cb48e6961fcbedf8216efc26a35c39e4"
ACTUAL_POLICY_SET=p106.POLICIES
CONTROL_SET=("a_opt","greedy_goal","physical_balance")
MAX_ACTIONS=96
WORLDS_PER_SPLIT=48
PROJECT_ROOT=Path(__file__).resolve().parents[1]


def frozen():
    """Validate frozen protocol and current P106 source with zero F6 effects."""
    raw=(PROJECT_ROOT/"experiments/p107_f6_owner_contract_freeze.json").read_bytes()
    if p106._git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("P107_FROZEN_PROTOCOL_DRIFT")
    f=json.loads(raw)
    if (f.get("schema")!="ttt-hybrid.p107-owner-firstuse-readiness-contract-freeze.v1"
        or f.get("parent",{}).get("sha")!=PARENT_SHA
        or f["dual_protocol_conflict"]["status"]!="UNRESOLVED_NO_F6_ACTIVATION"
        or f["dual_protocol_conflict"]["must_not_implicitly_pick_one"] is not True
        or f["owner"]["run_owner_scoped_authorization"]!="REQUIRED_NOT_GRANTED"
        or f["future_science"]["F6_effect_attempts"]!=0
        or f["future_science"]["scientific_pass"] is not False
        or f["immutable_effect_receipt_contract"]["cal_before_fit"].startswith("Deny CAL") is not True):
        raise ValueError("P107_PROTOCOL_OR_AUTHORITY_DRIFT")
    code=(PROJECT_ROOT/"experiments/p106_f6_source_neutral_policies.py").read_bytes()
    if p106._git_blob(code)!=P106_BLOB:
        raise ValueError("P107_P106_SOURCE_BLOB_DRIFT")
    p106.locked_protocol()
    return f


def _canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")


def _sha(obj):
    return hashlib.sha256(_canonical(obj)).hexdigest()


def _finite(v):
    if type(v) not in (int,float) or not math.isfinite(v):
        raise ValueError("P107_NONFINITE_OR_NONNUMERIC")
    return float(v)


def _state4(value):
    if not isinstance(value,(list,tuple)) or len(value)!=4:
        raise ValueError("P107_WRONG_STATE_DIMENSION")
    return tuple(_finite(x) for x in value)


def _session_key(split,block,index,policy):
    if (split!="FIT" or type(block) is not int or block not in range(3)
        or type(index) is not int or index not in range(16)
        or policy not in ACTUAL_POLICY_SET):
        raise PermissionError("P107_CAL_OOD_AND_UNKNOWN_SOURCE_DENIED")
    seed=p106.future_seed_schedule("FIT",block,index)
    return f"fixture-FIT-{seed}-{policy}"


def _exclusive_file(path:Path,payload:dict):
    """Atomic create + file fsync; NOT proof of WSL owner directory durability."""
    content=_canonical(payload)+b"\n"
    flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL
    fd=os.open(path,flags,0o600)
    try:
        with os.fdopen(fd,"wb",closefd=True) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # Leave any claimed path in place; never delete/retry an
        # unknown-outcome attempt.
        raise


def _load(path):
    data=path.read_bytes()
    parsed=json.loads(data)
    if data!=_canonical(parsed)+b"\n":
        raise ValueError("P107_NONCANONICAL_OR_TAMPERED_FIXTURE")
    return parsed


class FixtureLedger:
    """Synthetic test-only session receipt ordering; no external effect adapter.

    O_EXCL prevents two fixture claims or writes of the same event. Any
    intended event missing its outcome forbids a following event.
    A terminal/truncated outcome forbids any later step.
    """
    def __init__(self,root):
        frozen()
        self.root=Path(root)
        if not self.root.is_dir() or self.root.is_symlink():
            raise ValueError("P107_FIXTURE_ROOT_REQUIRED")
        self.root=self.root.resolve()

    def claim(self,split,block,index,policy):
        identity=_session_key(split,block,index,policy)
        path=self.root/identity
        # On filesystem collision, preserve the existing attempted session.
        path.mkdir(mode=0o700,exist_ok=False)
        try:
            _exclusive_file(path/"claim.json",{
                "schema":"p107.fixture-session.v1",
                "split":split,"block":block,"index":index,"policy":policy,
                "trial_seed":p106.future_seed_schedule(split,block,index),
                "not_external_firstuse":True,
                "F6_effects_executed":False,
            })
        except BaseException:
            # Do not automatically unlink an uncertain claim.
            raise
        return identity

    def _session(self,identity):
        # Re-validate entire identity against frozen protocol. Prevent path
        # traversal, symlink injection, and accidental CAL/OOD fixture birth.
        if not isinstance(identity,str):
            raise ValueError("P107_BAD_SESSION_ID")
        name=identity.split("-")
        if len(name)!=4 or name[0:2]!=["fixture","FIT"]:
            raise ValueError("P107_BAD_SESSION_ID")
        try:
            seed=int(name[2])
        except ValueError as e:
            raise ValueError("P107_BAD_SESSION_ID") from e
        policy=name[3]
        if policy not in ACTUAL_POLICY_SET:
            raise ValueError("P107_BAD_SESSION_ID")
        allowed={p106.future_seed_schedule("FIT",block,idx)
                 for block in range(3) for idx in range(16)}
        if seed not in allowed:
            raise ValueError("P107_BAD_SESSION_ID")
        path=self.root/identity
        if not path.is_dir() or path.is_symlink() or path.resolve().parent!=self.root:
            raise ValueError("P107_BAD_OR_UNCLAIMED_SESSION")
        claim=_load(path/"claim.json")
        if (claim.get("schema")!="p107.fixture-session.v1"
            or claim.get("not_external_firstuse") is not True
            or claim.get("F6_effects_executed") is not False
            or claim.get("split")!="FIT" or claim.get("policy")!=policy
            or claim.get("trial_seed")!=seed
            or _session_key(claim.get("split"),claim.get("block"),
                            claim.get("index"),claim.get("policy"))!=identity):
            raise ValueError("P107_TAMPERED_SESSION_CLAIM")
        return path

    def intend(self,identity,step,action,state_before):
        path=self._session(identity)
        if type(step) is not int or not 0<=step<MAX_ACTIONS:
            raise ValueError("P107_BAD_EFFECT_INDEX")
        if type(action) is not int or action not in (0,1):
            raise ValueError("P107_NOT_DISCRETE_FACTUAL_ACTION")
        state=_state4(state_before)
        if (path/"unknown.json").exists():
            raise PermissionError("P107_UNKNOWN_EFFECT_REPLAY_DENIED")
        if step:
            previous=_load(path/f"{step-1:03d}-outcome.json")
            if previous["terminated"] or previous["truncated"]:
                raise PermissionError("P107_POST_TERMINAL_ATTEMPT_DENIED")
        payload={
            "schema":"p107.fixture-intent.v1",
            "step":step,"action":action,"before":state,
            "session":identity,"unselected_effect":"UNOBSERVED",
        }
        _exclusive_file(path/f"{step:03d}-intent.json",payload)
        return {"step":step,"intent_sha256":_sha(payload),"action":action}

    def confirm_fixture(self,identity,step,intent_sha256,action,
                        state_after,reward,terminated,truncated,receipt_ref):
        path=self._session(identity)
        if (path/"unknown.json").exists():
            raise PermissionError("P107_UNKNOWN_EFFECT_REPLAY_DENIED")
        if type(step) is not int or not 0<=step<MAX_ACTIONS:
            raise ValueError("P107_BAD_EFFECT_INDEX")
        intent=_load(path/f"{step:03d}-intent.json")
        if (intent.get("schema")!="p107.fixture-intent.v1"
            or intent.get("session")!=identity or intent.get("step")!=step
            or intent.get("unselected_effect")!="UNOBSERVED"
            or not isinstance(intent_sha256,str) or _sha(intent)!=intent_sha256
            or type(action) is not int or intent.get("action")!=action):
            raise ValueError("P107_WRONG_OR_UNCOMMITTED_ACTION_CREDIT")
        if (not isinstance(receipt_ref,str) or
            not receipt_ref.startswith("fixture:") or len(receipt_ref)>128):
            raise PermissionError("P107_ONLY_FIXTURE_RECEIPT_PERMITTED")
        if type(terminated) is not bool or type(truncated) is not bool:
            raise ValueError("P107_BAD_TERMINAL_STATUS")
        post=_state4(state_after)
        gain=_finite(reward)
        outcome={
            "schema":"p107.fixture-outcome.v1",
            "step":step,"action":action,"intent_sha256":intent_sha256,
            "state_after":post,"reward":gain,
            "terminated":terminated,"truncated":truncated,
            "receipt_ref":receipt_ref,"external_do_executed":False,
        }
        _exclusive_file(path/f"{step:03d}-outcome.json",outcome)
        return {"outcome_sha256":_sha(outcome),"step":step,
                "not_external_science":True}

    def mark_unknown(self,identity,step):
        path=self._session(identity)
        if type(step) is not int or not 0<=step<MAX_ACTIONS:
            raise ValueError("P107_BAD_EFFECT_INDEX")
        if not (path/f"{step:03d}-intent.json").exists():
            raise ValueError("P107_NO_INTENT_TO_QUARANTINE")
        if (path/f"{step:03d}-outcome.json").exists():
            raise ValueError("P107_ALREADY_CONFIRMED_NO_UNKNOWN")
        _exclusive_file(path/"unknown.json",{
            "schema":"p107.fixture-unknown.v1",
            "step":step,"cannot_replay":True,
            "source_attempt":"NONE_FIXTURE_ONLY",
        })


def fit_select_fixture(rows):
    """Mock-only computation of fixed-source FIT incumbent selection.

    Never qualifies a real baseline or unlocks CAL; required independent
    SSM checkpoint/owner provenance absent from this synthetic test seam.
    """
    frozen()
    if not isinstance(rows,dict) or set(rows)!=set(CONTROL_SET):
        raise ValueError("P107_INCOMPLETE_STRONG_CONVENTIONAL_ARMS")
    result={}
    for control in CONTROL_SET:
        samples=rows[control]
        if not isinstance(samples,list) or len(samples)!=WORLDS_PER_SPLIT:
            raise ValueError("P107_INCOMPLETE_FIT_EPISODES")
        observed=set()
        lengths=[]
        penalties=[]
        for item in samples:
            if not isinstance(item,dict) or set(item)!={
                "block","index","survival_steps","physical_loss",
                "terminated","truncated","receipt_ref"}:
                raise ValueError("P107_WRONG_FIT_RECEIPT_FIELDS")
            block,index=item["block"],item["index"]
            if (type(block) is not int or block not in range(3)
                or type(index) is not int or index not in range(16)
                or (block,index) in observed):
                raise ValueError("P107_MISSING_OR_DUPLICATE_FIT_WORLD")
            observed.add((block,index))
            steps=item["survival_steps"]
            if type(steps) is not int or not 1<=steps<=MAX_ACTIONS:
                raise ValueError("P107_BAD_SURVIVAL")
            loss=_finite(item["physical_loss"])
            if (loss<0 or type(item["terminated"]) is not bool
                or type(item["truncated"]) is not bool
                or (steps<MAX_ACTIONS and not
                    (item["terminated"] or item["truncated"]))
                or not isinstance(item["receipt_ref"],str)
                or not item["receipt_ref"].startswith("fixture:")):
                raise ValueError("P107_MISSING_OR_FORGED_FIT_OUTCOME")
            lengths.append(steps)
            penalties.append(loss)
        result[control]={
            "mean_survival":sum(lengths)/len(lengths),
            "mean_physical_loss":sum(penalties)/len(penalties),
        }
    best=min(CONTROL_SET,key=lambda arm:(
        -result[arm]["mean_survival"],result[arm]["mean_physical_loss"],
        CONTROL_SET.index(arm)))
    return {
        "selected":best,"controls":result,
        "competence_fixture_only":result[best]["mean_survival"]>=24.,
        "real_FIT_train_checkpoint":"UNVERIFIED",
        "real_CAL_admission":False,
        "owner_approval":"ABSENT",
        "science_pass":False,
    }


def deny_external_firstuse(*_args,**_kwargs):
    """Only a separately authorized F6 owner adapter may ever act."""
    raise PermissionError("P107_EXTERNAL_F6_FIRSTUSE_UNAUTHORIZED_PROTOCOL_CONFLICT")


def engineering_receipt():
    f=frozen()
    return {
        "schema":"ttt-hybrid.p107-owner-f6-readiness.v1",
        "status":"SOURCE_NEUTRAL_ENG_ONLY_BLOCK_F6",
        "freeze_commit":FREEZE_COMMIT,"freeze_blob":FREEZE_BLOB,
        "parent_sha":PARENT_SHA,
        "dual_protocol_fork_unresolved":f["dual_protocol_conflict"]["status"]=="UNRESOLVED_NO_F6_ACTIVATION",
        "fixture_ledger_not_owner_source":True,
        "proposed_sessions_if_authorized":384,
        "proposed_max_effects_if_authorized":36864,
        "real_source_sessions":0,"real_effects":0,
        "cal_unlocked":False,"owner_firstuse_authorized":False,
        "runner_dispatched":False,"new_workflows":False,
        "scientific_pass":False,
    }
