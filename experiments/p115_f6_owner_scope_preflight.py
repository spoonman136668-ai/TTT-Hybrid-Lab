"""P115: detect unapproved F6 protocol forks and rehearse POSIX custody.

NO original simulator import or operation. Every claimed session, intent,
outcome and checkpoint ID in this module is fixture-only and must never be
accepted by an owner first-use route. A Linux CI directory-fsync smoke does
NOT qualify actual WSL owner, storage, or original scientific data.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re

from experiments import p106_f6_source_neutral_policies as p106
from experiments import p107_f6_owner_readiness as p107
from experiments import p110_f6_five_arm_readiness as p110
from experiments import p111_f6_fit16_preflight as p111
from experiments import p112_f6_holdout_preflight as p112
from experiments import p114_f6_native_optimizer_smoke as p114

ROOT=Path(__file__).resolve().parents[1]
FROZEN_BLOB="66efbd48afe9bb1b7e591c397638669826fcff98"
FROZEN_COMMIT="aef25179307db97eb84ce6eeda3abd79a56adf17"
PARENT_SHA="5523c445140bb6feefcdcc0535d905194d8374a9"
SESSIONS_PER_FIT=240
A5=p110.ARMS
HEX64=re.compile(r"[0-9a-f]{64}\Z")


def _canon(obj):
    return json.dumps(obj,sort_keys=True,separators=(",",":"),
                      allow_nan=False).encode("utf-8")+b"\n"


def _digest(obj):
    return hashlib.sha256(_canon(obj)).hexdigest()


def locked():
    raw=(ROOT/"experiments/p115_f6_owner_scope_freeze.json").read_bytes()
    if p106._git_blob(raw)!=FROZEN_BLOB:
        raise ValueError("P115_FROZEN_BLOB_DRIFT")
    f=json.loads(raw)
    if (f.get("schema")!="ttt-hybrid.p115-f6-protocol-collision-and-posix-custody-rehearsal.v1"
        or f["parent"]["exact_head"]!=PARENT_SHA
        or f["competing"]["P106_A4"]["freeze_blob"]!=p106.FREEZE_BLOB
        or f["competing"]["P110_A5_P112_REVISION"]["freeze_blob"]!=p110.FREEZE_BLOB
        or f["competing"]["P106_B4"]["freeze_blob"]!="42db0228c1c4561333058a3b9d849b8c3db1183c"
        or f["proposed_only"]["proposed_FIT_sessions"]!=SESSIONS_PER_FIT
        or f["proposed_only"]["proposed_FIT_actions"]!=SESSIONS_PER_FIT*96
        or f["proposed_only"]["not_authorized_revision"] is not True
        or f["custody_rehearsal"]["no_admission_to_source"] is not True
        or f["authority"]["protocol_fork_selected"] is not False
        or f["authority"]["F6_FIT_source_firstuse_authorized"] is not False
        or f["authority"]["CAL_source_firstuse_authorized"] is not False
        or f["authority"]["owner_wsl_native_custody_qualified"] is not False
        or f["authority"]["scientific_pass"] is not False):
        raise ValueError("P115_PROTOCOL_OR_AUTHORITY_DRIFT")
    p107.frozen()
    p110.locked_plan()
    p111.locked_protocol()
    p112.frozen()
    p114.locked()
    return f


def protocol_matrix():
    f=locked()
    a=p107.frozen()["dual_protocol_conflict"]
    proposed=p110.locked_plan()
    revised=p112.frozen()
    first,other=a["first"],a["other"]
    choices=f["competing"]
    if (first["freeze_blob"]!=choices["P106_A4"]["freeze_blob"]
        or other["freeze_blob"]!=choices["P106_B4"]["freeze_blob"]
        or first["policy_arms"]!=choices["P106_A4"]["arms"]
        or other["policy_arms"]!=choices["P106_B4"]["arms"]
        or proposed["experiment"]["policy_order"]!=choices["P110_A5_P112_REVISION"]["arms"]
        or proposed["experiment"]["first_use_FIT_only_max_sessions"]!=SESSIONS_PER_FIT
        or proposed["experiment"]["first_use_FIT_only_max_effects"]!=23040
        or revised["data"]["train_worlds_n"]!=24
        or revised["data"]["validation_worlds_n"]!=8
        or revised["result"]["P110_protocol_changed"] is not False
        or revised["result"]["owner_protocol_A5_selected"] is not False):
        raise ValueError("P115_INCOMPATIBLE_FROZEN_SOURCE_MATRIX")
    fit_keys={(1820000000+1000*b+i,policy)
              for b in range(3) for i in range(16) for policy in A5}
    if len(fit_keys)!=SESSIONS_PER_FIT:
        raise ValueError("P115_FIT_SCOPE_DUPLICATION")
    return {
        "schema":"ttt-hybrid.p115-f6-unapproved-protocol-review.v1",
        "protocol_choices":tuple(choices),
        "legacy_A4_FIT_sessions":choices["P106_A4"]["sessions"],
        "alternative_B4_FIT_sessions":choices["P106_B4"]["sessions"],
        "proposed_A5_FIT_sessions":len(fit_keys),
        "proposed_A5_FIT_max_effects":23040,
        "future_CAL_max_effects_if_separately_approved":23040,
        "SSM_training_worlds":24,
        "SSM_seed_selection_worlds":8,
        "FIT_conventional_selection_worlds_per_arm":16,
        "world_partition_disjoint":True,
        "multiple_incompatible_original_proposals":True,
        "recommendation_only":"P110_A5_WITH_P112_24TRAIN_8VAL",
        "protocol_selected":False,
        "real_FIT_source_authorized":False,
        "owner_native_storage_qualified":False,
        "authentic_trained_SSM_qualified":False,
        "source_effects_executed":0,
        "science_pass":False,
        "deployment":False,
    }


def _hex(value):
    if not isinstance(value,str) or HEX64.fullmatch(value) is None:
        raise ValueError("P115_INVALID_MODEL_DIGEST")
    return value


def _state(value):
    if not isinstance(value,(list,tuple)) or len(value)!=4:
        raise ValueError("P115_BAD_PHYSICAL_STATE")
    if any(type(x) not in (float,int) or not math.isfinite(x) for x in value):
        raise ValueError("P115_NONFINITE_PHYSICAL_STATE")
    return list(map(float,value))


def _dir_fsync(path):
    if os.name!="posix" or not hasattr(os,"O_DIRECTORY"):
        raise PermissionError("P115_POSIX_DIR_FSYNC_UNAVAILABLE")
    flags=os.O_RDONLY|os.O_DIRECTORY
    if hasattr(os,"O_NOFOLLOW"):
        flags|=os.O_NOFOLLOW
    fd=os.open(path,flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write(path,record):
    # Never replace, truncate, unlink or replay an uncertain attempt.
    data=_canon(record)
    flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL
    if hasattr(os,"O_NOFOLLOW"):
        flags|=os.O_NOFOLLOW
    fd=os.open(path,flags,0o600)
    with os.fdopen(fd,"wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    _dir_fsync(path.parent)
    return _digest(record)


def _read(path):
    if path.is_symlink():
        raise ValueError("P115_SYMLINKED_RECEIPT")
    raw=path.read_bytes()
    obj=json.loads(raw)
    if type(obj) is not dict or raw!=_canon(obj):
        raise ValueError("P115_NONCANONICAL_RECEIPT")
    return obj


class FixtureJournal:
    """Only fake, never externally replayable action intents/outcomes.

    A pre-source durable write-ahead prototype for Linux-hosted CI.
    Actual Windows->WSL owner directory persistence is NOT established here.
    """
    def __init__(self,root):
        protocol_matrix()
        if os.name!="posix":
            raise PermissionError("P115_WSL_OWNER_DURABILITY_UNVERIFIED_WINDOWS")
        root=Path(root)
        if not root.is_dir() or root.is_symlink():
            raise ValueError("P115_REQUIRED_UNLINKED_DIRECTORY")
        self.root=root.resolve()
        _dir_fsync(self.root)

    def claim(self,block,index,policy,model_sha):
        model_sha=_hex(model_sha)
        if (type(block) is not int or block not in range(3)
            or type(index) is not int or index not in range(16)
            or policy not in A5):
            raise PermissionError("P115_ONLY_FIT_A5_FIXTURE_ALLOWED")
        seed=p106.future_seed_schedule("FIT",block,index)
        name=f"fixture-FIT-{seed}-{policy}"
        location=self.root/name
        location.mkdir(mode=0o700,exist_ok=False)
        _dir_fsync(self.root)
        _write(location/"claim.json",{
            "schema":"p115.fixture-claim.v1",
            "variant":"UNAPPROVED_P110_A5_P112_REHEARSAL",
            "block":block,"index":index,"policy":policy,
            "seed":seed,"model_sha256":model_sha,
            "source_attempted":False,"real_F6_authority":False})
        return name

    def _session(self,name):
        if not isinstance(name,str):
            raise ValueError("P115_BAD_SESSION")
        m=re.fullmatch(r"fixture-FIT-(\d+)-([a-z][a-z0-9_]*)",name)
        if m is None or m.group(2) not in A5:
            raise ValueError("P115_BAD_SESSION")
        location=self.root/name
        if (not location.is_dir() or location.is_symlink()
            or location.resolve().parent!=self.root):
            raise ValueError("P115_UNCLAIMED_OR_SYMLINKED_SESSION")
        claim=_read(location/"claim.json")
        if (set(claim)!={"schema","variant","block","index","policy",
                         "seed","model_sha256","source_attempted","real_F6_authority"}
            or claim["schema"]!="p115.fixture-claim.v1"
            or claim["variant"]!="UNAPPROVED_P110_A5_P112_REHEARSAL"
            or claim["source_attempted"] is not False
            or claim["real_F6_authority"] is not False
            or claim["policy"]!=m.group(2)
            or type(claim["block"]) is not int or claim["block"] not in range(3)
            or type(claim["index"]) is not int or claim["index"] not in range(16)
            or claim["seed"]!=p106.future_seed_schedule(
                "FIT",claim["block"],claim["index"])
            or str(claim["seed"])!=m.group(1)):
            raise ValueError("P115_TAMPERED_CLAIM")
        _hex(claim["model_sha256"])
        return location,claim

    def intend(self,name,step,action,before,model_sha):
        location,claim=self._session(name)
        if (location/"unknown.json").exists():
            raise PermissionError("P115_UNKNOWN_EFFECT_NO_REPLAY")
        if (type(step) is not int or not 0<=step<96 or
            type(action) is not int or action not in (0,1) or
            _hex(model_sha)!=claim["model_sha256"]):
            raise ValueError("P115_INVALID_INTENT_OR_MODEL")
        if step:
            prev=_read(location/f"{step-1:03d}-outcome.json")
            if prev.get("terminated") or prev.get("truncated"):
                raise PermissionError("P115_AFTER_TERMINAL")
            if _state(before)!=_state(prev["after"]):
                raise ValueError("P115_BROKEN_PREVIOUS_STATE")
        row={"schema":"p115.fixture-intent.v1","session":name,
             "step":step,"action":action,"before":_state(before),
             "model_sha256":model_sha,"unselected_effect":"UNOBSERVED",
             "real_effect_executed":False}
        return _write(location/f"{step:03d}-intent.json",row)

    def confirm_fixture(self,name,step,intent_digest,action,after,reward,
                        terminated,truncated,receipt_ref):
        location,_=self._session(name)
        if (location/"unknown.json").exists():
            raise PermissionError("P115_UNKNOWN_EFFECT_NO_REPLAY")
        if type(step) is not int or not 0<=step<96:
            raise ValueError("P115_BAD_EFFECT_STEP")
        intent=_read(location/f"{step:03d}-intent.json")
        if (intent.get("schema")!="p115.fixture-intent.v1"
            or intent.get("session")!=name or intent.get("step")!=step
            or intent.get("unselected_effect")!="UNOBSERVED"
            or intent.get("real_effect_executed") is not False
            or _digest(intent)!=intent_digest
            or type(action) is not int or action!=intent["action"]):
            raise ValueError("P115_ACTION_INTENT_MISMATCH")
        if (not isinstance(receipt_ref,str)
            or not receipt_ref.startswith("fixture:")
            or len(receipt_ref)>128):
            raise PermissionError("P115_NOT_A_FIXTURE_RECEIPT")
        if (type(terminated) is not bool or type(truncated) is not bool
            or terminated and truncated or
            type(reward) not in (int,float) or not math.isfinite(reward)):
            raise ValueError("P115_BAD_FACTUAL_EFFECT")
        record={"schema":"p115.fixture-outcome.v1",
                "session":name,"step":step,
                "intent_sha256":intent_digest,"action":action,
                "after":_state(after),"reward":float(reward),
                "terminated":terminated,"truncated":truncated,
                "receipt_ref":receipt_ref,"external_source":False}
        return _write(location/f"{step:03d}-outcome.json",record)

    def quarantine(self,name,step):
        location,_=self._session(name)
        if type(step) is not int or not 0<=step<96:
            raise ValueError("P115_UNKNOWN_STEP")
        if (not (location/f"{step:03d}-intent.json").exists()
            or (location/f"{step:03d}-outcome.json").exists()):
            raise ValueError("P115_ONLY_PENDING_MAY_QUARANTINE")
        return _write(location/"unknown.json",{
            "schema":"p115.fixture-unknown.v1",
            "step":step,"uncertain_effect_replay_allowed":False,
            "real_source_attempted":False})

    def audit(self,name):
        location,_=self._session(name)
        intents={}
        outcomes={}
        for path in location.iterdir():
            if path.is_symlink():
                raise ValueError("P115_SYMLINKED_RECEIPT")
            if path.name in ("claim.json","unknown.json"):
                continue
            m=re.fullmatch(r"(\d{3})-(intent|outcome)\.json",path.name)
            if m is None or not 0<=int(m.group(1))<96:
                raise ValueError("P115_UNKNOWN_RECEIPT_FILE")
            record=_read(path)
            d=int(m.group(1))
            (intents if m.group(2)=="intent" else outcomes)[d]=record
        if (set(intents)!=set(range(len(intents)))
            or not set(outcomes).issubset(set(intents))
            or set(outcomes)!=set(range(len(outcomes)))):
            raise ValueError("P115_BROKEN_RECEIPT_CHAIN")
        for step,intent in intents.items():
            if (intent.get("schema")!="p115.fixture-intent.v1"
                or intent.get("step")!=step
                or intent.get("session")!=name
                or intent.get("real_effect_executed") is not False
                or intent.get("unselected_effect")!="UNOBSERVED"):
                raise ValueError("P115_TAMPERED_INTENT")
            if step in outcomes:
                out=outcomes[step]
                if (out.get("schema")!="p115.fixture-outcome.v1"
                    or out.get("session")!=name or out.get("step")!=step
                    or out.get("action")!=intent.get("action")
                    or out.get("intent_sha256")!=_digest(intent)
                    or out.get("external_source") is not False
                    or not str(out.get("receipt_ref","")).startswith("fixture:")):
                    raise ValueError("P115_TAMPERED_OUTCOME")
                if step and _state(intent["before"])!=_state(outcomes[step-1]["after"]):
                    raise ValueError("P115_BROKEN_PHYSICAL_CHAIN")
                if step+1 in intents and (out.get("terminated") or out.get("truncated")):
                    raise ValueError("P115_POST_TERMINAL_CHAIN")
        unknown=location/"unknown.json"
        if unknown.exists():
            mark=_read(unknown)
            if (mark.get("schema")!="p115.fixture-unknown.v1"
                or mark.get("uncertain_effect_replay_allowed") is not False
                or mark.get("real_source_attempted") is not False
                or mark.get("step")!=len(outcomes)):
                raise ValueError("P115_BAD_QUARANTINE")
        return {
            "status":"UNKNOWN_BLOCKED" if unknown.exists() else (
              "UNCONFIRMED_INTENT_BLOCKED" if len(intents)>len(outcomes)
              else "CLOSED_FIXTURE_ONLY"),
            "intents_fixture":len(intents),
            "outcomes_fixture":len(outcomes),
            "posix_dir_fsync_test_only":True,
            "native_owner_WSL_durability_verified":False,
            "real_source_actions":0,
            "protocol_approved":False,
            "science_pass":False,
        }


def deny_source(*_args,**_kwargs):
    raise PermissionError("P115_F6_FIRST_USE_AND_PROTOCOL_CHOICE_NOT_AUTHORIZED")
