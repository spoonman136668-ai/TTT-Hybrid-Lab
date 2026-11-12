"""P108: exact-confirmed-action RLS fixture, full 16D precision guard.

Pure deterministic fixture only. No simulator import, reset/do, source archive,
native owner, trained SSM or first-use. P107 ledger has fixture:-only outcomes.
This module does not make any external result or policy competence claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from experiments import p80mr_causalgym_trace_adapter as p80
from experiments import p106_f6_source_neutral_policies as p106
from experiments import p107_f6_owner_readiness as p107

FREEZE_COMMIT="122bbf7201e62cb4c2dcdfcd2089af5f963ed5c5"
FREEZE_BLOB="07f786a898cfe0c6d9841f2b76e1bdf1590f2fd5"
PARENT_SHA="4704a7a7c0beec44c5ca4e4fe86ed3f55703fe7d"
D=16
BLOBS={
    "model.py":"5d9ea57b827aa1796548447868fa5ee14c251b49",
    "experiments/p80mr_causalgym_trace_adapter.py":"24eb0e050e90466bd42e3626e419fbac08c94c72",
    "experiments/p106_f6_source_neutral_policies.py":"e7899e53cb48e6961fcbedf8216efc26a35c39e4",
    "experiments/p107_f6_owner_contract_freeze.json":"95c5d4fe840287c1c86aacd54b6eadd60f2e0d0f",
    "experiments/p107_f6_owner_readiness.py":"0ea2cc00b5a7d86bfaa2a10e3a1585560748e504",
}


def frozen():
    root=Path(__file__).resolve().parents[1]
    raw=(root/"experiments/p108_f6_confirmed_rls_freeze.json").read_bytes()
    if p106._git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("P108_FROZEN_PROTOCOL_BLOB_DRIFT")
    f=json.loads(raw)
    if (f["schema"]!="ttt-hybrid.p108-confirmed-factual-rls-fixture-contract.v1"
        or f["source_identity"]["parent_head"]!=PARENT_SHA
        or f["source_identity"]["canonical_pr"]!=233
        or f["science"]["owner_firstuse_authorized"] is not False
        or f["science"]["external_F6_effects"]!=0
        or f["science"]["source_protocol_fork_unresolved"] is not True
        or f["qualification"]["no_external_firstuse"] is not True):
        raise ValueError("P108_PROTOCOL_SCOPE_DRIFT")
    for path,digest in BLOBS.items():
        if p106._git_blob((root/path).read_bytes())!=digest:
            raise ValueError("P108_PINNED_SOURCE_BLOB_DRIFT")
    p107.frozen()
    return f


@dataclass(frozen=True)
class Model:
    W:torch.Tensor
    P:torch.Tensor
    count:int
    journal_sha256:str


@dataclass(frozen=True)
class Proposal:
    session:str
    step:int
    action:int
    before:tuple[float,float,float,float]
    policy:str
    model_sha256:str
    intent_sha256:str
    seal_sha256:str


def initial():
    return Model(torch.zeros((D,D),dtype=torch.float64),
                 torch.eye(D,dtype=torch.float64)*2.,0,"0"*64)


def validate(model):
    """Full SPD, including nine P80-unused dimensions, not just k'Pk."""
    if (not isinstance(model,Model) or
        not isinstance(model.W,torch.Tensor) or not isinstance(model.P,torch.Tensor)
        or model.W.shape!=(D,D) or model.P.shape!=(D,D)
        or model.W.dtype!=torch.float64 or model.P.dtype!=torch.float64
        or not bool(torch.isfinite(model.W).all())
        or not bool(torch.isfinite(model.P).all())
        or type(model.count) is not int or model.count<0
        or not isinstance(model.journal_sha256,str)
        or len(model.journal_sha256)!=64
        or any(c not in "0123456789abcdef" for c in model.journal_sha256)):
        raise ValueError("P108_BAD_MODEL_STATE")
    if not bool(torch.allclose(model.P,model.P.T,rtol=0.,atol=1e-11)):
        raise ValueError("P108_NONSYMMETRIC_FULL_PRECISION")
    if float(torch.linalg.eigvalsh(model.P).min())<1e-10:
        raise ValueError("P108_NOT_FULL_16D_SPD")
    return model


def model_sha(model):
    validate(model)
    data=(model.W.contiguous().numpy().tobytes()
          +model.P.contiguous().numpy().tobytes()
          +str(model.count).encode("ascii")
          +model.journal_sha256.encode("ascii"))
    return hashlib.sha256(data).hexdigest()


def _key(before,action):
    if type(action) is not int or action not in (0,1):
        raise ValueError("P108_ONLY_AUTHENTIC_BINARY_ACTIONS")
    state=p107._state4(before)
    return torch.tensor(p80._action_key(state,action),dtype=torch.float64)


def _seal(proposal):
    content={
        "session":proposal.session,"step":proposal.step,
        "action":proposal.action,"before":proposal.before,
        "policy":proposal.policy,"model_sha256":proposal.model_sha256,
        "intent_sha256":proposal.intent_sha256,
    }
    return hashlib.sha256(p107._canonical(content)).hexdigest()


def propose(ledger,session,model,policy,before,step,*,epsilon=.05):
    """Creates only a durable P107 FIXTURE intent, never source.do()."""
    frozen()
    validate(model)
    if not isinstance(ledger,p107.FixtureLedger):
        raise PermissionError("P108_ONLY_FIXTURE_LEDGER")
    state=p107._state4(before)
    if type(step) is not int or step!=model.count:
        raise ValueError("P108_MODEL_STEP_DOES_NOT_MATCH_RECEIPTS")
    if policy not in p106.POLICIES:
        raise ValueError("P108_UNKNOWN_POLICY")
    # Initial or continuing fixture episode state must follow last confirmed
    # observed state; no fabricated jump to a different state.
    path=ledger._session(session)
    if step:
        earlier=p107._load(path/f"{step-1:03d}-outcome.json")
        if (earlier.get("external_do_executed") is not False
            or earlier.get("terminated") or earlier.get("truncated")
            or tuple(earlier["state_after"])!=state):
            raise PermissionError("P108_STATE_OR_TERMINAL_CHAIN_DRIFT")
    action=p106.select_policy(
        policy,state,model.W.tolist(),model.P.tolist(),epsilon=epsilon)
    result=ledger.intend(session,step,action,state)
    unsealed=Proposal(session,step,action,state,policy,model_sha(model),
                      result["intent_sha256"],"")
    return Proposal(session,step,action,state,policy,unsealed.model_sha256,
                    unsealed.intent_sha256,_seal(unsealed))


def confirm_fixture(ledger,session,model,proposal,*,outcome_sha256):
    """Only an already-persisted, matching P107 fixture result can update RLS.

    The caller supplies NO outcome, effect or hidden oracle. No external
    action receipt admitted, even if a previously confirmed real one exists.
    """
    frozen()
    validate(model)
    if not isinstance(ledger,p107.FixtureLedger):
        raise PermissionError("P108_ONLY_FIXTURE_LEDGER")
    if (not isinstance(proposal,Proposal)
        or not isinstance(session,str)
        or proposal.session!=session
        or type(proposal.step) is not int or proposal.step!=model.count
        or proposal.action not in (0,1)
        or proposal.model_sha256!=model_sha(model)
        or proposal.seal_sha256!=_seal(proposal)):
        raise ValueError("P108_MISMATCHED_OR_TAMPERED_PROPOSAL")
    path=ledger._session(session)
    if (path/"unknown.json").exists():
        raise PermissionError("P108_UNKNOWN_EFFECT_REPLAY_DENIED")
    intent=p107._load(path/f"{proposal.step:03d}-intent.json")
    if (intent.get("schema")!="p107.fixture-intent.v1"
        or intent.get("session")!=session
        or intent.get("step")!=proposal.step
        or type(intent.get("action")) is not int
        or intent.get("action")!=proposal.action
        or tuple(intent.get("before",[]))!=proposal.before
        or p107._sha(intent)!=proposal.intent_sha256
        or intent.get("unselected_effect")!="UNOBSERVED"):
        raise ValueError("P108_WRONG_INTENT_OR_STATE")
    outcome=p107._load(path/f"{proposal.step:03d}-outcome.json")
    if (outcome.get("schema")!="p107.fixture-outcome.v1"
        or outcome.get("step")!=proposal.step
        or outcome.get("action")!=proposal.action
        or outcome.get("intent_sha256")!=proposal.intent_sha256
        or not isinstance(outcome_sha256,str) or len(outcome_sha256)!=64
        or p107._sha(outcome)!=outcome_sha256
        or outcome.get("external_do_executed") is not False
        or not isinstance(outcome.get("receipt_ref"),str)
        or not outcome["receipt_ref"].startswith("fixture:")
        or type(outcome.get("terminated")) is not bool
        or type(outcome.get("truncated")) is not bool):
        raise ValueError("P108_UNAUTHENTICATED_FIXTURE_OUTCOME")
    state_after=p107._state4(outcome["state_after"])
    reward=p107._finite(outcome["reward"])
    effect=torch.tensor(
        p80._observed_effect(proposal.before,state_after,reward,
                            outcome["terminated"],outcome["truncated"]),
        dtype=torch.float64)
    k=_key(proposal.before,proposal.action)
    g=model.P@k
    denom=1.+k@g
    if not bool(torch.isfinite(denom)) or float(denom)<=0:
        raise ValueError("P108_INVALID_CONFIRMED_RLS_DENOM")
    W=model.W+torch.outer(effect-(model.W@k),g)/denom
    P=model.P-torch.outer(g,g)/denom
    next_hash=hashlib.sha256((
        model.journal_sha256+"|"+proposal.intent_sha256+"|"
        +p107._sha(outcome)).encode("ascii")).hexdigest()
    next_state=Model(W,P,model.count+1,next_hash)
    validate(next_state)
    return next_state


def read(model,before,action):
    validate(model)
    return model.W@_key(before,action)


def snapshot(model):
    validate(model)
    return Model(model.W.clone(),model.P.clone(),
                 model.count,model.journal_sha256)


def refuse_external_firstuse(*_args,**_kwargs):
    raise PermissionError("P108_F6_EXTERNAL_SCIENCE_FIRSTUSE_DENIED")
