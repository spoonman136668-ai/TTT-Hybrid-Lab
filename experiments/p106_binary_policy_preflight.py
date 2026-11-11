"""P106: source-neutral, two-action causal-design policy engineering preflight.

NOT an F6 owner route. No real simulator import/reset/do, no network, no
archive access, no runner dispatch or model training. Reproduce the exact
P80 public key and P101 RLS algebra on a pure fixture, with P104-informed
A-optimal design as a strong conventional policy *interface*.

An eventual F6 first-use is a separate scoped authority and frozen trial.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch

from experiments import p105_f3_action_support_preflight as p105


FROZEN_COMMIT="383298f9c5913b77d80ed77fd7c364158657498d"
FROZEN_BLOB="42db0228c1c4561333058a3b9d849b8c3db1183c"
D=16
SCALES=(2.4,3.0,0.2095,3.5)
POLICIES=("max_variance","a_opt","hash_passive","cyclic")
PINNED={
    "model.py":"5d9ea57b827aa1796548447868fa5ee14c251b49",
    "experiments/p80mr_causalgym_trace_adapter.py":"24eb0e050e90466bd42e3626e419fbac08c94c72",
    "experiments/p105_authentic_action_support_freeze.json":"cd1bb40fd74df9d693067798fa5360a0cb205cbb",
    "experiments/p105_f3_action_support_preflight.py":"ab1bcb17da2e4eade5b0422e627cedb7b41da544",
}


def frozen():
    root=Path(__file__).resolve().parents[1]
    file=root/"experiments/p106_binary_policy_freeze.json"
    if p105._git_blob(file.read_bytes())!=FROZEN_BLOB:
        raise ValueError("P106_FROZEN_BLOB_DRIFT")
    f=json.loads(file.read_text(encoding="utf-8"))
    if (f.get("schema")!="ttt-hybrid.p106-f6-two-action-policy-preflight.v1"
        or f["identity"]["base_sha"]!="955d032ac21b71b9d5e02061ae9727600205d569"
        or f["factual_task"]["actual_action_domain"]!=[0,1]
        or f["factual_task"]["state_scales"]!=list(SCALES)
        or f["prospective_F6_design_only"]["authorization"]!="NOT_GRANTED_BY_P105_OR_THIS_FREEZE"
        or f["implementation_contract"]["no_env_constructor_reset_do"] is not True
        or f["implementation_contract"]["no_experimental_source_birth"] is not True):
        raise ValueError("P106_SCOPE_DRIFT")
    for rel,sha in PINNED.items():
        if p105._git_blob((root/rel).read_bytes())!=sha:
            raise ValueError("P106_PINNED_ORIGINAL_SOURCE_DRIFT")
    return f


def _four(state):
    if (not isinstance(state,(list,tuple)) or len(state)!=4
        or any(type(x) not in (int,float) or not math.isfinite(x) for x in state)):
        raise ValueError("P106_BAD_PUBLIC_PHYSICAL_STATE")
    return tuple(float(v) for v in state)


def public_action_key(state,action):
    """Bitwise-compatible operation order with P80's 16D state/action key."""
    s=_four(state)
    if type(action) is not int or action not in (0,1):
        raise ValueError("P106_ONLY_TWO_CAUSAL_ACTIONS")
    raw=[1.]+[math.tanh(x/scale) for x,scale in zip(s,SCALES)]
    raw+=[float(action==0),float(action==1)]
    raw+=[0.]*9
    denom=math.sqrt(math.fsum(x*x for x in raw))
    return torch.tensor([v/denom for v in raw],dtype=torch.float64)


def action_pair(state):
    return torch.stack((public_action_key(state,0),public_action_key(state,1)))


@dataclass(frozen=True)
class RLS:
    weights:torch.Tensor
    precision:torch.Tensor
    confirmed_count:int
    journal_sha256:str


@dataclass(frozen=True)
class Proposal:
    event_number:int
    action:int
    policy:str
    state_before:tuple[float,float,float,float]
    model_state_sha256:str
    opaque_trial_key:str
    public_step:int
    commitment_sha256:str


def initial():
    # Exact P101 prior 0.5 -> P=I/0.5 and W=zeros 16x16, float64.
    return RLS(torch.zeros((D,D),dtype=torch.float64),
               2.*torch.eye(D,dtype=torch.float64),0,"0"*64)


def _check_model(model):
    if (not isinstance(model,RLS) or
        model.weights.shape!=(D,D) or model.precision.shape!=(D,D)
        or model.weights.dtype!=torch.float64 or
        model.precision.dtype!=torch.float64
        or not bool(torch.isfinite(model.weights).all())
        or not bool(torch.isfinite(model.precision).all())
        or not bool(torch.allclose(model.precision,model.precision.T,rtol=1e-11,atol=1e-11))
        or float(torch.linalg.eigvalsh(model.precision).min()) <= 0.
        or type(model.confirmed_count) is not int or model.confirmed_count<0
        or not isinstance(model.journal_sha256,str)
        or len(model.journal_sha256)!=64):
        raise ValueError("P106_BAD_PERSISTENT_MODEL_STATE")


def model_sha(model):
    _check_model(model)
    raw=model.weights.contiguous().numpy().tobytes()+model.precision.contiguous().numpy().tobytes()
    raw+=str(model.confirmed_count).encode("ascii")+model.journal_sha256.encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _check_sigma(sigma):
    if (not isinstance(sigma,torch.Tensor) or sigma.shape!=(D,D)
        or sigma.dtype!=torch.float64 or
        not bool(torch.isfinite(sigma).all())
        or not bool(torch.allclose(sigma,sigma.T,rtol=1e-12,atol=1e-12))
        or float(torch.linalg.eigvalsh(sigma).min())< -1e-12
        or float(sigma.trace())<=0.):
        raise ValueError("P106_UNQUALIFIED_PUBLIC_FIT_COVARIANCE")


def choose(model,state_before,policy,*,sigma=None,
           trial_public_key,step):
    """NO true map, no reward, no unchosen effect, no future query input."""
    _check_model(model)
    _four(state_before)
    if (policy not in POLICIES or
        not isinstance(trial_public_key,str) or not 1<=len(trial_public_key)<=128
        or type(step) is not int or step<0):
        raise ValueError("P106_POLICY_SCOPE")
    keys=action_pair(state_before)
    if policy=="cyclic":
        return step%2
    if policy=="hash_passive":
        data=f"p106-f6-future|{trial_public_key}|{step}".encode("utf-8")
        return hashlib.sha256(data).digest()[0]&1
    P=model.precision
    Pg=keys@P
    uncertainty=(Pg*keys).sum(-1)
    if policy=="max_variance":
        return int(torch.argmax(uncertainty))
    _check_sigma(sigma)
    # Standard A-optimal one-step query covariance reduction, FIT-only
    # sigma must be externally selected without any CAL/heldout outcome.
    gain=((Pg@sigma)*Pg).sum(-1)/(1.+uncertainty)
    return int(torch.argmax(gain))


def commit(model,state_before,policy,*,sigma=None,trial_public_key,step):
    """Pure proposal: no physical action is executed or even available."""
    frozen()
    state=_four(state_before)
    chosen=choose(model,state,policy,sigma=sigma,
                  trial_public_key=trial_public_key,step=step)
    model_id=model_sha(model)
    event_number=model.confirmed_count+1
    payload=json.dumps({
        "event":event_number,"action":chosen,"model_sha":model_id,
        "public_trial_key":trial_public_key,"public_step":step,
        "state_before":state,"policy":policy,
    },sort_keys=True,separators=(",",":")).encode()
    return Proposal(event_number,chosen,policy,state,model_id,
                    trial_public_key,step,hashlib.sha256(payload).hexdigest())


def confirmed_write(model,proposal,*,executed_action,state_before,
                    observed_effect,confirmed,receipt_ref):
    """Read-only data step applied to pure RLS, NOT external source admission.

    This method cannot contact a simulator. A caller must independently
    obtain an authorized, factual, exact-action confirmation receipt.
    """
    frozen()
    _check_model(model)
    if not isinstance(proposal,Proposal):
        raise ValueError("P106_UNCOMMITTED_OUTCOME")
    if (proposal.event_number!=model.confirmed_count+1 or
        proposal.model_state_sha256!=model_sha(model) or
        _four(state_before)!=proposal.state_before):
        raise ValueError("P106_UNKNOWN_OR_REPLAYED_PROPOSAL")
    if (proposal.policy not in POLICIES or
        type(proposal.action) is not int or proposal.action not in (0,1) or
        not isinstance(proposal.opaque_trial_key,str) or
        not 1<=len(proposal.opaque_trial_key)<=128 or
        type(proposal.public_step) is not int or proposal.public_step<0):
        raise ValueError("P106_PROPOSAL_SCOPE_DRIFT")
    payload=json.dumps({
        "event":proposal.event_number,"action":proposal.action,
        "model_sha":proposal.model_state_sha256,
        "public_trial_key":proposal.opaque_trial_key,
        "public_step":proposal.public_step,
        "state_before":proposal.state_before,"policy":proposal.policy,
    },sort_keys=True,separators=(",",":")).encode()
    if hashlib.sha256(payload).hexdigest()!=proposal.commitment_sha256:
        raise ValueError("P106_TAMPERED_PROPOSAL_COMMITMENT")
    if (type(executed_action) is not int or executed_action!=proposal.action
        or executed_action not in (0,1)):
        raise ValueError("P106_WRONG_ACTION_EFFECT_CREDIT")
    if confirmed is not True:
        raise PermissionError("P106_UNCONFIRMED_EFFECT_DENIED")
    if (not isinstance(receipt_ref,str) or not receipt_ref.strip()
        or not receipt_ref.startswith("fixture:")):
        # Current module is a pure fixture-only ENGINEERING preflight;
        # do not turn a GitHub/owner receipt into executable F6 admission.
        raise PermissionError("P106_ONLY_STATIC_TEST_FIXTURE_RECEIPTS")
    if (not isinstance(observed_effect,torch.Tensor)
        or observed_effect.shape!=(D,) or observed_effect.dtype!=torch.float64
        or not bool(torch.isfinite(observed_effect).all())):
        raise ValueError("P106_BAD_OBSERVED_EFFECT")
    action=public_action_key(proposal.state_before,proposal.action)
    g=model.precision@action
    denom=1.+action@g
    if not bool(torch.isfinite(denom)) or float(denom)<=0.:
        raise ValueError("P106_RLS_INVALID_DENOM")
    residual=observed_effect-(model.weights@action)
    new_weights=model.weights+torch.outer(residual,g)/denom
    new_precision=model.precision-torch.outer(g,g)/denom
    audit=hashlib.sha256(
        (model.journal_sha256+"|"+proposal.commitment_sha256+"|"+receipt_ref).encode()
        +observed_effect.contiguous().numpy().tobytes()).hexdigest()
    next_state=RLS(new_weights,new_precision,model.confirmed_count+1,audit)
    _check_model(next_state)
    return next_state


def snapshot(model):
    _check_model(model)
    return RLS(model.weights.clone(),model.precision.clone(),
               model.confirmed_count,model.journal_sha256)


def read(model,state_before,action):
    _check_model(model)
    return model.weights @ public_action_key(state_before,action)
