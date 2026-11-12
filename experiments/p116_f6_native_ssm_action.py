"""P116 original F4 SSM 2-action prediction and action choice (fixture-only).

Read *only* prior executed action effects into the original P69/P85 native
256-state SSM, and compare P80 pre-action public keys for both legal next
actions. Predictions for the two *unexecuted* options are model estimates,
not witnessed counterfactuals. Never import an external simulator.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from experiments import p106_f6_source_neutral_policies as p106
from experiments import p113_f6_normalized_ssm_preflight as p113
from experiments import p114_f6_native_optimizer_smoke as p114
from p69_trainable_competitors import DiagonalContinuousSSM

ROOT=Path(__file__).resolve().parents[1]
FREEZE_BLOB="099c2ce38bb0115e8ac9851134a7ea8c61aede39"
FREEZE_COMMIT="9ed98407bca1a1545b5c96c5a9291cccfed38727"
PARENT="2de9ac5546398c50a63f78a6841c1c1158dfb20d"
MAX_HISTORY=95
STATE_SIZE=256
PINNED={
    "experiments/p85mr_dynamics_baselines.py":"aa7f6c34a68ee3118f0fe9e44659aa3dec5524bd",
    "experiments/p69_trainable_competitors.py":"ad0d3712e4eff289ee1ab782820e4c3fc80a95aa",
    "experiments/p80mr_causalgym_trace_adapter.py":"24eb0e050e90466bd42e3626e419fbac08c94c72",
}


def locked():
    raw=(ROOT/"experiments/p116_f6_native_ssm_action_freeze.json").read_bytes()
    if p106._git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("P116_FROZEN_SOURCE_DRIFT")
    f=json.loads(raw)
    if (f.get("schema")!="ttt-hybrid.p116-f6-native-ssm-causal-action-readout.v1"
        or f["parent"]["sha"]!=PARENT
        or f["model"]["prediction_shape"]!=[2,16]
        or f["model"]["history_state_floats"]!=STATE_SIZE
        or f["model"]["fixed_tie_action"]!=0
        or f["input"]["history_steps_max"]!=MAX_HISTORY
        or f["classification"]["original_simulator_steps"]!=0
        or f["classification"]["source_firstuse_authorized"] is not False
        or f["classification"]["protocol_fork_selected"] is not False
        or f["classification"]["science_pass"] is not False):
        raise ValueError("P116_SCOPE_OR_FIRSTUSE_DRIFT")
    for path,digest in PINNED.items():
        if p106._git_blob((ROOT/path).read_bytes())!=digest:
            raise ValueError("P116_ORIGINAL_NATIVE_SOURCE_DRIFT")
    p114.locked()
    return f


def _facts_before(history,state4):
    """Immutable-by-value fixture prefix of selected actions and outcomes."""
    if not isinstance(history,(list,tuple)) or len(history)>MAX_HISTORY:
        raise ValueError("P116_INVALID_FACTUAL_HISTORY_LENGTH")
    curr=p113.state4(state4)
    previous=None
    actions=[]
    effects=[]
    for index,event in enumerate(history,1):
        if not isinstance(event,dict) or set(event)!=p113.STEP_FIELDS:
            raise ValueError("P116_UNCHOSEN_EFFECT_OR_UNKNOWN_STEP_FIELD")
        if (type(event["event_id"]) is not int or event["event_id"]!=index
            or event["mode"]!="do" or
            type(event["executed_action"]) is not int
            or event["executed_action"] not in (0,1)):
            raise ValueError("P116_NOT_A_CONFIRMED_BINARY_ACTION")
        before=p113.state4(event["state_before"])
        after=p113.state4(event["state_after"])
        if previous is not None and before!=previous:
            raise ValueError("P116_BROKEN_FACTUAL_PREFIX")
        previous=after
        if (type(event["terminated"]) is not bool
            or type(event["truncated"]) is not bool):
            raise ValueError("P116_INVALID_TERMINATION")
        if event["terminated"] or event["truncated"]:
            raise PermissionError("P116_AFTER_TERMINAL_NO_NEXT_ACTION")
        p113._real(event["reward"],"P113_BAD_REALIZED_REWARD")
        if event["receipt_ref"]!="fixture:"+p113._sha({
              key:value for key,value in event.items()
              if key!="receipt_ref"}):
            raise ValueError("P116_UNSEALED_FIXTURE_OUTCOME")
        action=event["executed_action"]
        actions.append(p106.action_key(before,action))
        effects.append(p113.label_from_fact(
            before,after,event["reward"],False,False))
    if previous is not None and previous!=curr:
        raise ValueError("P116_NEXT_STATE_DOES_NOT_MATCH_FACTUAL_LAST")
    if not history:
        return None
    return {
        "action":torch.tensor([actions],dtype=torch.float32,device="cpu"),
        "effect":torch.tensor([effects],dtype=torch.float32,device="cpu"),
    }


def _checked_model(model):
    if type(model) is not DiagonalContinuousSSM or model.training:
        raise ValueError("P116_ORIGINAL_F4_SSM_EVAL_REQUIRED")
    params=list(model.parameters())
    if (not params or
        any(param.dtype!=torch.float32 or param.device.type!="cpu"
            or not bool(torch.isfinite(param).all()) for param in params)):
        raise ValueError("P116_CPU_FLOAT32_FINITE_WEIGHTS_REQUIRED")
    return model


def _cost(state,predicted):
    future=[]
    for value,estimate,scale in zip(state,predicted[:4],p106.SCALES):
        safe=max(-.98,min(.98,estimate))
        future.append(value+math.atanh(safe)*scale)
    score=math.fsum(w*(value/scale)**2
        for w,value,scale in zip(p106.GOAL_WEIGHTS,future,p106.SCALES))
    if not math.isfinite(score):
        raise ValueError("P116_NONFINITE_PHYSICAL_COST")
    return score


def infer_two_actions(model,state4,history=()):
    """Forecast both *hypothetical* next actions; return no source effect."""
    locked()
    original=_checked_model(model)
    state=p113.state4(state4)
    facts=_facts_before(history,state)
    keys=torch.tensor(
        [p106.action_key(state,action) for action in (0,1)],
        dtype=torch.float32,device="cpu")
    with torch.no_grad():
        if facts is None:
            carry=torch.zeros((1,STATE_SIZE),dtype=torch.float32)
        else:
            carry=original.encode(facts)
        if carry.shape!=(1,STATE_SIZE) or not bool(torch.isfinite(carry).all()):
            raise ValueError("P116_INVALID_NATIVE_MEMORY")
        output=original.read(carry.expand(2,-1),keys)
    if output.shape!=(2,16) or not bool(torch.isfinite(output).all()):
        raise ValueError("P116_INVALID_NATIVE_FORECAST")
    predictions=output.detach().cpu().tolist()
    penalty=[_cost(state,p) for p in predictions]
    action=min((0,1),key=lambda a:(penalty[a],a))
    return {
        "policy":"ssm256_model_based",
        "action":action,
        "estimated_physical_costs_fixture":penalty,
        "predicted_normalized_effects_fixture":predictions,
        "before_write_factual_prefix_steps":len(history),
        "memory_state_floats":STATE_SIZE,
        "model_checkpoint_sha256_fixture_only":p114._fingerprint(original),
        "unselected_source_effects_observed":0,
        "original_source_effects_executed":0,
        "F6_real_trained_model_qualified":False,
        "F6_first_use_authorized":False,
        "scientific_pass":False,
    }


def refuse_external_source(*_args,**_kwargs):
    raise PermissionError("P116_REAL_F6_FIRSTUSE_NOT_AUTHORIZED")
