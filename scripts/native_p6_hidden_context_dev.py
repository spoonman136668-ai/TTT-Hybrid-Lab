"""P6 engineering: infer hidden action-effect context from prior factual consequences.

The two analytic contexts have IDENTICAL current sensor/time/action distributions.
Only the realized factual next-sensor change distinguishes their causal slopes.
The inferred router has NO phase index, context label, phase schedule, or target
at prediction time. The oracle alone receives latent context as an upper bound.
This is deterministic analytic development, never scientific first-use.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from native_continuous_state import ContinuousState, NativeContinuousHybrid
from scripts.native_prequential_gate import NativeCausalEpisode
from scripts.native_p5_equal_capacity_dev import RoutedEpisode
from scripts.native_p4_interference_dev import MODEL_INITIALIZATIONS

SCHEMA="ttt-hybrid.native-p6.implicit-context-causal-evidence-dev.v1"
WIDTH=12
HALF=6
BLOCK=24
N_BLOCKS=4
EVENTS=BLOCK*N_BLOCKS
INITIALIZATIONS=MODEL_INITIALIZATIONS
ORDERS=(("A","B","A","B"),("B","A","B","A"))
ARMS=("inferred_context_12_slots","oracle_context_12_slots",
      "original_shared_12_slots","action_route_12_slots",
      "constant_sensor_route_12_slots","no_feedback")
FAST_RATE=.25
SLOPE_SIGN_FLOOR=.2
WALL_BUDGET=45
WINDOWS={"initial_late":(16,24),"second_late":(40,48),
         "revisit_1_immediate":(48,52),"revisit_1_late":(64,72),
         "revisit_2_immediate":(72,76),"revisit_2_late":(88,96)}


def fixture_event(index:int,order:tuple[str,...]) -> tuple[str,torch.Tensor,torch.Tensor,torch.Tensor]:
    """Evaluator only; this latent label must never enter inferred.predict."""
    if type(index) is not int or not 0<=index<EVENTS or order not in ORDERS:
        raise ValueError("P6_FIXTURE_EVENT_OUT_OF_RANGE")
    label=order[index//BLOCK]
    sensor=torch.full((1,1),.07,dtype=torch.float32)
    action=torch.tensor([[1. if index%2 else -1.]],dtype=torch.float32)
    dt=torch.full((1,1),.2,dtype=torch.float32)
    return label,sensor,action,dt


def observed_factual(label:str,sensor:torch.Tensor,action:torch.Tensor)->torch.Tensor:
    if label not in ("A","B") or sensor.shape!=(1,1) or action.shape!=(1,1):
        raise ValueError("P6_FIXTURE_INVALID")
    if not bool(torch.isfinite(sensor).all() and torch.isfinite(action).all()):
        raise ValueError("P6_NONFINITE")
    return sensor + .05 + (.45 if label=="A" else -.45)*action


class FactualSlopeEvidence:
    """Infers signed effect ONLY from *already observed* opposite actions.

    With a single action result, a free intercept is not identifiable; no
    sign update is permitted until both action polarities were observed.
    Fixed threshold is engineering-only and frozen before execution.
    """
    def __init__(self):
        self._last:dict[int,float]={}
        self._choice=1  # explicit initial tie-break, not a learned context label
        self._observations=0
        self._change_events=[]

    @property
    def choice(self)->int:
        return self._choice

    @property
    def observations(self)->int:
        return self._observations

    def observe(self,action:torch.Tensor,observed:torch.Tensor)->None:
        if not isinstance(action,torch.Tensor) or not isinstance(observed,torch.Tensor):
            raise ValueError("P6_REALIZED_TENSOR_REQUIRED")
        if action.shape!=(1,1) or observed.shape!=(1,1) or (
            not bool(torch.isfinite(action).all() and torch.isfinite(observed).all())
            or abs(abs(float(action.item()))-1.)>1e-6):
            raise ValueError("P6_REALIZED_PAIR_INVALID")
        a=1 if action.item()>0 else -1
        self._last[a]=float(observed.item())
        self._observations+=1
        if len(self._last)==2:
            # y(+1)-y(-1) cancels common sensor/intercept; no latent labels.
            slope=.5*(self._last[1]-self._last[-1])
            if abs(slope)>=SLOPE_SIGN_FLOOR:
                candidate=1 if slope>0 else -1
                if candidate!=self._choice:
                    self._change_events.append(self._observations)
                self._choice=candidate


def half_key(model:NativeContinuousHybrid,action:torch.Tensor,
             choice:int)->torch.Tensor:
    if choice not in (-1,1):
        raise ValueError("P6_CHOICE_INVALID")
    raw=model.action_encoder(action)
    mask=torch.zeros_like(raw)
    if choice==1:
        mask[:,:HALF]=1
    else:
        mask[:,HALF:]=1
    key=raw*mask
    if not bool((torch.linalg.vector_norm(key,dim=-1)>1e-8).all()):
        raise ValueError("P6_ZERO_KEY")
    return F.normalize(key,dim=-1)


def predict_masked(model:NativeContinuousHybrid,prior:ContinuousState,
                   sensor:torch.Tensor,action:torch.Tensor,dt:torch.Tensor,
                   choice:int):
    zero=ContinuousState(prior.recurrent,torch.zeros_like(prior.fast_weights),
                         prior.history)
    base,advanced=model.step(sensor,action,dt,zero)
    key=half_key(model,action,choice)
    correction=torch.bmm(prior.fast_weights,key.unsqueeze(-1)).squeeze(-1)
    prediction=base+correction
    next_state=ContinuousState(advanced.recurrent,
                               prior.fast_weights.detach().clone(),
                               advanced.history)
    return prediction,next_state,key


class ContextEpisode:
    """Inferred branch has NO label/event parameter; oracle only uses label."""
    def __init__(self,model:NativeContinuousHybrid,*,oracle:bool):
        self.model=model
        self.oracle=oracle
        self.evidence=None if oracle else FactualSlopeEvidence()
        self.state:ContinuousState|None=None
        self.pending:tuple[torch.Tensor,torch.Tensor,torch.Tensor]|None=None
        self.events=0
        self.choices=[]

    def predict_pair(self,sensor:torch.Tensor,action:torch.Tensor,
                     dt:torch.Tensor,*,oracle_label:str|None=None):
        if self.pending is not None:
            raise RuntimeError("P6_PENDING_OBSERVATION")
        if self.oracle:
            if oracle_label not in ("A","B"):
                raise ValueError("P6_ORACLE_REQUIRES_EXPLICIT_LABEL")
            choice=1 if oracle_label=="A" else -1
        else:
            if oracle_label is not None:
                raise ValueError("P6_HIDDEN_LABEL_TO_INFERRED_FORBIDDEN")
            choice=self.evidence.choice
        with torch.no_grad():
            prior=self.state if self.state is not None else self.model.initial_state(sensor)
            pred,next_state,key=predict_masked(
                self.model,prior,sensor,action,dt,choice)
            alternate,_,_=predict_masked(
                self.model,prior,sensor,-action,dt,choice)
            self.state=next_state
            self.pending=(pred.detach().clone(),key.detach().clone(),
                          action.detach().clone())
            self.choices.append(choice)
            self.events+=1
            return pred.detach().clone(),alternate.detach().clone()

    def observe(self,observed:torch.Tensor,*,factual:bool):
        if self.pending is None:
            raise RuntimeError("P6_NO_PENDING_OBSERVATION")
        if factual is not True:
            raise ValueError("P6_COUNTERFACTUAL_FEEDBACK_FORBIDDEN")
        if (not isinstance(observed,torch.Tensor) or observed.shape!=(1,1) or
            observed.dtype!=torch.float32 or
            not bool(torch.isfinite(observed).all())):
            raise ValueError("P6_REALIZED_OUTCOME_INVALID")
        with torch.no_grad():
            pred,key,action=self.pending
            correction=FAST_RATE*(observed-pred).unsqueeze(-1)*key.unsqueeze(1)
            self.state=ContinuousState(self.state.recurrent,
                                       self.state.fast_weights+correction,
                                       self.state.history)
            # Crucial: update inferred context AFTER applying the factual
            # outcome to the bank chosen BEFORE this observation existed.
            if self.evidence is not None:
                self.evidence.observe(action,observed)
            self.pending=None


def _mean(vals:list[float],window:tuple[int,int])->float:
    start,end=window
    return sum(vals[start:end])/(end-start)


def run_initialization(initialization:int,order:tuple[str,...])->dict:
    if type(initialization) is not int or initialization not in INITIALIZATIONS:
        raise ValueError("P6_UNREGISTERED_DEV_INITIALIZATION")
    if order not in ORDERS:
        raise ValueError("P6_UNREGISTERED_DEV_ORDER")
    torch.manual_seed(initialization)
    model=NativeContinuousHybrid(1,1,width=WIDTH,history_limit=4,
                                 fast_rate=FAST_RATE).eval()
    inferred=ContextEpisode(model,oracle=False)
    oracle=ContextEpisode(model,oracle=True)
    shared=NativeCausalEpisode(model)
    negatives={
        ARMS[3]:RoutedEpisode(model,"action_sign"),
        ARMS[4]:RoutedEpisode(model,"sensor_sign"),
    }
    no_state:ContinuousState|None=None
    mse={arm:[] for arm in ARMS}
    effect={arm:[] for arm in ARMS}
    inferred_correct=[]
    matching_recurrence=True
    no_feedback_zero=True
    same_capacity=True
    with torch.no_grad():
        for event in range(EVENTS):
            # The fixture owns the ground-truth label; the inferred episode
            # receives only sensor, factual action, positive elapsed time.
            label,sensor,action,dt=fixture_event(event,order)
            predicted_route=inferred.evidence.choice
            inferred_correct.append(predicted_route==(1 if label=="A" else -1))
            pi,ai=inferred.predict_pair(sensor,action,dt)
            po,ao=oracle.predict_pair(sensor,action,dt,oracle_label=label)
            ticket,as_=shared.predict_pair(sensor,action,-action,dt)
            preds={ARMS[0]:pi,ARMS[1]:po,ARMS[2]:ticket.prediction}
            alternates={ARMS[0]:ai,ARMS[1]:ao,ARMS[2]:as_}
            for arm,ep in negatives.items():
                preds[arm],alternates[arm]=ep.predict_pair(sensor,action,dt)
            prior=no_state if no_state is not None else model.initial_state(sensor)
            pn,next_no=model.step(sensor,action,dt,prior)
            an,_=model.step(sensor,-action,dt,prior)
            preds[ARMS[5]],alternates[ARMS[5]]=pn,an

            outcome=observed_factual(label,sensor,action)
            for arm in ARMS:
                err=float((preds[arm]-outcome).square().item())
                sign_effect=(preds[arm]-alternates[arm]) if action.item()>0 else (
                    alternates[arm]-preds[arm])
                mse[arm].append(err)
                effect[arm].append(float(sign_effect.item()))
                if not math.isfinite(err) or not math.isfinite(effect[arm][-1]):
                    raise RuntimeError("P6_NONFINITE_PREDICTION")
            inferred.observe(outcome,factual=True)
            oracle.observe(outcome,factual=True)
            shared.observe(ticket,outcome,factual=True)
            for ep in negatives.values():
                ep.observe(outcome,factual=True)
            no_state=next_no
            reference=shared.snapshot()
            for state in [inferred.state,oracle.state,*[x.state for x in negatives.values()],no_state]:
                matching_recurrence &= bool(torch.equal(reference.recurrent,state.recurrent)
                    and torch.equal(reference.history,state.history))
                same_capacity &= state.fast_weights.shape==(1,1,WIDTH)
            no_feedback_zero &= bool(torch.count_nonzero(no_state.fast_weights)==0)

    metric={window:{arm:_mean(mse[arm],scope) for arm in ARMS}
            for window,scope in WINDOWS.items()}
    effect_sign={}
    for k in ("revisit_1_immediate","revisit_2_immediate"):
        truth=1 if order[2 if k.startswith("revisit_1") else 3]=="A" else -1
        start,end=WINDOWS[k]
        effect_sign[k]={}
        for arm in ARMS:
            raw=effect[arm][start:end]
            eligible=[v for v in raw if abs(v)>=.05]
            effect_sign[k][arm]={
                "accuracy":sum((x*truth)>0 for x in eligible)/len(eligible)
                           if eligible else 0.0,
                "coverage":len(eligible)/len(raw)}
    inference={
        "prediction_correct_rate_all":sum(inferred_correct)/EVENTS,
        "preobserved_route_at_initial":inferred_correct[0],
        "route_correct_on_first_revisit_1":inferred_correct[48],
        "route_correct_on_first_revisit_2":inferred_correct[72],
        "correct_on_revisit_1_after_two_results":inferred_correct[50],
        "correct_on_revisit_2_after_two_results":inferred_correct[74],
        "change_evidence_counts":list(inferred.evidence._change_events),
        "factual_observations":inferred.evidence.observations,
        "route_never_given_context_label":True}
    comparisons={
        "inferred_beats_shared_on_both_late_revisits":all(
            metric[w][ARMS[0]]<metric[w][ARMS[2]] for w in
            ("revisit_1_late","revisit_2_late")),
        "inferred_beats_action_route_on_both_late_revisits":all(
            metric[w][ARMS[0]]<metric[w][ARMS[3]] for w in
            ("revisit_1_late","revisit_2_late")),
        "inferred_beats_constant_sensor_route_both_late":all(
            metric[w][ARMS[0]]<metric[w][ARMS[4]] for w in
            ("revisit_1_late","revisit_2_late")),
        "oracle_beats_inferred_on_both_immediate_revisits":all(
            metric[w][ARMS[1]]<metric[w][ARMS[0]] for w in
            ("revisit_1_immediate","revisit_2_immediate")),
        "inferred_recovers_after_two_factual_events":(
            inference["correct_on_revisit_1_after_two_results"] and
            inference["correct_on_revisit_2_after_two_results"]),
    }
    checks={
        "all_prequential_events_scored":all(len(mse[a])==EVENTS for a in ARMS),
        "same_native_base_model_weights":True,
        "identical_recurrence_history_all_arms":matching_recurrence,
        "all_arms_exactly_12_fast_storage_scalars":same_capacity,
        "no_feedback_fast_memory_never_updated":no_feedback_zero,
        "no_counterfactual_outcome_learning":True,
        "inferred_outcome_ledger_96_factual":inference["factual_observations"]==EVENTS,
        "inferred_no_oracle_label_or_event_argument":True,
        "positive_elapsed_time_and_fixed_sensor_for_both_contexts":True,
        "no_pending_episode":inferred.pending is None and oracle.pending is None
        and all(x.pending is None for x in negatives.values()),
    }
    return {
        "model_initialization":initialization,
        "block_order":list(order),
        "status":"PASS_DEV_INTEGRITY" if all(checks.values()) else "FAIL_DEV_INTEGRITY",
        "checks":checks,
        "inference":inference,
        "hypotheses":comparisons,
        "phase_mse":metric,
        "revisit_action_effect_sign":effect_sign,
        "oracle_uses_true_latent_label":"UPPER_BOUND_ONLY",
        "current_observation_label_signal":"NONE",
        "memory_budget_scalars":WIDTH,
    }


def run()->dict:
    if os.environ.get("NATIVE_P6_DEV_ONLY","1")!="1":
        raise RuntimeError("P6_SCIENCE_FIRST_USE_FORBIDDEN")
    torch.set_num_threads(1)
    start=time.perf_counter()
    records=[run_initialization(seed,order) for order in ORDERS
             for seed in INITIALIZATIONS]
    elapsed=time.perf_counter()-start
    integrity=(len(records)==16 and all(x["status"]=="PASS_DEV_INTEGRITY"
              for x in records) and elapsed< WALL_BUDGET)
    support={h:sum(bool(x["hypotheses"][h]) for x in records)
             for h in records[0]["hypotheses"]}
    return {
        "schema":SCHEMA,
        "status":"PASS_DEV_INTEGRITY_NOT_SCIENCE" if integrity else
                  "FAIL_DEV_INTEGRITY_NOT_SCIENCE",
        "hypothesis_support_count_of_16":support,
        "records":records,
        "model_initializations":list(INITIALIZATIONS),
        "block_orders":[list(x) for x in ORDERS],
        "total_events_per_record":EVENTS,
        "world_seeds_consumed":[],
        "science_first_use":"NOT_RUN",
        "worlds_constructed":0,
        "optimizer_updates":0,
        "model_weights_frozen":True,
        "memory_scalar_budget_each_arm":WIDTH,
        "limitations":"Analytic fixture, fixed action alternation, initialized plus-route tie-break, fixed observation offset, paired slope requires observed opposite actions, no new-world source, untrained native model, true-label oracle upper bound only; not scientific transfer or matched-compute superiority.",
        "whole_process_wall_seconds":elapsed
    }


def main()->int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--receipt",type=Path,required=True)
    args=parser.parse_args()
    result=run()
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "status":result["status"],
        "hypothesis_support_count_of_16":result["hypothesis_support_count_of_16"],
        "record_count":len(result["records"]),
        "science_first_use":result["science_first_use"],
        "whole_process_wall_seconds":result["whole_process_wall_seconds"],
        "inferred_correct_rate_overall":sum(r["inference"]["prediction_correct_rate_all"]
            for r in result["records"])/len(result["records"])
    },sort_keys=True))
    return 0 if result["status"]=="PASS_DEV_INTEGRITY_NOT_SCIENCE" else 1

if __name__=="__main__":
    raise SystemExit(main())
