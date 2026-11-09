"""P42 DEV: token-free finite-history causal contrast from actual consequences.

A bounded, factual-only regression of continuous sensor change on an intercept,
time trend and *executed* action. This is a zero-trained evidence estimator,
not a general learned world model, hidden-mode oracle, or scientific result.

Assumes a locally affine exogenous drift over the observation window. A quadratic
trend or unannounced change of causal law can invalidate this estimate; this
failure is an explicit engineering negative rather than silently hidden.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class CausalEvidenceState:
    times: torch.Tensor       # [batch, window], factual timestamps
    actions: torch.Tensor     # [batch, window], executed scalar actions
    deltas: torch.Tensor      # [batch, window, sensor_dim], actual outcome minus sensor
    count: torch.Tensor       # [batch], valid trailing history entries
    observations: torch.Tensor  # [batch], total factual events
    coefficients: torch.Tensor  # [batch, 3, sensor_dim]: drift, slope, action effect
    identifiable: torch.Tensor  # [batch], rank-sufficient factual window


class DriftAwareCausalEvidence:
    """Causal action-effect contrast with an explicit local affine-drift assumption.

    One inference consumes only factual (sensor, executed action, time, next sensor)
    events. Read-only previews never advance the state. Rank-deficient histories
    fail closed instead of inventing an effect. No hidden physics/oracle input.
    """

    def __init__(self, sensor_dim: int, window: int = 6, min_samples: int = 4):
        if type(sensor_dim) is not int or sensor_dim <= 0:
            raise ValueError('P42_POSITIVE_SENSOR_DIM_REQUIRED')
        if (type(window) is not int or not 4 <= window <= 64 or
            type(min_samples) is not int or not 4 <= min_samples <= window):
            raise ValueError('P42_BOUNDED_WINDOW_REQUIRED')
        self.sensor_dim = sensor_dim
        self.window = window
        self.min_samples = min_samples

    def _sensor(self, x, b=None, role='SENSOR'):
        if (not isinstance(x, torch.Tensor) or x.ndim != 2
            or x.shape[1] != self.sensor_dim or x.shape[0] < 1
            or (b is not None and x.shape[0] != b)
            or x.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(x).all())):
            raise ValueError('P42_INVALID_' + role)

    def initial_state(self, sensor: torch.Tensor) -> CausalEvidenceState:
        self._sensor(sensor)
        b = sensor.shape[0]
        z = sensor.new_zeros
        integer = lambda shape: torch.zeros(shape, dtype=torch.int64,device=sensor.device)
        return CausalEvidenceState(
            z((b,self.window)), z((b,self.window)),
            z((b,self.window,self.sensor_dim)), integer((b,)), integer((b,)),
            z((b,3,self.sensor_dim)),
            torch.zeros((b,),dtype=torch.bool,device=sensor.device))

    def _state(self, sensor: torch.Tensor, s: CausalEvidenceState):
        self._sensor(sensor)
        if not isinstance(s,CausalEvidenceState):
            raise ValueError('P42_STATE_REQUIRED')
        b= sensor.shape[0]
        sizes=((b,self.window),(b,self.window),(b,self.window,self.sensor_dim),
               (b,),(b,),(b,3,self.sensor_dim),(b,))
        vals=(s.times,s.actions,s.deltas,s.count,s.observations,s.coefficients,s.identifiable)
        if any(not isinstance(v,torch.Tensor) or tuple(v.shape)!=shape or v.device!=sensor.device
               for v,shape in zip(vals,sizes)):
            raise ValueError('P42_STATE_SHAPE_OR_DEVICE_DRIFT')
        if (any(v.dtype!=sensor.dtype for v in (s.times,s.actions,s.deltas,s.coefficients))
            or any(v.dtype!=torch.int64 for v in (s.count,s.observations))
            or s.identifiable.dtype != torch.bool):
            raise ValueError('P42_STATE_DTYPE_DRIFT')
        if (not all(bool(torch.isfinite(v).all()) for v in (s.times,s.actions,s.deltas,s.coefficients))
            or not bool(((s.count>=0)&(s.count<=self.window)).all())
            or not bool((s.observations>=s.count).all())
            or not bool((s.identifiable <= (s.count>=self.min_samples)).all())):
            raise ValueError('P42_INVALID_STATE_VALUES')
        for i in range(b):
            n=int(s.count[i])
            if n>1 and not bool((s.times[i,1:n]>s.times[i,:n-1]).all()):
                raise ValueError('P42_STATE_TIME_ORDER_DRIFT')
            if n and not bool((s.actions[i,:n].abs()>1e-8).all()):
                raise ValueError('P42_STATE_ACTION_ZERO')

    def _action_time(self,sensor,action,timestamp,candidates=False):
        b=sensor.shape[0]
        shape=(b,None,1) if candidates else (b,1)
        if (not isinstance(action,torch.Tensor) or action.ndim!=len(shape)
            or any(y is not None and x!=y for x,y in zip(action.shape,shape))
            or (candidates and not 1<=action.shape[1]<=16)
            or action.dtype!=sensor.dtype or action.device!=sensor.device
            or not bool(torch.isfinite(action).all())
            or not bool((action.abs()>1e-8).all())):
            raise ValueError('P42_INVALID_ACTION')
        if (not isinstance(timestamp,torch.Tensor) or timestamp.shape!=(b,1)
            or timestamp.dtype!=sensor.dtype or timestamp.device!=sensor.device
            or not bool(torch.isfinite(timestamp).all())):
            raise ValueError('P42_INVALID_TIMESTAMP')

    @torch.no_grad()
    def observe(self,sensor: torch.Tensor,executed_action: torch.Tensor,
                factual_timestamp: torch.Tensor,observed_next: torch.Tensor,
                state: CausalEvidenceState) -> CausalEvidenceState:
        """Update on a *completed* factual transition, never a speculative one.

        External source authentication remains the caller's responsibility.
        No source data is replayed automatically; old state is not mutated.
        """
        self._state(sensor,state)
        self._action_time(sensor,executed_action,factual_timestamp)
        self._sensor(observed_next,sensor.shape[0],'OBSERVED_NEXT')
        if observed_next.device!=sensor.device or observed_next.dtype!=sensor.dtype:
            raise ValueError('P42_OUTCOME_DEVICE_DTYPE_DRIFT')
        b=sensor.shape[0]
        times=state.times.clone();actions=state.actions.clone();deltas=state.deltas.clone()
        counts=state.count.clone();coefficients=state.coefficients.clone()
        known=state.identifiable.clone()
        for i in range(b):
            n=int(counts[i]);t=factual_timestamp[i,0]
            if n and not bool(t>times[i,n-1]):
                raise ValueError('P42_NONMONOTONE_FACTUAL_TIMESTAMP')
            if n==self.window:
                times[i,:-1]=times[i,1:].clone()
                actions[i,:-1]=actions[i,1:].clone()
                deltas[i,:-1]=deltas[i,1:].clone()
                n-=1
            times[i,n]=t
            actions[i,n]=executed_action[i,0]
            deltas[i,n]=observed_next[i]-sensor[i]
            n+=1;counts[i]=n
            known[i]=False;coefficients[i].zero_()
            if n<self.min_samples:
                continue
            # Column 1: unknown nuisance intercept. Column 2: nuisance slope.
            # Column 3: factual action effect, identifiable only when rank 3.
            rel=(times[i,:n]-times[i,n-1]).double()
            x=torch.stack((torch.ones_like(rel),rel,actions[i,:n].double()),dim=-1)
            singular=torch.linalg.svdvals(x)
            if not bool(singular[-1] > torch.maximum(singular[0]*1e-6,
                                                       singular.new_tensor(1e-8))):
                continue
            fit=torch.linalg.lstsq(x,deltas[i,:n].double()).solution
            if not bool(torch.isfinite(fit).all()):
                raise ValueError('P42_NONFINITE_FACTUAL_FIT')
            coefficients[i]=fit.to(sensor.dtype)
            known[i]=True
        result=CausalEvidenceState(times,actions,deltas,counts,state.observations+1,
                                   coefficients,known)
        self._state(sensor,result)
        return result

    def preview(self,sensor: torch.Tensor,candidates: torch.Tensor,
                future_timestamp: torch.Tensor,state: CausalEvidenceState) -> torch.Tensor:
        """Read-only conditional predictions; cannot fill unobserved outcomes."""
        self._state(sensor,state)
        self._action_time(sensor,candidates,future_timestamp,candidates=True)
        if not bool(state.identifiable.all()):
            raise ValueError('P42_INSUFFICIENT_FACTUAL_ACTION_IDENTIFIABILITY')
        rows=torch.arange(sensor.shape[0],device=sensor.device)
        last=state.times[rows,state.count-1].unsqueeze(1)
        if not bool((future_timestamp>last).all()):
            raise ValueError('P42_PREVIEW_TIME_NOT_FUTURE')
        elapsed=future_timestamp-last
        drift=state.coefficients[:,0,:]+elapsed*state.coefficients[:,1,:]
        effect=state.coefficients[:,2,:]
        return sensor.unsqueeze(1)+drift.unsqueeze(1)+candidates*effect.unsqueeze(1)

    def choose(self,sensor: torch.Tensor,candidates: torch.Tensor,
               future_timestamp: torch.Tensor,explicit_goal: torch.Tensor,
               state: CausalEvidenceState) -> torch.Tensor:
        predicted=self.preview(sensor,candidates,future_timestamp,state)
        self._sensor(explicit_goal,sensor.shape[0],'EXPLICIT_GOAL')
        if explicit_goal.device!=sensor.device or explicit_goal.dtype!=sensor.dtype:
            raise ValueError('P42_GOAL_DEVICE_DTYPE_DRIFT')
        return ((predicted-explicit_goal.unsqueeze(1)).square().sum(-1)).argmin(-1)
