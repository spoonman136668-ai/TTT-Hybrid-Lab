"""P40 DEV: online token-free belief over factual continuous action effects.

Learns a small bank of dynamical effect prototypes exclusively from observed
(s, executed_action, dt, observed_next) tuples. Hidden mode IDs are NEVER
supplied. This is an engineering mechanism, NOT learned general reasoning,
external-source authenticity or an independently qualified scientific gain.
"""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class HiddenModeBelief:
    effects: torch.Tensor          # [batch, capacity, sensor_dim]
    observations: torch.Tensor     # [batch, capacity], int64
    probabilities: torch.Tensor    # [batch, capacity], nonnegative, sums to 1 if known
    factual_steps: torch.Tensor    # [batch], int64


class FactualModeInference:
    """Action-normalized continuous effect prototypes with posterior switching.

    Model assumption: locally x_next - x = dt * action_scalar * effect(mode).
    This restriction is explicit: tests do not claim nonlinear causal transfer.
    Previously unknown effects are admitted only with unused capacity; a
    third incompatible effect fails closed rather than corrupting prototypes.
    """

    def __init__(self, sensor_dim: int, capacity: int = 2,
                 novelty: float = 0.15, noise: float = 0.05,
                 stay_probability: float = 0.8, learning_rate: float = 0.25):
        if type(sensor_dim) is not int or sensor_dim < 1 or type(capacity) is not int or not 2 <= capacity <= 16:
            raise ValueError('P40_INVALID_DIMENSIONS_OR_CAPACITY')
        if not all(math.isfinite(v) for v in (novelty,noise,stay_probability,learning_rate)) or not (
            novelty > 0 and noise > 0 and 0 < stay_probability < 1 and 0 < learning_rate <= 1):
            raise ValueError('P40_INVALID_BELIEF_HYPERPARAMETERS')
        self.sensor_dim = sensor_dim
        self.capacity = capacity
        self.novelty = novelty
        self.noise = noise
        self.stay_probability = stay_probability
        self.learning_rate = learning_rate

    def _sensor(self, x: torch.Tensor, batch: int | None = None, name: str = 'SENSOR') -> None:
        if (not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != self.sensor_dim
            or x.shape[0] < 1 or (batch is not None and x.shape[0] != batch)
            or x.dtype not in (torch.float32,torch.float64) or not bool(torch.isfinite(x).all())):
            raise ValueError('P40_INVALID_' + name)

    def initial_state(self, sensor: torch.Tensor) -> HiddenModeBelief:
        self._sensor(sensor)
        b = sensor.shape[0]
        return HiddenModeBelief(sensor.new_zeros((b,self.capacity,self.sensor_dim)),
                                torch.zeros((b,self.capacity),dtype=torch.int64,device=sensor.device),
                                sensor.new_zeros((b,self.capacity)),
                                torch.zeros((b,),dtype=torch.int64,device=sensor.device))

    def _check(self, sensor: torch.Tensor, state: HiddenModeBelief) -> None:
        self._sensor(sensor)
        if not isinstance(state,HiddenModeBelief):
            raise ValueError('P40_BELIEF_STATE_REQUIRED')
        b= sensor.shape[0]
        if (state.effects.shape != (b,self.capacity,self.sensor_dim)
            or state.observations.shape != (b,self.capacity)
            or state.probabilities.shape != (b,self.capacity)
            or state.factual_steps.shape != (b,)):
            raise ValueError('P40_BELIEF_SHAPE_DRIFT')
        if (any(x.device != sensor.device for x in (state.effects,state.observations,state.probabilities,state.factual_steps))
            or state.effects.dtype != sensor.dtype or state.probabilities.dtype != sensor.dtype
            or state.observations.dtype != torch.int64 or state.factual_steps.dtype != torch.int64):
            raise ValueError('P40_BELIEF_DEVICE_DTYPE_DRIFT')
        if (not bool(torch.isfinite(state.effects).all()) or not bool(torch.isfinite(state.probabilities).all())
            or not bool((state.observations>=0).all()) or not bool((state.factual_steps>=0).all())
            or not bool((state.probabilities>=0).all())
            or not bool((state.probabilities[state.observations==0]==0).all())
            or not bool((state.observations.sum(1)==state.factual_steps).all())):
            raise ValueError('P40_INVALID_BELIEF_STATE')
        summed=state.probabilities.sum(1)
        known=state.factual_steps>0
        if not bool(torch.allclose(summed[known],torch.ones_like(summed[known]),atol=1e-6,rtol=0)) or not bool((summed[~known]==0).all()):
            raise ValueError('P40_INVALID_BELIEF_NORMALIZATION')

    def _action_time(self, sensor: torch.Tensor, action: torch.Tensor, dt: torch.Tensor,
                     *, candidates: bool) -> None:
        a_shape=(sensor.shape[0],None,1) if candidates else (sensor.shape[0],1)
        if (not isinstance(action,torch.Tensor) or action.ndim != len(a_shape)
            or any(y is not None and x != y for x,y in zip(action.shape,a_shape))
            or (candidates and not 1 <= action.shape[1] <= 16)
            or action.device != sensor.device or action.dtype != sensor.dtype
            or not bool(torch.isfinite(action).all())
            or not bool((action.abs()>1e-8).all())):
            raise ValueError('P40_INVALID_ACTION')
        if (not isinstance(dt,torch.Tensor) or dt.shape != (sensor.shape[0],1)
            or dt.device != sensor.device or dt.dtype != sensor.dtype
            or not bool(torch.isfinite(dt).all()) or not bool((dt>0).all())):
            raise ValueError('P40_INVALID_ELAPSED_TIME')

    def preview(self, sensor: torch.Tensor, candidates: torch.Tensor, dt: torch.Tensor,
                state: HiddenModeBelief) -> torch.Tensor:
        """Read-only same-state counterfactual predictions, never an outcome write."""
        self._check(sensor,state)
        self._action_time(sensor,candidates,dt,candidates=True)
        expected=(state.probabilities.unsqueeze(-1)*state.effects).sum(1)
        return sensor.unsqueeze(1) + (dt.unsqueeze(1)*candidates)*expected.unsqueeze(1)

    def choose(self, sensor: torch.Tensor, candidates: torch.Tensor, dt: torch.Tensor,
               goal: torch.Tensor, state: HiddenModeBelief) -> torch.Tensor:
        predicted=self.preview(sensor,candidates,dt,state)
        self._sensor(goal,sensor.shape[0],'GOAL')
        if goal.device!=sensor.device or goal.dtype!=sensor.dtype:
            raise ValueError('P40_GOAL_DEVICE_DTYPE_DRIFT')
        return ((predicted-goal.unsqueeze(1)).square().sum(-1)).argmin(-1)

    @torch.no_grad()
    def observe(self, sensor: torch.Tensor, executed_action: torch.Tensor,
                dt: torch.Tensor, observed_next: torch.Tensor,
                state: HiddenModeBelief) -> HiddenModeBelief:
        """One factual update after caller has authenticated the real consequence.

        No source owner authentication is possible inside this module.
        A failed batch leaves the old immutable state unmodified.
        """
        self._check(sensor,state)
        self._action_time(sensor,executed_action,dt,candidates=False)
        self._sensor(observed_next,sensor.shape[0],'OBSERVED_NEXT')
        if observed_next.device!=sensor.device or observed_next.dtype!=sensor.dtype:
            raise ValueError('P40_OUTCOME_DEVICE_DTYPE_DRIFT')
        effect=(observed_next-sensor)/(dt*executed_action)
        effects=state.effects.clone()
        counts=state.observations.clone()
        probabilities=state.probabilities.clone()
        for b in range(sensor.shape[0]):
            valid=counts[b]>0
            if not bool(valid.any()):
                slot=0
                effects[b,slot]=effect[b]
                counts[b,slot]=1
                probabilities[b].zero_();probabilities[b,slot]=1
                continue
            distance=(effects[b]-effect[b]).square().sum(-1).sqrt()
            distance[~valid]=float('inf')
            slot=int(distance.argmin())
            if float(distance[slot]) > self.novelty:
                free=(~valid).nonzero().flatten()
                if not len(free):
                    raise ValueError('P40_UNMODELED_DYNAMICS_CAPACITY')
                slot=int(free[0]);effects[b,slot]=effect[b];counts[b,slot]=1
            else:
                effects[b,slot]+=self.learning_rate*(effect[b]-effects[b,slot])
                counts[b,slot]+=1
            used=counts[b]>0
            k=int(used.sum())
            if k==1:
                probabilities[b].zero_();probabilities[b,slot]=1
            else:
                # Predictive mode persistence with nonzero switching probability,
                # updated by factual action consequences (not hidden mode labels).
                prior=(1-self.stay_probability)*(1-probabilities[b])/(k-1) + self.stay_probability*probabilities[b]
                prior=torch.where(used,prior,torch.zeros_like(prior))
                # New slots get nonzero prior for a genuine novel observation.
                prior=torch.where((used)&(prior==0),torch.full_like(prior,1e-8),prior)
                logp=torch.log(prior.clamp_min(1e-12)) - (effects[b]-effect[b]).square().sum(-1)/(2*self.noise**2)
                logp[~used]=-float('inf')
                probabilities[b]=torch.softmax(logp,dim=0)
        result=HiddenModeBelief(effects,counts,probabilities,state.factual_steps+1)
        self._check(sensor,result)
        return result
