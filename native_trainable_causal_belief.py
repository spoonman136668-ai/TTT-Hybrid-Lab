"""P43 DEV: small trainable token-free causal-state interface.

A GRUCell learns an episode state from *completed* analog action/outcome facts.
Two prediction heads structurally separate shared exogenous changes from
candidate-action effects. This separation does not prove causal identification:
observational correlations, nonlinear nuisances, insufficient interventions or
unseen dynamics can defeat it. No hidden mode labels or oracle action outcomes.

Neither this module nor tests authenticate an external action/outcome witness.
Owner-controlled science first use and exact-source acceptance remain separate.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class NativeCausalBeliefState:
    hidden: torch.Tensor  # [batch, width] learned factual history only
    observations: torch.Tensor  # [batch] int64, count of completed actions
    elapsed: torch.Tensor  # [batch,1] cumulative observed physical time


@dataclass(frozen=True)
class NativeCausalPreview:
    predictions: torch.Tensor  # [batch,candidate_count,sensor_dim]
    factual_ready: torch.Tensor  # [batch] bool


class NativeTrainableCausalBelief(nn.Module):
    """Trainable recurrent state on continuous sensor/action/outcome observations.

    Trained end-to-end using *factual* next-observation targets and optimizer
    steps that are explicitly controlled by a caller. Recurrent state is not
    silently updated by candidate predictions. No vocabulary or token path.
    """

    def __init__(self, sensor_dim: int, width: int = 24, min_facts: int = 2):
        super().__init__()
        if (type(sensor_dim) is not int or sensor_dim < 1
            or type(width) is not int or not 4 <= width <= 256
            or type(min_facts) is not int or not 1 <= min_facts <= 64):
            raise ValueError('P43_INVALID_DIMENSIONS')
        self.sensor_dim = sensor_dim
        self.width = width
        self.min_facts = min_facts
        # Both current sensor and realized delta are factual, NOT future branches.
        self.memory_update = nn.GRUCell(2 * sensor_dim + 2, width)
        self.common = nn.Sequential(nn.Linear(width + sensor_dim + 1, width), nn.Tanh())
        self.exogenous_head = nn.Linear(width, sensor_dim)
        self.action_effect_head = nn.Linear(width, sensor_dim)

    def _analog(self, t: torch.Tensor, expected: tuple, name: str) -> None:
        if (not isinstance(t, torch.Tensor) or t.ndim != len(expected)
            or any(size is not None and t.shape[i] != size
                   for i,size in enumerate(expected))
            or t.dtype not in (torch.float32, torch.float64)
            or t.shape[0] < 1 or not bool(torch.isfinite(t).all())):
            raise ValueError('P43_INVALID_' + name)

    def _on_model(self, *values: torch.Tensor) -> None:
        parameter = next(self.parameters())
        if any(x.device != parameter.device or x.dtype != parameter.dtype
               for x in values):
            raise ValueError('P43_MODEL_DEVICE_DTYPE_DRIFT')

    def initial_state(self, sensors: torch.Tensor) -> NativeCausalBeliefState:
        self._analog(sensors, (None,self.sensor_dim), 'SENSOR')
        self._on_model(sensors)
        batch=sensors.shape[0]
        return NativeCausalBeliefState(
            sensors.new_zeros((batch,self.width)),
            torch.zeros((batch,),dtype=torch.int64,device=sensors.device),
            sensors.new_zeros((batch,1)))

    def _state(self, sensor: torch.Tensor, state: NativeCausalBeliefState) -> None:
        self._analog(sensor,(None,self.sensor_dim),'SENSOR')
        self._on_model(sensor)
        if not isinstance(state,NativeCausalBeliefState):
            raise ValueError('P43_STATE_REQUIRED')
        batch=sensor.shape[0]
        if (state.hidden.shape != (batch,self.width)
            or state.observations.shape != (batch,)
            or state.elapsed.shape != (batch,1)):
            raise ValueError('P43_STATE_SHAPE_DRIFT')
        if (state.hidden.device != sensor.device or state.elapsed.device != sensor.device
            or state.observations.device != sensor.device or state.hidden.dtype != sensor.dtype
            or state.elapsed.dtype != sensor.dtype
            or state.observations.dtype != torch.int64):
            raise ValueError('P43_STATE_DEVICE_DTYPE_DRIFT')
        if (not bool(torch.isfinite(state.hidden).all())
            or not bool(torch.isfinite(state.elapsed).all())
            or not bool((state.observations >= 0).all())
            or not bool((state.elapsed >= 0).all())):
            raise ValueError('P43_STATE_VALUES_DRIFT')

    def _dt(self, sensor, dt):
        self._analog(dt,(sensor.shape[0],1),'ELAPSED_TIME')
        self._on_model(dt)
        if not bool((dt>0).all()):
            raise ValueError('P43_NONPOSITIVE_TIME')

    def _action(self,sensor,action,candidates=False):
        shape=(sensor.shape[0],None,1) if candidates else (sensor.shape[0],1)
        self._analog(action,shape,'CANDIDATE_ACTIONS' if candidates else 'EXECUTED_ACTION')
        self._on_model(action)
        if candidates and not 1 <= action.shape[1] <= 16:
            raise ValueError('P43_INVALID_CANDIDATE_COUNT')
        if not bool((action.abs()>1e-8).all()):
            raise ValueError('P43_ZERO_ACTION_UNIDENTIFIABLE')

    def observe(self, sensor: torch.Tensor, executed_action: torch.Tensor,
                dt: torch.Tensor, factual_next: torch.Tensor,
                state: NativeCausalBeliefState) -> NativeCausalBeliefState:
        """Single factual recurrent update; gradients flow through entire history.

        The caller MUST witness completed action and its actual outcome. A
        hypothetical branch cannot enter via preview or choose.
        """
        self._state(sensor,state)
        self._action(sensor,executed_action)
        self._dt(sensor,dt)
        self._analog(factual_next,tuple(sensor.shape),'FACTUAL_OUTCOME')
        self._on_model(factual_next)
        update=torch.cat((sensor,factual_next-sensor,executed_action,torch.log1p(dt)),dim=-1)
        hidden=self.memory_update(update,state.hidden)
        return NativeCausalBeliefState(hidden,state.observations+1,state.elapsed+dt)

    def preview(self, sensor: torch.Tensor, candidate_actions: torch.Tensor,
                dt: torch.Tensor, state: NativeCausalBeliefState) -> NativeCausalPreview:
        """Pure hypothetical comparison; no state or parameter modification."""
        self._state(sensor,state)
        self._action(sensor,candidate_actions,candidates=True)
        self._dt(sensor,dt)
        features=self.common(torch.cat((state.hidden,sensor,torch.log1p(dt)),dim=-1))
        exogenous=self.exogenous_head(features)
        action_effect=self.action_effect_head(features)
        predicted=sensor.unsqueeze(1)+exogenous.unsqueeze(1)+candidate_actions*action_effect.unsqueeze(1)
        if not bool(torch.isfinite(predicted).all()):
            raise ValueError('P43_NONFINITE_PREDICTION')
        return NativeCausalPreview(predicted,state.observations>=self.min_facts)

    def select(self, sensor: torch.Tensor, candidate_actions: torch.Tensor,
               dt: torch.Tensor, external_goal: torch.Tensor,
               state: NativeCausalBeliefState) -> torch.Tensor:
        """Abstain if insufficient factual observations, never invent a goal."""
        out=self.preview(sensor,candidate_actions,dt,state)
        self._analog(external_goal,tuple(sensor.shape),'EXTERNAL_GOAL')
        self._on_model(external_goal)
        if not bool(out.factual_ready.all()):
            raise ValueError('P43_INSUFFICIENT_FACTUAL_HISTORY')
        return ((out.predictions-external_goal.unsqueeze(1)).square().sum(-1)).argmin(-1)
