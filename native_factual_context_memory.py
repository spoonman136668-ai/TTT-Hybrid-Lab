"""P39 DEV: bounded token-free context-indexed factual consequence memory.

A narrow engineering candidate that adds episodic, action-specific residual recall
around the existing NativeContinuousHybrid. It is NOT evidence of learned transfer,
model superiority, goal learning, or an authenticated external science run.

Only the caller can authenticate that an action was actually executed and that
observed_next came from that action. Preview performs no state/parameter writes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from native_continuous_state import ContinuousState, NativeContinuousHybrid
from native_intervention_choice import preview_interventions, MAX_CANDIDATES


@dataclass(frozen=True)
class FactualContextState:
    native: ContinuousState
    contexts: torch.Tensor  # [batch, slots, sensor_dim + action_dim]
    residuals: torch.Tensor  # [batch, slots, sensor_dim]
    occupied: torch.Tensor  # [batch, slots], bool
    next_slot: torch.Tensor  # [batch], int64 bounded circular index
    observations: torch.Tensor  # [batch], int64 factual-only counter


class FactualContextMemory:
    """Bounded continuous factual cache; fixed key features, not learned embeddings.

    `radius` limits how far evidence is reused in continuous sensor space.
    `action_tolerance` ensures a different action cannot borrow a factual
    residual. Neighbors are weighted by an isotropic RBF within the radius.
    Overwriting the oldest slot is deterministic and costs bounded memory.
    """

    def __init__(self, model: NativeContinuousHybrid, slots: int = 16,
                 radius: float = 0.30, action_tolerance: float = 1e-6):
        if not isinstance(model, NativeContinuousHybrid):
            raise TypeError('P39_NATIVE_MODEL_REQUIRED')
        if type(slots) is not int or not 1 <= slots <= 512:
            raise ValueError('P39_BOUNDED_SLOTS_REQUIRED')
        if not all(math.isfinite(x) and x > 0 for x in (radius, action_tolerance)):
            raise ValueError('P39_INVALID_RADIUS_OR_ACTION_TOLERANCE')
        self.model = model
        self.slots = slots
        self.radius = float(radius)
        self.action_tolerance = float(action_tolerance)

    def initial_state(self, sensor: torch.Tensor) -> FactualContextState:
        native = self.model.initial_state(sensor)
        batch, dim = sensor.shape
        device = sensor.device
        return FactualContextState(
            native,
            sensor.new_zeros((batch, self.slots, dim + self.model.action_dim)),
            sensor.new_zeros((batch, self.slots, dim)),
            torch.zeros((batch, self.slots), device=device, dtype=torch.bool),
            torch.zeros((batch,), device=device, dtype=torch.int64),
            torch.zeros((batch,), device=device, dtype=torch.int64),
        )

    def _validate(self, sensor: torch.Tensor, state: FactualContextState) -> None:
        self.model._validate_float(sensor, (None, self.model.sensor_dim), 'SENSOR')
        if not isinstance(state, FactualContextState):
            raise ValueError('P39_FACTUAL_STATE_REQUIRED')
        self.model._validate_state(state.native, sensor)
        b = sensor.shape[0]
        expected = ((b, self.slots, self.model.sensor_dim + self.model.action_dim),
                    (b, self.slots, self.model.sensor_dim),
                    (b, self.slots), (b,), (b,))
        values = (state.contexts, state.residuals, state.occupied,
                  state.next_slot, state.observations)
        if any(not isinstance(v, torch.Tensor) or tuple(v.shape) != shape
               or v.device != sensor.device for v, shape in zip(values, expected)):
            raise ValueError('P39_STATE_SHAPE_OR_DEVICE_DRIFT')
        if (state.contexts.dtype != sensor.dtype or state.residuals.dtype != sensor.dtype
            or state.occupied.dtype != torch.bool
            or state.next_slot.dtype != torch.int64
            or state.observations.dtype != torch.int64):
            raise ValueError('P39_STATE_DTYPE_DRIFT')
        if (not bool(torch.isfinite(state.contexts).all())
            or not bool(torch.isfinite(state.residuals).all())
            or not bool(((state.next_slot >= 0) & (state.next_slot < self.slots)).all())
            or not bool((state.observations >= 0).all())
            or not bool((state.occupied.sum(1) <= state.observations).all())):
            raise ValueError('P39_INVALID_FACTUAL_STATE_VALUES')

    def _correction(self, sensor: torch.Tensor, candidates: torch.Tensor,
                    state: FactualContextState) -> torch.Tensor:
        # Memory is keyed on the *actual* action as well as raw analog context.
        stored_sensor = state.contexts[..., :self.model.sensor_dim]
        stored_action = state.contexts[..., self.model.sensor_dim:]
        sensor_distance_sq = (sensor[:, None, None, :] - stored_sensor[:, None, :, :]).square().sum(-1)
        action_distance = (candidates[:, :, None, :] - stored_action[:, None, :, :]).abs().amax(-1)
        accepted = (sensor_distance_sq <= self.radius ** 2) & (action_distance <= self.action_tolerance)
        accepted = accepted & state.occupied[:, None, :]
        weights = torch.exp(-sensor_distance_sq / (2 * self.radius ** 2)) * accepted
        total = weights.sum(-1, keepdim=True)
        return (weights.unsqueeze(-1) * state.residuals[:, None, :, :]).sum(-2) / total.clamp_min(1e-12)

    def preview(self, sensor: torch.Tensor, candidates: torch.Tensor,
                dt: torch.Tensor, state: FactualContextState) -> torch.Tensor:
        """Same prior for every candidate; returns predictions, NEVER memory writes."""
        self._validate(sensor, state)
        base = preview_interventions(self.model, sensor, candidates, dt, state.native)
        return base + self._correction(sensor, candidates, state)

    def choose(self, sensor: torch.Tensor, candidates: torch.Tensor,
               dt: torch.Tensor, explicit_target: torch.Tensor,
               state: FactualContextState) -> tuple[torch.Tensor, torch.Tensor]:
        """Caller-supplied goal only; no learned objective or external actuation."""
        predicted = self.preview(sensor, candidates, dt, state)
        self.model._validate_float(explicit_target, tuple(sensor.shape), 'EXPLICIT_GOAL')
        if explicit_target.device != sensor.device or explicit_target.dtype != sensor.dtype:
            raise ValueError('P39_GOAL_DEVICE_DTYPE_DRIFT')
        costs = (predicted - explicit_target[:, None, :]).square().sum(-1)
        if not bool(torch.isfinite(costs).all()):
            raise ValueError('P39_NONFINITE_CHOICE_COST')
        return costs.argmin(-1), costs

    @torch.no_grad()
    def commit_factual(self, sensor: torch.Tensor, action_actually_executed: torch.Tensor,
                       dt: torch.Tensor, observed_next_from_source: torch.Tensor,
                       state: FactualContextState) -> tuple[torch.Tensor, FactualContextState]:
        """Advance exactly ONE factual action after observed outcome supplied.

        No hypothetical branch can be committed through preview/choose. This API
        itself cannot authenticate an external outcome: caller/owner must do so.
        """
        self._validate(sensor, state)
        self.model._validate_float(observed_next_from_source,
                                   tuple(sensor.shape), 'OBSERVED_OUTCOME')
        if (observed_next_from_source.device != sensor.device
            or observed_next_from_source.dtype != sensor.dtype):
            raise ValueError('P39_OUTCOME_DEVICE_DTYPE_DRIFT')
        baseline, native = self.model.step(sensor, action_actually_executed, dt, state.native)
        key = torch.cat((sensor, action_actually_executed), dim=-1)
        residual = observed_next_from_source - baseline
        contexts = state.contexts.clone()
        residuals = state.residuals.clone()
        occupied = state.occupied.clone()
        rows = torch.arange(sensor.shape[0], device=sensor.device)
        columns = state.next_slot
        contexts[rows, columns] = key
        residuals[rows, columns] = residual
        occupied[rows, columns] = True
        result = FactualContextState(native, contexts, residuals, occupied,
                                     (columns + 1) % self.slots,
                                     state.observations + 1)
        return baseline, result
