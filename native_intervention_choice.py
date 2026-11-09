"""Token-free native Hybrid candidate-action preview (DEV only).

Every candidate starts from the same factual continuous state. Hypothetical
predictions do not observe outcomes, update fast memory, or advance the caller's
recurrent state. Goal selection uses a CALLER-SUPPLIED continuous target; the
Hybrid has not learned the objective or a long-horizon planner here.

The owner of an environment must authenticate the separately supplied actual
outcome before calling commit_factual_outcome. No owner, model, or science
first-use authority is conferred by this module.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
from native_continuous_state import ContinuousState, NativeContinuousHybrid

MAX_CANDIDATES = 16


@dataclass(frozen=True)
class GoalActionChoice:
    predictions: torch.Tensor  # [batch, candidates, continuous sensor dimensions]
    squared_target_errors: torch.Tensor  # [batch, candidates]
    chosen_indices: torch.Tensor  # [batch], indexes caller's supplied candidate list
    chosen_actions: torch.Tensor  # [batch, continuous action dimensions]


def _validate_common(model: NativeContinuousHybrid, sensor: torch.Tensor,
                     candidates: torch.Tensor, dt: torch.Tensor,
                     state: ContinuousState | None) -> ContinuousState:
    if not isinstance(model, NativeContinuousHybrid):
        raise ValueError('NATIVE_INTERVENTION_MODEL_REQUIRED')
    model._validate_float(sensor, (None, model.sensor_dim), 'SENSOR')
    model._validate_float(candidates, (sensor.shape[0], None, model.action_dim),
                          'CANDIDATE_ACTIONS')
    if not 1 <= candidates.shape[1] <= MAX_CANDIDATES:
        raise ValueError('NATIVE_INVALID_CANDIDATE_COUNT')
    model._validate_float(dt, (sensor.shape[0], 1), 'ELAPSED_TIME')
    if not bool(torch.all(dt > 0)):
        raise ValueError('NATIVE_NONPOSITIVE_ELAPSED_TIME')
    if (sensor.device != candidates.device or sensor.device != dt.device
        or sensor.dtype != candidates.dtype or sensor.dtype != dt.dtype):
        raise ValueError('NATIVE_INTERVENTION_DEVICE_OR_DTYPE_DRIFT')
    if state is None:
        state = model.initial_state(sensor)
    model._validate_state(state, sensor)
    return state


def preview_interventions(model: NativeContinuousHybrid, sensor: torch.Tensor,
                          candidates: torch.Tensor, dt: torch.Tensor,
                          state: ContinuousState | None = None) -> torch.Tensor:
    """Predict all candidates, discard all speculative states; never learn.

    Shape: [batch, candidate count, sensor_dim]. No parameter, fast-weight,
    or recurrent-memory update. Gradients remain available for future training
    experiments, but this function performs no optimizer step.
    """
    state = _validate_common(model, sensor, candidates, dt, state)
    batch, count, action_dim = candidates.shape
    repeated = ContinuousState(
        state.recurrent.repeat_interleave(count, dim=0),
        state.fast_weights.repeat_interleave(count, dim=0),
        state.history.repeat_interleave(count, dim=0),
    )
    flat_sensor = sensor.repeat_interleave(count, dim=0)
    flat_dt = dt.repeat_interleave(count, dim=0)
    flat_action = candidates.reshape(batch * count, action_dim)
    predictions, _discarded_speculative_states = model.step(
        flat_sensor, flat_action, flat_dt, repeated)
    return predictions.reshape(batch, count, model.sensor_dim)


def select_goal_action(model: NativeContinuousHybrid, sensor: torch.Tensor,
                       candidates: torch.Tensor, dt: torch.Tensor,
                       explicit_target: torch.Tensor,
                       state: ContinuousState | None = None) -> GoalActionChoice:
    """Choose next action minimizing distance to caller-supplied native goal.

    The supplied objective is NOT learned. Exact equal costs break ties toward
    the first supplied candidate; that candidate order is recorded by caller.
    """
    predictions = preview_interventions(model, sensor, candidates, dt, state)
    model._validate_float(explicit_target, tuple(sensor.shape), 'EXPLICIT_GOAL')
    if (explicit_target.device != sensor.device or
        explicit_target.dtype != sensor.dtype):
        raise ValueError('NATIVE_EXPLICIT_GOAL_DEVICE_OR_DTYPE_DRIFT')
    errors = (predictions - explicit_target.unsqueeze(1)).square().sum(dim=-1)
    if not bool(torch.isfinite(errors).all()):
        raise ValueError('NATIVE_NONFINITE_GOAL_SCORE')
    chosen = errors.argmin(dim=1)
    actual_actions = candidates[torch.arange(sensor.shape[0], device=sensor.device), chosen]
    return GoalActionChoice(predictions, errors, chosen, actual_actions)


def commit_factual_outcome(model: NativeContinuousHybrid, sensor: torch.Tensor,
                           action_actually_executed: torch.Tensor, dt: torch.Tensor,
                           observed_next_from_source: torch.Tensor,
                           state: ContinuousState | None = None
                           ) -> tuple[torch.Tensor, ContinuousState]:
    """Write ONE source-verified actual outcome, never a hypothetical branch.

    The caller, not this function, is responsible for verifying that the action
    was physically executed and the next observation is authentic. The state
    returned by preview_interventions is intentionally never available here.
    """
    prediction, actual_provisional = model.step(
        sensor, action_actually_executed, dt, state)
    learned = model.learn_outcome(actual_provisional, action_actually_executed,
                                  prediction, observed_next_from_source)
    return prediction, learned
