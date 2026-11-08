"""Synthetic-only causal-order boundary for an untrained continuous-state Hybrid.

No world generator, training loop, seeds, network, runner, scheduler, or science
first-use. Enforces *API* event order; cannot authenticate real-world timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from native_continuous_state import ContinuousState, NativeContinuousHybrid


@dataclass(eq=False, frozen=True)
class PredictionTicket:
    """Opaque single-use event capability. The displayed estimate is only a copy."""

    event_index: int
    prediction: torch.Tensor


class NativeCausalEpisode:
    """Read-before-outcome, factual-only updates, one pending event per episode.

    Paired counterfactual estimates are read from the *same prior state*. Only
    the factual transition updates that state. A ticket must be returned to
    consume an observation; changing its exposed tensor cannot alter learning.
    """

    def __init__(self, model: NativeContinuousHybrid):
        if not isinstance(model, NativeContinuousHybrid):
            raise TypeError("NATIVE_MODEL_REQUIRED")
        self._model = model.eval()
        self._state: ContinuousState | None = None
        self._pending: PredictionTicket | None = None
        self._pending_action: torch.Tensor | None = None
        self._pending_prediction: torch.Tensor | None = None
        self._event_index = 0

    @staticmethod
    def _copy_state(state: ContinuousState | None) -> ContinuousState | None:
        if state is None:
            return None
        return ContinuousState(state.recurrent.detach().clone(),
                               state.fast_weights.detach().clone(),
                               state.history.detach().clone())

    def snapshot(self) -> ContinuousState | None:
        """Return a defensive copy; external mutation never alters the episode."""
        return self._copy_state(self._state)

    def _begin(self, sensor: torch.Tensor, action: torch.Tensor,
               dt: torch.Tensor, alternate: torch.Tensor | None):
        if self._pending is not None:
            raise RuntimeError("NATIVE_PENDING_OUTCOME_NOT_OBSERVED")
        # No state changes until both paired estimates have succeeded.
        with torch.no_grad():
            prior = self._copy_state(self._state)
            factual, new_state = self._model.step(sensor, action, dt, prior)
            counterfactual = None
            if alternate is not None:
                counterfactual, _ = self._model.step(
                    sensor, alternate, dt, self._copy_state(self._state))
            self._event_index += 1
            ticket = PredictionTicket(self._event_index, factual.detach().clone())
            self._state = self._copy_state(new_state)
            self._pending = ticket
            self._pending_action = action.detach().clone()
            self._pending_prediction = factual.detach().clone()
            return ticket, None if counterfactual is None else counterfactual.detach().clone()

    def predict(self, sensor: torch.Tensor, action: torch.Tensor,
                dt: torch.Tensor) -> PredictionTicket:
        ticket, _ = self._begin(sensor, action, dt, None)
        return ticket

    def predict_pair(self, sensor: torch.Tensor, factual_action: torch.Tensor,
                     alternate_action: torch.Tensor, dt: torch.Tensor
                     ) -> tuple[PredictionTicket, torch.Tensor]:
        """Score paired potential outcomes without learning the unavailable one."""
        ticket, counterfactual = self._begin(sensor, factual_action, dt, alternate_action)
        assert counterfactual is not None
        return ticket, counterfactual

    def observe(self, ticket: PredictionTicket, observed: torch.Tensor,
                *, factual: bool) -> None:
        if ticket is not self._pending:
            raise RuntimeError("NATIVE_STALE_OR_FOREIGN_PREDICTION")
        if factual is not True:
            raise ValueError("NATIVE_COUNTERFACTUAL_FEEDBACK_FORBIDDEN")
        with torch.no_grad():
            # If input validation fails, ticket stays pending and can be retried
            # with the authentic factual outcome. No partial state mutation.
            updated = self._model.learn_outcome(
                self._state, self._pending_action, self._pending_prediction, observed)
            self._state = self._copy_state(updated)
        self._pending = None
        self._pending_action = None
        self._pending_prediction = None

    def reset(self) -> None:
        if self._pending is not None:
            raise RuntimeError("NATIVE_PENDING_OUTCOME_NOT_OBSERVED")
        self._state = None
        self._event_index += 1
