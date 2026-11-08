"""Narrow reusable P6 evidence estimator for C7 DEV checkpoint integration.

Behavior copied from FactualSlopeEvidence in P6 PR #60 at exact source blob
7f21b73d6255f2611512969e87446352c1dee90c. No hidden-context label,
sequence index, future outcome, trainer, or optimizer is used here. P6 remains
independent; this file has no imports of that PR's separate P4/P5 experiments.
"""
from __future__ import annotations

import torch

SLOPE_SIGN_FLOOR = .2


class FactualSlopeEvidence:
    """Infers signed effect ONLY from *already observed* opposite actions.

    With a single action result, a free intercept is not identifiable; no
    sign update is permitted until both action polarities were observed.
    Fixed threshold is engineering-only and frozen before execution.
    """
    def __init__(self):
        self._last: dict[int, float] = {}
        self._choice = 1  # explicit initial tie-break, not a learned context label
        self._observations = 0
        self._change_events = []

    @property
    def choice(self) -> int:
        return self._choice

    @property
    def observations(self) -> int:
        return self._observations

    def observe(self, action: torch.Tensor, observed: torch.Tensor) -> None:
        if not isinstance(action, torch.Tensor) or not isinstance(observed, torch.Tensor):
            raise ValueError("P6_REALIZED_TENSOR_REQUIRED")
        if action.shape != (1, 1) or observed.shape != (1, 1) or (
            not bool(torch.isfinite(action).all() and torch.isfinite(observed).all())
            or abs(abs(float(action.item())) - 1.) > 1e-6
        ):
            raise ValueError("P6_REALIZED_PAIR_INVALID")
        a = 1 if action.item() > 0 else -1
        self._last[a] = float(observed.item())
        self._observations += 1
        if len(self._last) == 2:
            # y(+1)-y(-1) cancels common sensor/intercept; no latent labels.
            slope = .5 * (self._last[1] - self._last[-1])
            if abs(slope) >= SLOPE_SIGN_FLOOR:
                candidate = 1 if slope > 0 else -1
                if candidate != self._choice:
                    self._change_events.append(self._observations)
                self._choice = candidate
