"""G7-S25 deterministic, numeric prequential gate policy (ENGINEERING ONLY).

No model, optimizer, experiment, world simulator, heldout samples or training.
Update decisions are made after *only the current* paired observations arrive.
All K=8 policies consume exactly eight update slots; full-update is separate.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
import random

STEPS = 32
MATCHED_UPDATES = 8
MIN_SIGNED_EFFECT = 0.012
MIN_CONTRAST_DISAGREEMENT = 0.018
SURPRISE_THRESHOLD = 0.018
CONTRADICTION_WINDOW = 3
CONTRADICTION_COUNT = 2
MODES = frozenset(("periodic", "random", "surprise", "contradiction", "full"))


@dataclass(frozen=True)
class GateDecision:
    step: int
    update: bool
    mode: str
    trigger: bool
    reason: str
    contradiction_evidence_count: int
    real_updates_used: int


class PrequentialGate:
    """Select fast-weight updates causally, without oracle labels or lookahead."""

    def __init__(self, mode: str, *, engineering_seed: int):
        if mode not in MODES:
            raise ValueError("UNKNOWN_GATE_MODE")
        if not isinstance(engineering_seed, int) or engineering_seed < 0:
            raise ValueError("INVALID_GATE_SEED")
        self.mode = mode
        self._rng = random.Random(engineering_seed)
        self._next_step = 0
        self._real_updates = 0
        self._budget = STEPS if mode == "full" else MATCHED_UPDATES
        self._history: deque[bool] = deque(maxlen=CONTRADICTION_WINDOW)

    @property
    def updates_used(self) -> int:
        return self._real_updates

    def observe(
        self,
        step: int,
        *,
        predicted_minus: float,
        predicted_plus: float,
        observed_minus: float,
        observed_plus: float,
        primary_residual: float,
    ) -> GateDecision:
        """Consume current pre-update predictions and *observed* action pair.

        The API has no hidden world state, family ID, future observation, labels,
        answer keys or dynamics coefficients. Model update itself lives outside
        this policy; selected updates must be counted by the later runner.
        """
        if step != self._next_step or step < 0 or step >= STEPS:
            raise ValueError("NONCAUSAL_OR_OUT_OF_ORDER_GATE_STEP")
        data = (
            predicted_minus, predicted_plus, observed_minus,
            observed_plus, primary_residual,
        )
        if any(not isinstance(n, (int, float)) or not isfinite(n) for n in data):
            raise ValueError("NONFINITE_GATE_OBSERVATION")
        pred_contrast = float(predicted_plus - predicted_minus)
        obs_contrast = float(observed_plus - observed_minus)
        contradictory = (
            pred_contrast * obs_contrast < 0
            and abs(pred_contrast) >= MIN_SIGNED_EFFECT
            and abs(obs_contrast) >= MIN_SIGNED_EFFECT
            and abs(pred_contrast - obs_contrast) >= MIN_CONTRAST_DISAGREEMENT
        )
        self._history.append(contradictory)
        count = sum(self._history)
        if self.mode == "contradiction":
            trigger = count >= CONTRADICTION_COUNT
        elif self.mode == "surprise":
            trigger = abs(primary_residual) >= SURPRISE_THRESHOLD
        elif self.mode == "periodic":
            trigger = step % (STEPS // MATCHED_UPDATES) == (
                STEPS // MATCHED_UPDATES - 1)
        elif self.mode == "full":
            trigger = True
        else:
            remaining_steps = STEPS - step
            remaining_updates = self._budget - self._real_updates
            trigger = (self._rng.random()
                       < (remaining_updates / remaining_steps))
        remaining_steps = STEPS - step
        remaining_updates = self._budget - self._real_updates
        forced = remaining_updates == remaining_steps
        update = remaining_updates > 0 and (trigger or forced)
        if update:
            self._real_updates += 1
        self._next_step += 1
        reason = ("forced_budget" if forced and not trigger else
                  "trigger" if trigger and update else
                  "budget_exhausted" if trigger else "not_triggered")
        return GateDecision(step, bool(update), self.mode, bool(trigger),
                            reason, count, self._real_updates)

    def close(self) -> int:
        if self._next_step != STEPS or self._real_updates != self._budget:
            raise ValueError("UNFINISHED_OR_UNMATCHED_GATE_EPISODE")
        return self._real_updates
