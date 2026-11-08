"""G7-S26 quarter-reserved numeric gate, distinct from frozen S25.

One independent hypothesis motivated by the S25 development-only update-
depletion null. It DOES NOT revise S25, read hidden switch time, tune any
threshold or admit scientific seeds. It is a source-only causal policy.

Exactly two of eight real updates are consumed in EACH fixed eight-step
quarter of a 32-step episode, including the final quarter. All four
S26 comparator modes share the same temporal reserve constraint.
"""
from __future__ import annotations

from collections import deque
import random
from math import isfinite

from g7_s25_contradiction_gate import (
    GateDecision, STEPS, MATCHED_UPDATES, MIN_SIGNED_EFFECT,
    MIN_CONTRAST_DISAGREEMENT, SURPRISE_THRESHOLD, CONTRADICTION_WINDOW,
    CONTRADICTION_COUNT,
)

QUARTER_STEPS = 8
QUARTER_BUDGET = 2
QUARTER_ENDS = (7, 15, 23, 31)
RESERVE_MODES = ("periodic", "random", "surprise", "contradiction")
DEV_SEEDS = (3101, 3109, 3119)


class QuarterReserveGate:
    """S25-compatible gate API with non-oracle, block-local update credits."""

    def __init__(self, mode: str, *, engineering_seed: int):
        if mode not in RESERVE_MODES:
            raise ValueError("S26_UNKNOWN_RESERVE_MODE")
        if type(engineering_seed) is not int or engineering_seed < 0:
            raise ValueError("S26_INVALID_GATE_SEED")
        self.mode = mode
        self._rng = random.Random(engineering_seed)
        self._step = 0
        self._used = 0
        self._quarter_used = 0
        self._history = deque(maxlen=CONTRADICTION_WINDOW)

    @property
    def updates_used(self) -> int:
        return self._used

    def observe(
        self, step: int, *, predicted_minus: float,
        predicted_plus: float, observed_minus: float,
        observed_plus: float, primary_residual: float,
    ) -> GateDecision:
        if step != self._step or not 0 <= step < STEPS:
            raise ValueError("S26_NONCAUSAL_OR_OUT_OF_ORDER_STEP")
        data=(predicted_minus,predicted_plus,observed_minus,
              observed_plus,primary_residual)
        if any(type(x) not in (int,float) or not isfinite(x) for x in data):
            raise ValueError("S26_NONFINITE_GATE_OBSERVATION")
        quarter=step // QUARTER_STEPS
        inside=step % QUARTER_STEPS
        if inside == 0:
            if step and self._quarter_used != QUARTER_BUDGET:
                raise ValueError("S26_PREVIOUS_QUARTER_NOT_BALANCED")
            self._quarter_used=0
        predicted_contrast=float(predicted_plus-predicted_minus)
        observed_contrast=float(observed_plus-observed_minus)
        contradiction=(
            predicted_contrast*observed_contrast < 0
            and abs(predicted_contrast) >= MIN_SIGNED_EFFECT
            and abs(observed_contrast) >= MIN_SIGNED_EFFECT
            and abs(predicted_contrast-observed_contrast) >= MIN_CONTRAST_DISAGREEMENT
        )
        self._history.append(contradiction)
        count=sum(self._history)
        remaining=QUARTER_BUDGET-self._quarter_used
        slots=QUARTER_STEPS-inside
        if self.mode=="contradiction":
            trigger=count>=CONTRADICTION_COUNT
        elif self.mode=="surprise":
            trigger=abs(primary_residual)>=SURPRISE_THRESHOLD
        elif self.mode=="periodic":
            trigger=inside in (3,7)
        else:
            trigger=self._rng.random()<remaining/slots
        forced=remaining == slots and remaining > 0
        update=remaining > 0 and (trigger or forced)
        if update:
            self._used += 1
            self._quarter_used += 1
        self._step += 1
        reason=("forced_budget" if forced and not trigger else
                "trigger" if trigger and update else
                "budget_exhausted" if trigger else "not_triggered")
        return GateDecision(step,bool(update),self.mode,bool(trigger),
                            reason,count,self._used)

    def close(self) -> int:
        if self._step != STEPS or self._used != MATCHED_UPDATES or \
           self._quarter_used != QUARTER_BUDGET:
            raise ValueError("S26_UNFINISHED_OR_UNMATCHED_QUARTER_BUDGET")
        return self._used


def inject_quarter_reserve(episode, *, seed: int):
    """Swap only gate instances in a NEW S25 numeric episode.

    Retains the S25 network, fast weights, causal observations, masking and
    episode bookkeeping. Does not modify or monkeypatch S25 files.
    """
    from g7_s25_numeric_ttt import PrequentialTTTEpisode
    if not isinstance(episode,PrequentialTTTEpisode) or \
       episode.steps != 0 or episode.update_masks:
        raise ValueError("S26_FRESH_S25_EPISODE_REQUIRED")
    if seed not in DEV_SEEDS:
        raise ValueError("S26_ENGINEERING_SEEDS_ONLY")
    if not episode.gates or any(
        g.mode != episode.gates[0].mode for g in episode.gates
    ) or episode.gates[0].mode not in RESERVE_MODES:
        raise ValueError("S26_INVALID_OR_MIXED_GATE_POLICY")
    mode=episode.gates[0].mode
    episode.gates=[
        QuarterReserveGate(mode,engineering_seed=seed+101*j)
        for j in range(len(episode.gates))
    ]
    return episode
