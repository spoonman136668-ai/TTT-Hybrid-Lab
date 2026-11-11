"""Isolated, opt-in experiment adapter for the existing HybridPilot TTT switch.

Only previously observed consequences are permitted as gate evidence.
Default is the authentic existing all-write baseline. This adapter does NOT
change model.py, run training, access holdouts or authorize FORGE dispatch.
"""
from __future__ import annotations

import math


def choose_ttt_update(*, previous_observed_error: float,
                      previous_evidence_confidence: float,
                      mode: str = "all-write") -> bool:
    """Make one *sequence-level* update choice before the next model call.

    Inputs must be derived from already observed public action-consequence
    pairs; the caller remains responsible for authenticity and separation
    from future target labels. This is not per-token/per-step TTT gating.
    """
    if type(previous_observed_error) not in (int, float) or (
        type(previous_evidence_confidence) not in (int, float)
    ):
        raise ValueError("HYBRID_GATE_OBSERVATION_TYPE")
    error = float(previous_observed_error)
    confidence = float(previous_evidence_confidence)
    if not math.isfinite(error) or error < 0.0 or (
        not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0
    ):
        raise ValueError("HYBRID_GATE_OBSERVATION_INVALID")
    if mode == "all-write":
        return True
    raise ValueError("HYBRID_GATE_MODE_UNQUALIFIED")


def forward_existing_hybrid(model, inputs, *,
                            previous_observed_error: float,
                            previous_evidence_confidence: float,
                            mode: str = "all-write"):
    """Use existing HybridPilot.forward(x, ttt_update=...) with no mutation."""
    update = choose_ttt_update(
        previous_observed_error=previous_observed_error,
        previous_evidence_confidence=previous_evidence_confidence,
        mode=mode,
    )
    return model.forward(inputs, ttt_update=update)
