"""Posthoc analytic checks only. Does not instantiate a simulator or train."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from experiments.p78mr_gain_posthoc_audit import (
    RESULT, RESULT_BLOB, REFERENCE, REFERENCE_BLOB,
    SEEDS, audit, gain, logit, read_locked,
)


def test_original_frozen_evidence_and_equation_are_byte_pinned():
    assert read_locked(RESULT, RESULT_BLOB)
    assert read_locked(REFERENCE, REFERENCE_BLOB)


def test_all_three_saved_parameter_sets_have_inverse_error_slope():
    evidence = audit()
    assert evidence["classification"] == "POSTHOC_EQUATION_FINGERPRINT_ONLY"
    assert tuple(row["seed"] for row in evidence["rows"]) == SEEDS
    assert evidence["scientific_pass"] is False
    assert evidence["source_birth"] is False
    assert evidence["training"] is False
    assert evidence["heldout_scored"] is False
    for row in evidence["rows"]:
        assert -1.5 < row["error_log_coefficient"] < -1.3
        assert 0.3 < row["prediction_log_coefficient"] < 0.4
        low, mid, high = row["illustrative_gains_error_mse_0p01_0p1_1"]
        assert 0 < high < mid < low < 1
        assert row["inverse_error_slope_at_fixed_pred_energy"]


def test_gain_matches_exact_frozen_p77_equation_numerically():
    # Independent analytic reference; no P78 data or learned weights consumed.
    theta = (-1.0, -1.4, 0.34)
    for resid, pred in [(0.01, 0.1), (0.1, 0.1), (1.0, 0.1), (0.0, 0.0)]:
        z = theta[0] + theta[1] * math.log(resid + 1e-6) + theta[2] * math.log(pred + 1e-6)
        expected = 1.0 / (1.0 + math.exp(-z))
        assert gain(theta, resid, pred) == pytest.approx(expected, abs=1e-12)
        assert logit(theta, resid, pred) == pytest.approx(z, abs=1e-12)


def test_extreme_error_features_are_numerically_safe():
    assert 0.0 <= gain((-1.0, -1.4, 0.34), 1e-40, 1.0) <= 1.0
    assert 0.0 <= gain((-1.0, -1.4, 0.34), 1e30, 1.0) <= 1.0


@pytest.mark.parametrize("bad", [
    (float("nan"), 0.1), (-1.0, 0.1),
    (0.1, float("inf")), (0.1, -0.01),
])
def test_invalid_fabricated_feature_values_denied(bad):
    with pytest.raises(ValueError, match="P78MR_POSTHOC_INVALID_FEATURE"):
        gain((-1.0, -1.4, 0.34), *bad)


def test_source_drift_denial(tmp_path):
    altered = tmp_path / "receipt.json"
    altered.write_bytes(RESULT.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="P78MR_POSTHOC_SOURCE_DRIFT"):
        read_locked(altered, RESULT_BLOB)


def test_audit_has_no_source_or_training_entrypoint():
    script = (Path(__file__).resolve().parents[1] /
              "experiments" / "p78mr_gain_posthoc_audit.py").read_text()
    for forbidden in ("make_batch(", "source.make_world(", "optimizer.step(",
                      "torch.load(", "requests.get(", "urlopen("):
        assert forbidden not in script
