"""C4 factual measured-vs-claimed policy integration, DEV data only."""
from copy import deepcopy
import hashlib
import json

import pytest
import torch

from native_capability_registry import RegistryRejected, review_challenger
from scripts import native_c4_measured_retention as c4


def test_actual_checkpoint_scores_are_measured_and_auditable(tmp_path):
    torch.set_num_threads(1)
    seed = c4._spec()["initializations"][0]
    folder = tmp_path / "training"
    folder.mkdir()
    result = c4.measure_one(seed, folder)
    assert result["status"] == "PASS_MEASURED_DEV_POLICY_INTEGRITY_ONLY"
    assert result["champion_tip_sha256"] != result["challenger_tip_sha256"]
    assert result["evaluation_data_quality"] == "PUBLIC_TRAINING_REGIMES_NOT_HELDOUT"
    assert result["unchanged_control"]["status"] == "BLOCKED"
    assert result["review"]["scientifically_accepted"] is False
    assert result["review"]["promoted"] is False
    assert result["review"]["evidence_authenticated"] is False
    assert result["science_first_use"] == "NOT_RUN"
    assert not (folder / "accepted").exists()
    assert not (folder / "latest").exists()
    for measurement in result["measurements"]:
        evidence = result["measurement_vectors"][measurement["id"]]
        assert len(evidence["champion_squared_errors"]) == 16
        assert len(evidence["challenger_squared_errors"]) == 16
        assert evidence["champion_checkpoint_sha256"] == result["champion_tip_sha256"]
        assert evidence["challenger_checkpoint_sha256"] == result["challenger_tip_sha256"]
        raw = c4._canonical(evidence)
        assert hashlib.sha256(raw).hexdigest() == measurement["evidence_sha256"]
        assert measurement["champion"] == pytest.approx(
            sum(evidence["champion_squared_errors"]) / 16, rel=0, abs=1e-12)
        assert measurement["challenger"] == pytest.approx(
            sum(evidence["challenger_squared_errors"]) / 16, rel=0, abs=1e-12)


def test_frozen_policy_rejects_claimed_retention_regression():
    original = c4._spec()
    spec = deepcopy(original)
    policy = {"schema": c4.POLICY_SCHEMA,
              "champion_checkpoint_sha256": "a" * 64,
              "protocol_sha256": "b" * 64,
              "harness_sha256": "c" * 64,
              "evaluation_split_sha256": "d" * 64,
              "capabilities": deepcopy(spec["capabilities"]),
              "budget": deepcopy(spec["budget"])}
    digest = hashlib.sha256(c4._canonical(policy)).hexdigest()
    receipt = {
        "schema": c4.REVIEW_SCHEMA,
        "policy_sha256": digest,
        "champion_checkpoint_sha256": policy["champion_checkpoint_sha256"],
        "challenger_checkpoint_sha256": "e" * 64,
        "protocol_sha256": policy["protocol_sha256"],
        "harness_sha256": policy["harness_sha256"],
        "evaluation_split_sha256": policy["evaluation_split_sha256"],
        "measurements": [
            {"id": "old_regime_mse", "champion": 0.3,
             "challenger": 0.4, "samples": 16,
             "evidence_sha256": "1" * 64},
            {"id": "postreversal_mse", "champion": 0.5,
             "challenger": 0.1, "samples": 16,
             "evidence_sha256": "2" * 64},
        ],
        "cost": {"wall_seconds": 1, "gpu_seconds": 0,
                 "resident_bytes": 1000000},
    }
    # This is an intentional fabricated negative CONTROL, not a measurement.
    review = review_challenger(policy, digest, receipt)
    assert review["status"] == "BLOCKED"
    assert "old_regime_mse:RETENTION_REGRESSION" in review["reasons"]
    assert "postreversal_mse" in review["claimed_gains"]
    assert review["promoted"] is False
    tampered = deepcopy(receipt)
    tampered["evaluation_split_sha256"] = "9" * 64
    with pytest.raises(RegistryRejected, match="IDENTITY_DRIFT"):
        review_challenger(policy, digest, tampered)


def test_science_first_use_denied_before_training(monkeypatch):
    monkeypatch.setenv("NATIVE_C4_ENGINEERING_ONLY", "0")
    with pytest.raises(AssertionError, match="SCIENCE_FIRST_USE_FORBIDDEN"):
        c4.run()
    monkeypatch.delenv("NATIVE_C4_ENGINEERING_ONLY")
    monkeypatch.setenv("NATIVE_P3_ENGINEERING_ONLY", "0")
    with pytest.raises(AssertionError, match="SCIENCE_FIRST_USE_FORBIDDEN"):
        c4.run()
