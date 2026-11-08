"""Synthetic, untrusted-score policy tests only; no original data or training."""
import copy
import hashlib
import json

import pytest

from native_capability_registry import (RegistryRejected, freeze_policy,
                                        read_policy, review_challenger,
                                        REVIEW_SCHEMA, SCHEMA)


def digest(key):
    return hashlib.sha256(key.encode()).hexdigest()


def policy():
    return {
        "schema": SCHEMA,
        "champion_checkpoint_sha256": digest("champion"),
        "protocol_sha256": digest("protocol"),
        "harness_sha256": digest("harness"),
        "evaluation_split_sha256": digest("frozen-holdout"),
        "capabilities": [
            {"id": "causal.intervention", "direction": "higher", "floor": .7,
             "retention_tolerance": .02, "meaningful_gain": .06,
             "minimum_samples": 40},
            {"id": "cold.retention", "direction": "lower", "floor": .3,
             "retention_tolerance": .015, "meaningful_gain": .05,
             "minimum_samples": 50}],
        "budget": {"max_wall_seconds": 120, "max_gpu_seconds": 60,
                   "max_resident_bytes": 3000000},
    }


def receipt(p, sha):
    return {
        "schema": REVIEW_SCHEMA, "policy_sha256": sha,
        "champion_checkpoint_sha256": p["champion_checkpoint_sha256"],
        "challenger_checkpoint_sha256": digest("challenger"),
        "protocol_sha256": p["protocol_sha256"],
        "harness_sha256": p["harness_sha256"],
        "evaluation_split_sha256": p["evaluation_split_sha256"],
        "measurements": [
            {"id": "causal.intervention", "champion": .8, "challenger": .9,
             "samples": 45, "evidence_sha256": digest("intervention-evidence")},
            {"id": "cold.retention", "champion": .12, "challenger": .12,
             "samples": 55, "evidence_sha256": digest("retention-evidence")}],
        "cost": {"wall_seconds": 50, "gpu_seconds": 30,
                 "resident_bytes": 1000000},
    }


def base(tmp_path):
    p = policy()
    sha = freeze_policy(tmp_path, p)
    return p, sha, receipt(p, sha)


def test_policy_is_immutable_content_addressed_and_explicit(tmp_path):
    p, sha, r = base(tmp_path)
    assert freeze_policy(tmp_path, p) == sha
    assert read_policy(tmp_path, sha) == p
    assert not (tmp_path / "accepted").exists()
    assert not (tmp_path / "latest").exists()
    result = review_challenger(p, sha, r)
    assert result["status"] == "CANDIDATE_REQUIRES_INDEPENDENT_QUALIFICATION"
    assert result["claimed_gains"] == ["causal.intervention"]
    assert result["evidence_authenticated"] is False
    assert result["scientifically_accepted"] is False
    assert result["promoted"] is False
    assert result["training_authorized"] is False


def test_corrupt_policy_and_unfrozen_threshold_rejected(tmp_path):
    p, sha, r = base(tmp_path)
    (tmp_path / (sha + ".json")).write_text('{}')
    with pytest.raises(RegistryRejected, match="TAMPERED"):
        read_policy(tmp_path, sha)
    p["capabilities"][0]["retention_tolerance"] = .2
    with pytest.raises(RegistryRejected, match="GAIN_MUST_EXCEED_TOLERANCE"):
        review_challenger(p, sha, r)


def test_retention_regression_blocks_even_if_new_task_gains(tmp_path):
    p, sha, r = base(tmp_path)
    r["measurements"][1]["challenger"] = .20
    outcome = review_challenger(p, sha, r)
    assert outcome["status"] == "BLOCKED"
    assert "cold.retention:RETENTION_REGRESSION" in outcome["reasons"]


def test_no_gain_and_incompetent_champion_block(tmp_path):
    p, sha, r = base(tmp_path)
    r["measurements"][0]["challenger"] = .81
    assert "NO_PREDECLARED_MEANINGFUL_GAIN" in review_challenger(p, sha, r)["reasons"]
    r["measurements"][0]["champion"] = .6
    assert "causal.intervention:INCOMPETENT_CHAMPION" in review_challenger(p, sha, r)["reasons"]


def test_budget_and_sample_shortfall_block(tmp_path):
    p, sha, r = base(tmp_path)
    r["cost"]["gpu_seconds"] = 61
    r["measurements"][1]["samples"] = 1
    reasons = review_challenger(p, sha, r)["reasons"]
    assert "gpu_seconds:RESOURCE_BUDGET_EXCEEDED" in reasons
    assert "cold.retention:INSUFFICIENT_SAMPLES" in reasons


def test_sealed_eval_drift_or_checkpoint_reset_rejected(tmp_path):
    p, sha, r = base(tmp_path)
    r["evaluation_split_sha256"] = digest("changed")
    with pytest.raises(RegistryRejected, match="IDENTITY_DRIFT"):
        review_challenger(p, sha, r)
    r = receipt(p, sha)
    r["challenger_checkpoint_sha256"] = p["champion_checkpoint_sha256"]
    with pytest.raises(RegistryRejected, match="NO_DISTINCT_CHALLENGER"):
        review_challenger(p, sha, r)


def test_missing_duplicate_or_unaccounted_capability_rejected(tmp_path):
    p, sha, r = base(tmp_path)
    r["measurements"] = r["measurements"][:1]
    with pytest.raises(RegistryRejected, match="INCOMPLETE_CAPABILITIES"):
        review_challenger(p, sha, r)
    r = receipt(p, sha)
    r["measurements"][1]["id"] = r["measurements"][0]["id"]
    with pytest.raises(RegistryRejected, match="DUPLICATE_CAPABILITY"):
        review_challenger(p, sha, r)


def test_nan_boolean_and_reused_evidence_rejected(tmp_path):
    p, sha, r = base(tmp_path)
    r["measurements"][0]["challenger"] = float("nan")
    with pytest.raises(RegistryRejected, match="MEASUREMENT_INVALID"):
        review_challenger(p, sha, r)
    r = receipt(p, sha)
    r["cost"]["resident_bytes"] = True
    with pytest.raises(RegistryRejected, match="COST_INVALID"):
        review_challenger(p, sha, r)
    r = receipt(p, sha)
    r["measurements"][1]["evidence_sha256"] = r["measurements"][0]["evidence_sha256"]
    with pytest.raises(RegistryRejected, match="MEASUREMENT_INVALID"):
        review_challenger(p, sha, r)


def test_unqualified_policy_and_evaluation_origin_cannot_pass(tmp_path):
    p, sha, r = base(tmp_path)
    p["capabilities"].append(copy.deepcopy(p["capabilities"][0]))
    with pytest.raises(RegistryRejected, match="CAPABILITY_ID"):
        review_challenger(p, sha, r)
    p, sha, r = base(tmp_path)
    r["policy_sha256"] = digest("fake")
    with pytest.raises(RegistryRejected, match="SCHEMA_OR_POLICY"):
        review_challenger(p, sha, r)
