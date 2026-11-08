"""P6 bounded remote-artifact contract tests: synthetic CPU only."""
import json
import shutil

import pytest
import torch

from scripts.native_p6_artifact_recovery import produce, restore

SHA = "c" * 40


def fixture(tmp_path):
    artifact = tmp_path / "upload"
    staging = tmp_path / "fresh"
    result = produce(artifact, SHA)
    return artifact, staging, result


def test_exact_artifact_cold_restore_and_independent_reference(tmp_path):
    upload, staging, result = fixture(tmp_path)
    recovered = restore(upload, staging, SHA)
    assert recovered["status"] == "PASS_REMOTE_ARTIFACT_COLD_RESTORE_DEV"
    assert recovered["full_ancestry_verified"] is True
    assert recovered["independent_fixture_within_numeric_tolerance"] is True
    assert recovered["cold_state_equal"] is True
    assert recovered["objects_verified"] == 4
    assert recovered["tip_manifest_sha256"] == result["tip_manifest_sha256"]
    assert recovered["storage_retention_guarantee_verified"] is False
    assert recovered["owner_authenticity_verified"] is False
    assert recovered["model_promoted"] is False
    assert recovered["training_activated"] is False
    assert not (staging / "accepted").exists()


def test_producer_refuses_to_replace_any_existing_payload(tmp_path):
    upload, staging, result = fixture(tmp_path)
    with pytest.raises(ValueError, match="OUTPUT_MUST_BE_ABSENT"):
        produce(upload, SHA)
    assert len(list(upload.iterdir())) == 5


def test_missing_or_mutated_payload_is_rejected(tmp_path):
    upload, staging, result = fixture(tmp_path)
    blob = next(upload.glob("*.pt"))
    blob.write_bytes(blob.read_bytes() + b"tampers")
    with pytest.raises(Exception):
        restore(upload, staging, SHA)
    assert not staging.exists()


def test_missing_parent_manifest_is_rejected(tmp_path):
    upload, staging, result = fixture(tmp_path)
    (upload / (result["root_manifest_sha256"] + ".json")).unlink()
    with pytest.raises(Exception):
        restore(upload, staging, SHA)
    assert not staging.exists()


def test_modified_source_or_identity_is_rejected(tmp_path):
    upload, staging, result = fixture(tmp_path)
    with pytest.raises(Exception):
        restore(upload, staging, "d" * 40)
    assert not staging.exists()


def test_no_extra_injected_file_or_symlink(tmp_path):
    upload, staging, result = fixture(tmp_path)
    (upload / "latest").write_text(result["tip_manifest_sha256"])
    with pytest.raises(ValueError, match="ARTIFACT_FILE_COUNT"):
        restore(upload, staging, SHA)
    assert not staging.exists()


def test_tampered_metadata_cannot_select_new_tip(tmp_path):
    upload, staging, result = fixture(tmp_path)
    meta_file = upload / "metadata.json"
    meta = json.loads(meta_file.read_text("utf-8"))
    meta["tip_manifest_sha256"] = "e" * 64
    meta_file.write_text(json.dumps(meta))
    with pytest.raises(Exception):
        restore(upload, staging, SHA)
    assert not staging.exists()


def test_new_staging_required_and_existing_not_mutated(tmp_path):
    upload, staging, result = fixture(tmp_path)
    staging.mkdir()
    (staging / "sentinel.txt").write_text("keep")
    with pytest.raises(ValueError, match="STAGING_MUST_BE_ABSENT"):
        restore(upload, staging, SHA)
    assert (staging / "sentinel.txt").read_text() == "keep"


def test_byte_budget_refuses_large_artifact(tmp_path):
    upload, staging, result = fixture(tmp_path)
    (upload / "metadata.json").write_bytes(b"x" * (16 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="ARTIFACT_BUDGET_EXCEEDED"):
        restore(upload, staging, SHA)
    assert not staging.exists()

def test_independent_numeric_fixture_tolerance_does_not_weaken_saved_bits():
    from scripts.native_p6_artifact_recovery import _close_reference, _equal
    recorded = {"w": torch.tensor([1.0, 0.25], dtype=torch.float32)}
    rounded = {"w": torch.tensor([1.0000002, 0.25], dtype=torch.float32)}
    divergent = {"w": torch.tensor([1.001, 0.25], dtype=torch.float32)}
    assert not _equal(recorded, rounded)
    assert _close_reference(recorded, rounded)
    assert not _close_reference(recorded, divergent)
