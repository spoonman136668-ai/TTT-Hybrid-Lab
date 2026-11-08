"""C1 remote checkpoint-to-next-session integration; synthetic CPU only."""
import json
from pathlib import Path
import pytest

from scripts.native_c1_remote_resume import (
    PRODUCER_SHA, TEST_KEY, continue_remote, verify_frozen_trainer_compatibility,
)
from scripts.native_p6_artifact_recovery import produce


def previous_run_fixture(tmp_path):
    downloaded = tmp_path / "downloaded-p6-artifact"
    produced = produce(downloaded, PRODUCER_SHA)
    return downloaded, produced


def test_remote_training_continuity_preserves_lineage_and_memory(tmp_path):
    downloaded, first = previous_run_fixture(tmp_path)
    result = continue_remote(downloaded, tmp_path / "scratch")
    assert result["status"] == "PASS_SYNTHETIC_REMOTE_RESUME_AND_ADVANCE"
    assert result["parent_tip_sha256"] == first["tip_manifest_sha256"]
    assert result["parent_steps"] == 1 and result["child_steps"] == 2
    assert result["final_consumed_examples"] == 2
    assert result["model_optimizer_and_fast_memory_advanced"] is True
    assert result["cold_resume_child_bit_identical"] is True
    assert result["stale_custody_record_rejected"] is True
    assert result["new_child_inventory_objects"] == 6
    assert result["backup_owner_authenticated"] is False
    assert result["physical_recovery_host_separate"] is False
    assert result["continuous_trainer_activated"] is False
    assert result["model_promoted"] is False
    assert result["scientific_learning_advantage"] == "UNVERIFIED"


def test_replay_same_staging_refused_without_modification(tmp_path):
    downloaded, first = previous_run_fixture(tmp_path)
    scratch = tmp_path / "scratch"
    continue_remote(downloaded, scratch)
    files = sorted(p.name for p in (scratch / "stage").iterdir())
    with pytest.raises(ValueError, match="SCRATCH_MUST_BE_ABSENT"):
        continue_remote(downloaded, scratch)
    assert sorted(p.name for p in (scratch / "stage").iterdir()) == files


def test_source_identity_and_test_key_admission(tmp_path):
    downloaded, first = previous_run_fixture(tmp_path)
    with pytest.raises(ValueError, match="UNREGISTERED_SOURCE"):
        continue_remote(downloaded, tmp_path / "scratch",
                        source_sha="f" * 40)
    with pytest.raises(ValueError, match="TEST_KEY_MISSING"):
        continue_remote(downloaded, tmp_path / "scratch", test_key=b"short")
    assert not (tmp_path / "scratch").exists()


def test_tampered_remote_payload_cannot_start_next_session(tmp_path):
    downloaded, first = previous_run_fixture(tmp_path)
    payload = next(downloaded.glob("*.pt"))
    payload.write_bytes(payload.read_bytes() + b"tamper")
    with pytest.raises(Exception):
        continue_remote(downloaded, tmp_path / "scratch")
    assert not (tmp_path / "scratch" / "stage" / "accepted").exists()


def test_frozen_producer_functions_and_checkpoint_serializer_verified():
    verify_frozen_trainer_compatibility()
