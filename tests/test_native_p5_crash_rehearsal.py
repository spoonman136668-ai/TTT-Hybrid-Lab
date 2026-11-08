"""P5 process-crash proof on synthetic public data; no science first-use."""
from pathlib import Path
import pytest

from scripts.native_p5_crash_rehearsal import rehearse


def test_abrupt_child_process_death_and_explicit_cold_replica_recovery(tmp_path):
    result = rehearse(tmp_path / "rehearsal")
    assert result["status"] == "PASS_SYNTHETIC_PROCESS_CRASH_RECOVERY_ONLY"
    assert result["child_exit_code"] == 73
    assert result["partial_tip_rejected"] is True
    assert result["corrupt_primary_rejected"] is True
    assert result["replica_chain_verified"] is True
    assert result["cold_model_optimizer_native_sampler_equal"] is True
    assert result["objects_verified"] == 4
    assert result["off_host_failure_domain_verified"] is False
    assert result["model_promoted"] is False
    assert result["scientific_accepted"] is False
    assert result["training_activated"] is False


def test_nonempty_recovery_scratch_refuses_reuse(tmp_path):
    root = tmp_path / "occupied"
    root.mkdir()
    (root / "do-not-overwrite").write_text("sentinel", encoding="utf-8")
    with pytest.raises(ValueError, match="SCRATCH_MUST_BE_EMPTY"):
        rehearse(root)
    assert (root / "do-not-overwrite").read_text(encoding="utf-8") == "sentinel"
