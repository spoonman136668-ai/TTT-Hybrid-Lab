"""P7 offline backup integrity tests: synthetic-only, no external storage."""
from copy import deepcopy
import hashlib

import pytest
import torch

from native_backup_custody import (
    BackupCustodyRejected, seal_backup_inventory, verify_backup_inventory
)
from native_training_checkpoint import save_checkpoint, CheckpointRejected


TEST_KEY = b"p7-synthetic-test-key-not-for-production-0001"
WRONG_KEY = b"p7-synthetic-test-key-not-for-production-0002"


def identity():
    keys = ("architecture_sha256", "interface_sha256", "protocol_sha256",
            "data_split_sha256", "harness_sha256", "environment_sha256")
    return {"source_sha": "c" * 40,
            **{k: hashlib.sha256(("p7-" + k).encode()).hexdigest() for k in keys}}


def checkpoint(tmp_path):
    directory = tmp_path / "isolated"
    directory.mkdir()
    torch.manual_seed(7017)
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=0.005)
    native = {"recurrent": torch.ones(1, 2),
              "fast_weights": torch.ones(1, 2, 2),
              "history": torch.ones(1, 1, 2)}
    root = save_checkpoint(directory, model=model, optimizer=opt,
                           identity=identity(), native_state=native,
                           sampler_state={"position": 0}, steps=0,
                           consumed_examples=0, parent_manifest_sha256=None)
    opt.zero_grad()
    model(torch.tensor([[.2, .7]])).square().sum().backward()
    opt.step()
    native["fast_weights"].add_(.2)
    tip = save_checkpoint(directory, model=model, optimizer=opt,
                          identity=identity(), native_state=native,
                          sampler_state={"position": 1}, steps=1,
                          consumed_examples=1, parent_manifest_sha256=root)
    return directory, root, tip


def sealed(tmp_path):
    directory, root, tip = checkpoint(tmp_path)
    envelope = seal_backup_inventory(
        directory, tip_manifest_sha256=tip, identity=identity(),
        backup_domain_claim="synthetic-artifact-service",
        issued_unix=1000, expires_unix=2000, trust_key=TEST_KEY)
    return directory, root, tip, envelope


def verify(directory, tip, envelope, **kwargs):
    return verify_backup_inventory(
        directory, envelope, expected_tip_sha256=tip,
        identity=kwargs.pop("identity", identity()),
        expected_backup_domain_claim=kwargs.pop(
            "expected_backup_domain_claim", "synthetic-artifact-service"),
        as_of_unix=kwargs.pop("as_of_unix", 1100),
        trust_key=kwargs.pop("trust_key", TEST_KEY), **kwargs)


def test_keyed_exact_checkpoint_inventory_never_promotes(tmp_path):
    directory, root, tip, envelope = sealed(tmp_path)
    assert len(envelope["record"]["objects"]) == 4
    result = verify(directory, tip, envelope)
    assert result["status"] == "HASH_AND_TEST_KEY_VERIFIED_ONLY"
    assert result["verified_objects"] == 4
    assert result["checkpoint_steps"] == 1
    assert result["checkpoint_consumed_examples"] == 1
    assert result["hmac_verified_against_supplied_key"] is True
    assert result["key_owner_authenticated"] is False
    assert result["independent_failure_domain_verified"] is False
    assert result["actual_retention_sla_verified"] is False
    assert result["checkpoint_promoted"] is False
    assert result["training_activated"] is False
    assert not (directory / "latest").exists()
    assert not (directory / "accepted").exists()


def test_wrong_key_and_missing_key_fail_closed(tmp_path):
    directory, root, tip, envelope = sealed(tmp_path)
    with pytest.raises(BackupCustodyRejected, match="MAC_VERIFICATION_FAILED"):
        verify(directory, tip, envelope, trust_key=WRONG_KEY)
    with pytest.raises(BackupCustodyRejected, match="TRUST_KEY_MISSING_OR_SHORT"):
        verify(directory, tip, envelope, trust_key=b"bad")


def test_modified_signed_tip_and_claim_rejected(tmp_path):
    directory, root, tip, envelope = sealed(tmp_path)
    edited = deepcopy(envelope)
    edited["record"]["tip_manifest_sha256"] = root
    with pytest.raises(BackupCustodyRejected, match="MAC_VERIFICATION_FAILED"):
        verify(directory, tip, edited)
    edited = deepcopy(envelope)
    edited["record"]["backup_domain_claim"] = "different-backup"
    with pytest.raises(BackupCustodyRejected, match="MAC_VERIFICATION_FAILED"):
        verify(directory, tip, edited)


def test_untrusted_domain_claim_does_not_prove_independence(tmp_path):
    directory, root, tip, envelope = sealed(tmp_path)
    with pytest.raises(BackupCustodyRejected, match="DOMAIN_CLAIM_DRIFT"):
        verify(directory, tip, envelope,
               expected_backup_domain_claim="unrelated-destination")
    assert verify(directory, tip, envelope)["physical_loss_recovery_verified"] is False


def test_expired_and_not_yet_valid_envelopes_rejected(tmp_path):
    directory, root, tip, envelope = sealed(tmp_path)
    for invalid in (999, 2000, 2001):
        with pytest.raises(BackupCustodyRejected, match="EXPIRED_OR_NOT_YET_VALID"):
            verify(directory, tip, envelope, as_of_unix=invalid)


def test_corrupted_payload_rejected_despite_valid_mac(tmp_path):
    directory, root, tip, envelope = sealed(tmp_path)
    blob = next(directory.glob("*.pt"))
    blob.write_bytes(blob.read_bytes() + b"tampered")
    with pytest.raises((BackupCustodyRejected, CheckpointRejected)):
        verify(directory, tip, envelope)


def test_missing_parent_denies_recovery(tmp_path):
    directory, root, tip, envelope = sealed(tmp_path)
    (directory / (root + ".json")).unlink()
    with pytest.raises((BackupCustodyRejected, CheckpointRejected, FileNotFoundError)):
        verify(directory, tip, envelope)


def test_extra_mutable_or_unknown_files_fail_closed(tmp_path):
    directory, root, tip, envelope = sealed(tmp_path)
    (directory / "latest").write_text(tip, encoding="ascii")
    with pytest.raises(BackupCustodyRejected, match="UNEXPECTED_OR_MISSING_FILES"):
        verify(directory, tip, envelope)
    assert (directory / "latest").read_text(encoding="ascii") == tip


def test_source_identity_and_explicit_tip_must_match(tmp_path):
    directory, root, tip, envelope = sealed(tmp_path)
    bad = identity()
    bad["source_sha"] = "d" * 40
    with pytest.raises(BackupCustodyRejected, match="CHECKPOINT_IDENTITY_DRIFT"):
        verify(directory, tip, envelope, identity=bad)
    with pytest.raises(BackupCustodyRejected, match="CHECKPOINT_IDENTITY_DRIFT"):
        verify(directory, root, envelope)


def test_bounded_size_and_key_policy(tmp_path):
    directory, root, tip, envelope = sealed(tmp_path)
    with pytest.raises(BackupCustodyRejected, match="SIZE_BUDGET_EXCEEDED"):
        verify(directory, tip, envelope, max_total_bytes=1)
    with pytest.raises(BackupCustodyRejected, match="TRUST_KEY_MISSING_OR_SHORT"):
        seal_backup_inventory(directory, tip_manifest_sha256=tip,
                              identity=identity(),
                              backup_domain_claim="synthetic-service",
                              issued_unix=1000, expires_unix=2000,
                              trust_key=b"short")
    with pytest.raises(BackupCustodyRejected, match="ATTESTATION_WINDOW_INVALID"):
        seal_backup_inventory(directory, tip_manifest_sha256=tip,
                              identity=identity(),
                              backup_domain_claim="synthetic-service",
                              issued_unix=2000, expires_unix=1000,
                              trust_key=TEST_KEY)
