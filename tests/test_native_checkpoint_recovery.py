"""P4 engineering-only replay/storage tests: no research data or GPU training."""
import hashlib

import pytest
import torch

import native_checkpoint_recovery as recovery
from native_checkpoint_recovery import RecoveryRejected, mirror_verified_chain
from native_training_checkpoint import (CheckpointRejected, load_checkpoint,
                                        save_checkpoint, validate_chain)


def identity():
    keys = ("architecture_sha256", "interface_sha256", "protocol_sha256",
            "data_split_sha256", "harness_sha256", "environment_sha256")
    return {"source_sha": "a" * 40, **{k: hashlib.sha256(k.encode()).hexdigest()
                                         for k in keys}}


def build(tmp_path):
    source = tmp_path / "primary"
    replica = tmp_path / "replica"
    staging = tmp_path / "staging"
    for path in (source, replica, staging):
        path.mkdir()
    torch.manual_seed(1911)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    memory = {"recurrent": torch.ones(1, 2),
              "fast_weights": torch.ones(1, 2, 2),
              "history": torch.ones(1, 1, 2)}
    root = save_checkpoint(source, model=model, optimizer=optimizer,
                           identity=identity(), native_state=memory,
                           sampler_state={"position": 0}, steps=0,
                           consumed_examples=0, parent_manifest_sha256=None)
    x = torch.ones(1, 2)
    optimizer.zero_grad()
    model(x).square().sum().backward()
    optimizer.step()
    memory["fast_weights"].add_(0.25)
    tip = save_checkpoint(source, model=model, optimizer=optimizer,
                          identity=identity(), native_state=memory,
                          sampler_state={"position": 1}, steps=1,
                          consumed_examples=1, parent_manifest_sha256=root)
    return source, replica, staging, root, tip, model


def test_mirror_and_cold_recover_complete_trainable_state(tmp_path):
    source, replica, staging, root, tip, model = build(tmp_path)
    result = mirror_verified_chain(source, replica, tip_sha256=tip, identity=identity())
    assert result["state"] == "COPY_VERIFIED_UNAUTHENTICATED"
    assert result["objects_verified"] == 4
    assert result["steps"] == 1 and result["consumed_examples"] == 1
    assert result["independent_storage_qualified"] is False
    assert result["accepted_pointer_changed"] is False
    again = mirror_verified_chain(replica, staging, tip_sha256=tip,
                                  identity=identity())
    assert again["tip_sha256"] == tip
    cold = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(cold.parameters(), lr=0.01)
    native, sampler = load_checkpoint(staging, tip, model=cold, optimizer=opt,
                                     identity=identity())
    for name, weight in model.state_dict().items():
        torch.testing.assert_close(cold.state_dict()[name], weight, atol=0, rtol=0)
    assert sampler == {"position": 1}
    assert native["fast_weights"][0, 0, 0] == 1.25
    assert opt.state_dict()["state"]
    for d in (source, replica, staging):
        assert not (d / "accepted").exists()
        assert not (d / "latest").exists()


def test_interrupted_copy_remains_unselected_and_retry_is_safe(tmp_path, monkeypatch):
    source, replica, _, root, tip, _ = build(tmp_path)
    original = recovery._publish_object
    count = 0

    def inject(*args, **kwargs):
        nonlocal count
        result = original(*args, **kwargs)
        count += 1
        if count == 2:
            raise RuntimeError("SIMULATED_STOP_AFTER_PARTIAL_COPY")
        return result

    monkeypatch.setattr(recovery, "_publish_object", inject)
    with pytest.raises(RuntimeError, match="SIMULATED_STOP"):
        mirror_verified_chain(source, replica, tip_sha256=tip, identity=identity())
    assert not (replica / (tip + ".json")).exists()
    monkeypatch.setattr(recovery, "_publish_object", original)
    result = mirror_verified_chain(source, replica, tip_sha256=tip,
                                   identity=identity())
    assert result["objects_verified"] == 4
    assert validate_chain(replica, tip, identity())["steps"] == 1


def test_corrupt_primary_is_recovered_only_from_explicit_valid_replica(tmp_path):
    source, replica, staging, _, tip, _ = build(tmp_path)
    mirror_verified_chain(source, replica, tip_sha256=tip, identity=identity())
    obj = next(source.glob("*.pt"))
    obj.write_bytes(obj.read_bytes() + b"malformed")
    with pytest.raises((CheckpointRejected, RecoveryRejected)):
        mirror_verified_chain(source, staging, tip_sha256=tip, identity=identity())
    assert mirror_verified_chain(replica, staging, tip_sha256=tip,
                                 identity=identity())["steps"] == 1


def test_existing_corrupt_replica_object_fails_closed_without_overwriting(tmp_path):
    source, replica, _, root, tip, _ = build(tmp_path)
    original = (source / (root + ".json")).read_bytes()
    (replica / (root + ".json")).write_bytes(b"corrupted")
    with pytest.raises(RecoveryRejected, match="IMMUTABLE_COLLISION"):
        mirror_verified_chain(source, replica, tip_sha256=tip, identity=identity())
    assert (replica / (root + ".json")).read_bytes() == b"corrupted"
    assert (source / (root + ".json")).read_bytes() == original


def test_bad_tip_or_drift_cannot_choose_fallback(tmp_path):
    source, replica, _, _, tip, _ = build(tmp_path)
    with pytest.raises(RecoveryRejected, match="TIP_INVALID"):
        mirror_verified_chain(source, replica, tip_sha256="latest", identity=identity())
    wrong = identity()
    wrong["source_sha"] = "b" * 40
    with pytest.raises(CheckpointRejected):
        mirror_verified_chain(source, replica, tip_sha256=tip, identity=wrong)
    assert not list(replica.glob("*.json"))


def test_overlap_and_missing_storage_rejected(tmp_path):
    source, replica, _, _, tip, _ = build(tmp_path)
    (source / "child").mkdir()
    with pytest.raises(RecoveryRejected, match="DIRECTORY_OVERLAP"):
        mirror_verified_chain(source, source / "child", tip_sha256=tip, identity=identity())
    with pytest.raises(RecoveryRejected, match="DIRECTORY_MISSING"):
        mirror_verified_chain(source, tmp_path / "unprovisioned", tip_sha256=tip,
                              identity=identity())


def test_transfer_budget_rejects_without_exposing_validated_tip(tmp_path):
    source, replica, _, _, tip, _ = build(tmp_path)
    with pytest.raises(RecoveryRejected, match="BUDGET_EXCEEDED"):
        mirror_verified_chain(source, replica, tip_sha256=tip, identity=identity(),
                              max_total_bytes=10)
    assert not list(replica.glob("*.json"))
    assert not list(replica.glob("*.pt"))
    assert not list(replica.glob(".recovery-*"))


def test_repeated_transfer_has_deterministic_integrity_and_no_mutable_pointer(tmp_path):
    source, replica, _, root, tip, _ = build(tmp_path)
    first = mirror_verified_chain(source, replica, tip_sha256=tip, identity=identity())
    second = mirror_verified_chain(source, replica, tip_sha256=tip, identity=identity())
    assert first == second
    assert len(list(replica.glob("*.json"))) == 2
    assert len(list(replica.glob("*.pt"))) == 2
    assert not any(p.name in ("accepted", "latest") for p in replica.iterdir())
