"""C1 DEV: advance previously remotely recovered P2 training state by one step.

Consumes only the synthetic, already-published P6 GitHub artifact. No owner
backup key, model promotion, original science, external volume or live trainer.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from native_backup_custody import (
    BackupCustodyRejected, seal_backup_inventory, verify_backup_inventory,
)
from native_training_checkpoint import (
    CheckpointRejected, load_checkpoint, save_checkpoint, validate_chain,
)
from scripts import native_p6_artifact_recovery as p6

PRODUCER_SHA = "3ed4a9119900c3e356794973562c9fcbbeb727a9"
SCHEMA = "ttt-hybrid.c1-remote-training-continuity-dev.v1"
TEST_KEY = b"PUBLIC-SYNTHETIC-C1-DEV-KEY-NOT-FOR-AUTHENTICATION-2026"


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError("C1_" + reason)


def _same(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(
            _same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, type(a)) and len(a) == len(b) and all(
            _same(x, y) for x, y in zip(a, b))
    return a == b


def _semantic_functions(source: str) -> dict:
    tree = ast.parse(source)
    names = {"_identity", "_model", "_native", "_step"}
    functions = {node.name: ast.dump(node, include_attributes=False)
                 for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names}
    _require(set(functions) == names, "MISSING_FROZEN_PRODUCER_FUNCTION")
    return functions


def verify_frozen_trainer_compatibility() -> None:
    """Fail if synthetic training semantics changed since artifact creation."""
    prior = subprocess.run(
        ["git", "-C", str(ROOT), "show",
         PRODUCER_SHA + ":scripts/native_p6_artifact_recovery.py"],
        capture_output=True, text=True, check=False, timeout=10)
    _require(prior.returncode == 0, "ORIGINAL_SOURCE_UNAVAILABLE")
    current = (ROOT / "scripts/native_p6_artifact_recovery.py").read_text(
        encoding="utf-8")
    _require(_semantic_functions(prior.stdout) == _semantic_functions(current),
             "FROZEN_TRAINER_SOURCE_DRIFT")
    original_checkpoint = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse",
         PRODUCER_SHA + ":native_training_checkpoint.py"],
        capture_output=True, text=True, check=False, timeout=10)
    actual_checkpoint = subprocess.run(
        ["git", "-C", str(ROOT), "hash-object",
         "native_training_checkpoint.py"],
        capture_output=True, text=True, check=False, timeout=10)
    _require(original_checkpoint.returncode == 0
             and actual_checkpoint.returncode == 0
             and original_checkpoint.stdout.strip() == actual_checkpoint.stdout.strip(),
             "FROZEN_CHECKPOINT_SOURCE_DRIFT")


def continue_remote(downloaded: Path, scratch: Path, *,
                    source_sha: str = PRODUCER_SHA,
                    now_unix: int = 1794140000,
                    test_key: bytes = TEST_KEY) -> dict:
    """Remote -> verified restorable state -> second gradient step -> child.

    Uses no new world data; each update is a deterministic public CPU fixture.
    Same-platform cold roundtrip must match exactly. Cross-platform model math
    equivalence, genuine cognitive retention and off-host durability are not
    asserted.
    """
    _require(source_sha == PRODUCER_SHA, "UNREGISTERED_SOURCE")
    verify_frozen_trainer_compatibility()
    _require(not scratch.exists(), "SCRATCH_MUST_BE_ABSENT")
    _require(type(now_unix) is int and now_unix >= 0, "CLOCK_INVALID")
    _require(type(test_key) is bytes and len(test_key) >= 32,
             "TEST_KEY_MISSING")
    identity = p6._identity(source_sha)
    original = json.loads((downloaded / "metadata.json").read_text("utf-8"))
    tip = original["tip_manifest_sha256"]
    stage = scratch / "stage"
    scratch.mkdir(parents=True)
    restored = p6.restore(downloaded, stage, source_sha)
    _require(restored["cold_state_equal"] is True
             and restored["full_ancestry_verified"] is True
             and restored["objects_verified"] == 4, "REMOTE_RESTORE_UNVERIFIED")
    initial = seal_backup_inventory(
        stage, tip_manifest_sha256=tip, identity=identity,
        backup_domain_claim="synthetic-remote-artifact",
        issued_unix=now_unix, expires_unix=now_unix + 3600,
        trust_key=test_key)
    proof = verify_backup_inventory(
        stage, initial, expected_tip_sha256=tip, identity=identity,
        expected_backup_domain_claim="synthetic-remote-artifact",
        as_of_unix=now_unix + 1, trust_key=test_key)
    _require(proof["checkpoint_steps"] == 1
             and proof["checkpoint_consumed_examples"] == 1,
             "CHECKPOINT_NOT_TRAINED")
    trained, optimizer = p6._model()
    native, sampler = load_checkpoint(
        stage, tip, model=trained, optimizer=optimizer, identity=identity)
    _require(sampler == {"position": 1}
             and bool(optimizer.state_dict()["state"]),
             "REMOTE_OPTIMIZER_OR_CURSOR_RESET")
    prior_model = deepcopy(trained.state_dict())
    prior_opt = deepcopy(optimizer.state_dict())
    prior_native = deepcopy(native)
    rng_before = torch.get_rng_state().clone()
    p6._step(trained, optimizer)
    native["fast_weights"].add_(0.0625)
    _require(not _same(prior_model, trained.state_dict())
             and not _same(prior_opt, optimizer.state_dict())
             and not _same(prior_native, native),
             "TRAINING_DID_NOT_ADVANCE")
    child = save_checkpoint(
        stage, model=trained, optimizer=optimizer, identity=identity,
        native_state=native, sampler_state={"position": 2},
        steps=2, consumed_examples=2, parent_manifest_sha256=tip)
    manifest = validate_chain(stage, child, identity)
    _require(manifest["parent_manifest_sha256"] == tip
             and manifest["steps"] == 2 and manifest["consumed_examples"] == 2,
             "LINEAGE_REGRESSION")
    try:
        verify_backup_inventory(
            stage, initial, expected_tip_sha256=tip, identity=identity,
            expected_backup_domain_claim="synthetic-remote-artifact",
            as_of_unix=now_unix + 2, trust_key=test_key)
    except BackupCustodyRejected:
        pass
    else:
        raise AssertionError("C1_STALE_ATTESTATION_WAS_ACCEPTED")
    renewal = seal_backup_inventory(
        stage, tip_manifest_sha256=child, identity=identity,
        backup_domain_claim="synthetic-remote-artifact",
        issued_unix=now_unix, expires_unix=now_unix + 3600,
        trust_key=test_key)
    next_proof = verify_backup_inventory(
        stage, renewal, expected_tip_sha256=child, identity=identity,
        expected_backup_domain_claim="synthetic-remote-artifact",
        as_of_unix=now_unix + 3, trust_key=test_key)
    cold, cold_opt = p6._model()
    cold_native, cold_cursor = load_checkpoint(
        stage, child, model=cold, optimizer=cold_opt, identity=identity)
    _require(_same(trained.state_dict(), cold.state_dict())
             and _same(optimizer.state_dict(), cold_opt.state_dict())
             and _same(native, cold_native)
             and cold_cursor == {"position": 2},
             "SECOND_SESSION_NOT_COLD_RESTORABLE")
    _require(torch.equal(torch.get_rng_state(), rng_before),
             "SECOND_SESSION_RNG_NOT_RESTORED")
    _require(not (stage / "latest").exists()
             and not (stage / "accepted").exists(),
             "UNAUTHORIZED_ACCEPTED_POINTER")
    return {
        "schema": SCHEMA, "status": "PASS_SYNTHETIC_REMOTE_RESUME_AND_ADVANCE",
        "producer_source_sha": source_sha, "root_sha256": original["root_manifest_sha256"],
        "parent_tip_sha256": tip, "new_child_tip_sha256": child,
        "parent_steps": 1, "child_steps": 2,
        "final_consumed_examples": 2,
        "model_optimizer_and_fast_memory_advanced": True,
        "cold_resume_child_bit_identical": True,
        "stale_custody_record_rejected": True,
        "new_child_inventory_objects": next_proof["verified_objects"],
        "synthetic_test_key_only": True, "backup_owner_authenticated": False,
        "physical_recovery_host_separate": False,
        "continuous_trainer_activated": False,
        "scientific_learning_advantage": "UNVERIFIED",
        "model_promoted": False,
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloaded", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = continue_remote(args.downloaded, args.scratch)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                            encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
