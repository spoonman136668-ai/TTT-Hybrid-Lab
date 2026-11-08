"""P6 project-owned GitHub Actions artifact backup/retrieval rehearsal (CPU DEV).

Two *separate* workflow jobs: synthetic checkpoint creation and remote artifact
upload, then a fresh checkout and remote artifact download/cold restore.
No production backup, private model data, live trainer or external disk writes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from native_checkpoint_recovery import mirror_verified_chain
from native_training_checkpoint import load_checkpoint, save_checkpoint, validate_chain

SCHEMA = "ttt-hybrid.remote-artifact-recovery-eng.v1"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
MAX_METADATA_BYTES = 8192


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError("P6_" + reason)


def _identity(source_sha: str) -> dict:
    _require(type(source_sha) is str and HEX40.fullmatch(source_sha) is not None,
             "SOURCE_SHA_INVALID")
    keys = ("architecture_sha256", "interface_sha256", "protocol_sha256",
            "data_split_sha256", "harness_sha256", "environment_sha256")
    return {"source_sha": source_sha,
            **{key: hashlib.sha256(("p6-synthetic:" + key).encode()).hexdigest()
               for key in keys}}


def _model() -> tuple:
    net = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.01)
    return net, optimizer


def _native() -> dict:
    return {"recurrent": torch.ones(1, 2),
            "fast_weights": torch.ones(1, 2, 2),
            "history": torch.ones(1, 1, 2)}


def _step(model, optimizer) -> None:
    x = torch.tensor([[0.25, 0.75]])
    optimizer.zero_grad(set_to_none=True)
    model(x).square().sum().backward()
    optimizer.step()


def _reference() -> tuple:
    torch.manual_seed(64511)
    model, optimizer = _model()
    native = _native()
    _step(model, optimizer)
    native["fast_weights"].add_(0.125)
    return deepcopy(model.state_dict()), deepcopy(optimizer.state_dict()), native


def _equal(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(
            _equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, type(a)) and len(a) == len(b) and all(
            _equal(x, y) for x, y in zip(a, b))
    return a == b


def _bounded_artifact(directory: Path) -> set[str]:
    _require(directory.is_dir() and not directory.is_symlink(),
             "ARTIFACT_DIRECTORY_INVALID")
    objects = list(directory.iterdir())
    _require(len(objects) == 5, "ARTIFACT_FILE_COUNT")
    total = 0
    for obj in objects:
        _require(obj.is_file() and not obj.is_symlink(), "ARTIFACT_NONREGULAR_FILE")
        _require(obj.name == "metadata.json" or
                 (len(obj.stem) == 64 and HEX64.fullmatch(obj.stem) is not None
                  and obj.suffix in (".json", ".pt")), "ARTIFACT_FILENAME")
        total += obj.stat().st_size
        _require(total <= MAX_ARTIFACT_BYTES, "ARTIFACT_BUDGET_EXCEEDED")
    return {obj.name for obj in objects}


def produce(destination: Path, source_sha: str) -> dict:
    """Create exactly one deterministic synthetic checkpoint chain for upload."""
    _require(not destination.exists(), "OUTPUT_MUST_BE_ABSENT")
    destination.mkdir(parents=True)
    identity = _identity(source_sha)
    torch.manual_seed(64511)
    model, optimizer = _model()
    native = _native()
    root = save_checkpoint(destination, model=model, optimizer=optimizer,
                           identity=identity, native_state=native,
                           sampler_state={"position": 0}, steps=0,
                           consumed_examples=0, parent_manifest_sha256=None)
    _step(model, optimizer)
    native["fast_weights"].add_(0.125)
    tip = save_checkpoint(destination, model=model, optimizer=optimizer,
                          identity=identity, native_state=native,
                          sampler_state={"position": 1}, steps=1,
                          consumed_examples=1, parent_manifest_sha256=root)
    metadata = {"schema": SCHEMA, "source_sha": source_sha,
                "root_manifest_sha256": root,
                "tip_manifest_sha256": tip,
                "step": 1, "consumed_examples": 1,
                "science_original": False}
    (destination / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8")
    _require(len(_bounded_artifact(destination)) == 5, "PRODUCER_INCOMPLETE")
    validate_chain(destination, tip, identity)
    return {"status": "SYNTHETIC_REMOTE_UPLOAD_CANDIDATE",
            "schema": SCHEMA, "tip_manifest_sha256": tip,
            "root_manifest_sha256": root,
            "files": 5, "source_sha": source_sha,
            "training_activated": False, "scientific_first_use": False}


def restore(downloaded: Path, staging: Path, source_sha: str) -> dict:
    """Cold-restore only from a separately downloaded, bounded artifact folder.

    Never invokes GitHub APIs, chooses a 'latest' tip, or accesses producer dir.
    """
    expected_identity = _identity(source_sha)
    names = _bounded_artifact(downloaded)
    raw = (downloaded / "metadata.json").read_bytes()
    _require(len(raw) <= MAX_METADATA_BYTES, "METADATA_TOO_LARGE")
    try:
        meta = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("P6_METADATA_UNREADABLE") from exc
    _require(type(meta) is dict and set(meta) == {
        "schema", "source_sha", "root_manifest_sha256",
        "tip_manifest_sha256", "step", "consumed_examples",
        "science_original"}, "METADATA_FIELDS")
    _require(meta["schema"] == SCHEMA and meta["source_sha"] == source_sha
             and meta["step"] == 1 and meta["consumed_examples"] == 1
             and meta["science_original"] is False, "METADATA_IDENTITY")
    root, tip = meta["root_manifest_sha256"], meta["tip_manifest_sha256"]
    _require(all(type(x) is str and HEX64.fullmatch(x) is not None
                 for x in (root, tip)) and root != tip, "METADATA_DIGEST")
    verified = validate_chain(downloaded, tip, expected_identity)
    root_manifest = json.loads((downloaded / (root + ".json")).read_text("utf-8"))
    _require(root_manifest["parent_manifest_sha256"] is None
             and root_manifest["steps"] == 0
             and verified["parent_manifest_sha256"] == root
             and verified["steps"] == 1
             and verified["consumed_examples"] == 1,
             "CHAIN_NOT_EXACT_TWO_STEP")
    _require(names == {"metadata.json", root + ".json", tip + ".json",
                       root_manifest["payload_sha256"] + ".pt",
                       verified["payload_sha256"] + ".pt"},
             "ARTIFACT_EXTRA_OR_MISSING_OBJECTS")
    _require(not staging.exists(), "STAGING_MUST_BE_ABSENT")
    staging.mkdir(parents=True)
    mirrored = mirror_verified_chain(downloaded, staging, tip_sha256=tip,
                                     identity=expected_identity,
                                     max_total_bytes=MAX_ARTIFACT_BYTES)
    expected_model, expected_optimizer, expected_native = _reference()
    cold, optimizer = _model()
    restored_native, cursor = load_checkpoint(staging, tip, model=cold,
                                              optimizer=optimizer,
                                              identity=expected_identity)
    _require(_equal(expected_model, cold.state_dict()),
             "COLD_MODEL_NOT_EQUAL_INDEPENDENT_FIXTURE")
    _require(_equal(expected_optimizer, optimizer.state_dict()),
             "COLD_OPTIMIZER_NOT_EQUAL_INDEPENDENT_FIXTURE")
    _require(_equal(expected_native, restored_native)
             and cursor == {"position": 1},
             "COLD_NATIVE_OR_CURSOR_NOT_EQUAL")
    _require(not any((staging / name).exists()
                     for name in ("accepted", "latest")), "POINTER_MUTATION")
    return {"schema": SCHEMA, "status": "PASS_REMOTE_ARTIFACT_COLD_RESTORE_DEV",
            "source_sha": source_sha, "tip_manifest_sha256": tip,
            "objects_verified": mirrored["objects_verified"],
            "cold_state_equal": True, "full_ancestry_verified": True,
            "independent_fixture_equal": True,
            "remote_artifact_transport": "EXTERNAL_WORKFLOW_DOWNLOAD",
            "storage_retention_guarantee_verified": False,
            "power_loss_survivability_verified": False,
            "independent_production_backup_qualified": False,
            "owner_authenticity_verified": False, "training_activated": False,
            "scientific_first_use": False, "model_promoted": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("produce", "restore"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--staging", type=Path)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "produce":
        _require(args.staging is None, "PRODUCER_STAGING_INVALID")
        outcome = produce(args.directory, args.source_sha)
    else:
        _require(args.staging is not None, "RESTORER_STAGING_REQUIRED")
        outcome = restore(args.directory, args.staging, args.source_sha)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(outcome, sort_keys=True, indent=2) + "\n",
                            encoding="utf-8")
    print(json.dumps(outcome, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
