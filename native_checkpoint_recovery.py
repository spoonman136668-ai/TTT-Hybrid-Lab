"""Inactive, explicitly selected immutable P2 checkpoint replication/recovery.

Not an owner, scheduler, trainer, accepted-pointer writer, off-host backup,
or proof of authentication or power-loss durability. Never selects a tip.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping

from native_training_checkpoint import validate_chain

HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_ANCESTORS = 256
DEFAULT_MAX_TOTAL_BYTES = 16 * 1024**3
CHUNK = 1024 * 1024


class RecoveryRejected(ValueError):
    pass


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise RecoveryRejected(reason)


def _valid_hash(value: object) -> bool:
    return type(value) is str and HEX64.fullmatch(value) is not None


def _digest_file(path: Path, max_bytes: int) -> tuple[str, int]:
    _require(path.is_file() and not path.is_symlink(), "RECOVERY_OBJECT_INVALID")
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as inp:
        while data := inp.read(CHUNK):
            count += len(data)
            _require(count <= max_bytes, "RECOVERY_BUDGET_EXCEEDED")
            digest.update(data)
    return digest.hexdigest(), count


def _objects(source: Path, tip: str, identity: Mapping) -> list[tuple[str, str]]:
    """Enumerate only immutable files in the fully verified parent chain."""
    _require(_valid_hash(tip), "RECOVERY_TIP_INVALID")
    validate_chain(source, tip, identity)
    current = tip
    seen = set()
    backwards = []
    for _ in range(MAX_ANCESTORS):
        _require(current not in seen, "RECOVERY_CHAIN_CYCLE")
        seen.add(current)
        path = source / (current + ".json")
        _require(path.is_file() and not path.is_symlink(), "RECOVERY_MANIFEST_INVALID")
        raw = path.read_bytes()
        _require(len(raw) <= 8192 and hashlib.sha256(raw).hexdigest() == current,
                 "RECOVERY_MANIFEST_TAMPERED")
        try:
            manifest = json.loads(raw)
        except (ValueError, UnicodeError, TypeError) as exc:
            raise RecoveryRejected("RECOVERY_MANIFEST_UNREADABLE") from exc
        blob = manifest.get("payload_sha256") if isinstance(manifest, dict) else None
        _require(_valid_hash(blob), "RECOVERY_PAYLOAD_HASH_INVALID")
        backwards.extend(((current, ".json"), (blob, ".pt")))
        parent = manifest.get("parent_manifest_sha256")
        if parent is None:
            return list(reversed(backwards))
        _require(_valid_hash(parent), "RECOVERY_PARENT_INVALID")
        current = parent
    raise RecoveryRejected("RECOVERY_CHAIN_TOO_DEEP")


def _publish_object(source: Path, dest: Path, digest: str, max_bytes: int) -> int:
    """Verify while copying; hardlink atomically without replacing an object."""
    _require(source.is_file() and not source.is_symlink(), "RECOVERY_SOURCE_OBJECT_INVALID")
    _require(not dest.is_symlink(), "RECOVERY_DEST_SYMLINK")
    fd, temporary = tempfile.mkstemp(prefix=".recovery-", dir=dest.parent)
    temporary_path = Path(temporary)
    try:
        h, count = hashlib.sha256(), 0
        with source.open("rb") as reader, os.fdopen(fd, "wb") as writer:
            while chunk := reader.read(CHUNK):
                count += len(chunk)
                _require(count <= max_bytes, "RECOVERY_BUDGET_EXCEEDED")
                h.update(chunk)
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        _require(h.hexdigest() == digest, "RECOVERY_SOURCE_TAMPERED")
        try:
            os.link(temporary_path, dest)
        except FileExistsError:
            expected, size = _digest_file(dest, max_bytes)
            _require(expected == digest and size == count,
                     "RECOVERY_IMMUTABLE_COLLISION")
        verify, size = _digest_file(dest, max_bytes)
        _require(verify == digest and size == count, "RECOVERY_DEST_TAMPERED")
        return count
    finally:
        temporary_path.unlink(missing_ok=True)


def mirror_verified_chain(source: str | Path, destination: str | Path, *,
                          tip_sha256: str, identity: Mapping,
                          max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES) -> dict:
    """Replicate a caller-selected P2 chain to an already provisioned directory.

    This API never chooses a latest/accepted checkpoint. The caller must verify
    storage independence, ownership, credentials, and actual host durability.
    Same-host two-directory testing only establishes the local copying contract.
    """
    _require(type(max_total_bytes) is int and max_total_bytes > 0,
             "RECOVERY_BUDGET_INVALID")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    _require(source.is_dir() and destination.is_dir(), "RECOVERY_DIRECTORY_MISSING")
    _require(source != destination and source not in destination.parents
             and destination not in source.parents, "RECOVERY_DIRECTORY_OVERLAP")
    objs = _objects(source, tip_sha256, identity)
    unique = []
    seen = set()
    for digest, extension in objs:
        key = digest + extension
        if key not in seen:
            unique.append((digest, extension))
            seen.add(key)
    _require(len(unique) <= 2 * MAX_ANCESTORS, "RECOVERY_TOO_MANY_OBJECTS")
    total = 0
    # Copy parent-root first; interrupted candidate remains unselected.
    for digest, extension in unique:
        obj_source = source / (digest + extension)
        _require(obj_source.is_file() and not obj_source.is_symlink(),
                 "RECOVERY_SOURCE_OBJECT_INVALID")
        bytes_left = max_total_bytes - total
        _require(bytes_left > 0, "RECOVERY_BUDGET_EXCEEDED")
        size = _publish_object(obj_source, destination / (digest + extension), digest, bytes_left)
        total += size
    # A successful transport still means only locally verified bytes, not
    # owner-authenticated accepted state or independently durable storage.
    verified = validate_chain(destination, tip_sha256, identity)
    return {
        "schema": "ttt-hybrid.checkpoint-copy-review.v1",
        "tip_sha256": tip_sha256,
        "steps": verified["steps"],
        "consumed_examples": verified["consumed_examples"],
        "objects_verified": len(unique),
        "logical_bytes_verified": total,
        "state": "COPY_VERIFIED_UNAUTHENTICATED",
        "owner_authenticated": False,
        "independent_storage_qualified": False,
        "accepted_pointer_changed": False,
        "training_resumed": False,
    }
