"""Offline, content-bound P2 backup custody verification (inactive P7 DEV).

HMAC proves possession of a caller-provided verification key, not operator
identity, physical backup independence, service retention, or owner acceptance.
No key creation/storage, external transport, trainer operation or promotion.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
import re
from typing import Mapping

from native_checkpoint_recovery import _objects
from native_training_checkpoint import validate_chain

SCHEMA = "ttt-hybrid.offline-backup-custody.v1"
REPO = "spoonman136668-ai/TTT-Hybrid-Lab"
TAG = re.compile(r"^[a-z][a-z0-9_.-]{2,63}$")
HEX = re.compile(r"^[0-9a-f]{64}$")
MAX_OBJECTS = 512
MAX_RECORD_BYTES = 65536
DEFAULT_MAX_BYTES = 64 * 1024 * 1024
MAX_LIFETIME_SECONDS = 90 * 24 * 3600
DOMAIN_SEPARATOR = b"TTT-HYBRID/P7/BACKUP-CUSTODY/V1\n"


class BackupCustodyRejected(ValueError):
    pass


def _require(ok: bool, error: str) -> None:
    if not ok:
        raise BackupCustodyRejected(error)


def _canonical(obj: dict) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def _validate_key(key: bytes) -> None:
    _require(type(key) is bytes and len(key) >= 32,
             "BACKUP_TRUST_KEY_MISSING_OR_SHORT")


def _validate_window(issued_unix: int, expires_unix: int) -> None:
    _require(type(issued_unix) is int and type(expires_unix) is int
             and issued_unix >= 0 and issued_unix < expires_unix
             and expires_unix - issued_unix <= MAX_LIFETIME_SECONDS,
             "BACKUP_ATTESTATION_WINDOW_INVALID")


def _file_digest(path: Path, budget: int) -> tuple[str, int]:
    _require(path.is_file() and not path.is_symlink(),
             "BACKUP_OBJECT_NOT_REGULAR")
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            _require(size <= budget, "BACKUP_SIZE_BUDGET_EXCEEDED")
            h.update(chunk)
    return h.hexdigest(), size


def _inventory(directory: Path, tip: str, identity: Mapping,
               max_total_bytes: int) -> tuple[list[dict], dict]:
    """Enumerate an isolated *complete* immutable P2 lineage; reject extras."""
    _require(type(max_total_bytes) is int and 0 < max_total_bytes,
             "BACKUP_BUDGET_INVALID")
    _require(not directory.is_symlink() and directory.is_dir(),
             "BACKUP_DIRECTORY_INVALID")
    _require(type(tip) is str and HEX.fullmatch(tip) is not None,
             "BACKUP_TIP_INVALID")
    verified = validate_chain(directory, tip, identity)
    objects = _objects(directory, tip, identity)
    names = sorted({sha + extension for sha, extension in objects})
    _require(0 < len(names) <= MAX_OBJECTS, "BACKUP_OBJECT_COUNT")
    children = list(directory.iterdir())
    _require(all(child.is_file() and not child.is_symlink() for child in children),
             "BACKUP_UNSAFE_ENTRY")
    _require({child.name for child in children} == set(names)
             and len(children) == len(names), "BACKUP_UNEXPECTED_OR_MISSING_FILES")
    result = []
    total = 0
    for name in names:
        budget = max_total_bytes - total
        _require(budget > 0, "BACKUP_SIZE_BUDGET_EXCEEDED")
        digest, size = _file_digest(directory / name, budget)
        _require(digest == name.split(".")[0], "BACKUP_OBJECT_HASH_MISMATCH")
        total += size
        result.append({"name": name, "sha256": digest, "bytes": size})
    return result, verified


def _record_shape(record: dict) -> None:
    _require(type(record) is dict and set(record) == {
        "schema", "repo", "identity", "tip_manifest_sha256",
        "backup_domain_claim", "issued_unix", "expires_unix", "objects"
    }, "BACKUP_RECORD_FIELDS")
    _require(record["schema"] == SCHEMA and record["repo"] == REPO,
             "BACKUP_RECORD_SCOPE")
    _require(type(record["backup_domain_claim"]) is str
             and TAG.fullmatch(record["backup_domain_claim"]) is not None,
             "BACKUP_DOMAIN_CLAIM_INVALID")
    _validate_window(record["issued_unix"], record["expires_unix"])
    _require(type(record["tip_manifest_sha256"]) is str
             and HEX.fullmatch(record["tip_manifest_sha256"]) is not None,
             "BACKUP_RECORD_TIP")
    _require(type(record["identity"]) is dict, "BACKUP_RECORD_IDENTITY")
    _require(type(record["objects"]) is list
             and 0 < len(record["objects"]) <= MAX_OBJECTS,
             "BACKUP_RECORD_OBJECTS")
    seen = set()
    for obj in record["objects"]:
        _require(type(obj) is dict and set(obj) == {"name", "sha256", "bytes"},
                 "BACKUP_RECORD_OBJECT_FIELDS")
        _require(type(obj["name"]) is str
                 and re.fullmatch(r"[0-9a-f]{64}\.(?:json|pt)", obj["name"])
                 and obj["sha256"] == obj["name"][:64]
                 and type(obj["bytes"]) is int and obj["bytes"] >= 0
                 and obj["name"] not in seen, "BACKUP_RECORD_OBJECT_INVALID")
        seen.add(obj["name"])
    _require([o["name"] for o in record["objects"]] == sorted(seen),
             "BACKUP_RECORD_OBJECT_ORDER")


def seal_backup_inventory(directory: str | Path, *, tip_manifest_sha256: str,
                          identity: Mapping, backup_domain_claim: str,
                          issued_unix: int, expires_unix: int,
                          trust_key: bytes,
                          max_total_bytes: int = DEFAULT_MAX_BYTES) -> dict:
    """Prepare an integrity envelope from synthetic or owner-scoped bytes.

    The caller holds the key in memory; it is never embedded in this output.
    No claim of authentic owner or independent backup is licensed by this API.
    """
    _validate_key(trust_key)
    _validate_window(issued_unix, expires_unix)
    _require(type(backup_domain_claim) is str
             and TAG.fullmatch(backup_domain_claim) is not None,
             "BACKUP_DOMAIN_CLAIM_INVALID")
    objs, _ = _inventory(Path(directory), tip_manifest_sha256, identity,
                         max_total_bytes)
    record = {
        "schema": SCHEMA, "repo": REPO, "identity": dict(identity),
        "tip_manifest_sha256": tip_manifest_sha256,
        "backup_domain_claim": backup_domain_claim,
        "issued_unix": issued_unix, "expires_unix": expires_unix,
        "objects": objs,
    }
    _record_shape(record)
    payload = _canonical(record)
    _require(len(payload) <= MAX_RECORD_BYTES, "BACKUP_RECORD_TOO_LARGE")
    mac = hmac.new(trust_key, DOMAIN_SEPARATOR + payload, hashlib.sha256).hexdigest()
    return {"record": record, "hmac_sha256": mac}


def verify_backup_inventory(directory: str | Path, envelope: dict, *,
                            expected_tip_sha256: str, identity: Mapping,
                            expected_backup_domain_claim: str,
                            as_of_unix: int, trust_key: bytes,
                            max_total_bytes: int = DEFAULT_MAX_BYTES) -> dict:
    """Verify hash-bound bytes and bounded-key MAC; never authorize recovery.

    Caller must establish actual key custody and physical independence
    out of band. No output of this function is production admission.
    """
    _validate_key(trust_key)
    _require(type(as_of_unix) is int and as_of_unix >= 0,
             "BACKUP_CLOCK_INVALID")
    _require(type(envelope) is dict and set(envelope) == {"record", "hmac_sha256"},
             "BACKUP_ENVELOPE_FIELDS")
    _require(type(envelope["hmac_sha256"]) is str
             and HEX.fullmatch(envelope["hmac_sha256"]) is not None,
             "BACKUP_MAC_INVALID")
    record = envelope["record"]
    _record_shape(record)
    raw = _canonical(record)
    _require(len(raw) <= MAX_RECORD_BYTES, "BACKUP_RECORD_TOO_LARGE")
    expected_mac = hmac.new(trust_key, DOMAIN_SEPARATOR + raw,
                            hashlib.sha256).hexdigest()
    _require(hmac.compare_digest(expected_mac, envelope["hmac_sha256"]),
             "BACKUP_MAC_VERIFICATION_FAILED")
    _require(record["tip_manifest_sha256"] == expected_tip_sha256
             and record["identity"] == dict(identity),
             "BACKUP_CHECKPOINT_IDENTITY_DRIFT")
    _require(record["backup_domain_claim"] == expected_backup_domain_claim,
             "BACKUP_DOMAIN_CLAIM_DRIFT")
    _require(record["issued_unix"] <= as_of_unix < record["expires_unix"],
             "BACKUP_ENVELOPE_EXPIRED_OR_NOT_YET_VALID")
    observed, verified = _inventory(Path(directory), expected_tip_sha256,
                                    identity, max_total_bytes)
    _require(observed == record["objects"], "BACKUP_INVENTORY_CHANGED")
    return {
        "schema": "ttt-hybrid.backup-custody-verification.v1",
        "status": "HASH_AND_TEST_KEY_VERIFIED_ONLY",
        "tip_manifest_sha256": expected_tip_sha256,
        "checkpoint_steps": verified["steps"],
        "checkpoint_consumed_examples": verified["consumed_examples"],
        "verified_objects": len(observed),
        "hmac_verified_against_supplied_key": True,
        "key_owner_authenticated": False,
        "independent_failure_domain_verified": False,
        "physical_loss_recovery_verified": False,
        "actual_retention_sla_verified": False,
        "checkpoint_promoted": False,
        "training_activated": False,
        "science_qualified": False,
    }
