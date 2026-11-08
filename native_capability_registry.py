"""Inactive, content-addressed capability policy and champion/challenger review.

Scores and receipt digests are *untrusted assertions*. This module neither
trains, authenticates evidence, chooses an accepted checkpoint, nor promotes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

SCHEMA = "ttt-hybrid.capability-registry.v1"
REVIEW_SCHEMA = "ttt-hybrid.challenger-review.v1"
HEX = re.compile(r"^[0-9a-f]{64}$")
NAME = re.compile(r"^[a-z][a-z0-9_.-]{1,79}$")
MAX_CAPABILITIES = 128
MAX_BYTES = 65536


class RegistryRejected(ValueError):
    pass


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise RegistryRejected(reason)


def _hex(value: object) -> bool:
    return type(value) is str and HEX.fullmatch(value) is not None


def _number(value: object, *, nonnegative: bool = False) -> bool:
    return (type(value) in (int, float) and math.isfinite(value)
            and (not nonnegative or value >= 0))


def _canonical(record: dict) -> bytes:
    return (json.dumps(record, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def validate_policy(policy: dict) -> None:
    _require(type(policy) is dict and set(policy) == {
        "schema", "champion_checkpoint_sha256", "protocol_sha256",
        "harness_sha256", "evaluation_split_sha256", "capabilities", "budget"
    }, "REGISTRY_POLICY_FIELDS")
    _require(policy["schema"] == SCHEMA, "REGISTRY_POLICY_SCHEMA")
    _require(all(_hex(policy[k]) for k in (
        "champion_checkpoint_sha256", "protocol_sha256", "harness_sha256",
        "evaluation_split_sha256")), "REGISTRY_POLICY_FINGERPRINT")
    specs = policy["capabilities"]
    _require(type(specs) is list and 0 < len(specs) <= MAX_CAPABILITIES,
             "REGISTRY_CAPABILITY_COUNT")
    ids: set[str] = set()
    for spec in specs:
        _require(type(spec) is dict and set(spec) == {
            "id", "direction", "floor", "retention_tolerance",
            "meaningful_gain", "minimum_samples"
        }, "REGISTRY_CAPABILITY_FIELDS")
        ident = spec["id"]
        _require(type(ident) is str and NAME.fullmatch(ident) is not None and ident not in ids,
                 "REGISTRY_CAPABILITY_ID")
        ids.add(ident)
        _require(spec["direction"] in ("higher", "lower"), "REGISTRY_DIRECTION")
        _require(_number(spec["floor"]), "REGISTRY_FLOOR")
        _require(_number(spec["retention_tolerance"], nonnegative=True)
                 and _number(spec["meaningful_gain"], nonnegative=True)
                 and spec["meaningful_gain"] > spec["retention_tolerance"],
                 "REGISTRY_GAIN_MUST_EXCEED_TOLERANCE")
        _require(type(spec["minimum_samples"]) is int
                 and spec["minimum_samples"] > 0, "REGISTRY_MINIMUM_SAMPLES")
    _require(type(policy["budget"]) is dict and set(policy["budget"]) == {
        "max_wall_seconds", "max_gpu_seconds", "max_resident_bytes"
    }, "REGISTRY_BUDGET_FIELDS")
    b = policy["budget"]
    _require(_number(b["max_wall_seconds"]) and b["max_wall_seconds"] > 0
             and _number(b["max_gpu_seconds"], nonnegative=True)
             and type(b["max_resident_bytes"]) is int and b["max_resident_bytes"] > 0,
             "REGISTRY_BUDGET_INVALID")


def freeze_policy(directory: str | Path, policy: dict) -> str:
    """Publish an immutable, content-addressed candidate policy, not an accepted ref."""
    validate_policy(policy)
    raw = _canonical(policy)
    _require(len(raw) <= MAX_BYTES, "REGISTRY_TOO_LARGE")
    digest = hashlib.sha256(raw).hexdigest()
    directory = Path(directory)
    _require(directory.is_dir(), "REGISTRY_DIRECTORY_NOT_READY")
    dest = directory / (digest + ".json")
    fd, tmp = tempfile.mkstemp(prefix=".registry-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(tmp, dest)
        except FileExistsError:
            _require(dest.read_bytes() == raw, "REGISTRY_IMMUTABLE_COLLISION")
    finally:
        os.unlink(tmp)
    return digest


def read_policy(directory: str | Path, digest: str) -> dict:
    _require(_hex(digest), "REGISTRY_DIGEST_INVALID")
    raw = (Path(directory) / (digest + ".json")).read_bytes()
    _require(len(raw) <= MAX_BYTES and hashlib.sha256(raw).hexdigest() == digest,
             "REGISTRY_POLICY_TAMPERED")
    try:
        policy = json.loads(raw)
        validate_policy(policy)
    except (UnicodeError, ValueError, TypeError, KeyError) as exc:
        raise RegistryRejected("REGISTRY_POLICY_INVALID") from exc
    _require(_canonical(policy) == raw, "REGISTRY_NONCANONICAL")
    return policy


def review_challenger(policy: dict, policy_digest: str, receipt: dict) -> dict:
    """Compare claims against frozen controls; never authorize promotion or training.

    The caller MUST separately authenticate artifact bytes, owner, evaluation
    origin, checkpoint ancestry and frozen scientific qualifications.
    """
    validate_policy(policy)
    _require(_hex(policy_digest) and hashlib.sha256(_canonical(policy)).hexdigest()
             == policy_digest, "REVIEW_POLICY_DIGEST_MISMATCH")
    _require(type(receipt) is dict and set(receipt) == {
        "schema", "policy_sha256", "champion_checkpoint_sha256",
        "challenger_checkpoint_sha256", "protocol_sha256", "harness_sha256",
        "evaluation_split_sha256", "measurements", "cost"
    }, "REVIEW_RECEIPT_FIELDS")
    _require(receipt["schema"] == REVIEW_SCHEMA
             and receipt["policy_sha256"] == policy_digest,
             "REVIEW_SCHEMA_OR_POLICY")
    _require(all(receipt[k] == policy[k] for k in (
        "champion_checkpoint_sha256", "protocol_sha256", "harness_sha256",
        "evaluation_split_sha256")), "REVIEW_IDENTITY_DRIFT")
    _require(_hex(receipt["challenger_checkpoint_sha256"])
             and receipt["challenger_checkpoint_sha256"] != policy["champion_checkpoint_sha256"],
             "REVIEW_NO_DISTINCT_CHALLENGER")
    values = receipt["measurements"]
    _require(type(values) is list and len(values) == len(policy["capabilities"]),
             "REVIEW_INCOMPLETE_CAPABILITIES")
    _require(type(receipt["cost"]) is dict and set(receipt["cost"]) == {
        "wall_seconds", "gpu_seconds", "resident_bytes"
    }, "REVIEW_COST_FIELDS")
    cost = receipt["cost"]
    _require(_number(cost["wall_seconds"], nonnegative=True)
             and _number(cost["gpu_seconds"], nonnegative=True)
             and type(cost["resident_bytes"]) is int and cost["resident_bytes"] >= 0,
             "REVIEW_COST_INVALID")
    seen: dict[str, dict] = {}
    proof_digests: set[str] = set()
    for item in values:
        _require(type(item) is dict and set(item) == {
            "id", "champion", "challenger", "samples", "evidence_sha256"
        }, "REVIEW_MEASUREMENT_FIELDS")
        ident = item["id"]
        _require(type(ident) is str and ident not in seen, "REVIEW_DUPLICATE_CAPABILITY")
        _require(_number(item["champion"]) and _number(item["challenger"])
                 and type(item["samples"]) is int and item["samples"] > 0
                 and _hex(item["evidence_sha256"])
                 and item["evidence_sha256"] not in proof_digests,
                 "REVIEW_MEASUREMENT_INVALID")
        seen[ident] = item
        proof_digests.add(item["evidence_sha256"])
    _require(set(seen) == {s["id"] for s in policy["capabilities"]},
             "REVIEW_CAPABILITY_SET_CHANGED")
    reasons: list[str] = []
    gains: list[str] = []
    for spec in policy["capabilities"]:
        item = seen[spec["id"]]
        ident = spec["id"]
        if item["samples"] < spec["minimum_samples"]:
            reasons.append(ident + ":INSUFFICIENT_SAMPLES")
        direction = 1 if spec["direction"] == "higher" else -1
        if direction * (item["champion"] - spec["floor"]) < 0:
            reasons.append(ident + ":INCOMPETENT_CHAMPION")
        if direction * (item["challenger"] - spec["floor"]) < 0:
            reasons.append(ident + ":CHALLENGER_FLOOR_FAIL")
        delta = direction * (item["challenger"] - item["champion"])
        if delta < -spec["retention_tolerance"]:
            reasons.append(ident + ":RETENTION_REGRESSION")
        if delta >= spec["meaningful_gain"]:
            gains.append(ident)
    for name, limit in (("wall_seconds", "max_wall_seconds"),
                        ("gpu_seconds", "max_gpu_seconds"),
                        ("resident_bytes", "max_resident_bytes")):
        if cost[name] > policy["budget"][limit]:
            reasons.append(name + ":RESOURCE_BUDGET_EXCEEDED")
    if not gains:
        reasons.append("NO_PREDECLARED_MEANINGFUL_GAIN")
    return {
        "schema": "ttt-hybrid.capability-review-result.v1",
        "status": "BLOCKED" if reasons else "CANDIDATE_REQUIRES_INDEPENDENT_QUALIFICATION",
        "reasons": reasons, "claimed_gains": gains,
        "policy_sha256": policy_digest,
        "champion_checkpoint_sha256": policy["champion_checkpoint_sha256"],
        "challenger_checkpoint_sha256": receipt["challenger_checkpoint_sha256"],
        "evidence_authenticated": False, "scientifically_accepted": False,
        "training_authorized": False, "promoted": False,
    }
