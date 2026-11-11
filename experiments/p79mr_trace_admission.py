"""P79-MR independent trace *schema* admission; science first-use remains prohibited.

Read-only, pure, offline validation. Neither provenance labels nor checksums
prove independence: external evidence needs a separately authorized witness.
No dataset download, environment actuation, model inference or scoring here.
"""
import hashlib
import json
import math
import re

SCHEMA = "ttt-hybrid.p79mr.trace.v1"
KINDS = frozenset(("toy_engineering", "external_simulator", "physical"))
GROUPS = frozenset(("changed", "stable", "unknown"))
VEC_DIM = 16
MAX_WORLDS = 256
MAX_EVENTS = 4096
MAX_PROBES = 256


def canonical_sha256(value):
    """Content identity only: not a signature or admission proof."""
    try:
        raw = json.dumps(value, ensure_ascii=True, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("P79MR_UNSERIALIZABLE") from exc
    return hashlib.sha256(raw).hexdigest()


def _keys(value, required, error):
    if not isinstance(value, dict) or set(value) != set(required):
        raise ValueError(error)


def _name(value, error):
    if (not isinstance(value, str) or not 1 <= len(value) <= 128 or
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/@-]*", value)):
        raise ValueError(error)


def _vec(value, name, *, action=False):
    if not isinstance(value, (tuple, list)) or len(value) != VEC_DIM:
        raise ValueError("P79MR_BAD_" + name)
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ValueError("P79MR_BAD_" + name)
    norm = math.sqrt(math.fsum(float(x) * float(x) for x in value))
    if action and abs(norm-1.0) > 1e-4:
        raise ValueError("P79MR_ACTION_NOT_UNIT")
    return tuple(float(x) for x in value)


def validate_trace(manifest, worlds):
    """Validate custody SHAPE and digest, not independence or scientific rights.

    Source "kind" and "producer_id" are untrusted labels. A valid manifest is
    always ENGINEERING_SCHEMA_ONLY; it is never automatic project admission.
    """
    _keys(manifest, ("schema", "source_kind", "source_id", "producer_id",
                     "custody_ref", "expected_sha256"), "P79MR_BAD_MANIFEST")
    if manifest["schema"] != SCHEMA or manifest["source_kind"] not in KINDS:
        raise ValueError("P79MR_BAD_MANIFEST")
    for name in ("source_id", "producer_id", "custody_ref"):
        _name(manifest[name], "P79MR_BAD_MANIFEST")
    checksum = manifest["expected_sha256"]
    if (not isinstance(checksum, str) or
            not re.fullmatch("[0-9a-f]{64}", checksum)):
        raise ValueError("P79MR_BAD_DIGEST")
    if not isinstance(worlds, list) or not 1 <= len(worlds) <= MAX_WORLDS:
        raise ValueError("P79MR_BAD_WORLDS")
    seen_worlds = set()
    n_probes = 0
    for world in worlds:
        _keys(world, ("world_id", "events", "probes"), "P79MR_BAD_WORLD")
        _name(world["world_id"], "P79MR_BAD_WORLD_ID")
        if world["world_id"] in seen_worlds:
            raise ValueError("P79MR_DUPLICATE_WORLD")
        seen_worlds.add(world["world_id"])
        events, probes = world["events"], world["probes"]
        if not isinstance(events, list) or not 1 <= len(events) <= MAX_EVENTS:
            raise ValueError("P79MR_BAD_EVENTS")
        if not isinstance(probes, list) or not 1 <= len(probes) <= MAX_PROBES:
            raise ValueError("P79MR_BAD_PROBES")
        for index, event in enumerate(events):
            _keys(event, ("event_id", "action", "effect", "receipt_ref"),
                  "P79MR_BAD_EVENT")
            if type(event["event_id"]) is not int or event["event_id"] != index+1:
                raise ValueError("P79MR_EVENT_ORDER")
            _vec(event["action"], "ACTION", action=True)
            _vec(event["effect"], "EFFECT")
            _name(event["receipt_ref"], "P79MR_MISSING_EVENT_CUSTODY")
        seen_probes = set()
        for probe in probes:
            _keys(probe, ("probe_id", "after_events", "action",
                          "realized_effect", "group", "receipt_ref"),
                  "P79MR_BAD_PROBE")
            _name(probe["probe_id"], "P79MR_BAD_PROBE_ID")
            if probe["probe_id"] in seen_probes:
                raise ValueError("P79MR_DUPLICATE_PROBE")
            seen_probes.add(probe["probe_id"])
            cut = probe["after_events"]
            if type(cut) is not int or not 1 <= cut <= len(events):
                raise ValueError("P79MR_BAD_PROBE_CUT")
            if probe["group"] not in GROUPS:
                raise ValueError("P79MR_BAD_GROUP")
            _vec(probe["action"], "ACTION", action=True)
            _vec(probe["realized_effect"], "EFFECT")
            _name(probe["receipt_ref"], "P79MR_MISSING_PROBE_CUSTODY")
        n_probes += len(probes)
    digest = canonical_sha256(worlds)
    if digest != checksum:
        raise ValueError("P79MR_DIGEST_MISMATCH")
    return {
        "schema": SCHEMA,
        "worlds": len(worlds),
        "probes": n_probes,
        "worlds_sha256": digest,
        "source_kind": manifest["source_kind"],
        "admission": "SCHEMA_PASS_SCIENCE_UNVERIFIED",
        "external_first_use": "NOT_AUTHORIZED",
        "scientific_pass": False,
    }


def toy_manifest(worlds, *, source_id="toy-test-001"):
    """Only for in-memory engineering fixtures, never external provenance."""
    _name(source_id, "P79MR_BAD_MANIFEST")
    return {
        "schema": SCHEMA,
        "source_kind": "toy_engineering",
        "source_id": source_id,
        "producer_id": "unit-test",
        "custody_ref": "toy-fixture:in-memory",
        "expected_sha256": canonical_sha256(worlds),
    }
