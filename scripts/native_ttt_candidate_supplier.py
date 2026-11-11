"""Private project-owned, once-only native Hybrid source proposal generator.

This module never sends private code to public FORGE, runs inference, opens an
owner ticket, schedules work, changes the accepted source or authorizes science.
Independent private Git/tree/blob, original history, signed owner intent and
source-owned GPU/CPU qualification remain external requirements.
"""
from __future__ import annotations

import hashlib
import json
import re

LANE = "TTT-Hybrid"
REPO = "spoonman136668-ai/TTT-Hybrid-Lab"
PARENT = "6beeb1edd53606bc2aedad48022ec70192df4b43"
PATH = "scripts/native_ttt_write_admission.py"
BASE_BLOB = "49d5647408ae9c75b5aa2d2912e47215d0da285c"
SCHEMA = "hybrid.private.native-candidate-source.v1"
MECHANISM = "prior-observation-contradiction-ttt-admission-v1"
ANCHOR = (
    '    if mode == "all-write":\n'
    '        return True\n'
    '    raise ValueError("HYBRID_GATE_MODE_UNQUALIFIED")\n'
)
REPLACEMENT = (
    '    if mode == "all-write":\n'
    '        return True\n'
    '    if mode == "observed-contradiction":\n'
    '        # Only evidence observed before this forward call is eligible.\n'
    '        return confidence >= 0.75 and error >= 0.25\n'
    '    raise ValueError("HYBRID_GATE_MODE_UNQUALIFIED")\n'
)
SHA64 = re.compile(r"[a-f0-9]{64}\Z")


class PrivateCandidateDenied(ValueError):
    pass


def require(value: bool, code: str) -> None:
    if not value:
        raise PrivateCandidateDenied("HYBRID_PRIVATE_SUPPLIER_" + code)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(data: dict) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def git_blob(raw: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
    ).hexdigest()


def mechanism_id() -> str:
    return sha(canonical({
        "schema": SCHEMA, "parent": PARENT, "path": PATH,
        "mechanism": MECHANISM, "replacement_sha256": sha(REPLACEMENT.encode()),
    }))


def propose_private(*, parent_commit: str, source_path: str,
                    original_bytes: bytes, original_blob: str,
                    original_sha256: str, original_owner_history_sha256: str,
                    consumed_mechanisms: set[str],
                    consumed_candidate_sha256: set[str],
                    scientific_first_use: bool = False) -> dict:
    """Return private source and a distinct *public-safe* hash-only identity.

    Never publish the entire result into public FORGE; its 'private_source'
    member belongs to the project-owned private source compartment alone.
    """
    require(parent_commit == PARENT and source_path == PATH, "LINEAGE_DRIFT")
    require(type(original_bytes) is bytes
            and 100 <= len(original_bytes) <= 32768
            and original_blob == BASE_BLOB
            and git_blob(original_bytes) == original_blob
            and type(original_sha256) is str
            and sha(original_bytes) == original_sha256
            and type(original_owner_history_sha256) is str
            and SHA64.fullmatch(original_owner_history_sha256) is not None
            and type(consumed_mechanisms) is set
            and type(consumed_candidate_sha256) is set
            and len(consumed_mechanisms) <= 10000
            and len(consumed_candidate_sha256) <= 10000
            and all(type(x) is str and SHA64.fullmatch(x)
                    for x in (consumed_mechanisms | consumed_candidate_sha256))
            and scientific_first_use is False, "ORIGIN_OR_AUTHORITY_DRIFT")
    require(mechanism_id() not in consumed_mechanisms,
            "MECHANISM_ALREADY_CONSUMED")
    try:
        before = original_bytes.decode("utf-8", "strict")
    except UnicodeError as exc:
        raise PrivateCandidateDenied("HYBRID_PRIVATE_SUPPLIER_INVALID_UTF8") from exc
    require(before.count(ANCHOR) == 1
            and "def choose_ttt_update(" in before
            and "def forward_existing_hybrid(" in before
            and "previous_observed_error" in before
            and "previous_evidence_confidence" in before,
            "BASE_SOURCE_DRIFT")
    after = before.replace(ANCHOR, REPLACEMENT, 1).encode("utf-8")
    require(after != original_bytes and len(after) <= 32768,
            "OUTPUT_NOT_BOUNDED")
    after_sha = sha(after)
    require(after_sha not in consumed_candidate_sha256,
            "SOURCE_ALREADY_CONSUMED")
    # This can be independently rebound into the existing private reference
    # contract after the original signed worker/source gates are qualified.
    public_identity = {
        "schema": "hybrid.private.candidate-identity.v1",
        "lane": LANE, "repository": REPO, "source_parent_sha": PARENT,
        "source_path": PATH, "before_sha256": original_sha256,
        "after_sha256": after_sha, "mechanism_sha256": mechanism_id(),
        "history_sha256": original_owner_history_sha256,
        "private_provider_authenticated": False,
        "owner_signed": False, "original_worker_qualified": False,
        "scientific_first_use": False, "deployment_authorized": False,
    }
    return {
        "private_source": after,
        "public_identity": public_identity,
        "public_identity_sha256": sha(canonical(public_identity)),
        "candidate_sha256": after_sha,
        "candidate_kind": "observed-causal-contradiction-sequence-write",
        "independent_source_authentication_required": True,
        "existing_owner_only": True,
        "private_only": True,
    }
