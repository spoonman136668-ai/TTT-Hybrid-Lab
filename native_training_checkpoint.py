"""Inactive, exact-source native Hybrid training-state checkpoint contract.

No training, first-use, model acceptance, scheduling, or runner integration.
Explicit callers own artifact storage and authorization. Checkpoint manifests
are *local integrity* evidence, never independently authenticated science.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import tempfile
from typing import Mapping

import torch

SCHEMA = "ttt-hybrid.native-training-checkpoint.v1"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_ANCESTORS = 256
IDENTITY_KEYS = ("architecture_sha256", "interface_sha256", "protocol_sha256",
                 "data_split_sha256", "harness_sha256", "environment_sha256")


class CheckpointRejected(ValueError):
    pass


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json(raw: dict) -> bytes:
    return (json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n").encode("utf-8")


def _publish(path: Path, data: bytes) -> None:
    """Immutable same-directory publication; orphan blobs are not commits."""
    fd, tmp = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(tmp, path)  # EEXIST cannot overwrite an accepted artifact.
        except FileExistsError:
            if path.read_bytes() != data:
                raise CheckpointRejected("CHECKPOINT_IMMUTABLE_COLLISION")
        # Do not write mutable 'latest' or 'accepted' pointers here.
    finally:
        os.unlink(tmp)


def _identity(identity: Mapping) -> dict:
    if not isinstance(identity, Mapping) or set(identity) != {"source_sha", *IDENTITY_KEYS}:
        raise CheckpointRejected("CHECKPOINT_IDENTITY_FIELDS")
    if type(identity["source_sha"]) is not str or not HEX40.fullmatch(identity["source_sha"]):
        raise CheckpointRejected("CHECKPOINT_SOURCE_SHA")
    if any(type(identity[k]) is not str or not HEX64.fullmatch(identity[k])
           for k in IDENTITY_KEYS):
        raise CheckpointRejected("CHECKPOINT_FINGERPRINT")
    return dict(identity)


def _state(state: Mapping) -> dict:
    if not isinstance(state, Mapping) or set(state) != {"recurrent", "fast_weights", "history"}:
        raise CheckpointRejected("CHECKPOINT_NATIVE_STATE_INCOMPLETE")
    if any(not isinstance(v, torch.Tensor) or not v.is_floating_point() or
           not bool(torch.isfinite(v).all()) for v in state.values()):
        raise CheckpointRejected("CHECKPOINT_NATIVE_STATE_INVALID")
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def _rng() -> dict:
    data = {"torch_cpu": torch.get_rng_state(), "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}
    # NumPy is optional and its RNG must not silently fall out of a resume.
    try:
        import numpy as np
        n = np.random.get_state()
        data["numpy"] = (n[0], n[1].tolist(), int(n[2]), int(n[3]), float(n[4]))
    except ImportError:
        data["numpy"] = None
    return data


def _restore_rng(data: dict) -> None:
    if not isinstance(data, dict) or set(data) != {"torch_cpu", "python", "cuda", "numpy"}:
        raise CheckpointRejected("CHECKPOINT_RNG_INCOMPLETE")
    if not isinstance(data["torch_cpu"], torch.Tensor) or not isinstance(data["cuda"], list):
        raise CheckpointRejected("CHECKPOINT_RNG_INVALID")
    if data["cuda"] and (not torch.cuda.is_available() or
                         len(data["cuda"]) != torch.cuda.device_count()):
        raise CheckpointRejected("CHECKPOINT_CUDA_RNG_ENVIRONMENT_DRIFT")
    if data["numpy"] is not None:
        try:
            import numpy as np
        except ImportError as exc:
            raise CheckpointRejected("CHECKPOINT_NUMPY_RNG_UNAVAILABLE") from exc
        n = data["numpy"]
        if not (isinstance(n, (tuple, list)) and len(n) == 5):
            raise CheckpointRejected("CHECKPOINT_NUMPY_RNG_INVALID")
    # Do not touch any global RNG until all structure checks are complete.
    random.setstate(data["python"])
    torch.set_rng_state(data["torch_cpu"])
    if data["cuda"]:
        torch.cuda.set_rng_state_all(data["cuda"])
    if data["numpy"] is not None:
        np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))


def _read(directory: Path, manifest_sha256: str) -> tuple[dict, dict]:
    if type(manifest_sha256) is not str or not HEX64.fullmatch(manifest_sha256):
        raise CheckpointRejected("CHECKPOINT_DIGEST_INVALID")
    raw = (directory / (manifest_sha256 + ".json")).read_bytes()
    if _sha(raw) != manifest_sha256 or len(raw) > 8192:
        raise CheckpointRejected("CHECKPOINT_MANIFEST_TAMPERED")
    try:
        manifest = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise CheckpointRejected("CHECKPOINT_MANIFEST_INVALID") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise CheckpointRejected("CHECKPOINT_SCHEMA")
    blob_hash = manifest.get("payload_sha256")
    if type(blob_hash) is not str or not HEX64.fullmatch(blob_hash):
        raise CheckpointRejected("CHECKPOINT_PAYLOAD_HASH_INVALID")
    blob = (directory / (blob_hash + ".pt")).read_bytes()
    if _sha(blob) != blob_hash:
        raise CheckpointRejected("CHECKPOINT_PAYLOAD_TAMPERED")
    try:
        payload = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointRejected("CHECKPOINT_PAYLOAD_UNREADABLE") from exc
    if not isinstance(payload, dict) or set(payload) != {
            "model", "optimizer", "scheduler", "native_state", "sampler", "rng"}:
        raise CheckpointRejected("CHECKPOINT_PAYLOAD_INCOMPLETE")
    _state(payload["native_state"])
    if not isinstance(payload["sampler"], dict) or not isinstance(payload["model"], dict) or not isinstance(payload["optimizer"], dict):
        raise CheckpointRejected("CHECKPOINT_STATE_INVALID")
    if not isinstance(payload["rng"], dict) or set(payload["rng"]) != {"torch_cpu", "python", "cuda", "numpy"}:
        raise CheckpointRejected("CHECKPOINT_RNG_INCOMPLETE")
    return manifest, payload


def validate_chain(directory: str | Path, tip_sha256: str, identity: Mapping) -> dict:
    """Verify a bounded local lineage. Never choose a tip automatically."""
    directory = Path(directory)
    expected = _identity(identity)
    seen = set()
    digest = tip_sha256
    child = None
    tip = None
    for _ in range(MAX_ANCESTORS):
        if digest in seen:
            raise CheckpointRejected("CHECKPOINT_CYCLE")
        seen.add(digest)
        manifest, _ = _read(directory, digest)
        if manifest.get("identity") != expected or type(manifest.get("steps")) is not int or manifest["steps"] < 0 or type(manifest.get("consumed_examples")) is not int or manifest["consumed_examples"] < 0:
            raise CheckpointRejected("CHECKPOINT_IDENTITY_OR_CURSOR_DRIFT")
        if child is not None and (child["steps"] <= manifest["steps"] or child["consumed_examples"] <= manifest["consumed_examples"]):
            raise CheckpointRejected("CHECKPOINT_NONADVANCING_RESTART")
        if tip is None:
            tip = manifest
        parent = manifest.get("parent_manifest_sha256")
        if parent is None:
            if manifest["steps"] != 0 or manifest["consumed_examples"] != 0:
                raise CheckpointRejected("CHECKPOINT_MISSING_TRAINED_ANCESTRY")
            return tip
        if type(parent) is not str or not HEX64.fullmatch(parent):
            raise CheckpointRejected("CHECKPOINT_PARENT_INVALID")
        child = manifest
        digest = parent
    raise CheckpointRejected("CHECKPOINT_LINEAGE_TOO_DEEP")


def save_checkpoint(directory: str | Path, *, model, optimizer, identity: Mapping,
                    native_state: Mapping, sampler_state: dict, steps: int,
                    consumed_examples: int, parent_manifest_sha256: str | None,
                    scheduler=None) -> str:
    """Emit a candidate only, with complete resumable state and no promotion."""
    directory = Path(directory)
    if not directory.is_dir():
        raise CheckpointRejected("CHECKPOINT_DIRECTORY_NOT_READY")
    identity = _identity(identity)
    if not isinstance(sampler_state, dict) or not sampler_state or type(steps) is not int or steps < 0 or type(consumed_examples) is not int or consumed_examples < 0:
        raise CheckpointRejected("CHECKPOINT_CURSOR_INVALID")
    if parent_manifest_sha256 is None:
        if steps != 0 or consumed_examples != 0:
            raise CheckpointRejected("CHECKPOINT_UNDECLARED_RESTART")
    else:
        parent = validate_chain(directory, parent_manifest_sha256, identity)
        if steps <= parent["steps"] or consumed_examples <= parent["consumed_examples"]:
            raise CheckpointRejected("CHECKPOINT_NONADVANCING_RESTART")
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict() if scheduler is not None else None,
               "native_state": _state(native_state), "sampler": sampler_state,
               "rng": _rng()}
    data = io.BytesIO()
    torch.save(payload, data)
    blob = data.getvalue()
    blob_sha = _sha(blob)
    manifest = {"schema": SCHEMA, "identity": identity,
                "payload_sha256": blob_sha,
                "parent_manifest_sha256": parent_manifest_sha256,
                "steps": steps, "consumed_examples": consumed_examples,
                "has_scheduler": scheduler is not None}
    record = _json(manifest)
    manifest_sha = _sha(record)
    _publish(directory / (blob_sha + ".pt"), blob)
    _publish(directory / (manifest_sha + ".json"), record)
    return manifest_sha


def load_checkpoint(directory: str | Path, manifest_sha256: str, *, model, optimizer,
                    identity: Mapping, scheduler=None) -> tuple[dict, dict]:
    """Verify chain and restore exact state; caller resumes sampler at cursor.

    Do not call on a live model mid-step. No automatic fallback to initial weights.
    """
    directory = Path(directory)
    verified = validate_chain(directory, manifest_sha256, identity)
    _, payload = _read(directory, manifest_sha256)
    if verified["has_scheduler"] != (scheduler is not None) or (
            scheduler is not None and not isinstance(payload["scheduler"], dict)):
        raise CheckpointRejected("CHECKPOINT_SCHEDULER_DRIFT")
    if not isinstance(payload["model"], dict) or set(payload["model"]) != set(model.state_dict()):
        raise CheckpointRejected("CHECKPOINT_MODEL_SCHEMA_DRIFT")
    if not isinstance(payload["optimizer"], dict) or not payload["optimizer"].get("param_groups"):
        raise CheckpointRejected("CHECKPOINT_OPTIMIZER_MISSING")
    # These are exact checkpoint values, not new seed/reset initialization.
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    _restore_rng(payload["rng"])
    return {k: v.clone() for k, v in payload["native_state"].items()}, payload["sampler"].copy()
