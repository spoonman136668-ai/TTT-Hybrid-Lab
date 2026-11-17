"""R108 development-only public two-context memory CUSTODY BASELINE.

This deterministic eight-address, two-bank factual-only reference is NOT the
trainable hybrid candidate, original science, a reward predictor benchmark,
or evidence of generalization. It supplies falsifiable invariants for the
future frozen R108 experiment: correction must not clobber earlier context,
no private law is read, and memory survives exact cold serialization.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/r108_evidence_routed_memory_freeze.json"
FREEZE_BLOB = "04d31d52df674c72dfc4abaac2f50d81312fd340"
SCHEMA = "ttt-hybrid.r108.evidence-routed-memory.unit-only-preimplementation-freeze.v1"
STATE_SCHEMA = "ttt-hybrid.r108.public-factual-phase-memory.unit-baseline.v1"
EXPECTED_SEEDS = (1008011, 1008023, 1008037)


def audit_freeze():
    """Fail closed on exact frozen Git bytes and previously unspent UNIT panels."""
    raw = FREEZE.read_bytes()
    canonical = raw.replace(b"\r\n", b"\n")
    if b"\r" in canonical:
        raise RuntimeError("R108_FREEZE_NONCANONICAL_LINE_ENDING")
    actual = hashlib.sha1(b"blob " + str(len(canonical)).encode()
                          + b"\0" + canonical).hexdigest()
    if actual != FREEZE_BLOB:
        raise RuntimeError("R108_PREIMPLEMENTATION_FREEZE_BLOB_DRIFT")
    f = json.loads(raw)
    source = f["source"]
    seen = [int(x, 16) for x in source["unseen_heldout_masks_hex"]]
    trained = [int(x, 16) for x in source["known_fit_masks_hex"]]
    prior = [int(x, 16) for masks in
             source["prior_disjoint_unseen_masks_hex"].values()
             for x in masks]
    ham = lambda a, b: (a ^ b).bit_count()
    authority = f["authority"]
    if (f["schema"] != SCHEMA
        or f["stage"] != "PROSPECTIVE_FROZEN_BEFORE_IMPLEMENTATION_UNIT_ONLY_DEV_NO_PROTECTED_FIRST_USE"
        or tuple(source["seeds"]) != EXPECTED_SEEDS
        or len(seen) != 8 or len(set(seen)) != 8
        or set(seen) & (set(trained) | set(prior))
        or min(ham(a, b) for a in seen for b in trained) < 2
        or min(ham(a, b) for i, a in enumerate(seen)
               for j, b in enumerate(seen) if i != j) < 4
        or authority["protected_original_scientific_first_use"] is not False
        or authority["GPU_training_authorized_by_this_freeze"] is not False
        or authority["merge_authorized_by_this_freeze"] is not False
        or authority["forge_integration_authorized"] is not False
        or authority["deployment_authorized"] is not False
        or not source["no_unseen_world_generation_until_every_trainable_arm_finishes"]):
        raise PermissionError("R108_SOURCE_NOVELTY_OR_AUTHORITY_DRIFT")
    return {"freeze_blob": actual, "new_seeds": list(EXPECTED_SEEDS),
            "fresh_unseen_masks": [f"{x:02x}" for x in seen],
            "protected_firstuse": False, "scientific_pass": False}


def _address(public_role_factors):
    try:
        factors = np.asarray(public_role_factors, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError("R108_PUBLIC_ROLE_INPUT_INVALID") from exc
    if factors.shape != (3,) or not bool(np.isfinite(factors).all()) or bool((factors == 0).any()):
        raise ValueError("R108_PUBLIC_ROLE_INPUT_INVALID")
    return 4 * int(factors[0] > 0) + 2 * int(factors[1] > 0) + int(factors[2] > 0)


def _phase(public_context):
    if type(public_context) is not int or public_context not in (0, 1):
        raise ValueError("R108_PUBLIC_CONTEXT_REQUIRED")
    return public_context


def _action(actual_chosen_action):
    if type(actual_chosen_action) is not int or actual_chosen_action not in (0, 2):
        raise ValueError("R108_ONLY_ACTUALLY_CHOSEN_PHYSICAL_ACTION")
    return 0 if actual_chosen_action == 0 else 1


class PublicPhaseMemory:
    """Hand-coded, bounded baseline; NEVER score this as learned Hybrid."""

    def __init__(self, update_weight=0.75):
        weight = float(update_weight)
        if not math.isfinite(weight) or not 0.0 < weight <= 1.0:
            raise ValueError("R108_INVALID_PUBLIC_UPDATE_WEIGHT")
        self.update_weight = weight
        self.values = np.zeros((2, 8, 2), dtype=np.float64)
        self.counts = np.zeros((2, 8, 2), dtype=np.int64)

    def observe(self, public_context, public_role_factors, actual_chosen_action,
                actual_factual_return):
        phase = _phase(public_context)
        slot = _address(public_role_factors)
        action = _action(actual_chosen_action)
        try:
            reward = float(actual_factual_return)
        except (TypeError, ValueError) as exc:
            raise ValueError("R108_INVALID_FACTUAL_OUTCOME") from exc
        if not math.isfinite(reward) or abs(reward) > 10.0:
            raise ValueError("R108_INVALID_FACTUAL_OUTCOME")
        old_count = int(self.counts[phase, slot, action])
        if old_count == 0:
            self.values[phase, slot, action] = reward
        else:
            w = self.update_weight
            self.values[phase, slot, action] = (
                (1.0 - w) * self.values[phase, slot, action] + w * reward)
        self.counts[phase, slot, action] += 1

    def estimate(self, public_context, public_role_factors, actual_chosen_action):
        phase = _phase(public_context)
        slot = _address(public_role_factors)
        action = _action(actual_chosen_action)
        if self.counts[phase, slot, action] == 0:
            return None
        return float(self.values[phase, slot, action])

    def snapshot(self):
        payload = {"schema": STATE_SCHEMA, "update_weight": self.update_weight,
                   "values": self.values.tolist(), "counts": self.counts.tolist()}
        canonical = json.dumps(payload, sort_keys=True,
                               separators=(",", ":"), allow_nan=False)
        seal = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return json.dumps({**payload, "sha256": seal}, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)

    @classmethod
    def cold_restore(cls, serialized):
        try:
            obj = json.loads(serialized)
            actual = obj.pop("sha256")
            canonical = json.dumps(obj, sort_keys=True,
                                   separators=(",", ":"), allow_nan=False)
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if (obj["schema"] != STATE_SCHEMA or digest != actual
                or set(obj) != {"schema", "update_weight", "values", "counts"}):
                raise ValueError("R108_COLD_RESTORE_SOURCE_OR_SEAL_INVALID")
            state = cls(obj["update_weight"])
            values = np.asarray(obj["values"], dtype=np.float64)
            counts = np.asarray(obj["counts"])
            if (values.shape != (2, 8, 2) or counts.shape != (2, 8, 2)
                or not bool(np.isfinite(values).all())
                or counts.dtype.kind not in "iu" or bool((counts < 0).any())
                or not bool((values[np.asarray(counts) == 0] == 0).all())):
                raise ValueError("R108_COLD_RESTORE_STATE_INVALID")
            state.values = values
            state.counts = counts.astype(np.int64)
            return state
        except (KeyError, TypeError, json.JSONDecodeError, OverflowError) as exc:
            raise ValueError("R108_COLD_RESTORE_STATE_INVALID") from exc
