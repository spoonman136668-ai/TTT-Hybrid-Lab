"""R110 original preimplementation freeze custody; DEV_ONLY, never science authority."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/r110_public_contradiction_credit_freeze.json"
FREEZE_BLOB = "8b334ba1503539bef6665bbff5146ebf6f95f01b"
SCHEMA = "ttt-hybrid.r110.public-contradiction-credit.unit-only-preimplementation-freeze.v1"
EXPECTED_SEEDS = (1010011, 1010023, 1010037)


def audit_freeze():
    raw = FREEZE.read_bytes()
    canonical = raw.replace(b"\r\n", b"\n")
    if b"\r" in canonical:
        raise RuntimeError("R110_FREEZE_NONCANONICAL_LINE_ENDING")
    digest = hashlib.sha1(b"blob " + str(len(canonical)).encode() +
                          b"\0" + canonical).hexdigest()
    if digest != FREEZE_BLOB:
        raise RuntimeError("R110_PREIMPLEMENTATION_FREEZE_BLOB_DRIFT")
    f = json.loads(raw)
    src, auth = f["source"], f["authority"]
    unseen = [int(x, 16) for x in src["unseen_heldout_masks_hex"]]
    train = [int(x, 16) for x in src["known_fit_masks_hex"]]
    prior = [int(x, 16) for panel in
             src["prior_disjoint_unseen_masks_hex"].values() for x in panel]
    if (f["schema"] != SCHEMA
        or f["stage"] !=
            "PROSPECTIVE_FROZEN_BEFORE_IMPLEMENTATION_UNIT_ONLY_DEV_NO_PROTECTED_FIRST_USE"
        or tuple(src["seeds"]) != EXPECTED_SEEDS
        or len(unseen) != 8 or len(set(unseen)) != 8
        or set(unseen) & (set(train) | set(prior))
        or min((a ^ b).bit_count() for a in unseen for b in train) < 2
        or min((a ^ b).bit_count() for i, a in enumerate(unseen)
               for j, b in enumerate(unseen) if i != j) < 4
        or not src["no_unseen_world_generation_until_all_trainable_arms_finish"]
        or any(auth[k] is not False for k in
               ("protected_original_scientific_first_use",
                "GPU_training_authorized", "merge_authorized",
                "forge_integration_authorized", "deployment_authorized"))
        or auth["no_new_runner_controller_scheduler_or_paid_api"] is not True
        or f["evaluation"]["protected_scientific_pass_from_unit_only"] is not False):
        raise PermissionError("R110_FREEZE_NOVELTY_OR_AUTHORITY_DRIFT")
    return {"freeze_blob": digest, "seeds": list(EXPECTED_SEEDS),
            "fresh_unseen_masks": [f"{n:02x}" for n in unseen],
            "protected_firstuse": False, "scientific_pass": False}
