"""R111 immutable preimplementation source/provenance admission; read only, CPU UNIT_ONLY."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r111_active_probe_freeze.json"
FREEZE_BLOB="6864909e0b783c8cc2779b1cea9123ae1ee9d2cc"
SCHEMA="ttt-hybrid.r111.active-physical-causal-probes.unit-only-preimplementation-freeze.v1"
SEEDS=(1011011,1011023,1011037)


def audit_freeze():
    raw=FREEZE.read_bytes()
    canonical=raw.replace(b"\r\n",b"\n")
    if b"\r" in canonical:
        raise RuntimeError("R111_FREEZE_NONCANONICAL_EOL")
    actual=hashlib.sha1(b"blob "+str(len(canonical)).encode()+b"\0"+canonical).hexdigest()
    if actual!=FREEZE_BLOB:
        raise RuntimeError("R111_PREIMPLEMENTATION_FREEZE_BLOB_DRIFT")
    f=json.loads(raw)
    src=f["source"];auth=f["authority"]
    train=[int(x,16) for x in src["train_masks_hex"]]
    unseen=[int(x,16) for x in src["fresh_unseen_masks_hex"]]
    spent=[int(x,16) for v in src["spent_prior_unseen_masks_hex"].values() for x in v]
    if (f["schema"]!=SCHEMA
        or f["stage"]!="PROSPECTIVE_FROZEN_BEFORE_IMPLEMENTATION_UNIT_ONLY_DEV_NO_PROTECTED_FIRST_USE"
        or tuple(src["seeds"])!=SEEDS
        or len(unseen)!=8 or len(set(unseen))!=8
        or set(unseen)&set(train+spent)
        or min((a^b).bit_count() for a in unseen for b in train)<2
        or min((a^b).bit_count() for i,a in enumerate(unseen)
               for j,b in enumerate(unseen) if i!=j)<4
        or src["support_probe_budget_per_phase"]!=6
        or src["candidate_actions_at_each_step"]!=2
        or src["total_chosen_physical_probes_per_arm_per_world"]!=12
        or src["no_unseen_world_or_queries_until_all_trainable_models_complete"] is not True
        or f["training"]["fit_ONLY_training_gradients"] is not True
        or f["training"]["no_training_updates_on_UNIT_UNSEEN"] is not True
        or f["evaluation"]["original_protected_scientific_pass_from_unit_only"] is not False
        or any(v is not False for v in auth.values())):
        raise PermissionError("R111_FROZEN_SOURCE_NOVELTY_OR_AUTHORITY_DRIFT")
    return {"freeze_blob":actual,"seeds":list(SEEDS),
            "novel_masks":[f"{x:02x}" for x in unseen],
            "original_protected_first_use":False,"scientific_pass":False}
