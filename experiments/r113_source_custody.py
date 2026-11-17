"""R113 prospective two-stage causal mechanism source and authority custody."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments import r112_portable_custody as r112

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/r113_structured_probe_freeze.json"
FREEZE_BLOB = "53d9250db550019ebfc01e34685021927af93536"
SCHEMA = "ttt-hybrid.r113.structured-compositional-active-causal-intervention.unit-only-prospective-freeze.v1"
SEEDS = (1130011,1130023,1130037)


def audit_freeze():
    # Preserve all earlier FIT/VALID and scored R111 custody invariants.
    r112.audit_freeze()
    raw=FREEZE.read_bytes()
    canonical=raw.replace(b"\r\n",b"\n")
    if b"\r" in canonical:
        raise RuntimeError("R113_NONCANONICAL_SOURCE_FREEZE")
    sha=hashlib.sha1(b"blob "+str(len(canonical)).encode()+b"\0"+canonical).hexdigest()
    if sha!=FREEZE_BLOB:
        raise RuntimeError("R113_PREIMPLEMENTATION_GIT_BLOB_FREEZE_DRIFT")
    f=json.loads(raw)
    src=f["source"]; training=f["training"]; ev=f["evaluation"]
    pairs=src["unseen_compositions"]
    operators=src["operators"]
    if (f["schema"]!=SCHEMA
        or f["stage"]!="PREIMPLEMENTATION_FROZEN_UNIT_ONLY_DEV_NO_PROTECTED_FIRST_USE"
        or tuple(src["seeds"])!=SEEDS
        or operators!=["AND","OR","XOR","NAND","NOR","XNOR"]
        or len(pairs)!=8 or len(set(tuple(p) for p in pairs))!=8
        or not all(len(p)==2 and all(o in operators for o in p) for p in pairs)
        or src["unseen_source_generated_only_after_every_trainable_arm_512_updates"] is not True
        or src["no_unseen_world_generation_in_tests"] is not True
        or src["no_spent_R111_original_unseen_replay"] is not True
        or src["worlds_per_seed"]!={"UNIT_FIT":176,"UNIT_VALID":88,"UNIT_UNSEEN":128}
        or src["chosen_action_budget_per_phase"]!=6
        or src["candidate_actions_per_step"]!=2
        or src["outcome_flip_probability"]!=.15
        or training["AdamW_updates_each"]!=512 or training["batch_size"]!=16
        or training["trainable_arms"]!=3
        or ev["protected_scientific_pass_from_unit_only"] is not False
        or any(x is not False for x in f["authority"].values())):
        raise PermissionError("R113_SOURCE_NOVELTY_OR_AUTHORITY_DRIFT")
    return f


def guard_panel(panel):
    audit_freeze()
    if panel not in ("UNIT_FIT","UNIT_VALID","UNIT_UNSEEN"):
        raise PermissionError("R113_UNKNOWN_ORIGINAL_PROTECTED_SOURCE_DENIED")
    if panel=="UNIT_UNSEEN":
        # Never allow public helpers/tests to open unseen. The sole evaluator
        # passes a private module-local proof token, created only after all
        # three trained arms completed 512 FIT updates.
        raise PermissionError("R113_UNSEEN_PRIVATE_EVALUATOR_ONLY")
    return panel
