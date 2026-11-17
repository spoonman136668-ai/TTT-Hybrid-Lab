"""R115 source novelty/authority verification before original UNIT_ONLY generation."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from experiments import r114_source_custody as r114

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r115_query_targeted_causal_freeze.json"
FREEZE_BLOB="1c67231758eeb85644f5844f0c940f412ccc3cce"
SCHEMA="ttt-hybrid.r115.query-targeted-four-factor-causal-voi.unit-only-preimplementation-freeze.v1"
SEEDS=(1150011,1150023,1150037)

def audit_freeze():
    r114.audit_freeze()
    raw=FREEZE.read_bytes()
    canonical=raw.replace(b"\r\n",b"\n")
    if b"\r" in canonical:
        raise RuntimeError("R115_NONCANONICAL_FREEZE_EOL")
    actual=hashlib.sha1(b"blob "+str(len(canonical)).encode()+b"\0"+canonical).hexdigest()
    if actual!=FREEZE_BLOB:
        raise RuntimeError("R115_PREIMPLEMENTATION_GIT_BLOB_DRIFT")
    f=json.loads(raw)
    src=f["source"];tr=f["training"];ev=f["evaluation"];authority=f["authority"]
    if (f["schema"]!=SCHEMA or
        f["stage"]!="PROSPECTIVE_FROZEN_BEFORE_IMPLEMENTATION_UNIT_ONLY_DEV_NO_PROTECTED_FIRST_USE" or
        tuple(src["seeds"])!=SEEDS or
        f["lineage"]["parent_R114_exact_sha"]!="c6da381aa584625fe4904d99951c7be17b6ae579" or
        len(src["unseen_triples"])!=8 or
        len({tuple(x) for x in src["unseen_triples"]})!=8 or
        src["worlds_per_seed"]!={"UNIT_FIT":256,"UNIT_VALID":96,"UNIT_UNSEEN":128} or
        src["chosen_probes_per_phase"]!=5 or
        src["total_chosen_probes_per_world"]!=10 or
        src["candidate_actions_per_step"]!=2 or
        src["query_public_before_support"] is not True or
        src["query_true_outcome_hidden_until_all_probes"] is not True or
        src["unseen_generation_forbidden_until_every_trainable_arm_512_steps_complete"] is not True or
        src["tests_may_generate_only_FIT_and_VALID"] is not True or
        tr["updates_per_arm"]!=512 or tr["batch_size"]!=16 or
        tr["FIT_training_examples_per_seed"]!=2560 or tr["teacher_target_scale"]!=20 or
        ev["protected_scientific_pass_from_unit_only"] is not False or
        any(x is not False for x in authority.values())):
        raise PermissionError("R115_FROZEN_SOURCE_OR_AUTHORITY_DRIFT")
    return f

def guard_public_panel(panel):
    audit_freeze()
    if panel not in ("UNIT_FIT","UNIT_VALID"):
        raise PermissionError("R115_SCORING_SOURCE_PROTECTED_ONLY_AFTER_TRAINING")
    return panel
