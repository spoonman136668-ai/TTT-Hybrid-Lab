"""R116 prospective evidence/sign-conditional source and first-use custody."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from experiments import r115_source_custody as r115

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r116_feedback_pair_freeze.json"
FREEZE_BLOB="0aeda5baff861abb8ebc10b4d19ae844216bb39d"
SCHEMA="ttt-hybrid.r116.paired-factual-feedback-conditional-probing.unit-only-prospective-freeze.v1"
SEEDS=(1160011,1160023,1160037)

def audit_freeze():
    # Requires earlier R115 scored-source custody without replay.
    r115.audit_freeze()
    raw=FREEZE.read_bytes()
    canonical=raw.replace(b"\r\n",b"\n")
    if b"\r" in canonical:
        raise RuntimeError("R116_FREEZE_EOL_DRIFT")
    git_blob=hashlib.sha1(b"blob "+str(len(canonical)).encode()+b"\0"+canonical).hexdigest()
    if git_blob!=FREEZE_BLOB:
        raise RuntimeError("R116_PROSPECTIVE_GIT_BLOB_DRIFT")
    f=json.loads(raw)
    src=f["source"];t=f["training"];ev=f["evaluation"]
    if (f["schema"]!=SCHEMA or
        f["stage"]!="PROSPECTIVE_PREIMPLEMENTATION_UNIT_ONLY_DEV_NEVER_PROTECTED_SCIENCE" or
        f["lineage"]["base_R115_exact_sha"]!="89adbdb916b9fb21ab4e383cfd2577eaf12f9d77" or
        tuple(src["seeds"])!=SEEDS or
        src["public_calibration_actions"]!=[8,9] or
        src["public_diagnostic_action"]!=7 or
        src["public_next_offered_actions"]!=[12,14] or
        src["query_pattern"]!=15 or
        src["per_world_total_physical_actions"]!=4 or
        src["public_calibration_expected_signs"]!=[1,-1] or
        src["noise_probability"]!=0 or
        src["FIT_worlds_per_seed"]!=256 or
        src["VALID_worlds_per_seed"]!=96 or
        src["UNIT_UNSEEN_worlds_per_seed"]!=128 or
        src["heldout_world_count_per_reserved_mask_per_seed"]!=16 or
        src["never_generate_UNIT_UNSEEN_in_tests"] is not True or
        src["UNIT_UNSEEN_generation_only_after_all_trainable_512_updates"] is not True or
        src["final_query_answer_sealed_until_after_4th_action"] is not True or
        t["updates_per_trainable_arm"]!=512 or
        t["fit_samples_per_seed"]!=256 or
        t["batch_size"]!=16 or
        len(ev["all_3_seed_checks_frozen"])!=15 or
        ev["all_15_checks_required_on_every_seed"] is not True or
        ev["protected_scientific_PASS_from_UNIT_ONLY"] is not False or
        any(x is not False for x in f["authority"].values())):
        raise PermissionError("R116_SOURCE_AUTHORSHIP_OR_FROZEN_AUTHORITY_DRIFT")
    return f

def permit_public_panel(panel):
    audit_freeze()
    if panel not in ("UNIT_FIT","UNIT_VALID"):
        raise PermissionError("R116_SPENT_OR_PROTECTED_FIRSTUSE_DENIED")
    return panel
