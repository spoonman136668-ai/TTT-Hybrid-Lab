"""R120 preregistered native-only canonical FP32 gradient clipping source."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from experiments import r119_source_custody as r119

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r120_canonical_gradient_clip_freeze.json"
FREEZE_BLOB="af6596e9bd40d8212e186dfe57c06f41339b3024"
SCHEMA="ttt-hybrid.r120.canonical-gradient-L2-norm-clip-FIT-VALID.v1"
SEEDS=(1160011,1160023,1160037)
SNAPSHOTS=(0,1,32,128,256,512)

def audit_freeze():
    r119.audit_freeze()
    raw=FREEZE.read_bytes()
    canonical=raw.replace(b"\r\n",b"\n")
    if b"\r" in canonical:raise RuntimeError("R120_FROZEN_EOL_DRIFT")
    blob=hashlib.sha1(b"blob "+str(len(canonical)).encode()+b"\0"+canonical).hexdigest()
    if blob!=FREEZE_BLOB:
        raise RuntimeError("R120_PREIMPLEMENTATION_GIT_BLOB_DRIFT")
    f=json.loads(raw);src=f["source"];cl=src["new_clipping"];ev=f["evaluation"]
    if (f["schema"]!=SCHEMA or
        f["stage"]!="PROSPECTIVE_PREIMPLEMENTATION_FIT_VALID_ENGINEERING_ONLY" or
        f["lineage"]["parent_R119_exact_SHA"]!="9d3547baf8fd0914ce33aa0f4000ff217ade84ea" or
        tuple(src["seeds"])!=SEEDS or
        src["allowed_panels"]!=["UNIT_FIT","UNIT_VALID"] or
        src["forbid_scored_R116_UNIT_UNSEEN"] is not True or
        src["forbid_protected_original_science"] is not True or
        src["public_FIT_samples_per_seed"]!=256 or
        src["public_VALID_worlds_per_seed"]!=96 or
        src["only_native_gradient_L2_clipping_implementation_changed"] is not True or
        src["source_and_teacher_unmodified"] is not True or
        src["trainer"].find("512 updates")<0 or
        cl["epsilon"]!=1e-6 or
        cl["max_norm"]!=1.0 or
        cl["no_norm_threshold_or_hyperparameter_tuning"] is not True or
        src["public_VALID_physical_interventions_per_world"]!=4 or
        src["original_calibration_do"]!=[8,9,7] or
        src["last_action_choice"]!=[12,14] or
        tuple(ev["require_exact_first_step_native_FP32_all_3_seeds"] for _ in [0])!=(True,) or
        ev["scientific_pass_from_FIT_VALID"] is not False or
        any(v is not False for v in f["authority"].values())):
        raise PermissionError("R120_FROZEN_SOURCE_OR_PERMISSION_DRIFT")
    return f

def allow_public(panel):
    audit_freeze()
    if panel not in ("UNIT_FIT","UNIT_VALID"):
        raise PermissionError("R120_SPENT_OR_PROTECTED_CAUSAL_SOURCE_DENIED")
    return panel
