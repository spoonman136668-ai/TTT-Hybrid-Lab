"""R122 immutable FIT-only all-512-updates numeric provenance custody."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from experiments import r121_source_custody as r121

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r122_full_fit_gradient_provenance_freeze.json"
FREEZE_BLOB="ebf335b8873cee4109b34644aa3498eeb7791557"
SCHEMA="ttt-hybrid.r122.all-512-public-FIT-native-stable-clip-gradient-boundaries.v1"
SEEDS=(1160011,1160023,1160037)
STAGES=("minibatch_indices_int64","pre_update_named_FP32_weights",
        "forward_linear1","forward_gelu1","forward_linear2",
        "forward_gelu2","forward_linear3_logits","MSE_loss_FP32",
        "raw_named_FP32_gradients","canonical_fsum_squares_F64",
        "canonical_sqrt_norm_F64","canonical_clip_coefficient_FP32",
        "clipped_named_FP32_gradients","post_AdamW_named_FP32_weights",
        "post_AdamW_exp_avg_FP32","post_AdamW_exp_avg_sq_FP32",
        "post_AdamW_step_FP32")
UPDATES=512
WORKERS=("A","B")

def audit_freeze():
    r121.audit_freeze()
    raw=FREEZE.read_bytes()
    canonical=raw.replace(b"\r\n",b"\n")
    if b"\r" in canonical:
        raise RuntimeError("R122_NONCANONICAL_FREEZE_LINE_ENDINGS")
    blob=hashlib.sha1(b"blob "+str(len(canonical)).encode()+b"\0"+canonical).hexdigest()
    if blob!=FREEZE_BLOB:
        raise RuntimeError("R122_PREIMPLEMENTATION_GIT_BLOB_DRIFT")
    f=json.loads(raw)
    i=f["inputs"];c=f["capture"];a=f["acceptance"]
    if (f["schema"]!=SCHEMA or
        f["stage"]!="PREIMPLEMENTATION_FROZEN_ENGINEERING_FIT_ONLY" or
        f["lineage"]["parent_R121_source_sha"]!="7a34405e687878ac05a4489d1603beacb1b04e11" or
        tuple(i["seed_ids"])!=SEEDS or i["allowed_panels"]!=["UNIT_FIT"] or
        i["forbidden_panels"]!=["UNIT_VALID","UNIT_UNSEEN","original_scientific","protected_science"] or
        i["FIT_world_count"]!=256 or
        i["trainable_native_parameter_count"]!=3409 or
        i["num_workers_each_OS"]!=2 or
        i["model_updates_per_seed_each_worker"]!=UPDATES or
        i["number_modes"]!=1 or
        i["mode_name"]!="R120_F64_fsum_clipped_native" or
        i["no_R116_original_scored_UNSEEN_replay"] is not True or
        tuple(c["ordered_step_stages"])!=STAGES or
        c["all_512_updates_all_stages"] is not True or
        c["read_only_hooks_unhooked_negative_control_first_1_update"] is not True or
        a["scientific_pass"] is not False or
        any(v is not False for v in f["authority"].values())):
        raise PermissionError("R122_FROZEN_SOURCE_OR_AUTHORITY_DRIFT")
    return f

def allow(panel):
    audit_freeze()
    if panel!="UNIT_FIT":
        raise PermissionError("R122_ONLY_ORIGINAL_PUBLIC_FIT_ALLOWED")
    return panel
