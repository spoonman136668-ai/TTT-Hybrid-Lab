"""Prospective frozen first-update numeric boundary and first-use fence."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from experiments import r118_source_custody as r118

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r119_first_update_numeric_freeze.json"
FREEZE_BLOB="a1529830e8e89f46762c10667abc15e79040f124"
SCHEMA="ttt-hybrid.r119.first-update-FIT-only-numerical-boundary.v1"
SEEDS=(1160011,1160023,1160037)
STAGES=("fit_n_first_batch","fit_m_first_batch","fit_g_first_batch",
        "fit_target_first_batch","fit_minibatch_indices",
        "before_first_update_native_raw_FP32_parameters",
        "forward.linear1_FP32_output","forward.gelu1_FP32_output",
        "forward.linear2_FP32_output","forward.gelu2_FP32_output",
        "forward.linear3_FP32_logits","mse_loss_FP32_scalar",
        "raw_gradient_native_named_parameters","raw_gradient_global_norm_FP32",
        "clip_returned_preclip_norm_FP32",
        "clipped_gradient_native_named_parameters",
        "after_AdamW_step_native_named_parameters",
        "after_AdamW_step_optimizer_exp_avg_by_parameter",
        "after_AdamW_step_optimizer_exp_avg_sq_by_parameter",
        "after_AdamW_step_optimizer_step_by_parameter")

def audit_freeze():
    r118.audit_freeze()
    b=FREEZE.read_bytes()
    canonical=b.replace(b"\r\n",b"\n")
    if b"\r" in canonical:
        raise RuntimeError("R119_FROZEN_TEXT_EOL_DRIFT")
    blob=hashlib.sha1(b"blob "+str(len(canonical)).encode()+b"\0"+canonical).hexdigest()
    if blob!=FREEZE_BLOB:
        raise RuntimeError("R119_PREIMPLEMENTATION_GIT_BLOB_DRIFT")
    f=json.loads(b)
    src=f["inputs"];trace=f["trace"];ev=f["evaluation"]
    if (f["schema"]!=SCHEMA or
        f["stage"]!="PROSPECTIVELY_FROZEN_BEFORE_DIAGNOSTIC_IMPLEMENTATION" or
        f["lineage"]["parent_R118_SHA"]!="eca962608adfb1f409003f112bf30e2edd59e33b" or
        tuple(src["seeds"])!=SEEDS or
        src["panels"]!=["UNIT_FIT"] or
        src["FIT_worlds_per_seed"]!=256 or
        src["no_heldout_scored_data"] is not True or
        src["no_original_R116_UNIT_UNSEEN"] is not True or
        src["optimizer"].startswith("R116 native AdamW lr=.002") is not True or
        trace["no_model_updates_beyond_first_step"] is not True or
        tuple(trace["stages"])!=STAGES or
        ev["scientific_pass"] is not False or
        any(value is not False for value in f["authority"].values())):
        raise PermissionError("R119_FROZEN_SOURCE_AND_AUTHORITY_DRIFT")
    return f

def allow_fit(panel):
    audit_freeze()
    if panel!="UNIT_FIT":
        raise PermissionError("R119_ONLY_PUBLIC_FIT_NO_SPENT_SCIENCE")
    return panel
