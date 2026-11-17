"""Prospectively frozen FIT-only first-step independent-process custody."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from experiments import r120_source_custody as r120

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r121_first_step_repeat_freeze.json"
FREEZE_BLOB="1df5056dab851178d4ae3cd2ad168faec00d1f4d"
SCHEMA="ttt-hybrid.r121.first-FIT-step-independent-process-source-and-grad-repeat.v1"
SEEDS=(1160011,1160023,1160037)
MODES=("torch_original_clip_grad_norm_one","r120_fsum_f64_canonical_clip")
STAGES=("FIT_first_batch_n","FIT_first_batch_m","FIT_first_batch_g",
        "FIT_first_batch_target","FIT_first_batch_indices",
        "initial_native_FP32_named_parameters","forward_linear1",
        "forward_gelu1","forward_linear2","forward_gelu2",
        "forward_linear3_logits","MSE_loss_FP32",
        "all_raw_FP32_named_gradients","global_raw_grad_norm_F64_fsum_squared_IEEE754",
        "original_torch_global_raw_gradient_norm_FP32",
        "actual_clip_coefficient_FP32","all_clipped_FP32_named_gradients",
        "all_post_AdamW_FP32_named_parameters","optimizer_exp_avg_FP32",
        "optimizer_exp_avg_sq_FP32","optimizer_step_scalar")

def audit_freeze():
    r120.audit_freeze()
    raw=FREEZE.read_bytes()
    canonical=raw.replace(b"\r\n",b"\n")
    if b"\r" in canonical:
        raise RuntimeError("R121_FROZEN_NONCANONICAL_TEXT_EOL")
    blob=hashlib.sha1(b"blob "+str(len(canonical)).encode()+b"\0"+canonical).hexdigest()
    if blob!=FREEZE_BLOB:
        raise RuntimeError("R121_PREIMPLEMENTATION_GIT_BLOB_DRIFT")
    f=json.loads(raw);src=f["source"];cap=f["capture"];control=f["controls"];eva=f["evaluation"]
    if (f["schema"]!=SCHEMA or
        f["stage"]!="PROSPECTIVE_PREIMPLEMENTATION_FIT_ONLY_DIAGNOSTIC_NOT_SCIENTIFIC_FIRST_USE" or
        f["lineage"]["parent_R120_SHA"]!="99fcefefc49d38f199b79ffbee456c084a8d1128" or
        tuple(src["seeds"])!=SEEDS or src["allowed_panel"]!="UNIT_FIT" or
        src["FIT_sample_count"]!=256 or
        src["number_independent_processes_per_platform"]!=2 or
        tuple(src["two_clip_methods"])!=MODES or
        src["number_original_adamw_steps_per_process_seed_clip_mode"]!=1 or
        src["original_R116_TEST_UNSEEN_forbidden"] is not True or
        src["protected_science_firstuse_forbidden"] is not True or
        tuple(cap["frozen_stages"])!=STAGES or
        cap["capture_entire_3409_gradients_by_name_as_canonical_FP32_little_endian_base64"] is not True or
        control["each_process_traces_all_three_seeds_and_both_clipping_methods"] is not True or
        control["one_update_only"] is not True or
        eva["scientific_pass"] is not False or
        any(x is not False for x in f["authority"].values())):
        raise PermissionError("R121_SOURCE_AUTHORITY_OR_FREEZE_DRIFT")
    return f

def allow(panel):
    audit_freeze()
    if panel!="UNIT_FIT":
        raise PermissionError("R121_PUBLIC_FIT_ONLY_SPENT_SCIENCE_DENIED")
    return panel
