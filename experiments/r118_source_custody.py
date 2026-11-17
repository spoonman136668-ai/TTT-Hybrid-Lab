"""R118 immutable canonical-initial-state/CPU optimizer source fence."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from experiments import r117_source_custody as r117

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r118_shared_initial_state_freeze.json"
FREEZE_BLOB="567f553bc260183508627714971eee90f55f0176"
SCHEMA="ttt-hybrid.r118.canonical-same-bytes-initialization-FIT-VALID-only.v1"
SEEDS=(1160011,1160023,1160037)
SCALES=(384,256,128,64,32,16,8)
CHECKPOINTS=(0,1,32,128,256,512)

def audit_freeze():
    r117.audit_freeze()
    raw=FREEZE.read_bytes()
    canon=raw.replace(b"\r\n",b"\n")
    if b"\r" in canon: raise RuntimeError("R118_NONCANONICAL_FREEZE_EOL")
    blob=hashlib.sha1(b"blob "+str(len(canon)).encode()+b"\0"+canon).hexdigest()
    if blob!=FREEZE_BLOB:
        raise RuntimeError("R118_PREIMPLEMENTATION_GIT_BLOB_DRIFT")
    f=json.loads(raw);s=f["source"];d=f["diagnostic"];e=f["evaluation"]
    if (
        f["schema"]!=SCHEMA or
        f["stage"]!="PROSPECTIVE_PREIMPLEMENTATION_DIAGNOSTIC_NO_SCIENTIFIC_FIRST_USE" or
        f["lineage"]["parent_R117_exact_SHA"]!="8a0da2dc2f907ce342bcfcd21bf1970e8c44a1c3" or
        tuple(s["seeds"])!=SEEDS or
        s["allowed_panels"]!=["UNIT_FIT","UNIT_VALID"] or
        s["forbidden_panels"]!=["UNIT_UNSEEN","R116_original_spent","original_scientific","protected"] or
        s["exact_FIT_AdamW_updates_per_arm"]!=512 or
        s["exact_minibatch_shape"]!=[512,16] or
        s["trainable_arms"]!=["R116_NATIVE_SIGN_INTEGER","R116_CONVENTIONAL_MLP","R116_CONVENTIONAL_GRU"] or
        s["FIT_examples_per_seed"]!=256 or s["VALID_examples_per_seed"]!=96 or
        s["canonical_init_recipe"]["version"]!="r118.seed-sha256-fixedgrid-per-parameter-v1" or
        s["canonical_init_recipe"]["signed_integer"]!="int.from_bytes(digest[0:2],'little') - 32768" or
        s["canonical_init_recipe"]["model_parameter_value"]!="exact signed_integer / 262144, representable in FP32" or
        s["canonical_initialization_applies_to_all_three_arms"] is not True or
        s["FP32_model_optimizer_unchanged_from_R116"] is not True or
        tuple(d["all_native_raw_snapshot_update_indices"])!=CHECKPOINTS or
        tuple(d["additional_integer_scales_descending"])!=SCALES or
        d["public_VALID_actual_chosen_probe_budget"]!=4 or
        e["scientific_acceptance"] is not False or
        any(v is not False for v in f["authority"].values())
    ): raise PermissionError("R118_FROZEN_SOURCE_OR_AUTHORITY_DRIFT")
    return f

def allow(panel):
    audit_freeze()
    if panel not in ("UNIT_FIT","UNIT_VALID"):
        raise PermissionError("R118_SPENT_PROTECTED_OR_UNSEEN_FIRST_USE_DENIED")
    return panel
