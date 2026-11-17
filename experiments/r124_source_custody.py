"""R124 frozen native-only GELU replacement on PUBLIC FIT/VALID only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from experiments import r123_source_custody as r123

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r124_native_erf_portability_freeze.json"
FREEZE_BLOB="dbff7d2b2ccef7a0371848f97732bdf34d0921b2"
SCHEMA="ttt-hybrid.r124.native-explicit-erf-GELU-original-FIT-VALID-portability.v1"
SEEDS=(1160011,1160023,1160037)
UPDATES=512

def audit_freeze():
    r123.audit_freeze()
    raw=FREEZE.read_bytes()
    canonical=raw.replace(b"\r\n",b"\n")
    if b"\r" in canonical:raise RuntimeError("R124_NONCANONICAL_FREEZE_EOL")
    blob=hashlib.sha1(b"blob "+str(len(canonical)).encode()+b"\0"+canonical).hexdigest()
    if blob!=FREEZE_BLOB:
        raise RuntimeError("R124_PREIMPLEMENTATION_GIT_BLOB_DRIFT")
    f=json.loads(raw);s=f["source"];c=f["capture"];e=f["evaluation"]
    if (f["schema"]!=SCHEMA or
        f["stage"]!="PREIMPLEMENTATION_FROZEN_NATIVE_ONLY_CPU_DEVELOPMENT" or
        f["lineage"]["parent_R123_SHA"]!="179c3c886da1701fdf0f6c34b16626c82c7245d9" or
        tuple(s["seeds"])!=SEEDS or
        s["panels"]!=["UNIT_FIT","UNIT_VALID"] or
        s["forbidden"]!=["UNIT_UNSEEN","original_scientific","protected","previous_scored"] or
        s["FIT_examples_per_seed"]!=256 or
        s["VALID_worlds_per_seed"]!=96 or
        s["AdamW_steps_per_arm"]!=UPDATES or
        s["train_minibatch_size"]!=16 or
        s["trainable_arms"]!=["new_native_erf","original_native_GELU_control","conventional_MLP","conventional_GRU"] or
        s["new_GELU_exact_expression"]!="(0.5 * x) * (1.0 + torch.erf(x / math.sqrt(2.0))) in actual native torch FP32 forward; original PyTorch autograd on same graph, no manual backward modification" or
        s["first_primitive_from_R123"]!="torch_explicit_erf" or
        s["model_parameters_native_expected"]!=3409 or
        s["no_original_spent_R116_UNSEEN_reuse"] is not True or
        c["all_native_513_model_FP32_training_weight_SHA_by_update"] is not True or
        e["public_VALID_new_native_correct_min_per_seed"]!=96 or
        e["scientific_pass"] is not False or
        any(v is not False for v in f["authority"].values())):
        raise PermissionError("R124_FROZEN_SOURCE_OR_AUTHORITY_DRIFT")
    return f

def allow(panel):
    audit_freeze()
    if panel not in ("UNIT_FIT","UNIT_VALID"):
        raise PermissionError("R124_SPENT_OR_PROTECTED_ORIGINAL_SCIENCE_DENIED")
    return panel
