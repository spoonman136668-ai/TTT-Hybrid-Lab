"""R117 FIT/VALID-only fresh weight-portability custody. No original scored source."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments import r116_source_custody as r116

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r117_native_weight_portability_freeze.json"
FREEZE_BLOB="9e342ddcbec2122c8fc6486ab1b83bd21de7e827"
SCHEMA="ttt-hybrid.r117.fit-valid-native-optimizer-rounding-portability.preregistered-v1"
SEEDS=(1160011,1160023,1160037)
SCALES=(384,256,192,128,96,64,48,32,24,16,8)
CHECKPOINTS=(0,1,32,128,256,512)


def audit_freeze():
    r116.audit_freeze()
    raw=FREEZE.read_bytes()
    canonical=raw.replace(b"\r\n",b"\n")
    if b"\r" in canonical:
        raise RuntimeError("R117_NONCANONICAL_FREEZE_EOL")
    sha=hashlib.sha1(b"blob "+str(len(canonical)).encode()+b"\0"+canonical).hexdigest()
    if sha!=FREEZE_BLOB:
        raise RuntimeError("R117_ORIGINAL_PREIMPLEMENTATION_GIT_BLOB_DRIFT")
    f=json.loads(raw)
    source=f["source"];diagnostic=f["diagnostics"];evaluation=f["evaluation"]
    if (
        f["schema"]!=SCHEMA
        or f["stage"]!="FROZEN_PREIMPLEMENTATION_FIT_VALID_ONLY_NO_ORIGINAL_HELDOUT_FIRSTUSE"
        or source["parent_exact_R116_source_sha"]!="653c240fcddb75e3888f74b0a36ee00d93f42215"
        or tuple(source["seeds"])!=SEEDS
        or source["allowed_panels"]!=["UNIT_FIT","UNIT_VALID"]
        or source["do_not_generate_R116_UNIT_UNSEEN"] is not True
        or source["no_original_unseen_generation_tests"] is not True
        or source["trainable_arms"]!=3
        or source["AdamW_updates_per_arm"]!=512
        or source["matched_trainable_minibatch_size"]!=16
        or source["FIT_worlds_per_seed"]!=256
        or source["VALID_worlds_per_seed"]!=96
        or source["per_VALID_world_physical_actions"]!=4
        or source["VALID_same_public_calibration"]!=[8,9,7]
        or source["VALID_fourth_action_candidates"]!=[12,14]
        or tuple(diagnostic["snapshot_native_raw_FP32_at_updates"])!=CHECKPOINTS
        or tuple(diagnostic["candidate_integer_scales_descending"])!=SCALES
        or diagnostic["no_R116_unit_unseen_replay"] is not True
        or evaluation["scientific_pass_from_this_study"] is not False
        or any(x is not False for x in f["authority"].values())
    ):
        raise PermissionError("R117_FROZEN_SOURCE_OR_PERMISSION_DRIFT")
    return f


def permit_public_panel(panel):
    audit_freeze()
    if panel not in ("UNIT_FIT","UNIT_VALID"):
        raise PermissionError("R117_SPENT_SCIENTIFIC_FIRST_USE_DENIED")
    return panel
