"""Prospective R112 FIXED FIT/VALID-only portability custody.

This cannot qualify independent science or allow re-use of R111 UNIT_UNSEEN.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from experiments import r111_source_custody as r111

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r112_portable_probe_freeze.json"
FREEZE_BLOB="b41eb9dd59e27cc86f1cb0c153b2da1c79431e1e"
SCHEMA="ttt-hybrid.r112.portable-fixed-point-active-probe.fit-valid-only-freeze.v1"
SEEDS=(1011011,1011023,1011037)
Q=384


def audit_freeze():
    old=r111.audit_freeze()
    raw=FREEZE.read_bytes()
    canonical=raw.replace(b"\r\n",b"\n")
    if b"\r" in canonical:
        raise RuntimeError("R112_NONCANONICAL_FREEZE_LINE_ENDING")
    actual=hashlib.sha1(b"blob "+str(len(canonical)).encode()+b"\0"+canonical).hexdigest()
    if actual!=FREEZE_BLOB:
        raise RuntimeError("R112_PROSPECTIVE_FREEZE_GIT_BLOB_DRIFT")
    f=json.loads(raw)
    lineage,src,port,ev,authority=(f[k] for k in ("lineage","source",
                                                "portable_inference","evaluation","authority"))
    if (f["schema"]!=SCHEMA or
        f["stage"]!="PROSPECTIVE_FROZEN_BEFORE_IMPLEMENTATION_FIT_VALID_ONLY_NO_HELDOUT_OR_PROTECTED_FIRST_USE" or
        tuple(src["seeds"])!=SEEDS or tuple(old["seeds"])!=SEEDS or
        lineage["base_R111_exact_sha"]!="557e79e2be25851fa57eef0be0997936cf36af41" or
        lineage["base_R111_reproducibility"]!="STRICT_CROSS_OS_PROBE_SCORE_PARITY_FAIL" or
        src["public_panels_allowed"]!=["UNIT_FIT","UNIT_VALID"] or
        src["public_panels_forbidden"]!=["UNIT_UNSEEN","original_scientific_source","protected"] or
        src["R111_fit_worlds_per_seed"]!=176 or
        src["R111_valid_worlds_per_seed"]!=88 or
        src["fit_trainable_native_AdamW_steps"]!=512 or
        src["chosen_action_budget_per_world"]!=12 or
        src["no_original_R111_unseen_reuse"] is not True or
        port["fixed_point_scale"]!=Q or
        port["layers"]!=[5,32,16,1] or
        ev["SCIENCE_PASS_from_THIS_study"] is not False or
        any(v is not False for v in authority.values())):
        raise PermissionError("R112_UNAUTHORIZED_SOURCE_OR_FREEZE_DRIFT")
    return f


def fit_valid_only(panel):
    audit_freeze()
    if panel not in ("UNIT_FIT","UNIT_VALID"):
        raise PermissionError("R112_UNSEEN_AND_ORIGINAL_SCIENCE_SOURCE_DENIED")
    return panel
