"""R114 prospective exact-integer Bayesian prediction custody; FIT/VALID only."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from experiments import r113_source_custody as r113

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r114_exact_readout_freeze.json"
FREEZE_BLOB="a00e102ae6c7ae19c8f03a14ee54f7481f6e3961"
SCHEMA="ttt-hybrid.r114.exact-integer-compositional-bayes-readout.fit-valid-only-freeze.v1"
SEEDS=(1130011,1130023,1130037)


def audit_freeze():
    prior=r113.audit_freeze()
    raw=FREEZE.read_bytes()
    canon=raw.replace(b"\r\n",b"\n")
    if b"\r" in canon:
        raise RuntimeError("R114_FREEZE_LINE_ENDING_DRIFT")
    sha=hashlib.sha1(b"blob "+str(len(canon)).encode()+b"\0"+canon).hexdigest()
    if sha!=FREEZE_BLOB:
        raise RuntimeError("R114_PREIMPLEMENTATION_GIT_BLOB_DRIFT")
    f=json.loads(raw)
    src=f["source"];num=f["exact_prediction"];authority=f["authority"]
    if (f["schema"]!=SCHEMA or
        f["stage"]!="PROSPECTIVE_PREIMPLEMENTATION_FIT_VALID_ONLY_NO_R113_UNSEEN_REUSE" or
        tuple(src["seeds"])!=SEEDS or tuple(prior["source"]["seeds"])!=SEEDS or
        src["allowed_panels"]!=["UNIT_FIT","UNIT_VALID"] or
        src["forbidden_panels"]!=["UNIT_UNSEEN","original_protected","R111_original","R113_consumed"] or
        src["R113_no_scored_holdout_generated"] is not True or
        src["training_updates_per_trainable_arm"]!=512 or
        src["trainable_arms"]!=3 or
        num["likelihood_match_integer"]!=17 or
        num["likelihood_mismatch_integer"]!=3 or
        num["likelihood_denominator_integer"]!=20 or
        num["known_template_count"]!=36 or
        num["support_sample_limit_per_phase"]!=6 or
        num["never_float_in_final_score"] is not True or
        f["evaluation"]["original_scientific_PASS_from_fit_valid"] is not False or
        any(v is not False for v in authority.values())):
        raise PermissionError("R114_SOURCE_INTEGRITY_OR_AUTHORITY_DRIFT")
    return f


def allow(panel):
    audit_freeze()
    if panel not in ("UNIT_FIT","UNIT_VALID"):
        raise PermissionError("R114_SPENT_HELDOUT_AND_PROTECTED_SOURCE_DENIED")
    return panel
