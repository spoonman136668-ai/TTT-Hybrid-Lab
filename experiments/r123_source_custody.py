"""R123 public-FIT only GELU primitive prospective freeze and source custody."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from experiments import r122_source_custody as r122

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r123_gelu_primitive_freeze.json"
FREEZE_BLOB="c615affeb16477fa62e2ca287891f87d4e289226"
SCHEMA="ttt-hybrid.r123.GELU-primitive-forward-gradient-CPU-bitwise-FIT-only.v1"
SEEDS=(1160011,1160023,1160037)
CANDIDATES=("torch_original_GELU_none","torch_GELU_tanh",
            "torch_explicit_erf","decimal_erf_series_96")
LAYERS=("GELU1","GELU2")
WORKERS=("A","B")
PI="3.14159265358979323846264338327950288419716939937510582097494459230781640628620899"

def audit_freeze():
    r122.audit_freeze()
    raw=FREEZE.read_bytes()
    canon=raw.replace(b"\r\n",b"\n")
    if b"\r" in canon:
        raise RuntimeError("R123_FREEZE_NONCANONICAL_EOL")
    sha=hashlib.sha1(b"blob "+str(len(canon)).encode()+b"\0"+canon).hexdigest()
    if sha!=FREEZE_BLOB:
        raise RuntimeError("R123_PREIMPLEMENTATION_GIT_BLOB_DRIFT")
    f=json.loads(raw);s=f["source"];c=f["capture"];e=f["evaluation"]
    if (f["schema"]!=SCHEMA or
        f["stage"]!="PROSPECTIVE_PREIMPLEMENTATION_PUBLIC_FIT_ONLY" or
        f["lineage"]["parent_R122_source_sha"]!="4b0225b3d34bc640d3a7c0bc18a43178525b7448" or
        f["lineage"]["parent_R122_tree"]!="010bd73da185a6c12deef60adcf8376e5bb96cbe" or
        s["allowed_panel"]!="UNIT_FIT" or
        s["forbidden_panels"]!=["UNIT_VALID","UNIT_UNSEEN","protected","original_science"] or
        tuple(s["seeds"])!=SEEDS or
        s["public_FIT_worlds_per_seed"]!=256 or
        s["tensor_count_per_seed_two_native_GELU_activations"]!=2304 or
        s["worker_count_per_OS"]!=2 or
        s["model_inference_only_no_optimizer_step"] is not True or
        tuple(x["name"] for x in f["candidates"])!=CANDIDATES or
        c["decimal_precision"]!=70 or
        c["decimal_erf_taylor_terms"]!=96 or
        c["high_precision_pi"]!=PI or
        c["decimal_input_abs_overflow_limit"]!=5.0 or
        c["pass_max_decimal_forward_abs_error"]!=1e-5 or
        c["pass_max_decimal_gradient_abs_error"]!=1e-5 or
        e["scientific_pass"] is not False or
        any(v is not False for v in f["authority"].values())):
        raise PermissionError("R123_FROZEN_PROTOCOL_OR_AUTHORITY_DRIFT")
    return f

def allow(panel):
    audit_freeze()
    if panel!="UNIT_FIT":
        raise PermissionError("R123_SPENT_SCIENCE_OR_NONFIT_PANEL_DENIED")
    return panel
