"""R77: custody-preserving, single original FIT continuation 1024 -> 2048.

Import exclusively the verified R76 original GitHub artifact. Preserve R75
model, optimizer and NumPy PCG64 sampling stream EXACTLY, with no replay of
the original 1024 steps, no changing R74/R75 training math, and no DEV/OOD.
The 2048 snapshot is an operational transfer escrow, not a frozen R74
scientific evaluation checkpoint. Existing G4 PR workflow only, CPU-only.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import urllib.request

import numpy as np
import torch

from experiments import r74_capability_ceiling as r74
from experiments import r75_triangular_fastweight as r75
from experiments import r75_cumulative_training as train

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r77_resume_freeze.json"
FREEZE_BLOB="7fc188627bc2081d2dd1f657fd430b3541df851e"
SCHEMA="ttt-hybrid.r77-exact-custody-continuation-v1"
BRANCH="research/r77-verified-1024-to-2048-resume-20261115"
REPO="spoonman136668-ai/TTT-Hybrid-Lab"
ORIGINAL_SHA="b034eeb054763b5f8423a3413af2655ff0facb03"
ORIGINAL_ARTIFACT_ID=14160994105
ARMS=("HybridPilot_FULL","GRU64","MLP64")
WIDTH=1024
SEQUENCE=128
SEED=997601
START=1024
STOP=2048
SOURCE_DIR=ROOT/"local_runs/r77_input"
OUTPUT_DIR=ROOT/"local_runs/r77_intermediate"
SOURCE_API=ROOT/"local_runs/r77_verified_original_artifact.json"
RECEIPT=ROOT/"local_runs/r77_continuation_receipt.json"


def git_blob(raw):
    canonical=raw.replace(b"\r\n",b"\n")
    if b"\r" in canonical:
        raise ValueError("R77_INVALID_TEXT_CARRIAGE_RETURN")
    return hashlib.sha1(b"blob "+str(len(canonical)).encode()+b"\0"+canonical).hexdigest()


def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for piece in iter(lambda:handle.read(4*1024*1024),b""):
            h.update(piece)
    return h.hexdigest()


def frozen():
    if git_blob(FREEZE.read_bytes())!=FREEZE_BLOB:
        raise RuntimeError("R77_PROSPECTIVE_FREEZE_DRIFT")
    z=json.loads(FREEZE.read_text(encoding="utf-8"))
    s=z["source"]
    e=z["execution"]
    if (z["schema"]!=SCHEMA
        or z["stage"]!="PROSPECTIVE_FREEZE_BEFORE_R77_HANDOFF_IMPLEMENTATION_AND_EXECUTION"
        or s["original_parent_SHA"]!=ORIGINAL_SHA
        or s["original_artifact_id"]!=ORIGINAL_ARTIFACT_ID
        or s["original_artifact_name"]!="ttt-hybrid-r76-original-FIT-b034eeb054763b5f8423a3413af2655ff0facb03-43711769367"
        or s["original_engine"]!=train.ENGINE
        or s["immutable_R74_freeze_blob"]!=r74.FREEZE_BLOB
        or s["immutable_R75_freeze_blob"]!=r75.FREEZE_BLOB
        or e["from_updates"]!=START or e["to_updates"]!=STOP
        or e["arms"]!=list(ARMS)
        or e["width"]!=WIDTH or e["sequence"]!=SEQUENCE
        or e["batch"]!=2 or e["seed"]!=SEED
        or e["max_wall_minutes"]!=29
        or e["max_concurrent_workers"]!=1
        or any(z["authority"].values())):
        raise RuntimeError("R77_PROSPECTIVE_SEMANTIC_FREEZE_DRIFT")
    r75.frozen()
    source_checks={
      "R75_math_engine_source_sha256":train.canonical_sha(r75.__file__),
      "R75_training_driver_source_sha256":train.canonical_sha(train.__file__)
    }
    for key,value in source_checks.items():
        if value!=s["original_"+key]:
            raise PermissionError("R77_TRAINING_ENGINE_CHANGED_NOT_COMPATIBLE")
    if str(torch.__version__)!=s["original_torch"] or np.__version__!=s["original_numpy"]:
        raise PermissionError("R77_ORIGINAL_TORCH_NUMPY_TOOLCHAIN_CHANGED")
    return z


def admit_first_pr_runner():
    frozen()
    if (os.environ.get("GITHUB_ACTIONS")!="true"
        or os.environ.get("GITHUB_EVENT_NAME")!="pull_request"
        or os.environ.get("GITHUB_HEAD_REF")!=BRANCH
        or os.environ.get("GITHUB_REPOSITORY")!=REPO
        or os.environ.get("RUNNER_ENVIRONMENT")!="github-hosted"
        or os.environ.get("RUNNER_OS")!="Linux"
        or os.environ.get("GITHUB_RUN_ATTEMPT")!="1"
        or "/merge" not in os.environ.get("GITHUB_REF","")):
        raise PermissionError("R77_UNAUTHORIZED_FIRST_ORIGINAL_PR_HOST")
    got=subprocess.run(["git","rev-parse","HEAD"],cwd=str(ROOT),
                       text=True,capture_output=True,check=False,timeout=10)
    dirty=subprocess.run(["git","diff","--quiet","HEAD","--"],
                         cwd=str(ROOT),capture_output=True,check=False,timeout=10)
    expected=os.environ.get("EXPECTED_CANDIDATE_SHA","")
    if (got.returncode!=0 or dirty.returncode!=0
        or len(expected)!=40 or got.stdout.strip()!=expected):
        raise PermissionError("R77_EXACT_SHA_OR_TRACKED_SOURCE_DRIFT")
    return expected


def api_artifact_metadata():
    """Only GitHub's official read-only Actions endpoint, no third-party API."""
    candidate=admit_first_pr_runner()
    token=os.environ.get("GITHUB_TOKEN","")
    if not token:
        raise PermissionError("R77_READ_ONLY_GITHUB_ARTIFACT_TOKEN_MISSING")
    url=f"https://api.github.com/repos/{REPO}/actions/artifacts/{ORIGINAL_ARTIFACT_ID}"
    req=urllib.request.Request(url,headers={
        "Authorization":f"Bearer {token}",
        "Accept":"application/vnd.github+json",
        "X-GitHub-Api-Version":"2022-11-28",
        "User-Agent":"R77-source-provenance-only"})
    with urllib.request.urlopen(req,timeout=20) as response:
        data=json.loads(response.read(65536))
    s=frozen()["source"]
    run=data.get("workflow_run") or {}
    if (data.get("id")!=ORIGINAL_ARTIFACT_ID
        or data.get("name")!=s["original_artifact_name"]
        or data.get("digest")!="sha256:"+s["original_artifact_zip_sha256"]
        or data.get("size_in_bytes")!=s["original_artifact_size"]
        or data.get("expired") is not False
        or run.get("id")!=s["original_G4_run_id"]
        or run.get("head_sha")!=ORIGINAL_SHA):
        raise PermissionError("R77_SOURCE_ARTIFACT_ID_DIGEST_SIZE_OR_RUN_MISMATCH")
    receipt={
      "schema":SCHEMA,"R77_candidate_SHA":candidate,
      "original_run_id":s["original_G4_run_id"],
      "original_artifact_id":ORIGINAL_ARTIFACT_ID,
      "original_artifact_digest":data["digest"],
      "original_artifact_size":data["size_in_bytes"],
      "original_artifact_name":data["name"],
      "original_head_sha":ORIGINAL_SHA,
      "original_artifact_expired":False,"source_verified_by":"GITHUB_READ_ONLY_API"
    }
    SOURCE_API.parent.mkdir(exist_ok=True,parents=True)
    with SOURCE_API.open("x",encoding="utf-8") as f:
        json.dump(receipt,f,sort_keys=True,indent=2)
        f.write("\n")
    return receipt


def original_name(arm):
    if arm not in ARMS:
        raise PermissionError("R77_UNKNOWN_ORIGINAL_ARM")
    return f"R75-{arm}-width{WIDTH}-seq{SEQUENCE}-seed{SEED}.pt"


def old_identity(arm):
    current=train.identity(arm,WIDTH,SEQUENCE,SEED)
    current["checkout_commit_sha"]=ORIGINAL_SHA
    return current


def source_preflight():
    """Hash/size/actual checkpoint identity BEFORE any update, no replay."""
    candidate=admit_first_pr_runner()
    spec=frozen()
    s=spec["source"]
    if not SOURCE_API.is_file():
        raise FileNotFoundError("R77_SOURCE_ARTIFACT_API_CUSTODY_RECEIPT_MISSING")
    metadata=json.loads(SOURCE_API.read_text(encoding="utf-8"))
    if (metadata.get("R77_candidate_SHA")!=candidate
        or metadata.get("original_artifact_digest")!=
           "sha256:"+s["original_artifact_zip_sha256"]
        or metadata.get("original_artifact_id")!=ORIGINAL_ARTIFACT_ID):
        raise PermissionError("R77_SOURCE_API_READBACK_DRIFT")
    receipt_path=SOURCE_DIR/s["original_inner_receipt"]
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise FileNotFoundError("R77_ORIGINAL_RESULT_RECEIPT_MISSING")
    old=json.loads(receipt_path.read_text(encoding="utf-8"))
    if (old.get("candidate_SHA")!=ORIGINAL_SHA
        or old.get("status")!="PASS_1024_FIT_CPU_CHECKPOINTS_ONLY"
        or old.get("freeze_blob")!=s["immutable_R76_freeze_blob"]
        or old.get("width")!=WIDTH
        or old.get("sequence")!=SEQUENCE
        or old.get("seed")!=SEED
        or old.get("updates_target")!=START
        or old.get("DEV_OOD_generated") is not False
        or old.get("scientific_pass")!="UNVERIFIED"):
        raise PermissionError("R77_ORIGINAL_1024_RECEIPT_INCONSISTENT")
    rows={row["arm"]:row for row in old["training_arms_completed"]}
    if set(rows)!=set(ARMS):
        raise PermissionError("R77_ORIGINAL_CONTROL_INCOMPLETE")
    sources={}
    for arm in ARMS:
        row=rows[arm]
        path=SOURCE_DIR/s["original_inner_paths_prefix"]/original_name(arm)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError("R77_ORIGINAL_CHECKPOINT_NOT_FOUND")
        if (path.stat().st_size!=row["checkpoint_bytes"]
            or path.stat().st_size!=s["original_inner_bytes"][arm]
            or sha256_file(path)!=s["original_inner_sha256"][arm]
            or row["checkpoint_sha256"]!=s["original_inner_sha256"][arm]
            or row["update_count"]!=START
            or row["selected_factual_events"]!=START*2*SEQUENCE
            or abs(row["final_FIT_validation_MSE"]-
                   s["original_FIT_validation_MSE_1024"][arm])>1e-12):
            raise PermissionError("R77_ORIGINAL_CHECKPOINT_HASH_OR_METRIC_DRIFT")
        state=torch.load(path,map_location="cpu",weights_only=True)
        if (state.get("identity")!=old_identity(arm)
            or state.get("updates_completed")!=START
            or len(state.get("FIT_validation_history",[]))!=1
            or state["FIT_validation_history"][0]["updates"]!=START
            or not isinstance(state.get("FIT_sampling_rng"),dict)
            or "model_state" not in state or "optimizer_state" not in state):
            raise PermissionError("R77_ORIGINAL_MODEL_OPTIMIZER_RNG_IDENTITY_MISMATCH")
        if any(not bool(torch.isfinite(t).all()) for t in state["model_state"].values()
               if isinstance(t,torch.Tensor)):
            raise ValueError("R77_NONFINITE_ORIGINAL_WEIGHT")
        sources[arm]={"name":path.name,"bytes":path.stat().st_size,
                      "sha256":s["original_inner_sha256"][arm],
                      "initial_1024_MSE":row["final_FIT_validation_MSE"]}
        del state
        gc.collect()
    return sources


def resume_one(arm,verified,input_source,validation):
    """The R75 one_fit_update and exact original PCG64 stream, never warm start."""
    file=SOURCE_DIR/"r75_cumulative_fit"/original_name(arm)
    origin=torch.load(file,map_location="cpu",weights_only=True)
    if origin["identity"]!=old_identity(arm) or origin["updates_completed"]!=START:
        raise PermissionError("R77_ORIGINAL_INPUT_DRIFT_AFTER_PREFLIGHT")
    model=train.optimized_model(arm,WIDTH,SEED)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    model.load_state_dict(origin["model_state"],strict=True)
    optimizer.load_state_dict(origin["optimizer_state"])
    rng=np.random.default_rng(0)
    rng.bit_generator.state=origin["FIT_sampling_rng"]
    initial=train.score_fit_validation(model,validation)
    if not math.isfinite(initial) or abs(initial-verified["initial_1024_MSE"])>1e-6:
        raise RuntimeError("R77_ORIGINAL_FIT_VALIDATION_NOT_REPRODUCIBLE")
    started=time.perf_counter()
    loss=0.0
    for step in range(START+1,STOP+1):
        selected=rng.integers(0,len(input_source["input"]),2)
        loss=train.one_fit_update(model,optimizer,input_source,selected)
    final=train.score_fit_validation(model,validation)
    if not math.isfinite(final):
        raise RuntimeError("R77_NONFINITE_INTERMEDIATE_FIT_VALIDATION")
    if not all(bool(torch.isfinite(v).all()) for v in model.parameters()):
        raise ValueError("R77_NONFINITE_AFTER_CONTINUATION")
    candidate=admit_first_pr_runner()
    lineage={
        "schema":SCHEMA,"parent_checkpoint_SHA256":verified["sha256"],
        "parent_original_git_SHA":ORIGINAL_SHA,
        "continuation_candidate_git_SHA":candidate,
        "from_updates":START,"to_updates":STOP,
        "original_engine_identity_unchanged":True,
        "only_FIT_and_FIT_VALIDATION":True,
        "not_R74_frozen_checkpoint_4096":True
    }
    checkpoint={
        "identity":origin["identity"],  # DO NOT rewrite original 1024 SHA custody.
        "continuation_lineage":lineage, # additional proof for subsequent escrow.
        "updates_completed":STOP,
        "model_state":model.state_dict(),
        "optimizer_state":optimizer.state_dict(),
        "FIT_sampling_rng":rng.bit_generator.state,
        "FIT_validation_history":origin["FIT_validation_history"]  # Only original scientific checkpoints, no 2048 entry.
    }
    dest=OUTPUT_DIR/f"R77-{arm}-width{WIDTH}-seq{SEQUENCE}-seed{SEED}-u{STOP}.pt"
    if dest.exists() or dest.with_suffix(".tmp").exists():
        raise FileExistsError("R77_DUPLICATE_OR_UNCERTAIN_CHECKPOINT_REPLAY")
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix(".tmp")
    torch.save(checkpoint,tmp)
    os.replace(tmp,dest)
    result={"arm":arm,"from_updates":START,"to_updates":STOP,
            "added_actual_selected_FIT_events":(STOP-START)*2*SEQUENCE,
            "cumulative_actual_selected_FIT_events":STOP*2*SEQUENCE,
            "initial_1024_FIT_validation_MSE":initial,
            "intermediate_2048_FIT_validation_MSE":final,
            "FIT_validation_relative_change":(initial-final)/max(initial,1e-12),
            "parameters":sum(p.numel() for p in model.parameters()),
            "training_wall_seconds":time.perf_counter()-started,
            "checkpoint_basename":dest.name,
            "checkpoint_size":dest.stat().st_size,
            "checkpoint_sha256":sha256_file(dest),
            "parent_checkpoint_sha256":verified["sha256"],
            "optimizer_and_rng_state_preserved":True,
            "original_science_checkpoint_schedule_not_tuned":True,
            "DEV_OOD_scored":False}
    del origin,checkpoint,optimizer,model
    gc.collect()
    return result


def run():
    """One admitted original PR job only; never auto-replay on uncertain state."""
    candidate=admit_first_pr_runner()
    if RECEIPT.exists() or OUTPUT_DIR.exists():
        raise FileExistsError("R77_EXISTING_CONTINUATION_STATE_REFUSED")
    original=source_preflight()
    torch.set_num_threads(1)
    started=time.perf_counter()
    report={"schema":SCHEMA,"R77_freeze_git_blob":FREEZE_BLOB,
            "R76_original_SHA":ORIGINAL_SHA,"R77_candidate_SHA":candidate,
            "original_artifact_id":ORIGINAL_ARTIFACT_ID,
            "original_source_1024_checkpoints":original,
            "started_at_updates":START,"target_cumulative_updates":STOP,
            "arms":list(ARMS),"completed_arms":[],
            "CPU_only":True,"DEV_OOD_generated":False,
            "independent_scientific_pass":False,
            "frozen_R74_4096_checkpoint_completed":False,
            "deployment":"NONE",
            "status":"INCOMPLETE"}
    try:
        fit=r74.factual_worlds("FIT",sequence=SEQUENCE)
        valid=r74.factual_worlds("FIT_VALIDATION",sequence=SEQUENCE)
        if (fit["source_panel"]!="FIT" or fit["source_count"]!=512
            or valid["source_panel"]!="FIT_VALIDATION" or valid["source_count"]!=128):
            raise PermissionError("R77_SOURCE_FIT_COHORT_CHANGED")
        for arm in ARMS:
            row=resume_one(arm,original[arm],fit,valid)
            report["completed_arms"].append(row)
        if len(report["completed_arms"])!=len(ARMS):
            raise RuntimeError("R77_INCOMPLETE_ALL_CONTROL_COHORT")
        report["status"]="PASS_FIT_ONLY_2048_CONTINUATION"
    except BaseException as ex:
        report["status"]="INCOMPLETE_ENGINEERING_STOP"
        report["exception"]=type(ex).__name__
        raise
    finally:
        report["wall_seconds"]=time.perf_counter()-started
        RECEIPT.parent.mkdir(exist_ok=True,parents=True)
        with RECEIPT.open("x",encoding="utf-8") as out:
            json.dump(report,out,sort_keys=True,indent=2,allow_nan=False)
            out.write("\n")
    return report


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--verify-artifact",action="store_true")
    parser.add_argument("--run",action="store_true")
    args=parser.parse_args()
    if args.verify_artifact:
        result=api_artifact_metadata()
    elif args.run:
        result=run()
    else:
        parser.error("choose --verify-artifact or --run")
    print(json.dumps({"status":result.get("status","PARENT_ARTIFACT_VERIFIED"),
        "R77_candidate":result.get("R77_candidate_SHA"),
        "DEV_OOD_generated":False},sort_keys=True))


if __name__=="__main__":
    main()
