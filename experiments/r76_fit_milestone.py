"""R76 single bounded top-width FIT milestone on the existing Ubuntu G4 host.

Runs exactly the R75 accelerated native reference and fully trained
conventional GRU64/MLP64 controls for the first original frozen 1024 updates
at width1024/sequence128, seed997601. Only FIT and FIT_VALIDATION are ever
constructed; scoring OOD/DEV or GPU first-use is explicitly prohibited.
The existing PR G4 workflow is the only execution route (not a scheduler).
Original full AdamW/NumPy checkpoint is retained for cold continuation.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch

from experiments import r74_capability_ceiling as r74
from experiments import r75_triangular_fastweight as r75
from experiments import r75_cumulative_training as training

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r76_fit_milestone_freeze.json"
FREEZE_BLOB="fde88a6cdd0a2956e91b412aaa1567275055babc"
SCHEMA="ttt-hybrid.r76-high-first-fit-milestone.v1"
PARENT="131f9f5de17073b25f7008fbe6dbcb21f22b7f43"
BRANCH="research/r76-max-width-1024-fit-checkpoint-20261115"
ARMS=("HybridPilot_FULL","GRU64","MLP64")
SEED=997601
WIDTH=1024
SEQUENCE=128
UPDATES=1024
RECEIPT=ROOT/"local_runs/r76_fit_milestone.json"
ACTUAL_SELECTED_FIT_EVENTS=UPDATES*2*SEQUENCE
PREVIOUS_R75_FREEZE="2cf720b2214715c7977ffc5b87ea45ed193e3087"


def git_blob(raw:bytes)->str:
    original=raw.replace(b"\r\n",b"\n")
    if b"\r" in original:
        raise ValueError("R76_UNSUPPORTED_TEXT_CARRIAGE_RETURN")
    return hashlib.sha1(b"blob "+str(len(original)).encode()+b"\0"+original).hexdigest()


def frozen():
    s=json.loads(FREEZE.read_text(encoding="utf-8"))
    if git_blob(FREEZE.read_bytes())!=FREEZE_BLOB:
        raise RuntimeError("R76_IMMUTABLE_FREEZE_CHANGED")
    if (s["schema"]!=SCHEMA or s["stage"]!="FROZEN_BEFORE_R76_TRAINING_PR_AND_ORIGINAL_CI"
        or s["parent"]["R75_exact_SHA"]!=PARENT
        or s["parent"]["R75_freeze_blob"]!=PREVIOUS_R75_FREEZE
        or s["parent"]["R74_freeze_blob"]!=r74.FREEZE_BLOB
        or s["training"]["architecture"]!=list(ARMS)
        or s["training"]["seed"]!=SEED
        or s["training"]["width"]!=WIDTH
        or s["training"]["sequence"]!=SEQUENCE
        or s["training"]["batch"]!=2
        or s["training"]["optim_updates"]!=UPDATES
        or s["training"]["selected_real_FIT_events_per_arm"]!=ACTUAL_SELECTED_FIT_EVENTS
        or s["execution"]["timeout_minutes"]!=29
        or s["execution"]["trigger"]!="ONE_PULL_REQUEST_ON_R76_BRANCH_ONLY_NO_SCHEDULE_NO_WORKFLOW_DISPATCH"
        or any(s["authority"].values())):
        raise RuntimeError("R76_ORIGINAL_PROTOCOL_SEMANTICS_CHANGED")
    r75.frozen()
    return s


def checkout_readback():
    sha_cmd=subprocess.run(
        ["git","rev-parse","HEAD"],cwd=str(ROOT),
        capture_output=True,text=True,check=False,timeout=10)
    status=subprocess.run(
        ["git","status","--porcelain","--untracked-files=all"],cwd=str(ROOT),
        capture_output=True,text=True,check=False,timeout=10)
    if (sha_cmd.returncode!=0 or status.returncode!=0
        or not sha_cmd.stdout.strip() or status.stdout.strip()):
        raise PermissionError("R76_CHECKOUT_NOT_CLEAN_OR_IDENTITY_UNKNOWN")
    return sha_cmd.stdout.strip()


def admit_cpu_execution():
    frozen()
    if (os.environ.get("GITHUB_ACTIONS")!="true"
        or os.environ.get("GITHUB_EVENT_NAME")!="pull_request"
        or os.environ.get("GITHUB_HEAD_REF")!=BRANCH
        or os.environ.get("RUNNER_OS")!="Linux"
        or os.environ.get("RUNNER_ENVIRONMENT")!="github-hosted"
        or os.environ.get("GITHUB_RUN_ATTEMPT")!="1"
        or os.environ.get("GITHUB_REPOSITORY")!="spoonman136668-ai/TTT-Hybrid-Lab"):
        raise PermissionError("R76_ONLY_FIRST_ORIGINAL_HOSTED_UBUNTU_PR_RUN_ALLOWED")
    expected=os.environ.get("EXPECTED_CANDIDATE_SHA")
    observed=checkout_readback()
    if (not expected or len(expected)!=40 or observed!=expected
        or os.environ.get("GITHUB_REF","").find("/merge")==-1):
        raise PermissionError("R76_EXACT_PR_HEAD_SHA_OR_REF_DENIED")
    return observed


def hash_file(path:Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initial_fit_validation(arm):
    model=training.optimized_model(arm,WIDTH,SEED)
    validation=r74.factual_worlds("FIT_VALIDATION",sequence=SEQUENCE)
    if validation["source_panel"]!="FIT_VALIDATION" or validation["source_count"]!=128:
        raise RuntimeError("R76_ONLY_ORIGINAL_FIT_VALIDATION_ALLOWED")
    score=training.score_fit_validation(model,validation)
    del model,validation
    gc.collect()
    if not (score>=0 and score<float("inf")):
        raise RuntimeError("R76_INVALID_INITIAL_FIT_VALIDATION")
    return score


def run():
    """Starts only through the approved existing branch-guarded G4 job."""
    expected_sha=admit_cpu_execution()
    torch.set_num_threads(1)
    start=time.perf_counter()
    if RECEIPT.exists():
        raise FileExistsError("R76_DUPLICATE_SCORED_OR_INCOMPLETE_RECEIPT_DENIED")
    REPORT={
        "schema":SCHEMA,"freeze_blob":FREEZE_BLOB,
        "R75_parent_SHA":PARENT,
        "candidate_SHA":expected_sha,
        "platform":platform.platform(),
        "CPU_torch":str(torch.__version__),
        "source":"ORIGINAL_R74_FIT_AND_FIT_VALIDATION_ONLY",
        "width":WIDTH,"sequence":SEQUENCE,"seed":SEED,
        "updates_target":UPDATES,"arms_requested":list(ARMS),
        "training_arms_completed":[],
        "original_R75_engine":training.ENGINE,
        "DEV_OOD_generated":False,
        "GPU_or_external_firstuse":False,
        "fully_trained_65536":False,
        "scientific_pass":"UNVERIFIED",
        "status":"INCOMPLETE"
    }
    try:
        for arm in ARMS:
            destination=training.checkpoint_path(arm,WIDTH,SEQUENCE,SEED)
            if destination.exists():
                raise FileExistsError("R76_REFUSE_EXISTING_CHECKPOINT_REPLAY")
            before=initial_fit_validation(arm)
            result=training.train_explicit_fit_only(
                arm,WIDTH,SEQUENCE,SEED,UPDATES)
            if result["updates_completed"]!=UPDATES or result["identity"]["checkout_commit_sha"]!=expected_sha:
                raise RuntimeError("R76_ORIGINAL_CUMULATIVE_UPDATE_IDENTITY_BAD")
            if result["training_exposures"]!=ACTUAL_SELECTED_FIT_EVENTS:
                raise RuntimeError("R76_ORIGINAL_SELECTED_FIT_EVENT_BUDGET_DRIFT")
            if (not destination.is_file() or destination.stat().st_size==0):
                raise FileNotFoundError("R76_ORIGINAL_FULL_OPTIMIZER_CHECKPOINT_NOT_WRITTEN")
            trained=torch.load(destination,map_location="cpu",weights_only=True)
            if (trained["identity"]!=training.identity(arm,WIDTH,SEQUENCE,SEED)
                or trained["updates_completed"]!=UPDATES
                or len(trained["FIT_validation_history"])!=1
                or trained["FIT_validation_history"][0]["updates"]!=UPDATES):
                raise RuntimeError("R76_SOURCE_OR_OPTIMIZER_CUSTODY_FAILED")
            after=float(trained["FIT_validation_history"][0]["FIT_validation_MSE"])
            REPORT["training_arms_completed"].append({
                "arm":arm,"update_count":UPDATES,
                "selected_factual_events":ACTUAL_SELECTED_FIT_EVENTS,
                "parameters":result["trainable_parameters"],
                "initial_FIT_validation_MSE":before,
                "final_FIT_validation_MSE":after,
                "FIT_validation_relative_improvement":(before-after)/max(before,1e-12),
                "trainer_seconds":result["time_this_invocation_seconds"],
                "checkpoint_path":str(destination.relative_to(ROOT)),
                "checkpoint_bytes":destination.stat().st_size,
                "checkpoint_sha256":hash_file(destination),
                "checkpoint_engine":trained["identity"]["engine"],
                "full_AdamW_RNG_state":True,
                "results_are_FIT_only":True
            })
            del trained
            gc.collect()
            # The receipt is immutable once final; intermediate progress is
            # kept in RAM. Original checkpoint files remain retained by CI.
        if len(REPORT["training_arms_completed"])!=3:
            raise RuntimeError("R76_NOT_ALL_THREE_ARMS_TRAINED")
        REPORT["wall_seconds"]=time.perf_counter()-start
        REPORT["status"]="PASS_1024_FIT_CPU_CHECKPOINTS_ONLY"
    except BaseException as e:
        REPORT["status"]="INCOMPLETE_FIT_ENGINEERING"
        REPORT["error_class"]=type(e).__name__
        REPORT["wall_seconds"]=time.perf_counter()-start
        raise
    finally:
        RECEIPT.parent.mkdir(exist_ok=True,parents=True)
        with RECEIPT.open("x",encoding="utf-8") as h:
            json.dump(REPORT,h,sort_keys=True,indent=2,allow_nan=False)
            h.write("\n")
    return REPORT


if __name__=="__main__":
    result=run()
    print(json.dumps({"status":result["status"],
        "SHA":result["candidate_SHA"],
        "arms":len(result["training_arms_completed"]),
        "wall_seconds":result["wall_seconds"],
        "DEV_OOD_generated":False,
        "scientific_pass":"UNVERIFIED"},sort_keys=True,allow_nan=False))
