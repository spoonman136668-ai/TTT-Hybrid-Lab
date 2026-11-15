"""R78: exact-custody maximum-width native FIT 2048 -> 4096 in two bounded jobs.

The only input to stage 1 is the immutable R77 2048 checkpoint artifact.
The only input to stage 2 is the exact stage-1 artifact from THIS run after
stage-1 success. Neither stage ever replays earlier selected-action updates,
reinitializes optimizer state, changes source/learning, or touches DEV/OOD.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import urllib.request

import numpy as np
import torch

from experiments import r74_capability_ceiling as r74
from experiments import r75_cumulative_training as train
from experiments import r77_verified_resume as r77

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r78_4096_freeze.json"
FREEZE_BLOB="b33ae6b2801753f9a7d3183aad75cc273248e34b"
SCHEMA="ttt-hybrid.r78-2048-to-4096-two-bounded-stages.v1"
BRANCH="research/r78-verified-two-chunk-2048-to-4096-20261115"
REPO="spoonman136668-ai/TTT-Hybrid-Lab"
ORIGINAL_IDENTITY_SHA="b034eeb054763b5f8423a3413af2655ff0facb03"
R77_PARENT_SHA="7af6784d814224988a77aab98146d61c906325cc"
ARMS=("HybridPilot_FULL","GRU64","MLP64")
WIDTH=1024
LENGTH=128
SEED=997601
STAGES={1:(2048,3072),2:(3072,4096)}
SRC={1:ROOT/"local_runs/r78_import_r77",2:ROOT/"local_runs/r78_import_stage1"}
OUTPUT={1:ROOT/"local_runs/r78_stage1",2:ROOT/"local_runs/r78_stage2"}
API={1:ROOT/"local_runs/r78_api_stage1.json",2:ROOT/"local_runs/r78_api_stage2.json"}
RECEIPT={1:ROOT/"local_runs/r78_stage1_receipt.json",2:ROOT/"local_runs/r78_stage2_receipt.json"}


def git_blob(raw):
    return r77.git_blob(raw)


def hash_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for piece in iter(lambda:f.read(1024*1024),b""):
            h.update(piece)
    return h.hexdigest()


def frozen():
    if git_blob(FREEZE.read_bytes())!=FREEZE_BLOB:
        raise RuntimeError("R78_IMMUTABLE_FREEZE_HASH_MISMATCH")
    obj=json.loads(FREEZE.read_text(encoding="utf-8"))
    s=obj["source_checkpoint"]
    e=obj["execution"]
    p=obj["parent"]
    if (obj["schema"]!=SCHEMA
        or obj["stage"]!="PROSPECTIVE_FROZEN_BEFORE_R78_SOURCE_TEST_WORKFLOW_FIRST_USE"
        or p["R77_exact_SHA"]!=R77_PARENT_SHA
        or p["R77_artifact_digest"]!="sha256:2bde1690c31ec381c4e444853f0262b75a11d6120c4f767c05433b3e895173df"
        or p["R77_freeze_git_blob"]!=r77.FREEZE_BLOB
        or p["R76_immutable_original_git_SHA"]!=ORIGINAL_IDENTITY_SHA
        or s["checkpoint_original_identity_sha"]!=ORIGINAL_IDENTITY_SHA
        or s["checkpoint_R77_lineage_sha"]!=R77_PARENT_SHA
        or obj["learning"]["arms"]!=list(ARMS)
        or obj["learning"]["width"]!=WIDTH
        or obj["learning"]["sequence"]!=LENGTH
        or obj["learning"]["seed"]!=SEED
        or obj["learning"]["stages"]!=[
            {"id":1,"start":2048,"end":3072},
            {"id":2,"start":3072,"end":4096}]
        or e["max_wall_minutes_each"]!=29
        or e["max_parallel_fit_training_workers"]!=1
        or e["first_original_GITHUB_RUN_ATTEMPT_only"] is not True
        or any(obj["authority"].values())):
        raise PermissionError("R78_ORIGINAL_FROZEN_SEMANTICS_MISMATCH")
    r77.frozen()  # exact prior torch/numpy, original source pins
    return obj


def run_identity(stage):
    if stage not in STAGES:
        raise ValueError("R78_INVALID_STAGE")
    freeze=frozen()
    if (os.getenv("GITHUB_ACTIONS")!="true"
        or os.getenv("GITHUB_EVENT_NAME")!="pull_request"
        or os.getenv("GITHUB_HEAD_REF")!=BRANCH
        or os.getenv("RUNNER_OS")!="Linux"
        or os.getenv("RUNNER_ENVIRONMENT")!="github-hosted"
        or os.getenv("GITHUB_RUN_ATTEMPT")!="1"
        or os.getenv("GITHUB_REPOSITORY")!=REPO
        or "/merge" not in os.getenv("GITHUB_REF","")
        or os.getenv("R78_STAGE")!=str(stage)):
        raise PermissionError("R78_ONLY_FIRST_SAME_REPO_HOSTED_PR_CPU_JOB")
    sha=subprocess.run(["git","rev-parse","HEAD"],cwd=str(ROOT),
                       text=True,capture_output=True,timeout=10,check=False)
    status=subprocess.run(["git","diff","--quiet","HEAD","--"],
                          cwd=str(ROOT),capture_output=True,timeout=10,check=False)
    candidate=os.getenv("EXPECTED_CANDIDATE_SHA","")
    run=os.getenv("GITHUB_RUN_ID","")
    if (len(candidate)!=40 or sha.returncode!=0
        or sha.stdout.strip()!=candidate or status.returncode!=0
        or not run.isdigit()):
        raise PermissionError("R78_UNKNOWN_OR_DIRTY_CANDIDATE_OR_RUN")
    return {"candidate_SHA":candidate,"run_id":int(run),"stage":stage}


def _github_json(url):
    token=os.getenv("GITHUB_TOKEN","")
    if not token:
        raise PermissionError("R78_ORIGINAL_READ_ONLY_GITHUB_TOKEN_MISSING")
    req=urllib.request.Request(url,headers={
        "Authorization":"Bearer "+token,
        "Accept":"application/vnd.github+json",
        "X-GitHub-Api-Version":"2022-11-28",
        "User-Agent":"R78-frozen-model-custody-only"})
    with urllib.request.urlopen(req,timeout=25) as response:
        return json.loads(response.read(100000))


def verified_source_artifact(stage):
    ident=run_identity(stage)
    obj=frozen()
    p=obj["parent"]
    if stage==1:
        data=_github_json(f"https://api.github.com/repos/{REPO}/actions/artifacts/{p['R77_private_artifact_id']}")
        expected={"id":p["R77_private_artifact_id"],
                  "name":p["R77_private_artifact_name"],
                  "digest":p["R77_artifact_digest"],
                  "size_in_bytes":p["R77_artifact_bytes"]}
        for key,value in expected.items():
            if data.get(key)!=value:
                raise PermissionError("R78_ORIGINAL_R77_ARTIFACT_CUSTODY_MISMATCH_"+key)
        run=data.get("workflow_run") or {}
        if run.get("id")!=p["R77_G4_run_id"] or run.get("head_sha")!=R77_PARENT_SHA:
            raise PermissionError("R78_ORIGINAL_R77_ARTIFACT_RUN_SHA_DRIFT")
    else:
        name=f"ttt-hybrid-r78-stage1-{ident['candidate_SHA']}-{ident['run_id']}"
        data=_github_json(
            f"https://api.github.com/repos/{REPO}/actions/runs/{ident['run_id']}/artifacts?per_page=100")
        found=[item for item in data.get("artifacts",[]) if item.get("name")==name]
        if len(found)!=1:
            raise PermissionError("R78_STAGE1_SAME_RUN_PRIVATE_ARTIFACT_NOT_UNIQUE")
        data=found[0]
        run=data.get("workflow_run") or {}
        if run.get("id")!=ident["run_id"] or run.get("head_sha")!=ident["candidate_SHA"]:
            raise PermissionError("R78_STAGE1_SOURCE_RUN_SHA_DRIFT")
    if (data.get("expired") is not False
        or not re.fullmatch(r"sha256:[0-9a-f]{64}",str(data.get("digest","")))
        or not isinstance(data.get("size_in_bytes"),int)
        or data["size_in_bytes"]<=0):
        raise PermissionError("R78_SOURCE_ARTIFACT_EXPIRED_OR_DIGEST_UNKNOWN")
    if API[stage].exists():
        raise FileExistsError("R78_DUPLICATE_API_RECEIPT_REFUSED")
    result={"schema":SCHEMA,"stage":stage,**ident,
            "original_artifact_id":data["id"],
            "artifact_name":data["name"],
            "artifact_digest":data["digest"],
            "artifact_bytes":data["size_in_bytes"],
            "artifact_head_SHA":run.get("head_sha"),
            "artifact_run_id":run.get("id"),
            "not_expired":True}
    API[stage].parent.mkdir(parents=True,exist_ok=True)
    API[stage].write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    return result


def checkpoint_name(arm,updates,stage):
    if arm not in ARMS or stage not in STAGES:
        raise ValueError("R78_UNKNOWN_ARM_OR_STAGE")
    return f"R78-{arm}-width{WIDTH}-seq{LENGTH}-seed{SEED}-u{updates}.pt"


def original_identity(arm):
    return r77.old_identity(arm)


def preflight(stage):
    ident=run_identity(stage)
    spec=frozen()
    p=spec["parent"]
    source=spec["source_checkpoint"]
    if not API[stage].is_file():
        raise FileNotFoundError("R78_OFFICIAL_GITHUB_SOURCE_METADATA_NOT_VERIFIED")
    metadata=json.loads(API[stage].read_text())
    if (metadata["candidate_SHA"]!=ident["candidate_SHA"]
        or metadata["stage"]!=stage or metadata["run_id"]!=ident["run_id"]
        or metadata["not_expired"] is not True):
        raise PermissionError("R78_ARTIFACT_METADATA_CANDIDATE_DRIFT")
    if stage==1:
        if (metadata["original_artifact_id"]!=p["R77_private_artifact_id"]
            or metadata["artifact_digest"]!=p["R77_artifact_digest"]):
            raise PermissionError("R78_SOURCE_PARENT_ARTIFACT_DIGEST_DRIFT")
        record_path=SRC[stage]/source["original_inner_receipt"]
        if not record_path.is_file() or record_path.is_symlink():
            raise FileNotFoundError("R78_R77_RECEIPT_MISSING")
        record=json.loads(record_path.read_text())
        if (record.get("status")!="PASS_FIT_ONLY_2048_CONTINUATION"
            or record.get("R77_candidate_SHA")!=R77_PARENT_SHA
            or record.get("R76_original_SHA")!=ORIGINAL_IDENTITY_SHA
            or record.get("target_cumulative_updates")!=2048
            or record.get("CPU_only") is not True
            or record.get("DEV_OOD_generated") is not False):
            raise PermissionError("R78_R77_ORIGINAL_RECEIPT_DRIFT")
        rows={x["arm"]:x for x in record["completed_arms"]}
    else:
        record_path=SRC[stage]/"r78_stage1_receipt.json"
        if not record_path.is_file() or record_path.is_symlink():
            raise FileNotFoundError("R78_STAGE1_SAME_RUN_RECEIPT_MISSING")
        record=json.loads(record_path.read_text())
        if (record.get("status")!="PASS_FIT_ONLY_STAGE1_3072"
            or record.get("candidate_SHA")!=ident["candidate_SHA"]
            or record.get("run_id")!=ident["run_id"]
            or record.get("source_original_R77_SHA")!=R77_PARENT_SHA
            or record.get("target_updates")!=3072
            or record.get("CPU_only") is not True
            or record.get("DEV_OOD_generated") is not False):
            raise PermissionError("R78_UNQUALIFIED_STAGE1_CUSTODY")
        rows={x["arm"]:x for x in record["completed_arms"]}
    if set(rows)!=set(ARMS):
        raise PermissionError("R78_INCOMPLETE_OR_DUPLICATED_CONTROL_ARMS")
    valid={}
    for arm in ARMS:
        row=rows[arm]
        if stage==1:
            path=SRC[1]/source["directory"]/f"R77-{arm}-width{WIDTH}-seq{LENGTH}-seed{SEED}-u2048.pt"
            sha=source["checkpoint_sha256"][arm]
            length=source["checkpoint_bytes"][arm]
            mse=source["measured_FIT_validation_at_2048"][arm]
            if (row["checkpoint_sha256"]!=sha
                or row["checkpoint_size"]!=length
                or abs(row["intermediate_2048_FIT_validation_MSE"]-mse)>1e-12
                or row["to_updates"]!=2048):
                raise PermissionError("R78_FROZEN_R77_CHECKPOINT_META_DRIFT")
        else:
            path=SRC[2]/"r78_stage1"/checkpoint_name(arm,3072,1)
            sha=row["checkpoint_sha256"]
            length=row["checkpoint_bytes"]
            mse=row["FIT_validation_MSE_after"]
            if row["to_updates"]!=3072 or row["from_updates"]!=2048:
                raise PermissionError("R78_STAGE1_INTERVAL_DRIFT")
        if (not path.is_file() or path.is_symlink()
            or path.stat().st_size!=length or hash_file(path)!=sha):
            raise PermissionError("R78_SOURCE_CHECKPOINT_BYTES_OR_HASH_DRIFT")
        state=torch.load(path,map_location="cpu",weights_only=True)
        if (state.get("identity")!=original_identity(arm)
            or state.get("updates_completed")!=STAGES[stage][0]
            or "model_state" not in state or "optimizer_state" not in state
            or not isinstance(state.get("FIT_sampling_rng"),dict)
            or state["FIT_sampling_rng"].get("bit_generator")!="PCG64"
            or state.get("continuation_lineage",{}).get("continuation_candidate_git_SHA")!=R77_PARENT_SHA
            or len(state.get("FIT_validation_history",[]))!=1
            or state["FIT_validation_history"][0]["updates"]!=1024):
            raise PermissionError("R78_ORIGINAL_MODEL_OPTIMIZER_RNG_LINEAGE_DRIFT")
        if stage==2:
            chain=state.get("R78_stages",[])
            if (len(chain)!=1 or chain[0].get("stage")!=1
                or chain[0].get("candidate_SHA")!=ident["candidate_SHA"]
                or chain[0].get("parent_checkpoint_sha256")!=
                   source["checkpoint_sha256"][arm]):
                raise PermissionError("R78_STAGE1_PARENT_CHAIN_DRIFT")
        if not all(bool(torch.isfinite(v).all()) for v in state["model_state"].values()
                   if isinstance(v,torch.Tensor)):
            raise ValueError("R78_SOURCE_NONFINITE_PARAMETERS")
        valid[arm]={"path":path,"hash":sha,"bytes":length,
                    "starting_FIT_validation_MSE":float(mse)}
        del state
        gc.collect()
    return valid


def resume_arm(stage,arm,original,fit,validation,run):
    start,end=STAGES[stage]
    parent=torch.load(original["path"],map_location="cpu",weights_only=True)
    if parent["updates_completed"]!=start or parent["identity"]!=original_identity(arm):
        raise PermissionError("R78_SOURCE_REPLACED_AFTER_PREFLIGHT")
    model=train.optimized_model(arm,WIDTH,SEED)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    model.load_state_dict(parent["model_state"],strict=True)
    opt.load_state_dict(parent["optimizer_state"])
    rng=np.random.default_rng(0)
    rng.bit_generator.state=parent["FIT_sampling_rng"]
    current=train.score_fit_validation(model,validation)
    if not math.isfinite(current) or abs(current-original["starting_FIT_validation_MSE"])>1e-6:
        raise RuntimeError("R78_STORED_FIT_VALIDATION_NOT_REPRODUCIBLE")
    began=time.perf_counter()
    last_loss=0.0
    for update in range(start+1,end+1):
        selection=rng.integers(0,len(fit["input"]),2)
        last_loss=train.one_fit_update(model,opt,fit,selection)
    final=train.score_fit_validation(model,validation)
    if not math.isfinite(final):
        raise RuntimeError("R78_NONFINITE_NEW_FIT_VALIDATION")
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise RuntimeError("R78_NONFINITE_NEW_WEIGHTS")
    lineage={"stage":stage,"from":start,"to":end,
             "candidate_SHA":run["candidate_SHA"],"source_parent_SHA":R77_PARENT_SHA,
             "parent_checkpoint_sha256":original["hash"],"original_first_R76_identity":ORIGINAL_IDENTITY_SHA,
             "real_selected_FIT_events_added":(end-start)*2*LENGTH}
    chain=[*parent.get("R78_stages",[]),lineage]
    history=list(parent["FIT_validation_history"])
    if stage==2:
        if end!=4096 or len(chain)!=2:
            raise RuntimeError("R78_FROZEN_4096_CUSTODY_NOT_COMPLETE")
        history.append({"updates":4096,
                        "FIT_last_selected_MSE":last_loss,
                        "FIT_validation_MSE":final,
                        "R78_exact_lineage_candidate":run["candidate_SHA"],
                        "original_R74_milestone":True})
    new={"identity":parent["identity"],"continuation_lineage":parent["continuation_lineage"],
         "R78_stages":chain,"updates_completed":end,
         "model_state":model.state_dict(),
         "optimizer_state":opt.state_dict(),
         "FIT_sampling_rng":rng.bit_generator.state,
         "FIT_validation_history":history}
    dest=OUTPUT[stage]/checkpoint_name(arm,end,stage)
    OUTPUT[stage].mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix(".tmp")
    if dest.exists() or tmp.exists():
        raise FileExistsError("R78_UNCERTAIN_EFFECT_REPLAY_REFUSED")
    torch.save(new,tmp)
    os.replace(tmp,dest)
    result={"arm":arm,"from_updates":start,"to_updates":end,
            "parameters":sum(p.numel() for p in model.parameters()),
            "added_selected_FIT_events":(end-start)*2*LENGTH,
            "cumulative_selected_FIT_events":end*2*LENGTH,
            "FIT_validation_MSE_before":current,
            "FIT_validation_MSE_after":final,
            "FIT_validation_relative_gain":(current-final)/max(current,1e-12),
            "training_wall_seconds":time.perf_counter()-began,
            "checkpoint_name":dest.name,
            "checkpoint_bytes":dest.stat().st_size,
            "checkpoint_sha256":hash_file(dest),
            "parent_checkpoint_sha256":original["hash"],
            "original_AdamW_and_PCG64_restored":True,
            "FIT_only":True,
            "fully_original_frozen_R74_4096_milestone_for_arm":stage==2}
    del parent,model,opt,rng,new
    gc.collect()
    return result


def execute(stage):
    run=run_identity(stage)
    spec=frozen()
    if RECEIPT[stage].exists() or OUTPUT[stage].exists():
        raise FileExistsError("R78_EXISTING_OUTPUT_OR_UNCERTAIN_REPLAY_DENIED")
    preflight_checks=preflight(stage)
    torch.set_num_threads(1)
    started=time.perf_counter()
    src=r74.factual_worlds("FIT",sequence=LENGTH)
    val=r74.factual_worlds("FIT_VALIDATION",sequence=LENGTH)
    if (src["source_count"]!=512 or src["source_panel"]!="FIT"
        or val["source_count"]!=128 or val["source_panel"]!="FIT_VALIDATION"):
        raise PermissionError("R78_CHANGED_ORIGINAL_FIT_SOURCE")
    start,end=STAGES[stage]
    result={"schema":SCHEMA,"freeze_blob":FREEZE_BLOB,
            "stage":stage,"candidate_SHA":run["candidate_SHA"],
            "run_id":run["run_id"],"source_original_R77_SHA":R77_PARENT_SHA,
            "source_artifact":json.loads(API[stage].read_text()),
            "from_updates":start,"target_updates":end,
            "CPU_only":True,"DEV_OOD_generated":False,
            "independent_scientific_pass":False,"deployment":"NONE",
            "completed_arms":[],"status":"INCOMPLETE"}
    try:
        for arm in ARMS:
            row=resume_arm(stage,arm,preflight_checks[arm],src,val,run)
            result["completed_arms"].append(row)
        if len(result["completed_arms"])!=3:
            raise RuntimeError("R78_ORIGINAL_THREE_ARMS_INCOMPLETE")
        result["status"]="PASS_FIT_ONLY_STAGE1_3072" if stage==1 else "PASS_FIT_ONLY_STAGE2_4096"
    except BaseException as exc:
        result["status"]="INCOMPLETE_ENGINEERING_OR_RESOURCE_LIMIT"
        result["error_type"]=type(exc).__name__
        raise
    finally:
        result["wall_seconds"]=time.perf_counter()-started
        RECEIPT[stage].parent.mkdir(parents=True,exist_ok=True)
        with RECEIPT[stage].open("x",encoding="utf-8") as out:
            json.dump(result,out,sort_keys=True,indent=2,allow_nan=False)
            out.write("\n")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",type=int,choices=(1,2),required=True)
    p.add_argument("--verify-artifact",action="store_true")
    p.add_argument("--run",action="store_true")
    a=p.parse_args()
    if a.verify_artifact == a.run:
        p.error("choose exactly one of --verify-artifact or --run")
    answer=verified_source_artifact(a.stage) if a.verify_artifact else execute(a.stage)
    print(json.dumps({"stage":a.stage,
        "status":answer.get("status","ORIGINAL_ARTIFACT_METADATA_VERIFIED"),
        "candidate":answer.get("candidate_SHA"),
        "DEV_OOD_generated":False},sort_keys=True))


if __name__=="__main__":
    main()
