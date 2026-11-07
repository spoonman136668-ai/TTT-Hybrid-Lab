"""One-use exact-SHA, nonroot CPU admission for G7-S23 persistent context-error hysteresis banks vs S22 naive, random, shared and rehearsal.

No system Python package-version check: the pinned package manifest is
validated by the scientific child INSIDE the prequalified R3 virtualenv.
Never print subprocess stderr, command-line secrets, or workspace paths.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
import g7_r5_oneshot as custody

class Blocked(Exception):
    pass

def guard(ok,reason):
    if not ok:
        raise Blocked(reason)

def dpkg_hash():
    fmt="-f=" + chr(36)+"{binary:Package}\t"+chr(36)+"{Status}\t"+chr(36)+"{Version}\n"
    result=subprocess.run(
        ["/usr/bin/dpkg-query","-W",fmt],
        cwd=ROOT,text=True,capture_output=True,check=False,timeout=30)
    guard(result.returncode==0,"DPKG_READ_ONLY_INVENTORY_FAILED")
    return hashlib.sha256(result.stdout.encode()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--expected-sha",required=True)
    a=parser.parse_args()
    source=ROOT/"scripts/g7_s23_persistent_banks.py"
    protocol=ROOT/"experiments/g7_s23_frozen_persistent_change.json"
    receipt=ROOT/"local_runs/g7_s23_persistent_banks_evidence.json"
    admission="BLOCKED_NOT_STARTED"
    science="NOT_RUN"
    before=None
    status_code=2
    try:
        guard(os.geteuid()==1000 and
              "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower(),
              "NORMAL_WSL_USER_REQUIRED")
        guard(custody.git(ROOT,"rev-parse","HEAD")==a.expected_sha and
              not custody.git(ROOT,"status","--porcelain",
                              "--untracked-files=all"),
              "EXACT_SOURCE_SHA_OR_DIRTY_TREE")
        guard(custody.sha256(ROOT/"source_pins.json")==
              custody.R3_PINS_SHA256,"SOURCE_PINS_DRIFT")
        ast.parse(source.read_text(encoding="utf-8"),filename=str(source))
        frozen=json.loads(protocol.read_text(encoding="utf-8"))
        guard(
            frozen.get("schema")==
                "ttt-hybrid.g7-s23.hysteretic-changepoint-banks.v1" and
            frozen["predecessor"]["s22_science_candidate_sha"]==
                "065e8bdfd79b4624e2c3d8fed23bbba7b12453de" and
            frozen["predecessor"]["s22_run"]==42408137211 and
            frozen["training"]["fit_seeds"]==[1901,2011,2131] and
            frozen["training"]["total_updates_per_trainable_arm"]==528 and
            frozen["evaluation"]["heldout_ood_seeds"]["sine"]==
                [4409,4421,4441] and
            frozen["evaluation"]["heldout_ood_seeds"]["rational"]==
                [4507,4519,4523] and
            frozen["calibration"]["kernel_source_seeds"]==
                [4201,4211,4229] and
            frozen["unconventional"]["detection"]["window_length"]==6 and
            frozen["unconventional"]["detection"]["required_high"]==5 and
            frozen["unconventional"]["detection"]["post_allocation_cooldown_updates"]==36 and
            frozen["unconventional"]["max_banks"]==3 and
            frozen["unconventional"]["training_router_task_label_used"] is False and
            frozen["unconventional"]["evaluation_router_task_label_used"] is False and
            frozen["environment"]["no_tokenizer"] is True and
            frozen["environment"]["no_attention"] is True and
            frozen["authority"]["cpu_only"] is True and
            frozen["authority"]["no_package_install"] is True,
            "FROZEN_TOKENLESS_STUDY_PROTOCOL_DRIFT")
        venv=Path.home().resolve()/".local/share/ttt-hybrid"/(
            "g7-r3-"+custody.R3_SOURCE_SHA[:12])/"venv/bin/python"
        guard(venv.is_file(),"ORIGINAL_PINNED_CPU_PYTHON_MISSING")
        before=dpkg_hash()
        env=dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"]="-1"
        env["PYTHONDONTWRITEBYTECODE"]="1"
        env["OMP_NUM_THREADS"]="1"
        env["MKL_NUM_THREADS"]="1"
        done=subprocess.run(
            [str(venv),"-I","-B",str(source),
             "--expected-sha",a.expected_sha,
             "--protocol",str(protocol),
             "--receipt",str(receipt)],
            cwd=ROOT,env=env,text=True,capture_output=True,
            check=False,timeout=275)
        guard(receipt.is_file(),"S23_SCIENCE_RECEIPT_NOT_WRITTEN")
        record=json.loads(receipt.read_text(encoding="utf-8"))
        science=record.get("status","SCIENTIFIC_STATUS_MISSING")
        after=dpkg_hash()
        record["dpkg_before_sha256"]=before
        record["dpkg_after_sha256"]=after
        record["installed_packages_changed"]=before!=after
        record["r3_expected_manifest_sha256"]=custody.R3_MANIFEST_SHA256
        receipt.write_text(
            json.dumps(record,sort_keys=True,indent=2)+"\n",
            encoding="utf-8")
        guard(before==after,"INSTALLED_PACKAGE_INVENTORY_CHANGED")
        guard(
            done.returncode==0 and
            record.get("candidate_sha")==a.expected_sha and
            record.get("status") in (
                "PRELIMINARY_PERSISTENT_UNLABELED_CHANGE_MEMORY_NOT_SCIENTIFIC_PASS",
                "PERSISTENT_CHANGEPOINT_BANK_ROUTING_OR_RETENTION_NULL") and
            record.get("source_kernel_positive_control_pass") is True and
            record.get("oracle_positive_control_pass") is True and
            record.get("persistent_change_detection_used") is True and
            record.get("S22_original_single_step_novelty_matched_control") is True and
            record.get("tokenizer_used") is False and
            record.get("attention_used") is False and
            record.get("task_labels_passed_to_bank_router") is False and
            record.get("heldout_targets_used_for_bank_router") is False and
            record.get("old_train_worlds_replayed_in_blind_bank") is False and
            record.get("pinned_r3_manifest_sha256")==
                custody.R3_MANIFEST_SHA256 and
            record.get("updates_per_trainable_arm_per_seed")==528 and
            record.get("worlds_per_trainable_arm_per_seed")==25344 and
            record.get("cold_restores_all_models_exact") is True and
            len(record.get("per_seed",[]))==3 and
            record.get("deployment")=="NOT_AUTHORIZED",
            "S23_COMPLETED_SCIENCE_OR_CONTROL_RECEIPT_INVALID")
        admission="PASS_PERSISTENT_CHANGEPOINT_MEMORY_STUDY_MEASURED"
        status_code=0
    except (Blocked,custody.Blocked) as e:
        allowed={
            "NORMAL_WSL_USER_REQUIRED","EXACT_SOURCE_SHA_OR_DIRTY_TREE",
            "SOURCE_PINS_DRIFT","FROZEN_TOKENLESS_STUDY_PROTOCOL_DRIFT",
            "ORIGINAL_PINNED_CPU_PYTHON_MISSING",
            "DPKG_READ_ONLY_INVENTORY_FAILED",
            "S23_SCIENCE_RECEIPT_NOT_WRITTEN",
            "INSTALLED_PACKAGE_INVENTORY_CHANGED",
            "S23_COMPLETED_SCIENCE_OR_CONTROL_RECEIPT_INVALID"
        }
        admission="BLOCKED_"+(
            str(e) if str(e) in allowed else "UNRECOGNIZED_SOURCE_GUARD")
    except Exception as e:
        admission="BLOCKED_"+type(e).__name__
    finally:
        print("G7_S23_ADMISSION="+admission,flush=True)
        print("G7_S23_SCIENCE="+science,flush=True)
        print("G7_S23_CUDA=NOT_USED",flush=True)
        print("G7_S23_PACKAGE_PROVENANCE="+(
            "RECEIPT_IF_PRESENT" if before else "UNVERIFIED"),flush=True)
    return status_code

if __name__=="__main__":
    raise SystemExit(main())
