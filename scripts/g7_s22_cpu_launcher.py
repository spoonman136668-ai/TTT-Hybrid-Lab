"""One-use exact-SHA, nonroot CPU admission for G7-S22 surprise-addressed unlabeled multiple continuous banks vs random, shared, rehearsal and oracle controls.

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
    source=ROOT/"scripts/g7_s22_surprise_banks.py"
    protocol=ROOT/"experiments/g7_s22_frozen_surprise_banks.json"
    receipt=ROOT/"local_runs/g7_s22_surprise_banks_evidence.json"
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
                "ttt-hybrid.g7-s22.surprise-addressed-tokenless-memory.v1" and
            frozen["predecessor"]["s21_sha"]==
                "5855881a2d148160a3414b843e458ddba2f15747" and
            frozen["predecessor"]["s21_run"]==42406899997 and
            frozen["training"]["fit_seeds"]==[1601,1709,1823] and
            frozen["training"]["total_updates_per_trainable_arm"]==528 and
            frozen["evaluation"]["heldout_ood_seeds"]["sine"]==
                [3803,3821,3833] and
            frozen["evaluation"]["heldout_ood_seeds"]["rational"]==
                [3907,3917,3929] and
            frozen["calibration"]["kernel_source_seeds"]==
                [3613,3623,3631] and
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
        guard(receipt.is_file(),"S22_SCIENCE_RECEIPT_NOT_WRITTEN")
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
                "PRELIMINARY_UNLABELED_MEMORY_BANK_RETENTION_SIGNAL_NOT_SCIENTIFIC_PASS",
                "UNLABELED_MEMORY_BANK_ROUTING_OR_PLASTICITY_FROZEN_NULL") and
            record.get("source_kernel_positive_control_pass") is True and
            record.get("oracle_positive_control_pass") is True and
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
            "S22_COMPLETED_SCIENCE_OR_CONTROL_RECEIPT_INVALID")
        admission="PASS_UNLABELED_SURPRISE_BANK_STUDY_MEASURED"
        status_code=0
    except (Blocked,custody.Blocked) as e:
        allowed={
            "NORMAL_WSL_USER_REQUIRED","EXACT_SOURCE_SHA_OR_DIRTY_TREE",
            "SOURCE_PINS_DRIFT","FROZEN_TOKENLESS_STUDY_PROTOCOL_DRIFT",
            "ORIGINAL_PINNED_CPU_PYTHON_MISSING",
            "DPKG_READ_ONLY_INVENTORY_FAILED",
            "S22_SCIENCE_RECEIPT_NOT_WRITTEN",
            "INSTALLED_PACKAGE_INVENTORY_CHANGED",
            "S22_COMPLETED_SCIENCE_OR_CONTROL_RECEIPT_INVALID"
        }
        admission="BLOCKED_"+(
            str(e) if str(e) in allowed else "UNRECOGNIZED_SOURCE_GUARD")
    except Exception as e:
        admission="BLOCKED_"+type(e).__name__
    finally:
        print("G7_S22_ADMISSION="+admission,flush=True)
        print("G7_S22_SCIENCE="+science,flush=True)
        print("G7_S22_CUDA=NOT_USED",flush=True)
        print("G7_S22_PACKAGE_PROVENANCE="+(
            "RECEIPT_IF_PRESENT" if before else "UNVERIFIED"),flush=True)
    return status_code

if __name__=="__main__":
    raise SystemExit(main())
