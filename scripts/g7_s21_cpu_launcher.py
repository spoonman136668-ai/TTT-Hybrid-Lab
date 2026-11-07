"""One-use exact-SHA, nonroot CPU admission for G7-S21 gradient-subspace no-replay interference protection with matched random and rehearsal controls.

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
    source=ROOT/"scripts/g7_s21_orthogonal_memory.py"
    protocol=ROOT/"experiments/g7_s21_frozen_orthogonal_memory.json"
    receipt=ROOT/"local_runs/g7_s21_orthogonal_memory_evidence.json"
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
                "ttt-hybrid.g7-s21.frozen-orthogonal-continuous-memory.v1" and
            frozen["predecessor"]["s20_sha"]==
                "dfdf6706a69097027490d580659be6c0f6401aea" and
            frozen["predecessor"]["s20_run"]==42400091370 and
            frozen["training"]["seeds"]==[1361,1429,1531] and
            frozen["training"]["total_updates_per_model"]==528 and
            frozen["evaluation"]["seeds"]["sine"]==
                [3001,3011,3023] and
            frozen["evaluation"]["seeds"]["rational"]==
                [3109,3121,3137] and
            frozen["calibration"]["seeds"]==[2819,2833,2843] and
            frozen["mechanism"]["capture_last_steps_per_family"]==12 and
            frozen["environment"]["tokenizer"] is False and
            frozen["environment"]["attention"] is False and
            frozen["authority"]["cpu_only"] is True and
            frozen["authority"]["no_package_change"] is True,
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
        guard(receipt.is_file(),"S21_SCIENCE_RECEIPT_NOT_WRITTEN")
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
                "PRELIMINARY_ORTHOGONAL_TOKENLESS_RETENTION_GAIN_NOT_SCIENTIFIC_PASS",
                "ORTHOGONAL_MEMORY_FROZEN_RETENTION_OR_PLASTICITY_NULL") and
            record.get("source_oracle_positive_controls_pass") is True and
            record.get("source_kernel_positive_controls_pass") is True and
            record.get("tokenizer_used") is False and
            record.get("attention_used") is False and
            record.get("protected_no_old_episode_replay") is True and
            record.get("r3_pinned_manifest_sha256")==
                custody.R3_MANIFEST_SHA256 and
            record.get("train_updates_per_arm_per_seed")==528 and
            record.get("train_worlds_per_arm_per_seed")==25344 and
            record.get("all_trained_model_cold_restores_exact") is True and
            len(record.get("per_seed",[]))==3 and
            record.get("deployment")=="NOT_AUTHORIZED",
            "S21_COMPLETED_SCIENCE_OR_CONTROL_RECEIPT_INVALID")
        admission="PASS_ORTHOGONAL_EPISODIC_GRADIENT_STUDY_MEASURED"
        status_code=0
    except (Blocked,custody.Blocked) as e:
        allowed={
            "NORMAL_WSL_USER_REQUIRED","EXACT_SOURCE_SHA_OR_DIRTY_TREE",
            "SOURCE_PINS_DRIFT","FROZEN_TOKENLESS_STUDY_PROTOCOL_DRIFT",
            "ORIGINAL_PINNED_CPU_PYTHON_MISSING",
            "DPKG_READ_ONLY_INVENTORY_FAILED",
            "S21_SCIENCE_RECEIPT_NOT_WRITTEN",
            "INSTALLED_PACKAGE_INVENTORY_CHANGED",
            "S21_COMPLETED_SCIENCE_OR_CONTROL_RECEIPT_INVALID"
        }
        admission="BLOCKED_"+(
            str(e) if str(e) in allowed else "UNRECOGNIZED_SOURCE_GUARD")
    except Exception as e:
        admission="BLOCKED_"+type(e).__name__
    finally:
        print("G7_S21_ADMISSION="+admission,flush=True)
        print("G7_S21_SCIENCE="+science,flush=True)
        print("G7_S21_CUDA=NOT_USED",flush=True)
        print("G7_S21_PACKAGE_PROVENANCE="+(
            "RECEIPT_IF_PRESENT" if before else "UNVERIFIED"),flush=True)
    return status_code

if __name__=="__main__":
    raise SystemExit(main())
