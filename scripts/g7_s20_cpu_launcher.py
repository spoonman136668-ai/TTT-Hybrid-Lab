"""One-use exact-SHA, nonroot CPU admission for G7-S20 hidden-state four-phase retention and partial-observation inference.

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
    source=ROOT/"scripts/g7_s20_hidden_multicycle.py"
    protocol=ROOT/"experiments/g7_s20_frozen_hidden_multicycle.json"
    receipt=ROOT/"local_runs/g7_s20_hidden_multicycle_evidence.json"
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
                "ttt-hybrid.g7-s20.hidden-state-multicycle.v1" and
            frozen["predecessor"]["s19_sha"]==
                "65308a66956b2bca70c48600fd5a879b5b22d388" and
            frozen["predecessor"]["s19_run"]==42398803072 and
            frozen["training"]["seeds"]==[1091,1217,1301] and
            frozen["training"]["total_updates_per_trained_arm_per_seed"]==528 and
            frozen["evaluation"]["heldout_seeds"]["sine"]==
                [2609,2617,2621] and
            frozen["evaluation"]["heldout_seeds"]["rational"]==
                [2707,2713,2729] and
            frozen["calibration"]["seeds"]==[2401,2411,2423] and
            frozen["environment"]["no_tokenizer"] is True and
            frozen["environment"]["no_attention"] is True and
            frozen["authority"]["cpu_only"] is True and
            frozen["authority"]["no_install"] is True,
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
        guard(receipt.is_file(),"S20_SCIENCE_RECEIPT_NOT_WRITTEN")
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
                "PRELIMINARY_PARTIAL_OBSERVATION_MULTICYCLE_TRANSFER_NOT_SCIENTIFIC_PASS",
                "PARTIAL_OBSERVATION_OR_MULTICYCLE_LEARNING_FROZEN_NULL") and
            record.get("oracle_controls_pass") is True and
            record.get("source_conventional_controls_pass") is True and
            record.get("tokenizer_used") is False and
            record.get("vocabulary_used") is False and
            record.get("attention_used") is False and
            record.get("hidden_state_or_latent_coefficients_supplied_to_predictor") is False and
            record.get("r3_manifest_sha256")==
                custody.R3_MANIFEST_SHA256 and
            record.get("steps_per_model_per_seed")==528 and
            record.get("worlds_per_model_per_seed")==25344 and
            record.get("all_model_cold_restores_exact") is True and
            record.get("untrained_kernel_and_random_controls_unchanged") is True and
            len(record.get("per_seed",[]))==3 and
            record.get("deployment")=="NOT_AUTHORIZED",
            "S20_COMPLETED_SCIENCE_OR_CONTROL_RECEIPT_INVALID")
        admission="PASS_CPU_HIDDEN_STATE_FOUR_PHASE_STUDY_MEASURED"
        status_code=0
    except (Blocked,custody.Blocked) as e:
        allowed={
            "NORMAL_WSL_USER_REQUIRED","EXACT_SOURCE_SHA_OR_DIRTY_TREE",
            "SOURCE_PINS_DRIFT","FROZEN_TOKENLESS_STUDY_PROTOCOL_DRIFT",
            "ORIGINAL_PINNED_CPU_PYTHON_MISSING",
            "DPKG_READ_ONLY_INVENTORY_FAILED",
            "S20_SCIENCE_RECEIPT_NOT_WRITTEN",
            "INSTALLED_PACKAGE_INVENTORY_CHANGED",
            "S20_COMPLETED_SCIENCE_OR_CONTROL_RECEIPT_INVALID"
        }
        admission="BLOCKED_"+(
            str(e) if str(e) in allowed else "UNRECOGNIZED_SOURCE_GUARD")
    except Exception as e:
        admission="BLOCKED_"+type(e).__name__
    finally:
        print("G7_S20_ADMISSION="+admission,flush=True)
        print("G7_S20_SCIENCE="+science,flush=True)
        print("G7_S20_CUDA=NOT_USED",flush=True)
        print("G7_S20_PACKAGE_PROVENANCE="+(
            "RECEIPT_IF_PRESENT" if before else "UNVERIFIED"),flush=True)
    return status_code

if __name__=="__main__":
    raise SystemExit(main())
