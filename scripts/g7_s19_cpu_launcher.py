"""One-use exact-SHA, nonroot CPU admission for G7-S19 two-novel-family transfer and sequential interference.

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
    source=ROOT/"scripts/g7_s19_novel_dynamics.py"
    protocol=ROOT/"experiments/g7_s19_frozen_novel_dynamics.json"
    receipt=ROOT/"local_runs/g7_s19_novel_dynamics_evidence.json"
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
                "ttt-hybrid.g7-s19.novel-dynamics-sequential-interference.v1" and
            frozen["predecessor"]["s18_exact_sha"]==
                "affe574f00d69abb8d816589ed12d81e716fc458" and
            frozen["predecessor"]["s18_science_run"]==42396385331 and
            frozen["frozen_training"]["seeds"]==[701,809,907] and
            frozen["frozen_training"]["steps"]==240 and
            frozen["evaluation"]["split_seeds"]["novel_sine_ood_switch"]==
                [2101,2113,2129] and
            frozen["evaluation"]["split_seeds"]["novel_rational_ood_switch"]==
                [2203,2213,2221] and
            frozen["evaluation"]["sequential_interference"]["phase_B_updates"]==96 and
            frozen["calibration"]["seeds"]==[1901,1913,1931] and
            frozen["world"]["no_tokenizer"] is True and
            frozen["world"]["no_attention"] is True and
            frozen["resource_authority"]["cpu_only"] is True and
            frozen["resource_authority"]["no_package_install"] is True,
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
            check=False,timeout=240)
        guard(receipt.is_file(),"S17_SCIENCE_RECEIPT_NOT_WRITTEN")
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
                "PRELIMINARY_CROSS_FAMILY_TOKENLESS_TRANSFER_REQUIRES_INDEPENDENT_QUALIFICATION",
                "CROSS_FAMILY_RAW_FEATURE_TRANSFER_NULL_OR_NEGATIVE") and
            record.get("original_source_oracle_controls_pass") is True and
            record.get("original_source_conventional_controls_pass") is True and
            record.get("tokenizer_used") is False and
            record.get("vocabulary_used") is False and
            record.get("attention_used") is False and
            record.get("true_nonlinear_feature_formula_exposed_to_models") is False and
            record.get("r3_pinned_package_manifest_sha256")==
                custody.R3_MANIFEST_SHA256 and
            record.get("trained_model_steps_phase_A_per_seed")==240 and
            record.get("trained_model_steps_phase_B_per_seed")==96 and
            record.get("training_worlds_per_seed_per_model")==16128 and
            record.get("all_trained_cold_restores_exact") is True and
            len(record.get("per_seed",[]))==3 and
            record.get("deployment")=="NOT_AUTHORIZED",
            "S19_COMPLETED_SCIENCE_OR_CONTROL_RECEIPT_INVALID")
        admission="PASS_CPU_CROSS_FAMILY_TRANSFER_AND_INTERFERENCE_MEASURED"
        status_code=0
    except (Blocked,custody.Blocked) as e:
        allowed={
            "NORMAL_WSL_USER_REQUIRED","EXACT_SOURCE_SHA_OR_DIRTY_TREE",
            "SOURCE_PINS_DRIFT","FROZEN_TOKENLESS_STUDY_PROTOCOL_DRIFT",
            "ORIGINAL_PINNED_CPU_PYTHON_MISSING",
            "DPKG_READ_ONLY_INVENTORY_FAILED",
            "S17_SCIENCE_RECEIPT_NOT_WRITTEN",
            "INSTALLED_PACKAGE_INVENTORY_CHANGED",
            "S17_COMPLETED_SCIENCE_OR_CONTROL_RECEIPT_INVALID"
        }
        admission="BLOCKED_"+(
            str(e) if str(e) in allowed else "UNRECOGNIZED_SOURCE_GUARD")
    except Exception as e:
        admission="BLOCKED_"+type(e).__name__
    finally:
        print("G7_S19_ADMISSION="+admission,flush=True)
        print("G7_S19_SCIENCE="+science,flush=True)
        print("G7_S19_CUDA=NOT_USED",flush=True)
        print("G7_S19_PACKAGE_PROVENANCE="+(
            "RECEIPT_IF_PRESENT" if before else "UNVERIFIED"),flush=True)
    return status_code

if __name__=="__main__":
    raise SystemExit(main())
