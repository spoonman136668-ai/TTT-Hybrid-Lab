"""G7-S15: one-shot normal WSL CPU scientific admission, no installs or CUDA."""
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

def require(ok,reason):
    if not ok:
        raise Blocked(reason)

def dpkg_hash():
    fmt="-f=" + chr(36)+"{binary:Package}\t"+chr(36)+"{Status}\t"+chr(36)+"{Version}\n"
    result=subprocess.run(
        ["/usr/bin/dpkg-query","-W",fmt],capture_output=True,
        text=True,cwd=ROOT,timeout=30,check=False)
    require(result.returncode==0,"READ_ONLY_DPKG_INVENTORY_UNAVAILABLE")
    return hashlib.sha256(result.stdout.encode()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--expected-sha",required=True)
    args=parser.parse_args()
    receipt=ROOT/"local_runs/g7_s15_raw_analog_results.json"
    before=None
    result=None
    outcome="BLOCKED_NOT_STARTED"
    exit_status=2
    try:
        require(os.geteuid()==1000 and
                "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower(),
                "NORMAL_WSL_USER_REQUIRED")
        require(custody.git(ROOT,"rev-parse","HEAD")==args.expected_sha
                and not custody.git(ROOT,"status","--porcelain",
                                    "--untracked-files=all"),
                "UNQUALIFIED_CANDIDATE_TREE")
        require(custody.sha256(ROOT/"source_pins.json")==
                custody.R3_PINS_SHA256,
                "SOURCE_PIN_DRIFT")
        # The system Python is only the nonroot launcher. Its distribution
        # inventory is NOT the pinned R3 venv manifest. The child validates
        # exact torch/triton packages inside the qualified research venv.
        source=ROOT/"scripts/g7_s15_raw_analog_dynamics.py"
        spec=ROOT/"experiments/g7_s15_frozen_raw_analog_dynamics.json"
        ast.parse(source.read_text(encoding="utf-8"),filename=str(source))
        frozen=json.loads(spec.read_text(encoding="utf-8"))
        require(frozen["schema"]==
                "ttt-hybrid.g7-s15.raw-analog-dynamics.v1"
                and frozen["evaluation"]["seeds"]==[149,271,397]
                and frozen["evaluation"]["ood_seeds"]==[557,613,719]
                and frozen["numerical_input"]["tokenizer"] is False
                and frozen["numerical_input"]["vocabulary"] is False
                and frozen["resource_limits"]["cpu_only"] is True
                and frozen["resource_limits"]["no_new_packages"] is True,
                "FROZEN_TOKENIZATION_FREE_PROTOCOL_DRIFT")
        python=(Path.home().resolve()/".local/share/ttt-hybrid"/
                ("g7-r3-"+custody.R3_SOURCE_SHA[:12])/"venv/bin/python")
        require(python.is_file(),"PINNED_LOCAL_R3_PYTHON_MISSING")
        before=dpkg_hash()
        env=dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"]="-1"
        env["OMP_NUM_THREADS"]="1"
        env["MKL_NUM_THREADS"]="1"
        env["PYTHONDONTWRITEBYTECODE"]="1"
        command=[str(python),"-I","-B",str(source),
                 "--expected-sha",args.expected_sha,
                 "--protocol",str(spec),
                 "--receipt",str(receipt)]
        done=subprocess.run(
            command,cwd=ROOT,env=env,capture_output=True,
            text=True,check=False,timeout=210)
        require(receipt.is_file(),"SCIENTIFIC_RECEIPT_NOT_WRITTEN")
        result=json.loads(receipt.read_text(encoding="utf-8"))
        after=dpkg_hash()
        result["dpkg_before_sha256"]=before
        result["dpkg_after_sha256"]=after
        result["installed_packages_changed"]=before!=after
        result["pinned_R3_manifest_sha256"]=custody.R3_MANIFEST_SHA256
        receipt.write_text(
            json.dumps(result,sort_keys=True,indent=2)+"\n",
            encoding="utf-8")
        require(before==after,"INSTALLED_PACKAGE_INVENTORY_DRIFT")
        require(done.returncode==0
                and result.get("candidate_sha")==args.expected_sha
                and result.get("status") in (
                    "ANALOG_RLS_PASS_GRU_TRANSFER_NULL_OR_NEGATIVE",
                    "TRAINABLE_CONTINUOUS_GRU_ADAPTATION_SIGNAL")
                and result.get("tokenizer_used") is False
                and result.get("vocabulary_used") is False
                and result.get("discrete_key_ids_used") is False
                and result.get("next_token_loss_used") is False
                and result.get("numerical_time_sampling_used") is True
                and result.get("r3_manifest_sha256") ==
                    custody.R3_MANIFEST_SHA256
                and result.get("oracle_controls_pass") is True
                and result.get("strong_rls_adaptation_controls_pass") is True
                and result.get("training_updates_per_seed")==240
                and result.get("training_worlds_per_seed")==11520,
                "S15_CONTINUOUS_WORLD_RESULTS_NOT_QUALIFIED")
        outcome="PASS_CPU_RAW_CONTINUOUS_EXPERIMENT_MEASURED"
        exit_status=0
    except (Blocked,custody.Blocked) as exc:
        # Fixed local classifications only; never emit arbitrary traceback,
        # workspace paths, command-line arguments or subprocess stderr.
        allowed={
            "NORMAL_WSL_USER_REQUIRED",
            "UNQUALIFIED_CANDIDATE_TREE",
            "SOURCE_PIN_DRIFT",
            "PINNED_VENV_PACKAGE_MANIFEST_DRIFT",
            "FROZEN_TOKENIZATION_FREE_PROTOCOL_DRIFT",
            "PINNED_LOCAL_R3_PYTHON_MISSING",
            "READ_ONLY_DPKG_INVENTORY_UNAVAILABLE",
            "SCIENTIFIC_RECEIPT_NOT_WRITTEN",
            "INSTALLED_PACKAGE_INVENTORY_DRIFT",
            "S15_CONTINUOUS_WORLD_RESULTS_NOT_QUALIFIED",
        }
        code=str(exc)
        outcome="BLOCKED_"+(code if code in allowed else "UNKNOWN_GUARD")
    except (KeyError,ValueError,OSError,subprocess.TimeoutExpired,
            SyntaxError) as exc:
        outcome="BLOCKED_"+type(exc).__name__
    except Exception as exc:
        outcome="BLOCKED_"+type(exc).__name__
    finally:
        print("G7_S15_ADMISSION="+outcome,flush=True)
        print("G7_S15_SCIENCE="+(
            result.get("status","NOT_MEASURED")
            if isinstance(result,dict) else "NOT_MEASURED"),flush=True)
        print("G7_S15_CPU_ONLY=TRUE",flush=True)
        print("G7_S15_DPKG="+(
            str(result.get("installed_packages_changed","UNVERIFIED"))
            if isinstance(result,dict) else "UNVERIFIED"),flush=True)
    return exit_status

if __name__=="__main__":
    raise SystemExit(main())
