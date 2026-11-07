"""G7-S6 source-bound normal-user launcher for frozen CPU-only baseline.

Runs only the pinned R3 venv and transfers a sanitized science receipt.
No downloads, CUDA operations, installation, network, or persistent training.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import g7_r5_oneshot as custody

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--expected-sha",required=True)
    args=p.parse_args()
    report_path=ROOT/"local_runs/g7_s6_trained_baseline.json"
    try:
        custody.guard(os.geteuid()==1000 and "microsoft" in
                      Path("/proc/sys/kernel/osrelease").read_text().lower(),
                      "NORMAL_WSL_USER_REQUIRED")
        custody.guard(custody.git(ROOT,"rev-parse","HEAD")==args.expected_sha
                      and not custody.git(ROOT,"status","--porcelain",
                                          "--untracked-files=all"),
                      "EXACT_SOURCE_SHA_OR_DIRTY")
        script=ROOT/"scripts/g7_s6_trained_conventional_baseline.py"
        protocol=ROOT/"experiments/g7_s6_frozen_conventional_baseline.json"
        ast.parse(script.read_text(encoding="utf-8"),filename=str(script))
        payload=json.loads(protocol.read_text(encoding="utf-8"))
        custody.guard(payload.get("schema")==
                      "ttt-hybrid.g7-s6.conventional-associative-baseline.v1"
                      and payload["baseline"]["steps"]==512,
                      "FROZEN_PROTOCOL_INVALID")
        venv=(Path.home().resolve()/".local/share/ttt-hybrid"/
              ("g7-r3-"+custody.R3_SOURCE_SHA[:12])/"venv/bin/python")
        custody.guard(venv.is_file(),"PINNED_R3_VENV_MISSING")
        env=dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"]="1"
        env["OMP_NUM_THREADS"]="1"
        env["MKL_NUM_THREADS"]="1"
        p=subprocess.run(
            [str(venv),"-I","-B",str(script),
             "--expected-sha",args.expected_sha,
             "--protocol",str(protocol),
             "--receipt",str(report_path)],
            cwd=ROOT,env=env,check=False,text=True,
            capture_output=True,timeout=290
        )
        custody.guard(report_path.is_file(),"SCIENCE_RECEIPT_MISSING")
        data=json.loads(report_path.read_text(encoding="utf-8"))
        custody.guard(
            p.returncode==0 and
            data.get("candidate_sha")==args.expected_sha and
            data.get("status") in (
                "TASK_LEARNABILITY_POSITIVE_CONTROL_PASS",
                "TASK_LEARNABILITY_POSITIVE_CONTROL_FAIL") and
            data.get("package_changes")=="NONE" and
            data.get("gpu_execution")=="NOT_RUN" and
            data.get("hybrid_superiority")=="NOT_TESTED",
            "FROZEN_BASELINE_NOT_QUALIFIED"
        )
        print("G7_S6_ADMISSION=PASS_MEASURED",flush=True)
        print("G7_S6_SCIENCE="+data["status"],flush=True)
        print("G7_S6_GPU=NOT_RUN",flush=True)
        return 0
    except (custody.Blocked, OSError, ValueError,
            subprocess.TimeoutExpired, SyntaxError) as exc:
        category=(str(exc) if isinstance(exc,custody.Blocked) else
                  type(exc).__name__)
        print("G7_S6_ADMISSION=BLOCKED_"+category,flush=True)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
