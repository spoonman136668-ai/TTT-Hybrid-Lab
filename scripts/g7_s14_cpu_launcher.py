"""One-use, no-install, non-root admission for G7-S14 CPU-only sketch study."""
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
sys.path.insert(0,str(ROOT/"scripts"))
import g7_r5_oneshot as custody

def guard(x,why):
    if not x:
        raise custody.Blocked(why)

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--expected-sha",required=True)
    args=a.parse_args()
    report=ROOT/"local_runs/g7_s14_catsem_evidence.json"
    try:
        guard(os.geteuid()==1000 and
              "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower(),
              "NORMAL_WSL_USER_REQUIRED")
        guard(custody.git(ROOT,"rev-parse","HEAD")==args.expected_sha
              and not custody.git(ROOT,"status","--porcelain",
                                  "--untracked-files=all"),
              "SOURCE_SHA_OR_WORKTREE_DRIFT")
        script=ROOT/"scripts/g7_s14_catsem_memory.py"
        protocol=ROOT/"experiments/g7_s14_frozen_trisketch.json"
        ast.parse(script.read_text(encoding="utf-8"),filename=str(script))
        p=json.loads(protocol.read_text(encoding="utf-8"))
        guard(p["schema"]==
              "ttt-hybrid.g7-s14.catsem-attention-free-exploration.v1"
              and p["authority"]["no_cuda"] is True
              and p["frozen_experiment"]["seeds"]==[137,257,389]
              and p["controls"]["wall_seconds_cap"]==180,
              "FROZEN_STUDY_OR_AUTHORITY_DRIFT")
        venv=(Path.home().resolve()/".local/share/ttt-hybrid"/
              ("g7-r3-"+custody.R3_SOURCE_SHA[:12])/"venv/bin/python")
        guard(venv.is_file(),"QUALIFIED_PINNED_R3_VENV_NOT_FOUND")
        env=dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"]="1"
        env["CUDA_VISIBLE_DEVICES"]="-1"
        env["OMP_NUM_THREADS"]="1"
        env["MKL_NUM_THREADS"]="1"
        p=subprocess.run(
            [str(venv),"-I","-B",str(script),
             "--expected-sha",args.expected_sha,
             "--protocol",str(protocol),
             "--receipt",str(report)],
            cwd=ROOT,env=env,text=True,capture_output=True,
            check=False,timeout=220)
        guard(report.is_file(),"SANITIZED_SCIENCE_RECEIPT_NOT_CREATED")
        r=json.loads(report.read_text(encoding="utf-8"))
        guard(p.returncode==0
              and r.get("candidate_sha")==args.expected_sha
              and r.get("status") in (
                  "ATTENTION_FREE_EPISODIC_SKETCH_RETRIEVAL_POSITIVE_ENGINEERING_EXPLORATION",
                  "ATTENTION_FREE_SKETCH_RETRIEVAL_FROZEN_LIMITS_NOT_MET")
              and r.get("installed_packages_changed") is False
              and r.get("does_mamba_or_ttt_or_cuda_execution") is False
              and r.get("no_training") is True
              and r.get("learned_parameters")==0
              and r.get("heldout_episodes_per_seed")==192
              and r.get("peak_allocated_sketch_state_bytes",0)<=16777216,
              "S14_CPU_EXPLORATION_OR_CUSTODY_INVALID")
        print("G7_S14_ADMISSION=PASS_MEASURED_CPU_ONLY",flush=True)
        print("G7_S14_SCIENCE="+r["status"],flush=True)
        print("G7_S14_NO_GPU=PASS",flush=True)
        return 0
    except (custody.Blocked,ValueError,KeyError,OSError,subprocess.TimeoutExpired,
            SyntaxError) as exc:
        print("G7_S14_ADMISSION=BLOCKED_"+type(exc).__name__,flush=True)
        return 2

if __name__=="__main__":
    raise SystemExit(main())
