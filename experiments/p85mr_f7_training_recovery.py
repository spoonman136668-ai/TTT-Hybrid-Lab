"""F7 single-use *TRAINING ONLY* from the already-consumed original FIT source.

Reads the prior native ext4 O_EXCL journal; never imports the original
simulator, resets an environment, opens external sessions, or repeats do().
The original first-use owner identity is immutable and DISTINCT from this
new training-only workflow identity. A single new fit training attempt is
allowed; any unknown output is preserved and is never replayed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"experiments"))

from experiments import p85mr_f7_protocol as cfg
from experiments import p85mr_f7_train_fit as trainer

FREEZE_BLOB="fb0538ec47e2bcb3862915049979299318a2dd20"
PARENT_SHA="e683462e328d4db795b56c3dd70f841a951fcbe1"
BRANCH="research/p85mr-f7-native-raw-bypass-freshfit-20261112"
WORKFLOW=".github/workflows/p85mr-f7-fit-training-recovery-once.yml"
SOURCE_ROOT="f7-freshfit-1821000000-once-v1"
RECOVERY_ROOT="f7-fit-training-recovery-43195349254-once-v1"
ORIGINAL_SHA="57ff8520e0f2c94c913e35a2aa19c6162843f3e3"
ORIGINAL_RUN="43195349254"
CAPTURE_SHA="7df8e2a2dc276cb90fbcac91619083dec7911453d5af10d2ccbdcfd45e5007d4"
SUMMARY_SHA="bc31da3abf0a0367f8e3f49985cb6d1d0e9f0e6da7c38c97cfc6c1646ec08ece"
SOURCE_OWNER={
  "actor":"spoonman136668-ai",
  "runner":"TTT-HYBRID-GPU-WIN",
  "sha":ORIGINAL_SHA,
  "run_id":ORIGINAL_RUN,
  "frozen_blob":"31c6107766389b73a228c36cb2bfa2e1a0de1f24",
}


def locked():
    raw=(ROOT/"experiments/p85mr_f7_train_recovery_freeze.json").read_bytes()
    if cfg.git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("F7_RECOVERY_FROZEN_BLOB_DRIFT")
    freeze=json.loads(raw)
    a=freeze["original_collection"]
    p=freeze["parent"]
    c=freeze["recovery"]
    if (freeze.get("schema")!="ttt-hybrid.f7-one-shot-training-only-source-recovery.v1"
        or p["exact_sha"]!=PARENT_SHA
        or a["owner_run"]!=int(ORIGINAL_RUN)
        or a["sha"]!=ORIGINAL_SHA
        or a["source_capture_sha256"]!=CAPTURE_SHA
        or a["source_summary_sha256"]!=SUMMARY_SHA
        or a["source_actions_consumed"]!=12288
        or a["never_retry_source"] is not True
        or c["one_shot_only"] is not True
        or c["no_upstream_checkout"] is not True
        or c["no_source_api_import_constructor_reset_do"] is not True
        or c["workflow"]!=WORKFLOW
        or freeze["authority"]["no_CAL_or_OOD"] is not True
        or freeze["authority"]["new_external_firstuse"] is not False
        or freeze["science"]["scientific_PASS_possible"] is not False):
        raise ValueError("F7_RECOVERY_SCOPE_OR_AUTHORITY_DRIFT")
    cfg.frozen()
    return freeze


def owner(environ=None):
    locked()
    env=os.environ if environ is None else environ
    expect={
        "GITHUB_REPOSITORY":"spoonman136668-ai/TTT-Hybrid-Lab",
        "GITHUB_ACTOR":"spoonman136668-ai",
        "GITHUB_EVENT_NAME":"push",
        "GITHUB_REF":"refs/heads/"+BRANCH,
        "GITHUB_WORKFLOW_REF":"spoonman136668-ai/TTT-Hybrid-Lab/"+WORKFLOW+"@refs/heads/"+BRANCH,
        "GITHUB_RUN_NUMBER":"1","GITHUB_RUN_ATTEMPT":"1",
        "RUNNER_NAME":"TTT-HYBRID-GPU-WIN",
        "RUNNER_OS":"Windows","RUNNER_ARCH":"X64",
        "P85MR_F7_TRAIN_RECOVERY_ONLY":"1","CUDA_VISIBLE_DEVICES":"-1",
    }
    if any(env.get(k)!=v for k,v in expect.items()):
        raise PermissionError("F7_RECOVERY_NOT_UNIQUE_TRAINING_OWNER")
    if (os.name!="posix" or not hasattr(os,"O_DIRECTORY")
        or os.getuid()!=1000 or sys.version_info[:2]!=(3,11)
        or "microsoft" not in os.uname().release.lower()):
        raise PermissionError("F7_RECOVERY_NOT_NATIVE_OWNER_WSL")
    me=Path.home().name
    cpu=f"/home/{me}/.local/share/ttt-hybrid/g7-s25-cpu311-r1/venv/bin/python"
    if Path(sys.executable).resolve()!=Path(cpu).resolve():
        raise PermissionError("F7_RECOVERY_NOT_PINNED_CPU_RUNTIME")
    run=env.get("GITHUB_RUN_ID","")
    head=env.get("GITHUB_SHA","")
    if (not re.fullmatch(r"[0-9]{6,20}",run)
        or run==ORIGINAL_RUN or
        not re.fullmatch("[0-9a-f]{40}",head)
        or head==ORIGINAL_SHA):
        raise PermissionError("F7_RECOVERY_BAD_NEW_TRAINING_RUN_ID")
    return {"sha":head,"run_id":run,
            "actor":env["GITHUB_ACTOR"],
            "runner":env["RUNNER_NAME"],
            "original_firstuse_run":ORIGINAL_RUN}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canon(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),
                      allow_nan=False).encode("utf-8")+b"\n"


def _read_exact(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("F7_RECOVERY_MISSING_UNLINKED_NATIVE_FILE")
    raw=path.read_bytes()
    obj=json.loads(raw)
    if _canon(obj)!=raw:
        raise ValueError("F7_RECOVERY_NONCANONICAL_NATIVE_RECEIPT")
    return obj,raw


def _native_directory():
    if os.name!="posix":
        raise PermissionError("F7_RECOVERY_NEEDS_NATIVE_EXT4")
    base=(Path.home()/".local/share/ttt-hybrid").resolve(strict=True)
    if Path.home().resolve().as_posix()!=f"/home/{Path.home().name}":
        raise PermissionError("F7_RECOVERY_NOT_NATIVE_WSL_HOME")
    if base.is_symlink() or not base.is_dir():
        raise ValueError("F7_RECOVERY_INVALID_NATIVE_ROOT")
    # Longest mountpoint wins; a different ext4 volume cannot prove this root.
    path=base.as_posix()
    mounts=[]
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left,sep,right=line.partition(" - ")
        if not sep:continue
        f=left.split()
        rest=right.split()
        if len(f)>=5 and len(rest)>=1:
            m=f[4].replace("\\040"," ")
            if path==m or path.startswith(m.rstrip("/")+"/"):
                mounts.append((len(m),rest[0],m))
    if not mounts or max(mounts)[1]!="ext4":
        raise PermissionError("F7_RECOVERY_ORIGINAL_JOURNAL_NOT_NATIVE_EXT4")
    return base


def _fsync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)


def audit_original_source():
    """No journal changes or simulator access: compare original immutable evidence."""
    locked()
    base=_native_directory()
    root=base/SOURCE_ROOT
    if not root.is_dir() or root.is_symlink() or root.resolve().parent!=base:
        raise ValueError("F7_RECOVERY_ORIGINAL_FIRSTUSE_JOURNAL_MISSING")
    original,_=_read_exact(root/"ATTEMPT.json")
    result,_=_read_exact(root/"RESULT.json")
    capture,raw=_read_exact(root/"FIT_CAPTURE.json")
    if (original.get("run")!=SOURCE_OWNER
        or original.get("replay_permitted") is not False
        or result.get("owner")!=SOURCE_OWNER
        or result.get("replay_allowed") is not False
        or result.get("source_sessions")!=128
        or result.get("source_effects")!=12288
        or result.get("CAL_OOD_NEGATIVE_created") is not False
        or result.get("scientific_pass") is not False
        or result.get("capture_sha256")!=CAPTURE_SHA
        or result.get("status")!="F7_128_FIT_SOURCE_COLLECTED_DEV_NOT_SCIENCE_PASS"
        or _sha(raw)!=CAPTURE_SHA
        or len(capture.get("worlds",[]))!=128
        or (root/"FAILED.json").exists()):
        raise ValueError("F7_RECOVERY_ORIGINAL_FIRSTUSE_CUSTODY_MISMATCH")
    summary={
      "external_actions":result["source_effects"],
      "native_ext4_receipt":str(root),
      "owner":SOURCE_OWNER,
      "result":result,
    }
    if _sha(_canon(summary))!=SUMMARY_SHA:
        raise ValueError("F7_RECOVERY_ARCHIVED_OWNER_SUMMARY_MISMATCH")
    return root,summary


def _create_single_attempt(new_owner,summary):
    base=_native_directory()
    claim=base/RECOVERY_ROOT
    # No replay even if a partial training attempt creates only a directory.
    claim.mkdir(mode=0o700,exist_ok=False)
    _fsync_dir(base)
    _fsync_dir(claim)
    descriptor={
       "schema":"ttt-hybrid.f7-fit-training-only-attempt.v1",
       "owner":new_owner,
       "original_firstuse_run":ORIGINAL_RUN,
       "firstuse_source_sha256":CAPTURE_SHA,
       "source_summary_sha256":SUMMARY_SHA,
       "new_source_effects":0,
       "replay_allowed":False,
       "science_pass":False,
    }
    raw=_canon(descriptor)
    fd=os.open(claim/"ATTEMPT.json",os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as file:
        file.write(raw);file.flush();os.fsync(file.fileno())
    _fsync_dir(claim)
    return claim


def run(*,models,receipt):
    new_owner=owner()
    root,summary=audit_original_source()
    # Authenticate every factual action against original source native intent
    # and outcome O_EXCL records BEFORE making new training attempt.
    train,heldout,source_meta=trainer.audit_capture(
       root/"FIT_CAPTURE.json",_make_summary_file(summary),SOURCE_OWNER)
    claim=_create_single_attempt(new_owner,summary)
    data=trainer.train_admitted_fit(
       train=train,holdout=heldout,source_meta=source_meta,owner=new_owner,
       output=models,receipt=receipt)
    return {"status":data["status"],"FIT_candidate":data["FIT_development_mechanism_candidate"],
            "source_effects_new":0,"scientific_pass":False,"CAL_used":False,
            "original_firstuse_run":ORIGINAL_RUN}


def _make_summary_file(summary):
    # Caller passes a unique local_runs owner receipt path; do not copy/overwrite
    # the actual original journal or rename a prior source run's evidence.
    path=ROOT/"local_runs"/"p85mr_f7_recovery_original_summary.json"
    path.parent.mkdir(exist_ok=True)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,"wb") as out:
        out.write(_canon(summary));out.flush();os.fsync(out.fileno())
    return path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--models",type=Path,required=True)
    ap.add_argument("--receipt",type=Path,required=True)
    args=ap.parse_args()
    print(json.dumps(run(models=args.models,receipt=args.receipt),
                     sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
