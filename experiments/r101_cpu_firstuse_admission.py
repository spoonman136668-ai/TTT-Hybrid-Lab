"""R101 scoped one-shot CPU scientific admission, not a controller or scheduler.

This file is development source; only an *additional append-only owner approval*
commit binds its exact parent and original first workflow run. The immutable
freeze-only proposal stays frozen (the original freeze's authority flags remain
false). This separate owner decision never grants GPU, merge or promotion.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = "spoonman136668-ai/TTT-Hybrid-Lab"
BRANCH = "research/r101-unknown-operator-induction-20261116"
JOB = "r101-unknown-original"
PR = 297
APPROVAL_PATH = "experiments/r101_cpu_firstuse_authorization.json"
AUTH_PATH = ROOT / APPROVAL_PATH
SEAL_PATH = ROOT / "experiments/r101_scientific_firstuse_seal.json"
FIRSTUSE_SCHEMA = "ttt-hybrid.r101.owner-scoped-firstuse.v1"
# Separate proposal freeze remains unchanged, including its original false
# authority flags. The user expressly approved only this one CPU experiment.
FROZEN_BLOB = "38146a196b59bc78ca66d43d2179e8e465dd54fa"
SEEDS = (1002511,1002523,1002537)


def approval():
    if not AUTH_PATH.is_file():
        raise PermissionError("R101_PROTECTED_ORIGINAL_FIRST_USE_NOT_AUTHORIZED")
    payload = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != FIRSTUSE_SCHEMA
        or payload.get("scope") != "ONE_ORIGINAL_THREE_SEED_GITHUB_HOSTED_CPU_TEST_ONLY"
        or payload.get("approved_on") != "2026-11-16"
        or payload.get("owner_approval") != "USER_EXPLICITLY_AUTHORIZED_R101_CPU_FIRST_USE"
        or payload.get("pr_number") != PR
        or payload.get("branch") != BRANCH
        or tuple(payload.get("seeds",())) != SEEDS
        or payload.get("immutable_freeze_blob") != FROZEN_BLOB
        or payload.get("no_prior_source_replay") is not True
        or payload.get("no_gpu_merge_promotion_forge") is not True
        or not isinstance(payload.get("exact_parent_sha"),str)
        or len(payload["exact_parent_sha"]) != 40
    ):
        raise PermissionError("R101_FIRST_USE_OWNER_SCOPE_MANIFEST_INVALID")
    return payload


def _git(*arguments):
    result = subprocess.run(
        ("git",)+arguments,cwd=ROOT,stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,text=True,timeout=10,check=False)
    if result.returncode:
        raise PermissionError("R101_EXACT_SOURCE_GIT_VERIFICATION_FAILED")
    return result.stdout.strip()


def _read_api(url, token):
    req = urllib.request.Request(url,headers={
        "Authorization":"Bearer "+token,
        "Accept":"application/vnd.github+json",
        "X-GitHub-Api-Version":"2022-11-28"})
    with urllib.request.urlopen(req,timeout=12) as response:
        return json.loads(response.read(2500000).decode("utf-8"))


def _first_attempt_ci_custody(sha, run_id):
    """Independently consult GitHub's source-bound, original workflow runs.

    Refuse a later run of the same SHA (or a rerun, checked separately).
    Refuse originals unless the same-SHA G5 and G7 have passed.
    """
    token = os.getenv("GITHUB_TOKEN","")
    if not token:
        raise PermissionError("R101_NO_CI_READ_TOKEN_FOR_FIRST_USE_CUSTODY")
    root = "https://api.github.com/repos/"+REPO
    q = "?"+urllib.parse.urlencode({
        "head_sha":sha,"event":"pull_request","per_page":100})
    try:
        workflow = _read_api(
            root+"/actions/workflows/g4-cpu-qualification.yml/runs"+q,token)
        g4 = [x for x in workflow.get("workflow_runs",[])
              if x.get("head_sha")==sha and x.get("name")=="G4 CPU qualification"]
        if not g4 or min(int(x["id"]) for x in g4)!=run_id:
            raise PermissionError("R101_NOT_CANONICAL_FIRST_RUN_FOR_EXACT_SHA")
        first = next(x for x in g4 if int(x["id"])==run_id)
        if first.get("run_attempt")!=1 or first.get("event")!="pull_request":
            raise PermissionError("R101_NOT_GITHUB_FIRST_ATTEMPT")
        # A bounded in-job read, never new dispatch or polling automation.
        for _ in range(16):
            related = _read_api(root+"/actions/runs"+q,token)
            rows = [x for x in related.get("workflow_runs",[])
                    if x.get("head_sha")==sha]
            success = {x.get("name") for x in rows
                       if x.get("conclusion")=="success"}
            if {"G5 authentic TTT CPU","G7 pinned Mamba2 CPU import"} <= success:
                return {"g4_run_id":run_id,
                        "same_sha_G5_G7_verified":True,
                        "original_attempt":1}
            if any(x.get("name") in ("G5 authentic TTT CPU",
                                      "G7 pinned Mamba2 CPU import")
                   and x.get("conclusion") not in (None,"success") for x in rows):
                raise PermissionError("R101_SAME_SHA_G5_G7_FAIL")
            time.sleep(5)
        raise PermissionError("R101_SAME_SHA_G5_G7_NOT_YET_VERIFIED")
    except PermissionError:
        # Deliberate custody rejection is NOT a transport problem.
        raise
    except (OSError,ValueError,KeyError) as exc:
        raise PermissionError(
            "R101_INDEPENDENT_GITHUB_CI_CUSTODY_UNAVAILABLE") from exc


# Original R101 scientific source is permanently consumed. This is the
# post-scoring source seal, NOT a new scientific source authorization.
SPENT_ORIGINAL_SHA = "ad21af19d342d239207629d562a0b4990e5fd163"
SPENT_RUN_ID = 43893297635
SPENT_SEEDS = SEEDS


def admit(seed, candidate_sha=None):
    """All attempted R101 first-use cohorts are spent; no science replay."""
    raise PermissionError("R101_ORIGINAL_FIRST_USE_PERMANENTLY_CONSUMED")
