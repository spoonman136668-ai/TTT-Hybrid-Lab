"""R103 owner-scoped single original CPU first-use admission, not a controller.

The immutable preimplementation freeze's false authority flags are NEVER
modified. Only a separate, final one-file owner manifest child commit can
enable this exact GitHub-hosted, first-attempt, three-seed original source.
No GPU, merged code, FORGE integration, scheduler, runner or paid API.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
REPO="spoonman136668-ai/TTT-Hybrid-Lab"
BRANCH="research/r103-relational-binding-20261116"
JOB="r103-relational-original"
PR=299
APPROVAL_PATH="experiments/r103_cpu_firstuse_authorization.json"
AUTH_PATH=ROOT/APPROVAL_PATH
FIRSTUSE_SCHEMA="ttt-hybrid.r103.owner-scoped-firstuse.v1"
FROZEN_BLOB="e9a12e3e5224abf162ae00e82b0781987c00484d"
SEEDS=(1003011,1003023,1003037)
MAX_BUDGET_SECONDS=900


def approval():
    if not AUTH_PATH.is_file():
        raise PermissionError("R103_PROTECTED_ORIGINAL_FIRST_USE_NOT_AUTHORIZED")
    f=json.loads(AUTH_PATH.read_text(encoding="utf-8"))
    if (f.get("schema")!=FIRSTUSE_SCHEMA
        or f.get("scope")!="ONE_ORIGINAL_THREE_SEED_GITHUB_HOSTED_CPU_TEST_ONLY"
        or not isinstance(f.get("approved_on"),str) or len(f["approved_on"])!=10
        or f.get("owner_approval")!="USER_EXPLICITLY_AUTHORIZED_R103_CPU_FIRST_USE"
        or f.get("repo")!=REPO or f.get("pr_number")!=PR or f.get("branch")!=BRANCH
        or tuple(f.get("seeds",()))!=SEEDS
        or f.get("immutable_freeze_blob")!=FROZEN_BLOB
        or f.get("exact_parent_sha") is None
        or not isinstance(f["exact_parent_sha"],str)
        or len(f["exact_parent_sha"])!=40
        or f.get("no_prior_source_replay") is not True
        or f.get("no_gpu_merge_promotion_forge") is not True
        or f.get("no_r102_original_cohort_reuse") is not True
        or f.get("per_seed_CPU_budget_seconds")!=MAX_BUDGET_SECONDS):
        raise PermissionError("R103_OWNER_FIRSTUSE_MANIFEST_INVALID")
    return f


def _git(*args):
    result=subprocess.run(("git",)+args,cwd=ROOT,capture_output=True,
                          text=True,timeout=10,check=False)
    if result.returncode:
        raise PermissionError("R103_EXACT_SOURCE_GIT_CUSTODY_FAILED")
    return result.stdout.strip()


def _read_api(url,token):
    req=urllib.request.Request(url,headers={
        "Authorization":"Bearer "+token,
        "Accept":"application/vnd.github+json",
        "X-GitHub-Api-Version":"2022-11-28"})
    with urllib.request.urlopen(req,timeout=15) as resp:
        return json.loads(resp.read(3000000).decode("utf-8"))


def _first_attempt_ci_custody(sha,run_id):
    """Verify canonical original G4 and SAME SOURCE G5/G7 before FIT source."""
    token=os.getenv("GITHUB_TOKEN","")
    if not token:
        raise PermissionError("R103_MISSING_GITHUB_ACTIONS_READ_TOKEN")
    root="https://api.github.com/repos/"+REPO
    query="?"+urllib.parse.urlencode({
        "head_sha":sha,"event":"pull_request","per_page":100})
    try:
        workflow=_read_api(
            root+"/actions/workflows/g4-cpu-qualification.yml/runs"+query,token)
        g4=[x for x in workflow.get("workflow_runs",[])
            if x.get("head_sha")==sha and x.get("name")=="G4 CPU qualification"]
        if (not g4 or min(int(x["id"]) for x in g4)!=run_id):
            raise PermissionError("R103_NOT_FIRST_CANONICAL_G4_FOR_EXACT_SHA")
        first=next(x for x in g4 if int(x["id"])==run_id)
        if (first.get("run_attempt")!=1 or first.get("event")!="pull_request"):
            raise PermissionError("R103_NOT_ORIGINAL_FIRST_GITHUB_ATTEMPT")
        # In addition to workflow needs, independently prove both CPU
        # platforms successfully ran full regression at the original SHA.
        for _ in range(24):
            jobs=_read_api(root+f"/actions/runs/{run_id}/jobs?per_page=100",token)
            verified={
                x["name"] for x in jobs.get("jobs",[])
                if x.get("conclusion")=="success"
            }
            if not {"CPU qualification / ubuntu-latest",
                    "CPU qualification / windows-latest"} <= verified:
                if any(x.get("name") in (
                        "CPU qualification / ubuntu-latest",
                        "CPU qualification / windows-latest")
                       and x.get("conclusion") not in (None,"success")
                       for x in jobs.get("jobs",[])):
                    raise PermissionError("R103_G4_FULL_DUAL_OS_FAILED")
                time.sleep(5)
                continue
            related=_read_api(root+"/actions/runs"+query,token)
            same=[x for x in related.get("workflow_runs",[])
                  if x.get("head_sha")==sha]
            successes={x.get("name") for x in same
                       if x.get("conclusion")=="success"}
            if {"G5 authentic TTT CPU","G7 pinned Mamba2 CPU import"} <= successes:
                branch_ref=_read_api(
                    root+"/git/ref/heads/"+urllib.parse.quote(BRANCH,safe="/"),token)
                if branch_ref.get("object",{}).get("sha")!=sha:
                    raise PermissionError("R103_RESEARCH_BRANCH_DRIFT_BEFORE_FIRST_USE")
                return {"original_G4_run_id":run_id,"run_attempt":1,
                        "same_source_G4_linux_windows":True,
                        "same_source_G5_G7":True,
                        "research_branch_head_sha_verified":sha}
            if any(x.get("name") in ("G5 authentic TTT CPU",
                                     "G7 pinned Mamba2 CPU import")
                   and x.get("conclusion") not in (None,"success")
                   for x in same):
                raise PermissionError("R103_SAME_SHA_NATIVE_OR_MAMBA_QUALIFICATION_FAIL")
            time.sleep(5)
        raise PermissionError("R103_SAME_SHA_CUSTODY_NOT_FULLY_QUALIFIED")
    except PermissionError:
        raise
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise PermissionError("R103_INDEPENDENT_GITHUB_CUSTODY_UNAVAILABLE") from exc


def admit(seed,candidate_sha=None):
    """Only an explicit later one-file owner approval final commit can admit."""
    from experiments import r103_relational_binding_diagnostic as r103
    immutable=r103.frozen()
    if (immutable["authority"]["protected_scored_first_use"]
        or tuple(immutable["source"]["seeds"])!=SEEDS
        or immutable["training"]["per_seed_total_CPU_wall_budget_seconds"]!=MAX_BUDGET_SECONDS):
        raise PermissionError("R103_IMMUTABLE_FROZEN_SCOPE_DRIFT")
    scope=approval()  # Fail before touching environment/network if absent.
    required={
        "GITHUB_ACTIONS":"true",
        "GITHUB_EVENT_NAME":"pull_request",
        "GITHUB_REPOSITORY":REPO,
        "GITHUB_HEAD_REF":BRANCH,
        "GITHUB_JOB":JOB,
        "RUNNER_OS":"Linux",
        "RUNNER_ENVIRONMENT":"github-hosted",
        "GITHUB_RUN_ATTEMPT":"1",
        "R103_SEED":str(seed),
    }
    if (seed not in SEEDS or any(os.getenv(k)!=v for k,v in required.items())
        or os.getenv("GITHUB_REF")!=f"refs/pull/{PR}/merge"):
        raise PermissionError("R103_PROTECTED_ORIGINAL_FIRST_USE_NOT_AUTHORIZED")
    run_id_text=os.getenv("GITHUB_RUN_ID","")
    sha=os.getenv("EXPECTED_CANDIDATE_SHA","")
    if (not run_id_text.isdecimal() or int(run_id_text)<=0
        or not isinstance(candidate_sha,str) or len(sha)!=40
        or sha!=candidate_sha):
        raise PermissionError("R103_EXACT_SOURCE_RUN_SEED_NOT_ADMITTED")
    event_path=os.getenv("GITHUB_EVENT_PATH","")
    if not event_path or not Path(event_path).is_file():
        raise PermissionError("R103_GITHUB_PR_EVENT_MISSING")
    event=json.loads(Path(event_path).read_text(encoding="utf-8"))
    pr=event.get("pull_request",{})
    if (pr.get("number")!=PR or pr.get("head",{}).get("sha")!=sha
        or pr.get("head",{}).get("ref")!=BRANCH
        or pr.get("head",{}).get("repo",{}).get("full_name")!=REPO):
        raise PermissionError("R103_PR_OWNER_HEAD_SOURCE_MISMATCH")
    if (_git("rev-parse","HEAD")!=sha
        or _git("rev-parse","HEAD^")!=scope["exact_parent_sha"]):
        raise PermissionError("R103_FINAL_SINGLE_FILE_APPROVAL_PARENT_MISMATCH")
    if (_git("diff-tree","--no-commit-id","--name-only","-r","HEAD")
        !=APPROVAL_PATH):
        raise PermissionError("R103_FINAL_APPROVAL_COMMIT_HAS_EXTRA_CHANGES")
    if (_git("status","--porcelain")
        or _git("hash-object",str(r103.FREEZE))!=FROZEN_BLOB):
        raise PermissionError("R103_DIRTY_SOURCE_OR_FROZEN_BLOB_DRIFT")
    custody=_first_attempt_ci_custody(sha,int(run_id_text))
    return {"seed":seed,"source_sha":sha,"original_run_id":int(run_id_text),
            "owner_scope":scope["scope"],
            "approval_parent":scope["exact_parent_sha"],**custody}
