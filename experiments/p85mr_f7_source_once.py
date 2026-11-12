"""F7 one-shot external FIT source capture. NEVER import in normal CI.

All first-use effects require exact owner, branch, workflow, Windows runner and
native Ubuntu WSL/ext4 filesystem. A permanent O_EXCL owner claim, session
claims, and every constructor/reset/do intent+observed effect are fsynced with
their parent directory. Unknown outcome is NON-REPLAYABLE. F3/F5 historical
data and spent routes are never used.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"experiments"))
from experiments import p85mr_f7_protocol as cfg
import p85mr_fitcal_owner_firstuse as legacy

BRANCH="research/p85mr-f7-native-raw-bypass-freshfit-20261112"
WORKFLOW=".github/workflows/p85mr-f7-fit-only-once.yml"
PROJECT="spoonman136668-ai/TTT-Hybrid-Lab"
SOURCE_SHA="c94e9a4cd3641588b6b8507e71ccee5c27032bdf"
EXCLUSIVE_ROOT="f7-freshfit-1821000000-once-v1"
PACKAGES={"numpy":"1.26.4","gymnasium":"0.29.1","networkx":"3.4.2",
          "pydash":"8.0.5","toposort":"1.10","cloudpickle":"3.0.0",
          "farama-notifications":"0.0.4","typing-extensions":"4.12.2"}
DEADLINE_SECONDS=2400


def _canon(d):
    return (json.dumps(d,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def _fsync_dir(path):
    if os.name!="posix":
        raise PermissionError("F7_POSIX_NATIVE_CUSTODY_REQUIRED")
    flags=os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW
    fd=os.open(path,flags)
    try:os.fsync(fd)
    finally:os.close(fd)


def _exclusive(path,obj):
    raw=_canon(obj)
    flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW
    fd=os.open(path,flags,0o600)
    with os.fdopen(fd,"wb") as out:
        out.write(raw);out.flush();os.fsync(out.fileno())
    _fsync_dir(path.parent)
    return hashlib.sha256(raw).hexdigest()


def _mkdir(path):
    path.mkdir(mode=0o700,exist_ok=False)
    _fsync_dir(path.parent)
    _fsync_dir(path)


def _git(root,*args):
    run=subprocess.run(["git","-C",str(root),*args],
                       capture_output=True,text=True,timeout=20,check=False)
    if run.returncode:raise ValueError("F7_GIT_CUSTODY_FAIL")
    return run.stdout.strip()


def _owner(environ):
    f=cfg.frozen()
    expect={
        "GITHUB_REPOSITORY":PROJECT,
        "GITHUB_ACTOR":"spoonman136668-ai",
        "GITHUB_REF":"refs/heads/"+BRANCH,
        "GITHUB_EVENT_NAME":"push",
        "GITHUB_WORKFLOW_REF":PROJECT+"/"+WORKFLOW+"@refs/heads/"+BRANCH,
        "GITHUB_RUN_NUMBER":"1","GITHUB_RUN_ATTEMPT":"1",
        "RUNNER_NAME":"TTT-HYBRID-GPU-WIN","RUNNER_OS":"Windows",
        "RUNNER_ARCH":"X64","P85MR_F7_FIT_SOURCE_ONLY":"1",
    }
    if any(environ.get(k)!=v for k,v in expect.items()):
        raise PermissionError("F7_UNTRUSTED_OWNER_OR_REPLAY_ATTEMPT")
    if (os.name!="posix" or not hasattr(os,"O_DIRECTORY")
        or not hasattr(signal,"SIGALRM") or os.getuid()!=1000
        or sys.version_info[:2]!=(3,11)
        or "microsoft" not in os.uname().release.lower()):
        raise PermissionError("F7_NOT_OWNER_WSL_LINUX")
    who=os.environ.get("USER","")
    correct=f"/home/{who}/.local/share/ttt-hybrid/p85mr-causalgym-cpu311-r3/venv/bin/python"
    if (not re.fullmatch(r"[a-z_][a-z0-9_-]{0,31}",who)
        or Path(sys.executable).resolve()!=Path(correct).resolve()):
        raise PermissionError("F7_UNPINNED_SIMULATOR_RUNTIME")
    sha=environ.get("GITHUB_SHA","")
    run=environ.get("GITHUB_RUN_ID","")
    if (not re.fullmatch("[0-9a-f]{40}",sha)
        or not re.fullmatch("[0-9]{6,20}",run)):
        raise PermissionError("F7_MISSING_GITHUB_SOURCE_CUSTODY")
    for name,version in PACKAGES.items():
        if importlib.metadata.version(name)!=version:
            raise PermissionError("F7_PINNED_SIMULATOR_DEPENDENCY_DRIFT")
    return {"sha":sha,"run_id":run,"actor":"spoonman136668-ai",
            "runner":"TTT-HYBRID-GPU-WIN","frozen_blob":cfg.FREEZE_BLOB}


def _preflight(project,source,owner):
    protocol=cfg.frozen()
    if _git(project,"rev-parse","HEAD")!=owner["sha"]:
        raise ValueError("F7_PROJECT_SHA_DRIFT")
    if _git(project,"status","--porcelain","--untracked-files=all"):
        raise ValueError("F7_DIRTY_PROJECT")
    if _git(source,"rev-parse","HEAD")!=SOURCE_SHA:
        raise ValueError("F7_UPSTREAM_SHA_DRIFT")
    if _git(source,"status","--porcelain","--untracked-files=all"):
        raise ValueError("F7_UPSTREAM_DIRTY")
    for rel,expected in protocol["scope"]["source_blobs"].items():
        if cfg.git_blob((source/rel).read_bytes())!=expected:
            raise ValueError("F7_UPSTREAM_BLOB_DRIFT")
    if protocol["authority"]["approve_only_F7_FIT_new_source"] is not True:
        raise PermissionError("F7_FIT_APPROVAL_MISSING")
    if protocol["authority"]["approve_F7_CAL"] or protocol["authority"]["science_effects_started"]:
        raise PermissionError("F7_EXCLUSIVE_FIRSTUSE_DRIFT")


def _native_root():
    home=Path.home().resolve(strict=True)
    if str(home)!="/home/"+home.name:
        raise PermissionError("F7_HOME_MUST_BE_WSL_EXT4")
    root=home/".local"/"share"/"ttt-hybrid"
    if root.is_symlink() or not root.is_dir():
        raise PermissionError("F7_NATIVE_EXT4_ROOT_MISSING")
    # WSL ext4 VHD: fail closed on Windows-mounted or overlay persistence.
    mounts=Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    if " - ext4 " not in mounts:
        raise PermissionError("F7_NATIVE_EXT4_NOT_FOUND")
    _fsync_dir(root)
    return root


def _source_state(obs):
    return list(cfg.state4(legacy._state(obs)))


def _one_episode(factory,task,index,owner,root,deadline):
    seed=cfg.seed(index)
    session=root/f"fit-{index:03d}-{seed}"
    _mkdir(session)
    _exclusive(session/"claim.json",{
        "schema":"f7.source-session-claim.v1","index":index,
        "seed":seed,"run":owner["run_id"],"source_sha":SOURCE_SHA,
        "mode":"FIT_ONLY","no_retry":True,
    })
    _exclusive(session/"constructor.intent.json",{
        "index":index,"seed":seed,"run":owner["run_id"],"attempt":True,
    })
    env=factory(max_episode_steps=cfg.HORIZON,task=task,
                wind_mean=0,wind_std=.01,init_theta_mean=0,init_theta_std=0)
    try:
        _exclusive(session/"constructor.outcome.json",{
            "index":index,"seed":seed,"constructed":True,
        })
        _exclusive(session/"reset.intent.json",{
            "index":index,"seed":seed,"attempt":True,
        })
        observed=env.reset(seed=seed)
        if not isinstance(observed,(tuple,list)) or len(observed)!=2:
            raise ValueError("F7_UNKNOWN_RESET_OUTCOME")
        before=_source_state(observed[0])
        _exclusive(session/"reset.outcome.json",{
            "seed":seed,"state":before,"session":index,
        })
        transitions=[]
        for step in range(1,cfg.HORIZON+1):
            if time.monotonic()>deadline:
                raise TimeoutError("F7_SOURCE_WALL_CAP")
            action=cfg.factual_policy(index,step,before)
            expected=list(before)
            _exclusive(session/f"{step:03d}.intent.json",{
                "index":index,"seed":seed,"step":step,"action":action,
                "state_before":expected,"run":owner["run_id"],
                "unselected_effect":"UNOBSERVED",
            })
            calls=[0]
            def policy(obs):
                calls[0]+=1
                if calls[0]!=1 or any(abs(a-b)>1e-5 for a,b in zip(
                    _source_state(obs),expected)):
                    raise ValueError("F7_POLICY_INPUT_OR_DUPLICATE_CALL")
                return action
            observed=env.do(policy)
            if not isinstance(observed,(tuple,list)) or len(observed)!=5 or calls[0]!=1:
                raise ValueError("F7_UNKNOWN_FACTUAL_DO_OUTCOME")
            after,reward,terminated,truncated,info=observed
            if (not isinstance(info,dict) or type(info.get("action")) is not int
                or info.get("action")!=action
                or type(reward) not in (int,float) or not math.isfinite(reward)
                or type(terminated) is not bool or type(truncated) is not bool):
                raise ValueError("F7_ACTION_CREDIT_OR_RESULT_INVALID")
            after=_source_state(after)
            receipt=f"ghrun:{owner['run_id']}:F7-fit:{index}:{step}"
            event={
                "event_id":step,"state_before":before,
                "executed_action":action,"state_after":after,
                "reward":float(reward),"terminated":terminated,
                "truncated":truncated,"mode":"do","receipt_ref":receipt,
            }
            _exclusive(session/f"{step:03d}.outcome.json",event)
            transitions.append(event)
            before=after
            if terminated or truncated:break
        result={
            "world_id":f"f7-fit-{index:03d}-{seed}",
            "episode_seed":seed,"transitions":transitions,
            "probe_event_ids":list(range(2,len(transitions)+1)),
        }
        _exclusive(session/"world.json",result)
        return result
    finally:
        env.close()  # closing a possibly-failed session NEVER retries a do.


def collect(*,project_root,source_root,environ=None):
    owner=_owner(os.environ if environ is None else environ)
    project=Path(project_root).resolve(strict=True)
    src=Path(source_root).resolve(strict=True)
    _preflight(project,src,owner)
    native=_native_root()
    claim=native/EXCLUSIVE_ROOT
    _mkdir(claim)
    _exclusive(claim/"ATTEMPT.json",{
        "schema":"ttt-hybrid.f7-firstuse-attempt.v1",
        "run":owner,"firstuse":True,
        "new_fit_seed_base":cfg.SEED_BASE,
        "max_actions":cfg.MAX_ACTIONS,
        "CAL_OOD_NEGATIVE_opened":False,
        "replay_permitted":False,"science_pass":False,
    })
    started=time.monotonic()
    previous=signal.getsignal(signal.SIGALRM)
    def expired(_sig,_frame):raise TimeoutError("F7_EXCLUSIVE_OWNER_DEADLINE")
    signal.signal(signal.SIGALRM,expired)
    signal.alarm(DEADLINE_SECONDS)
    worlds=[];effect_count=0
    try:
        factory,task=legacy._load_scoped_source(src)
        for index in range(cfg.FIT_WORLDS):
            world=_one_episode(factory,task,index,owner,claim,started+DEADLINE_SECONDS)
            worlds.append(world)
            effect_count+=len(world["transitions"])
            if effect_count>cfg.MAX_ACTIONS:
                raise ValueError("F7_MAX_EFFECT_CAP")
        if len(worlds)!=cfg.FIT_WORLDS:
            raise ValueError("F7_MISSING_WORLDS")
        capture={
            "schema":"ttt-hybrid.p80mr.cartpole-do-capture.v1",
            "upstream_repo":"CausalAILab/Causal-Gymnasium",
            "upstream_commit":SOURCE_SHA,
            "upstream_path":"causal_gym/envs/cartpole_wind.py",
            "upstream_blob":cfg.frozen()["scope"]["source_blobs"]["causal_gym/envs/cartpole_wind.py"],
            "source_id":f"p85mr-f7-fit-{owner['run_id']}",
            "producer_id":"ttt-hybrid-f7-owner",
            "custody_ref":"ghrun:"+owner["run_id"],
            "worlds":worlds,
        }
        capture_sha=_exclusive(claim/"FIT_CAPTURE.json",capture)
        result={
            "schema":"ttt-hybrid.f7-owner-fit-source-receipt.v1",
            "status":"F7_128_FIT_SOURCE_COLLECTED_DEV_NOT_SCIENCE_PASS",
            "owner":owner,"capture_sha256":capture_sha,
            "source_effects":effect_count,"source_sessions":len(worlds),
            "short_worlds":sum(len(w["transitions"])<2 for w in worlds),
            "source_wall_s":time.monotonic()-started,
            "CAL_OOD_NEGATIVE_created":False,
            "new_firstuse_effects_consumed":True,"replay_allowed":False,
            "scientific_pass":False,
        }
        _exclusive(claim/"RESULT.json",result)
        # Safe artifact export, after durable source work; never a replay path.
        export=project/"local_runs"/"p85mr_f7_owner_source.json"
        export.parent.mkdir(parents=True,exist_ok=True)
        _exclusive(export,{
            "owner":owner,"result":result,"native_ext4_receipt":str(claim),
            "external_actions":effect_count,
        })
        exported=project/"local_runs"/"p85mr_f7_fit_capture.json"
        _exclusive(exported,capture)
        if hashlib.sha256(exported.read_bytes()).hexdigest()!=capture_sha:
            raise ValueError("F7_EXPORTED_CAPTURE_HASH_DRIFT")
        return result
    except BaseException as e:
        if not (claim/"FAILED.json").exists():
            _exclusive(claim/"FAILED.json",{
                "schema":"ttt-hybrid.f7-owner-abort.v1",
                "failure_type":type(e).__name__,
                "completed_source_sessions":len(worlds),
                "effects_recorded":effect_count,
                "CAL_OOD_NEGATIVE_created":False,
                "replay_allowed":False,"scientific_pass":False,
            })
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM,previous)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",type=Path,required=True)
    ap.add_argument("--upstream",type=Path,required=True)
    a=ap.parse_args()
    print(json.dumps(collect(project_root=a.project,source_root=a.upstream),
                     sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
