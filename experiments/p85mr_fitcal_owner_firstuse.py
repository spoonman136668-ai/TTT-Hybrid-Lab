"""One-shot operator-approved external CartPoleWind FIT/CAL source collector.

NEVER called from tests, automatic PR CI or a scheduler. No runner dispatch,
external downloads, model training, scoring, OOD source or paid APIs.
A GH Actions environment is checked defensively; its claims MUST ALSO be
verified independently against the actual GitHub run and owner custody.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
import types

from p84mr_partition_preflight import preflight_partition, planned_discrete_action, proposed_seed
from p80mr_causalgym_trace_adapter import adapt_capture, UPSTREAM_REPO, UPSTREAM_SHA, UPSTREAM_BLOB, UPSTREAM_PATH

SCOPE_BLOB = "cde1faee905027b4069deb329149e73d48292394"
PROJECT = "spoonman136668-ai/TTT-Hybrid-Lab"
SOURCE_BLOBS = {
    "causal_gym/envs/cartpole_wind.py":"d65a6a31941709dc9201259d80dd4ebbcf48b4fa",
    "causal_gym/core/pch.py":"ff37780c6b791877c70583b8d5f699be7fcd8817",
    "causal_gym/core/task.py":"668262f3f5c88fa2865b08ef10bbf96b09a3b73e",
    "causal_gym/core/scm.py":"f66f2638c797cfeaf0e96d4e13c06d4fabc2c8d5",
}
DEPS={"numpy":"1.26.4","gymnasium":"0.29.1","networkx":"3.4.2",
      "pydash":"8.0.5","toposort":"1.10"}
COHORTS=(("fit",128),("cal",64))
MAX_STEPS=96
MAX_ACTIONS=18432
MAX_SECONDS=2400
MAX_OUTPUT=18000000


def git(root,*args):
    return subprocess.run(["git","-C",str(root),*args],check=True,
        capture_output=True,text=True,timeout=15).stdout.strip()


def blob(raw):
    return hashlib.sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest()


def locked_scope():
    raw=Path(__file__).with_name("p85mr_fitcal_firstuse_authorization_freeze.json").read_bytes()
    if blob(raw)!=SCOPE_BLOB:
        raise ValueError("P85F1_SCOPE_DRIFT")
    scope=json.loads(raw)
    if (scope["schema"]!="ttt-hybrid.p85mr-first-source-fitcal-authorized-scope-freeze.v1"
        or scope["operator_approval"]["approved_in_chat"] is not True
        or scope["operator_approval"]["effects_started"] is not False
        or scope["operator_approval"]["owner_execution_route_verified"] is not False
        or scope["upstream"]["sha"]!=UPSTREAM_SHA):
        raise ValueError("P85F1_BAD_AUTHORIZATION_SCOPE")
    return scope


def owner_context(environ):
    needed={"GITHUB_REPOSITORY":PROJECT,"GITHUB_ACTOR":"spoonman136668-ai",
            "GITHUB_EVENT_NAME":"workflow_dispatch","GITHUB_REF":"refs/heads/main",
            "GITHUB_REF_PROTECTED":"true","GITHUB_RUN_ATTEMPT":"1",
            "RUNNER_NAME":"TTT-HYBRID-GPU-WIN"}
    if any(environ.get(k)!=v for k,v in needed.items()):
        raise PermissionError("P85F1_TRUSTED_OWNER_ROUTE_NOT_VERIFIED")
    run=environ.get("GITHUB_RUN_ID","")
    sha=environ.get("GITHUB_SHA","")
    if (not run.isdecimal() or not 1<=len(run)<=20 or len(sha)!=40
        or any(ch not in "0123456789abcdef" for ch in sha)):
        raise PermissionError("P85F1_GITHUB_CUSTODY_MISSING")
    return {"run_id":run,"project_sha":sha,"actor":needed["GITHUB_ACTOR"],
            "runner":needed["RUNNER_NAME"],"attempt":1}


def preflight(source_root,project_root,owner):
    scope=locked_scope()
    partition=preflight_partition()
    if partition["cross_split_collisions"] or partition["new_unique_seeds"]!=448:
        raise ValueError("P85F1_SEED_CUSTODY_FAILURE")
    src=Path(source_root).resolve(strict=True)
    project=Path(project_root).resolve(strict=True)
    for root,sha,origins in (
        (src,UPSTREAM_SHA,("https://github.com/CausalAILab/Causal-Gymnasium.git",
                           "https://github.com/CausalAILab/Causal-Gymnasium",
                           "git@github.com:CausalAILab/Causal-Gymnasium.git")),
        (project,owner["project_sha"],("https://github.com/"+PROJECT+".git",
                                       "https://github.com/"+PROJECT,
                                       "git@github.com:"+PROJECT+".git"))):
        if git(root,"rev-parse","HEAD")!=sha:
            raise ValueError("P85F1_CHECKOUT_SHA_MISMATCH")
        if git(root,"status","--porcelain"):
            raise ValueError("P85F1_DIRTY_SOURCE")
        if git(root,"remote","get-url","origin") not in origins:
            raise ValueError("P85F1_SOURCE_ORIGIN_MISMATCH")
    for rel,expected in SOURCE_BLOBS.items():
        f=src/rel
        if not f.is_file() or blob(f.read_bytes())!=expected:
            raise ValueError("P85F1_SOURCE_BLOB_MISMATCH")
    if sys.version_info[:2]!=(3,11):
        raise ValueError("P85F1_PYTHON_VERSION")
    for pkg,version in DEPS.items():
        if importlib.metadata.version(pkg)!=version:
            raise ValueError("P85F1_DEPENDENCY_DRIFT")
    if [(k,scope["cohorts"][k]["n"]) for k,_ in COHORTS]!=list(COHORTS):
        raise ValueError("P85F1_COHORT_DRIFT")
    return src,project,scope


def _state(obs):
    if not isinstance(obs,(list,tuple)) and callable(getattr(obs,"tolist",None)):
        obs=obs.tolist()
    if (not isinstance(obs,(list,tuple)) or len(obs)!=4
        or any(type(x) not in (int,float) or not math.isfinite(x) for x in obs)):
        raise ValueError("P85F1_BAD_STATE")
    return [float(x) for x in obs]


def _load_scoped_source(src):
    # Exact upstream CartPoleWind module; avoid unrelated envs barrel imports.
    if "causal_gym" in sys.modules:
        raise ValueError("P85F1_UPSTREAM_ALREADY_IMPORTED")
    pack=types.ModuleType("causal_gym")
    pack.__path__=[str(src/"causal_gym")]
    envs=types.ModuleType("causal_gym.envs")
    envs.__path__=[str(src/"causal_gym"/"envs")]
    sys.modules["causal_gym"]=pack
    sys.modules["causal_gym.envs"]=envs
    imported=importlib.import_module("causal_gym.envs.cartpole_wind")
    tasks=importlib.import_module("causal_gym.core.task")
    return imported.CartPoleWindPCH,tasks.Task(learning_regime=tasks.LearningRegime.do)


def observed_episode(factory,task,split,index,regime,run_id,deadline):
    """Execute only factual do effects, exactly once per step/seed."""
    seed=proposed_seed(split,index)
    if time.monotonic()>deadline:
        raise TimeoutError("P85F1_EXPIRED")
    env=factory(max_episode_steps=MAX_STEPS,task=task,
        wind_mean=regime["wind_mean"],wind_std=regime["wind_std"],
        init_theta_mean=regime["init_theta_mean"],init_theta_std=regime["init_theta_std"])
    events=[]
    try:
        reset=env.reset(seed=seed)
        if not isinstance(reset,(tuple,list)) or len(reset)!=2 or not isinstance(reset[1],dict):
            raise ValueError("P85F1_BAD_RESET")
        previous=_state(reset[0])
        for step in range(1,MAX_STEPS+1):
            if time.monotonic()>deadline:
                raise TimeoutError("P85F1_EXPIRED")
            desired=planned_discrete_action(split,index,step)
            seen=[0]
            def policy(obs):
                seen[0]+=1
                if seen[0]!=1 or any(abs(a-b)>1e-5 for a,b in zip(_state(obs),previous)):
                    raise ValueError("P85F1_WRONG_POLICY_OBSERVATION")
                return desired
            # Effectful: one call only, no retries even if outcome unknown.
            result=env.do(policy)
            if (not isinstance(result,(tuple,list)) or len(result)!=5 or seen[0]!=1):
                raise ValueError("P85F1_UNKNOWN_ACTION_OUTCOME")
            after,reward,terminated,truncated,info=result
            if (not isinstance(info,dict) or type(info.get("action")) is not int
                or info["action"]!=desired):
                raise ValueError("P85F1_ACTION_MISMATCH")
            if (type(reward) not in (int,float) or not math.isfinite(reward)
                or type(terminated) is not bool or type(truncated) is not bool):
                raise ValueError("P85F1_BAD_EFFECT_RECEIPT")
            current=_state(after)
            events.append({
                "event_id":step,"state_before":list(previous),
                "executed_action":desired,"state_after":current,"reward":float(reward),
                "terminated":terminated,"truncated":truncated,"mode":"do",
                "receipt_ref":f"ghrun:{run_id}:{split}:{index}:{step}"})
            previous=current
            if terminated or truncated:
                break
    finally:
        env.close()  # once only; no compensating simulation
    return {"world_id":f"{split}-{index:03d}-{seed}",
            "episode_seed":seed,"transitions":events,
            "probe_event_ids":list(range(2,len(events)+1))}


def _put_new(path,object_):
    raw=(json.dumps(object_,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    if len(raw)>MAX_OUTPUT:
        raise ValueError("P85F1_EVIDENCE_BUDGET")
    with path.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    return hashlib.sha256(raw).hexdigest()


def _append_progress(path, event):
    raw=(json.dumps(event,sort_keys=True,allow_nan=False)+"\n").encode()
    if path.exists() and path.stat().st_size+len(raw)>MAX_OUTPUT:
        raise ValueError("P85F1_PROGRESS_BUDGET")
    with path.open("ab") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def collect_one_shot(*,source_root,project_root,environ=None):
    owner=owner_context(os.environ if environ is None else environ)
    # The authorized owner route executes in WSL Linux, never on GH-hosted
    # PR runners. A hard alarm bounds a stalled external env.do call.
    if (os.name!="posix" or not hasattr(signal,"SIGALRM") or
            "microsoft" not in os.uname().release.lower()):
        raise PermissionError("P85F1_WRONG_EXECUTION_OS")
    src,project,scope=preflight(source_root,project_root,owner)
    base=project/"evidence"/"p85mr_fitcal_firstuse"
    base.mkdir(parents=True,exist_ok=True)
    # Fixed attempt directory. Never replay by changing GitHub run id.
    attempt=base/"ONCE"
    attempt.mkdir(mode=0o700,exist_ok=False)
    _put_new(attempt/"ATTEMPT.json",{
        "status":"ATTEMPTED_NO_REPLAY","owner_context":owner,
        "upstream_sha":UPSTREAM_SHA,"cohorts":["fit","cal"],
        "max_actions":MAX_ACTIONS,"ood_opened":False})
    begun=time.monotonic()
    previous_handler=signal.getsignal(signal.SIGALRM)
    def _time_limit(_signum,_frame):
        raise TimeoutError("P85F1_HARD_WALL_CLOCK_STOP")
    signal.signal(signal.SIGALRM,_time_limit)
    signal.alarm(MAX_SECONDS)
    total=0
    summary={}
    try:
        factory,task=_load_scoped_source(src)
        for split,n in COHORTS:
            regime=scope["cohorts"][split]
            accepted=[]
            short=[]
            for index in range(n):
                world=observed_episode(factory,task,split,index,regime,owner["run_id"],begun+MAX_SECONDS)
                # Preserve every completed episode BEFORE the next effectful
                # reset, including early-short and terminal observations.
                _append_progress(attempt/(split+".ndjson"),world)
                total+=len(world["transitions"])
                if total>MAX_ACTIONS:
                    raise ValueError("P85F1_ACTION_BUDGET")
                if len(world["transitions"])<2:
                    short.append(index)
                else:
                    accepted.append(world)
            if len(short)/n > 0.15:
                raise ValueError("P85F1_TOO_MANY_SHORT_EPISODES")
            if not accepted:
                raise ValueError("P85F1_EMPTY_SOURCE")
            capture={
                "schema":"ttt-hybrid.p80mr.cartpole-do-capture.v1",
                "upstream_repo":UPSTREAM_REPO,"upstream_commit":UPSTREAM_SHA,
                "upstream_path":UPSTREAM_PATH,"upstream_blob":UPSTREAM_BLOB,
                "source_id":f"p85mr-{split}-{owner['run_id']}",
                "producer_id":"ttt-hybrid-owner",
                "custody_ref":f"ghrun:{owner['run_id']}",
                "worlds":accepted,
            }
            if adapt_capture(capture)["custody"]["scientific_pass"] is not False:
                raise ValueError("P85F1_ADMISSION_BYPASS")
            digest=_put_new(attempt/(split+".json"),capture)
            summary[split]={"scheduled":n,"valid":len(accepted),
                            "short_episode_indices":short,"sha256":digest}
        out={"schema":"ttt-hybrid.p85mr-fitcal-firstuse-unqualified-receipt.v1",
             "status":"SOURCE_COLLECTED_WITNESS_VERIFICATION_PENDING",
             "owner":owner,"upstream_sha":UPSTREAM_SHA,"cohorts":summary,
             "executed_actions":total,"wall_seconds":time.monotonic()-begun,
             "ood_created":False,"external_dev_created":False,
             "trained_baselines":False,"science_pass":False}
        _put_new(attempt/"RESULT.json",out)
        return out
    except BaseException as exc:
        _put_new(attempt/"FAILED.json",{"status":"ABORTED_NO_REPLAY",
            "failure_type":type(exc).__name__,"actions_so_far":total,
            "ood_created":False,"science_pass":False})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM,previous_handler)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-checkout",required=True)
    parser.add_argument("--project-checkout",required=True)
    parser.add_argument("--approved-once",action="store_true")
    args=parser.parse_args()
    if not args.approved_once:
        raise PermissionError("P85F1_EXPLICIT_EFFECTFUL_FLAG_REQUIRED")
    collect_one_shot(source_root=args.source_checkout,project_root=args.project_checkout)


if __name__=="__main__":
    main()
