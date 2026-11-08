"""S29: MIT Farama Gymnasium CartPole public engineering API smoke.

This is NOT CausalAILab/Causal-Gymnasium, not wind, not scientific first-use.
The pinned Farama package is loaded ONLY from a clean read-only Git checkout;
no installer, dependency resolution, network, training, latent physics or model.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.native_s28_causal_wind_offline import (
    FactualWindTransitionLedger,
    canonical_json,
    git_blob_sha,
)

FROZEN=ROOT/"experiments"/"s29_gymnasium_contract_smoke_frozen.json"
SCHEMA="ttt-hybrid.s29.mit-gymnasium-contract-smoke-result.v1"
PINS={
    "LICENSE":"ac10bf443756cdc02724943ac09021092ce9e737",
    "pyproject.toml":"f7af86ade9bb45181760cb9ec393d56436e3bc5f",
    "gymnasium/__init__.py":"a5066ac6bae2208d25896cabd8f2447f23fdcd3b",
    "gymnasium/envs/classic_control/cartpole.py":"9bd08a0153bc68f23776e83a975a6bbca6bb9a11",
}


class S29Blocked(RuntimeError):
    pass


def require(ok:bool,reason:str):
    if not ok:
        raise S29Blocked("S29_"+reason)


def frozen():
    v=json.loads(FROZEN.read_text("utf-8"))
    up=v["upstream"]; s=v["smoke"]; q=v["qualification"]
    require(
        v["schema"]=="ttt-hybrid.s29.mit-gymnasium-actual-api-engineering-smoke.v1"
        and v["source_parent_s28_sha"]=="1041d1496c7c39182694a508778d1defab07e26c"
        and up["repo"]=="Farama-Foundation/Gymnasium"
        and up["exact_commit"]=="17eff00220d210beda933f78b0e52850022b0690"
        and up["git_blobs"]==PINS and up["license"]=="MIT"
        and s["only_env_id"]=="CartPole-v1"
        and s["public_engineering_smoke_seeds"]==[42,43]
        and s["repeat_per_seed"]==2 and s["max_steps_per_repetition"]==32
        and s["action_space_n"]==2 and s["expected_observation_dim"]==4
        and s["causal_gym_source_import_allowed"] is False
        and s["protected_first_use_allowed"] is False
        and q["no_environment_install_or_global_mutation"] is True
        and q["no_merge_or_model_promotion"] is True,
        "FROZEN_PROTOCOL_DRIFT")
    return v


def git(root:Path,*args:str)->str:
    try:
        process=subprocess.run(
            ["git","-C",str(root),*args],
            capture_output=True,text=True,check=False,timeout=20)
    except (OSError,subprocess.TimeoutExpired) as e:
        raise S29Blocked("S29_GIT_UNAVAILABLE") from e
    require(process.returncode==0,"GIT_FAILURE_"+args[0])
    return process.stdout.strip()


def source_identity(source:Path,v:dict)->dict:
    root=Path(source).resolve(strict=True)
    require(root.is_dir(),"UPSTREAM_SOURCE_MISSING")
    require(git(root,"rev-parse","HEAD")==v["upstream"]["exact_commit"],
            "GYMNASIUM_SOURCE_SHA_MISMATCH")
    require(git(root,"status","--porcelain","--untracked-files=all")=="",
            "GYMNASIUM_CHECKOUT_DIRTY")
    blobs={}
    for rel,pin in sorted(PINS.items()):
        file=root/rel
        require(file.is_file() and not file.is_symlink(),"REQUIRED_FILE_MISSING")
        part=root
        for segment in Path(rel).parts:
            part=part/segment
            require(not part.is_symlink(),"SOURCE_PARENT_SYMLINK")
        require(file.resolve(strict=True).is_relative_to(root),"SOURCE_PATH_ESCAPE")
        sha=git_blob_sha(file.read_bytes())
        require(sha==pin,"SOURCE_BLOB_CHANGED_"+rel)
        blobs[rel]=sha
    lic=(root/"LICENSE").read_text(encoding="utf-8")
    require("The MIT License" in lic and "Permission is hereby granted" in lic
            and "THE SOFTWARE IS PROVIDED" in lic
            and "Copyright (c) 2022 Farama Foundation" in lic
            and "@@ -" not in lic and len(lic)>900,
            "MIT_LICENSE_NOT_COMPLETE")
    py=tomllib.loads((root/"pyproject.toml").read_text(encoding="utf-8"))
    deps=py["project"]["dependencies"]
    require(py["project"]["name"]=="gymnasium"
            and py["project"]["license"]["text"]=="MIT License"
            and any(x.startswith("numpy ") for x in deps)
            and any(x.startswith("cloudpickle ") for x in deps)
            and any(x.startswith("typing-extensions ") for x in deps)
            and not any("gymnasium[all]" in x for x in deps),
            "SOURCE_PYPROJECT_DEPENDENCY_DRIFT")
    return {
        "status":"PASS_PINNED_FARAMA_SOURCE_AND_COMPLETE_MIT_LICENSE",
        "upstream_sha":v["upstream"]["exact_commit"],
        "files_verified":blobs,
        "license_sha256":hashlib.sha256(lic.encode("utf-8")).hexdigest(),
        "license_complete_mit":True,
        "license_applies_only_to_farama_not_causalailab":True,
        "upstream_code_vendored_to_hybrid_repo":False,
    }


def dependencies_preexisting()->dict:
    found={}
    for name in ("numpy","cloudpickle","typing_extensions"):
        try:
            found[name]=metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name]=None
    require(all(found.values()),"PREEXISTING_MINIMAL_DEPENDENCIES_MISSING")
    return {"status":"PASS_PREEXISTING_BASE_DEPENDENCY_METADATA",
            "distributions":found,"new_dependencies_installed":0,
            "broad_optional_extra_installed":False}


def _one_episode(gym,seed:int,max_steps:int)->dict:
    env=gym.make("CartPole-v1",render_mode=None)
    require(env.action_space.n==2 and env.observation_space.shape==(4,),
            "PUBLIC_ACTION_OBSERVATION_INTERFACE_DRIFT")
    ledger=FactualWindTransitionLedger()
    try:
        first,info=env.reset(seed=seed)
        ledger.start(first,info)
        for i in range(max_steps):
            action=i%2
            transition=env.step(action)
            event=ledger.add_completed_transition(action,transition)
            if event["terminated"] or event["truncated"]:
                break
        out=ledger.receipt()
        require(out["events"]>=1 and out["events"]<=max_steps,
                "SMOKE_EVENT_COUNT_BOUNDS")
        return out
    finally:
        env.close()


def run(source:Path)->dict:
    if any(os.environ.get(k,"1")!="1" for k in (
        "NATIVE_S29_ENGINEERING_ONLY","NATIVE_S28_ENGINEERING_ONLY")):
        raise S29Blocked("S29_SCIENCE_FIRST_USE_DENIED")
    t0=time.perf_counter()
    v=frozen()
    upstream=source_identity(source,v)
    deps=dependencies_preexisting()
    # Import only pinned MIT Gymnasium in the in-memory Python path; do not
    # use the host's potentially different global Gymnasium.
    sys.path.insert(0,str(Path(source).resolve()))
    gym=import_module("gymnasium")
    module=Path(gym.__file__).resolve(strict=True)
    require(module==Path(source).resolve()/"gymnasium"/"__init__.py",
            "IMPORTED_UNPINNED_GYMNASIUM")
    require("causal_gym" not in sys.modules,"UNAUTHORIZED_CAUSALGYM_IMPORT")
    records=[]
    max_steps=v["smoke"]["max_steps_per_repetition"]
    for seed in v["smoke"]["public_engineering_smoke_seeds"]:
        pair=[_one_episode(gym,seed,max_steps) for _ in range(2)]
        require(pair[0]["sha256"]==pair[1]["sha256"]
                and pair[0]["records"]==pair[1]["records"],
                "SAME_SEED_SAME_ACTIONS_NONDETERMINISTIC")
        records.append({
            "public_engineering_seed":seed,
            "episodes_replayed":2,
            "steps":pair[0]["events"],
            "factual_transition_sha256":pair[0]["sha256"],
            "factual_transitions":pair[0]["records"],
            "termination_flags":{
                "terminated":pair[0]["records"][-1]["terminated"],
                "truncated":pair[0]["records"][-1]["truncated"],
            },
            "bit_exact_within_this_os":True,
        })
    require("causal_gym" not in sys.modules,"CAUSALGYM_WAS_IMPORTED")
    return {
        "schema":SCHEMA,
        "status":"PASS_ACTUAL_FARAMA_CARTPOLE_ENGINEERING_SMOKE_NOT_CAUSAL_WIND",
        "upstream":upstream,"dependency_metadata":deps,
        "gymnasium_runtime_version_reported_by_source":gym.__version__,
        "environment":"CartPole-v1",
        "same_version_source_runtime_import_verified":True,
        "record_count":len(records),
        "total_episodes":4,
        "source_qualified_real_environment_steps":sum(x["steps"] for x in records)*2,
        "episodes":records,
        "frozen_protocol_sha256":hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        "original_protected_science_first_use":"NOT_RUN",
        "scientific_benchmark_or_transfer_claim":False,
        "real_causal_wind_environment_executed":False,
        "trained_agent_or_model_executed":False,
        "installed_any_package":False,
        "accepted_model_promoted":False,
        "elapsed_wall_seconds":round(time.perf_counter()-t0,3),
    }


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--receipt",type=Path,required=True)
    args=p.parse_args()
    result=run(args.source_root)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "status":result["status"],"total_episodes":result["total_episodes"],
        "source":result["upstream"]["upstream_sha"],
        "factual_steps":result["source_qualified_real_environment_steps"],
        "runtime_version":result["gymnasium_runtime_version_reported_by_source"],
        "protected_science_first_use":result["original_protected_science_first_use"],
        "causal_wind_executed":result["real_causal_wind_environment_executed"],
    },sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
