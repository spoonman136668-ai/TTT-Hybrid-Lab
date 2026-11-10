"""P59: Source-pinned installed Gymnasium runtime qualification, not scientific first use.

Refuses to claim full-runtime parity without the actual installed package, the
exact upstream class source blob, Gymnasium.make registration and wrappers.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import inspect
import json
import math
import platform
import time
from pathlib import Path
from typing import Any
import numpy as np
from native_p54_mountaincar_factual_bridge import (
    ENV_ID, UPSTREAM_COMMIT, UPSTREAM_GIT_BLOB, published_reset, published_step, source_git_blob)
from experiments.p58_long_horizon_action_value_dev import action_for

REGISTERED_TIME_LIMIT = 999
ENGINEERING_ATOL = 5e-7
P58_REUSED_SOURCE_SEEDS = tuple(range(985100,985116))
POLICIES = ('neutral','momentum_bangbang')
HORIZON = 256

class OfficialPackageUnavailable(RuntimeError): pass
class SourceQualificationFailure(RuntimeError): pass


def require_source_file(path: str | Path) -> str:
    try: payload=Path(path).read_bytes()
    except (OSError,TypeError) as e:
        raise SourceQualificationFailure('P59_ORIGINAL_SOURCE_FILE_UNREADABLE') from e
    blob=source_git_blob(payload)
    if blob!=UPSTREAM_GIT_BLOB:
        raise SourceQualificationFailure('P59_PINNED_ORIGINAL_SOURCE_MISMATCH: '+blob)
    return blob


def import_installed_gymnasium() -> Any:
    try: return importlib.import_module('gymnasium')
    except (ModuleNotFoundError,ImportError) as e:
        raise OfficialPackageUnavailable('P59_OFFICIAL_GYMNASIUM_NOT_INSTALLED') from e


def make_verified_env(gym: Any):
    try: env=gym.make(ENV_ID,render_mode=None)
    except Exception as e:
        raise SourceQualificationFailure('P59_OFFICIAL_ENV_REGISTRATION_FAILED') from e
    try:
        cls=type(env.unwrapped)
        if (cls.__name__!='Continuous_MountainCarEnv' or
            cls.__module__!='gymnasium.envs.classic_control.continuous_mountain_car'):
            raise SourceQualificationFailure('P59_UNEXPECTED_OFFICIAL_ENV_CLASS')
        file=inspect.getsourcefile(cls)
        if not file:
            raise SourceQualificationFailure('P59_OFFICIAL_ENV_SOURCE_MISSING')
        blob=require_source_file(file)
        spec=getattr(env,'spec',None)
        if spec is None or getattr(spec,'id',None)!=ENV_ID:
            raise SourceQualificationFailure('P59_OFFICIAL_ENV_SPEC_DRIFT')
        if spec.max_episode_steps!=REGISTERED_TIME_LIMIT:
            raise SourceQualificationFailure('P59_OFFICIAL_TIMELIMIT_SPEC_DRIFT')
        return env,str(Path(file).resolve()),blob
    except Exception:
        env.close()
        raise


def verify_episode(env: Any,seed: int,policy: str) -> dict:
    actual,info=env.reset(seed=seed)
    if not isinstance(info,dict): raise AssertionError('P59_RESET_INFO_NOT_DICT')
    reference=published_reset(seed)
    maxerr=float(np.abs(actual.astype(np.float64)-reference.astype(np.float64)).max())
    if maxerr>ENGINEERING_ATOL:raise AssertionError('P59_SEEDED_RESET_PARITY_FAIL')
    digest=hashlib.sha256();digest.update(np.asarray(actual,np.float32).tobytes())
    cumulative=0.0
    for step in range(HORIZON):
        action=action_for(policy,actual,None) # no future outcome supplied to policy
        observed,reward,term,trunc,info=env.step(np.array([action],np.float32))
        expected=published_step(reference,action)
        delta=float(np.abs(observed.astype(np.float64)-expected.next_sensor.astype(np.float64)).max())
        maxerr=max(maxerr,delta)
        if (delta>ENGINEERING_ATOL or abs(float(reward)-expected.reward)>1e-9 or
            bool(term)!=expected.terminated or bool(trunc) or
            not np.isfinite(observed).all() or not math.isfinite(float(reward))):
            raise AssertionError('P59_OFFICIAL_STEP_PARITY_FAIL')
        actual=np.asarray(observed,np.float32);reference=expected.next_sensor
        cumulative+=float(reward)
        digest.update(np.array([action],np.float32).tobytes())
        digest.update(actual.tobytes());digest.update(np.array([reward],np.float64).tobytes())
        digest.update(bytes((int(bool(term)),int(bool(trunc)))))
        if term:break
    return dict(seed=seed,policy=policy,actual_completed_steps=step+1,
                reached_goal=bool(term),cumulative_reward=cumulative,
                max_state_abs_error=maxerr,trace_sha256=digest.hexdigest())


def verify_wrapper_time_limit(env: Any) -> dict:
    env.reset(seed=P58_REUSED_SOURCE_SEEDS[0])
    for i in range(REGISTERED_TIME_LIMIT):
        obs,reward,term,trunc,info=env.step(np.array([0.],np.float32))
        if term or bool(trunc)!=(i==REGISTERED_TIME_LIMIT-1):
            raise AssertionError('P59_OFFICIAL_TIMELIMIT_BEHAVIOR_DRIFT')
        if float(reward)!=0.:raise AssertionError('P59_NEUTRAL_MOTOR_REWARD_DRIFT')
    return dict(actual_steps=i+1,final_truncated=bool(trunc),final_terminated=bool(term))


def qualify() -> dict:
    gym=import_installed_gymnasium();start=time.perf_counter()
    env,filename,blob=make_verified_env(gym);env.close()
    episodes=[]
    for seed in P58_REUSED_SOURCE_SEEDS:
        for policy in POLICIES:
            env,_,_=make_verified_env(gym)
            try:episodes.append(verify_episode(env,seed,policy))
            finally:env.close()
    env,_,_=make_verified_env(gym)
    try:limit=verify_wrapper_time_limit(env)
    finally:env.close()
    return dict(schema='ttt-hybrid.p59.official-installed-gymnasium-runtime-qualification.v1',
                classification='PINNED_INSTALLED_RUNTIME_ENGINEERING_NOT_SCIENTIFIC_FIRST_USE',
                status='PASS',upstream_commit=UPSTREAM_COMMIT,installed_file=filename,
                pinned_file_git_blob=blob,gymnasium_version=getattr(gym,'__version__',None),
                seeds=list(P58_REUSED_SOURCE_SEEDS),policies=list(POLICIES),horizon=HORIZON,
                atol=ENGINEERING_ATOL,episodes=episodes,official_wrapper=limit,
                total_completed_steps=sum(x['actual_completed_steps'] for x in episodes),
                max_state_abs_error=max(x['max_state_abs_error'] for x in episodes),
                elapsed_seconds=time.perf_counter()-start,python=platform.python_version(),
                scientific_learning_benefit='NOT_TESTED',FORGE='UNCHANGED')


def main(argv: list[str] | None=None) -> int:
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);args=p.parse_args(argv)
    try:result=qualify();code=0
    except OfficialPackageUnavailable as e:
        result=dict(schema='ttt-hybrid.p59.official-installed-gymnasium-runtime-qualification.v1',
                    status='UNAVAILABLE',reason=str(e),upstream_commit=UPSTREAM_COMMIT,
                    pinned_file_git_blob=UPSTREAM_GIT_BLOB,scientific_learning_benefit='NOT_TESTED');code=2
    except (SourceQualificationFailure,AssertionError) as e:
        result=dict(schema='ttt-hybrid.p59.official-installed-gymnasium-runtime-qualification.v1',
                    status='FAIL',reason=str(e),upstream_commit=UPSTREAM_COMMIT,
                    pinned_file_git_blob=UPSTREAM_GIT_BLOB,scientific_learning_benefit='NOT_TESTED');code=3
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({x:result.get(x) for x in ('status','reason','pinned_file_git_blob',
      'gymnasium_version','total_completed_steps','max_state_abs_error','scientific_learning_benefit')},sort_keys=True))
    return code

if __name__=='__main__': raise SystemExit(main())
