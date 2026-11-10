"""P58 frozen DEV: long-horizon actual-action reward discrimination, no neural model.

This runs P54's published equation transcription (not an installed official Gymnasium
package) with pre-registered controllers, seeds and stop criteria. No future physical
observations are given to any policy. No scientific learning advantage is claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from native_p54_mountaincar_factual_bridge import published_reset, published_step, source_git_blob

SOURCE_BLOB = '9ecfc5822989336b3bbb14f29b8ebf84296c7039'
FIRST_SEED = 985100
LAST_SEED = 985195
HORIZON = 256
POLICIES = ('neutral','random3','always_right','momentum_bangbang')


def verify_source():
    import native_p54_mountaincar_factual_bridge as source
    if source_git_blob(Path(source.__file__).read_bytes()) != SOURCE_BLOB:
        raise ValueError('P58_SOURCE_BLOB_MISMATCH')


def action_for(policy: str, sensor: np.ndarray, rng: np.random.Generator | None):
    if policy == 'neutral':
        return 0.0
    if policy == 'always_right':
        return 1.0
    if policy == 'momentum_bangbang':
        return -1.0 if float(sensor[1]) < 0.00005 else 1.0
    if policy == 'random3':
        if rng is None:
            raise ValueError('P58_RANDOM_SOURCE_REQUIRED')
        return float(rng.choice(np.array([-1.,0.,1.],dtype=np.float32)))
    raise ValueError('P58_UNREGISTERED_POLICY')


def episode(seed: int, policy: str):
    if not (type(seed) is int and FIRST_SEED <= seed <= LAST_SEED):
        raise ValueError('P58_UNREGISTERED_SEED')
    if policy not in POLICIES:
        raise ValueError('P58_UNREGISTERED_POLICY')
    state = published_reset(seed)
    rng = np.random.default_rng(990000+seed) if policy == 'random3' else None
    actions=[]; transitions=[]; rewards=[]; hit=False
    digest=hashlib.sha256()
    digest.update(state.tobytes())
    for t in range(HORIZON):
        a=action_for(policy,state,rng)
        if not math.isfinite(a) or not -1.0 <= a <= 1.0:
            raise ValueError('P58_OUT_OF_BOUNDS_POLICY_ACTION')
        before=state.copy()
        observed=published_step(before,a)  # truly executed policy action in reference simulator
        state=observed.next_sensor
        if not np.isfinite(state).all() or not math.isfinite(observed.reward):
            raise ValueError('P58_NONFINITE_PHYSICAL_OUTCOME')
        actions.append(a); transitions.append((before.copy(),state.copy()))
        rewards.append(observed.reward)
        digest.update(np.asarray([a],np.float32).tobytes())
        digest.update(state.tobytes())
        digest.update(np.asarray([observed.reward],np.float64).tobytes())
        digest.update(bytes([int(observed.terminated)]))
        if observed.terminated:
            hit=True
            break
    n=len(actions)
    assert 1<=n<=HORIZON
    return {
       'seed':seed,'policy':policy,'success':hit,'steps':n,
       'cumulative_reward':float(math.fsum(rewards)),
       'energy':float(sum(a*a for a in actions)),
       'final_position':float(state[0]),'final_velocity':float(state[1]),
       'trace_sha256':digest.hexdigest(),
       # Retain only a compact proof of real action execution in receipt.
       'first_actions':actions[:8],
       'last_actions':actions[-8:]
    }


def study():
    verify_source()
    start=time.perf_counter()
    entries={name:[] for name in POLICIES}
    for seed in range(FIRST_SEED,LAST_SEED+1):
        for name in POLICIES:
            entries[name].append(episode(seed,name))
    # Original frozen source-reward custody gate: terminal bonus occurs once
    # and every real control action incurs precisely its published energy cost.
    exact_rewards = all(
        abs(v['cumulative_reward'] - (100.0 if v['success'] else 0.0) + 0.1*v['energy']) <= 1e-8
        for rows in entries.values() for v in rows
    )
    # Source-only evaluator contrast, not supplied to any controller.
    one_step_effects = []
    for seed in range(FIRST_SEED,LAST_SEED+1):
        initial=published_reset(seed)
        left=published_step(initial,-1.0).next_sensor
        right=published_step(initial,+1.0).next_sensor
        one_step_effects.append(float(np.abs((right-left)*0.5).mean()))
    out={}
    for name,rows in entries.items():
        successful=[v for v in rows if v['success']]
        out[name]={
          'episodes':len(rows),
          'success_count':len(successful),
          'success_rate':len(successful)/len(rows),
          'mean_undiscounted_return':float(np.mean([r['cumulative_reward'] for r in rows])),
          'mean_executed_steps':float(np.mean([r['steps'] for r in rows])),
          'mean_actual_motor_energy':float(np.mean([r['energy'] for r in rows])),
          'median_successful_hitting_steps':None if not successful else float(np.median([r['steps'] for r in successful])),
          'mean_final_position':float(np.mean([r['final_position'] for r in rows])),
          'mean_final_velocity':float(np.mean([r['final_velocity'] for r in rows])),
          'source_episode_trace_sha256':hashlib.sha256((''.join(r['trace_sha256'] for r in rows)).encode()).hexdigest(),
          'first_and_last_episode': [rows[0],rows[-1]]
        }
    m=out['momentum_bangbang']['success_rate']
    n=out['neutral']['success_rate']
    random=out['random3']['success_rate']
    gates={
      'momentum_goal_rate_at_least_75_percent':m>=0.75,
      'neutral_goal_rate_at_most_1_percent':n<=0.01,
      'momentum_minus_random_goal_rate_at_least_50_points':m-random>=0.5,
      'all_source_actions_and_terminations_finite':True,
      'source_pinned_git_blob':True,
      'source_reward_replay_exact':exact_rewards
    }
    return {
      'schema':'ttt-hybrid.p58.long-horizon-source-discrimination-dev-result.v1',
      'source_kind':'PUBLISHED_EQUATIONS_TRANSCRIPTION_NOT_OFFICIAL_GYMNASIUM_RUNTIME',
      'source_git_blob':SOURCE_BLOB,'seeds':[FIRST_SEED,LAST_SEED],
      'horizon':HORIZON,'policies':out,'gates':gates,
      'mean_abs_one_step_action_effect_per_motor_unit':float(np.mean(one_step_effects)),
      'all_frozen_gates':all(gates.values()),
      'elapsed_seconds':time.perf_counter()-start,
      'science_first_use':False,'neural_learning_tested':False,
      'FORGE_controller_or_queue_changed':False
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=study()
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'result':result['all_frozen_gates'],'gates':result['gates'],
       'policies':{n:{k:v for k,v in row.items() if not k.startswith('first_and_last')}
                   for n,row in result['policies'].items()},
       'time':result['elapsed_seconds']},indent=2,sort_keys=True))

if __name__ == '__main__':
    main()
