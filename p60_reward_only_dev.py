"""P60-POL: reward-only search on partially observed published MountainCar equations.

Only actual executed position and *completed* position changes enter the
searchable controllers; policy parameters are ranked by factual net returns.
This is a 3-parameter conventional CEM experiment, NOT native Hybrid training,
not official installed Gymnasium, not independent scientific first use.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
from native_p54_mountaincar_factual_bridge import published_reset,published_step

PHYSICS_GIT_BLOB='9ecfc5822989336b3bbb14f29b8ebf84296c7039'
P58_MOMENTUM_TRACE='1686baec1451b442b818a37f35843f75a3f0c8c42ae4a677c3b4f13d07652853'
TRAIN_SEEDS=tuple(range(989000,989008))
VALID_SEEDS=tuple(range(989100,989132))
OOD_A=tuple(range(989200,989264))
OOD_B=tuple(range(989400,989464))
MODEL_SEEDS=(9741,9763,9781)
HORIZON=256
GENERATIONS=6
CANDIDATES=20
ELITES=5
ARMS=('history_cem','memoryless_cem','history_random','wrong_credit_cem')
EVAL_ARMS=ARMS+('history_zero','position_delta_sign','full_state_momentum','random_actions')


def git_blob(raw:bytes)->str:
    return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()


def actual_action(theta,position,last_position,kind):
    if kind=='history':
        observed_delta=0. if last_position is None else (position-last_position)*100.
        features=(observed_delta,4*(position+.5),1.)
    elif kind=='memoryless':
        f=4*(position+.5)
        features=(f,f**3,1.)
    elif kind=='zero_history':
        features=(0.,4*(position+.5),1.)
    else: raise ValueError('P60_UNKNOWN_SEARCH_POLICY')
    return 1. if float(np.dot(theta,features))>=0 else -1.


def physical_episode(seed,policy,theta=None,*,low=-.6,high=-.4):
    if type(seed) is not int or seed<0:raise ValueError('P60_BAD_PHYSICAL_SEED')
    if policy not in ('history','memoryless','zero_history','position_delta_sign','full_state_momentum','random_actions'):
        raise ValueError('P60_UNKNOWN_PHYSICAL_POLICY')
    if policy in ('history','memoryless','zero_history'):
        if theta is None or np.asarray(theta).shape!=(3,) or not np.isfinite(theta).all():
            raise ValueError('P60_BAD_REAL_REWARD_CONTROLLER')
    state=published_reset(seed,low=low,high=high)
    random=np.random.default_rng(990000+seed) if policy=='random_actions' else None
    prev=None; total=0.;energy=0.;hit=False
    transcript=hashlib.sha256()
    transcript.update(state.tobytes())
    for t in range(HORIZON):
        pos=float(state[0])
        if policy in ('history','memoryless','zero_history'):
            action=actual_action(theta,pos,prev,policy)
        elif policy=='position_delta_sign':
            action=(-1. if prev is None or (pos-prev)<0.00005 else 1.)
        elif policy=='full_state_momentum':
            action=(-1. if float(state[1])<.00005 else 1.)
        else:
            action=float(random.choice(np.array([-1.,1.],dtype=np.float32)))
        observed=published_step(state,action)
        prev=pos;state=observed.next_sensor
        total+=observed.reward;energy+=action*action
        transcript.update(np.asarray([action],dtype=np.float32).tobytes())
        transcript.update(state.tobytes())
        transcript.update(np.asarray([observed.reward],dtype=np.float64).tobytes())
        transcript.update(bytes((int(observed.terminated),)))
        if observed.terminated:hit=True;break
    steps=t+1
    if abs(total-((100. if hit else 0.)-.1*energy))>1e-7:
        raise AssertionError('P60_FACTUAL_REWARD_ACCOUNTING_FAIL')
    return dict(success=bool(hit),net_return=float(total),steps=steps,
                physical_action_energy=float(energy),trace_sha256=transcript.hexdigest())


def check_original_p58_trace():
    """Validate local physics-equation mirror against original P58 exact source trace.

    This test reuses original P58 source ONLY for physics source validation;
    these consumed source seeds are NEVER used for P60 fit/evaluation.
    """
    accum=[]
    for s in range(985100,985196):
        accum.append(physical_episode(s,'full_state_momentum')['trace_sha256'])
    return hashlib.sha256(''.join(accum).encode()).hexdigest()


def fit_search(seed,arm):
    if seed not in MODEL_SEEDS or arm not in ARMS:raise ValueError('P60_FROZEN_FIT_SOURCE_REQUIRED')
    rng=np.random.default_rng(seed)
    mu=np.zeros(3);sd=np.array([1.,1.,.25])
    baseline_sd=sd.copy()
    memory='memoryless' if arm=='memoryless_cem' else 'history'
    started=time.perf_counter();evaluated=0;steps=0;best_true=float('-inf');best_observed=float('-inf');chosen=None
    hist=[]
    for gen in range(GENERATIONS):
        if arm=='history_random':
            candidates=rng.normal(0,baseline_sd,size=(CANDIDATES,3))
        else:
            candidates=rng.normal(mu,sd,size=(CANDIDATES,3))
        actual=np.empty(CANDIDATES,dtype=np.float64)
        success=np.empty(CANDIDATES,dtype=np.float64)
        for i,params in enumerate(candidates):
            episodes=[physical_episode(s,memory,params) for s in TRAIN_SEEDS]
            actual[i]=float(np.mean([v['net_return'] for v in episodes]))
            success[i]=float(np.mean([v['success'] for v in episodes]))
            steps+=sum(v['steps'] for v in episodes)
            evaluated+=len(episodes)
        credited=actual.copy()
        if arm=='wrong_credit_cem':
            credited=actual[rng.permutation(len(actual))]
        elite=np.argsort(-credited,kind='stable')[:ELITES]
        idx=int(np.argmax(credited))
        if credited[idx]>best_observed:
            best_observed=float(credited[idx]);best_true=float(actual[idx]);chosen=candidates[idx].copy()
        if arm not in ('history_random',):
            mu=.5*mu+.5*candidates[elite].mean(0)
            sd=.5*sd+.5*np.maximum(.05,candidates[elite].std(0))
        hist.append(dict(generation=gen,physical_mean_return=float(actual.mean()),
                         actual_goal_fraction=float(success.mean()),top_true=float(np.max(actual)),
                         best_credited=float(credited[idx]),elite_mean=mu.tolist(),elite_sigma=sd.tolist()))
    assert evaluated==GENERATIONS*CANDIDATES*len(TRAIN_SEEDS)
    assert chosen is not None
    return dict(seed=seed,arm=arm,theta=chosen.tolist(),true_training_reward_at_selection=best_true,
                training_return_used_for_selection=best_observed,train_episode_count=evaluated,
                physical_train_steps=steps,train_wall_seconds=time.perf_counter()-started,
                generation_history=hist)


def eval_arm(seeds,arm,theta=None,low=-.6,high=-.4):
    p={'history_cem':'history','memoryless_cem':'memoryless','history_random':'history',
       'wrong_credit_cem':'history','history_zero':'zero_history',
       'position_delta_sign':'position_delta_sign','full_state_momentum':'full_state_momentum',
       'random_actions':'random_actions'}[arm]
    started=time.perf_counter()
    rows=[physical_episode(s,p,theta,low=low,high=high) for s in seeds]
    return dict(episodes=len(rows),success_count=sum(v['success'] for v in rows),
                success_fraction=float(np.mean([v['success'] for v in rows])),
                mean_net_return=float(np.mean([v['net_return'] for v in rows])),
                total_physical_steps=sum(v['steps'] for v in rows),
                mean_steps=float(np.mean([v['steps'] for v in rows])),
                total_motor_energy=float(sum(v['physical_action_energy'] for v in rows)),
                trace_sha256=hashlib.sha256(''.join(v['trace_sha256'] for v in rows).encode()).hexdigest(),
                wall_seconds=time.perf_counter()-started)


def trial(seed):
    fit={a:fit_search(seed,a) for a in ARMS}
    evaluation={}
    for name,seeds,lo,hi in [('validation',VALID_SEEDS,-.6,-.4),('oodA',OOD_A,-.6,-.4),('oodB',OOD_B,-.75,-.25)]:
        states={}
        for arm in EVAL_ARMS:
            theta=fit['history_cem']['theta'] if arm=='history_zero' else fit[arm]['theta'] if arm in ARMS else None
            states[arm]=eval_arm(seeds,arm,theta,low=lo,high=hi)
        evaluation[name]=states
    gates={}
    for key in ('oodA','oodB'):
        v=evaluation[key];cand=v['history_cem'];mem=v['memoryless_cem'];rnd=v['random_actions'];fd=v['position_delta_sign'];hr=v['history_random'];wrong=v['wrong_credit_cem'];zero=v['history_zero']
        gates[key]={
            'history_success_at_least_90pct':cand['success_fraction']>=.90,
            'history_beats_memoryless_50pp':cand['success_fraction']>=mem['success_fraction']+.50,
            'history_beats_random_actions_50pp':cand['success_fraction']>=rnd['success_fraction']+.50,
            'history_near_finite_diff_5pp':cand['success_fraction']>=fd['success_fraction']-.05,
            'history_net_reward_near_finite_diff_10':cand['mean_net_return']>=fd['mean_net_return']-10,
            'strict_beats_equal_budget_random_search_5pp':cand['success_fraction']>=hr['success_fraction']+.05,
            'strict_beats_random_search_reward_5':cand['mean_net_return']>=hr['mean_net_return']+5,
            'strict_beats_reward_shuffle_25pp':cand['success_fraction']>=wrong['success_fraction']+.25,
            'history_necessity_50pp':cand['success_fraction']>=zero['success_fraction']+.50,
        }
    # Frozen controller sufficient stats restore exactly through JSON, not weights.
    saved=json.loads(json.dumps(fit['history_cem']['theta']))
    import numpy as np
    replay=np.array([actual_action(saved,float(x),float(x-.01),'history') for x in (-.45,-.1,.1)])
    raw=np.array([actual_action(fit['history_cem']['theta'],float(x),float(x-.01),'history') for x in (-.45,-.1,.1)])
    return dict(seed=seed,fit=fit,evaluation=evaluation,gates=gates,cold_exact=bool(np.array_equal(replay,raw)),
                frozen_all_gates=all(all(v for v in x.values()) for x in gates.values()) and np.array_equal(replay,raw))


def run():
    started=time.perf_counter()
    physics_trace=check_original_p58_trace()
    if physics_trace!=P58_MOMENTUM_TRACE:raise RuntimeError('P60_LOCAL_PHYSICS_SOURCE_TRACE_NOT_IDENTICAL_TO_PINNED_P58')
    rows=[trial(s) for s in MODEL_SEEDS]
    return dict(schema='ttt-hybrid.p60-pol.reward-only-hidden-velocity-search-authored-dev.v1',
        source_type='PINNED_PUBLISHED_EQUATIONS_MIRROR_NOT_OFFICIAL_INSTALLED_GYMNASIUM',
        physics_P58_exact_trace_match=True,physics_P58_trace_sha256=physics_trace,
        source_seeds=dict(train=[TRAIN_SEEDS[0],TRAIN_SEEDS[-1]],valid=[VALID_SEEDS[0],VALID_SEEDS[-1]],
                          oodA=[OOD_A[0],OOD_A[-1]],oodB=[OOD_B[0],OOD_B[-1]]),
        first_use='AUTHOR_GENERATED_DEV_NOT_INDEPENDENT_SCIENCE',
        prefreeze_pilot_excluded=True,model_seeds=list(MODEL_SEEDS),
        training_episode_budget_per_arm=GENERATIONS*CANDIDATES*len(TRAIN_SEEDS),
        rows=rows,all_frozen_gates=all(x['frozen_all_gates'] for x in rows),
        total_wall_seconds=time.perf_counter()-started,
        original_full_Windows_Ubuntu_regression='NOT_RUN',trained_neural_Hybrid_GRU_MLP='NOT_RUN',FORGE='UNTOUCHED')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--receipt',type=Path,required=True)
    args=ap.parse_args();out=run()
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'source':out['physics_P58_trace_sha256'],'all':out['all_frozen_gates'],
      'rows':[{ 'seed':r['seed'],'theta':r['fit']['history_cem']['theta'],'training_episode_counts':{k:v['train_episode_count'] for k,v in r['fit'].items()},
        'ood':{cohort:{arm:{k:v[k] for k in ('success_count','mean_net_return','total_physical_steps')} for arm,v in arms.items()} for cohort,arms in r['evaluation'].items() if cohort!='validation'},'gates':r['gates']} for r in out['rows']],
      'wall':out['total_wall_seconds']},indent=2))

if __name__=='__main__':main()
