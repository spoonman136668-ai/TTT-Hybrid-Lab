"""P51 DEV: active rank repair with two completed physical interventions.

No tokenization, neural training, external first-use, or unrealized-outcome access.
All arms share the exact six-action factual history, physical parameters and
exogenous noise tape, and each executes exactly two *actual* next actions.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import time
from pathlib import Path
import numpy as np

N = 8
T = np.arange(N, dtype=np.float64) / N
SOURCE_SEEDS = {800101: (.35, .9), 800103: (.95, 1.35), 800107: (1.4, 2.)}
WORLDS = 1536
ARMS = ("orthogonal", "doptimal", "flip", "random", "passive")
POLICIES = ("constant", "feedback", "sticky", "random")
KINDS = ("stable", "causal_reverse", "exogenous_jump", "no_effect_sham")


def authored(seed: int) -> dict:
    if type(seed) is not int or seed not in SOURCE_SEEDS:
        raise ValueError('P51_UNREGISTERED_DEVELOPMENT_SOURCE')
    g = np.random.default_rng(seed)
    n = WORLDS
    kind = np.repeat(np.arange(4, dtype=np.int64), n // 4)
    policy = np.repeat(np.arange(4, dtype=np.int64), n // 4)
    g.shuffle(kind)
    g.shuffle(policy)
    k = g.choice(np.array([3,4,5]), n)
    u = lambda lo, hi: g.uniform(lo, hi, n)
    b0 = u(-.2, .2); b1=u(-.4,.4); b2=u(-.1,.1)
    amp=u(0,.08); phase=u(-math.pi,math.pi); freq=u(*SOURCE_SEEDS[seed])
    beta=g.choice(np.array([-1.,1.]), n)*u(.25,.45)
    beta[kind==3] = 0.0
    jump = g.choice(np.array([-1.,1.]),n)*u(.4,.7)
    x0 = u(-.3,.3)
    initial_action=g.choice(np.array([-1.,1.]),n)
    # RNG tape is born BEFORE the arms run: policy stochasticity is arm-independent.
    random_signs=g.choice(np.array([-1.,1.]),size=(n,N))
    stochastic=g.random((n,N))
    noise=g.normal(0,.02,size=(n,N))
    effect=beta[:,None]*np.where((kind[:,None]==1)&(np.arange(N)[None,:]>=k[:,None]),-1.,1.)
    step=(kind[:,None]==2)&(np.arange(N)[None,:]>=k[:,None])
    base=(b0[:,None]+b1[:,None]*T+b2[:,None]*T*T+
          amp[:,None]*np.sin(2*math.pi*freq[:,None]*T+phase[:,None])+
          jump[:,None]*step+noise)
    actual_a=np.zeros((n,6),dtype=np.float64)
    actual_y=np.zeros((n,6),dtype=np.float64)
    current=x0.copy()
    previous=initial_action.copy()
    for i in range(6):
        action=policy_action(policy,initial_action,previous,current,random_signs[:,i],stochastic[:,i])
        change=base[:,i]+effect[:,i]*action
        actual_a[:,i]=action; actual_y[:,i]=change
        current+=change; previous=action
    digest=hashlib.sha256(b''.join(z.tobytes() for z in (kind,policy,k,initial_action,
          base,effect,random_signs,stochastic,actual_a,actual_y))).hexdigest()
    return dict(seed=seed,kind=kind,policy=policy,k=k,initial=initial_action,
                random_signs=random_signs,stochastic=stochastic,base=base,
                effect=effect,current=current,initial_a=actual_a,initial_y=actual_y,
                source_sha256=digest)


def policy_action(policy,initial,previous,current,random_sign,stochastic):
    # The action chooser sees only previous real sensor, action and its random tape.
    feedback=np.where(current>0,-1.,1.)
    return np.where(policy==0,initial,
        np.where(policy==1,np.where(stochastic<.2,random_sign,feedback),
            np.where(policy==2,np.where(stochastic<.8,previous,-previous),random_sign)))


def score_actions(past_a:np.ndarray, step:int, method:str) -> np.ndarray:
    if step not in (6,7) or past_a.shape[1]!=step or method not in ('orthogonal','doptimal'):
        raise ValueError('P51_INVALID_PROSPECTIVE_ACTION_DESIGN')
    ts=T[:step+1]
    b=np.stack([np.ones_like(ts),ts,ts*ts],axis=-1)
    reg=0. if method=='orthogonal' else 1.e-4
    # Schur complement gives the *exact* D-optimal action-column criterion.
    # A full determinant need not be recomputed for every candidate/world.
    projection=np.eye(step+1)-b@np.linalg.solve(b.T@b+reg*np.eye(3),b.T)
    plus=np.concatenate((past_a,np.ones((len(past_a),1))),axis=1)
    minus=np.concatenate((past_a,-np.ones((len(past_a),1))),axis=1)
    scores_plus=np.einsum('ni,ij,nj->n',plus,projection,plus)
    scores_minus=np.einsum('ni,ij,nj->n',minus,projection,minus)
    return np.where(scores_plus>scores_minus+1.e-12,1.,
                    np.where(scores_minus>scores_plus+1.e-12,-1.,-past_a[:,-1]))


def estimate(a:np.ndarray,y:np.ndarray,start:int,method:str) -> tuple[np.ndarray,np.ndarray]:
    if method=='recent':
        assert start==5
        ts=T[5:];signs=a[:,5:];outcomes=y[:,5:]
        deg=1
    else:
        assert start==0 and method=='full'
        ts=T;signs=a;outcomes=y
        deg=2
    n=len(a)
    result=np.full(n,np.nan)
    # Only 2^8 possible sign histories: group and cache the exact conventional
    # factual OLS pseudoinverse. Never use latent source parameters here.
    keys=((signs==1).astype(np.uint16)*(2**np.arange(signs.shape[1],dtype=np.uint16))[None,:]).sum(1)
    for key in np.unique(keys):
        idx=np.flatnonzero(keys==key)
        vec=signs[idx[0]]
        x=np.stack([*(ts**j for j in range(deg+1)),vec],axis=1)
        singular=np.linalg.svd(x,compute_uv=False)
        if singular[-1]<=1e-10*singular[0] or singular[0]/singular[-1]>1e6:
            continue
        pin=np.linalg.pinv(x,rcond=1e-10)
        result[idx]=outcomes[idx]@pin[-1,:]
    return result,np.isfinite(result)


def executed_arm(world:dict,arm:str) -> dict:
    if arm not in ARMS:raise ValueError('P51_UNKNOWN_ACTUAL_PROBE_POLICY')
    t0=time.perf_counter();n=len(world['kind'])
    a=world['initial_a'].copy()
    y=world['initial_y'].copy()
    current=world['current'].copy()
    for step in (6,7):
        if arm in ('orthogonal','doptimal'):
            actual=score_actions(a,step,arm)
        elif arm=='flip':actual=-a[:,-1]
        elif arm=='random':actual=world['random_signs'][:,step]
        else:actual=policy_action(world['policy'],world['initial'],a[:,-1],current,
                         world['random_signs'][:,step],world['stochastic'][:,step])
        # This is the one executed physical action. Neither alternative is
        # queried; any unchosen-action consequence is never generated.
        observed_delta=world['base'][:,step]+world['effect'][:,step]*actual
        a=np.column_stack((a,actual));y=np.column_stack((y,observed_delta))
        current=current+observed_delta
    recent,recent_valid=estimate(a,y,5,'recent')
    full,full_valid=estimate(a,y,0,'full')
    elapsed=time.perf_counter()-t0
    sha=hashlib.sha256(b''.join(z.tobytes() for z in (a,y))).hexdigest()
    return dict(a=a,y=y,recent=recent,full=full,recent_valid=recent_valid,
                full_valid=full_valid,elapsed_seconds=elapsed,action_outcome_sha256=sha)


def metrics(world,arms):
    truth=world['effect'][:,-1]
    labels={name:world['kind']==i for i,name in enumerate(KINDS)}
    policies={name:world['policy']==i for i,name in enumerate(POLICIES)}
    def evaluate(arm):
        r=arms[arm]
        valid=r['recent_valid'];ef=np.abs(r['recent']-truth)
        p_err=np.where(valid,ef,1.0)
        full=r['full_valid']
        result=dict(
            valid_full_fraction=float(full.mean()),valid_recent_fraction=float(valid.mean()),
            recent_penalized_MAE=float(p_err.mean()),
            recent_conditional_MAE=float(ef[valid].mean()) if valid.any() else None,
            causal_switch_penalized_MAE=float(p_err[labels['causal_reverse']].mean()),
            no_effect_sham_conditional_MAE=float(ef[valid&labels['no_effect_sham']].mean()) if (valid&labels['no_effect_sham']).any() else None,
            action_outcome_sha256=r['action_outcome_sha256'],
            process_seconds=r['elapsed_seconds'],
            group={name:dict(n=int(mask.sum()),recent_valid=float(valid[mask].mean()),
                full_valid=float(full[mask].mean()),penalized_MAE=float(p_err[mask].mean()))
                for name,mask in labels.items()},
            policy={name:dict(n=int(mask.sum()),recent_valid=float(valid[mask].mean()),
                full_valid=float(full[mask].mean()),penalized_MAE=float(p_err[mask].mean()))
                for name,mask in policies.items()})
        return result
    return {arm:evaluate(arm) for arm in ARMS}


def trial(seed):
    world=authored(seed)
    results={arm:executed_arm(world,arm) for arm in ARMS}
    measurements=metrics(world,results)
    # Persist and cold-restore ONLY actual selected actions and observed deltas,
    # not latent effect, goal, counterfactuals or unknown regime.
    cold={}
    for arm,r in results.items():
        file=io.BytesIO();np.savez_compressed(file,a=r['a'],y=r['y']);file.seek(0)
        with np.load(file,allow_pickle=False) as restored:
            rec,rv=estimate(restored['a'],restored['y'],5,'recent')
            ful,fv=estimate(restored['a'],restored['y'],0,'full')
        cold[arm]=bool(np.array_equal(rec,r['recent'],equal_nan=True) and
                       np.array_equal(ful,r['full'],equal_nan=True) and
                       np.array_equal(rv,r['recent_valid']) and np.array_equal(fv,r['full_valid']))
    o=measurements['orthogonal'];d=measurements['doptimal'];r=measurements['random'];f=measurements['flip']
    checks={
       'causal_MAE_better_strong_10pct':o['causal_switch_penalized_MAE'] <= .9*min(
           d['causal_switch_penalized_MAE'],r['causal_switch_penalized_MAE'],f['causal_switch_penalized_MAE']),
       'constant_full_rank_95pct':o['policy']['constant']['full_valid'] >= .95,
       'constant_coverage_gain_vs_random_10pp':o['policy']['constant']['full_valid'] >= r['policy']['constant']['full_valid']+.10,
       'sham_effect_MAE_under_point06':o['no_effect_sham_conditional_MAE'] is not None and o['no_effect_sham_conditional_MAE']<=.06,
       'cold_exact_all':all(cold.values()),
       'cost_under_125pct_Dopt':o['process_seconds']<=1.25*d['process_seconds']
    }
    return dict(seed=seed,world_sha256=world['source_sha256'],arms=measurements,cold_exact=cold,
                checks=checks,all_frozen_checks=all(checks.values()),
                common_initial_source_sha256=hashlib.sha256(world['initial_a'].tobytes()+world['initial_y'].tobytes()).hexdigest())


def run():
    t0=time.perf_counter();rows=[trial(seed) for seed in SOURCE_SEEDS]
    return dict(schema='ttt-hybrid.p51.rank-repair-dev-receipt.v1',source_type='AUTHORED_DEV_NOT_EXTERNAL_OR_TRAINED_SCIENCE',
                seeds=list(SOURCE_SEEDS),worlds_per_cohort=WORLDS,arm_order=list(ARMS),
                arms_per_world_actions=8,actual_probe_slots=[6,7],
                rows=rows,all_three_positive=all(row['all_frozen_checks'] for row in rows),
                process_seconds=time.perf_counter()-t0,
                independent_scientific_first_use='NOT_RUN',fully_trained_GRU_MLP='NOT_RUN',
                final_dual_platform_regression='NOT_RUN',FORGE='UNTOUCHED')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--receipt',type=Path,required=True)
    args=ap.parse_args();z=run()
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(z,indent=2,sort_keys=True)+'\n',encoding='utf8')
    print(json.dumps({'rows':[
        {'seed':q['seed'],'sha256':q['world_sha256'],'arms':{name:{
        'full_coverage':a['valid_full_fraction'],
        'recent_coverage':a['valid_recent_fraction'],
        'switch_penalized_mae':a['causal_switch_penalized_MAE'],
        'constant_full_coverage':a['policy']['constant']['full_valid'],
        'sham_effect_mae':a['no_effect_sham_conditional_MAE'],
        'wall':a['process_seconds']} for name,a in q['arms'].items()},
        'checks':q['checks']} for q in z['rows']],
        'all_positive':z['all_three_positive'],'seconds':z['process_seconds']},indent=2))

if __name__=='__main__':main()
