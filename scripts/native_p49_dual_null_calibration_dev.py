"""P49 prospective authored-DEV two-stage null calibration.

The original P47/P48 implementations, data and their frozen results are left
untouched. This is a conventional statistical selection control, not neural
causal learning or independent science. Evaluate only new preregistered seeds.
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
from scripts import native_p47_change_vs_nuisance_dev as p47
from scripts import native_p48_calibrated_causal_switch_dev as p48

COHORTS = {780001:(4096,(.35,2.0),'calibration'),
           780101:(1536,(.35,.90),'evaluation'),
           780103:(1536,(.95,1.35),'evaluation'),
           780107:(1536,(1.40,2.0),'evaluation')}
Q=.99


def source(seed):
    if type(seed) is not int or seed not in COHORTS:
        raise ValueError('P49_UNREGISTERED_SOURCE')
    n,band,role=COHORTS[seed]
    g=np.random.default_rng(seed)
    if role=='calibration':
        kind=np.full(n,2,dtype=np.int64)
    else:
        kind=np.repeat(np.array([0,1,2],dtype=np.int64),[768,384,384]);g.shuffle(kind)
    k=g.choice(np.array(p47.SPLITS,dtype=np.int64),n)
    initial=g.choice(np.array([-1.,1.]),n)
    a=initial[:,None]*np.power(-1.,np.arange(p47.N))[None,:]
    u=lambda lo,hi:g.uniform(lo,hi,(n,1))
    b0,b1,b2=u(-.20,.20),u(-.40,.40),u(-.10,.10)
    amp=u(0,.08);phase=u(-math.pi,math.pi);freq=u(*band)
    beta=g.choice(np.array([-1.,1.]),(n,1))*u(.25,.45)
    flip=(kind[:,None]==0)&(np.arange(p47.N)[None,:]>=k[:,None])
    beta_hist=beta*np.where(flip,-1.,1.)
    jump=g.choice(np.array([-1.,1.]),(n,1))*u(.40,.70)
    shock=(kind[:,None]==1)&(np.arange(p47.N)[None,:]>=k[:,None])
    nuisance=(b0+b1*p47.T+b2*p47.T**2
              +amp*np.sin(2*math.pi*freq*p47.T+phase)+jump*shock)
    y=nuisance+beta_hist*a+g.normal(0,.02,(n,p47.N))
    true_beta=beta_hist[:,-1].copy()
    sha=hashlib.sha256(b''.join(x.tobytes() for x in
         (a,y,true_beta,kind,k))).hexdigest()
    return dict(a=a,y=y,kind=kind,k=k,beta=true_beta,seed=seed,
                role=role,sha=sha)


def null_evidence(a,y):
    """Only original P47 unchanged/nuisance hypotheses and BIC scores.

    Margins are improvement in BIC from unchanged to the best nuisance jump.
    """
    p47.estimate(a,y)  # original fixed action and finite-source validation
    n=len(a)
    best_null_bic=np.full(n,np.inf)
    best_nuisance_bic=np.full(n,np.inf)
    full_effect=np.zeros(n)
    nuisance_effect=np.zeros(n)
    nuisance_k=np.full(n,-1,dtype=np.int64)
    for sign in (-1.,1.):
        ids=np.flatnonzero(a[:,0]==sign)
        if not len(ids):continue
        full,fullbic=p47.fit(y[ids],sign)
        full_effect[ids]=full;best_null_bic[ids]=fullbic
        for k in p47.SPLITS:
            effect,bic=p47.fit(y[ids],sign,'nuisance',k)
            sub=(bic<best_nuisance_bic[ids]);selected=ids[sub]
            best_nuisance_bic[selected]=bic[sub]
            nuisance_effect[selected]=effect[sub]
            nuisance_k[selected]=k
    return dict(margin=best_null_bic-best_nuisance_bic,
                full_effect=full_effect,nuisance_effect=nuisance_effect,
                nuisance_k=nuisance_k)


def calibrate(cal):
    if cal['seed']!=780001 or cal['role']!='calibration':
        raise ValueError('P49_CALIBRATION_CUSTODY')
    if not np.all(cal['kind']==2):
        raise ValueError('P49_NONSTABLE_CALIBRATION')
    causal_margin=p48.fitted_hypotheses(cal['a'],cal['y'])['margin']
    nuisance_margin=null_evidence(cal['a'],cal['y'])['margin']
    c=float(np.quantile(causal_margin,Q,method='higher'))
    n=float(np.quantile(nuisance_margin,Q,method='higher'))
    if not np.isfinite(c) or not np.isfinite(n):
        raise ValueError('P49_NONFINITE_CALIBRATION')
    digests={key:hashlib.sha256(x.tobytes()).hexdigest() for key,x in
              [('causal_margin',causal_margin),('nuisance_margin',nuisance_margin)]}
    return (c,n),digests


def predict(a,y,thresholds):
    if (not isinstance(thresholds,(tuple,list)) or len(thresholds)!=2
        or not all(np.isscalar(x) and np.isfinite(x) for x in thresholds)):
        raise ValueError('P49_INVALID_FROZEN_THRESHOLDS')
    c_threshold,n_threshold=thresholds
    p48_result=p48.predict(a,y,float(c_threshold))
    null=null_evidence(a,y)
    # Never change calibrated P48's causal branch or its original controls.
    causal=p48_result['model_kind']==0
    nuisance=(null['margin']>max(0.,float(n_threshold))) & ~causal
    ret={key:val.copy() for key,val in p48_result.items()}
    ret['candidate']=np.where(causal,p48_result['candidate'],
                       np.where(nuisance,null['nuisance_effect'],null['full_effect']))
    ret['model_kind']=np.where(causal,0,np.where(nuisance,1,-1))
    ret['chosen_k']=np.where(causal,p48_result['chosen_k'],
                     np.where(nuisance,null['nuisance_k'],-1))
    return ret


def run():
    t0=time.perf_counter();cal=source(780001)
    thresholds,digests=calibrate(cal)
    cohorts=[]
    for seed in (780101,780103,780107):
        d=source(seed)
        original=p47.metrics(d,p47.estimate(d['a'],d['y']))
        p48_ref=p47.metrics(d,p48.predict(d['a'],d['y'],thresholds[0]))
        new_pred=predict(d['a'],d['y'],thresholds)
        candidate=p47.metrics(d,new_pred)
        b=io.BytesIO();np.savez_compressed(b,a=d['a'],y=d['y']);b.seek(0)
        with np.load(b,allow_pickle=False) as z:
            cold=predict(z['a'],z['y'],thresholds)
        exact=all(np.array_equal(new_pred[k],cold[k]) for k in new_pred)
        nuisance_mae_new=candidate['results']['candidate']['exogenous']['mae']
        nuisance_mae_p48=p48_ref['results']['candidate']['exogenous']['mae']
        extra_gate=(nuisance_mae_new<=1.10*nuisance_mae_p48)
        gates=dict(candidate['gates'],
                   nuisance_effect_MAE_110pct_P48=bool(extra_gate),cold_exact=bool(exact))
        cohorts.append(dict(seed=seed,trace_sha256=d['sha'],original_p47=original,
                            original_p48=p48_ref,p49=candidate,gates=gates,
                            all_gates=all(gates.values())))
    return dict(schema='ttt-hybrid.p49.dual-null-calibration-dev-result.v1',
                scope='PROSPECTIVE_AUTHORED_DEV_ONLY_NOT_INDEPENDENT_SCIENCE',
                calibration_seed=780001,
                calibration_source_sha256=cal['sha'],calibration_margins_sha256=digests,
                threshold_quantile=Q,thresholds={'causal':thresholds[0],
                                                  'nuisance':thresholds[1]},
                cohorts=cohorts,all_frozen_gates=all(x['all_gates'] for x in cohorts),
                elapsed_seconds=time.perf_counter()-t0,
                original_p48_source_sha='3227cdd4fcbd4f6821d3bc369c18d326df70105d',
                full_windows_linux_regression='NOT_RUN',
                independent_science='NOT_RUN',
                FORGE='UNCHANGED')


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args();result=run()
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'thresholds':result['thresholds'],
        'cohorts':[{'seed':row['seed'],'all_gates':row['all_gates'],
          'p49_stable_false_alarm':row['p49']['detection']['stable_false_alarm'],
          'p49_causal_recall':row['p49']['detection']['causal_recall'],
          'p49_stable_MAE':row['p49']['results']['candidate']['stable']['mae'],
          'p49_nuisance_MAE':row['p49']['results']['candidate']['exogenous']['mae'],
          'gates':row['gates']} for row in result['cohorts']],
          'all_frozen_gates':result['all_frozen_gates'],
          'wall':result['elapsed_seconds']},indent=2))

if __name__=='__main__':main()
