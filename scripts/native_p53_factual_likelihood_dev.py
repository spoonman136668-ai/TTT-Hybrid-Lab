"""P53 DEV: learn the value of a real probe from only logged physical consequences.

The learned Gaussian likelihood uses signed physical rewards to reconstruct the
post-episode label. Training never reads latent theta or unexecuted diagnostic
measurements. The Gaussian-family inductive bias is deliberately conventional.
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
from scipy.special import ndtr, expit
import torch
from scripts import native_p52_probe_value_dev as p52

SOURCE = {830101: (89600, 'fit'), 830103: (1024, 'validation'),
          830107: (2048, 'ood1'), 830109: (2048, 'ood2')}
MODEL_SEEDS = (8311, 8323, 8331)
BOOTSTRAP_SEED = 830901


def source(seed: int) -> dict:
    if type(seed) is not int or seed not in SOURCE:
        raise ValueError('P53_UNREGISTERED_SOURCE')
    n, kind = SOURCE[seed]
    rng = np.random.default_rng(seed)
    if kind in ('fit', 'validation'):
        cg, pg, cs, ps, cost = ((.1,1.2),(.1,1.2),(.55,.95),(.55,.95),(.05,.4))
    elif kind == 'ood1':
        cg, pg, cs, ps, cost = ((0,1.5),(0,1.6),(.95,1.25),(.95,1.25),(.05,.65))
    else:
        cg, pg, cs, ps, cost = ((0,2.0),(0,2.0),(1.1,1.4),(1.1,1.4),(.01,.75))
    sample = lambda interval: rng.uniform(*interval, n).astype(np.float32)
    cue_gain, probe_gain = sample(cg), sample(pg)
    cue_sigma, probe_sigma, price = sample(cs), sample(ps), sample(cost)
    theta = rng.choice(np.array([-1.,1.],dtype=np.float32),size=n)
    noise_c = rng.normal(size=n).astype(np.float32)
    noise_p = rng.normal(size=n).astype(np.float32)
    cue = theta*cue_gain + cue_sigma*noise_c
    prior = np.stack((cue,cue_gain,cue_sigma,probe_gain,probe_sigma,price),axis=-1)
    log_probe = (rng.random(n)<.5).astype(np.float32)
    log_action = rng.choice(np.array([-1.,1.],dtype=np.float32),size=n)
    # The logged dataset contains ONLY one chosen diagnostic action, its actual
    # measurement if paid for, and one chosen final action plus observed reward.
    log_measure = np.where(log_probe>0,theta*probe_gain+probe_sigma*noise_p,0.).astype(np.float32)
    log_reward = (theta*log_action - price*log_probe).astype(np.float32)
    log = dict(prior=prior,actual_probe=log_probe,actual_measurement=log_measure,
               actual_final_action=log_action,observed_net_reward=log_reward)
    # Hidden values are retained ONLY in evaluator input, never in fitting API.
    data_np=dict(prior=prior,theta=theta,eps_probe=noise_p,
                 random_probe=log_probe,random_action=log_action)
    digest=hashlib.sha256(b''.join(x.tobytes() for x in data_np.values())).hexdigest()
    logged_hash=hashlib.sha256(b''.join(x.tobytes() for x in log.values())).hexdigest()
    return dict(seed=seed,kind=kind,n=n,log=log,log_sha256=logged_hash,
                data={k:torch.from_numpy(v.copy()) for k,v in data_np.items()},
                source_sha256=digest)


def fitted_likelihood(log: dict) -> dict:
    """Conventional Gaussian fit using actual observations and reward ONLY."""
    expected={'prior','actual_probe','actual_measurement','actual_final_action','observed_net_reward'}
    if set(log)!=expected:raise ValueError('P53_LOG_SCHEMA_REFUSED')
    c,probe,m,a,r=(np.asarray(log[k],dtype=np.float64) for k in
                   ('prior','actual_probe','actual_measurement','actual_final_action','observed_net_reward'))
    n=len(c)
    if c.shape!=(n,6) or any(z.shape!=(n,) for z in (probe,m,a,r)):
        raise ValueError('P53_BAD_ACTUAL_LOG_SHAPE')
    if not np.all(np.isin(probe,(0,1))) or not np.all(np.isin(a,(-1,1))):
        raise ValueError('P53_ILLEGAL_ACTUAL_ACTION')
    if not np.isfinite(c).all() or not all(np.isfinite(x).all() for x in (probe,m,a,r)):
        raise ValueError('P53_NONFINITE_LOG')
    if not np.all(m[probe==0]==0):raise ValueError('P53_UNEXECUTED_MEASUREMENT_LEAK')
    if (c[:,2]<=0).any() or (c[:,4]<=0).any():raise ValueError('P53_BAD_SIGMA')
    # The *observed* final physical reward in this particular toy exactly
    # identifies the latent sign after the episode. Never inspect source theta.
    sign=(r + c[:,5]*probe)/a
    if not np.all(np.isclose(np.abs(sign),1.,atol=5e-6,rtol=0)):
        raise ValueError('P53_REWARD_NOT_BINARY_EFFECT_IDENTIFIABLE')
    sign=np.where(sign>=0,1.,-1.)
    p=(sign>0).mean()
    observed=probe>0
    if not observed.any() or (~observed).sum()==0:raise ValueError('P53_NO_PROBE_COVERAGE')
    def slope_var(x,y):
        coef=float(np.dot(x,y)/np.dot(x,x))
        resid=y-coef*x
        var=float(np.dot(resid,resid)/len(resid))
        if coef<=0 or var<=0:raise ValueError('P53_UNIDENTIFIABLE_GAUSSIAN')
        return coef,var
    xcue=sign*c[:,1]/c[:,2]
    ycue=c[:,0]/c[:,2]
    cue_scale,cue_var=slope_var(xcue,ycue)
    xprobe=sign[observed]*c[observed,3]/c[observed,4]
    yprobe=m[observed]/c[observed,4]
    probe_scale,probe_var=slope_var(xprobe,yprobe)
    return dict(prior_positive=float(p),cue_scale=cue_scale,cue_variance=cue_var,
                probe_scale=probe_scale,probe_variance=probe_var,
                physical_training_facts=n,executed_training_probes=int(observed.sum()),
                physical_probe_cost_total=float(np.dot(probe,c[:,5])),
                reward_sign_max_residual=float(np.max(np.abs(np.abs((r+c[:,5]*probe)/a)-1.))))


def decision(data: dict, model: dict, policy='learned') -> dict:
    if policy not in ('learned','never','always'):raise ValueError('P53_UNKNOWN_POLICY')
    c=data['prior'].numpy().astype(np.float64)
    theta=data['theta'].numpy().astype(np.float64)
    ep=data['eps_probe'].numpy().astype(np.float64)
    cue,cg,cs,pg,ps,cost=c.T
    # All coefficients below were fitted from real training observations.
    logodds=math.log(model['prior_positive']/(1-model['prior_positive'])) + (
        2*model['cue_scale']*cg*cue/(model['cue_variance']*cs*cs))
    p=expit(logodds)
    no_value=np.abs(2*p-1)
    mu=model['probe_scale']*pg
    stdev=ps*math.sqrt(model['probe_variance'])
    boundary=-logodds*stdev*stdev/(2*np.maximum(mu,1e-10))
    expected_accuracy=p*ndtr((mu-boundary)/stdev)+(1-p)*ndtr((boundary+mu)/stdev)
    advantage=2*expected_accuracy-1-cost-no_value
    if policy=='never':actual_probe=np.zeros(len(c),dtype=np.float64)
    elif policy=='always':actual_probe=np.ones(len(c),dtype=np.float64)
    else:actual_probe=(advantage>0).astype(np.float64)
    # Model sees only the selected physical diagnostic measurement; the
    # simulator's counterfactual tape is *not* passed to this decision logic.
    obs=np.zeros(len(c),dtype=np.float64)
    mask=actual_probe>0
    obs[mask]=theta[mask]*pg[mask]+ps[mask]*ep[mask]
    updated=logodds + 2*model['probe_scale']*pg*obs/(model['probe_variance']*ps*ps)
    actual_action=np.where(updated>=0,1.,-1.)
    actual_reward=theta*actual_action-cost*actual_probe
    weak=(np.abs(cue)/cs<.5)&(pg>.8)
    low=weak&(cost<.2);high=weak&(cost>=.4)
    return dict(net_reward=float(actual_reward.mean()),probe_rate=float(actual_probe.mean()),
                reward=actual_reward,probe=actual_probe,action=actual_action,
                low_probe_fraction=float(actual_probe[low].mean()) if low.any() else None,
                high_probe_fraction=float(actual_probe[high].mean()) if high.any() else None,
                low_group_n=int(low.sum()),high_group_n=int(high.sum()),
                actual_probe_cost=float(np.dot(actual_probe,cost)),
                actual_executed_probes=int(actual_probe.sum()),
                decisions_sha256=hashlib.sha256(np.column_stack((actual_probe,actual_action,actual_reward)).tobytes()).hexdigest())


def paired_ci(x,y,seed=BOOTSTRAP_SEED):
    diff=np.asarray(x)-np.asarray(y)
    rng=np.random.default_rng(seed)
    ix=rng.integers(0,len(diff),size=(2000,len(diff)))
    boot=diff[ix].mean(axis=1)
    return {'difference':float(diff.mean()),'lower_95':float(np.percentile(boot,2.5)),
            'upper_95':float(np.percentile(boot,97.5)),
            'bootstrap_seed':seed,'bootstrap_resamples':2000}


def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    ds={seed:source(seed) for seed in SOURCE}
    fitted_start=time.perf_counter()
    params=fitted_likelihood(ds[830101]['log'])
    fit_time=time.perf_counter()-fitted_start
    three=[]
    idx=np.array_split(np.arange(ds[830101]['n']),3)
    for group in idx:
        partition={k:v[group] for k,v in ds[830101]['log'].items()}
        three.append(fitted_likelihood(partition))
    tests={}
    for seed in (830107,830109):
        data=ds[seed]['data']
        new=decision(data,params,'learned')
        never=decision(data,params,'never')
        always=decision(data,params,'always')
        bayes=p52.bayes_reference(data,'bayes')
        simple=p52.bayes_reference(data,'simple')
        ci=paired_ci(new['reward'],never['reward'],BOOTSTRAP_SEED+seed)
        tests[str(seed)]={
           'learned':{k:v for k,v in new.items() if k not in ('reward','probe','action')},
           'learned_never':{k:v for k,v in never.items() if k not in ('reward','probe','action')},
           'learned_always':{k:v for k,v in always.items() if k not in ('reward','probe','action')},
           'model_aware_bayes':{k:v for k,v in bayes.items() if k not in ('decision','probed')},
           'simple_model_aware':{k:v for k,v in simple.items() if k not in ('decision','probed')},
           'paired_bootstrap':ci,
           'gate':{
             'learned_beat_fitted_never_point02':new['net_reward']>=never['net_reward']+.02,
             'learned_within_point02_of_bayes':new['net_reward']>=bayes['net_reward']-.02,
             'nontrivial_probes':.05<=new['probe_rate']<=.95,
             'cost_sensitive':new['low_group_n']>=25 and new['high_group_n']>=25 and
                               new['low_probe_fraction']>=new['high_probe_fraction']+.15,
             'paired_CI_lower_positive':ci['lower_95']>0,
           }}
    # Strong directly reused P52 on-policy REINFORCE control on the fresh same
    # world contexts. Counts of *executed training probes* may differ from logs.
    rl=[]
    for seed in MODEL_SEEDS:
        arms={}
        for name,factory in (('factorized',p52.FactorizedPolicy),('gru',p52.ConventionalGRUPolicy)):
            torch.manual_seed(seed);m=factory()
            torch.manual_seed(seed+100000)
            tr=p52.train(m,ds[830101]['data'],ds[830103]['data'])
            ev={str(s):p52.summary(m,ds[s]['data']) for s in (830107,830109)}
            cold,sha=p52.cold(m,ds[830107]['data'],factory)
            arms[name]=dict(train=tr,eval=ev,params=p52.count_params(m),cold_exact=cold,cold_sha256=sha)
        rl.append(dict(seed=seed,arms=arms))
    return dict(schema='ttt-hybrid.p53.factual-reward-learned-VOI-result.v1',
      classification='AUTHOR_GENERATED_DEV_NOT_INDEPENDENT_SCIENCE',
      source_hashes={str(s):ds[s]['source_sha256'] for s in SOURCE},
      logged_fit_hash=ds[830101]['log_sha256'],
      data_sizes={str(s):ds[s]['n'] for s in SOURCE},
      learned_parameters=params,three_disjoint_fitted_partitions=three,
      learned_fitting_seconds=fit_time,ood=tests,original_P52_RL_on_new_source=rl,
      frozen_positive=all(all(v for v in r['gate'].values()) for r in tests.values()),
      full_wall_seconds=time.perf_counter()-start,
      original_full_regression='NOT_RUN',independent_scientific_first_use='NOT_RUN',FORGE='UNTOUCHED')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args();result=run();args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf8')
    print(json.dumps({
     'source_hashes':result['source_hashes'],
     'fitted':result['learned_parameters'],
     'ood':{s:{'learned':z['learned']['net_reward'],'never':z['learned_never']['net_reward'],
          'bayes':z['model_aware_bayes']['net_reward'],'probe_fraction':z['learned']['probe_rate'],
          'gain_95ci':z['paired_bootstrap'],'gates':z['gate']} for s,z in result['ood'].items()},
     'RL':{x['seed']:{k:{s:v['net_reward'] for s,v in arm['eval'].items()} for k,arm in x['arms'].items()} for x in result['original_P52_RL_on_new_source']},
     'all_frozen_gates':result['frozen_positive'],'seconds':result['full_wall_seconds']},indent=2))

if __name__=='__main__':main()
