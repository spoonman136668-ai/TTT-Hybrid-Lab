"""P56: fit Gaussian/Laplace observation families from physically logged rewards.

Pre-registered author-generated DEV. No latent label or unexecuted physical
measurement enters the model fitting API. This is a classical two-family
hypothesis selection comparison, not evidence of native Hybrid intelligence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
from scipy.special import expit, logsumexp, ndtr
from scripts import native_p54_noisy_factual_reward_dev as p54
from scripts import native_p55_laplace_transfer_dev as p55

FAMILY = {
  860001: ('fit', ('G','G','G')),
  860003: ('ood', ('G','G','G')),
  860101: ('fit', ('L','L','L')),
  860103: ('ood', ('L','L','L')),
  860201: ('fit', ('G','L','G')),
  860203: ('ood', ('G','L','G')),
  860301: ('sham', ('G','L','G')),
}
FIT_N=20480
SELECT_N=4096
OOD_N=8192
SHAM_N=4096
FIT_SEEDS=(860001,860101,860201)
OOD_SEEDS=(860003,860103,860203)
SHAM_SEED=860301
FROZEN_BOOTSTRAP_BASE=860901


def source(seed:int):
    if type(seed) is not int or seed not in FAMILY:
        raise ValueError('P56_UNREGISTERED_SOURCE')
    role, laws=FAMILY[seed]
    n={'fit':FIT_N,'ood':OOD_N,'sham':SHAM_N}[role]
    rng=np.random.default_rng(seed)
    ranges=((.1,1.2),(.1,1.2),(.55,.95),(.55,.95),(.05,.4),(.5,1.1)) if role=='fit' else (
        (0,1.6),(0,1.8),(.95,1.35),(.95,1.35),(.01,.7),(1.1,1.8))
    cg,pg,cs,ps,cost,rs=[rng.uniform(lo,hi,n).astype('float32') for lo,hi in ranges]
    if role=='sham':pg.fill(0)
    theta=rng.choice(np.array([-1.,1.],dtype='float32'),size=n)
    def noise(family):
        return (rng.standard_normal(size=n) if family=='G' else rng.laplace(scale=1/math.sqrt(2),size=n)).astype('float32')
    cue_noise=noise(laws[0]);probe_noise=noise(laws[1]);reward_noise=noise(laws[2])
    prior=np.stack((theta*cg+cs*cue_noise,cg,cs,pg,ps,cost),axis=1)
    logged_probe=(rng.random(n)<.5).astype('float32')
    logged_action=rng.choice(np.array([-1.,1.],dtype='float32'),size=n)
    measurement=np.where(logged_probe>0,theta*pg+ps*probe_noise,0).astype('float32')
    realized_net_reward=(theta*logged_action+rs*reward_noise-cost*logged_probe).astype('float32')
    log=dict(prior=prior,actual_probe=logged_probe,actual_measurement=measurement,
             actual_final_action=logged_action,observed_net_reward=realized_net_reward)
    # Separate evaluator-owned tape: never pass this to fit_model().
    evaluator=dict(prior=prior,theta=theta,probe_noise=probe_noise,
                   reward_noise=rs*reward_noise)
    signature=hashlib.sha256(b''.join(x.tobytes() for x in (prior,theta,probe_noise,rs*reward_noise,logged_probe,logged_action,measurement,realized_net_reward))).hexdigest()
    loghash=hashlib.sha256(b''.join(x.tobytes() for x in log.values())).hexdigest()
    return dict(seed=seed,role=role,laws=laws,log=log,eval=evaluator,sha256=signature,log_sha256=loghash)


def logpdf(x,mu,var,family):
    if not np.isfinite(var) or var<=0:
        raise ValueError('P56_INVALID_VARIANCE')
    if family=='G':
        return -.5*(math.log(2*math.pi*var)+(x-mu)**2/var)
    if family=='L':
        b=math.sqrt(var/2)
        return -math.log(2*b)-np.abs(x-mu)/b
    raise ValueError('P56_UNSUPPORTED_FAMILY')


def log_mixture(prior,z,reward_var,reward_family,signal=None,mu=None,var=None,signal_family=None):
    pi=np.array([1-prior,prior])
    likelihood=np.stack((np.log(pi[0])+logpdf(z,-1,reward_var,reward_family),
                         np.log(pi[1])+logpdf(z,1,reward_var,reward_family)),axis=1)
    if signal is not None:
        likelihood[:,0]+=logpdf(signal,-mu,var,signal_family)
        likelihood[:,1]+=logpdf(signal,mu,var,signal_family)
    return logsumexp(likelihood,axis=1)


def fit_model(train_log,selection_log):
    expected={'prior','actual_probe','actual_measurement','actual_final_action','observed_net_reward'}
    if set(train_log)!=expected or set(selection_log)!=expected:
        raise ValueError('P56_NONFACTUAL_LOG_SCHEMA')
    fitted=p54.fit_moments(train_log)
    cp=np.asarray(selection_log['prior'],dtype=np.float64)
    pr=np.asarray(selection_log['actual_probe'],dtype=np.float64)
    mm=np.asarray(selection_log['actual_measurement'],dtype=np.float64)
    aa=np.asarray(selection_log['actual_final_action'],dtype=np.float64)
    rr=np.asarray(selection_log['observed_net_reward'],dtype=np.float64)
    if len(cp)!=SELECT_N:raise ValueError('P56_SELECTION_SLICE_NOT_FROZEN')
    if np.any(mm[pr==0]!=0):raise ValueError('P56_UNEXECUTED_MEASUREMENT_LEAK')
    z=(rr+cp[:,5]*pr)/aa
    trainp=np.asarray(train_log['prior'],dtype=np.float64)
    trainz=(np.asarray(train_log['observed_net_reward'],dtype=np.float64)+trainp[:,5]*np.asarray(train_log['actual_probe'],dtype=np.float64))/np.asarray(train_log['actual_final_action'],dtype=np.float64)
    reward_var=max(.05,float(np.var(trainz)-1))
    prior=fitted['prior_positive']
    reward_scores={f:float(log_mixture(prior,z,reward_var,f).mean()) for f in ('G','L')}
    reward_family=max(reward_scores,key=reward_scores.get)
    cue=cp[:,0]/cp[:,2]
    mu_c=fitted['cue_scale']*cp[:,1]/cp[:,2]
    scores_cue={f:float(log_mixture(prior,z,reward_var,reward_family,cue,mu_c,fitted['cue_variance'],f).mean()) for f in ('G','L')}
    cue_family=max(scores_cue,key=scores_cue.get)
    ix=pr>0
    measured=mm[ix]/cp[ix,4]
    mu_p=fitted['probe_scale']*cp[ix,3]/cp[ix,4]
    scores_probe={f:float(log_mixture(prior,z[ix],reward_var,reward_family,measured,mu_p,fitted['probe_variance'],f).mean()) for f in ('G','L')}
    probe_family=max(scores_probe,key=scores_probe.get)
    return dict(fitted=fitted,reward_variance=reward_var,
                family={'reward':reward_family,'cue':cue_family,'probe':probe_family},
                selection_scores={'reward':reward_scores,'cue':scores_cue,'probe':scores_probe},
                fit_records=len(trainp),selection_records=len(cp),
                fit_executed_probes=int(np.sum(train_log['actual_probe'])),
                selection_executed_probes=int(ix.sum()))


def posterior_logratio(observed,mu,var,family):
    return logpdf(observed,mu,var,family)-logpdf(observed,-mu,var,family)


def diagnostic_gain(logodds,mu,var,family):
    p=expit(logodds)
    if family=='L':
        return p55.expected_laplace_value(logodds,mu,math.sqrt(var/2))
    if family=='G':
        std=math.sqrt(var)
        border=-logodds*var/(2*np.maximum(mu,1.e-10))
        correct=p*ndtr((mu-border)/std)+(1-p)*ndtr((mu+border)/std)
        return np.maximum(0,2*correct-1-np.abs(2*p-1))
    raise ValueError('P56_UNSUPPORTED_PROBE_FAMILY')


def decide(tape,model,policy='selected',override=None):
    if policy not in ('selected','gaussian','laplace','never','oracle','wrong'):
        raise ValueError('P56_BAD_POLICY')
    c=np.asarray(tape['prior'],dtype=np.float64)
    theta=np.asarray(tape['theta'],dtype=np.float64)
    npnoise=np.asarray(tape['probe_noise'],dtype=np.float64)
    reward_noise=np.asarray(tape['reward_noise'],dtype=np.float64)
    cue,cg,cs,pg,ps,cost=c.T
    if policy=='oracle':
        if override is None:raise ValueError('P56_ORACLE_FAMILY_NOT_SPECIFIED')
        prior=.5;sc=sp=vc=vp=1.
        cf,pf=override[0],override[1]
    else:
        params=model['fitted']
        prior=params['prior_positive']
        sc,sp,vc,vp=[params[k] for k in ('cue_scale','probe_scale','cue_variance','probe_variance')]
        if policy=='gaussian':cf=pf='G'
        elif policy=='laplace':cf=pf='L'
        else:cf,pf=model['family']['cue'],model['family']['probe']
    if not 0<prior<1 or min(vc,vp)<=0:raise ValueError('P56_BAD_LEARNED_MODEL')
    logodds=math.log(prior/(1-prior))+posterior_logratio(cue/cs,sc*cg/cs,vc,cf)
    value=diagnostic_gain(logodds,sp*pg/ps,vp,pf)
    if policy=='never':probe=np.zeros(len(c),dtype=np.float64)
    else:probe=(value>cost+1.e-12).astype(np.float64)
    measured=np.zeros(len(c))
    ix=probe>0
    measured[ix]=theta[ix]*pg[ix]+ps[ix]*npnoise[ix]
    updated=logodds+probe*posterior_logratio(measured/ps,sp*pg/ps,vp,pf)
    action=np.where(updated>=0,1.,-1.)
    reward=theta*action+reward_noise-cost*probe
    weak=(np.abs(cue)/cs<.5)&(pg>.8)
    low=weak&(cost<.2);high=weak&(cost>=.4)
    return dict(mean_reward=float(reward.mean()),probe_fraction=float(probe.mean()),
        actual_probe_count=int(probe.sum()),actual_probe=probe,actual_action=action,reward=reward,
        cost_probe_fraction_low=float(probe[low].mean()) if low.any() else None,
        cost_probe_fraction_high=float(probe[high].mean()) if high.any() else None,
        cost_low_n=int(low.sum()),cost_high_n=int(high.sum()),
        decisions_sha256=hashlib.sha256(np.column_stack((probe,action,reward)).tobytes()).hexdigest())


def paired_bootstrap(a,b,seed):
    d=a-b;g=np.random.default_rng(seed);means=np.empty(1000)
    for i in range(1000):means[i]=d[g.integers(0,len(d),len(d))].mean()
    return dict(mean=float(d.mean()),lo95=float(np.percentile(means,2.5)),hi95=float(np.percentile(means,97.5)),seed=seed,resamples=1000)


def experiment():
    started=time.perf_counter();records={}
    for fit_seed,ood_seed in zip(FIT_SEEDS,OOD_SEEDS):
        fit=source(fit_seed)
        train={key:val[:-SELECT_N] for key,val in fit['log'].items()}
        selection={key:val[-SELECT_N:] for key,val in fit['log'].items()}
        model=fit_model(train,selection)
        world=source(ood_seed)
        arms={k:decide(world['eval'],model,k,world['laws']) for k in ('selected','gaussian','laplace','never','oracle')}
        wrong_log=p54.shuffled_credit_log(train)
        wrong=p54.fit_moments(wrong_log)
        wrongmodel=dict(model,fitted=wrong)
        arms['wrong_credit']=decide(world['eval'],wrongmodel,'wrong')
        res={name:{key:val for key,val in value.items() if key not in ('actual_probe','actual_action','reward')} for name,value in arms.items()}
        chosen=arms['selected'];gauss=arms['gaussian'];lap=arms['laplace'];no=arms['never'];oracle=arms['oracle'];wrong=arms['wrong_credit']
        ci=paired_bootstrap(chosen['reward'],no['reward'],FROZEN_BOOTSTRAP_BASE+ood_seed)
        gains=dict(vs_gaussian=chosen['mean_reward']-gauss['mean_reward'],vs_laplace=chosen['mean_reward']-lap['mean_reward'],vs_stronger_fixed=chosen['mean_reward']-max(gauss['mean_reward'],lap['mean_reward']),vs_never=chosen['mean_reward']-no['mean_reward'])
        gates=dict(correct_family=model['family']==dict(zip(('cue','probe','reward'),world['laws'])),
            near_stronger_fixed=gains['vs_stronger_fixed']>=-.001,
            actually_probes=.05<=chosen['probe_fraction']<=.95,
            learns_positive_value=gains['vs_never']>=.02,
            within_model_aware=.03>=oracle['mean_reward']-chosen['mean_reward'],
            positive_paired_CI=ci['lo95']>0,
            wrong_credit_margin=chosen['mean_reward']>=wrong['mean_reward']+.025,
            mixed_strong_gain=(gains['vs_stronger_fixed']>=.003 if fit_seed==860201 else True))
        coldmodel=json.loads(json.dumps(model,sort_keys=True))
        cold=decide(world['eval'],coldmodel,'selected',world['laws'])
        gates['cold_exact']=cold['decisions_sha256']==chosen['decisions_sha256']
        records[str(ood_seed)]=dict(fit_seed=fit_seed,fit_sha256=fit['sha256'],fit_log_sha256=fit['log_sha256'],ood_sha256=world['sha256'],
            laws=world['laws'],fitted_model=model,arms=res,paired_ci=ci,gains=gains,gates=gates)
    sham=source(SHAM_SEED)
    mix=records['860203']['fitted_model']
    shamdecision=decide(sham['eval'],mix,'selected',sham['laws'])
    return dict(schema='ttt-hybrid.p56.factual-family-selection-dev-result.v1',
        classification='FROZEN_AUTHOR_GENERATED_DEV_NOT_INDEPENDENT_SCIENTIFIC_FIRST_USE',
        rows=records,sham=dict(source_sha256=sham['sha256'],actual_probe_count=shamdecision['actual_probe_count']),
        frozen_all_gates=all(all(row['gates'].values()) for row in records.values()) and shamdecision['actual_probe_count']==0,
        elapsed_seconds=time.perf_counter()-started,original_repo_full_Windows_Ubuntu_regression='NOT_RUN',
        scientifically_independent_first_use='NOT_RUN',FORGE='UNTOUCHED')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--receipt',type=Path,required=True)
    a=ap.parse_args();result=experiment();a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'family_decisions':{s:r['fitted_model']['family'] for s,r in result['rows'].items()},
        'by_cohort':{s:{'gains':r['gains'],'gates':r['gates'],'mean_rewards':{x:y['mean_reward'] for x,y in r['arms'].items()}} for s,r in result['rows'].items()},
        'all':result['frozen_all_gates'],'sham':result['sham'],'wall':result['elapsed_seconds']},indent=2))
if __name__=='__main__':main()
