"""P57 authored DEV: two physical diagnostics; buy second after first consequence.

The learner receives only same-probe logged factual experiences and never
receives hidden state, an unexecuted diagnostic outcome, or a counterfactual
reward. Gaussian/Laplace hypothesis family is supplied by the researcher.
"""
from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scripts import native_p56_family_selection_dev as p56

FAMILIES = {
    870101: ('fit', ('G','G','L','G')), 870103: ('ood', ('G','G','L','G')),
    870201: ('fit', ('L','L','G','L')), 870203: ('ood', ('L','L','G','L')),
    870301: ('fit', ('G','L','L','G')), 870303: ('ood', ('G','L','L','G')),
    870401: ('sham',('G','L','L','G')),
}
FIT_N, SELECT_N, OOD_N, SHAM_N = 28672,4096,16384,4096
PAIRS=((870101,870103),(870201,870203),(870301,870303))
FIRST_USE_SET=frozenset(FAMILIES)


def source(seed:int):
    if type(seed) is not int or seed not in FIRST_USE_SET:raise ValueError('P57_UNREGISTERED_SOURCE')
    role,laws=FAMILIES[seed];n={'fit':FIT_N,'ood':OOD_N,'sham':SHAM_N}[role]
    g=np.random.default_rng(seed)
    if role=='fit': ranges=((.1,1.2),(.55,.95),(.1,1.3),(.55,.95),(.015,.24),(.5,1.1))
    else:ranges=((0,1.6),(.95,1.35),(0,1.8),(.95,1.35),(.015,.35),(1.1,1.8))
    cg=g.uniform(*ranges[0],n).astype('float32')
    cs=g.uniform(*ranges[1],n).astype('float32')
    ga=g.uniform(*ranges[2],n).astype('float32')
    sa=g.uniform(*ranges[3],n).astype('float32')
    ca=g.uniform(*ranges[4],n).astype('float32')
    gb=g.uniform(*ranges[2],n).astype('float32')
    sb=g.uniform(*ranges[3],n).astype('float32')
    cb=g.uniform(*ranges[4],n).astype('float32')
    rs=g.uniform(*ranges[5],n).astype('float32')
    if role=='sham':ga.fill(0);gb.fill(0)
    theta=g.choice(np.array([-1.,1.],dtype='float32'),size=n)
    def noise(f):return (g.normal(size=n) if f=='G' else g.laplace(scale=1/math.sqrt(2),size=n)).astype('float32')
    ec,ea,eb,er=(noise(f) for f in laws)
    cue=(theta*cg+cs*ec).astype('float32')
    ctx=np.column_stack((cue,cg,cs,ga,sa,ca,gb,sb,cb,rs)).astype('float32')
    first=g.integers(0,3,size=n,dtype='int64')
    second=np.zeros(n,dtype='int64');want_second=g.random(n)<.5
    # Stage two logging randomly chooses only a test not already performed.
    second[want_second&(first==1)]=2
    second[want_second&(first==2)]=1
    t=g.integers(1,3,size=n)
    second[want_second&(first==0)]=t[want_second&(first==0)]
    maska=((first==1)|(second==1)).astype('float32')
    maskb=((first==2)|(second==2)).astype('float32')
    ma=np.where(maska>0,theta*ga+sa*ea,0.).astype('float32')
    mb=np.where(maskb>0,theta*gb+sb*eb,0.).astype('float32')
    final=g.choice(np.array([-1.,1.],dtype='float32'),size=n)
    reward=(theta*final+rs*er-ca*maska-cb*maskb).astype('float32')
    log=dict(context=ctx,first=first,second=second,mask_A=maska,mask_B=maskb,
             measurement_A=ma,measurement_B=mb,actual_final_action=final,observed_reward=reward)
    # All hidden physical facts stay exclusively in the evaluator tape.
    env=dict(context=ctx,theta=theta,noise_A=ea,noise_B=eb,reward_noise=rs*er)
    sh=hashlib.sha256(b''.join(x.tobytes() for x in (ctx,theta,ea,eb,rs*er,first,second,ma,mb,final,reward))).hexdigest()
    logsh=hashlib.sha256(b''.join(v.tobytes() for v in log.values())).hexdigest()
    return dict(seed=seed,role=role,laws=laws,log=log,env=env,sha256=sh,log_sha256=logsh)


def check_log(log):
    required={'context','first','second','mask_A','mask_B','measurement_A','measurement_B','actual_final_action','observed_reward'}
    if set(log)!=required:raise ValueError('P57_BAD_FACTUAL_LOG_SCHEMA')
    c=np.asarray(log['context']);n=len(c)
    if c.shape!=(n,10) or any(len(z)!=n for z in log.values()):raise ValueError('P57_BAD_LOG_SHAPE')
    if not np.isfinite(c).all() or any(not np.isfinite(z).all() for z in log.values()):raise ValueError('P57_NONFINITE_LOG')
    for name in ('A','B'):
        mask=log['mask_'+name];measure=log['measurement_'+name]
        if not np.isin(mask,[0.,1.]).all() or not np.all(measure[mask==0]==0):raise ValueError('P57_UNEXECUTED_MEASUREMENT_LEAK')
    if not np.isin(log['actual_final_action'],[-1.,1.]).all():raise ValueError('P57_INVALID_EXECUTED_ACTION')
    if np.any((log['first']!=0)&(log['first']==log['second'])):raise ValueError('P57_DUPLICATE_ACTION')


def moments(log):
    check_log(log)
    c=np.asarray(log['context'],dtype=np.float64)
    x=np.asarray(log['actual_final_action'],dtype=np.float64)
    z=(np.asarray(log['observed_reward'],dtype=np.float64)+c[:,5]*log['mask_A']+c[:,8]*log['mask_B'])/x
    def fit_channel(obs,gain,sigma,mask=None):
        if mask is not None:
            obs,gain,sigma=obs[mask],gain[mask],sigma[mask]
        v=gain/sigma;w=obs/sigma
        scale=float(np.mean(v*w*z[mask] if mask is not None else v*w*z)/np.mean(v*v))
        var=float(np.mean(w*w)-scale*scale*np.mean(v*v))
        return dict(scale=max(1e-3,scale),variance=max(.05,var),actual_observed=int(len(w)))
    fit=dict(prior_positive=float(np.clip((1+z.mean())/2,.01,.99)),
             reward_variance=max(.05,float(np.var(z)-1)),
             cue=fit_channel(c[:,0],c[:,1],c[:,2]))
    for label,col,maskcol in (('A',3,'mask_A'),('B',6,'mask_B')):
        # The sole observational difference is whether that diagnostic was actually purchased.
        ix=log[maskcol]>0
        fit[label]=fit_channel(np.asarray(log['measurement_'+label],dtype=np.float64),c[:,col],c[:,col+1],ix)
    fit['training_worlds']=len(c)
    fit['executed_training_A']=int(log['mask_A'].sum());fit['executed_training_B']=int(log['mask_B'].sum())
    return fit


def select_family(fit,observed):
    check_log(observed)
    if len(observed['context'])!=SELECT_N:raise ValueError('P57_SELECTION_HOLDOUT_WRONG_SIZE')
    c=np.asarray(observed['context'],dtype=np.float64)
    z=(np.asarray(observed['observed_reward'],dtype=np.float64)+c[:,5]*observed['mask_A']+c[:,8]*observed['mask_B'])/observed['actual_final_action']
    p=fit['prior_positive'];vr=fit['reward_variance']
    rscores={f:float(p56.log_mixture(p,z,vr,f).mean()) for f in ('G','L')};rf=max(rscores,key=rscores.get)
    labels={'reward':rf};scores={'reward':rscores}
    for key,col,measure,maskcol in (('cue',1,c[:,0],None),('A',3,observed['measurement_A'],observed['mask_A']),('B',6,observed['measurement_B'],observed['mask_B'])):
        ix=np.ones(len(c),dtype=bool) if maskcol is None else (maskcol>0)
        obs=measure[ix]/c[ix,col+1]
        mu=fit[key]['scale']*c[ix,col]/c[ix,col+1]
        s={f:float(p56.log_mixture(p,z[ix],vr,rf,obs,mu,fit[key]['variance'],f).mean()) for f in ('G','L')}
        labels[key]=max(s,key=s.get);scores[key]=s
    return dict(stats=fit,family=labels,selection_scores=scores,selection_cases=len(c),selection_acts_A=int(observed['mask_A'].sum()),selection_acts_B=int(observed['mask_B'].sum()))


def reward_shuffled_log(log,seed=870801):
    check_log(log)
    g=np.random.default_rng(seed);o={k:v.copy() for k,v in log.items()};c=o['context'];a=o['actual_final_action']
    z=(o['observed_reward']+c[:,5]*o['mask_A']+c[:,8]*o['mask_B'])/a
    o['observed_reward']=(z[g.permutation(len(z))]*a-c[:,5]*o['mask_A']-c[:,8]*o['mask_B']).astype('float32')
    return o


def log_odds(p):return math.log(p/(1-p))


def action_policy(context,model,policy,actual_measurements):
    """Client policy sees actual context and only chosen measurement values.

    The callback is a physical actuator; the callback never gets invoked for an
    unchosen test and does not expose hidden state to policy math.
    """
    c=np.asarray(context,dtype=np.float64);n=len(c)
    if policy not in ('selected','gaussian','laplace','never','cheapest','oracle','wrong'):
        raise ValueError('P57_UNKNOWN_POLICY')
    if policy=='oracle':
        st=dict(prior_positive=.5,cue=dict(scale=1,variance=1),A=dict(scale=1,variance=1),B=dict(scale=1,variance=1))
        fm=model['true_family']
    else:
        st=model['stats'];fm=model['family']
    if policy=='gaussian':fm=dict(cue='G',A='G',B='G')
    if policy=='laplace':fm=dict(cue='L',A='L',B='L')
    post=np.full(n,log_odds(st['prior_positive']))+p56.posterior_logratio(c[:,0]/c[:,2],st['cue']['scale']*c[:,1]/c[:,2],st['cue']['variance'],fm['cue'])
    def advantage(kind,bel):
        col=3 if kind==1 else 6;label='A' if kind==1 else 'B'
        return p56.diagnostic_gain(bel,st[label]['scale']*c[:,col]/c[:,col+1],st[label]['variance'],fm[label])-c[:,col+2]
    ga,gb=advantage(1,post),advantage(2,post)
    if policy=='never':first=np.zeros(n,dtype='int64')
    elif policy=='cheapest':first=np.where(c[:,5]<=c[:,8],1,2).astype('int64')
    else:first=np.where(np.maximum(ga,gb)>1e-12,np.where(ga>=gb,1,2),0).astype('int64')
    probe_A=(first==1);probe_B=(first==2)
    ma=np.zeros(n);mb=np.zeros(n)
    if probe_A.any():ma[probe_A]=actual_measurements('A',probe_A)
    if probe_B.any():mb[probe_B]=actual_measurements('B',probe_B)
    post+=probe_A*p56.posterior_logratio(ma/c[:,4],st['A']['scale']*c[:,3]/c[:,4],st['A']['variance'],fm['A'])
    post+=probe_B*p56.posterior_logratio(mb/c[:,7],st['B']['scale']*c[:,6]/c[:,7],st['B']['variance'],fm['B'])
    # Stage 2 is genuinely consequence-contingent: decide using the updated posterior.
    if policy=='never':second=np.zeros(n,dtype='int64')
    else:
        second=np.where((~probe_A)&(advantage(1,post)>1e-12),1,0)
        optionB=(~probe_B)&(advantage(2,post)>1e-12)
        second=np.where(optionB&((second==0)|(advantage(2,post)>advantage(1,post))),2,second).astype('int64')
    secondA=second==1;secondB=second==2
    if secondA.any():
        new=actual_measurements('A',secondA);ma[secondA]=new;probe_A|=secondA
        post+=secondA*p56.posterior_logratio(ma/c[:,4],st['A']['scale']*c[:,3]/c[:,4],st['A']['variance'],fm['A'])
    if secondB.any():
        new=actual_measurements('B',secondB);mb[secondB]=new;probe_B|=secondB
        post+=secondB*p56.posterior_logratio(mb/c[:,7],st['B']['scale']*c[:,6]/c[:,7],st['B']['variance'],fm['B'])
    return dict(first=first,second=second,mask_A=probe_A,mask_B=probe_B,measurement_A=ma,measurement_B=mb,
                action=np.where(post>=0,1.,-1.))


def executed(env,model,policy):
    c=np.asarray(env['context'],dtype=np.float64);n=len(c);theta=env['theta']
    def actuator(label,mask):
        col=3 if label=='A' else 6;noise=env['noise_A'] if label=='A' else env['noise_B']
        # This is the physically executed measurement only, never reveals the
        # other potential outcome to the caller.
        return theta[mask]*c[mask,col]+c[mask,col+1]*noise[mask]
    t0=time.perf_counter();record=action_policy(c,model,policy,actuator)
    cost=c[:,5]*record['mask_A']+c[:,8]*record['mask_B']
    reward=theta*record['action']+env['reward_noise']-cost
    h=hashlib.sha256(np.column_stack((record['first'],record['second'],record['action'],record['mask_A'],record['mask_B'],reward)).tobytes()).hexdigest()
    return dict(net=float(reward.mean()),probe_any=float((record['mask_A']|record['mask_B']).mean()),
                second_fraction=float((record['second']!=0).mean()),probes_A=int(record['mask_A'].sum()),probes_B=int(record['mask_B'].sum()),
                paid_cost=float(cost.sum()),runtime_s=time.perf_counter()-t0,history_sha256=h,
                reward=reward,record=record)


def bootstrap(a,b,seed):
    d=np.asarray(a)-np.asarray(b);g=np.random.default_rng(seed);out=np.empty(1000)
    for i in range(1000):out[i]=d[g.integers(0,len(d),len(d))].mean()
    return dict(mean=float(d.mean()),lower95=float(np.quantile(out,.025)),upper95=float(np.quantile(out,.975)),resamples=1000,seed=seed)


def run():
    start=time.perf_counter();rows={};hashes={}
    for fit_seed,ood_seed in PAIRS:
        src=source(fit_seed);log=src['log'];cut=FIT_N-SELECT_N
        tr={k:v[:cut] for k,v in log.items()};hold={k:v[cut:] for k,v in log.items()}
        fitted=select_family(moments(tr),hold);truth=src['laws']
        # Truth is passed only to the explicit privileged oracle, never to an ordinary policy.
        oracle_model={**fitted,'true_family':dict(zip(('cue','A','B','reward'),truth))}
        wrong=select_family(moments(reward_shuffled_log(tr)),hold)
        w=source(ood_seed);hashes[str(fit_seed)]=src['sha256'];hashes[str(ood_seed)]=w['sha256']
        methods={'selected':fitted,'gaussian':fitted,'laplace':fitted,'cheapest':fitted,'never':fitted,
                 'oracle':oracle_model,'wrong':wrong}
        observed={name:executed(w['env'],fitted if name not in ('oracle','wrong') else mm,name) for name,mm in methods.items()}
        sel=observed['selected'];fixed=max(observed[x]['net'] for x in ('gaussian','laplace','cheapest'))
        paired=bootstrap(sel['reward'],observed['never']['reward'],870901+ood_seed)
        gates={
            'correct_all_source_families':fitted['family']==dict(zip(('cue','A','B','reward'),truth)),
            'better_strong_fixed_0point005':sel['net']>=fixed+.005,
            'better_never_0point03':sel['net']>=observed['never']['net']+.03,
            'within_oracle_0point02':sel['net']>=observed['oracle']['net']-.02,
            'executes_at_least_4pct':sel['probe_any']>=.04,
            'outcome_contingent_second_probe_at_least_1pct':sel['second_fraction']>=.01,
            'paired_lower_positive':paired['lower95']>0,
            'wrong_credit_0point025':sel['net']>=observed['wrong']['net']+.025,
            'cold_exact':sel['history_sha256']==executed(w['env'],json.loads(json.dumps(fitted,sort_keys=True)),'selected')['history_sha256']
        }
        rows[str(ood_seed)]=dict(fit_seed=fit_seed,source_sha256=w['sha256'],fit_sha256=src['sha256'],fit_log_sha256=src['log_sha256'],
            true_family=dict(zip(('cue','A','B','reward'),truth)),fitted=fitted,
            arms={name:{k:v for k,v in result.items() if k not in ('reward','record')} for name,result in observed.items()},
            paired_ci=paired,gates=gates)
    sham=source(870401);hashes['870401']=sham['sha256'];m=rows['870303']['fitted']
    abst=executed(sham['env'],m,'selected');sham_pass=abst['probes_A']==0 and abst['probes_B']==0
    return dict(schema='ttt-hybrid.p57.two-physical-probe-factual-dev-receipt.v1',classification='AUTHOR_GENERATED_DEV_NO_INDEPENDENT_SCIENTIFIC_FIRST_USE',
        source_hashes=hashes,sham=dict(probe_count_A=abst['probes_A'],probe_count_B=abst['probes_B'],source_hash=sham['sha256']),
        rows=rows,all_frozen_pass=all(all(q['gates'].values()) for q in rows.values()) and sham_pass,
        elapsed_s=time.perf_counter()-start,full_final_Windows_Ubuntu_regression='NOT_RUN',external_scientific_first_use='NOT_RUN',FORGE='UNTOUCHED')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--receipt',type=Path,required=True)
    a=ap.parse_args();out=run();a.receipt.parent.mkdir(exist_ok=True,parents=True)
    a.receipt.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'all_pass':out['all_frozen_pass'],'rows':{k:{'families':x['fitted']['family'],
        'net':{n:v['net'] for n,v in x['arms'].items()},'probe_fractions':{n:(v['probe_any'],v['second_fraction']) for n,v in x['arms'].items()},
        'gates':x['gates']} for k,x in out['rows'].items()},'sham':out['sham'],'elapsed':out['elapsed_s']},indent=2))
if __name__=='__main__':main()
