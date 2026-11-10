"""P54 frozen DEV: noisy factual reward removes exact hidden-sign label shortcut.

A shared log of actually executed diagnostic actions and final rewards trains a
classical Gaussian moment policy and two ordinary equally exposed neural
reward-regression controls. This is not independent science or a native model.
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
from torch import nn
from scripts import native_p52_probe_value_dev as p52
from scripts import native_p53_factual_likelihood_dev as p53

SOURCES = {
    840001: (49152, 'fit'), 840101: (2048, 'validation'),
    840201: (8192, 'ood_a'), 840301: (8192, 'ood_b'),
    840401: (4096, 'sham')
}
MODEL_SEEDS = (8411,8423,8431)
TRAIN_UPDATES = 600
BATCH = 128
SHUFFLE_SEED = 840801
BOOTSTRAP_SEED = 840901


def source(seed):
    if type(seed) is not int or seed not in SOURCES:
        raise ValueError('P54_UNFROZEN_SOURCE')
    n, cohort = SOURCES[seed]
    g = np.random.default_rng(seed)
    if cohort in ('fit', 'validation'):
        c_ranges=((.1,1.2),(.1,1.2),(.55,.95),(.55,.95),(.05,.4),(.5,1.1))
    elif cohort in ('ood_a','sham'):
        c_ranges=((0,1.5),(0,1.6),(.95,1.25),(.95,1.25),(.05,.65),(1.1,1.6))
    else:
        c_ranges=((0,2),(0,2),(1.1,1.4),(1.1,1.4),(.01,.75),(1.4,2.0))
    arrays=[g.uniform(lo,hi,n).astype('float32') for lo,hi in c_ranges]
    cg,pg,cs,ps,cost,rs=arrays
    if cohort=='sham':pg.fill(0.)
    theta=g.choice(np.array([-1.,1.],dtype='float32'),n)
    ec=g.normal(size=n).astype('float32')
    ep=g.normal(size=n).astype('float32')
    er=g.normal(size=n).astype('float32')
    cue=theta*cg+cs*ec
    prior=np.stack((cue,cg,cs,pg,ps,cost),axis=-1)
    logged_probe=(g.random(n)<.5).astype('float32')
    logged_action=g.choice(np.array([-1.,1.],dtype='float32'),n)
    actual_measurement=np.where(logged_probe>0,theta*pg+ps*ep,0.).astype('float32')
    observed_net_reward=(theta*logged_action+rs*er-cost*logged_probe).astype('float32')
    # THESE ARE THE ONLY FIELDS MADE AVAILABLE TO ANY FITTER.
    log=dict(prior=prior,actual_probe=logged_probe,actual_measurement=actual_measurement,
             actual_final_action=logged_action,observed_net_reward=observed_net_reward)
    # Physics tape is for evaluation only; never given to a fit() method.
    evaluation=dict(prior=prior,theta=theta,eps_probe=ep,reward_noise=rs*er)
    digest=hashlib.sha256(b''.join(x.tobytes() for x in [prior,theta,ep,rs*er,logged_probe,logged_action,actual_measurement,observed_net_reward])).hexdigest()
    loghash=hashlib.sha256(b''.join(x.tobytes() for x in log.values())).hexdigest()
    return dict(seed=seed,kind=cohort,n=n,log=log,evaluation=evaluation,
                source_sha256=digest,logged_sha256=loghash)


def fit_moments(log):
    fields=['prior','actual_probe','actual_measurement','actual_final_action','observed_net_reward']
    if set(log)!=set(fields):raise ValueError('P54_UNAUTHORIZED_FIT_FIELD')
    c,p,m,a,r=[np.asarray(log[name],dtype=np.float64) for name in fields]
    n=len(c)
    if c.shape!=(n,6) or any(x.shape!=(n,) for x in [p,m,a,r]) or n<128:
        raise ValueError('P54_BAD_FACTUAL_SHAPE')
    if not all(np.isfinite(x).all() for x in (c,p,m,a,r)):
        raise ValueError('P54_NONFINITE')
    if not np.all(np.isin(p,(0,1))) or not np.all(np.isin(a,(-1.,1))):
        raise ValueError('P54_UNEXECUTED_ACTION')
    if np.any(m[p==0]!=0):raise ValueError('P54_UNEXECUTED_MEASUREMENT')
    if (c[:,2]<=0).any() or (c[:,4]<=0).any():raise ValueError('P54_INVALID_PHYSICAL_SIGMA')
    mask=p>0
    if mask.sum()<100 or (~mask).sum()<100:raise ValueError('P54_INSUFFICIENT_EXECUTED_PROBES')
    # z is a NOISY post-event reward-derived *proxy*, NOT the hidden sign:
    # E[z | theta] = theta; Var(z|theta) = reward noise variance >0.
    z=(r+c[:,5]*p)/a
    if np.max(np.abs(np.abs(z)-1))<1e-4:raise ValueError('P54_EXACT_SIGN_SHORTCUT_SOURCE')
    prior_positive=float(np.clip((1+z.mean())/2,.01,.99))
    def estimate_gain_variance(response,context,supervision):
        # Since reward noise is independent of cue/probe noise and actions,
        # E[z*signal] = learned_gain * known_strength, even without theta labels.
        scale=float(np.mean(supervision*response*context)/np.mean(context**2))
        scale=max(0.,scale)
        variance=float(np.mean(response**2)-scale*scale*np.mean(context**2))
        variance=max(.01,variance)
        return scale,variance
    cue_gain,cue_var=estimate_gain_variance(c[:,0]/c[:,2],c[:,1]/c[:,2],z)
    probe_gain,probe_var=estimate_gain_variance(m[mask]/c[mask,4],c[mask,3]/c[mask,4],z[mask])
    return dict(prior_positive=prior_positive,cue_scale=cue_gain,cue_variance=cue_var,
                probe_scale=probe_gain,probe_variance=probe_var,
                physical_training_facts=n,executed_training_probes=int(mask.sum()),
                reward_proxy_variance=float(np.var(z)),
                exact_sign_fraction=float((np.abs(np.abs(z)-1)<1e-4).mean()))


def shuffled_credit_log(log):
    rng=np.random.default_rng(SHUFFLE_SEED)
    indices=rng.permutation(len(log['actual_probe']))
    out={name:v.copy() for name,v in log.items()}
    a=out['actual_final_action'];p=out['actual_probe'];cost=out['prior'][:,5]
    # Keep same real-world reward proxy values, scramble *which world* receives
    # them, while maintaining form of the actual observed reward ledger.
    original_z=(out['observed_net_reward']+cost*p)/a
    out['observed_net_reward']=(original_z[indices]*a-cost*p).astype(np.float32)
    return out


def decisions(tape,model,policy='moment',exploit=None):
    if policy not in ('moment','never','always','bayes'):raise ValueError('P54_BAD_POLICY')
    c=np.asarray(tape['prior'],dtype=np.float64)
    th=np.asarray(tape['theta'],dtype=np.float64)
    ep=np.asarray(tape['eps_probe'],dtype=np.float64)
    reward_noise=np.asarray(tape['reward_noise'],dtype=np.float64)
    cue,cg,cs,pg,ps,cost=c.T
    if policy=='bayes': prior_positive,sg,vg,sp,vp=.5,1.,1.,1.,1.
    else:
        prior_positive=model['prior_positive']
        sg,vg,sp,vp=(model[k] for k in ('cue_scale','cue_variance','probe_scale','probe_variance'))
    logodds=math.log(prior_positive/(1-prior_positive)) + 2*sg*cg*cue/(vg*cs*cs)
    pp=expit(logodds);no_value=np.abs(2*pp-1)
    mu=sp*pg;std=ps*math.sqrt(vp)
    # When learned probe gain = 0 the diagnostic cannot improve a decision.
    boundary=-logodds*std*std/(2*np.maximum(mu,1e-10))
    expected_accuracy=pp*ndtr((mu-boundary)/std)+(1-pp)*ndtr((boundary+mu)/std)
    expected_gain=2*expected_accuracy-1-no_value
    if policy=='always':probe=np.ones(len(c),dtype=np.float64)
    elif policy=='never':probe=np.zeros(len(c),dtype=np.float64)
    else:probe=(expected_gain>cost+1e-12).astype(np.float64)
    measurement=np.zeros(len(c),dtype=np.float64)
    observed=probe==1
    measurement[observed]=th[observed]*pg[observed]+ps[observed]*ep[observed]
    if exploit is None:
        posterior=logodds+2*sp*pg*measurement/(vp*ps*ps)
        action=np.where(posterior>=0,1.,-1.)
    else:
        action=exploit(c,probe,measurement)
    net=th*action+reward_noise-cost*probe
    weak=(np.abs(cue)/cs<.5)&(pg>.8)
    low=weak&(cost<.2);high=weak&(cost>=.4)
    result=dict(net_reward=float(net.mean()),probe_fraction=float(probe.mean()),
        reward=net,actual_probe=probe,actual_action=action,actual_measurement=measurement,
        actual_executed_probes=int(probe.sum()),
        low_cost_probe=float(probe[low].mean()) if low.any() else None,
        high_cost_probe=float(probe[high].mean()) if high.any() else None,
        low_group_n=int(low.sum()),high_group_n=int(high.sum()),
        decisions_sha256=hashlib.sha256(np.column_stack((probe,action,net)).tobytes()).hexdigest())
    return result


class MLPRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cue=nn.Sequential(nn.Linear(6,31),nn.Tanh(),nn.Linear(31,1))
        self.post=nn.Sequential(nn.Linear(8,31),nn.Tanh(),nn.Linear(31,1))
    def forward(self,prior,flag,measurement):
        pre=self.cue(prior).squeeze(-1)
        post=self.post(torch.cat((prior,flag[:,None],measurement[:,None]),-1)).squeeze(-1)
        return pre,post


class GRURegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell=nn.GRUCell(6,10)
        self.first=nn.Linear(10,1)
        self.second=nn.Linear(10,1)
    def forward(self,prior,flag,measurement):
        h=self.cell(prior,prior.new_zeros((len(prior),10)))
        first=self.first(h).squeeze(-1)
        second_input=torch.stack((prior[:,0],flag,measurement,prior[:,3],prior[:,4],prior[:,5]),-1)
        v=self.cell(second_input,h)
        return first,self.second(v).squeeze(-1)


def fit_neural(model,train_log,val_log,indices):
    opt=torch.optim.AdamW(model.parameters(),lr=.002)
    def tensors(log):
        prior=torch.from_numpy(log['prior'].copy())
        p=torch.from_numpy(log['actual_probe'].copy());m=torch.from_numpy(log['actual_measurement'].copy())
        z=torch.from_numpy((log['observed_net_reward']+log['prior'][:,5]*log['actual_probe'])/log['actual_final_action'])
        return prior,p,m,z
    a=tensors(train_log);v=tensors(val_log);curve=[];started=time.perf_counter()
    for step in range(TRAIN_UPDATES):
        model.train();ix=indices[step]
        first,post=model(a[0][ix],a[1][ix],a[2][ix])
        z=a[3][ix]
        loss=((first-z)**2+(post-z)**2).mean()
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        if (step+1)%100==0:
            model.eval()
            with torch.no_grad():
                pred,post=model(*v[:3]);curve.append(float((((pred-v[3])**2+(post-v[3])**2).mean())))
    return dict(steps=TRAIN_UPDATES,seconds=time.perf_counter()-started,val_curve=curve,
                last_relative_change=abs(curve[-1]-curve[-2])/max(curve[-2],1e-8))


def neural_exploitation(model,tape,model_params):
    model.eval()
    def exploit(prior,probe,measurement):
        with torch.no_grad():
            p=torch.tensor(prior,dtype=torch.float32);f=torch.tensor(probe,dtype=torch.float32)
            m=torch.tensor(measurement,dtype=torch.float32)
            _,z=model(p,f,m)
            return np.where(z.numpy()>=0,1.,-1.)
    result=decisions(tape,model_params,'moment',exploit=exploit)
    return {k:v for k,v in result.items() if k not in ('reward','actual_probe','actual_action','actual_measurement')}


def bootstrap(a,b,seed):
    d=np.asarray(a)-np.asarray(b);g=np.random.default_rng(seed)
    vals=np.empty(1000)
    for i in range(1000):vals[i]=d[g.integers(0,len(d),len(d))].mean()
    return dict(mean=float(d.mean()),lower_95=float(np.percentile(vals,2.5)),
                upper_95=float(np.percentile(vals,97.5)),seed=seed,resamples=1000)


def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    started=time.perf_counter();src={s:source(s) for s in SOURCES}
    a=fit_moments(src[840001]['log'])
    wrong=fit_moments(shuffled_credit_log(src[840001]['log']))
    # Regression and legacy exact-sign fitting must reject noisy labels as exact signs.
    exact_sign_refused=False
    try:p53.fitted_likelihood(src[840001]['log'])
    except ValueError as error:exact_sign_refused='P53_REWARD_NOT_BINARY_EFFECT_IDENTIFIABLE' in str(error)
    results={}
    for s in (840201,840301,840401):
        tape=src[s]['evaluation']
        actions={name:decisions(tape,model,policy) for name,model,policy in (
            ('moment',a,'moment'),('never',a,'never'),('always',a,'always'),
            ('bayes',a,'bayes'),('shuffled',wrong,'moment'))}
        ci=bootstrap(actions['moment']['reward'],actions['never']['reward'],BOOTSTRAP_SEED+s)
        mo=actions['moment'];mn=actions['never'];ba=actions['bayes'];sh=actions['shuffled']
        passed={
          'better_than_never_0point02':mo['net_reward']>=mn['net_reward']+.02,
          'within_0point025_Bayes':mo['net_reward']>=ba['net_reward']-.025,
          'actual_probes_5_to_95pct':.05<=mo['probe_fraction']<=.95,
          'paired_CI_positive':ci['lower_95']>0,
          'cost_sensitive':(mo['low_group_n']>=30 and mo['high_group_n']>=30 and
                            mo['low_cost_probe']>=mo['high_cost_probe']+.15),
          'beating_wrong_credit_0point025':mo['net_reward']>=sh['net_reward']+.025,
        } if s!=840401 else {'zero_sham_probe':mo['actual_executed_probes']==0,
                               'model_aware_zero_sham':ba['actual_executed_probes']==0}
        results[str(s)]={'arms':{name:{k:v for k,v in arm.items() if k not in
                                 ('reward','actual_probe','actual_action','actual_measurement')}
                              for name,arm in actions.items()},
                         'paired_ci':ci,'gates':passed}
    # Identical *observed physical training logs* for both neural families and
    # moment model; each neural family uses the exact SAME batch indices.
    generator=torch.Generator().manual_seed(840701)
    indices=torch.randint(0,src[840001]['n'],(TRAIN_UPDATES,BATCH),generator=generator)
    indexhash=hashlib.sha256(indices.numpy().tobytes()).hexdigest()
    neural=[]
    for seed in MODEL_SEEDS:
        record={}
        for name,cls in (('MLP',MLPRegressor),('GRU',GRURegressor)):
            torch.manual_seed(seed)
            m=cls()
            trained=fit_neural(m,src[840001]['log'],src[840101]['log'],indices)
            evaluations={str(s):neural_exploitation(m,src[s]['evaluation'],a)
                         for s in (840201,840301)}
            # Check fresh instance with exact loaded parameters and same
            # pre/post diagnostic predictions, no stochastic replay shortcuts.
            buffer=io.BytesIO();torch.save(m.state_dict(),buffer);buffer.seek(0)
            weights=torch.load(buffer,weights_only=True)
            clone=cls();clone.load_state_dict(weights);clone.eval();m.eval()
            log=src[840101]['log'];p=torch.from_numpy(log['prior'][:16].copy())
            flag=torch.from_numpy(log['actual_probe'][:16].copy())
            meas=torch.from_numpy(log['actual_measurement'][:16].copy())
            with torch.no_grad():
                original=m(p,flag,meas);replay=clone(p,flag,meas)
            cold=all(torch.equal(a,b) for a,b in zip(original,replay))
            record[name]={'parameters':sum(x.numel() for x in m.parameters()),
                          'training':trained,'ood_under_same_moment_probe_actions':evaluations,
                          'cold_bit_exact':cold}
        neural.append(dict(seed=seed,arms=record))
    # Fitted model is pure learned sufficient statistics; exact JSON custody.
    model2=json.loads(json.dumps(a,sort_keys=True))
    cold=all(decisions(src[s]['evaluation'],a)['decisions_sha256']==
             decisions(src[s]['evaluation'],model2)['decisions_sha256'] for s in (840201,840301,840401))
    all_positive=exact_sign_refused and cold and all(r['gates'] and all(r['gates'].values()) for r in results.values())
    return dict(schema='ttt-hybrid.p54.noisy-reward-equal-log-dev-receipt.v1',
       source_sha256={str(s):w['source_sha256'] for s,w in src.items()},
       log_sha256={str(s):w['logged_sha256'] for s,w in src.items()},
       fit=a,wrong_credit=wrong,identical_logged_fit_across_methods=True,
       actual_training_probe_count=a['executed_training_probes'],
       exact_P53_sign_fitter_refused=exact_sign_refused,moment_cold_exact=cold,
       neural_batch_indices_sha256=indexhash,neural=neural,results=results,
       frozen_dev_positive=all_positive,wall_seconds=time.perf_counter()-started,
       independent_external_science='NOT_RUN',original_full_Windows_Ubuntu_regression='NOT_RUN',FORGE='UNTOUCHED')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--receipt',type=Path,required=True)
    args=ap.parse_args();out=run();args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'sources':out['source_sha256'],'fit':out['fit'],
      'ood':{s:{'rewards':{k:v['net_reward'] for k,v in row['arms'].items()},
                'probes':{k:v['probe_fraction'] for k,v in row['arms'].items()},
                'CI':row['paired_ci'],'gates':row['gates']} for s,row in out['results'].items()},
      'neural':{m['seed']:{name:{'params':v['parameters'],
        'validation_last_change':v['training']['last_relative_change'],
        'ood_reward':{s:q['net_reward'] for s,q in v['ood_under_same_moment_probe_actions'].items()}}
        for name,v in m['arms'].items()} for m in out['neural']},
      'all':out['frozen_dev_positive'],'seconds':out['wall_seconds']},indent=2))

if __name__=='__main__':main()
