"""P52: factual-reward policy-gradient learning of whether to pay for a probe.

Author-generated, precommitted DEV analogue. A diagnostic is executed only if
chosen; the policy receives only its actual physical measurement. There is no
unrealized reward, latent world sign, science source, or privileged Bayes law in
neural input or objective.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import ndtr
import torch
from torch import nn
from torch.nn import functional as F

N_STEPS=700
BATCH=128
TRAIN_SEED=820101
VALID_SEED=820103
OOD_SEED=820107
SEEDS=(8211,8223,8231)
CHECK_EVERY=100
EPS=1.e-8


def worlds(seed:int):
    if seed not in (TRAIN_SEED,VALID_SEED,OOD_SEED) or type(seed) is not int:
        raise ValueError('P52_SOURCE_NOT_FROZEN')
    g=np.random.default_rng(seed)
    n=N_STEPS*BATCH if seed==TRAIN_SEED else (1024 if seed==VALID_SEED else 2048)
    ood=seed==OOD_SEED
    u=lambda lo,hi:g.uniform(lo,hi,n).astype('float32')
    cue_gain=u(*( (0.,1.5) if ood else (.1,1.2) ))
    probe_gain=u(*( (0.,1.6) if ood else (.1,1.2) ))
    cue_sigma=u(*( (.95,1.25) if ood else (.55,.95) ))
    probe_sigma=u(*( (.95,1.25) if ood else (.55,.95) ))
    cost=u(*( (.05,.65) if ood else (.05,.4) ))
    theta=g.choice(np.array([-1.,1.],dtype='float32'),size=n)
    eps_c=g.normal(size=n).astype('float32')
    eps_p=g.normal(size=n).astype('float32')
    cue=(theta*cue_gain+cue_sigma*eps_c).astype('float32')
    prior=np.stack((cue,cue_gain,cue_sigma,probe_gain,probe_sigma,cost),axis=-1)
    # Independent experiment coin tape; not an observed probe or an oracle outcome.
    random_probe=(g.random(n)<.5).astype('float32')
    random_action=g.choice(np.array([-1.,1.],dtype='float32'),size=n)
    arrays=dict(prior=prior,theta=theta,eps_probe=eps_p,random_probe=random_probe,random_action=random_action)
    digest=hashlib.sha256(b''.join(x.tobytes() for x in arrays.values())).hexdigest()
    return {**{k:torch.from_numpy(v.copy()) for k,v in arrays.items()},'sha256':digest,'n':n,'seed':seed}


class FactorizedPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.probe=nn.Sequential(nn.Linear(6,50),nn.Tanh(),nn.Linear(50,1))
        self.exploit=nn.Sequential(nn.Linear(8,50),nn.Tanh(),nn.Linear(50,1))

    def first(self,prior):
        return self.probe(prior).squeeze(-1), None

    def second(self,prior,probe_flag,measurement,context=None):
        return self.exploit(torch.cat((prior,probe_flag[:,None],measurement[:,None]),dim=1)).squeeze(-1)


class ConventionalGRUPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell=nn.GRUCell(6,14)
        self.probe_head=nn.Linear(14,1)
        self.exploit_head=nn.Linear(14,1)

    def first(self,prior):
        h=self.cell(prior,prior.new_zeros((len(prior),14)))
        return self.probe_head(h).squeeze(-1),h

    def second(self,prior,probe_flag,measurement,context=None):
        if context is None: raise ValueError('P52_RECURRENT_STATE_MISSING')
        second=torch.stack((prior[:,0],probe_flag,measurement,prior[:,3],prior[:,4],prior[:,5]),dim=1)
        h=self.cell(second,context)
        return self.exploit_head(h).squeeze(-1)


def count_params(model):
    return sum(int(z.numel()) for z in model.parameters())


def physical_feedback(batch,probe):
    """Single realized diagnostic outcome. Unchosen measurement stays zero."""
    if not bool(((probe==0)|(probe==1)).all()): raise ValueError('P52_BAD_PHYSICAL_ACTION')
    theta=batch['theta']; gain=batch['prior'][:,3];sigma=batch['prior'][:,4]
    return torch.where(probe>0, theta*gain+sigma*batch['eps_probe'],torch.zeros_like(theta))


def signed_reward(batch,probe,action):
    """Only executed final action determines observed factual reward."""
    return batch['theta']*action - batch['prior'][:,5]*probe


def take_batch(data,ix):
    return {k:v[ix] for k,v in data.items() if isinstance(v,torch.Tensor)}


def act(model,batch,stochastic=False):
    prior=batch['prior'];logit,h=model.first(prior)
    if stochastic:
        probe=(torch.rand_like(logit)<torch.sigmoid(logit)).to(prior.dtype)
    else:probe=(logit>=0).to(prior.dtype)
    measurement=physical_feedback(batch,probe)
    final=model.second(prior,probe,measurement,h)
    if stochastic:positive=(torch.rand_like(final)<torch.sigmoid(final)).to(prior.dtype)
    else:positive=(final>=0).to(prior.dtype)
    action=positive*2-1
    reward=signed_reward(batch,probe,action)
    return dict(probe=probe,action=action,measurement=measurement,reward=reward,
                probe_logit=logit,action_logit=final,positive=positive)


def train(model,train_data,val_data,wrong_credit=False):
    optim=torch.optim.AdamW(model.parameters(),lr=.002)
    ema=0.0;curve=[];started=time.perf_counter()
    for step in range(N_STEPS):
        model.train();batch=take_batch(train_data,slice(step*BATCH,(step+1)*BATCH))
        o=act(model,batch,stochastic=True)
        factual=o['reward'].detach()
        credit=factual.roll(1) if wrong_credit else factual
        adv=credit-ema
        entropy_probe=F.binary_cross_entropy_with_logits(o['probe_logit'],o['probe'],reduction='none')
        entropy_action=F.binary_cross_entropy_with_logits(o['action_logit'],o['positive'],reduction='none')
        loss=((entropy_probe+entropy_action)*adv).mean()
        optim.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optim.step()
        ema=.95*ema+.05*float(factual.mean())
        if (step+1)%CHECK_EVERY==0:
            model.eval()
            with torch.no_grad():curve.append(float(act(model,val_data)['reward'].mean()))
    return dict(optimizer_updates=N_STEPS,training_seconds=time.perf_counter()-started,
                validation_reward_curve=curve,last_two_difference=abs(curve[-1]-curve[-2]),
                wrong_credit=wrong_credit)


def standard_normal_cdf(x):
    return ndtr(x)


def bayes_reference(data,policy):
    """Known-generative-law comparator, explicitly not an unprivileged learner.

    Its probe decision uses the Gaussian mixture predictive correct-decision
    probability; physical reward is still computed only for its actual choice.
    """
    prior=data['prior'].numpy();theta=data['theta'].numpy();eps=data['eps_probe'].numpy()
    cue,cg,cs,pg,ps,cost=prior.T
    log_odds=2*cg*cue/(cs*cs)
    prob=1/(1+np.exp(-np.clip(log_odds,-50,50)))
    expected_no=np.abs(2*prob-1)
    # Bayes action after probe is positive for a probe measurement > boundary.
    boundary=-log_odds*ps*ps/(2*np.maximum(pg,1.e-8))
    chance_correct=prob*standard_normal_cdf((pg-boundary)/ps)+(1-prob)*standard_normal_cdf((boundary+pg)/ps)
    expected_probe=2*chance_correct-1-cost
    if policy=='bayes':probe=(expected_probe>expected_no).astype('float32')
    elif policy=='always':probe=np.ones_like(theta)
    elif policy=='never':probe=np.zeros_like(theta)
    elif policy=='simple':probe=((np.abs(cue)/cs<.5)&(cost<.25)).astype('float32')
    elif policy=='random':probe=data['random_probe'].numpy()
    else:raise ValueError('P52_UNKNOWN_REFERENCE')
    # The unexecuted probe measurement is never constructed as an observation.
    measurement=np.zeros_like(theta)
    mask=probe>0
    measurement[mask]=theta[mask]*pg[mask]+ps[mask]*eps[mask]
    posterior=log_odds+probe*2*pg*measurement/(ps*ps)
    action=np.where(posterior>=0,1.,-1.)
    if policy=='random':action=data['random_action'].numpy()
    reward=theta*action-cost*probe
    return dict(net_reward=float(np.mean(reward)),probe_rate=float(np.mean(probe)),
                reward_by_cost=[float(np.mean(reward[(cost<.20)])),float(np.mean(reward[(cost>=.4)]))],
                decision=action,probed=probe,expected_optimal_probe_prob=float(np.mean(expected_probe>expected_no)))


def summary(model,data):
    model.eval();started=time.perf_counter()
    with torch.no_grad():x=act(model,data)
    p=data['prior'];cost=p[:,5];cue=p[:,0];cs=p[:,2];gain=p[:,3]
    weak=(torch.abs(cue)/cs<.5)&(gain>.8)
    low=weak&(cost<.2);high=weak&(cost>=.4)
    def mean_mask(x,mask):
        return float(x[mask].mean()) if bool(mask.any()) else None
    return dict(net_reward=float(x['reward'].mean()),probe_rate=float(x['probe'].mean()),
                accuracy=float((x['action']==data['theta']).float().mean()),
                probed_low_weak_highgain=mean_mask(x['probe'],low),
                probed_high_weak_highgain=mean_mask(x['probe'],high),
                group_low_count=int(low.sum()),group_high_count=int(high.sum()),
                inference_seconds=time.perf_counter()-started)


def cold(model,data,factory):
    model.eval();b=take_batch(data,slice(0,8))
    with torch.no_grad():
        a=act(model,b)
        buf=io.BytesIO();torch.save(model.state_dict(),buf);buf.seek(0)
        restored=torch.load(buf,weights_only=True)
        other=factory().eval();other.load_state_dict(restored)
        z=act(other,b)
    return (bool(all(torch.equal(a[k],z[k]) for k in ('probe_logit','action_logit','action','probe','reward'))),
            hashlib.sha256(buf.getvalue()).hexdigest())


def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    started=time.perf_counter()
    fit=worlds(TRAIN_SEED);val=worlds(VALID_SEED);ood=worlds(OOD_SEED)
    refs={k:{i:v for i,v in bayes_reference(ood,k).items() if i not in ('decision','probed')}
          for k in ('bayes','always','never','simple','random')}
    counts={'candidate':count_params(FactorizedPolicy()),'gru':count_params(ConventionalGRUPolicy()),'wrong_credit':count_params(FactorizedPolicy())}
    records=[]
    for seed in SEEDS:
        arms={}
        for name,factory in (('candidate',FactorizedPolicy),('gru',ConventionalGRUPolicy),('wrong_credit',FactorizedPolicy)):
            torch.manual_seed(seed);model=factory()
            torch.manual_seed(seed+100000)
            tr=train(model,fit,val,wrong_credit=(name=='wrong_credit'))
            result=summary(model,ood)
            same,sha=cold(model,ood,factory)
            arms[name]=dict(train=tr,ood=result,cold_exact=same,cold_sha256=sha)
        cand=arms['candidate']['ood'];gru=arms['gru']['ood'];wrong=arms['wrong_credit']['ood']
        lower=cand['probed_low_weak_highgain'];upper=cand['probed_high_weak_highgain']
        gates=dict(
            beats_trained_gru_and_simple_by_point02=cand['net_reward']>=max(gru['net_reward'],refs['always']['net_reward'],refs['never']['net_reward'],refs['simple']['net_reward'])+.02,
            within_point05_analytic_bayes=cand['net_reward']>=refs['bayes']['net_reward']-.05,
            beats_corrupt_reward_credit_point05=cand['net_reward']>=wrong['net_reward']+.05,
            learns_probe_cost_gradient=(lower is not None and upper is not None and lower>=upper+.20),
            all_cold_exact=all(q['cold_exact'] for q in arms.values()),
            convergence_check_all=all(x['train']['last_two_difference']<=.02 for x in arms.values()),
            )
        records.append(dict(seed=seed,arms=arms,gates=gates))
    return dict(schema='ttt-hybrid.p52.factual-probe-value-dev-receipt.v1',
                source_kind='NEW_AUTHORED_DEV_NOT_PROTECTED_SCIENTIFIC_FIRST_USE',
                cohort_sha256={str(d['seed']):d['sha256'] for d in (fit,val,ood)},
                train_episodes=fit['n'],validation_episodes=val['n'],ood_episodes=ood['n'],
                parameters=counts,updates_per_arm=N_STEPS,batch=BATCH,
                training_worlds_same_across_arms=True,training_model_seeds=list(SEEDS),
                analytic_controls=refs,records=records,
                all_predeclared_seed_gates=all(all(v for v in row['gates'].values()) for row in records),
                full_wall_seconds=time.perf_counter()-started,
                science_first_use='NOT_RUN',original_windows_ubuntu_full_regression='NOT_RUN',FORGE='UNTOUCHED')


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args();x=run();args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'sources':x['cohort_sha256'],'params':x['parameters'],
        'reference_rewards':{k:v['net_reward'] for k,v in x['analytic_controls'].items()},
        'per_seed':{row['seed']:{'reward':{k:v['ood']['net_reward'] for k,v in row['arms'].items()},
             'gates':row['gates'],'probe_rates':{k:v['ood']['probe_rate'] for k,v in row['arms'].items()},
             'train_s':{k:v['train']['training_seconds'] for k,v in row['arms'].items()}}
             for row in x['records']},
        'all':x['all_predeclared_seed_gates'],'wall':x['full_wall_seconds']},indent=2))

if __name__=='__main__':main()
