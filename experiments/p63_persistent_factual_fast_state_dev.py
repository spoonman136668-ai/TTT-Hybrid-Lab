"""P63: persistent online gradient fast weights vs conventional factual recurrence.

Frozen authored DEV only, no authentic external first use. No unrevealed
alternative outcomes or latent source parameters are used by the learners.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import platform
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn

SPLITS = {'fit': (995201,512,.10,[(16,.7)]),
          'val': (995203,128,.10,[(16,.7)]),
          'ood': (995207,128,.16,[(8,.8),(24,.8)])}
SEEDS=(996101,996103,996107)
UPDATES,BATCH,CHECK_EVERY=240,96,40
SAMPLER_SEED=996199
T=32


def source(name):
    if name not in SPLITS: raise ValueError('P63_UNKNOWN_SOURCE')
    seed,n,sigma,switches=SPLITS[name]
    rng=np.random.default_rng(seed)
    sign=rng.choice(np.array([-1,1],np.int8),n)
    actions=rng.choice(np.array([-1.,1.],np.float32),(n,T))
    confound=rng.uniform(-.15,.15,n)
    y=np.empty((n,T),np.float32)
    latent=np.empty((n,T),np.int8)
    for t in range(T):
        for step,prob in switches:
            if t==step:
                sign=np.where(rng.random(n)<prob,-sign,sign)
        confound=confound+rng.normal(0,.02,n)
        latent[:,t]=sign
        y[:,t]=(confound+.20*sign*actions[:,t]+rng.normal(0,sigma,n)).astype(np.float32)
    # Destructive wrong credit is independent and non-invertible, not a roll.
    corruption=np.random.default_rng(seed+909).choice(np.array([-1.,1.],np.float32),(n,T))
    raw={'executed_action':actions,'real_observed_delta':y,'wrong_action_credit':corruption,
         'hidden_evaluator_sign':latent}
    h=hashlib.sha256(b''.join(np.ascontiguousarray(v).tobytes() for v in raw.values())).hexdigest()
    return {k:torch.from_numpy(v) for k,v in raw.items()},h


def allowed(d,idx=None):
    names=('executed_action','real_observed_delta','wrong_action_credit')
    return {k:(d[k] if idx is None else d[k][idx]) for k in names}


class FastState(nn.Module):
    """Trainable online normalized two-feature gradient; no history replay.

    Factual state only [intercept,effect] for each independent world. Read
    for the next *executed* action before updating using the realized response.
    Persistent state can be serialized mid-stream and cold resumed exactly.
    """
    def __init__(self, learned=True):
        super().__init__()
        self.learned=learned
        if learned:
            self.logit_lr=nn.Parameter(torch.tensor(math.log(.2/.8)))
            self.logit_decay=nn.Parameter(torch.tensor(math.log(.97/.03)))
        else:
            self.register_buffer('fixed_lr',torch.tensor(.2))
            self.register_buffer('fixed_decay',torch.tensor(.97))

    def rates(self):
        if self.learned:
            return self.logit_lr.sigmoid(),self.logit_decay.sigmoid()
        return self.fixed_lr,self.fixed_decay

    def initial_state(self,batch,ref):
        return ref.new_zeros((batch,2))

    def read(self,state,action):
        if state.ndim!=2 or state.shape[1]!=2 or action.shape!=state[:,0].shape:
            raise ValueError('P63_INVALID_STATE_OR_ACTION')
        return state[:,0]+state[:,1]*action

    def observe(self,state,actual_action,actual_delta):
        if state.ndim!=2 or actual_action.shape!=state[:,0].shape or actual_delta.shape!=actual_action.shape:
            raise ValueError('P63_INVALID_FACTUAL_UPDATE')
        if not bool(torch.isfinite(actual_action).all() and torch.isfinite(actual_delta).all()):
            raise ValueError('P63_NONFINITE_FACT')
        if not bool((actual_action.abs()==1).all()):
            raise ValueError('P63_INVALID_EXECUTED_ACTION')
        lr,decay=self.rates()
        error=actual_delta-self.read(state,actual_action)
        features=torch.stack((torch.ones_like(actual_action),actual_action),dim=-1)
        return decay*state+lr*error[:,None]*features/2

    def forward(self,d,wrong_credit=False,carry=None):
        a=d['executed_action']; y=d['real_observed_delta']
        if a.ndim!=2 or a.shape!=y.shape or a.shape[1]<1 or not bool(torch.isfinite(a).all() and torch.isfinite(y).all()):
            raise ValueError('P63_INVALID_FACTUAL_STREAM')
        state=self.initial_state(len(a),a) if carry is None else carry
        if wrong_credit:
            fake=d['wrong_action_credit']
            if fake.shape!=a.shape:raise ValueError('P63_INVALID_WRONG_CREDIT')
        preds=[]
        for t in range(a.shape[1]):
            preds.append(self.read(state,a[:,t]))
            # Deliberately discredited arm learns with independent action labels;
            # the *actual* response remains the same. No hidden mode leakage.
            observed_action=fake[:,t] if wrong_credit else a[:,t]
            state=self.observe(state,observed_action,y[:,t])
        return torch.stack(preds,dim=1),state


class GRUPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.rec=nn.GRUCell(2,6)
        self.head=nn.Sequential(nn.Linear(7,16),nn.Tanh(),nn.Linear(16,1))

    def forward(self,d):
        a=d['executed_action']; y=d['real_observed_delta']
        b,t=a.shape
        h=a.new_zeros((b,6))
        output=[]
        for j in range(t):
            out=self.head(torch.cat((h,a[:,j,None]),dim=1))[:,0]
            output.append(out)
            # The recurrent learner reads only *completed* factual actions/outcomes.
            h=self.rec(torch.stack((a[:,j],y[:,j]),dim=-1),h)
        return torch.stack(output,dim=1)


def sliding_ridge(d,window=8):
    a,y=d['executed_action'],d['real_observed_delta']
    out=torch.zeros_like(y)
    for t in range(1,a.shape[1]):
        start=max(0,t-window)
        aa=a[:,start:t].double(); yy=y[:,start:t].double()
        mean_a=aa.mean(1); mean_y=yy.mean(1)
        den=((aa-mean_a[:,None])**2).sum(1)
        cov=((aa-mean_a[:,None])*(yy-mean_y[:,None])).sum(1)
        beta=torch.where(den>0,cov/den.clamp(min=1e-15),torch.zeros_like(den))
        offset=mean_y-beta*mean_a
        out[:,t]=(offset+beta*a[:,t].double()).float()
    return out


def score(pred,d):
    y=d['real_observed_delta']
    return float((pred[:,4:]-y[:,4:]).square().mean().detach())


def train(model,fit,val,indices,kind,wrong_credit=False):
    optimizer=torch.optim.Adam(model.parameters(),lr=.015)
    curve=[];st=time.perf_counter()
    for step,idx in enumerate(indices):
        d=allowed(fit,idx)
        model.train()
        pred=(model(d,wrong_credit=wrong_credit)[0] if kind=='fast' else model(d))
        loss=(pred[:,4:]-d['real_observed_delta'][:,4:]).square().mean()
        if not bool(torch.isfinite(loss)):raise ValueError('P63_NONFINITE_TRAIN_LOSS')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if (step+1)%CHECK_EVERY==0:
            model.eval()
            with torch.no_grad():
                vd=allowed(val)
                pv=(model(vd,wrong_credit=wrong_credit)[0] if kind=='fast' else model(vd))
                curve.append(score(pv,vd))
    plateau=abs(curve[-1]-curve[-2])/max(1e-15,curve[-2])<=.05
    return {'validation_MSE_every_40':curve,'validation_last_40_relative_change_le_5pct':plateau,
            'last_fit_MSE':float(loss.detach()),'fit_seconds':time.perf_counter()-st}


def cold_resume(model,d):
    facts=allowed(d,torch.arange(7))
    prefix={k:v[:,:16].clone() for k,v in facts.items()}
    suffix={k:v[:,16:].clone() for k,v in facts.items()}
    with torch.no_grad():
        p,fast=model(prefix)
        q,end=model(suffix,carry=fast)
        full,last=model(facts)
    buff=io.BytesIO()
    torch.save({'weights':model.state_dict(),'fast':fast,'suffix':suffix},buff)
    buff.seek(0)
    values=torch.load(buff,weights_only=True)
    clone=FastState(learned=model.learned)
    clone.load_state_dict(values['weights'])
    with torch.no_grad():
        resumed,resumed_final=clone(values['suffix'],carry=values['fast'])
    return (bool(torch.equal(torch.cat((p,q),1),full) and torch.equal(q,resumed)
                 and torch.equal(end,last) and torch.equal(end,resumed_final)),
            hashlib.sha256(buff.getvalue()).hexdigest())


def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    st=time.perf_counter()
    sources={s:source(s) for s in SPLITS}
    streams={k:v[0] for k,v in sources.items()}
    hashes={k:v[1] for k,v in sources.items()}
    gen=torch.Generator().manual_seed(SAMPLER_SEED)
    batches=torch.randint(512,(UPDATES,BATCH),generator=gen)
    sampler_sha=hashlib.sha256(batches.numpy().tobytes()).hexdigest()
    eval_facts=allowed(streams['ood'])
    controls={}
    with torch.no_grad():
        fixed=FastState(learned=False)
        fixed_pred,_=fixed(eval_facts)
        controls['fixed_fast_0p2']={'MSE':score(fixed_pred,eval_facts)}
        controls['sliding_OLS_8']={'MSE':score(sliding_ridge(eval_facts),eval_facts)}
        last=torch.cat((torch.zeros_like(eval_facts['real_observed_delta'][:,:1]),eval_facts['real_observed_delta'][:,:-1]),1)
        controls['last_actual_delta']={'MSE':score(last,eval_facts)}
    rows=[]
    for seed in SEEDS:
        arms={}
        for key,kind,wrong in [('fast','fast',False),('GRU','gru',False),('wrong_credit','fast',True)]:
            torch.manual_seed(seed)
            model=FastState(learned=True) if kind=='fast' else GRUPredictor()
            training=train(model,streams['fit'],streams['val'],batches,kind,wrong_credit=wrong)
            model.eval()
            with torch.no_grad():
                out=model(eval_facts,wrong_credit=wrong)[0] if kind=='fast' else model(eval_facts)
            outcome={'MSE':score(out,eval_facts),'params':sum(p.numel() for p in model.parameters()),
                     'train':training}
            if kind=='fast':
                exact,sha=cold_resume(model,streams['val'])
                outcome.update({'cold_exact':exact,'cold_sha256':sha,
                                'learning_rate':float(model.rates()[0].detach()),
                                'decay':float(model.rates()[1].detach())})
            arms[key]=outcome
            print('P63',seed,key,'ood_mse',outcome['MSE'],'sec',training['fit_seconds'],flush=True)
        ours=arms['fast']['MSE']
        best=min(arms['GRU']['MSE'],controls['sliding_OLS_8']['MSE'])
        gates={'10pct_better_than_best_GRU_or_OLS':ours<=.9*best,
               '10pct_better_than_fixed_fast':ours<=.9*controls['fixed_fast_0p2']['MSE'],
               '20pct_wrong_credit_MSE_increase':arms['wrong_credit']['MSE']>=1.2*ours,
               'cold_resume_exact':arms['fast']['cold_exact'],
               'all_trained_validation_plateau':all(arms[a]['train']['validation_last_40_relative_change_le_5pct'] for a in arms)}
        rows.append({'seed':seed,'arms':arms,'gates':gates,'all_pass':all(gates.values())})
    return {'schema':'ttt-hybrid.p63-persistent-factual-fast-weight-dev-result.v1',
            'freeze':'09425435616c1c9d02073e749ae5da8f5525bbc3',
            'source_sha256':hashes,'sampler_sha256':sampler_sha,'source':SPLITS,
            'seeds':SEEDS,'rows':rows,'controls':controls,
            'registered_all_seed_pass':all(row['all_pass'] for row in rows),
            'total_seconds':time.perf_counter()-st,'torch':torch.__version__,
            'python':platform.python_version(), 'external_science':'NOT_RUN',
            'original_model_modified':False, 'FORGE':'UNCHANGED',
            'mandatory_full_Windows_Linux_regression':'UNVERIFIED'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--receipt',type=Path,required=True)
    args=ap.parse_args()
    result=run()
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print('P63_REGISTERED_RESULT',json.dumps({'controls':result['controls'],
          'seeds':[{'seed':row['seed'],'gates':row['gates'],
                    'MSE':{k:v['MSE'] for k,v in row['arms'].items()}} for row in result['rows']],
          'all_pass':result['registered_all_seed_pass']}),flush=True)

if __name__=='__main__':main()
