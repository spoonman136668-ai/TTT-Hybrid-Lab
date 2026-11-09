"""P44: separate authored DEV comparison of the EXACT P43 trainable GRUCell.

No original scientific data, external source first use, runner, or promotion.
Freeze: experiments/native_p44_p43_original_factual_transfer_frozen.json.
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
import torch
from torch import nn
from native_trainable_causal_belief import NativeTrainableCausalBelief, NativeCausalBeliefState

SEEDS=(9203,9217,9221)
SOURCE_SEEDS={'fit':730001,'validation':730101,'unfamiliar_dev':730201}
COUNTS={'fit':2048,'validation':256,'unfamiliar_dev':512}
SAMPLES_SEED=731001
HISTORY=8
BATCH=128
UPDATES=600
DT=1.0/8


def source(cohort):
    """Separate realized actual actions, withheld hypothetical outcomes for scoring."""
    if cohort not in SOURCE_SEEDS:
        raise ValueError('P44_NONREGISTERED_SOURCE_COHORT')
    n=COUNTS[cohort]
    g=np.random.default_rng(SOURCE_SEEDS[cohort]); od=cohort=='unfamiliar_dev'
    u=lambda low,high:g.uniform(low,high,(n,1))
    b0=u(-.1,.1);b1=u(-.25,.25);b2=u(-.15,.15);amp=u(-.15,.15)
    phase=u(-math.pi,math.pi);f=u(*( (1.25,1.8) if od else (.45,1.) ))
    beta=g.choice(np.array([-1.,1.]),(n,1))*u(.2,.45)
    switch=np.zeros(n,dtype=bool)
    if od:switch[:n//4]=True
    actions=g.choice(np.array([-1.,1.]),(n,HISTORY))
    query_action=g.choice(np.array([-1.,1.]),(n,1))
    starts=u(-.3,.3)
    history=np.empty((n,HISTORY,4),dtype=np.float64)
    x=starts.copy()
    for i in range(HISTORY):
        t=i*DT
        physical=beta*np.where(switch[:,None] & (i>=4),-1.,1.)
        nuisance=b0+b1*t+b2*t*t+amp*np.sin(2*np.pi*f*t+phase)
        delta=nuisance+physical*actions[:,i:i+1]+g.normal(0,.01,(n,1))
        history[:,i,:]=np.concatenate((x,delta,actions[:,i:i+1],np.full((n,1),np.log1p(DT))),axis=1)
        x=x+delta
    t=1.0
    current_effect=beta*np.where(switch[:,None],-1.,1.)
    nuisance=b0+b1*t+b2*t*t+amp*np.sin(2*np.pi*f*t+phase)
    factual_y=x+nuisance+current_effect*query_action+g.normal(0,.01,(n,1))
    eval_actions=np.array([[-1.,1.]])
    scorer_both=x+nuisance+current_effect*eval_actions+g.normal(0,.01,(n,2))
    tensors={key:torch.tensor(value,dtype=torch.float32) for key,value in
             [('history',history),('query_sensor',x),('query_action',query_action),
              ('factual_query_y',factual_y),('evaluator_both',scorer_both),('effect',current_effect)]}
    tensors['switched']=torch.tensor(switch)
    tensors['source_sha256']=hashlib.sha256(b''.join(tensors[k].numpy().tobytes() for k in
        ('history','query_sensor','query_action','factual_query_y','evaluator_both','effect','switched'))).hexdigest()
    return tensors


class GRUConventional(nn.Module):
    """Same recurrent factual information, non-factorized query-action head."""
    def __init__(self,width=16):
        super().__init__()
        self.cell=nn.GRUCell(4,width)
        self.common=nn.Sequential(nn.Linear(width+3,width),nn.Tanh())
        self.head=nn.Linear(width,1)

    def forward(self,cohort,candidates,wrong_credit=False):
        hist=cohort['history']
        prior=hist.new_zeros((hist.shape[0],self.cell.hidden_size))
        for i in range(HISTORY):
            prior=self.cell(hist[:,i,:],prior)
        x=cohort['query_sensor']; log_dt=x.new_full(x.shape,math.log1p(DT))
        if candidates.ndim==2:candidates=candidates[:,None,:]
        scores=[]
        for j in range(candidates.shape[1]):
            state=self.common(torch.cat((prior,x,log_dt,candidates[:,j,:]),dim=-1))
            scores.append(x+self.head(state))
        return torch.stack(scores,dim=1)


class MLPConventional(nn.Module):
    """Same completed factual data flattened, query action separately supplied."""
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(HISTORY*4+3,24),nn.GELU(),
                               nn.Linear(24,16),nn.GELU(),nn.Linear(16,1))

    def forward(self,cohort,candidates,wrong_credit=False):
        h=cohort['history']; x=cohort['query_sensor']; n=h.shape[0]
        log_dt=x.new_full(x.shape,math.log1p(DT))
        if candidates.ndim==2:candidates=candidates[:,None,:]
        out=[]
        for j in range(candidates.shape[1]):
            inp=torch.cat((h.flatten(1),x,log_dt,candidates[:,j,:]),dim=-1)
            out.append(x+self.net(inp))
        return torch.stack(out,dim=1)


class OriginalP43(nn.Module):
    """Wraps the EXACT original pinned NativeTrainableCausalBelief public API."""
    def __init__(self):
        super().__init__()
        self.native=NativeTrainableCausalBelief(sensor_dim=1,width=16,min_facts=2)

    def factual_state(self,cohort,wrong_credit=False):
        h=cohort['history']; n=h.shape[0]
        state=self.native.initial_state(h[:,0,:1])
        for i in range(HISTORY):
            physical_action=h[:,(i-1)%HISTORY,2:3] if wrong_credit else h[:,i,2:3]
            actual_sensor=h[:,i,:1]
            actual_outcome=actual_sensor+h[:,i,1:2]
            dt=actual_sensor.new_full((n,1),DT)
            state=self.native.observe(actual_sensor,physical_action,dt,actual_outcome,state)
        return state

    def forward(self,cohort,candidates,wrong_credit=False):
        state=self.factual_state(cohort,wrong_credit=wrong_credit)
        if candidates.ndim==2:candidates=candidates[:,None,:]
        x=cohort['query_sensor'];dt=x.new_full(x.shape,DT)
        out=self.native.preview(x,candidates,dt,state)
        assert bool(out.factual_ready.all())
        return out.predictions


def params(model):return sum(p.numel() for p in model.parameters())


def batch_data(cohort,indices):
    return {k:v[indices] for k,v in cohort.items() if isinstance(v,torch.Tensor)}


def train(arm,cohorts,indices,wrong_credit=False):
    opt=torch.optim.AdamW(arm.parameters(),lr=.003)
    curve=[];start=time.perf_counter()
    for step in range(UPDATES):
        arm.train();sub=batch_data(cohorts['fit'],indices[step]);a=sub['query_action']
        out=arm(sub,a,wrong_credit=wrong_credit)[:,0,:]
        loss=(out-sub['factual_query_y']).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(arm.parameters(),1.0)
        opt.step()
        if (step+1)%100==0:
            arm.eval()
            with torch.no_grad():
                va=cohorts['validation'];p=arm(va,va['query_action'],wrong_credit=wrong_credit)[:,0,:]
                curve.append(float((p-va['factual_query_y']).square().mean()))
    return {'train_wall_seconds':time.perf_counter()-start,'updates':UPDATES,'val_curve':curve,
            'last_relative_improvement':(curve[-2]-curve[-1])/max(curve[-2],1e-9),
            'last_absolute_change':curve[-2]-curve[-1]}


def evaluate(arm,cohort,wrong_credit=False):
    arm.eval(); n=cohort['history'].shape[0]
    a=torch.tensor([[[-1.],[1.]]],dtype=torch.float32).expand(n,-1,-1)
    start=time.perf_counter()
    with torch.no_grad():pred=arm(cohort,a,wrong_credit=wrong_credit)[:,:,0]
    truth=cohort['evaluator_both']; mae=(.5*(pred[:,1]-pred[:,0])-cohort['effect'][:,0]).abs()
    opt=[]
    for direction in (-.35,.35):
        target=cohort['query_sensor'][:,0,None]+direction
        opt.append(((pred-target).square().argmin(1)==(truth-target).square().argmin(1)).float())
    accuracy=.5*(opt[0]+opt[1]); error=(pred-truth).square().mean(1)
    result={'mse':float(error.mean()),'effect_mae':float(mae.mean()),
            'goal_optimal':float(accuracy.mean()),'inference_seconds':time.perf_counter()-start}
    sw=cohort['switched']
    if bool(sw.any()):
        for label,mask in [('switch',sw),('no_switch',~sw)]:
            result[label]={'mse':float(error[mask].mean()),'effect_mae':float(mae[mask].mean()),
                           'goal_optimal':float(accuracy[mask].mean()),'n':int(mask.sum())}
    return result


def ridge(cohort):
    h=cohort['history'].double();n=h.shape[0]
    t=torch.arange(HISTORY,dtype=torch.float64)[None,:].expand(n,-1)*DT
    x=torch.stack((torch.ones_like(t),t,t*t,h[:,:,2]),dim=-1)
    y=h[:,:,1,None]
    rank=torch.linalg.matrix_rank(x)
    mat=x.transpose(1,2)@x+0.001*torch.eye(4,dtype=torch.float64).expand(n,-1,-1)
    solve=torch.linalg.solve(mat,x.transpose(1,2)@y).squeeze(-1)
    known=rank==4; solve[~known,3]=0
    drift=solve[:,:3].sum(1)
    sensor=cohort['query_sensor'][:,0].double()
    pred=torch.stack((sensor+drift-solve[:,3],sensor+drift+solve[:,3]),dim=1).float()
    truth=cohort['evaluator_both']; error=(pred-truth).square().mean(1)
    ma=(solve[:,3].float()-cohort['effect'][:,0]).abs()
    o=[]
    for d in (-.35,.35):
        target=cohort['query_sensor'][:,0,None]+d
        o.append(((pred-target).square().argmin(1)==(truth-target).square().argmin(1)).float())
    out={'mse':float(error.mean()),'effect_mae':float(ma.mean()),
         'goal_optimal':float((o[0]+o[1]).mean()*.5),'rank_full':int(known.sum()),'n':n}
    sw=cohort['switched']
    for label,mask in [('switch',sw),('no_switch',~sw)]:
        if bool(mask.any()):out[label]={'mse':float(error[mask].mean()),'effect_mae':float(ma[mask].mean()),
                                      'goal_optimal':float((o[0][mask]+o[1][mask]).mean()*.5),'n':int(mask.sum())}
    return out


def cold(model,cohort):
    """Bit-exact fresh instance and factual-state tensor reload for P43 original."""
    import pickle
    sample=batch_data(cohort,torch.arange(3))
    model.eval()
    with torch.no_grad():
        state=model.factual_state(sample)
        initial=model.native.preview(sample['query_sensor'],torch.tensor([[[-1.],[1.]]]).expand(3,-1,-1),
                                     torch.full((3,1),DT),state).predictions
    obj={'weights':model.state_dict(),'state':{'hidden':state.hidden,'observations':state.observations,'elapsed':state.elapsed}}
    buff=io.BytesIO();torch.save(obj,buff);buff.seek(0)
    restored=torch.load(buff,weights_only=True)
    fresh=OriginalP43().eval();fresh.load_state_dict(restored['weights'])
    s=NativeCausalBeliefState(**restored['state'])
    with torch.no_grad():
        replay=fresh.native.preview(sample['query_sensor'],torch.tensor([[[-1.],[1.]]]).expand(3,-1,-1),
                                    torch.full((3,1),DT),s).predictions
    return bool(torch.equal(initial,replay)),hashlib.sha256(buff.getvalue()).hexdigest()


def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    started=time.perf_counter()
    cohorts={name:source(name) for name in SOURCE_SEEDS}
    index=torch.randint(0,COUNTS['fit'],(UPDATES,BATCH),generator=torch.Generator().manual_seed(SAMPLES_SEED))
    index_sha=hashlib.sha256(index.numpy().tobytes()).hexdigest()
    families={'native_p43':OriginalP43,'gru':GRUConventional,'mlp':MLPConventional,'wrong_credit':OriginalP43}
    ps={name:params(f()) for name,f in families.items()}
    ridge_out=ridge(cohorts['unfamiliar_dev']);result=[]
    for seed in SEEDS:
        arms={}
        for name,factory in families.items():
            torch.manual_seed(seed)
            model=factory()
            wrong=name=='wrong_credit'
            tr=train(model,cohorts,index,wrong_credit=wrong)
            arms[name]={'fit':tr,'val':evaluate(model,cohorts['validation'],wrong_credit=wrong),
                        'ood':evaluate(model,cohorts['unfamiliar_dev'],wrong_credit=wrong)}
            if name=='native_p43':
                ok,digest=cold(model,cohorts['unfamiliar_dev'])
                arms[name]['cold_exact']=ok;arms[name]['cold_sha256']=digest
        p=arms['native_p43']['ood'];comp=[arms[k]['ood']['mse'] for k in ('gru','mlp')]
        passed={'ood_mse_beats_best_trainable_or_ridge_by_5pct':p['mse']<=.95*min(comp+[ridge_out['mse']]),
                'switch_goal_improvement_over_gru_3pp':p['switch']['goal_optimal']>=arms['gru']['ood']['switch']['goal_optimal']+.03,
                'wrong_credit_mse_margin_20pct':p['mse']<=.8*arms['wrong_credit']['ood']['mse'],
                'cold_exact':arms['native_p43']['cold_exact']}
        result.append({'seed':seed,'arms':arms,'gates':passed})
    return {'schema':'ttt-hybrid.native-p44.p43-grucell-dev-result.v1',
            'scope':'AUTHORED_NO_EXTERNAL_SOURCE_DEV_ONLY','source_sha256':{k:v['source_sha256'] for k,v in cohorts.items()},
            'shared_index_sha256':index_sha,'model_params':ps,'optimizer_updates_each':UPDATES,
            'source_seeds':SOURCE_SEEDS,'model_seeds':SEEDS,'ridge':ridge_out,
            'records':result,'all_three_seed_gates':all(all(s['gates'].values()) for s in result),
            'wall_seconds':time.perf_counter()-started,'science_first_use':'NOT_RUN',
            'full_windows_wsl_final_regression':'NOT_RUN','FORGE_integration':'NOT_ATTEMPTED'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);args=p.parse_args()
    r=run();args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(r,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'params':r['model_params'],'source_sha256':r['source_sha256'],
       'index_sha':r['shared_index_sha256'],'ridge':r['ridge'],
       'seed_gates':{x['seed']:x['gates'] for x in r['records']},
       'seed_MSE':{x['seed']:{name:v['ood']['mse'] for name,v in x['arms'].items()} for x in r['records']},
       'elapsed':r['wall_seconds']},indent=2))

if __name__=='__main__':main()
