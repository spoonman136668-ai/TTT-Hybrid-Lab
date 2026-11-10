"""P56 DEV: train the existing continuous Hybrid on completed published-equation physical steps.

Not official Gymnasium runtime: source is P54's independently transcribed equations.
All data here are authored source-controlled developmental fixtures, not external
scientific first-use. No hypothetical query outcome is supplied to training.
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
from model import Configuration, HybridPilot
from native_p54_mountaincar_factual_bridge import published_step

SPLITS = {'fit':(966101,768),'val':(966103,192),'ood':(966107,256)}
MODEL_SEEDS=(976101,976103,976107)
SAMPLE_SEED=976199
UPDATES=200
BATCH=64
CHOICES=np.asarray([-1.,0.,1.],dtype=np.float32)
FEATURES=7


def git_blob(path):
    x=Path(path).read_bytes()
    return hashlib.sha1(b'blob '+str(len(x)).encode()+b'\0'+x).hexdigest()


def source(which):
    if which not in SPLITS:
        raise ValueError('P56_UNREGISTERED_SOURCE_FIRST_USE')
    seed,n=SPLITS[which]
    rng=np.random.default_rng(seed)
    hist=[];current=[];observed_actions=[];actual_next=[];paired=[]
    rejected=0
    while len(hist)<n:
        if which=='ood':
            lo,hi=((-1.15,-.95) if rng.random()<.5 else (.10,.30))
            sensor=np.array([rng.uniform(lo,hi),rng.uniform(-.04,.04)],dtype=np.float32)
        else:
            sensor=np.array([rng.uniform(-.6,-.4),0.],dtype=np.float32)
        events=[]; terminated=False
        for step in range(8):
            action=float(rng.choice(CHOICES))
            actual=published_step(sensor,action)
            # Published actual completed physical step; never a speculated action.
            events.append(np.concatenate([sensor,actual.next_sensor-sensor,
                                          np.array([action,1.,0.],dtype=np.float32)]))
            sensor=actual.next_sensor
            if actual.terminated:
                terminated=True
                break
        if terminated:
            rejected+=1
            continue
        chosen=float(rng.choice(CHOICES))
        query=published_step(sensor,chosen)
        if query.terminated:
            rejected+=1
            continue
        hist.append(np.stack(events))
        current.append(sensor)
        observed_actions.append([chosen])
        actual_next.append(query.next_sensor)
        if which!='fit':
            # Only created for *scoring* and never passed to model inputs/trainer.
            paired.append(np.stack([published_step(sensor,float(a)).next_sensor for a in CHOICES]))
        if rejected>20000:
            raise RuntimeError('P56_SOURCE_REJECTION_BOUND_EXCEEDED')
    arrays=dict(history=np.asarray(hist,dtype=np.float32),sensor=np.asarray(current,dtype=np.float32),
         executed_query_action=np.asarray(observed_actions,dtype=np.float32),
         factual_next=np.asarray(actual_next,dtype=np.float32))
    if paired:
        arrays['scoring_only_potential_outcomes']=np.asarray(paired,dtype=np.float32)
    sha=hashlib.sha256(b''.join(a.tobytes() for a in arrays.values())).hexdigest()
    return {k:torch.from_numpy(v) for k,v in arrays.items()},sha,rejected


def form(history,sensor,actions,wrong_credit=False):
    b,k=actions.shape[:2]
    if history.ndim!=3 or history.shape!=(b,8,FEATURES) or sensor.shape!=(b,2) or actions.shape!=(b,k,1):
        raise ValueError('P56_BAD_COMPLETED_FACTUAL_HISTORY_OR_QUERY')
    if not bool(torch.isfinite(history).all() and torch.isfinite(sensor).all() and torch.isfinite(actions).all()):
        raise ValueError('P56_NONFINITE_ACTUAL_SOURCE')
    h=history.clone()
    if wrong_credit:
        h[:,:,4]=torch.roll(h[:,:,4],1,dims=1)
    probe=torch.zeros((b,k,FEATURES),dtype=history.dtype,device=history.device)
    probe[:,:,:2]=sensor[:,None,:]
    probe[:,:,4]=actions[:,:,0]
    probe[:,:,5]=1.0
    probe[:,:,6]=1.0
    return torch.cat((h[:,None,:,:].expand(-1,k,-1,-1),probe[:,:,None,:]),dim=2)


class Hybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.pilot=HybridPilot(FEATURES,2,Configuration(width=16,heads=2))
    def forward(self,d,actions,wrong_credit=False):
        z=form(d['history'],d['sensor'],actions,wrong_credit)
        b,k=z.shape[:2]
        return d['sensor'][:,None,:]+self.pilot(z.reshape(b*k,9,FEATURES))[:,-1,:].reshape(b,k,2)


class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru=nn.GRU(FEATURES,32,batch_first=True)
        self.head=nn.Linear(32,2)
    def forward(self,d,actions,wrong_credit=False):
        z=form(d['history'],d['sensor'],actions,wrong_credit)
        b,k=z.shape[:2]
        _,h=self.gru(z.reshape(b*k,9,FEATURES))
        return d['sensor'][:,None,:]+self.head(h[-1]).reshape(b,k,2)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(9*FEATURES,64),nn.GELU(),nn.Linear(64,32),nn.GELU(),nn.Linear(32,2))
    def forward(self,d,actions,wrong_credit=False):
        z=form(d['history'],d['sensor'],actions,wrong_credit)
        b,k=z.shape[:2]
        return d['sensor'][:,None,:]+self.net(z.reshape(b*k,9*FEATURES)).reshape(b,k,2)


def take(d,ids):
    return {k:v[ids] for k,v in d.items() if k!='scoring_only_potential_outcomes'}


def count(model):return sum(p.numel() for p in model.parameters())


def fit(model,train,val,indices,wrong_credit=False):
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002)
    wall=time.perf_counter();curve=[]
    for j in range(UPDATES):
        model.train();d=take(train,indices[j]);actual=d['executed_query_action'][:,None,:]
        pred=model(d,actual,wrong_credit)[:,0,:]
        loss=(pred-d['factual_next']).square().mean()
        if not bool(torch.isfinite(loss).all()):raise ValueError('P56_NONFINITE_OPTIMIZER_LOSS')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if (j+1)%40==0:
            model.eval()
            with torch.no_grad():
                v=model(val,val['executed_query_action'][:,None,:],wrong_credit)[:,0,:]
                curve.append(float((v-val['factual_next']).square().mean()))
    return {'training_seconds':time.perf_counter()-wall,'validation_curve':curve,'plateau':abs(curve[-1]-curve[-2])/max(curve[-2],1e-12)<=.01}


def metrics(pred,y):
    d=pred-y
    mse=float(d.square().mean())
    eff=(pred[:,2,:]-pred[:,0,:])*0.5
    eff_true=(y[:,2,:]-y[:,0,:])*0.5
    effect_mae=float((eff-eff_true).abs().mean())
    return {'mse':mse,'effect_mae':effect_mae,'neutral_mse':float(d[:,1,:].square().mean())}


def evaluate(model,d,wrong_credit=False):
    actions=torch.tensor([[-1.],[0.],[1.]],dtype=torch.float32)[None,:,:].expand(len(d['sensor']),-1,-1)
    model.eval()
    with torch.no_grad():pred=model(d,actions,wrong_credit)
    return metrics(pred,d['scoring_only_potential_outcomes'])


def cold(model,d,wrong_credit=False):
    model.eval();n=min(3,len(d['sensor']));sub=take(d,torch.arange(n))
    candidates=torch.tensor([[-1.],[0.],[1.]])[None,:,:].expand(n,-1,-1)
    with torch.no_grad():before=model(sub,candidates,wrong_credit)
    buf=io.BytesIO();torch.save({'weights':model.state_dict(),'history':sub['history'],'sensor':sub['sensor']},buf);buf.seek(0)
    obj=torch.load(buf,weights_only=True)
    restored=type(model)().eval();restored.load_state_dict(obj['weights'])
    inputs=dict(sub,history=obj['history'],sensor=obj['sensor'])
    with torch.no_grad():after=restored(inputs,candidates,wrong_credit)
    return bool(torch.equal(before,after) and torch.equal(sub['history'],obj['history']))


def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    began=time.perf_counter()
    sources={k:source(k) for k in SPLITS}
    d={k:v[0] for k,v in sources.items()};hashes={k:v[1] for k,v in sources.items()}
    rejected={k:v[2] for k,v in sources.items()}
    generator=torch.Generator().manual_seed(SAMPLE_SEED)
    indices=torch.randint(0,len(d['fit']['sensor']),(UPDATES,BATCH),generator=generator)
    indexhash=hashlib.sha256(indices.numpy().tobytes()).hexdigest()
    truth=d['ood']['scoring_only_potential_outcomes']
    last=d['ood']['sensor'][:,None,:]+d['ood']['history'][:,-1,None,2:4]
    classical=metrics(last.expand(-1,3,-1),truth)
    known=metrics(truth,truth)
    arms={'P55':Hybrid,'GRU':GRU,'MLP':MLP,'wrong_credit_P55':Hybrid}
    param={n:count(c()) for n,c in arms.items()}
    rows=[]
    for seed in MODEL_SEEDS:
        results={}
        for name,cls in arms.items():
            torch.manual_seed(seed);model=cls()
            wrong=name=='wrong_credit_P55'
            f=fit(model,d['fit'],d['val'],indices,wrong)
            e=evaluate(model,d['ood'],wrong)
            results[name]={'train':f,'metrics':e,'params':count(model)}
            if name=='P55':results[name]['cold_exact']=cold(model,d['ood'])
            print('seed',seed,'arm',name,'MSE',round(e['mse'],8),'effect_MAE',round(e['effect_mae'],8),flush=True)
        p=results['P55']['metrics']; best=min(classical['mse'],results['GRU']['metrics']['mse'],results['MLP']['metrics']['mse'])
        best_effect=min(results['GRU']['metrics']['effect_mae'],results['MLP']['metrics']['effect_mae'])
        gates={'better_than_strongest_MSE_5pct':p['mse']<=.95*best,
              'better_than_neural_effect_MAE_5pct':p['effect_mae']<=.95*best_effect,
              'wrong_credit_MSE_20pct':p['mse']<=.80*results['wrong_credit_P55']['metrics']['mse'],
              'cold_exact':results['P55']['cold_exact'],
              'all_neural_plateau':all(v['train']['plateau'] for v in results.values())}
        rows.append({'seed':seed,'arms':results,'gates':gates,'all_gates':all(gates.values())})
    return {'schema':'ttt-hybrid.p56.published-mountaincar-factual-learning-dev-result.v1',
            'scope':'TRANSCRIBED_PUBLISHED_SIMULATOR_DEV_NOT_OFFICIAL_GYMNASIUM_RUNTIME_OR_INDEPENDENT_SCIENCE',
            'seeds':SPLITS,'source_SHA256':hashes,'reject_terminal_counts':rejected,'index_sha256':indexhash,
            'model_seeds':MODEL_SEEDS,'params':param,'updates_per_arm':UPDATES,
            'classical_last_delta':classical,'known_deterministic_mechanics':known,'cohorts':rows,
            'all_gates':all(row['all_gates'] for row in rows),'wall_seconds':time.perf_counter()-began,
            'python':platform.python_version(),'torch':torch.__version__,
            'official_source_runtime_parity':'NOT_RUN','source_owner_scientific_acceptance':'NOT_RUN',
            'Windows_Linux_full_regression':'NOT_RUN','FORGE':'UNCHANGED'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=run();a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'classical':result['classical_last_delta'],'known':result['known_deterministic_mechanics'],
       'seeds':[{'seed':r['seed'],'gates':r['gates'],'P55':r['arms']['P55']['metrics'],
       'GRU':r['arms']['GRU']['metrics'],'MLP':r['arms']['MLP']['metrics']} for r in result['cohorts']],
       'all_gates':result['all_gates'],'wall_seconds':result['wall_seconds']},sort_keys=True,indent=2))

if __name__=='__main__':main()
