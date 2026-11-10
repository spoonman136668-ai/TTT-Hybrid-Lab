"""P51 frozen DEV: original P50 factual Hybrid vs trained conventional controls.

Only *completed* actions/outcomes are inputs; one factual query action provides
training supervision. Hidden modes and evaluator-only paired alternatives are
never used for gradient updates. This is authored development data, not original
owner-qualified scientific first-use, NOR a proof of causal generalization.
"""
from __future__ import annotations
import argparse, hashlib, io, json, math, platform, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from model import Configuration
from native_factual_hybrid_stream import NativeFactualHybridStream, FactualStreamState

SEEDS = {"fit":811101,"validation":811103,"unfamiliar_dev":811107}
N = {"fit":2048,"validation":256,"unfamiliar_dev":512}
INIT = (831101,831103,831107)
INDEX_SEED=831199
HISTORY=8
BATCH=64
UPDATES=160
DT=0.125


def source(name):
    if name not in SEEDS:
        raise ValueError('P51_UNREGISTERED_DEV_SOURCE')
    n=N[name]; g=np.random.default_rng(SEEDS[name]); ood=name=='unfamiliar_dev'
    u=lambda lo,hi:g.uniform(lo,hi,(n,1))
    x=u(-.3,.3)
    b0,b1,b2=u(-.1,.1),u(-.25,.25),u(-.15,.15)
    amp=u(-.15,.15);phase=u(-math.pi,math.pi)
    freq=u(1.2,1.8) if ood else u(.35,.8)
    beta=g.choice(np.array([-1.,1.]),(n,1))*u(.2,.45)
    switch=np.zeros((n,1),dtype=bool)
    if ood:
        switch[:n//4]=True
    a_hist=g.choice(np.array([-1.,1.]),(n,HISTORY))
    history=np.empty((n,HISTORY,5),dtype=np.float32)
    dt=np.full((n,1),DT,dtype=np.float64)
    for i in range(HISTORY):
        t=i*DT
        effect=beta*np.where(switch & (i>=4),-1.,1.)
        nuisance=b0+b1*t+b2*t*t+amp*np.sin(2*math.pi*freq*t+phase)
        a=a_hist[:,i:i+1]
        delta=nuisance+effect*a+g.normal(0,.01,(n,1))
        history[:,i,:]=np.concatenate((x,delta,a,dt,np.zeros((n,1))),axis=1)
        x=x+delta
    query_a=g.choice(np.array([-1.,1.]),(n,1))
    query_t=1.
    nuisance=b0+b1*query_t+b2*query_t**2+amp*np.sin(2*math.pi*freq*query_t+phase)
    effect=beta*np.where(switch,-1.,1.)
    factual_y=x+nuisance+effect*query_a+g.normal(0,.01,(n,1))
    # Evaluator-only alternatives, never passed to FIT model or optimizer.
    paired=None
    if name!='fit':
        paired=x+nuisance+effect*np.array([[-1.,1.]])+g.normal(0,.01,(n,2))
    raw={"history":history,"sensor":x.astype(np.float32),"query_a":query_a.astype(np.float32),
         "factual_y":factual_y.astype(np.float32),"effect":effect.astype(np.float32),
         "switch":switch.reshape(n),"paired":paired.astype(np.float32) if paired is not None else None}
    hashparts=[raw[k].tobytes() for k in ("history","sensor","query_a","factual_y","effect","switch")]
    if paired is not None:hashparts.append(raw['paired'].tobytes())
    digest=hashlib.sha256(b''.join(hashparts)).hexdigest()
    data={k:torch.tensor(v) for k,v in raw.items() if v is not None}
    return data,digest


def query_features(sensor, candidates):
    if candidates.ndim==2:
        candidates=candidates[:,None,:]
    b,k,_=candidates.shape
    z=torch.zeros((b,k,1),dtype=sensor.dtype,device=sensor.device)
    x=sensor[:,None,:].expand(-1,k,-1)
    dt=torch.full_like(z,DT)
    return torch.cat((x,z,candidates,dt,torch.ones_like(z)),dim=-1)


def history_with_credit(data,wrong_credit=False):
    h=data['history']
    if wrong_credit:
        h=h.clone()
        h[:,:,2]=torch.roll(h[:,:,2],1,dims=1)
    return h


class P50Original(nn.Module):
    def __init__(self):
        super().__init__()
        self.native=NativeFactualHybridStream(1,capacity=8,
            cfg=Configuration(width=16,heads=2),min_facts=3)

    def forward(self,data,candidates,wrong_credit=False):
        h=history_with_credit(data,wrong_credit)
        q=query_features(data['sensor'],candidates)
        b,k=q.shape[:2]
        seq=torch.cat((h[:,None,:,:].expand(-1,k,-1,-1),q[:,:,None,:]),dim=2)
        seq=seq.reshape(b*k,HISTORY+1,5)
        # Exact original P50 model branch: parallel episodes are independent,
        # no source aliases, future outcome nor hidden effect supplied.
        y=self.native.pilot(seq,ttt_update=True)[:,-1,:].reshape(b,k,1)
        return data['sensor'][:,None,:]+y


class GRUControl(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru=nn.GRU(5,33,batch_first=True)
        self.readout=nn.Sequential(nn.Linear(33,16),nn.GELU(),nn.Linear(16,1))

    def forward(self,data,candidates,wrong_credit=False):
        h=history_with_credit(data,wrong_credit)
        q=query_features(data['sensor'],candidates)
        b,k=q.shape[:2]
        seq=torch.cat((h[:,None,:,:].expand(-1,k,-1,-1),q[:,:,None,:]),dim=2)
        _,state=self.gru(seq.reshape(b*k,HISTORY+1,5))
        return data['sensor'][:,None,:]+self.readout(state[-1]).reshape(b,k,1)


class MLPControl(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(45,60),nn.GELU(),nn.Linear(60,30),
                               nn.GELU(),nn.Linear(30,1))

    def forward(self,data,candidates,wrong_credit=False):
        h=history_with_credit(data,wrong_credit)
        q=query_features(data['sensor'],candidates)
        b,k=q.shape[:2]
        flat=h.flatten(1)[:,None,:].expand(-1,k,-1)
        z=torch.cat((flat,q),dim=-1)
        return data['sensor'][:,None,:]+self.net(z)


def params(m):
    return sum(p.numel() for p in m.parameters())


def take(data,index):
    return {k:v[index] for k,v in data.items()}


def train(model, cohorts, indices, *, wrong_credit=False):
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
    vals=[];start=time.perf_counter()
    for step in range(UPDATES):
        model.train();d=take(cohorts['fit'],indices[step])
        pred=model(d,d['query_a'],wrong_credit=wrong_credit)[:,0,:]
        loss=(pred-d['factual_y']).square().mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError('P51_NONFINITE_TRAIN_LOSS')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if (step+1)%40==0:
            model.eval()
            with torch.no_grad():
                validation=cohorts['validation']
                va=model(validation,validation['query_a'],wrong_credit=wrong_credit)[:,0,:]
                vals.append(float(((va-validation['factual_y'])**2).mean()))
    return dict(wall_seconds=time.perf_counter()-start,
                last_loss=float(loss.detach()),validation_curve=vals,
                plateau_last40=abs(vals[-2]-vals[-1])/max(vals[-2],1e-9)<=.01)


def evaluate(model,data,*,wrong_credit=False):
    model.eval();n=len(data['history'])
    action=torch.tensor([[[-1.],[1.]]],dtype=torch.float32).expand(n,-1,-1)
    start=time.perf_counter()
    with torch.no_grad():
        out=[]
        for indices in torch.arange(n).split(128):
            out.append(model(take(data,indices),action[indices],wrong_credit=wrong_credit)[:,:,0])
        pred=torch.cat(out,dim=0)
    return scoring(pred,data,time.perf_counter()-start)


def scoring(pred,data,seconds=0.):
    truth=data['paired'];target=data['sensor']
    actual=data['effect'][:,0]
    causal=(pred[:,1]-pred[:,0])*.5
    error=(pred-truth).square().mean(-1)
    effect_err=(causal-actual).abs()
    goals=[]
    for d in (-.3,.3):
        goal=target[:,0,None]+d
        selected=(pred-goal).square().argmin(1)
        oracle=(truth-goal).square().argmin(1)
        goals.append((selected==oracle).float())
    goal_score=(goals[0]+goals[1])*.5
    masks={'all':torch.ones(len(pred),dtype=torch.bool),
           'switched':data['switch'].bool(),
           'unswitched':~data['switch'].bool()}
    metrics={}
    for name,mask in masks.items():
        if mask.sum().item()==0:continue
        metrics[name]=dict(n=int(mask.sum()),mse=float(error[mask].mean()),
            effect_mae=float(effect_err[mask].mean()),goal_optimal=float(goal_score[mask].mean()))
    return dict(metrics=metrics,inference_seconds=seconds)


def polynomial_ridge(data):
    # Convex fit using eight COMPLETED factual transitions, never query outcome.
    h=data['history'].double();n=h.shape[0]
    t=torch.arange(HISTORY,dtype=torch.float64)[None,:].expand(n,-1)*DT
    x=torch.stack((torch.ones_like(t),t,t*t,h[:,:,2]),dim=-1)
    y=h[:,:,1,None]
    eye=torch.eye(4,dtype=torch.float64)[None,:,:].expand(n,-1,-1)
    fit=torch.linalg.solve(x.transpose(1,2)@x+.001*eye,x.transpose(1,2)@y).squeeze(-1)
    nuisance=fit[:,:3].sum(-1)
    sensor=data['sensor'][:,0].double()
    both=torch.stack((sensor+nuisance-fit[:,3],sensor+nuisance+fit[:,3]),dim=-1).float()
    return scoring(both,data)


def last_effect(data):
    h=data['history'];x=data['sensor'][:,0]
    effect=h[:,-1,1]/h[:,-1,2]
    pred=torch.stack((x-effect,x+effect),dim=-1)
    return scoring(pred,data)


def cold_check(model,data):
    sample=take(data,torch.arange(3))
    model.eval()
    state=model.native.initial_state(sample['history'][:,0,:1])
    for i in range(HISTORY):
        h=sample['history'][:,i,:]
        state=model.native.observe(h[:,0:1],h[:,2:3],h[:,3:4],
                                   h[:,0:1]+h[:,1:2],state)
    actions=torch.tensor([[[-1.],[1.]]]).expand(3,-1,-1)
    dt=torch.full((3,1),DT)
    with torch.no_grad():
        prev=model.native.preview(sample['sensor'],actions,dt,state)
    buf=io.BytesIO();torch.save({'weights':model.state_dict(),
        'state':{key:getattr(state,key) for key in FactualStreamState.__dataclass_fields__}},buf)
    buf.seek(0);payload=torch.load(buf,weights_only=True)
    fresh=P50Original().eval();fresh.load_state_dict(payload['weights'])
    rs=FactualStreamState(**payload['state'])
    with torch.no_grad():
        replay=fresh.native.preview(sample['sensor'],actions,dt,rs)
    return (torch.equal(prev.predictions,replay.predictions)
            and torch.equal(prev.identified,replay.identified)
            and torch.equal(prev.facts_used,replay.facts_used))


def compare_direct_api(model,data):
    # Confirms batched training path is identical to original P50 preview for
    # a single fully factual episode, so benchmark does not swap architecture.
    one=take(data,torch.arange(1));h=one['history'];state=model.native.initial_state(h[:,0,:1])
    for t in range(HISTORY):
        state=model.native.observe(h[:,t,0:1],h[:,t,2:3],h[:,t,3:4],
                                   h[:,t,0:1]+h[:,t,1:2],state)
    action=torch.tensor([[[-1.],[1.]]]);dt=torch.full((1,1),DT)
    model.eval()
    with torch.no_grad():
        v=model.native.preview(one['sensor'],action,dt,state)
        w=model(one,action)
    return bool(torch.allclose(v.predictions,w,rtol=0,atol=1e-6))


def run():
    started=time.perf_counter();torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cohorts={};source_hash={}
    for name in SEEDS:
        cohorts[name],source_hash[name]=source(name)
    indices=torch.randint(0,N['fit'],(UPDATES,BATCH),
                           generator=torch.Generator().manual_seed(INDEX_SEED))
    sample_hash=hashlib.sha256(indices.numpy().tobytes()).hexdigest()
    arms={'p50':P50Original,'gru':GRUControl,'mlp':MLPControl,'wrong_credit_p50':P50Original}
    param_count={key:params(f()) for key,f in arms.items()}
    ridge=polynomial_ridge(cohorts['unfamiliar_dev'])
    last=last_effect(cohorts['unfamiliar_dev'])
    records=[]
    for seed in INIT:
        evidence={}
        for name,factory in arms.items():
            torch.manual_seed(seed);arm=factory()
            wrong=name=='wrong_credit_p50'
            tr=train(arm,cohorts,indices,wrong_credit=wrong)
            scored=evaluate(arm,cohorts['unfamiliar_dev'],wrong_credit=wrong)
            val=evaluate(arm,cohorts['validation'],wrong_credit=wrong)
            evidence[name]=dict(train=tr,unfamiliar=scored,validation=val)
            if name=='p50':
                evidence[name]['cold_exact']=cold_check(arm,cohorts['unfamiliar_dev'])
                evidence[name]['direct_api_exact']=compare_direct_api(arm,cohorts['unfamiliar_dev'])
        p=evidence['p50']['unfamiliar']['metrics']['all']
        comparator=min(evidence[name]['unfamiliar']['metrics']['all']['mse']
                       for name in ('gru','mlp'))
        comparator=min(comparator,ridge['metrics']['all']['mse'])
        mae_base=min([ridge['metrics']['all']['effect_mae']]+[
            evidence[name]['unfamiliar']['metrics']['all']['effect_mae']
            for name in ('gru','mlp')])
        negative=evidence['wrong_credit_p50']['unfamiliar']['metrics']['all']['mse']
        gates=dict(mse_5pct_strongest=p['mse']<=.95*comparator,
                   effect_mae_5pct_strongest=p['effect_mae']<=.95*mae_base,
                   wrong_credit_20pct=p['mse']<=.8*negative,
                   cold_exact=evidence['p50']['cold_exact'],
                   numerically_equivalent_P50_API_1e_6=evidence['p50']['direct_api_exact'],
                   all_neural_plateau=all(x['train']['plateau_last40'] for x in evidence.values()))
        records.append(dict(seed=seed,arms=evidence,gates=gates,all_gates=all(gates.values())))
    return dict(schema='ttt-hybrid.native-p51.p50-vs-trained-baselines-dev-result.v1',
        classification='AUTHORED_DEV_NOT_INDEPENDENT_SCIENCE',
        exact_p50_parent_sha='7d0c253fc884b1a8979cf762518d8b1bbfedbcbc',
        exact_model_blob='5d9ea57b827aa1796548447868fa5ee14c251b49',
        source_sha256=source_hash,sampling_indices_sha256=sample_hash,
        models_parameters=param_count,baselines={'ridge':ridge,'last_effect':last},
        model_seeds=list(INIT),optimizer_steps_per_arm=UPDATES,total_optimizer_updates=UPDATES*len(INIT)*len(arms),
        training_sample_budget_per_arm=UPDATES*BATCH,records=records,
        all_frozen_gates=all(x['all_gates'] for x in records),
        total_wall_seconds=time.perf_counter()-started,
        python=platform.python_version(),torch=torch.__version__,
        original_windows_linux_full_regression='NOT_RUN',
        independently_qualified_source='NOT_RUN',full_process_compute_matched='NOT_PROVEN',
        FORGE_integration='UNTOUCHED')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--receipt',type=Path,required=True)
    args=ap.parse_args();r=run();args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(source_sha256=r['source_sha256'],index_sha=r['sampling_indices_sha256'],
       params=r['models_parameters'],baseline_MSE=r['baselines']['ridge']['metrics']['all']['mse'],
       seeds=[{'seed':x['seed'],'gates':x['gates'],
               'MSE':{a:v['unfamiliar']['metrics']['all']['mse'] for a,v in x['arms'].items()},
               'effect_MAE':{a:v['unfamiliar']['metrics']['all']['effect_mae'] for a,v in x['arms'].items()}}
              for x in r['records']],all_gates=r['all_frozen_gates'],wall=r['total_wall_seconds']),indent=2))

if __name__=='__main__':main()
