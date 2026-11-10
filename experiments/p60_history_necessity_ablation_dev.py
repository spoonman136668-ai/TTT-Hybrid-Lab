"""P60 DEV: is completed action history necessary for expert-imitation success?

This is NOT reward-only action learning, a scientific first-use trial, or an
installed Gymnasium evaluation. The fixed teacher reads only current velocity.
Use identical trained weights to ablate historical evidence and TTT fast update.
"""
from __future__ import annotations
import argparse, hashlib, io, json, math, platform, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model import Configuration, HybridPilot
from native_p54_mountaincar_factual_bridge import published_reset, published_step

TRAIN_SEEDS=tuple(range(986000,986128))  # P59-POL prior FIT cohort, not evaluation
VAL_SEEDS=tuple(range(987200,987232))
OOD_SEEDS=tuple(range(987400,987496))
MODEL_SEEDS=(9611,9623,9631)
CAPACITY=4;HORIZON=256;UPDATES=400;BATCH=96;LR=.003
P59_FIT_SOURCE_HASH='fa101629292fafb9013d63f4c3626f5cde184039e86db41b17dcd3b77ff61c8f'
P59_REFERENCE_MODEL_GIT_BLOB='5d9ea57b827aa1796548447868fa5ee14c251b49'
P54_SOURCE_GIT_BLOB='9ecfc5822989336b3bbb14f29b8ebf84296c7039'


def git_blob(path):
    data=Path(path).read_bytes()
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def check_source():
    root=Path(__file__).resolve().parents[1]
    if git_blob(root/'model.py')!=P59_REFERENCE_MODEL_GIT_BLOB:
        raise ValueError('P60_PINNED_MODEL_DRIFT')
    if git_blob(root/'native_p54_mountaincar_factual_bridge.py')!=P54_SOURCE_GIT_BLOB:
        raise ValueError('P60_PINNED_PHYSICS_DRIFT')


def teacher_action(sensor):
    return -1.0 if float(sensor[1])<.00005 else 1.0


def witness(before,after,action):
    delta=np.asarray(after,dtype=np.float32)-np.asarray(before,dtype=np.float32)
    return np.array([before[0],before[1]*100.,delta[0]*100.,delta[1]*100.,action],np.float32)


def features(history,current):
    if len(history)>CAPACITY:raise ValueError('P60_UNBOUNDED_HISTORY')
    x=np.zeros((CAPACITY+1,5),dtype=np.float32)
    if len(history):x[CAPACITY-len(history):CAPACITY]=np.stack(history)
    x[-1,0]=current[0];x[-1,1]=current[1]*100.
    return x


def teacher_episodes(seeds,low=-.6,high=-.4):
    x=[];y=[];trace=hashlib.sha256();results=[]
    for seed in seeds:
        state=published_reset(int(seed),low=low,high=high);hist=[];reward=0.0;hit=False
        for t in range(HORIZON):
            a=teacher_action(state)
            x.append(features(hist,state));y.append(float(a>0.))
            actual=published_step(state,a)
            hist=(hist+[witness(state,actual.next_sensor,a)])[-CAPACITY:]
            trace.update(np.asarray([seed,t,a],dtype=np.float64).tobytes())
            trace.update(actual.next_sensor.tobytes())
            trace.update(np.asarray([actual.reward],dtype=np.float64).tobytes())
            state=actual.next_sensor;reward+=actual.reward
            if actual.terminated:hit=True;break
        results.append(dict(seed=int(seed),goal=hit,actual_steps=t+1,actual_reward=reward))
    return torch.from_numpy(np.stack(x)),torch.tensor(y,dtype=torch.float32),results,trace.hexdigest()


class HybridPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=HybridPilot(5,1,Configuration(width=8,heads=2,ssm=True,attention=True,ttt=True,depth=True))
    def forward(self,x,ttt_update=True):return self.body(x,ttt_update=ttt_update)[:,-1,0]


class CurrentOnlyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(2,34),nn.Tanh(),nn.Linear(34,32),nn.Tanh(),nn.Linear(32,1))
    def forward(self,x):return self.body(x[:,-1,:2]).squeeze(-1)


def model_params(m):return sum(p.numel() for p in m.parameters())


def ablated_logits(model,x,mode):
    if mode=='current_only':return model(x)
    if mode=='full':return model(x)
    if mode=='no_history':
        z=x.clone();z[:,:CAPACITY,:]=0.
        return model(z)
    if mode=='wrong_credit':
        z=x.clone();z[:,:CAPACITY,4]=torch.roll(z[:,:CAPACITY,4],shifts=1,dims=1)
        return model(z)
    if mode=='ttt_off':return model(x,ttt_update=False)
    raise ValueError('P60_UNKNOWN_ABLATION')


def rollout(model,seeds,mode,low=-.75,high=-.25):
    states=np.stack([published_reset(int(seed),low=low,high=high) for seed in seeds])
    n=len(seeds);history=np.zeros((n,CAPACITY,5),np.float32)
    done=np.zeros(n,bool);rewards=np.zeros(n,np.float64);actions=np.zeros(n,np.int32)
    digest=hashlib.sha256();model.eval();began=time.perf_counter()
    with torch.no_grad():
        for t in range(HORIZON):
            active=np.flatnonzero(~done)
            if not len(active):break
            x=np.zeros((len(active),CAPACITY+1,5),np.float32)
            x[:,:CAPACITY]=history[active];x[:,-1,0]=states[active,0];x[:,-1,1]=states[active,1]*100.
            logits=ablated_logits(model,torch.from_numpy(x),mode)
            a=np.where(logits.numpy()>=0.,1.,-1.)
            for j,i in enumerate(active):
                before=states[i].copy();action=float(a[j]);z=published_step(before,action)
                states[i]=z.next_sensor;actions[i]+=1;rewards[i]+=z.reward
                history[i,:-1]=history[i,1:].copy();history[i,-1]=witness(before,z.next_sensor,action)
                digest.update(np.asarray([int(seeds[i]),t,action],np.float64).tobytes())
                digest.update(z.next_sensor.tobytes())
                digest.update(np.asarray([z.reward],np.float64).tobytes())
                done[i]=z.terminated
    return dict(n=n,success=int(done.sum()),success_rate=float(done.mean()),mean_reward=float(rewards.mean()),
                mean_actual_steps=float(actions.mean()),total_actual_steps=int(actions.sum()),
                sha256=digest.hexdigest(),wall_seconds=time.perf_counter()-began)


def cold_restore(model,x):
    m=model.eval();sample=x[:12]
    with torch.no_grad():a=m(sample)
    b=io.BytesIO();torch.save(m.state_dict(),b);b.seek(0)
    restored=type(m)().eval();restored.load_state_dict(torch.load(b,weights_only=True)['state_dict'] if False else torch.load(b,weights_only=True))
    with torch.no_grad():c=restored(sample)
    return bool(torch.equal(a,c)),hashlib.sha256(b.getvalue()).hexdigest()


def train(model,train_x,train_y,val_x,val_y,indices):
    opt=torch.optim.AdamW(model.parameters(),lr=LR);curve=[];started=time.perf_counter()
    for step in range(UPDATES):
        model.train();ids=indices[step];guess=model(train_x[ids]);loss=F.binary_cross_entropy_with_logits(guess,train_y[ids])
        if not bool(torch.isfinite(loss)):raise ValueError('P60_NONFINITE_TRAIN_LOSS')
        opt.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if (step+1)%100==0:
            model.eval()
            with torch.no_grad():
                v=model(val_x);acc=float(((v>=0)==(val_y>=.5)).float().mean())
            curve.append(acc)
    return dict(training_seconds=time.perf_counter()-started,validation_curve=curve,
                plateau=abs(curve[-1]-curve[-2])<=.01,train_last_loss=float(loss.detach()))


def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    check_source();started=time.perf_counter()
    train_x,train_y,train_worlds,train_sha=teacher_episodes(TRAIN_SEEDS)
    if train_sha!=P59_FIT_SOURCE_HASH:raise ValueError('P60_PRIOR_EXPERT_FIT_SOURCE_REPRODUCIBILITY_FAILURE')
    val_x,val_y,val_worlds,val_sha=teacher_episodes(VAL_SEEDS)
    _,_,ood_worlds,ood_sha=teacher_episodes(OOD_SEEDS,-.75,-.25)
    ntrain=len(train_x);cases=[]
    for seed in MODEL_SEEDS:
        g=np.random.default_rng(910000+seed)
        indices=g.integers(0,ntrain,size=(UPDATES,BATCH))
        sampler_sha=hashlib.sha256(indices.tobytes()).hexdigest()
        models={};fits={};tests={};colds={}
        for name,typ in [('Hybrid',HybridPolicy),('CurrentOnly',CurrentOnlyPolicy)]:
            torch.manual_seed(seed);model=typ()
            fit=train(model,train_x,train_y,val_x,val_y,indices)
            exact,cold_sha=cold_restore(model,val_x)
            models[name]=model;fits[name]=fit;colds[name]=dict(exact=exact,weights_sha256=cold_sha)
        for mode in ('full','no_history','wrong_credit','ttt_off'):
            tests[mode]=rollout(models['Hybrid'],OOD_SEEDS,mode)
        tests['current_only']=rollout(models['CurrentOnly'],OOD_SEEDS,'current_only')
        orig=tests['full'];no=tests['no_history'];base=tests['current_only'];wrong=tests['wrong_credit'];ttt=tests['ttt_off']
        gate={
            'full_goal_rate_ge_95pct':orig['success_rate']>=.95,
            'full_success_gt_both_memoryless_by_10pp':all(orig['success_rate']-q['success_rate']>=.10 for q in (no,base)),
            'full_reward_gt_both_memoryless_by_5':all(orig['mean_reward']-q['mean_reward']>=5. for q in (no,base)),
            'wrong_credit_success_loss_ge_10pp':orig['success_rate']-wrong['success_rate']>=.10,
            'ttt_update_off_success_loss_ge_10pp':orig['success_rate']-ttt['success_rate']>=.10,
            'all_validation_plateau':all(z['plateau'] for z in fits.values()),
            'all_weight_cold_exact':all(z['exact'] for z in colds.values())}
        print(json.dumps(dict(seed=seed,goals={k:v['success'] for k,v in tests.items()},rewards={k:round(v['mean_reward'],3) for k,v in tests.items()},gates=gate)),flush=True)
        cases.append(dict(seed=seed,sampler_sha256=sampler_sha,train=fits,tests=tests,weights=colds,gates=gate,all_gates=all(gate.values())))
    return dict(schema='ttt-hybrid.p60.factual-memory-necessity-dev-result.v1',
      classification='AUTHORED_IMITATION_DEVELOPMENT_NOT_INDEPENDENT_CAUSAL_SCIENCE',
      pretrained_parent='P59-POL PR158: training examples recreated and SHA verified; original P59 module unchanged',
      fit_trace_sha256=train_sha,val_trace_sha256=val_sha,ood_trace_sha256=ood_sha,
      fit_executed_actions=len(train_x),val_episodes=len(val_worlds),ood_episodes=len(ood_worlds),
      teacher_ood_success=sum(q['goal'] for q in ood_worlds),
      parameters={'Hybrid':model_params(HybridPolicy()),'CurrentOnly':model_params(CurrentOnlyPolicy())},
      initializations=cases,all_frozen_gates=all(x['all_gates'] for x in cases),
      wall_seconds=time.perf_counter()-started,python=platform.python_version(),torch=torch.__version__,
      gymnasium_installed_original='NOT_RUN',external_scientific_source_first_use='NOT_RUN',
      exact_candidate_multi_platform_full_regression='NOT_RUN',FORGE='UNCHANGED')


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',required=True,type=Path)
    args=p.parse_args();result=run();args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('all_frozen_gates','parameters','teacher_ood_success','wall_seconds')},sort_keys=True))

if __name__=='__main__':main()
