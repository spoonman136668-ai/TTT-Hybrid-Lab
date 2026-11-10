"""P59 frozen DEV: closed-loop physical motor policy after factual expert imitation.

Training labels are handcoded conventional momentum actions on ACTUAL completed
expert trajectories: this is supervised imitation, not autonomous RL or causal
consequence learning. Physical source is P54's pinned published-equation
transcription, not an independently installed official Gymnasium runtime.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model import Configuration,HybridPilot
from native_p54_mountaincar_factual_bridge import published_reset,published_step

PHYSICS_GIT_BLOB='9ecfc5822989336b3bbb14f29b8ebf84296c7039'
MODEL_GIT_BLOB='5d9ea57b827aa1796548447868fa5ee14c251b49'
TRAIN_SEEDS=tuple(range(986000,986128))
VALID_SEEDS=tuple(range(986200,986232))
OOD_SEEDS=tuple(range(986400,986496))
MODEL_SEEDS=(9511,9523,9531)
HORIZON=256
CAPACITY=4
TRAIN_UPDATES=400
BATCH=96
LR=.003
ARMS=('Hybrid','GRU','MLP')


def git_blob(raw:bytes)->str:
    return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()


def verify_exact_parent_source():
    import model,native_p54_mountaincar_factual_bridge as physics
    if git_blob(Path(model.__file__).read_bytes())!=MODEL_GIT_BLOB:
        raise ValueError('P59_MODEL_SOURCE_DRIFT')
    if git_blob(Path(physics.__file__).read_bytes())!=PHYSICS_GIT_BLOB:
        raise ValueError('P59_PHYSICAL_SOURCE_DRIFT')


def teacher_action(sensor):
    return -1.0 if float(sensor[1])<.00005 else 1.0


def row(before,after,action):
    delta=np.asarray(after,dtype=np.float32)-np.asarray(before,dtype=np.float32)
    return np.array((before[0],before[1]*100.,delta[0]*100.,delta[1]*100.,action),dtype=np.float32)


def feature(history,current):
    if len(history)>CAPACITY:
        raise ValueError('P59_FACTUAL_CAPACITY_OVERFLOW')
    x=np.zeros((CAPACITY+1,5),dtype=np.float32)
    if history:x[CAPACITY-len(history):CAPACITY]=np.stack(history)
    x[-1,:2]=current[0],current[1]*100.
    return x


def expert_episodes(seeds,low=-.6,high=-.4):
    features=[];labels=[];trace=hashlib.sha256();worlds=[]
    for seed in seeds:
        state=published_reset(int(seed),low=low,high=high);history=[]
        cumulative=0.;steps=0;hit=False
        for t in range(HORIZON):
            a=teacher_action(state)
            features.append(feature(history,state));labels.append(float(a>0))
            nxt=published_step(state,a)
            history=(history+[row(state,nxt.next_sensor,a)])[-CAPACITY:]
            trace.update(np.asarray([seed,t,a],dtype=np.float64).tobytes())
            trace.update(nxt.next_sensor.tobytes())
            trace.update(np.asarray([nxt.reward],dtype=np.float64).tobytes())
            state=nxt.next_sensor;cumulative+=nxt.reward;steps+=1
            if nxt.terminated:hit=True;break
        worlds.append({'seed':int(seed),'hit':hit,'steps':steps,'reward':cumulative})
    x=np.stack(features).astype('float32');y=np.asarray(labels,dtype='float32')
    return x,y,worlds,trace.hexdigest()


class HybridPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=HybridPilot(5,1,Configuration(width=8,heads=2,ssm=True,attention=True,ttt=True,depth=True))
    def forward(self,x):return self.body(x)[:,-1,0]

class GRUPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.GRU(5,17,batch_first=True)
        self.head=nn.Linear(17,1)
    def forward(self,x):return self.head(self.body(x)[0][:,-1]).squeeze(-1)

class MLPPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(25,24),nn.Tanh(),nn.Linear(24,24),nn.Tanh(),nn.Linear(24,1))
    def forward(self,x):return self.body(x.reshape(len(x),25)).squeeze(-1)

FACTORIES={'Hybrid':HybridPolicy,'GRU':GRUPolicy,'MLP':MLPPolicy}


def parameter_counts():
    out={key:sum(int(v.numel()) for v in factory().parameters()) for key,factory in FACTORIES.items()}
    if any(abs(v/out['Hybrid']-1)>.1 for key,v in out.items() if key!='Hybrid'):
        raise ValueError('P59_PARAMETER_BUDGET_MISMATCH')
    return out


def signed_action(logits):
    return torch.where(logits>=0,torch.ones_like(logits),-torch.ones_like(logits))


def batch_rollout(policy,seeds,low,high):
    """Each neural arm controls an independent real physical session; no teacher call."""
    n=len(seeds)
    states=np.stack([published_reset(int(seed),low=low,high=high) for seed in seeds])
    histories=np.zeros((n,CAPACITY,5),dtype=np.float32)
    counts=np.zeros(n,dtype=np.int32)
    done=np.zeros(n,dtype=bool);rewards=np.zeros(n);steps=np.zeros(n,dtype=np.int32)
    digest=hashlib.sha256()
    started=time.perf_counter();policy.eval()
    with torch.no_grad():
        for t in range(HORIZON):
            idx=np.flatnonzero(~done)
            if not len(idx):break
            x=np.zeros((len(idx),CAPACITY+1,5),dtype=np.float32)
            x[:,:CAPACITY]=histories[idx]
            x[:,-1,0]=states[idx,0];x[:,-1,1]=states[idx,1]*100.
            a=signed_action(policy(torch.from_numpy(x))).numpy()
            for j,k in enumerate(idx):
                before=states[k].copy();action=float(a[j])
                observed=published_step(before,action)
                states[k]=observed.next_sensor;steps[k]+=1;rewards[k]+=observed.reward
                histories[k,:-1]=histories[k,1:].copy()
                histories[k,-1]=row(before,states[k],action)
                counts[k]=min(CAPACITY,counts[k]+1)
                digest.update(np.asarray([int(seeds[k]),t,action],dtype=np.float64).tobytes())
                digest.update(states[k].tobytes())
                digest.update(np.asarray([observed.reward],dtype=np.float64).tobytes())
                done[k]=observed.terminated
    elapsed=time.perf_counter()-started
    return dict(worlds=n,successes=int(done.sum()),success=float(done.mean()),
                mean_return=float(rewards.mean()),mean_steps=float(steps.mean()),
                total_actual_steps=int(steps.sum()),trace_sha256=digest.hexdigest(),seconds=elapsed)


def teacher_summary(worlds):
    return dict(worlds=len(worlds),successes=sum(int(v['hit']) for v in worlds),
                success=float(np.mean([v['hit'] for v in worlds])),
                mean_return=float(np.mean([v['reward'] for v in worlds])),
                mean_steps=float(np.mean([v['steps'] for v in worlds])))


def cold_restore(model,validation):
    b=torch.tensor(validation[:10].copy());model.eval()
    with torch.no_grad():expected=model(b).clone()
    buf=io.BytesIO();torch.save(model.state_dict(),buf);buf.seek(0)
    other=type(model)();other.load_state_dict(torch.load(buf,weights_only=True));other.eval()
    with torch.no_grad():actual=other(b)
    return bool(torch.equal(expected,actual)),hashlib.sha256(buf.getvalue()).hexdigest()


def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.perf_counter();parameter=parameter_counts()
    xfit,yfit,fitworlds,fitsha=expert_episodes(TRAIN_SEEDS)
    xval,yval,valworlds,valsha=expert_episodes(VALID_SEEDS)
    xood,yood,oodworlds,oodsha=expert_episodes(OOD_SEEDS,low=-.75,high=-.25)
    train_x=torch.from_numpy(xfit);train_y=torch.from_numpy(yfit)
    val_x=torch.from_numpy(xval);val_y=torch.from_numpy(yval)
    teacher={'validation':teacher_summary(valworlds),'OOD':teacher_summary(oodworlds)}
    reports=[]
    for seed in MODEL_SEEDS:
        rng=np.random.default_rng(910000+seed)
        indexes=rng.integers(0,len(xfit),size=(TRAIN_UPDATES,BATCH))
        batch_sha=hashlib.sha256(indexes.tobytes()).hexdigest()
        arms={}
        for name,factory in FACTORIES.items():
            torch.manual_seed(seed);m=factory()
            untrained=batch_rollout(m,OOD_SEEDS,-.75,-.25)
            opt=torch.optim.AdamW(m.parameters(),lr=LR)
            curve=[];started=time.perf_counter()
            for i in range(TRAIN_UPDATES):
                m.train();ix=indexes[i]
                pred=m(train_x[ix]);loss=F.binary_cross_entropy_with_logits(pred,train_y[ix])
                opt.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
                opt.step()
                if (i+1)%100==0:
                    m.eval()
                    with torch.no_grad():
                        v=signed_action(m(val_x))
                        rate=float((v==(val_y*2-1)).float().mean())
                    curve.append(rate)
            train_s=time.perf_counter()-started
            validation=batch_rollout(m,VALID_SEEDS,-.6,-.4)
            ood=batch_rollout(m,OOD_SEEDS,-.75,-.25)
            cold,sha=cold_restore(m,xval)
            arms[name]=dict(train_seconds=train_s,validation_expert_accuracy_curve=curve,
                convergence_last_two_diff=abs(curve[-1]-curve[-2]),
                final_train_loss=float(loss.detach()),validation=validation,OOD=ood,
                untrained_OOD=untrained,cold_exact=cold,weights_sha256=sha)
        hybrid=arms['Hybrid'];conv=max(arms['GRU']['OOD']['success'],arms['MLP']['OOD']['success'])
        checks=dict(validation_relative_to_teacher=hybrid['validation']['success']>=.95*teacher['validation']['success'],
            OOD_relative_to_teacher=hybrid['OOD']['success']>=.95*teacher['OOD']['success'],
            OOD_reward_relative_to_teacher=hybrid['OOD']['mean_return']>=teacher['OOD']['mean_return']-5.0,
            OOD_success_relative_to_best_conventional=hybrid['OOD']['success']>=conv-.05,
            cold_exact_all=all(a['cold_exact'] for a in arms.values()),
            convergence_all=all(a['convergence_last_two_diff']<=.01 for a in arms.values()))
        reports.append(dict(seed=seed,batch_index_sha256=batch_sha,arms=arms,gates=checks,
                            frozen_per_seed_pass=all(checks.values())))
    return dict(schema='ttt-hybrid.p59.factual-long-horizon-imitation-policy-dev.v1',
      classification='AUTHORED_SUPERVISED_IMITATION_DEV_NOT_ACTION_CONSEQUENCE_LEARNING',
      parent_source_git_blobs=dict(physical=PHYSICS_GIT_BLOB,hybrid_reference=MODEL_GIT_BLOB),
      source_sha256={'train_executed_trajectories':fitsha,'validation_executed_trajectories':valsha,
                     'OOD_executed_trajectories':oodsha},
      parameters=parameter,source_sizes={'training_trajectories':len(TRAIN_SEEDS),'training_completed_actions':len(xfit),
               'validation_trajectories':len(VALID_SEEDS),'OOD_trajectories':len(OOD_SEEDS)},
      teacher=teacher,model_seeds=list(MODEL_SEEDS),updates=TRAIN_UPDATES,batch=BATCH,
      results=reports,overall_dev_pass=all(r['frozen_per_seed_pass'] for r in reports),
      local_wall_seconds=time.perf_counter()-start,
      original_unchanged_git_sources_verified='REMOTE_VERIFICATION_REQUIRED',
      official_installed_gymnasium='NOT_AVAILABLE',independent_science='NOT_RUN',
      full_final_Windows_Ubuntu='NOT_RUN',FORGE='UNTOUCHED')


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args();out=run();args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'params':out['parameters'],'sizes':out['source_sizes'],'teacher':out['teacher'],
        'seeds':[{ 'seed':v['seed'], 'gates':v['gates'],
            'OOD':{name:{k:rec['OOD'][k] for k in ('success','mean_return','mean_steps')}
                       for name,rec in v['arms'].items()},
            'val_accuracy':{name:rec['validation_expert_accuracy_curve'] for name,rec in v['arms'].items()},
            'train_seconds':{name:round(rec['train_seconds'],2) for name,rec in v['arms'].items()}
                 } for v in out['results']],
        'overall':out['overall_dev_pass'],'wall':out['local_wall_seconds']},indent=2))
if __name__=='__main__':main()
