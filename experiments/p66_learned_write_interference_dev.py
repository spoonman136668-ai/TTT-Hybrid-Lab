"""P66 authored development experiment: train a causal TTT-matrix write gate.

This deliberately does not instantiate the original Hybrid neural architecture or
claim scientific first use. Event keys are factual executed actions; values are
observed outcomes. Query targets are used only to train/evaluate readouts, NEVER
as an online write feature. Full source and controls are fixed by P66 freeze.
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
import torch
from torch import nn
import torch.nn.functional as F

D = 16
TARGETS = 8
ETA = 0.5
SOURCES = {'fit': (81011, 192, 4, 64, False),
           'val': (81013, 48, 4, 64, False),
           'ood': (81017, 64, 3, 80, False),
           'orthogonal': (81019, 64, 3, 80, True)}
MODEL_SEEDS = (82031, 82033, 82037)
UPDATES, BATCH, CHECK = 120, 32, 40
THRESHOLDS = (0.03,0.06,0.1,0.16,0.25,0.4,0.7)


def source(name):
    if name not in SOURCES:
        raise ValueError('P66_UNKNOWN_SOURCE')
    seed,n,rep,noise,orth = SOURCES[name]
    gen = torch.Generator().manual_seed(seed)
    basis = torch.linalg.qr(torch.randn(n,D,D,generator=gen)).Q
    target_key = basis[:,:,:TARGETS].transpose(1,2).contiguous()
    target_value = torch.tanh(torch.randn(n,TARGETS,D,generator=gen))
    distractor_key = torch.randn(n,noise,D,generator=gen)
    if orth:
        distractor_key = torch.einsum('bdj,btj->btd',basis[:,:,TARGETS:],distractor_key[:,:,:D-TARGETS])
    distractor_key = F.normalize(distractor_key,dim=-1)
    distractor_value = torch.tanh(torch.randn(n,noise,D,generator=gen))
    actions = torch.cat([target_key.repeat(1,rep,1),distractor_key],dim=1)
    effects = torch.cat([target_value.repeat(1,rep,1),distractor_value],dim=1)
    labels = torch.cat([torch.ones(n,TARGETS*rep,dtype=torch.bool),
                        torch.zeros(n,noise,dtype=torch.bool)],dim=1)
    idx = torch.rand(n,actions.shape[1],generator=gen).argsort(dim=1)
    actions = actions.gather(1,idx[:,:,None].expand(-1,-1,D))
    effects = effects.gather(1,idx[:,:,None].expand(-1,-1,D))
    labels = labels.gather(1,idx)
    wrong = F.normalize(torch.randn(n,actions.shape[1],D,generator=gen),dim=-1)
    sha = hashlib.sha256(b''.join(z.contiguous().numpy().tobytes()
                    for z in (actions,effects,target_key,target_value,labels,wrong))).hexdigest()
    return dict(action=actions,effect=effects,target_key=target_key,
                target_value=target_value,evaluator_target=labels,wrong_action=wrong),sha


def batch_at(source_data,idx):
    return {k:v[idx] for k,v in source_data.items()}


class LearnedGate(nn.Module):
    """Content-sensitive online write rule on the P65 reference matrix equation."""
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(4))

    def forward(self,d,carry=None,lo=0,hi=None,mode='learned',threshold=None,random_p=None,
                random_seed=0,wrong_credit=False,return_mid=False):
        a,v=d['action'],d['effect']
        if a.shape!=v.shape or a.ndim!=3 or a.shape[2]!=D:
            raise ValueError('P66_INVALID_FACTUAL_STREAM')
        if not bool(torch.isfinite(a).all() and torch.isfinite(v).all()):
            raise ValueError('P66_NONFINITE_FACT')
        b,t,_=a.shape
        hi=t if hi is None else hi
        if not 0<=lo<=hi<=t: raise ValueError('P66_INVALID_RANGE')
        w=a.new_zeros(b,D,D) if carry is None else carry
        if w.shape!=(b,D,D): raise ValueError('P66_INVALID_FAST_STATE')
        rng=torch.Generator().manual_seed(random_seed)
        admissions=[];mid=None
        for j in range(lo,hi):
            key=d['wrong_action'][:,j] if wrong_credit else a[:,j]
            val=v[:,j]
            pred=torch.bmm(w,key[:,:,None]).squeeze(-1)
            residual=val-pred
            err=residual.square().mean(-1)
            if mode=='learned':
                feats=torch.stack((torch.ones_like(err),
                    torch.log(err+1e-6),torch.log(pred.square().mean(-1)+1e-6),
                    F.cosine_similarity(pred,val,dim=-1,eps=1e-6)),dim=1)
                gate=torch.sigmoid(feats@self.theta)
            elif mode=='all':gate=torch.ones_like(err)
            elif mode=='surprise':gate=(err<=threshold).to(err.dtype)
            elif mode=='random':
                gate=(torch.rand((b,),generator=rng)<random_p).to(err.dtype)
            elif mode=='oracle':gate=d['evaluator_target'][:,j].to(err.dtype)
            else:raise ValueError('P66_UNKNOWN_ARM')
            w=w+ETA*gate[:,None,None]*residual[:,:,None]*key[:,None,:]
            admissions.append(gate.detach())
            if return_mid and j==t//2-1:mid=w
        return w,(torch.stack(admissions,1).mean().item() if admissions else 0.),mid


def read(w,queries):
    return torch.einsum('bij,bkj->bki',w,queries)


def loss_final(w,d):
    return (read(w,d['target_key'])-d['target_value']).square().mean()


class TrainedGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.rec=nn.GRU(D*2,24,batch_first=True)
        self.readout=nn.Sequential(nn.Linear(24+D,64),nn.Tanh(),nn.Linear(64,D))

    def forward(self,d):
        features=torch.cat((d['action'],d['effect']),dim=-1)
        h,_=self.rec(features)
        queries=d['target_key']
        def predict(hh):
            return self.readout(torch.cat((hh[:,None,:].expand(-1,TARGETS,-1),queries),dim=-1))
        return predict(h[:,-1]),predict(h[:,h.shape[1]//2-1])


def objective(m,d,kind):
    if kind=='gate':
        full,_,mid=m(d,return_mid=True)
        return loss_final(full,d)+0.1*loss_final(mid,d)
    final,mid=m(d)
    return ((final-d['target_value']).square().mean()+
            0.1*(mid-d['target_value']).square().mean())


def train_model(m,kind,fit,val,indices):
    opt=torch.optim.Adam(m.parameters(),lr=0.015)
    checkpoints=[];start=time.perf_counter()
    for step,idx in enumerate(indices):
        d=batch_at(fit,idx)
        m.train();opt.zero_grad(set_to_none=True)
        loss=objective(m,d,kind)
        if not torch.isfinite(loss): raise ValueError('P66_NONFINITE_LOSS')
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
        opt.step()
        if (step+1)%CHECK==0:
            m.eval()
            with torch.no_grad():checkpoints.append(float(objective(m,val,kind)))
    return {'validation_loss_every_40':checkpoints,
            'last40_relative_change_le_5pct':abs(checkpoints[-1]-checkpoints[-2])<=0.05*max(1e-12,checkpoints[-2]),
            'train_seconds':time.perf_counter()-start,
            'params':sum(p.numel() for p in m.parameters())}


def fifo_mse(d):
    keys=d['action'][:,-TARGETS:,:]
    vals=d['effect'][:,-TARGETS:,:]
    sim=torch.einsum('bqd,bkd->bqk',d['target_key'],keys)
    nearest=sim.argmax(-1)
    matches=sim.gather(-1,nearest[:,:,None]).squeeze(-1)>1-1e-5
    pred=vals.gather(1,nearest[:,:,None].expand(-1,-1,D))
    return float(((pred*matches[:,:,None]-d['target_value'])**2).mean())


def negative_orthogonal(gate,orth):
    # All target events exactly retained in their original order; only nuisance
    # events are skipped in the comparator, which is an evaluator control.
    mask=orth['evaluator_target']
    idx=(~mask).to(torch.int64).argsort(dim=1,stable=True)
    target_count=int(mask[0].sum())
    clean={k:(v.gather(1,idx[:,:,None].expand_as(v))[:,:target_count] if v.ndim==3 and v.shape[1]==mask.shape[1]
              else v.gather(1,idx)[:,:target_count] if v.ndim==2 and v.shape[1]==mask.shape[1] else v)
           for k,v in orth.items()}
    with torch.no_grad():
        wfull,_,_=gate(orth,mode='all')
        wclean,_,_=gate(clean,mode='all')
    return abs(float(loss_final(wfull,orth))-float(loss_final(wclean,orth)))


def cold_exact(m,d):
    with torch.no_grad():
        full,_,_=m(d)
        midpoint=d['action'].shape[1]//2
        pref,_,_=m(d,lo=0,hi=midpoint)
        buff=io.BytesIO();torch.save({'weights':m.state_dict(),'state':pref},buff)
        buff.seek(0);payload=torch.load(buff,weights_only=True)
        fresh=LearnedGate();fresh.load_state_dict(payload['weights'])
        res,_,_=fresh(d,lo=midpoint,carry=payload['state'])
    return torch.equal(res,full),hashlib.sha256(buff.getvalue()).hexdigest()


def eval_fast(m,d,mode='learned',**kw):
    with torch.no_grad():
        w,frac,_=m(d,mode=mode,**kw)
        return float(loss_final(w,d)),frac


def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    fetched={k:source(k) for k in SOURCES}
    data={k:fetched[k][0] for k in SOURCES}
    digests={k:fetched[k][1] for k in SOURCES}
    generator=torch.Generator().manual_seed(82099)
    batches=torch.randint(192,(UPDATES,BATCH),generator=generator)
    batch_sha=hashlib.sha256(batches.numpy().tobytes()).hexdigest()
    gate_ref=LearnedGate()
    baseline_all,_=eval_fast(gate_ref,data['ood'],mode='all')
    tuned=[(tau,*eval_fast(gate_ref,data['val'],mode='surprise',threshold=tau)) for tau in THRESHOLDS]
    best_tau=min(tuned,key=lambda z:(z[1],z[0]))[0]
    surprise_ood,_=eval_fast(gate_ref,data['ood'],mode='surprise',threshold=best_tau)
    controls={'all_write_ood_mse':baseline_all,'surprise_ood_mse':surprise_ood,
              'surprise_val_mse_by_threshold':{str(t):e for t,e,_ in tuned},
              'surprise_threshold_selected_on_val':best_tau,
              'fifo_last8_ood_mse':fifo_mse(data['ood']),
              'oracle_ood_mse':eval_fast(gate_ref,data['ood'],mode='oracle')[0],
              'orthogonal_negative_loss_difference':negative_orthogonal(gate_ref,data['orthogonal'])}
    rows=[]
    for seed in MODEL_SEEDS:
        row={'seed':seed,'arms':{}}
        for kind in ('gate','gru'):
            torch.manual_seed(seed)
            model=LearnedGate() if kind=='gate' else TrainedGRU()
            training=train_model(model,kind,data['fit'],data['val'],batches)
            model.eval()
            if kind=='gate':
                value,write_fraction=eval_fast(model,data['ood'])
                frac_fit=eval_fast(model,batch_at(data['fit'],torch.arange(48)))[1]
                random_value,_=eval_fast(model,data['ood'],mode='random',
                                    random_p=frac_fit,random_seed=seed+20000)
                wrong_value,_=eval_fast(model,data['ood'],wrong_credit=True)
                cold,serialized_sha=cold_exact(model,batch_at(data['val'],torch.arange(8)))
                row['arms']['gate']={'ood_mse':value,'write_fraction_ood':write_fraction,
                    'write_fraction_fit':frac_fit,'random_matched_ood_mse':random_value,
                    'wrong_credit_ood_mse':wrong_value,'cold_restore_bitwise':cold,
                    'cold_sha256':serialized_sha,'gate_theta':model.theta.detach().tolist(),
                    'training':training}
            else:
                with torch.no_grad():
                    pred,_=model(data['ood'])
                    value=float(((pred-data['ood']['target_value'])**2).mean())
                row['arms']['gru']={'ood_mse':value,'training':training,
                    'state_floats':24,'capacity_match':False}
            print('P66',seed,kind,'OOD MSE',value,'training_s',round(training['train_seconds'],2),flush=True)
        gate=row['arms']['gate']; gru=row['arms']['gru']; gm=gate['ood_mse']
        best=min(baseline_all,surprise_ood,controls['fifo_last8_ood_mse'],gru['ood_mse'])
        row['conditions']={'gate_10pct_better_than_best_unprivileged':gm<=0.9*best,
            'gate_10pct_better_than_allwrite':gm<=0.9*baseline_all,
            'wrong_action_credit_20pct_worse':gate['wrong_credit_ood_mse']>=1.2*gm,
            'cold_resume_exact':gate['cold_restore_bitwise'],
            'training_plateau_both':all(x['training']['last40_relative_change_le_5pct'] for x in row['arms'].values())}
        row['strict_dev_pass']=all(row['conditions'].values())
        rows.append(row)
    return {'schema':'ttt-hybrid.p66-learned-write-mechanistic-dev-result.v1',
            'scope':'AUTHORED_DEV_ONLY_NOT_FULL_HYBRID_NOT_INDEPENDENT_SCIENCE',
            'sources_sha256':digests,'training_sampler_sha256':batch_sha,
            'model_seeds':MODEL_SEEDS,'controls':controls,'rows':rows,
            'strict_dev_success_all_seeds':all(r['strict_dev_pass'] for r in rows),
            'orthogonal_negative_pass':controls['orthogonal_negative_loss_difference']<=1e-5,
            'torch':torch.__version__,'python':platform.python_version(),
            'total_seconds':time.perf_counter()-start,
            'external_science':'NOT_RUN','original_model_modified':False,
            'training_capacity_and_compute_matched':False,
            'mandatory_full_Windows_Linux_native_GPU':'UNVERIFIED','FORGE':'UNCHANGED'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--receipt',type=Path,required=True)
    args=ap.parse_args()
    result=run();args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print('P66_RESULT',json.dumps({'strict_dev_success_all_seeds':result['strict_dev_success_all_seeds'],
        'orthogonal_negative_pass':result['orthogonal_negative_pass'],
        'controls':{k:v for k,v in result['controls'].items() if k!='surprise_val_mse_by_threshold'},
        'rows':[{'seed':r['seed'],'conditions':r['conditions'],
          'gate':r['arms']['gate']['ood_mse'],'gru':r['arms']['gru']['ood_mse']} for r in result['rows']]}),flush=True)


if __name__=='__main__':main()
