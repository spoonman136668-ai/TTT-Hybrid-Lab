"""P79 prospective fresh-source 420-update convergence and strong fixed-baseline DEV.

Frozen BEFORE P79 source birth: experiments/p79_convergence_frozen.json at
448a8f43b03ec0e2aa202c165b64865f8eb4385f. Original HybridPilot unedited.
Compare original native TTT fast operand vs action-identity-preserving operand
under equal supervised training of *only existing* TTT.proj and output head.
"""
from __future__ import annotations
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from model import HybridPilot, Configuration
from native_p78_bounded_raw_ttt import BoundedRawTTT
from experiments import p72_original_hybrid_transfer_dev as p72

D=16
SEEDS={'fit':790111,'val':790113,'ood_dev':790117,'negative':790119}
N={'fit':256,'val':128,'ood_dev':128,'negative':64}
CORE_SEEDS=(790201,790203,790207)
UPDATES,BATCH,CHECK=420,16,30
MAX_WALL=240


def sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def core_sha(core):
    return hashlib.sha256(b''.join(x.detach().contiguous().cpu().numpy().tobytes()
                 for x in core.state_dict().values())).hexdigest()


def make_world(split,n=None):
    if split not in SEEDS: raise ValueError('P78_UNDECLARED_SPLIT')
    n=N[split] if n is None else n
    if type(n) is not int or not 1<=n<=N[split]:raise ValueError('P78_BAD_WORLD_COUNT')
    rng=torch.Generator().manual_seed(SEEDS[split])
    basis=torch.linalg.qr(torch.randn(n,D,D,generator=rng)).Q
    target_key=basis[:,:,:8].transpose(1,2).contiguous()
    effect=torch.tanh(torch.randn(n,8,D,generator=rng))
    x=target_key.repeat(1,2,1)
    y=effect.repeat(1,2,1)
    noise=32 if split=='ood_dev' else 8
    nk=F.normalize(torch.randn(n,noise,D,generator=rng),dim=-1)
    ny=torch.tanh(torch.randn(n,noise,D,generator=rng))
    x=torch.cat((x,nk),1)
    y=torch.cat((y,ny),1)
    role=torch.cat((torch.ones(n,16,dtype=torch.bool),torch.zeros(n,noise,dtype=torch.bool)),1)
    order=torch.rand(n,x.shape[1],generator=rng).argsort(1)
    x=x.gather(1,order[:,:,None].expand_as(x))
    y=y.gather(1,order[:,:,None].expand_as(y))
    role=role.gather(1,order)
    chosen=torch.randint(8,(n,),generator=rng)
    q=target_key[torch.arange(n),chosen]
    label=effect[torch.arange(n),chosen]
    return {'facts':torch.cat((x,y),-1),'query':torch.cat((q,torch.zeros_like(q)),-1)[:,None],
            'label':label,'role_audit':role}


def world_sha(world):
    return hashlib.sha256(b''.join(world[k].contiguous().numpy().tobytes()
             for k in ('facts','query','label','role_audit'))).hexdigest()


def audit(w):
    role=w['role_audit'];count,t=role.shape
    quart=role.float().reshape(count,4,t//4).mean((0,2))
    depart=float((quart-role.float().mean()).abs().max())
    train,held=role[:count//2],role[count//2:]
    pred=(train.float().mean(0)>=.5)[None].expand_as(held)
    tpr=(pred & held).sum()/held.sum().clamp_min(1)
    tnr=((~pred)&(~held)).sum()/(~held).sum().clamp_min(1)
    balanced=float((tpr+tnr)/2)
    return {'quartile_imbalance':depart,'balanced_index_only':balanced,
            'pass':bool(depart<=.05 and balanced<=.55)}


def core(seed):
    torch.manual_seed(seed)
    m=HybridPilot(32,16,Configuration(width=16,heads=2,ssm=True,
                  attention=True,ttt=True,depth=True)).eval()
    for p in m.parameters():p.requires_grad_(False)
    for p in (*m.ttt.proj.parameters(),*m.head.parameters()):p.requires_grad_(True)
    return m


def native_memory(core,z):
    # Exact original online read-before-write feature projections.
    norm=core.ttt.norm(z)
    key=F.normalize(core.ttt.key(norm[:,:-1]),dim=-1)
    q=F.normalize(core.ttt.query(norm[:,-1]),dim=-1)
    values=torch.tanh(core.ttt.value(norm[:,:-1]))
    fast=z.new_zeros((len(z),D,D))
    for i in range(key.shape[1]):
        ki=key[:,i]
        estimated=torch.bmm(fast,ki.unsqueeze(-1)).squeeze(-1)
        fast=fast+core.ttt.eta*((values[:,i]-estimated).unsqueeze(-1)*ki.unsqueeze(1))
    return torch.bmm(fast,q.unsqueeze(-1)).squeeze(-1),fast


def features(m,w):
    with torch.no_grad():
        data=torch.cat((w['facts'],w['query']),dim=1)
        z=m.attn(m.ssm(m.stem(data)))
        native,state=native_memory(m,z)
        raw,raw_state=BoundedRawTTT(max_events=48).read_from_episode(w['facts'],w['query'][:,0,:D])
        return {'last':z[:,-1].detach(),'native':native.detach(),'raw':raw.detach(),
                'target':w['label'].detach(),'native_state':state.detach(),
                'raw_state':raw_state.fast_weights.detach()}


def pure_depth(model,z):
    d=model.depth
    norm=d.norm(z)
    prob=torch.sigmoid(d.router(norm))
    mask=(prob>=.5).squeeze(-1)
    out=torch.zeros_like(z)
    if bool(mask.any()):
        gated=1.+prob[mask]-prob[mask].detach()
        out[mask]=d.ffn(norm[mask])*gated
    return z+out


def output(model,feat,mode):
    if mode not in ('raw_internal','native_original','no_memory'):raise ValueError('P78_INVALID_ARM')
    val=feat['raw'] if mode=='raw_internal' else feat['native'] if mode=='native_original' else torch.zeros_like(feat['raw'])
    z=feat['last']+model.ttt.proj(val)
    z=pure_depth(model,z)
    return model.head(z)


def losses(pred,target):return (pred-target).square().mean(-1)


def train(seed,mode,fit,val,index,deadline):
    m=core(seed)
    init=core_sha(m)
    params=list(m.ttt.proj.parameters())+list(m.head.parameters())
    optimizer=torch.optim.AdamW(params,lr=.01,weight_decay=.01)
    best=float('inf'); best_step=None; best_weights=None;trace=[];start=time.perf_counter()
    for t in range(1,UPDATES+1):
        if time.perf_counter()>deadline:raise TimeoutError('P78_TRAINING_TIME_BUDGET')
        ix=index[t-1]
        optimizer.zero_grad(set_to_none=True)
        loss=losses(output(m,{k:v[ix] for k,v in fit.items() if k in ('last','raw','native')},mode),fit['target'][ix]).mean()
        if not bool(torch.isfinite(loss)):raise ValueError('P78_NONFINITE_TRAIN_LOSS')
        loss.backward();nn.utils.clip_grad_norm_(params,1.0);optimizer.step()
        if t%CHECK==0:
            with torch.no_grad():score=float(losses(output(m,val,mode),val['target']).mean())
            trace.append({'step':t,'val_mse':score})
            if score<best:
                best=score;best_step=t;best_weights=copy.deepcopy(m.state_dict())
    m.load_state_dict(best_weights)
    return m,{'seed':seed,'mode':mode,'init_sha':init,'best_val_mse':best,
              'best_step':best_step,'train_seconds':time.perf_counter()-start,
              'params_trained':sum(p.numel() for p in params), 'trace':trace,
              'best_checkpoint_sha':core_sha(m)}


def fixed_raw_fast_mse(f):return float(losses(f['raw'],f['target']).mean())


def frequency(w):return p72.cache_prediction(w)


def bootstrap_lower(x,y,seed=790401,reps=2000):
    idx=torch.randint(len(x),(reps,len(x)),generator=torch.Generator().manual_seed(seed))
    return float(torch.quantile(1-x[idx].mean(-1)/y[idx].mean(-1).clamp_min(1e-10),.025))


def cold_equivalence(w):
    t=BoundedRawTTT(max_events=48)
    f=w['facts'];q=w['query'][:,0,:D]
    s=t.initial_state(f[:,0])
    full=t.ingest_confirmed(f,s,torch.ones(f.shape[:2],dtype=torch.bool))
    mid=f.shape[1]//2
    half=t.ingest_confirmed(f[:,:mid],s,torch.ones(f.shape[0],mid,dtype=torch.bool))
    loaded=type(half)(half.fast_weights.clone(),half.confirmed_count.clone())
    rest=t.ingest_confirmed(f[:,mid:],loaded,torch.ones(f.shape[0],f.shape[1]-mid,dtype=torch.bool))
    return (torch.equal(full.fast_weights,rest.fast_weights) and torch.equal(full.confirmed_count,rest.confirmed_count) and
            torch.equal(t.read_only(q,full),t.read_only(q,rest)) and
            sum(a.numel() for a in (full.fast_weights,))==len(f)*256)


def study():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.perf_counter();deadline=start+MAX_WALL
    fit,val=make_world('fit'),make_world('val')
    receipt={'schema':'ttt-hybrid.p79-bounded-convergence-dev-result.v1','frozen_commit':'448a8f43b03ec0e2aa202c165b64865f8eb4385f',
        'classification':'AUTHORED_SYNTHETIC_DEV_NOT_SCIENTIFIC_PASS','OOD_generated':False,'OOD_scored':False,
        'source_hashes':{'fit':world_sha(fit),'val':world_sha(val)},'source_audits':{'fit':audit(fit),'val':audit(val)},
        'scientific_pass':False,'FORGE':'UNCHANGED','original_model_source':'UNCHANGED','full_regression':'NOT_RUN',
        'GPU':'NOT_RUN','status':'NOT_COMPLETED','whole_process_matched':False}
    if not all(x['pass'] for x in receipt['source_audits'].values()):
        receipt['status']='DEV_FAIL_SOURCE_AUDIT';return receipt
    receipt['cold_restore_pass']=cold_equivalence({k:v[:8] for k,v in val.items()})
    if not receipt['cold_restore_pass']:
        receipt['status']='DEV_FAIL_COLD_RESTORE';return receipt
    idx=torch.randint(N['fit'],(UPDATES,BATCH),generator=torch.Generator().manual_seed(790299))
    receipt['sampler_sha']=sha(idx)
    zero=float(val['label'].square().mean())
    freq_mse=float(losses(frequency(val),val['label']).mean())
    once=core(CORE_SEEDS[0]); val_feature=features(once,val)
    fast_mse=fixed_raw_fast_mse(val_feature)
    receipt['validation_controls']={'zero':zero,'frequency_256':freq_mse,'raw_fixed_fast_256':fast_mse,
       'competence_pass':bool(max(freq_mse,fast_mse)<=.75*zero)}
    if not receipt['validation_controls']['competence_pass']:
        receipt['status']='DEV_FAIL_STRONG_BASELINE_COMPETENCE';return receipt
    models=[];rows=[]
    for seed in CORE_SEEDS:
        ci=core(seed);before=core_sha(ci)
        timer=time.perf_counter();ff=features(ci,fit);vv=features(ci,val)
        feature_s=time.perf_counter()-timer
        if core_sha(ci)!=before:
            receipt['status']='DEV_FAIL_FROZEN_CORE_DRIFT';return receipt
        a,am=train(seed,'raw_internal',ff,vv,idx,deadline)
        b,bm=train(seed,'native_original',ff,vv,idx,deadline)
        n,nm=train(seed,'no_memory',ff,vv,idx,deadline)
        with torch.no_grad():
            native_exact=(ci(torch.cat((val['facts'][:4],val['query'][:4]),dim=1))[:,-1]
                if False else None)
            raw_val=losses(output(a,vv,'raw_internal'),vv['target'])
            native_val=losses(output(b,vv,'native_original'),vv['target'])
        row={'seed':seed,'initial_identical_sha':am['init_sha']==bm['init_sha']==nm['init_sha'],
             'raw_internal':am,'native_original':bm,'no_memory':nm,'features_seconds':feature_s,
             'raw_val_mse':float(raw_val.mean()),'native_val_mse':float(native_val.mean()),
             'eligible':bool(am['best_val_mse']<=.95*min(freq_mse,fast_mse)
                             and am['best_val_mse']<=.75*bm['best_val_mse']
                             and am['best_val_mse']<=.75*zero
                             and am['best_step']<=390
                             and all(x['val_mse']<=1.05*am['best_val_mse'] for x in am['trace'][-2:])),'float_TTT_state':256}
        row['raw_late_plateau']=all(x['val_mse']<=1.05*am['best_val_mse'] for x in am['trace'][-2:])
        rows.append(row);models.append((a,b))
    receipt['model_rows']=rows
    receipt['different_core_seeds']=len({r['raw_internal']['init_sha'] for r in rows})==3
    if not receipt['different_core_seeds'] or not all(x['initial_identical_sha'] and x['eligible'] for x in rows):
        receipt['status']='DEV_FAIL_OR_UNQUALIFIED_VAL_CONVERGENCE_OR_STRONG_BASELINE';receipt['seconds']=time.perf_counter()-start;return receipt
    if time.perf_counter()>deadline:
        receipt['status']='DEV_FAIL_CPU_BUDGET';return receipt
    # OOD source is generated ONCE only after all preregistered validation controls pass.
    ood=make_world('ood_dev')
    receipt['source_hashes']['ood_dev']=world_sha(ood)
    receipt['source_audits']['ood_dev']=audit(ood)
    if not receipt['source_audits']['ood_dev']['pass']:
        receipt['status']='DEV_FAIL_OOD_SOURCE_AUDIT';return receipt
    negative=make_world('negative')
    wrong={k:v.clone() for k,v in negative.items()}
    gen=torch.Generator().manual_seed(790119)
    order=torch.rand(len(negative['facts']),negative['facts'].shape[1],generator=gen).argsort(1)
    wrong['facts'][:,:,D:]=wrong['facts'][:,:,D:].gather(1,order[:,:,None].expand(len(order),24,D))
    receipt['OOD_generated']=True;receipt['source_hashes']['negative']=world_sha(negative)
    oodzero=float(ood['label'].square().mean())
    oodfreq=float(losses(frequency(ood),ood['label']).mean())
    receipt['ood_controls']={'zero':oodzero,'frequency_256':oodfreq}
    for row,(a,b) in zip(rows,models):
        fo=features(core(row['seed']),ood)
        ng=features(core(row['seed']),negative);wg=features(core(row['seed']),wrong)
        with torch.no_grad():
            r=losses(output(a,fo,'raw_internal'),fo['target']);base=losses(output(b,fo,'native_original'),fo['target'])
            clean=float(losses(output(a,ng,'raw_internal'),ng['target']).mean())
            corrupt=float(losses(output(a,wg,'raw_internal'),wg['target']).mean())
        fixed_fast=losses(fo['raw'],fo['target'])
        fixed_freq=losses(frequency(ood),ood['label'])
        strong_fixed=fixed_fast if float(fixed_fast.mean())<=float(fixed_freq.mean()) else fixed_freq
        lower=bootstrap_lower(r,strong_fixed)
        row['ood']={'mse':float(r.mean()),'native_mse':float(base.mean()),'raw_fast_fixed_mse':fixed_raw_fast_mse(fo),
                    'zero':oodzero,'freq_256_mse':oodfreq,'strongest_fixed_mse':float(strong_fixed.mean()),
                    'relative_gain_over_native':1-float(r.mean()/base.mean()),
                    'bootstrap_lower_95_relative':lower,'negative_clean_mse':clean,'negative_wrong_credit_mse':corrupt,
                    'pass':bool(float(r.mean())<=.90*float(strong_fixed.mean()) and float(r.mean())<=.75*float(base.mean()) and
                       lower>0 and corrupt>=1.4*clean)}
    receipt['OOD_scored']=True
    receipt['status']='DEV_PASS_BOUNDED_INTERNAL_REFERENCE' if all(row['ood']['pass'] for row in rows) else 'DEV_FAIL_BOUNDED_INTERNAL_REFERENCE'
    receipt['seconds']=time.perf_counter()-start
    return receipt

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--receipt',required=True,type=Path)
    args=p.parse_args();r=study();args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':r['status'],'OOD_scored':r['OOD_scored'],'seconds':r.get('seconds'),'scientific_pass':False}))
