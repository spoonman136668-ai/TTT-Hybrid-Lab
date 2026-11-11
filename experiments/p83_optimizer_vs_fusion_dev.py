"""P83 frozen new-source optimizer vs readable factual-memory diagnostic.

Frozen before source birth: f56eef527eba4b8d89d85b8535e7263ad156948e.
Source and original model unchanged. Author synthetic DEV only, no science claim.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from experiments import p82_capacity_matched_fusion_dev as p82
from experiments import p81_readout_information_dev as p81
from native_p78_bounded_raw_ttt import BoundedRawTTT

D=16
SEEDS={'fit':830111,'val':830113,'transfer':830117,'negative':830119}
COUNTS={'fit':224,'val':112,'transfer':128,'negative':64}
MODEL_SEEDS=(830201,830203,830207)
STEPS,BATCH,CHECK=300,16,30
FREEZE='f56eef527eba4b8d89d85b8535e7263ad156948e'
ARMS=('constant','cosine','original_fusion')

def world(split,n=None,*,qualified=False):
    if split not in SEEDS:raise ValueError('P83_UNDECLARED_SPLIT')
    if split in ('transfer','negative') and not qualified:
        raise ValueError('P83_PROTECTED_UNQUALIFIED_SOURCE')
    n=COUNTS[split] if n is None else n
    if type(n) is not int or n<8 or n>COUNTS[split] or n%2:
        raise ValueError('P83_BAD_WORLD_COUNT')
    g=torch.Generator().manual_seed(SEEDS[split])
    basis=torch.linalg.qr(torch.randn(n,D,D,generator=g)).Q
    target_key=basis[:,:,:8].transpose(1,2).contiguous()
    effects=torch.tanh(torch.randn(n,8,D,generator=g))
    x=target_key.repeat(1,2,1)
    y=effects.repeat(1,2,1)
    nuisance=32 if split=='transfer' else 8
    x=torch.cat((x,F.normalize(torch.randn(n,nuisance,D,generator=g),dim=-1)),1)
    y=torch.cat((y,torch.tanh(torch.randn(n,nuisance,D,generator=g))),1)
    role=torch.cat((torch.ones(n,16,dtype=torch.bool),torch.zeros(n,nuisance,dtype=torch.bool)),1)
    perm=torch.rand(n,x.shape[1],generator=g).argsort(1)
    x=x.gather(1,perm[:,:,None].expand_as(x))
    y=y.gather(1,perm[:,:,None].expand_as(y))
    role=role.gather(1,perm)
    chosen=torch.randint(8,(n,),generator=g)
    q=target_key[torch.arange(n),chosen]
    label=effects[torch.arange(n),chosen]
    return {'facts':torch.cat((x,y),-1),
            'query':torch.cat((q,torch.zeros_like(q)),-1)[:,None],
            'label':label,'role_audit':role}

def digest(w):
    return hashlib.sha256(b''.join(w[k].contiguous().cpu().numpy().tobytes()
        for k in ('facts','query','label','role_audit'))).hexdigest()

def initial(seed):
    return p82.model(seed)

def fingerprint(model):
    return p81.core_sha(model)

def features(model,w):
    return p82.features(model,w)

def output(m,f,arm):
    if arm not in ARMS:raise ValueError('P83_UNKNOWN_ARM')
    return p82.out(m,f,'original_raw_fusion' if arm=='original_fusion' else 'corrected')

def loss_each(m,f,arm):
    return p82.errors(output(m,f,arm),f['label'])

def ridge_probe(m,fit,val):
    with torch.no_grad():
        a=m.ttt.proj(fit['raw'])+.05*fit['depth_context']
        b=m.ttt.proj(val['raw'])+.05*val['depth_context']
    beta=p81.fit_ridge(a,fit['label'],alpha=.01)
    return {'validation_mse':float(p82.errors(p81.read_ridge(b,beta),val['label']).mean()),
            'trainable_ridge_coefficients':int(beta.numel()),
            'sha256':hashlib.sha256(beta.contiguous().numpy().tobytes()).hexdigest()}

def train(seed,arm,fit,val,indices,deadline):
    m=initial(seed)
    original_sha=fingerprint(m)
    params=list(m.ttt.proj.parameters())+list(m.head.parameters())
    assert sum(t.numel() for t in params)==544
    opt=torch.optim.AdamW(params,lr=.01,weight_decay=.01)
    best=float('inf');best_step=None;best_state=None;trace=[];t0=time.perf_counter()
    for step in range(STEPS):
        if time.perf_counter()>deadline:raise TimeoutError('P83_CPU_BUDGET')
        if arm=='cosine':
            opt.param_groups[0]['lr']=.01*(.01+.99*(1+math.cos(math.pi*step/STEPS))/2)
        ix=indices[step]
        batch={k:v[ix] for k,v in fit.items() if isinstance(v,torch.Tensor)}
        opt.zero_grad(set_to_none=True)
        objective=loss_each(m,batch,arm).mean()
        if not bool(torch.isfinite(objective)):
            raise ValueError('P83_NONFINITE_LOSS')
        objective.backward()
        grad=nn.utils.clip_grad_norm_(params,1.0)
        if not bool(torch.isfinite(grad)):
            raise ValueError('P83_NONFINITE_GRAD')
        opt.step()
        if (step+1)%CHECK==0:
            with torch.no_grad():v=float(loss_each(m,val,arm).mean())
            if not math.isfinite(v):raise ValueError('P83_NONFINITE_VALIDATION')
            trace.append({'step':step+1,'mse':v})
            if v<best:
                best=v;best_step=step+1;best_state=copy.deepcopy(m.state_dict())
    m.load_state_dict(best_state)
    return m,{'initial_sha256':original_sha,'best_validation_mse':best,
              'best_step':best_step,'best_model_sha256':fingerprint(m),
              'validation_trace':trace,'train_seconds':time.perf_counter()-t0,
              'trained_parameter_count':544,'updates':STEPS}

def fixed_controls(w):
    direct=BoundedRawTTT(max_events=48).read_from_episode(w['facts'],w['query'][:,0,:D])[0]
    cache=p82.cache_prediction(w)
    return {'zero_mse':float(w['label'].square().mean()),
            'raw_direct_mse':float(p82.errors(direct,w['label']).mean()),
            'frequency_256_mse':float(p82.errors(cache,w['label']).mean())}

def study():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    t0=time.perf_counter();deadline=t0+120
    receipt={'schema':'ttt-hybrid.p83-optimizer-vs-fusion-dev-result.v1',
        'freeze_commit':FREEZE,'status':'NOT_STARTED','scientific_pass':False,
        'OOD_generated':False,'OOD_scored':False,'negative_generated':False,
        'full_original_model_trained':False,'whole_process_matched':False}
    root=Path(__file__).resolve().parents[1]
    blobs={'model.py':'5d9ea57b827aa1796548447868fa5ee14c251b49',
           'native_p78_bounded_raw_ttt.py':'db948c53bb1a28af5fe0e6a62ef71463db6d95ba',
           'experiments/p82_capacity_matched_fusion_dev.py':'c44f77093db501502c3741fdcf70f606f2f202b7'}
    receipt['source_blobs']={path:{'expected':sha,'actual':p81.git_blob(root/path)}
                             for path,sha in blobs.items()}
    if any(x['expected']!=x['actual'] for x in receipt['source_blobs'].values()):
        receipt['status']='DEV_FAIL_SOURCE_CUSTODY';return receipt
    fit,val=world('fit'),world('val')
    receipt['source_sha256']={'fit':digest(fit),'val':digest(val)}
    receipt['source_audits']={'fit':p82.audit(fit),'val':p82.audit(val)}
    if not all(x['pass'] for x in receipt['source_audits'].values()):
        receipt['status']='DEV_FAIL_LEAKAGE';return receipt
    receipt['cold_bitwise']=p81.cold_restores({k:v[:8] for k,v in val.items()})
    if not receipt['cold_bitwise']:
        receipt['status']='DEV_FAIL_COLD';return receipt
    fixed=fixed_controls(val)
    fixed['strongest_mse']=min(fixed['raw_direct_mse'],fixed['frequency_256_mse'])
    fixed['competent']=max(fixed['raw_direct_mse'],fixed['frequency_256_mse'])<=.75*fixed['zero_mse']
    receipt['fixed_controls']=fixed
    if not fixed['competent']:
        receipt['status']='DEV_FAIL_CONVENTIONAL';return receipt
    indices=torch.randint(len(fit['label']),(STEPS,BATCH),
                   generator=torch.Generator().manual_seed(830299))
    receipt['sampler_sha256']=hashlib.sha256(indices.numpy().tobytes()).hexdigest()
    rows=[];all_models=[]
    for seed in MODEL_SEEDS:
        original=initial(seed)
        initial_sha=fingerprint(original)
        pre=time.perf_counter()
        f=features(original,fit);v=features(original,val)
        pre_seconds=time.perf_counter()-pre
        if fingerprint(original)!=initial_sha or f['state_floats']!=256 or v['native_state_floats']!=256:
            receipt['status']='DEV_FAIL_CORE_OR_STATE';return receipt
        ridge=ridge_probe(original,f,v)
        trained={};arms={}
        for arm in ARMS:
            model,meta=train(seed,arm,f,v,indices,deadline)
            trained[arm]=model;arms[arm]=meta
        cosine=arms['cosine'];constant=arms['constant'];baseline=arms['original_fusion']
        plateau=all(point['mse']<=1.05*cosine['best_validation_mse']
                    for point in cosine['validation_trace'][-3:])
        ridge_pass=ridge['validation_mse']<=.90*fixed['strongest_mse']
        accepted=bool(ridge_pass and cosine['best_validation_mse']<=.95*fixed['strongest_mse']
                and cosine['best_validation_mse']<=.80*constant['best_validation_mse']
                and cosine['best_validation_mse']<=.80*baseline['best_validation_mse']
                and cosine['best_step']<STEPS and plateau)
        rows.append({'seed':seed,'initial_full_core_sha256':initial_sha,
           'paired_bitwise_initialization':all(x['initial_sha256']==initial_sha for x in arms.values()),
           'preprocess_seconds':pre_seconds,'ridge':ridge,'ridge_sufficient':ridge_pass,
           'cosine_plateau':plateau,'cosine_eligible':accepted,'arms':arms})
        all_models.append(trained)
    receipt['rows']=rows
    receipt['distinct_original_inits']=len({r['initial_full_core_sha256'] for r in rows})==len(MODEL_SEEDS)
    receipt['total_seconds']=time.perf_counter()-t0
    if not receipt['distinct_original_inits'] or not all(r['paired_bitwise_initialization'] for r in rows):
        receipt['status']='DEV_FAIL_INIT_CUSTODY';return receipt
    if not all(r['ridge_sufficient'] for r in rows):
        receipt['status']='DEV_DIAGNOSTIC_FROZEN_FEATURE_INSUFFICIENT';return receipt
    if not all(r['cosine_eligible'] for r in rows):
        receipt['status']='DEV_DIAGNOSTIC_OPTIMIZER_UNRESOLVED';return receipt
    # Never create/score fresh transfer unless the entire prospectively frozen VAL gate passes.
    transfer=world('transfer',qualified=True)
    receipt['OOD_generated']=True
    receipt['source_sha256']['transfer']=digest(transfer)
    control=BoundedRawTTT(max_events=48).read_from_episode(transfer['facts'],transfer['query'][:,0,:D])[0]
    baseline=p82.cache_prediction(transfer)
    c1=p82.errors(control,transfer['label']);c2=p82.errors(baseline,transfer['label'])
    stronger=c1 if float(c1.mean())<=float(c2.mean()) else c2
    ix=torch.randint(len(transfer['label']),(2000,len(transfer['label'])),
                     generator=torch.Generator().manual_seed(830401))
    for r,models in zip(rows,all_models):
        f=features(initial(r['seed']),transfer)
        with torch.no_grad():err=loss_each(models['cosine'],f,'cosine')
        lower=float(torch.quantile(1-err[ix].mean(-1)/stronger[ix].mean(-1).clamp_min(1e-10),.025))
        r['transfer']={'mse':float(err.mean()),'strong_fixed_mse':float(stronger.mean()),
            'paired_bootstrap_lower_95':lower,'pass':bool(float(err.mean())<=.9*float(stronger.mean()) and lower>0)}
    receipt['OOD_scored']=True
    if not all(r['transfer']['pass'] for r in rows):
        receipt['status']='DEV_FAIL_TRANSFER';return receipt
    neg=world('negative',qualified=True)
    receipt['negative_generated']=True
    corrupted={k:v.clone() for k,v in neg.items()}
    g=torch.Generator().manual_seed(830399)
    perm=torch.rand(len(neg['facts']),24,generator=g).argsort(1)
    corrupted['facts'][:,:,D:]=neg['facts'][:,:,D:].gather(1,perm[:,:,None].expand(len(neg['facts']),24,D))
    receipt['source_sha256']['negative']=digest(neg)
    for r,models in zip(rows,all_models):
        m=initial(r['seed'])
        good,bad=features(m,neg),features(m,corrupted)
        with torch.no_grad():
            clean=float(loss_each(models['cosine'],good,'cosine').mean())
            wrong=float(loss_each(models['cosine'],bad,'cosine').mean())
        r['negative']={'clean':clean,'wrong_credit':wrong,'pass':bool(wrong>=1.4*clean)}
    receipt['status']='DEV_PASS_OPTIMIZER_AND_TRANSFER_ONLY' if all(r['negative']['pass'] for r in rows) else 'DEV_FAIL_NEGATIVE'
    receipt['total_seconds']=time.perf_counter()-t0
    return receipt

def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args();r=study()
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':r['status'],'OOD_scored':r['OOD_scored'],
                      'seconds':r.get('total_seconds'),'science_pass':r['scientific_pass']}))

if __name__=='__main__':main()
