"""P82 no-extra-parameter memory-priority fusion, author-synthetic DEV only.
Freeze before first P82 data: f293e221013ed2b425abf6d8bae848f9b92bb36e.
Imported comparator and reference original model are source byte pinned.
"""
from __future__ import annotations
import copy
import hashlib
import json
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from model import HybridPilot, Configuration
from native_p78_bounded_raw_ttt import BoundedRawTTT
from experiments import p81_readout_information_dev as p81
from experiments.p72_original_hybrid_transfer_dev import cache_prediction

D=16
SEEDS={'fit':820111,'val':820113,'transfer':820117,'negative':820119}
N={'fit':256,'val':128,'transfer':128,'negative':64}
CORE_SEEDS=(820201,820203,820207)
STEPS,BATCH,CHECK=300,16,30
MAX_SECONDS=180
ARMS=('corrected','original_raw_fusion','original_native_TTT','no_memory')
FREEZE='f293e221013ed2b425abf6d8bae848f9b92bb36e'


def sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def world(split,n=None):
    if split not in SEEDS:raise ValueError('P82_UNKNOWN_SPLIT')
    n=N[split] if n is None else n
    if type(n) is not int or n<8 or n>N[split] or n%2:raise ValueError('P82_INVALID_WORLD_COUNT')
    g=torch.Generator().manual_seed(SEEDS[split])
    basis=torch.linalg.qr(torch.randn(n,D,D,generator=g)).Q
    keys=basis[:,:,:8].transpose(1,2).contiguous()
    values=torch.tanh(torch.randn(n,8,D,generator=g))
    k=keys.repeat(1,2,1);v=values.repeat(1,2,1)
    nuisance=32 if split=='transfer' else 8
    k=torch.cat((k,F.normalize(torch.randn(n,nuisance,D,generator=g),dim=-1)),1)
    v=torch.cat((v,torch.tanh(torch.randn(n,nuisance,D,generator=g))),1)
    roles=torch.cat((torch.ones(n,16,dtype=torch.bool),torch.zeros(n,nuisance,dtype=torch.bool)),1)
    order=torch.rand(n,k.shape[1],generator=g).argsort(1)
    k=k.gather(1,order[:,:,None].expand_as(k));v=v.gather(1,order[:,:,None].expand_as(v))
    roles=roles.gather(1,order)
    chosen=torch.randint(8,(n,),generator=g)
    query=keys[torch.arange(n),chosen];label=values[torch.arange(n),chosen]
    return {'facts':torch.cat((k,v),-1),'query':torch.cat((query,torch.zeros_like(query)),-1)[:,None],
            'label':label,'role_audit':roles}


def world_sha(w):
    return hashlib.sha256(b''.join(w[k].contiguous().numpy().tobytes() for k in ('facts','query','label','role_audit'))).hexdigest()


def audit(w):
    r=w['role_audit'];n,t=r.shape
    q=(r.float().reshape(n,4,t//4).mean((0,2))-r.float().mean()).abs().max().item()
    tr,held=r[:n//2],r[n//2:]
    pred=(tr.float().mean(0)>=.5)[None].expand_as(held)
    ba=float(((pred&held).sum()/held.sum().clamp_min(1)+((~pred)&(~held)).sum()/(~held).sum().clamp_min(1))*.5)
    clean=bool(torch.count_nonzero(w['query'][:,:,D:])==0 and all(torch.isfinite(w[k]).all() for k in ('facts','query','label')))
    return {'quartile_imbalance':q,'position_balanced_accuracy':ba,'query_unknown_effect_zero':clean,'pass':bool(q<=.05 and ba<=.55 and clean)}


def model(seed):
    torch.manual_seed(seed)
    m=HybridPilot(32,16,Configuration(width=16,heads=2,ssm=True,attention=True,ttt=True,depth=True)).eval()
    for p in m.parameters():p.requires_grad_(False)
    for p in (*m.ttt.proj.parameters(),*m.head.parameters()):p.requires_grad_(True)
    return m


def native_memory(m,z):
    norm=m.ttt.norm(z)
    keys=F.normalize(m.ttt.key(norm[:,:-1]),dim=-1)
    q=F.normalize(m.ttt.query(norm[:,-1]),dim=-1)
    vals=torch.tanh(m.ttt.value(norm[:,:-1]))
    mat=z.new_zeros((len(z),D,D))
    for step in range(keys.shape[1]):
        k=keys[:,step];pred=torch.bmm(mat,k[:,:,None]).squeeze(-1)
        mat=mat+m.ttt.eta*(vals[:,step]-pred)[:,:,None]*k[:,None,:]
    return torch.bmm(mat,q[:,:,None]).squeeze(-1),mat


def features(m,w):
    with torch.no_grad():
        x=torch.cat((w['facts'],w['query']),dim=1)
        z=m.attn(m.ssm(m.stem(x)))
        raw,state=BoundedRawTTT(max_events=48).read_from_episode(w['facts'],w['query'][:,0,:D])
        native,mat=native_memory(m,z)
        context=z[:,-1]
        depth_context=m.depth(context)
    return {'context':context.detach(),'depth_context':depth_context.detach(),'raw':raw.detach(),
            'native':native.detach(),'label':w['label'].detach(),
            'state_floats':state.fast_weights[0].numel(),'native_state_floats':mat[0].numel()}


def out(m,f,arm):
    if arm not in ARMS:raise ValueError('P82_UNKNOWN_ARM')
    raw=f['raw'];ctx=f['context'];zero=torch.zeros_like(raw)
    if arm=='corrected':
        hidden=m.ttt.proj(raw)+.05*f['depth_context']
    else:
        read=raw if arm=='original_raw_fusion' else f['native'] if arm=='original_native_TTT' else zero
        hidden=m.depth(ctx+m.ttt.proj(read))
    return m.head(hidden)


def errors(pred,label):return (pred-label).square().mean(-1)


def train(seed,arm,fit,val,indices,deadline):
    m=model(seed)
    start_hash=p81.core_sha(m)
    params=list(m.ttt.proj.parameters())+list(m.head.parameters())
    opt=torch.optim.AdamW(params,lr=.01,weight_decay=.01)
    best=float('inf');step=None;weights=None;trace=[];started=time.perf_counter()
    for i in range(1,STEPS+1):
        if time.perf_counter()>deadline:raise TimeoutError('P82_CPU_BUDGET')
        ix=indices[i-1]
        opt.zero_grad(set_to_none=True)
        loss=errors(out(m,{key:value[ix] for key,value in fit.items() if isinstance(value,torch.Tensor)},arm),fit['label'][ix]).mean()
        if not bool(torch.isfinite(loss)):raise ValueError('P82_NONFINITE_LOSS')
        loss.backward();nn.utils.clip_grad_norm_(params,1.);opt.step()
        if i%CHECK==0:
            with torch.no_grad():value=float(errors(out(m,val,arm),val['label']).mean())
            if not torch.isfinite(torch.tensor(value)):raise ValueError('P82_NONFINITE_VAL')
            trace.append({'step':i,'mse':value})
            if value<best:
                best=value;step=i;weights=copy.deepcopy(m.state_dict())
    m.load_state_dict(weights)
    return m,{'seed':seed,'arm':arm,'initial_sha':start_hash,'best_mse':best,'best_step':step,
              'trace':trace,'best_model_sha':p81.core_sha(m),'elapsed_training_seconds':time.perf_counter()-started,
              'trainable_params':sum(p.numel() for p in params)}


def study():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    started=time.perf_counter();deadline=started+MAX_SECONDS
    receipt={'schema':'ttt-hybrid.p82-capacity-matched-original-fusion-dev-result.v1',
             'freeze_commit':FREEZE,'status':'NOT_STARTED','scientific_pass':False,
             'OOD_generated':False,'OOD_scored':False,'negative_generated':False,
             'original_source_unchanged':True,'full_original_model_training':False,
             'whole_process_compute_matched':False,'native_gpu':'NOT_RUN'}
    root=Path(__file__).resolve().parents[1]
    expected={'model.py':'5d9ea57b827aa1796548447868fa5ee14c251b49',
              'native_p78_bounded_raw_ttt.py':'db948c53bb1a28af5fe0e6a62ef71463db6d95ba',
              'experiments/p81_readout_information_dev.py':'80f136a3507cf53434695c1d083cb774d40857e9',
              'experiments/p72_original_hybrid_transfer_dev.py':'83099a6fb0ae3f37aea2715ff8351caf5f5cd39e'}
    receipt['custody']={k:{'expected':v,'observed':p81.git_blob(root/k)} for k,v in expected.items()}
    if any(v['expected']!=v['observed'] for v in receipt['custody'].values()):
        receipt['status']='DEV_FAIL_SOURCE_CUSTODY';return receipt
    fit=world('fit');val=world('val')
    receipt['source_audits']={'fit':audit(fit),'val':audit(val)}
    receipt['source_hashes']={'fit':world_sha(fit),'val':world_sha(val)}
    if not all(v['pass'] for v in receipt['source_audits'].values()):
        receipt['status']='DEV_FAIL_SOURCE_LEAKAGE';return receipt
    receipt['cold_restore']=p81.cold_restores({k:v[:8] for k,v in val.items()})
    if not receipt['cold_restore']:
        receipt['status']='DEV_FAIL_COLD_RESTORE';return receipt
    # Cache is original P72 streaming 8-slot 256-float comparator; no P82 role inputs.
    controls={'zero_mse':float(val['label'].square().mean()),
              'fixed_raw_mse':float(errors(BoundedRawTTT(max_events=48).read_from_episode(val['facts'],val['query'][:,0,:D])[0],val['label']).mean()),
              'frequency_256_mse':float(errors(cache_prediction(val),val['label']).mean())}
    controls['strongest_fixed_mse']=min(controls['fixed_raw_mse'],controls['frequency_256_mse'])
    controls['pass']=bool(max(controls['fixed_raw_mse'],controls['frequency_256_mse'])<=.75*controls['zero_mse'])
    receipt['validation_controls']=controls
    if not controls['pass']:
        receipt['status']='DEV_FAIL_CONVENTIONAL_CONTROLS';return receipt
    indices=torch.randint(N['fit'],(STEPS,BATCH),generator=torch.Generator().manual_seed(820299))
    receipt['sampler_sha']=sha(indices)
    rows=[];paired=[]
    for seed in CORE_SEEDS:
        m=model(seed);full_start=p81.core_sha(m)
        pre=time.perf_counter();ff=features(m,fit);fv=features(m,val);feature_seconds=time.perf_counter()-pre
        if full_start!=p81.core_sha(m) or ff['state_floats']!=256 or fv['native_state_floats']!=256:
            receipt['status']='DEV_FAIL_REFERENCE_OR_STATE_CUSTODY';return receipt
        models={};scores={}
        for arm in ARMS:
            trained,meta=train(seed,arm,ff,fv,indices,deadline)
            models[arm]=trained;scores[arm]=meta
        raw=scores['corrected'];orig=scores['original_raw_fusion'];native=scores['original_native_TTT']
        stable=all(c['mse']<=1.05*raw['best_mse'] for c in raw['trace'][-3:])
        qualified=bool(raw['best_mse']<=.95*controls['strongest_fixed_mse'] and
              raw['best_mse']<=.80*orig['best_mse'] and
              raw['best_mse']<=.75*native['best_mse'] and raw['best_step']<STEPS and stable)
        rows.append({'seed':seed,'model_initial_sha':full_start,
                     'identical_model_initializations':all(z['initial_sha']==full_start for z in scores.values()),
                     'core_state_floats':256,'feature_precompute_seconds':feature_seconds,
                     'corrected_plateau':stable,'qualified':qualified,'arms':scores})
        paired.append(models)
    receipt['model_rows']=rows
    receipt['distinct_seed_initializations']=len({x['model_initial_sha'] for x in rows})==3
    if not receipt['distinct_seed_initializations'] or not all(x['qualified'] and x['identical_model_initializations'] for x in rows):
        receipt['status']='DEV_FAIL_FROZEN_VAL_OR_CONVERGENCE';receipt['seconds']=time.perf_counter()-started;return receipt
    if time.perf_counter()>deadline:
        receipt['status']='DEV_FAIL_TIME_BUDGET';return receipt
    # Transfer is born only after all frozen FIT/VAL requirements qualify.
    ood=world('transfer');receipt['OOD_generated']=True
    receipt['source_hashes']['transfer']=world_sha(ood)
    g=features(model(CORE_SEEDS[0]),ood)
    fixed=errors(g['raw'],g['label']);freq=errors(cache_prediction(ood),ood['label'])
    strongest=fixed if float(fixed.mean())<=float(freq.mean()) else freq
    def boot(x,y):
        bs=torch.randint(len(x),(2000,len(x)),generator=torch.Generator().manual_seed(820401))
        return float(torch.quantile(1-x[bs].mean(-1)/y[bs].mean(-1).clamp_min(1e-10),.025))
    for row,arm_models in zip(rows,paired):
        f=features(model(row['seed']),ood)
        with torch.no_grad():
            e=errors(out(arm_models['corrected'],f,'corrected'),f['label'])
            base=errors(out(arm_models['original_raw_fusion'],f,'original_raw_fusion'),f['label'])
        lower=boot(e,strongest)
        row['transfer']={'mse':float(e.mean()),'orig_fusion_mse':float(base.mean()),
            'best_fixed_mse':float(strongest.mean()),'bootstrap_lower_vs_fixed':lower,
            'pass':bool(float(e.mean())<=.9*float(strongest.mean()) and float(e.mean())<=.8*float(base.mean()) and lower>0)}
    receipt['OOD_scored']=True
    if not all(x['transfer']['pass'] for x in rows):
        receipt['status']='DEV_FAIL_TRANSFER';receipt['seconds']=time.perf_counter()-started;return receipt
    # Negative independent source only on complete qualified transfer.
    neg=world('negative');receipt['negative_generated']=True
    corrupt={k:v.clone() for k,v in neg.items()}
    g=torch.Generator().manual_seed(820399);order=torch.rand(len(neg['facts']),24,generator=g).argsort(1)
    corrupt['facts'][:,:,D:]=neg['facts'][:,:,D:].gather(1,order[:,:,None].expand(len(order),24,D))
    receipt['source_hashes']['negative']=world_sha(neg)
    for row,arm_models in zip(rows,paired):
        m=model(row['seed']);clean=features(m,neg);wrong=features(m,corrupt)
        with torch.no_grad():
            a=float(errors(out(arm_models['corrected'],clean,'corrected'),clean['label']).mean())
            b=float(errors(out(arm_models['corrected'],wrong,'corrected'),wrong['label']).mean())
        row['negative']={'clean_mse':a,'wrong_credit_mse':b,'pass':bool(b>=1.4*a)}
    receipt['status']='DEV_PASS_CAPACITY_MATCHED_FUSION_ONLY' if all(x['negative']['pass'] for x in rows) else 'DEV_FAIL_NEGATIVE'
    receipt['seconds']=time.perf_counter()-started
    return receipt


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',required=True,type=Path)
    a=parser.parse_args();z=study();a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(z,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':z['status'],'seconds':z.get('seconds'),
          'OOD_scored':z['OOD_scored'],'scientific_pass':z['scientific_pass']}))

if __name__=='__main__':main()
