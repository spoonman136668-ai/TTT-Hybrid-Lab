"""P81: frozen, FIT-only ridge readout sufficiency in exact original HybridPilot.

Dev-only authored synthetic evidence. Prospective freeze commit:
e4498783e9d7025a9e1d365b2773521565cfe02a.
This does NOT train original hybrid weights or prove causal transfer.
No OOD split can be requested by this module.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import HybridPilot, Configuration
from native_p78_bounded_raw_ttt import BoundedRawTTT

D=16
SOURCE_SEEDS={'fit':810111,'val':810113,'negative':810119}
N={'fit':192,'val':128,'negative':64}
CORE_SEEDS=(810201,810203,810207)
RIDGE_ALPHA=0.01
FREEZE_COMMIT='e4498783e9d7025a9e1d365b2773521565cfe02a'


def git_blob(path:Path):
    b=path.read_bytes()
    return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()


def source(split, n=None):
    if split not in SOURCE_SEEDS:
        raise ValueError('P81_SEALED_OR_UNKNOWN_SPLIT')
    n=N[split] if n is None else n
    if type(n) is not int or not 1<=n<=N[split]:
        raise ValueError('P81_BAD_SOURCE_COUNT')
    g=torch.Generator().manual_seed(SOURCE_SEEDS[split])
    basis=torch.linalg.qr(torch.randn(n,D,D,generator=g)).Q
    targets=basis[:,:,:8].transpose(1,2).contiguous()
    effects=torch.tanh(torch.randn(n,8,D,generator=g))
    keys=targets.repeat(1,2,1)
    values=effects.repeat(1,2,1)
    nuisance=F.normalize(torch.randn(n,8,D,generator=g),dim=-1)
    nuisance_effects=torch.tanh(torch.randn(n,8,D,generator=g))
    keys=torch.cat((keys,nuisance),1)
    values=torch.cat((values,nuisance_effects),1)
    roles=torch.cat((torch.ones(n,16,dtype=torch.bool),torch.zeros(n,8,dtype=torch.bool)),1)
    order=torch.rand(n,24,generator=g).argsort(1)
    keys=keys.gather(1,order[:,:,None].expand_as(keys))
    values=values.gather(1,order[:,:,None].expand_as(values))
    roles=roles.gather(1,order)
    chosen=torch.randint(8,(n,),generator=g)
    query_action=targets[torch.arange(n),chosen]
    label=effects[torch.arange(n),chosen]
    return {'facts':torch.cat((keys,values),-1),
            'query':torch.cat((query_action,torch.zeros_like(query_action)),-1)[:,None,:],
            'label':label,'role_audit':roles}


def digest(world):
    return hashlib.sha256(b''.join(world[k].contiguous().cpu().numpy().tobytes()
        for k in ('facts','query','label','role_audit'))).hexdigest()


def audit(world):
    roles=world['role_audit'];n,t=roles.shape
    if t!=24 or n<8 or n%2:raise ValueError('P81_AUDIT_SHAPE')
    avg=roles.float().mean()
    quart=roles.float().reshape(n,4,6).mean((0,2))
    departure=float((quart-avg).abs().max())
    t0,held=roles[:n//2],roles[n//2:]
    guessed=(t0.float().mean(0)>=.5)[None,:].expand_as(held)
    tpr=(guessed&held).sum()/held.sum().clamp_min(1)
    tnr=((~guessed)&(~held)).sum()/(~held).sum().clamp_min(1)
    ba=float((tpr+tnr)*.5)
    future_hidden=bool(torch.count_nonzero(world['query'][:,:,D:])==0)
    schema=set(world)=={'facts','query','label','role_audit'}
    finite=all(bool(torch.isfinite(world[k]).all()) for k in ('facts','query','label'))
    return {'quartile_departure':departure,'index_only_balanced_accuracy':ba,
            'future_effect_hidden':future_hidden,'schema':schema,'finite':finite,
            'pass':bool(departure<=.05 and ba<=.55 and future_hidden and schema and finite)}


def core(seed):
    torch.manual_seed(seed)
    return HybridPilot(32,16,Configuration(width=16,heads=2,ssm=True,
                       attention=True,ttt=True,depth=True)).eval()


def core_sha(m):
    return hashlib.sha256(b''.join(t.detach().cpu().contiguous().numpy().tobytes()
         for t in m.state_dict().values())).hexdigest()


def factual_read(world):
    """P78 original 256 float32 state; only executed historical observations."""
    if not audit_local_schema(world):raise ValueError('P81_BAD_WORLD_SCHEMA')
    return BoundedRawTTT(max_events=24).read_from_episode(
        world['facts'],world['query'][:,0,:D])


def audit_local_schema(world):
    return (set(world)=={'facts','query','label','role_audit'}
            and world['facts'].ndim==3 and world['facts'].shape[1:]==(24,32)
            and world['query'].shape==(len(world['facts']),1,32)
            and bool(torch.count_nonzero(world['query'][:,:,D:])==0)
            and world['facts'].dtype==world['query'].dtype==torch.float32)


def cold_restores(world):
    f=world['facts'];q=world['query'][:,0,:D]
    t=BoundedRawTTT(max_events=24)
    initial=t.initial_state(f[:,0,:])
    conf=torch.ones((len(f),24),dtype=torch.bool)
    full=t.ingest_confirmed(f,initial,conf)
    first=t.ingest_confirmed(f[:,:12,:],initial,conf[:,:12])
    # Simulates weights_only=True restoration of only Tensors and primitive tag.
    copied=type(first)(first.fast_weights.clone(),first.confirmed_count.clone(),first.policy_tag)
    restored=t.ingest_confirmed(f[:,12:,:],copied,conf[:,12:])
    return bool(torch.equal(full.fast_weights,restored.fast_weights)
          and torch.equal(full.confirmed_count,restored.confirmed_count)
          and torch.equal(t.read_only(q,full),t.read_only(q,restored))
          and full.fast_weights.numel()==256*len(f))


def feature_views(m,world,raw):
    """Exact original four-block pre-TTT, native frozen projection and depth."""
    if not audit_local_schema(world):raise ValueError('P81_BAD_WORLD_SCHEMA')
    with torch.no_grad():
        seq=torch.cat((world['facts'],world['query']),dim=1)
        pre=m.attn(m.ssm(m.stem(seq)))
        context=pre[:,-1,:]
        projected=m.ttt.proj(raw)
        projected_depth=m.depth(projected)
        fused=context+projected
        fused_depth=m.depth(fused)
        # This is exactly P79's raw-internal original head/latent semantics.
        torch.testing.assert_close(m.head(fused_depth),m.head(m.depth(context+m.ttt.proj(raw))),atol=0,rtol=0)
        return {'raw_16':raw.detach(),
                'native_context_16':context.detach(),
                'memory_projected_16':projected.detach(),
                'memory_projected_after_depth_16':projected_depth.detach(),
                'sum_pre_depth_16':fused.detach(),
                'sum_post_depth_16':fused_depth.detach(),
                'concat_32':torch.cat((context,raw),dim=-1).detach(),
                'native_memory_actual':m(seq)[:,-1,:].detach()}


def fit_ridge(x,y,alpha=RIDGE_ALPHA):
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y):raise ValueError('P81_BAD_RIDGE_SHAPE')
    X=torch.cat((x.double(),torch.ones((len(x),1),dtype=torch.float64)),dim=1)
    Y=y.double()
    reg=torch.eye(X.shape[1],dtype=torch.float64)*alpha
    reg[-1,-1]=0
    params=torch.linalg.solve(X.T@X+reg,X.T@Y)
    return params


def read_ridge(x,params):
    X=torch.cat((x.double(),torch.ones((len(x),1),dtype=torch.float64)),dim=1)
    return (X@params).float()


def mse(x,y):
    return float((x-y).square().mean())


def negative_credit(w):
    out={k:v.clone() for k,v in w.items()}
    g=torch.Generator().manual_seed(810399)
    perm=torch.rand(len(w['facts']),24,generator=g).argsort(1)
    out['facts'][:,:,D:]=w['facts'][:,:,D:].gather(1,perm[:,:,None].expand(len(w['facts']),24,D))
    return out


def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    t0=time.perf_counter()
    root=Path(__file__).resolve().parents[1]
    blobs={'model.py':'5d9ea57b827aa1796548447868fa5ee14c251b49',
       'native_p78_bounded_raw_ttt.py':'db948c53bb1a28af5fe0e6a62ef71463db6d95ba'}
    custody={p:{'expected':expected,'actual':git_blob(root/p),
         'pass':git_blob(root/p)==expected} for p,expected in blobs.items()}
    result={'schema':'ttt-hybrid.p81-readout-ridge-mechanistic-dev-result.v1',
         'classification':'AUTHORED_SYNTHETIC_DEV_RANDOM_FROZEN_HYBRID_NOT_SCIENTIFIC_EVIDENCE',
         'freeze_commit':FREEZE_COMMIT,'original_blob_checks':custody,
         'OOD_generated':False,'OOD_scored':False,
         'science_pass':False,'whole_process_cost_matched':False,
         'original_hybrid_trained':False,'original_hybrid_changed':False,
         'status':'NOT_RUN'}
    if not all(x['pass'] for x in custody.values()):
        result['status']='DEV_FAIL_ORIGINAL_BLOB_DRIFT';return result
    fit,val=source('fit'),source('val')
    audits={k:audit(w) for k,w in [('fit',fit),('val',val)]}
    result['audits']=audits
    result['source_hashes']={k:digest(w) for k,w in [('fit',fit),('val',val)]}
    if not all(x['pass'] for x in audits.values()):
        result['status']='DEV_FAIL_SOURCE';return result
    cold=cold_restores({k:v[:8] for k,v in val.items()})
    result['cold_bitwise']=cold
    if not cold:
        result['status']='DEV_FAIL_COLD';return result
    fr,fs=factual_read(fit)
    vr,vs=factual_read(val)
    zero=float(val['label'].square().mean())
    fixed=mse(vr,val['label'])
    result['fixed_memory']={'zero_mse':zero,'raw_fixed_mse':fixed,'float_state_per_world':fs.fast_weights[0].numel(),
         'confirmed_events':int(fs.confirmed_count[0])}
    if fixed>.75*zero:
        result['status']='DEV_FAIL_FIXED_MEMORY_UNQUALIFIED';return result
    rows=[]
    for seed in CORE_SEEDS:
        m=core(seed)
        before=core_sha(m)
        f=feature_views(m,fit,fr);v=feature_views(m,val,vr)
        decoded={};params={}
        for kind in ('raw_16','native_context_16','memory_projected_16',
            'memory_projected_after_depth_16','sum_pre_depth_16',
            'sum_post_depth_16','concat_32'):
            beta=fit_ridge(f[kind],fit['label'])
            pred=read_ridge(v[kind],beta)
            params[kind]=beta
            decoded[kind]={'mse':mse(pred,val['label']),
                           'feature_dim':f[kind].shape[1],
                           'ridge_coefficients':beta.numel(),
                           'coefficient_sha256':hashlib.sha256(beta.contiguous().numpy().tobytes()).hexdigest(),
                           'rank_fit':int(torch.linalg.matrix_rank(f[kind].double()))}
        base=decoded['raw_16']['mse']
        row={'seed':seed,'original_core_sha256':before,
             'original_core_unchanged':before==core_sha(m),
             'original_native_prediction_mse':mse(v['native_memory_actual'],val['label']),
             'feature_readout':decoded,
             'raw_ridge_competent':bool(base<=.90*fixed),
             'concat_oracle_competent':bool(decoded['concat_32']['mse']<=1.10*base),
             'projection_decodable':bool(decoded['memory_projected_16']['mse']<=1.15*base),
             'depth_decodable':bool(decoded['memory_projected_after_depth_16']['mse']<=1.15*base),
             'additive_fusion_bottleneck':bool(decoded['sum_post_depth_16']['mse']>=1.25*base
                                    and decoded['concat_32']['mse']<=1.10*base)}
        rows.append(row)
    result['rows']=rows
    result['distinct_original_seeds']=len({x['original_core_sha256'] for x in rows})==3
    if not result['distinct_original_seeds'] or not all(r['original_core_unchanged'] for r in rows):
        result['status']='DEV_FAIL_MODEL_CUSTODY';return result
    if not all(r['raw_ridge_competent'] and r['concat_oracle_competent'] for r in rows):
        result['status']='DEV_FAIL_RIDGE_OR_ORACLE_QUALIFICATION'
        result['seconds']=time.perf_counter()-t0
        return result
    # Independent negative cohort only after predeclared eligibility.
    negative=source('negative')
    wrong=negative_credit(negative)
    result['source_hashes']['negative']=digest(negative)
    result['negative_source_audit']=audit(negative)
    if not result['negative_source_audit']['pass']:
        result['status']='DEV_FAIL_NEGATIVE_SOURCE';return result
    clean,_=factual_read(negative)
    corrupt,_=factual_read(wrong)
    wrong_error=mse(corrupt,negative['label'])
    clean_error=mse(clean,negative['label'])
    negative_pass=wrong_error>=2*clean_error and wrong_error>=.1
    result['negative_credit']={'clean_raw_mse':clean_error,
        'wrong_credit_raw_mse':wrong_error,'pass':negative_pass,
        'original_query_and_targets_unchanged':bool(torch.equal(wrong['query'],negative['query'])
                            and torch.equal(wrong['label'],negative['label']))}
    patterns={'projection_failure_3_of_3':all(not r['projection_decodable'] for r in rows),
          'depth_failure_3_of_3':all(not r['depth_decodable'] for r in rows),
          'additive_fusion_bottleneck_3_of_3':all(r['additive_fusion_bottleneck'] for r in rows)}
    result['frozen_patterns']=patterns
    qualified=negative_pass and any(patterns.values())
    result['status']='DEV_MECHANISTIC_DIAGNOSTIC_PASS_ONLY' if qualified else 'DEV_INCONCLUSIVE_OR_NEGATIVE'
    result['seconds']=time.perf_counter()-t0
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args()
    r=run()
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'status':r['status'],'seconds':r.get('seconds'),
                      'negative':r.get('negative_credit',{}).get('pass'),
                      'OOD_scored':False,'science_pass':False}),flush=True)

if __name__=='__main__':main()
