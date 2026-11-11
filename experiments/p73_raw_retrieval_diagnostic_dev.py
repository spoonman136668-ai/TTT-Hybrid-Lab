"""P73 frozen synthetic DEV: direct conventional retrieval vs exact original Hybrid.

Prospective freeze: experiments/p73_learnability_diagnostic_frozen.json commit
994dcd4ead1b2b176c5b612805e3a4aef78c1c7e. No P72 FIT/VAL reuse.

The conventional learner keeps the full factual history; this is not a
persistent-state or FLOP-matched architecture comparison, nor science first-use.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

try:
    from experiments import p72_original_hybrid_transfer_dev as p72
except ImportError:
    spec=importlib.util.spec_from_file_location('p72_for_p73', Path(__file__).with_name('p72_original_hybrid_transfer_dev.py'))
    p72=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p72)

D=16
SOURCE_SEEDS={'fit':730111,'val':730113,'negative':730119}
N={'fit':192,'val':128,'negative':64}
MODEL_SEEDS=(730201,730203,730207)
FIT_STEPS,BATCH,CHECK=120,8,20
P73_BATCH_SEED=730299
WRONG_CREDIT_SEED=730399
MAX_SECONDS=330


def tensor_sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def make_world(split,n=None):
    if split not in SOURCE_SEEDS: raise ValueError('P73_UNKNOWN_SOURCE')
    n=N[split] if n is None else n
    if not isinstance(n,int) or n<1 or n>N[split]: raise ValueError('P73_INVALID_COUNT')
    gen=torch.Generator().manual_seed(SOURCE_SEEDS[split]);
    basis=torch.linalg.qr(torch.randn(n,D,D,generator=gen)).Q
    targets=basis[:,:,:8].transpose(1,2).contiguous()
    effects=torch.tanh(torch.randn(n,8,D,generator=gen))
    key=targets.repeat(1,2,1)
    consequence=effects.repeat(1,2,1)
    nuisance=F.normalize(torch.randn(n,8,D,generator=gen),dim=-1)
    nuisance_effect=torch.tanh(torch.randn(n,8,D,generator=gen))
    key=torch.cat((key,nuisance),1)
    consequence=torch.cat((consequence,nuisance_effect),1)
    roles=torch.cat((torch.ones(n,16,dtype=torch.bool),torch.zeros(n,8,dtype=torch.bool)),1)
    order=torch.rand(n,24,generator=gen).argsort(1)
    key=key.gather(1,order[:,:,None].expand_as(key))
    consequence=consequence.gather(1,order[:,:,None].expand_as(consequence))
    roles=roles.gather(1,order)
    chosen=torch.randint(8,(n,),generator=gen)
    query=targets[torch.arange(n),chosen]
    labels=effects[torch.arange(n),chosen]
    return {'facts':torch.cat((key,consequence),-1),
            'query':torch.cat((query,torch.zeros_like(query)),-1)[:,None],
            'label':labels,'role_audit':roles}


def facts_only(world):
    """Training model may see only completed action/effect observations."""
    return world['facts']


def source_hash(w):
    return hashlib.sha256(b''.join(w[k].detach().contiguous().cpu().numpy().tobytes()
                 for k in ('facts','query','label','role_audit'))).hexdigest()


def wrong_credit(world,seed):
    """Shuffle observed consequences within each world, leave action/query/label fixed."""
    g=torch.Generator().manual_seed(seed)
    out={k:t.clone() for k,t in world.items()}
    f=out['facts']; count,steps,_=f.shape
    perm=torch.rand(count,steps,generator=g).argsort(1)
    f[:,:,D:]=f[:,:,D:].gather(1,perm[:,:,None].expand(count,steps,D))
    return out


class LearnedRetrieval(nn.Module):
    """A conventional trainable metricized raw-key/history retrieval comparator."""
    def __init__(self,seed):
        super().__init__()
        g=torch.Generator().manual_seed(seed)
        self.metric=nn.Parameter(torch.eye(D)+.012*torch.randn((D,D),generator=g))
        self.log_temperature=nn.Parameter(torch.tensor(-.9)+.012*torch.randn((),generator=g))

    def forward(self,events,query):
        if (not isinstance(events,torch.Tensor) or not isinstance(query,torch.Tensor)
            or events.ndim!=3 or query.ndim!=3 or query.shape[1]!=1 or
            events.shape[0]!=query.shape[0] or events.shape[-1]!=2*D or query.shape[-1]!=2*D
            or events.dtype!=torch.float32 or query.dtype!=torch.float32
            or events.device!=query.device or not bool(torch.isfinite(events).all())
            or not bool(torch.isfinite(query).all())):
            raise ValueError('P73_NONFACTUAL_OR_INVALID_HISTORY')
        key=F.normalize(events[:,:,:D]@self.metric,dim=-1)
        q=F.normalize(query[:,0,:D]@self.metric,dim=-1)
        sim=(key*q[:,None,:]).sum(-1)
        temperature=torch.exp(self.log_temperature).clamp(max=64.)
        w=torch.softmax(temperature*sim,dim=1)
        return (w[:,:,None]*events[:,:,D:]).sum(1)


def direct_eval(model,w):
    return p72.losses(model(facts_only(w),w['query']),w)


def fixed_raw_cosine(w,temperature=8.):
    f=w['facts']; key=F.normalize(f[:,:,:D],dim=-1)
    q=F.normalize(w['query'][:,0,:D],dim=-1)
    scores=(key*q[:,None,:]).sum(-1)
    weights=torch.softmax(temperature*scores,dim=-1)
    return (weights[:,:,None]*f[:,:,D:]).sum(1)


def train_conventional(seed,fit,val,idx,deadline):
    m=LearnedRetrieval(seed)
    init=tensor_sha(torch.cat((m.metric.flatten(),m.log_temperature.view(1))))
    with torch.no_grad(): initial_val=float(direct_eval(m,val).mean())
    opt=torch.optim.AdamW(m.parameters(),lr=.04,weight_decay=.01)
    best=float('inf');best_step=None;best_state=None;checks=[];start=time.perf_counter()
    for step in range(1,FIT_STEPS+1):
        if time.perf_counter()>deadline:
            return m,{'status':'TIME_BUDGET','steps':step-1,'seed':seed}
        batch=p72.sampled(fit,idx[step-1])
        opt.zero_grad(set_to_none=True)
        loss=direct_eval(m,batch).mean()
        if not bool(torch.isfinite(loss)):return m,{'status':'NONFINITE','steps':step,'seed':seed}
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(),1.)
        opt.step()
        if step%CHECK==0:
            with torch.no_grad(): mse=float(direct_eval(m,val).mean())
            checks.append({'step':step,'validation_mse':mse,'beta':float(torch.exp(m.log_temperature).detach())})
            if mse<best:
                best, best_step, best_state=mse,step,copy.deepcopy(m.state_dict())
    m.load_state_dict(best_state)
    return m,{'status':'DONE','seed':seed,'initial_model_sha256':init,
              'initial_val_mse':initial_val,'best_val_mse':best,'best_step':best_step,
              'steps':FIT_STEPS,'train_seconds':time.perf_counter()-start,
              'trained_parameters':sum(p.numel() for p in m.parameters()),
              'checks':checks,'final_beta':float(torch.exp(m.log_temperature).detach())}


def study():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.perf_counter();deadline=start+MAX_SECONDS
    fit,val=make_world('fit'),make_world('val')
    audit={k:p72.position_audit(w) for k,w in (('fit',fit),('val',val))}
    r={'schema':'ttt-hybrid.p73-full-hybrid-training-path-diagnostic-dev-result.v1',
       'classification':'AUTHORED_SYNTHETIC_DEV_ONLY_NO_INDEPENDENT_SCIENCE',
       'frozen_sha':'994dcd4ead1b2b176c5b612805e3a4aef78c1c7e',
       'source_seeds':SOURCE_SEEDS,'source_hashes':{'fit':source_hash(fit),'val':source_hash(val)},
       'source_audit':audit,'ood_generated':False,'ood_scored':False,
       'scientific_pass':False,'full_Windows_Linux_regression':'NOT_RUN',
       'gpu':'NOT_RUN','original_model_changed':False,'FORGE':'UNCHANGED',
       'status':'UNVERIFIED'}
    if not all(x['pass'] for x in audit.values()):
        r['status']='DEV_FAIL_SOURCE_POSITION_AUDIT';return r
    with torch.no_grad():
        zero=float(val['label'].square().mean())
        freq=float(p72.losses(p72.cache_prediction(val),val).mean())
        fixed=float(p72.losses(fixed_raw_cosine(val),val).mean())
    r['validation_controls']={'zero':zero,'fixed_cosine_temperature8':fixed,
        'frequency_cache_256f':freq,'frequency_cache_competent':freq<=.75*zero}
    if not r['validation_controls']['frequency_cache_competent']:
        r['status']='DEV_FAIL_DATA_LEARNABILITY_CONTROL';return r
    indices=torch.randint(len(fit['facts']),(FIT_STEPS,BATCH),generator=torch.Generator().manual_seed(P73_BATCH_SEED))
    r['sample_indices_sha256']=tensor_sha(indices)
    incorrect_fit=wrong_credit(fit,WRONG_CREDIT_SEED)
    incorrect_val=wrong_credit(val,WRONG_CREDIT_SEED+1)
    r['wrong_credit_fit_sha256']=source_hash(incorrect_fit)
    r['wrong_credit_val_sha256']=source_hash(incorrect_val)
    rows=[]
    for seed in MODEL_SEEDS:
        row={'seed':seed}
        model,metrics=train_conventional(seed,fit,val,indices,deadline)
        row['trained_retrieval']=metrics
        if metrics['status']!='DONE':
            r['status']='DEV_FAIL_BASELINE_TRAINING';r['rows']=rows+[row];return r
        _,negative=train_conventional(seed,incorrect_fit,incorrect_val,indices,deadline)
        row['wrong_credit_retrieval']=negative
        if negative['status']!='DONE':
            r['status']='DEV_FAIL_NEGATIVE_TRAINING';r['rows']=rows+[row];return r
        row['wrong_credit_ratio']=negative['best_val_mse']/max(metrics['best_val_mse'],1e-9)
        row['conventional_competence']=bool(metrics['best_val_mse']<=.60*zero and
                              metrics['best_val_mse']<=.85*metrics['initial_val_mse'])
        row['wrong_credit_pass']=bool(row['wrong_credit_ratio']>=1.5)
        rows.append(row)
    r['independent_conventional_init']=len({x['trained_retrieval']['initial_model_sha256'] for x in rows})==3
    r['rows']=rows
    if not (r['independent_conventional_init'] and all(x['conventional_competence'] and x['wrong_credit_pass'] for x in rows)):
        r['status']='DEV_FAIL_CONVENTIONAL_QUALIFICATION';r['seconds']=time.perf_counter()-start;return r
    for row in rows:
        seed=row['seed'];row['hybrid']={}
        for mode in ('all_write','learned'):
            _,meta=p72.train_arm(seed,mode,fit,val,indices,deadline)
            row['hybrid'][mode]=meta
            if meta['status']!='DONE':
                r['status']='DEV_FAIL_HYBRID_RUNTIME';r['seconds']=time.perf_counter()-start;return r
            row['hybrid'][mode]['val_competent']=meta['best_val_mse']<=.85*zero
        row['hybrid_same_initial_weights']=row['hybrid']['all_write']['init_sha256']==row['hybrid']['learned']['init_sha256']
    r['hybrid_pair_init_equal']=all(x['hybrid_same_initial_weights'] for x in rows)
    r['hybrid_independent_init']=len({x['hybrid']['all_write']['init_sha256'] for x in rows})==3
    r['hybrid_unqualified_pairs']=sum(not row['hybrid']['all_write']['val_competent']
        and not row['hybrid']['learned']['val_competent'] for row in rows)
    r['mean_val_mse']={key:sum(row['trained_retrieval']['best_val_mse'] if key=='retrieval'
          else row['hybrid'][key]['best_val_mse'] for row in rows)/len(rows)
          for key in ('retrieval','all_write','learned')}
    r['storage_and_compute']={'conventional_trainable_parameters':257,
             'conventional_factual_history_floats':24*32,
             'frequency_cache_floats':256,
             'hybrid_state_floats':'SSM16 + TTT256 + Attention K/V 24*16 = 656, plus weights',
             'whole_process_FLOPs_matched':False,
             'full_convergence_proven':False,
             'cpu_threads':1,
             'platform':platform.platform(),'torch':torch.__version__,
             'total_cpu_elapsed_seconds':time.perf_counter()-start}
    r['status']='DEV_DIAGNOSTIC_SUPPORTED_FULL_HYBRID_TRAINING_PATH' if (
         r['hybrid_pair_init_equal'] and r['hybrid_independent_init'] and
         r['hybrid_unqualified_pairs']>=2 and time.perf_counter()-start<=MAX_SECONDS) else 'DEV_DIAGNOSTIC_INCONCLUSIVE_OR_FALSIFIED'
    r['seconds']=time.perf_counter()-start
    return r


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--receipt',required=True,type=Path)
    args=parser.parse_args()
    result=study();args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':result['status'],'zero':result.get('validation_controls',{}).get('zero'),
        'ood_scored':False,'scientific_pass':False,'seconds':result.get('seconds')}),flush=True)

if __name__=='__main__':main()
