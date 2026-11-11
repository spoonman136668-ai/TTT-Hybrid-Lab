"""P77 preregistered frozen-original-Hybrid identity-bypass sidecar experiment.

Prospective authored synthetic DEV protocol: commit d01f9f4b7246ee88d10cb1d16ed8700b82cbc258.
Only historical executed (action, observed consequence) events are stored. Original
HybridPilot source and weights stay untouched. Bypass uses extra raw action state;
this is not the original TTT fast-weight or full-model output learning.
"""
from __future__ import annotations
import copy
import hashlib
import json
import time
import platform
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

try:
    from experiments import p76_native_alignment_dev as p76
except ImportError:
    import importlib.util
    _ref=Path(__file__).with_name('p76_native_alignment_dev.py')
    _spec=importlib.util.spec_from_file_location('p76_for_p77',_ref)
    p76=importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(p76)

D=16
SEEDS={'fit':770111,'val':770113,'ood_dev':770117,'negative':770119}
COUNTS={'fit':256,'val':128,'ood_dev':128,'negative':64}
MODEL_SEEDS=(770201,770203,770207)
UPDATES,BATCH,CHECK=180,32,30
SAMPLE_SEED=770299
BOOTSTRAP_SEED=770401
LIMIT_SECONDS=180


def make_world(split,n=None):
    if split not in SEEDS:raise ValueError('P77_UNKNOWN_SPLIT')
    n=COUNTS[split] if n is None else n
    if type(n) is not int or not 1<=n<=COUNTS[split]:raise ValueError('P77_BAD_COUNT')
    rng=torch.Generator().manual_seed(SEEDS[split])
    basis=torch.linalg.qr(torch.randn(n,D,D,generator=rng)).Q
    targets=basis[:,:,:8].transpose(1,2).contiguous()
    effects=torch.tanh(torch.randn(n,8,D,generator=rng))
    action=targets.repeat(1,2,1)
    observed=effects.repeat(1,2,1)
    noise=32 if split=='ood_dev' else 8
    distract=F.normalize(torch.randn(n,noise,D,generator=rng),dim=-1)
    distract_effect=torch.tanh(torch.randn(n,noise,D,generator=rng))
    action=torch.cat((action,distract),dim=1)
    observed=torch.cat((observed,distract_effect),dim=1)
    role=torch.cat((torch.ones(n,16,dtype=torch.bool),torch.zeros(n,noise,dtype=torch.bool)),dim=1)
    perm=torch.rand(n,16+noise,generator=rng).argsort(1)
    action=action.gather(1,perm[:,:,None].expand_as(action))
    observed=observed.gather(1,perm[:,:,None].expand_as(observed))
    role=role.gather(1,perm)
    chosen=torch.randint(8,(n,),generator=rng)
    query=targets[torch.arange(n),chosen]
    outcome=effects[torch.arange(n),chosen]
    return {'facts':torch.cat((action,observed),-1),
            'query':torch.cat((query,torch.zeros_like(query)),-1)[:,None],
            'label':outcome,'role_audit':role}


def source_sha(w):
    return hashlib.sha256(b''.join(w[k].contiguous().cpu().numpy().tobytes() for k in ('facts','query','label','role_audit'))).hexdigest()


def audit(w):
    return p76.audit(w)


def wrong_credit(world,seed=770119):
    shuffled={k:v.clone() for k,v in world.items()}
    n,t,_=shuffled['facts'].shape
    perm=torch.rand(n,t,generator=torch.Generator().manual_seed(seed)).argsort(1)
    shuffled['facts'][:,:,D:]=shuffled['facts'][:,:,D:].gather(1,perm[:,:,None].expand(n,t,D))
    return shuffled


def value_features(model,world):
    with torch.no_grad():
        z=model.attn(model.ssm(model.stem(torch.cat((world['facts'],world['query']),1))))
        t=model.ttt.norm(z)
        return torch.tanh(model.ttt.value(t[:,:-1]))


def ridge_fit(x,y,alpha=.01):
    if x.ndim!=2 or y.ndim!=2 or x.shape[0]!=y.shape[0] or x.shape[1]!=D or y.shape[1]!=D:
        raise ValueError('P77_BAD_RIDGE_DATA')
    xx=torch.cat((x.double(),torch.ones((len(x),1),dtype=torch.float64)),1)
    reg=torch.eye(D+1,dtype=torch.float64)*alpha
    reg[-1,-1]=0
    return torch.linalg.solve(xx.T@xx+reg,xx.T@y.double())


def ridge_apply(x,w):
    if x.shape[-1]!=D or w.shape!=(D+1,D):raise ValueError('P77_BAD_RIDGE_STATE')
    z=torch.cat((x.double(),torch.ones((*x.shape[:-1],1),dtype=torch.float64)),dim=-1)
    return (z@w).float()


class BypassAlignment(p76.Alignment):
    """513 parameters, raw action skip available at key and query addressing."""
    def __init__(self,seed):
        super().__init__(seed)
        self.mix_logit=nn.Parameter(torch.tensor(-1.3862943611,dtype=torch.float32))

    def logits(self,nkeys,nquery,rkeys,rquery):
        if not (nkeys.ndim==3 and rkeys.shape==nkeys.shape and nquery.shape==(len(nkeys),D)
                and rquery.shape==nquery.shape and bool(torch.isfinite(nkeys).all())
                and bool(torch.isfinite(rkeys).all()) and bool(torch.isfinite(nquery).all())
                and bool(torch.isfinite(rquery).all())):
            raise ValueError('P77_BAD_ALIGNED_OBSERVATION')
        alpha=torch.sigmoid(self.mix_logit)
        key=F.normalize((1-alpha)*F.normalize(nkeys@self.key_map,dim=-1)+alpha*rkeys,dim=-1)
        query=F.normalize((1-alpha)*F.normalize(nquery@self.query_map,dim=-1)+alpha*rquery,dim=-1)
        return p76.TEMP*(key*query[:,None,:]).sum(-1)


def logits(model,features,kind):
    if kind=='bypass':
        return model.logits(features['native_keys'],features['native_query'],features['raw_keys'],features['raw_query'])
    if kind=='native':return model.logits(features['native_keys'],features['native_query'])
    if kind=='raw':return model.logits(features['raw_keys'],features['raw_query'])
    raise ValueError('P77_BAD_ARM')


def evaluate(model,feature,kind,decoder=None,values=None):
    with torch.no_grad():
        score=logits(model,feature,kind)
        ix=score.argmax(1)
        idx=torch.arange(len(ix))
        observed=feature['observed_effects'][idx,ix]
        target=feature['labels']
        direct=(observed-target).square().mean(-1)
        hit=feature['positive'][idx,ix]
        o={'hit':float(hit.float().mean()),'mse':float(direct.mean()),
           'contrastive':float(p76.contrastive(score,feature['positive'])),'perworld':direct}
        if decoder is not None:
            if values is None:raise ValueError('P77_MISSING_NATIVE_VALUE')
            native_pred=ridge_apply(values[idx,ix],decoder)
            o['native_decoded_mse']=float((native_pred-target).square().mean())
        return o


def train_arm(seed,kind,fit,val,idx,start_time):
    module=BypassAlignment(seed) if kind=='bypass' else p76.Alignment(seed)
    initial_matrix_sha=p76.model_hash(module) if kind!='bypass' else hashlib.sha256(
        b''.join(p.detach().cpu().numpy().tobytes() for p in (module.key_map,module.query_map))).hexdigest()
    opt=torch.optim.AdamW(module.parameters(),lr=.02,weight_decay=.01)
    best=float('inf');weights=None;best_step=None;checks=[];t0=time.perf_counter()
    for step in range(UPDATES):
        if time.perf_counter()-start_time>LIMIT_SECONDS:raise TimeoutError('P77_CPU_BUDGET')
        ix=idx[step]
        opt.zero_grad(set_to_none=True)
        f={k:v[ix] for k,v in fit.items() if isinstance(v,torch.Tensor)}
        loss=p76.contrastive(logits(module,f,kind),f['positive'])
        if not bool(torch.isfinite(loss)):raise ValueError('P77_NONFINITE_TRAIN_LOSS')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(),1.0)
        opt.step()
        if (step+1)%CHECK==0:
            with torch.no_grad():val_loss=float(p76.contrastive(logits(module,val,kind),val['positive']))
            checks.append({'step':step+1,'val_contrastive':val_loss})
            if val_loss<best:
                best=val_loss;best_step=step+1;weights=copy.deepcopy(module.state_dict())
    if weights is None:raise ValueError('P77_NO_CHECKPOINT')
    module.load_state_dict(weights)
    new=BypassAlignment(seed+10000) if kind=='bypass' else p76.Alignment(seed+10000)
    new.load_state_dict(copy.deepcopy(module.state_dict()))
    cold=bool(torch.equal(logits(module,{k:v[:8] for k,v in val.items() if isinstance(v,torch.Tensor)},kind),
                            logits(new,{k:v[:8] for k,v in val.items() if isinstance(v,torch.Tensor)},kind)))
    result={'kind':kind,'best_val_loss':best,'best_step':best_step,'steps':UPDATES,
            'param_count':sum(p.numel() for p in module.parameters()),
            'initial_projection_sha256':initial_matrix_sha,
            'best_model_sha256':p76.model_hash(module),
            'checks':checks,'cold_bitwise':cold,'elapsed_seconds':time.perf_counter()-t0}
    if kind=='bypass':result['raw_skip_fraction']=float(torch.sigmoid(module.mix_logit).detach())
    return module,result


def strip_evaluation(x):
    return {k:v for k,v in x.items() if k!='perworld'}


def paired_lower(a,b,reps=2000,seed=BOOTSTRAP_SEED):
    index=torch.randint(len(a),(reps,len(a)),generator=torch.Generator().manual_seed(seed))
    return float(torch.quantile(1-a[index].mean(-1)/b[index].mean(-1).clamp_min(1e-10),.025))


def study():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    fit,val=make_world('fit'),make_world('val')
    audits={k:audit(w) for k,w in (('fit',fit),('val',val))}
    receipt={'schema':'ttt-hybrid.p77-identity-bypass-sidecar-dev.v1',
      'level':'AUTHORED_DEV_NOT_SCIENCE','freeze_commit':'d01f9f4b7246ee88d10cb1d16ed8700b82cbc258',
      'source_hashes':{'fit':source_sha(fit),'val':source_sha(val)},
      'source_audits':audits,'status':'NOT_COMPLETED','ood_generated':False,'ood_scored':False,
      'original_core_unchanged':False,'forge':'UNCHANGED','science_pass':False,
      'full_regression':'NOT_RUN','gpu':'NOT_RUN','whole_process_compute_matched':False}
    if not all(a['pass'] for a in audits.values()):
        receipt['status']='DEV_FAIL_SOURCE_AUDIT';return receipt
    raw=F.normalize(val['facts'][:,:,:D],dim=-1)
    rq=F.normalize(val['query'][:,0,:D],dim=-1)
    raw_idx=(raw*rq[:,None]).sum(-1).argmax(-1)
    raw_pred=val['facts'][torch.arange(len(raw_idx)),raw_idx,D:]
    zero=float(val['label'].square().mean())
    raw_mse=float((raw_pred-val['label']).square().mean())
    raw_hit=float(p76.labels_mask(val)[torch.arange(len(raw_idx)),raw_idx].float().mean())
    receipt['baseline_val']={'zero_mse':zero,'raw_exact_mse':raw_mse,'raw_exact_hit':raw_hit}
    if raw_hit<.99 or raw_mse>1e-5:
        receipt['status']='DEV_FAIL_RAW_BASELINE';return receipt
    sampler=torch.randint(COUNTS['fit'],(UPDATES,BATCH),generator=torch.Generator().manual_seed(SAMPLE_SEED))
    receipt['sampler_sha256']=p76.tensor_hash(sampler)
    models=[];rows=[]
    for seed in MODEL_SEEDS:
        core=p76.original(seed)
        baseline_sha=p76.model_hash(core)
        f,v=p76.features(core,fit),p76.features(core,val)
        fv=value_features(core,fit);vv=value_features(core,val)
        decoder=ridge_fit(fv.reshape(-1,D),fit['facts'][:,:,D:].reshape(-1,D))
        arms={};trained={}
        for kind in ('native','bypass','raw'):
            model,metadata=train_arm(seed,kind,f,v,sampler,start)
            ev=evaluate(model,v,kind,decoder if kind=='bypass' else None,
                         vv if kind=='bypass' else None)
            arms[kind]={**metadata,'val':strip_evaluation(ev)}
            trained[kind]=model
        native_init=arms['native']['initial_projection_sha256']
        same_init=native_init==arms['bypass']['initial_projection_sha256']
        unchanged=baseline_sha==p76.model_hash(core)
        good=bool(arms['bypass']['val']['hit']>=.90
                  and arms['bypass']['val']['mse']<=.15*zero
                  and arms['bypass']['val']['native_decoded_mse']<=.75*zero)
        rows.append({'seed':seed,'original_core_sha256':baseline_sha,'core_frozen':unchanged,
                     'native_vs_bypass_initial_projection_identical':same_init,
                     'native_value_fit_ridge_sha256':p76.tensor_hash(decoder),
                     'native_value_decoder_fit_mse':float((ridge_apply(fv,decoder)-fit['facts'][:,:,D:]).square().mean()),
                     'arms':arms,'bypass_val_eligible':good})
        models.append((core,trained,decoder))
    receipt['rows']=rows
    receipt['original_core_unchanged']=all(r['core_frozen'] for r in rows)
    receipt['independent_core_seeds']=len({r['original_core_sha256'] for r in rows})==3
    if not (receipt['original_core_unchanged'] and receipt['independent_core_seeds']
             and all(r['native_vs_bypass_initial_projection_identical'] and r['bypass_val_eligible']
                     and all(a['cold_bitwise'] for a in r['arms'].values()) for r in rows)):
        receipt['status']='DEV_FAIL_BYPASS_VAL_ELIGIBILITY';receipt['elapsed_seconds']=time.perf_counter()-start
        return receipt
    if time.perf_counter()-start>LIMIT_SECONDS:
        receipt['status']='DEV_FAIL_CPU_BUDGET';receipt['elapsed_seconds']=time.perf_counter()-start;return receipt
    # Only now materialize new and previously unconsumed authored OOD development data.
    ood=make_world('ood_dev')
    receipt['source_hashes']['ood_dev']=source_sha(ood)
    receipt['source_audits']['ood_dev']=audit(ood)
    if not receipt['source_audits']['ood_dev']['pass']:
        receipt['status']='DEV_FAIL_OOD_SOURCE_AUDIT';receipt['elapsed_seconds']=time.perf_counter()-start;return receipt
    wrong=wrong_credit(ood)
    receipt['ood_generated']=True
    for row,(core,trained,decoder) in zip(rows,models):
        f=p76.features(core,ood)
        native_val=value_features(core,ood)
        fb=evaluate(trained['bypass'],f,'bypass',decoder,native_val)
        fn=evaluate(trained['native'],f,'native')
        fr=evaluate(trained['raw'],f,'raw')
        negf=p76.features(core,wrong)
        wrong_b=evaluate(trained['bypass'],negf,'bypass')
        zero_ood=float(f['labels'].square().mean())
        better=1-fb['mse']/max(fn['mse'],1e-12)
        ci=paired_lower(fb['perworld'],fn['perworld'])
        raw_control=p76.fixed(f,'raw')
        passed=bool(fb['hit']>=.85 and fb['mse']<=.2*zero_ood and
                    fb['native_decoded_mse']<=.8*zero_ood and better>=.2 and ci>0 and
                    wrong_b['mse']>=2*max(fb['mse'],1e-5) and
                    raw_control['hit']>=.99 and raw_control['mse']<=1e-5)
        row['ood']={'bypass':strip_evaluation(fb),'native':strip_evaluation(fn),
                    'raw_train':strip_evaluation(fr),'raw_exact':strip_evaluation(raw_control),
                    'zero_mse':zero_ood,'wrong_credit_mse':wrong_b['mse'],
                    'relative_reduction_vs_native':better,'paired_lower_95':ci,'passed':passed}
    receipt['ood_scored']=True
    receipt['status']='DEV_PASS_IDENTITY_BYPASS_SIDE_CAR' if all(r['ood']['passed'] for r in rows) else 'DEV_FAIL_OOD_COMPARISON'
    receipt['elapsed_seconds']=time.perf_counter()-start
    receipt['original_hybrid_weights_trained']=False
    receipt['original_hybrid_outputs_scored']=False
    return receipt


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--receipt',type=Path,required=True)
    args=ap.parse_args()
    result=study();args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':result['status'],'ood_scored':result['ood_scored'],
                      'seconds':result.get('elapsed_seconds'),'science_pass':False}))


if __name__=='__main__':main()
