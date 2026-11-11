"""P76 prospective frozen-native query/key alignment, author-synthetic development only.

Protocol commit d0b658ec0dc935a57e729d5e2da0bb479075675f predates source use.
All original HybridPilot weights frozen; no modification to original model or PR #184.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from model import HybridPilot, Configuration

D=16
MODEL_SEEDS=(760201,760203,760207)
SEEDS={'fit':760111,'val':760113,'ood_dev':760117,'negative':760119}
COUNTS={'fit':256,'val':192,'ood_dev':256,'negative':64}
UPDATES=240
BATCH=32
CHECK=40
TEMP=12.0


def tensor_hash(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def model_hash(m):
    return hashlib.sha256(b''.join(v.detach().contiguous().cpu().numpy().tobytes()
                for v in m.state_dict().values())).hexdigest()


def make_world(split, n=None):
    if split not in SEEDS: raise ValueError('P76_UNKNOWN_SOURCE')
    n=COUNTS[split] if n is None else n
    if type(n) is not int or not 1<=n<=COUNTS[split]:raise ValueError('P76_BAD_COUNT')
    gen=torch.Generator().manual_seed(SEEDS[split])
    q=torch.linalg.qr(torch.randn(n,D,D,generator=gen)).Q
    targets=q[:,:,:8].transpose(1,2).contiguous()
    values=torch.tanh(torch.randn(n,8,D,generator=gen))
    keys=targets.repeat(1,2,1)
    ys=values.repeat(1,2,1)
    distractors=32 if split=='ood_dev' else 8
    dk=F.normalize(torch.randn(n,distractors,D,generator=gen),dim=-1)
    dy=torch.tanh(torch.randn(n,distractors,D,generator=gen))
    keys=torch.cat((keys,dk),1)
    ys=torch.cat((ys,dy),1)
    roles=torch.cat((torch.ones(n,16,dtype=torch.bool),torch.zeros(n,distractors,dtype=torch.bool)),1)
    order=torch.rand(n,keys.shape[1],generator=gen).argsort(1)
    keys=keys.gather(1,order[:,:,None].expand_as(keys))
    ys=ys.gather(1,order[:,:,None].expand_as(ys))
    roles=roles.gather(1,order)
    chosen=torch.randint(8,(n,),generator=gen)
    query=targets[torch.arange(n),chosen]
    answer=values[torch.arange(n),chosen]
    return {'facts':torch.cat((keys,ys),-1),
            'query':torch.cat((query,torch.zeros_like(query)),-1)[:,None],
            'label':answer,'role_audit':roles}


def source_sha(w):
    return hashlib.sha256(b''.join(w[k].contiguous().cpu().numpy().tobytes() for k in
              ('facts','query','label','role_audit'))).hexdigest()


def audit(w):
    roles=w['role_audit'];n,steps=roles.shape
    if steps%4 or n<8: raise ValueError('P76_BAD_AUDIT_SHAPE')
    fq=roles.float().reshape(n,4,steps//4).mean((0,2))
    im=float((fq-roles.float().mean()).abs().max())
    train,held=roles[:n//2],roles[n//2:]
    pred=(train.float().mean(0)>=.5)[None].expand_as(held)
    ba=float(.5*((pred&held).sum()/held.sum().clamp_min(1) +
                    ((~pred)&(~held)).sum()/(~held).sum().clamp_min(1)))
    return {'max_quartile_imbalance':im,'index_only_balanced_accuracy':ba,
            'pass':bool(im<=.05 and ba<=.55)}


def factual(w):
    if set(w)!={'facts','query','label','role_audit'}:
        raise ValueError('P76_WORLD_SCHEMA')
    x=w['facts']
    if (x.ndim!=3 or x.shape[-1]!=32 or w['query'].shape!=(len(x),1,32) or
            not bool(torch.equal(w['query'][:,:,D:],torch.zeros_like(w['query'][:,:,D:])))):
        raise ValueError('P76_FACTUAL_SHAPE')
    return x


def labels_mask(world):
    facts=factual(world)
    keys=F.normalize(facts[:,:,:D],dim=-1)
    query=F.normalize(world['query'][:,0,:D],dim=-1)
    mask=(keys*query[:,None,:]).sum(-1)>.99999
    if not bool(mask.any(-1).all()): raise ValueError('P76_QUERY_NOT_OBSERVED')
    return mask


def original(seed):
    torch.manual_seed(seed)
    return HybridPilot(32,16,Configuration(width=16,heads=2,ssm=True,
                        attention=True,ttt=True,depth=True)).eval()


def features(core,world):
    facts=factual(world)
    with torch.no_grad():
        seq=torch.cat((facts,world['query']),1)
        # Original, byte-pinned causal feature production. Original parameters
        # and all historical factual key/value inputs are unchanged.
        pre=core.attn(core.ssm(core.stem(seq)))
        t=core.ttt.norm(pre)
        k=F.normalize(core.ttt.key(t[:,:-1]),dim=-1)
        q=F.normalize(core.ttt.query(t[:,-1]),dim=-1)
    raw_key=F.normalize(facts[:,:,:D],dim=-1)
    raw_query=F.normalize(world['query'][:,0,:D],dim=-1)
    return {'native_keys':k,'native_query':q,'raw_keys':raw_key,'raw_query':raw_query,
            'observed_effects':facts[:,:,D:].detach(), 'labels':world['label'].detach(),
            'positive':labels_mask(world).detach()}


class Alignment(nn.Module):
    """512-param supervised factual-address projection; no role or label input."""
    def __init__(self,seed):
        super().__init__()
        g=torch.Generator().manual_seed(seed)
        eye=torch.eye(D)
        self.key_map=nn.Parameter(eye+.1*torch.randn(D,D,generator=g))
        self.query_map=nn.Parameter(eye+.1*torch.randn(D,D,generator=g))

    def logits(self,keys,query):
        if keys.ndim!=3 or query.ndim!=2 or keys.shape[-1]!=D or query.shape!=(len(keys),D):
            raise ValueError('P76_BAD_PROJECTION_INPUT')
        k=F.normalize(keys@self.key_map,dim=-1)
        q=F.normalize(query@self.query_map,dim=-1)
        return TEMP*(k*q[:,None,:]).sum(-1)


def pairs(f,kind):
    if kind=='native':return f['native_keys'],f['native_query']
    if kind=='raw':return f['raw_keys'],f['raw_query']
    raise ValueError('P76_BAD_KIND')


def contrastive(logits,positive):
    # Multiple positive historical observations of the same executed action.
    if logits.shape!=positive.shape or not bool(positive.any(-1).all()):
        raise ValueError('P76_BAD_POSITIVE_MASK')
    return (torch.logsumexp(logits,-1)-
            torch.logsumexp(logits.masked_fill(~positive,float('-inf')),-1)).mean()


def evaluate(model,f,kind):
    with torch.no_grad():
        k,q=pairs(f,kind)
        logits=model.logits(k,q)
        ix=logits.argmax(-1)
        pred=f['observed_effects'][torch.arange(len(ix)),ix]
        hit=f['positive'][torch.arange(len(ix)),ix]
        perworld=(pred-f['labels']).square().mean(-1)
        return {'mse':float(perworld.mean()),'hit':float(hit.float().mean()),
                'loss':float(contrastive(logits,f['positive'])),'perworld':perworld,
                'weights_sha':model_hash(model)}


def fixed(f,kind):
    with torch.no_grad():
        k,q=pairs(f,kind)
        ix=(k*q[:,None,:]).sum(-1).argmax(-1)
        pred=f['observed_effects'][torch.arange(len(ix)),ix]
        hit=f['positive'][torch.arange(len(ix)),ix]
        perworld=(pred-f['labels']).square().mean(-1)
        return {'mse':float(perworld.mean()),'hit':float(hit.float().mean()),'perworld':perworld}


def train(seed,fv,fit_idx,kind):
    torch.set_num_threads(1)
    model=Alignment(seed)
    initial_sha=model_hash(model)
    opt=torch.optim.AdamW(model.parameters(),lr=.02,weight_decay=.01)
    fit,val=fv
    best=float('inf'); best_state=None;checks=[]
    for step in range(UPDATES):
        ix=fit_idx[step]
        k,q=pairs(fit,kind)
        opt.zero_grad(set_to_none=True)
        loss=contrastive(model.logits(k[ix],q[ix]),fit['positive'][ix])
        if not bool(torch.isfinite(loss)):raise ValueError('P76_NONFINITE_TRAIN')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if (step+1)%CHECK==0:
            with torch.no_grad():
                vk,vq=pairs(val,kind)
                vloss=float(contrastive(model.logits(vk,vq),val['positive']))
            checks.append({'step':step+1,'loss':vloss})
            if vloss<best:
                best=vloss;best_state=copy.deepcopy(model.state_dict())
    if best_state is None:raise ValueError('P76_NO_BEST_STATE')
    model.load_state_dict(best_state)
    cold=Alignment(seed+10000)
    cold.load_state_dict(copy.deepcopy(best_state))
    k,q=pairs(val,kind)
    cold_equal=bool(torch.equal(model.logits(k[:8],q[:8]),cold.logits(k[:8],q[:8])))
    return model,{'seed':seed,'kind':kind,'initial_sha':initial_sha,
                  'final_sha':model_hash(model),'checks':checks,
                  'params':sum(p.numel() for p in model.parameters()),
                  'best_val_contrastive':best,'cold_bitwise':cold_equal,
                  'val':{k:v for k,v in evaluate(model,val,kind).items() if k!='perworld'}}


def lower_relative(candidate,reference,seed=760401,reps=2000):
    ids=torch.randint(len(candidate),(reps,len(candidate)),generator=torch.Generator().manual_seed(seed))
    return float(torch.quantile(1-candidate[ids].mean(-1)/reference[ids].mean(-1).clamp_min(1e-9),.025))


def study():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    fit,val=make_world('fit'),make_world('val')
    source_audits={'fit':audit(fit),'val':audit(val)}
    out={'schema':'ttt-hybrid.p76-frozen-native-projection-dev-result.v1',
         'classification':'AUTHORED_DEV_NOT_ORIGINAL_TRAINED_MODEL_NOT_SCIENCE',
         'freeze_commit':'d0b658ec0dc935a57e729d5e2da0bb479075675f',
         'source_hashes':{'fit':source_sha(fit),'val':source_sha(val)},
         'source_audits':source_audits,'OOD_generated':False,'OOD_scored':False,
         'status':'NOT_EVALUATED','science_pass':False,'original_hybrid_changed':False,
         'full_regression':'NOT_RUN','gpu':'NOT_RUN','FORGE':'UNCHANGED',
         'training_budget':UPDATES}
    if not all(x['pass'] for x in source_audits.values()):
        out['status']='DEV_FAIL_SOURCE_AUDIT';return out
    sample_idx=torch.randint(len(fit['facts']),(UPDATES,BATCH),generator=torch.Generator().manual_seed(760299))
    out['sample_indices_sha']=tensor_hash(sample_idx)
    out['raw_fixed_val']=None
    rows=[];models=[]
    for seed in MODEL_SEEDS:
        core=original(seed);ori=model_hash(core)
        ff=features(core,fit);vf=features(core,val)
        zero=float(vf['labels'].square().mean())
        rawfixed=fixed(vf,'raw')
        baseline=fixed(vf,'native')
        if out['raw_fixed_val'] is None:out['raw_fixed_val']={'mse':rawfixed['mse'],'hit':rawfixed['hit']}
        if rawfixed['hit']<.99 or rawfixed['mse']>1e-5:
            out['status']='DEV_FAIL_RAW_BASELINE';return out
        trained,info=train(seed,(ff,vf),sample_idx,'native')
        raw,rawinfo=train(seed,(ff,vf),sample_idx,'raw')
        row={'seed':seed,'original_model_sha':ori,'original_model_frozen':ori==model_hash(core),
             'zero_val_mse':zero,'frozen_native_val':{'hit':baseline['hit'],'mse':baseline['mse']},
             'native_trained':info,'matched_raw_trained':rawinfo,
             'native_val_eligible':bool(info['val']['hit']>=.35 and info['val']['mse']<=.85*zero),
             'checkpoints_distinct_from_raw':info['final_sha']!=rawinfo['final_sha']}
        rows.append(row)
        models.append((core,trained))
    out['rows']=rows
    out['different_native_model_initializations']=len({r['original_model_sha'] for r in rows})==3
    out['different_aligner_initializations']=len({r['native_trained']['initial_sha'] for r in rows})==3
    if not (out['different_native_model_initializations'] and out['different_aligner_initializations'] and
       all(r['original_model_frozen'] and r['native_trained']['cold_bitwise'] and
           r['matched_raw_trained']['cold_bitwise'] and r['native_val_eligible'] for r in rows)):
        out['status']='DEV_FAIL_NATIVE_VAL_QUALIFICATION';out['seconds']=time.perf_counter()-start
        return out
    if time.perf_counter()-start>300:
        out['status']='DEV_FAIL_CPU_BUDGET';out['seconds']=time.perf_counter()-start;return out
    ood=make_world('ood_dev')
    out['source_hashes']['ood_dev']=source_sha(ood)
    out['source_audits']['ood_dev']=audit(ood)
    if not out['source_audits']['ood_dev']['pass']:
        out['status']='DEV_FAIL_OOD_SOURCE_AUDIT';out['seconds']=time.perf_counter()-start;return out
    wrong={k:v.clone() for k,v in ood.items()}
    gen=torch.Generator().manual_seed(760119)
    perm=torch.rand(len(ood['facts']),ood['facts'].shape[1],generator=gen).argsort(1)
    wrong['facts'][:,:,D:]=wrong['facts'][:,:,D:].gather(1,perm[:,:,None].expand(len(perm),48,D))
    for row,(core,trained) in zip(rows,models):
        of=features(core,ood)
        og=evaluate(trained,of,'native')
        bf=fixed(of,'native')
        raw=fixed(of,'raw')
        wrongf=features(core,wrong)
        wrongret=evaluate(trained,wrongf,'native')
        zero=float(of['labels'].square().mean())
        improve=1-og['mse']/max(bf['mse'],1e-10)
        row['ood']={'candidate_hit':og['hit'],'candidate_mse':og['mse'],
                    'frozen_native_hit':bf['hit'],'frozen_native_mse':bf['mse'],
                    'raw_fixed_hit':raw['hit'],'raw_fixed_mse':raw['mse'],
                    'zero_mse':zero,'incorrect_credit_mse':wrongret['mse'],
                    'improvement_fraction':improve,
                    'bootstrap_lower_relative_vs_frozen':lower_relative(og['perworld'],bf['perworld']),
                    'passed':bool(og['hit']>=.50 and og['mse']<=.70*zero and improve>=.20
                         and lower_relative(og['perworld'],bf['perworld'])>0 and
                         wrongret['mse']>=2*og['mse'] and raw['hit']>=.99 and raw['mse']<=1e-5)}
    out['OOD_generated']=True;out['OOD_scored']=True
    out['status']='DEV_PASS_NATIVE_ALIGNER_TRANSFER' if all(r['ood']['passed'] for r in rows) else 'DEV_FAIL_NATIVE_ALIGNER_TRANSFER'
    out['seconds']=time.perf_counter()-start
    out['whole_process_compute_matched']=False
    out['original_hybrid_weights_trained']=False
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--receipt',required=True,type=Path)
    a=p.parse_args()
    result=study()
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'OOD_scored':result['OOD_scored'],
                      'seconds':result.get('seconds'),'science_pass':False}),flush=True)

if __name__=='__main__': main()
