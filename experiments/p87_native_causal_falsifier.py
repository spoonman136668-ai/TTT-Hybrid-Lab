"""P87 full original HybridPilot causal learning vs independent GRU/SSM and fixed P86.

Freeze BEFORE code/source birth: ab92a50e4c27261b0aa02aa31fd4263b17980ae7.
Author-synthetic DEV only, no external scientific first use. Holdouts fail closed.
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
from experiments import p86_portable_causal_preflight as p86
from model import HybridPilot, Configuration, SelectiveStateBlock
from native_p78_bounded_raw_ttt import BoundedRawTTT

D=16
SEEDS={'FIT':870111,'VAL':870113}
COUNTS={'FIT':384,'VAL':128}
RESERVED={'OOD':870117,'NEGATIVE':870119}
MODEL_SEEDS=(870201,870203,870207)
STEPS,BATCH,CHECK=400,16,40
FREEZE='ab92a50e4c27261b0aa02aa31fd4263b17980ae7'
EXPECTED_BLOBS={'model.py':'5d9ea57b827aa1796548447868fa5ee14c251b49',
  'native_p78_bounded_raw_ttt.py':'db948c53bb1a28af5fe0e6a62ef71463db6d95ba',
  'experiments/p86_portable_causal_preflight.py':'008492665ee9090db5791963f2e2336ea283ee62'}
ARMS=('hybrid','gru','ssm')


def source(split, n=None):
    if split not in SEEDS: raise ValueError('P87_PROTECTED_OR_UNKNOWN_SOURCE')
    n=COUNTS[split] if n is None else n
    if type(n) is not int or n<8 or n>COUNTS[split] or n%2:
        raise ValueError('P87_INVALID_SOURCE_COUNT')
    seed=SEEDS[split]
    switched=set(sorted(range(n),key=lambda w:p86.digest_int(seed,w,'regime',0,0,2**32))[:n//2])
    histories=[]; queries=[]; targets=[]; roles=[]
    for w in range(n):
        u0,v0=p86.basis(seed,w,'u0'),p86.basis(seed,w,'v0')
        u1,v1=p86.basis(seed,w,'u1'),p86.basis(seed,w,'v1')
        changed=w in switched
        hist=[]
        for t in range(49):
            ai=[p86.digest_int(seed,w,'action',t,j,193)-96 for j in range(D)]
            action=[x/128 for x in ai]
            u,v=(u0,v0) if (t<24 or not changed) else (u1,v1)
            outcome=p86.causal_effect(ai,u,v)
            if t<48: hist.append(action+outcome)
            else:
                queries.append(action+[0.]*D)
                targets.append(outcome)
        histories.append(hist);roles.append(changed)
    return {'history':torch.tensor(histories,dtype=torch.float32),
            'query':torch.tensor(queries,dtype=torch.float32),
            'effect':torch.tensor(targets,dtype=torch.float32),
            'changed_world':torch.tensor(roles,dtype=torch.bool)}


def checksum(w):return p86.world_sha(w)

def input_tensor(w): return torch.cat((w['history'], w['query'][:,None,:]),dim=1)

class GRUBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru=nn.GRU(input_size=32,hidden_size=16,num_layers=1,batch_first=True)
        self.head=nn.Linear(16,16)
    def forward(self,x):
        h,_=self.gru(x)
        return self.head(h[:,-1,:])

class SSMBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Linear(32,16)
        self.ssm=SelectiveStateBlock(16)
        self.head=nn.Linear(16,16)
    def forward(self,x):
        return self.head(self.ssm(self.stem(x))[:,-1,:])


def new_model(arm,seed):
    if arm not in ARMS:raise ValueError('P87_UNKNOWN_MODEL')
    torch.manual_seed(seed)
    if arm=='hybrid':return HybridPilot(32,16,Configuration(width=16,heads=2,ssm=True,attention=True,ttt=True,depth=True))
    if arm=='gru':return GRUBaseline()
    return SSMBaseline()


def fingerprint(model):
    return hashlib.sha256(b''.join(p.detach().cpu().contiguous().numpy().tobytes()
            for p in model.state_dict().values())).hexdigest()


def mse_by_regime(y,label,changed):
    e=(y-label).square().mean(-1)
    return {'combined':float(e.mean()),'stable':float(e[~changed].mean()),
            'changed':float(e[changed].mean())}


def exact_cold_restore(w):
    t=BoundedRawTTT(max_events=48)
    f=w['history'][:8]
    init=t.initial_state(f[:,0,:])
    mask=torch.ones((8,48),dtype=torch.bool)
    whole=t.ingest_confirmed(f,init,mask)
    partial=t.ingest_confirmed(f[:,:24,:],init,mask[:,:24])
    saved=type(partial)(partial.fast_weights.clone(),partial.confirmed_count.clone(),partial.policy_tag)
    resume=t.ingest_confirmed(f[:,24:,:],saved,mask[:,24:])
    return (torch.equal(whole.fast_weights,resume.fast_weights) and
            torch.equal(whole.confirmed_count,resume.confirmed_count) and
            torch.equal(t.read_only(w['query'][:8,:D],whole),t.read_only(w['query'][:8,:D],resume))
            and whole.fast_weights.numel()==8*256)


def score_model(model,x,w):
    model.eval()
    with torch.no_grad():
        y=model(x)
        if y.ndim==3:y=y[:,-1,:]
        return mse_by_regime(y,w['effect'],w['changed_world'])


def train(arm,seed,x,y,valx,val,indices,deadline):
    model=new_model(arm,seed)
    before=fingerprint(model)
    params=list(model.parameters())
    opt=torch.optim.AdamW(params,lr=.002,weight_decay=.01)
    best=float('inf');best_step=None;best_weights=None;traces=[]
    start=time.perf_counter()
    model.train()
    for step in range(1,STEPS+1):
        if time.perf_counter()>=deadline:raise TimeoutError('P87_TRAINING_BUDGET')
        ix=indices[step-1]
        opt.zero_grad(set_to_none=True)
        pred=model(x[ix]);pred=pred[:,-1,:] if pred.ndim==3 else pred
        loss=(pred-y[ix]).square().mean()
        if not bool(torch.isfinite(loss)):raise ValueError('P87_NONFINITE_LOSS')
        loss.backward()
        gradient=nn.utils.clip_grad_norm_(params,1.0)
        if not bool(torch.isfinite(gradient)):raise ValueError('P87_NONFINITE_GRADIENT')
        opt.step()
        if step%CHECK==0:
            score=score_model(model,valx,val)
            if not all(math.isfinite(z) for z in score.values()):raise ValueError('P87_NONFINITE_VAL')
            traces.append({'step':step,'mse':score})
            if score['combined']<best:
                best=score['combined'];best_step=step;best_weights=copy.deepcopy(model.state_dict())
            model.train()
    model.load_state_dict(best_weights)
    best_val=score_model(model,valx,val)
    return model,{'arm':arm,'seed':seed,'initial_sha256':before,'best_sha256':fingerprint(model),
       'params_trainable':sum(t.numel() for t in params),
       'best_step':best_step,'best_VAL_mse':best_val,'validation_trace':traces,
       'train_seconds':time.perf_counter()-start,'num_updates':STEPS}


def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start=time.perf_counter();deadline=start+160
    result={'schema':'ttt-hybrid.p87-native-causal-falsifier-author-dev-result.v1',
      'freeze_commit':FREEZE,'status':'NOT_RUN','scientific_pass':False,
      'OOD_generated':False,'negative_generated':False,'full_hybrid_trained':False,
      'whole_process_compute_matched':False}
    root=Path(__file__).resolve().parents[1]
    result['source_blobs']={k:{'expected':v,'actual':p86.blob(root/k)} for k,v in EXPECTED_BLOBS.items()}
    if any(v['actual']!=v['expected'] for v in result['source_blobs'].values()):
        result['status']='DEV_FAIL_SOURCE_BLOBS';return result
    fit,val=source('FIT'),source('VAL')
    result['cohorts']={'FIT':{'n':len(fit['effect']),'sha256':checksum(fit),'audit':p86.audit(fit)},
                       'VAL':{'n':len(val['effect']),'sha256':checksum(val),'audit':p86.audit(val)}}
    if not all(t['audit']['pass'] for t in result['cohorts'].values()):
        result['status']='DEV_FAIL_COHORT';return result
    result['cold_restore_bitwise_P78']=exact_cold_restore(val)
    if not result['cold_restore_bitwise_P78']:
        result['status']='DEV_FAIL_COLD_RESTORE';return result
    zero=mse_by_regime(torch.zeros_like(val['effect']),val['effect'],val['changed_world'])
    fixed=p86.error(p86.online(val,1.,.99)[0],val)
    original=p86.error(p86.p78_original(val)[0],val)
    ols=p86.error(p86.last24_ols(val),val)
    result['controls']={'zero':zero,'P86_CAL_frozen_eta1_decay099':fixed,
         'P78_original_eta05_decay1':original,'last24_OLS_unmatched':ols}
    strong=bool(fixed['stable']<=.75*zero['stable'] and fixed['changed']<=.80*zero['changed'])
    if not strong:
        result['status']='DEV_FAIL_STRONG_FIXED_CONVENTIONAL';return result
    g=torch.Generator().manual_seed(870299)
    ix=torch.randint(len(fit['effect']),(STEPS,BATCH),generator=g)
    result['sampler_sha256']=hashlib.sha256(ix.numpy().tobytes()).hexdigest()
    x,y,valx=input_tensor(fit),fit['effect'],input_tensor(val)
    rows=[]
    for seed in MODEL_SEEDS:
        row={'seed':seed,'models':{}}
        for arm in ARMS:
            m,meta=train(arm,seed,x,y,valx,val,ix,deadline)
            row['models'][arm]=meta
        rows.append(row)
    result['rows']=rows
    result['full_hybrid_trained']=all('hybrid' in r['models'] for r in rows)
    all_baseline=True;all_hybrid=True
    for r in rows:
        r['eligible']={}
        for arm in ARMS:
            z=r['models'][arm]
            best=z['best_VAL_mse']; last=z['validation_trace'][-2:]
            plateau=all(q['mse']['combined']<=1.10*best['combined'] for q in last)
            good=bool(best['combined']<=.75*zero['combined'] and z['best_step']<=360 and plateau)
            if arm in ('gru','ssm'):
                r['eligible'][arm]={'pass':good,'plateau':plateau,'nonterminal':z['best_step']<=360}
                all_baseline=all_baseline and good
            else:
                threshold=min(fixed['combined'], r['models']['gru']['best_VAL_mse']['combined'],
                              r['models']['ssm']['best_VAL_mse']['combined'])
                g2=bool(best['combined']<=.9*threshold and best['changed']<=.9*fixed['changed'] and
                        z['best_step']<=360 and plateau)
                r['eligible'][arm]={'pass':g2,'plateau':plateau,'nonterminal':z['best_step']<=360,
                                    'qualified_strongest_mse':threshold}
                all_hybrid=all_hybrid and g2
    result['frozen_eligibility']={'trained_gru_ssm_all_qualified':all_baseline,
                                  'original_hybrid_all_qualified':all_hybrid}
    if not all_baseline:
        result['status']='DEV_FAIL_TRAINED_CONVENTIONAL_UNQUALIFIED'
    elif not all_hybrid:
        result['status']='DEV_FAIL_ORIGINAL_FULL_HYBRID_VS_STRONG_CONTROLS'
    else:
        result['status']='DEV_FIT_VAL_ONLY_PASS_PROTECTED_FIRST_USE_REQUIRED'
    result['elapsed_seconds']=time.perf_counter()-start
    # STOP before protected OOD/negative even when VAL passes; separate owner authority needed.
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();r=run();a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'status':r['status'],'full_hybrid_trained':r['full_hybrid_trained'],
      'OOD_generated':r['OOD_generated'],'scientific_pass':r['scientific_pass'],
      'elapsed_seconds':r.get('elapsed_seconds')}))

if __name__=='__main__':main()
