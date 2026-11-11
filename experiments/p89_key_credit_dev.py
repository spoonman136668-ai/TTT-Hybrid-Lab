"""P89 frozen-source native action-key alignment (author-synthetic DEV only).

Freeze 750dd41bbd66e372ae13b4e1b2a1246f6f9a9d9e before any P89 source.
No independent scientific first-use, original model edits or OOD generation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from experiments import p86_portable_causal_preflight as p86
from experiments import p87_native_causal_falsifier as p87

D=16
SEEDS={'FIT':890111,'VAL':890113}
N={'FIT':128,'VAL':64}
MODEL_SEEDS=(890201,890203,890207)
FREEZE='750dd41bbd66e372ae13b4e1b2a1246f6f9a9d9e'
ORIGINAL={'model.py':'5d9ea57b827aa1796548447868fa5ee14c251b49',
 'experiments/p86_portable_causal_preflight.py':'008492665ee9090db5791963f2e2336ea283ee62',
 'experiments/p87_native_causal_falsifier.py':'18713cdd7c04f0c22ec33dad7d2f158b6fd59c24'}
ARMS=('raw_direct','stem_shared','context_shared','context_mismatched','ridge_recovered_action')


def source(split,n=None):
    if split not in SEEDS:raise ValueError('P89_PROTECTED_OR_UNKNOWN_SOURCE')
    n=N[split] if n is None else n
    if type(n) is not int or n<8 or n>N[split] or n%2:raise ValueError('P89_BAD_SOURCE_SIZE')
    seed=SEEDS[split]
    changed=set(sorted(range(n),key=lambda w:p86.digest_int(seed,w,'regime',0,0,2**32))[:n//2])
    aa=[];ee=[];reg=[]
    for w in range(n):
        u0,v0=p86.basis(seed,w,'u0'),p86.basis(seed,w,'v0')
        u1,v1=p86.basis(seed,w,'u1'),p86.basis(seed,w,'v1')
        a=[];e=[]
        for t in range(49):
            action=[p86.digest_int(seed,w,'action',t,j,193)-96 for j in range(D)]
            u,v=(u0,v0) if (t<24 or w not in changed) else (u1,v1)
            a.append([z/128 for z in action]);e.append(p86.causal_effect(action,u,v))
        aa.append(a);ee.append(e);reg.append(w in changed)
    return {'actions':torch.tensor(aa,dtype=torch.float32),
            'effects':torch.tensor(ee,dtype=torch.float32),
            'changed_world':torch.tensor(reg,dtype=torch.bool)}


def digest(w):
    return hashlib.sha256(b''.join(w[k].contiguous().numpy().tobytes()
            for k in ('actions','effects','changed_world'))).hexdigest()


def p86_world(w):
    a,e=w['actions'],w['effects']
    return {'history':torch.cat((a[:,:48],e[:,:48]),dim=-1),
      'query':torch.cat((a[:,48],torch.zeros_like(a[:,48])),dim=-1),
      'effect':e[:,48],'changed_world':w['changed_world']}


def paired(w):
    a,e=w['actions'],w['effects']
    x=a.new_zeros((len(a),97,32))
    x[:,::2,:16]=a
    x[:,1::2,:16]=a[:,:48]
    x[:,1::2,16:]=e[:,:48]
    return x


def features(model,w):
    with torch.no_grad():
        stream=paired(w)
        stem=model.stem(stream)
        z=model.attn(model.ssm(stem))
        def qmap(x):return F.normalize(model.ttt.query(model.ttt.norm(x)),dim=-1)
        return {'stem':qmap(stem[:,::2]),
          'context':qmap(z[:,::2]),
          'native_fact':F.normalize(model.ttt.key(model.ttt.norm(z[:,1::2])),dim=-1),
          'context_pre':z[:,::2]}


def ridge(fit_features,fit_actions):
    x=fit_features.reshape(-1,16).double()
    y=F.normalize(fit_actions.reshape(-1,16),dim=-1).double()
    X=torch.cat((x,torch.ones((len(x),1),dtype=torch.float64)),dim=1)
    penalize=torch.eye(17,dtype=torch.float64)*.01
    penalize[-1,-1]=0
    return torch.linalg.solve(X.T@X+penalize,X.T@y)


def restored_key(z,beta):
    X=torch.cat((z.double(),torch.ones((*z.shape[:-1],1),dtype=torch.float64)),dim=-1)
    return F.normalize((X@beta).float(),dim=-1)


def online(w,keys,queries,values=None):
    if keys.shape!=(len(w['actions']),48,16) or queries.shape!=(len(w['actions']),49,16):
        raise ValueError('P89_BAD_KEY_QUERY_SHAPE')
    obs=w['effects'][:,:48] if values is None else values
    state=torch.zeros((len(obs),16,16),dtype=torch.float32)
    predictions=[]
    for t in range(49):
        q=F.normalize(queries[:,t],dim=-1)
        predictions.append(torch.bmm(state,q.unsqueeze(-1)).squeeze(-1))
        if t<48:
            k=F.normalize(keys[:,t],dim=-1)
            err=obs[:,t]-torch.bmm(state,k.unsqueeze(-1)).squeeze(-1)
            state=.99*state+err.unsqueeze(-1)*k.unsqueeze(1)
    return torch.stack(predictions,1),state


def scores(pred,w):
    err=(pred-w['effects']).square().mean(-1)
    late=err[:,32:49].mean(-1);last=err[:,48];r=w['changed_world']
    return {'final':float(last.mean()),'stable':float(last[~r].mean()),
         'changed':float(last[r].mean()),'late':float(late.mean())}


def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    t0=time.perf_counter();root=Path(__file__).resolve().parents[1]
    result={'schema':'ttt-hybrid.p89-action-key-credit-frozen-dev-result.v1',
     'freeze_commit':FREEZE,'status':'NOT_RUN','science_pass':False,
     'OOD_generated':False,'full_model_trained':False,'whole_process_cost_matched':False}
    result['custody']={p:{'expected':sha,'actual':p86.blob(root/p)} for p,sha in ORIGINAL.items()}
    if any(x['expected']!=x['actual'] for x in result['custody'].values()):
        result['status']='DEV_FAIL_SOURCE_CUSTODY';return result
    fit,val=source('FIT'),source('VAL')
    result['source']={k:{'sha256':digest(w),'audit':p86.audit(p86_world(w)),'n':len(w['actions'])}
                      for k,w in [('FIT',fit),('VAL',val)]}
    if not all(x['audit']['pass'] for x in result['source'].values()):
        result['status']='DEV_FAIL_SOURCE_AUDIT';return result
    result['cold_restore_P78']=p87.exact_cold_restore(p86_world(val))
    if not result['cold_restore_P78']:
        result['status']='DEV_FAIL_COLD';return result
    aa=F.normalize(val['actions'],dim=-1)
    raw,state=online(val,aa[:,:48],aa)
    zero=scores(torch.zeros_like(raw),val)
    fixed=scores(raw,val)
    # Explicit reference: direct original P86 eta1 decay.99 final effect.
    p86_pred=p86.online(p86_world(val),1.,.99)[0]
    result['reference_parity']=bool(torch.allclose(raw[:,-1],p86_pred,atol=1e-6,rtol=1e-6)
      and state.numel()==len(aa)*256)
    perm=torch.arange(48).roll(1)
    wrong,_=online(val,aa[:,:48],aa,values=val['effects'][:,:48][:,perm])
    wrong_score=scores(wrong,val)
    result['controls']={'fixed':fixed,'zero':zero,'wrong_credit':wrong_score,
       'wrong_credit_pass':bool(wrong_score['final']>=1.5*fixed['final'] and wrong_score['final']>=.01)}
    if not (result['reference_parity'] and result['controls']['wrong_credit_pass']
       and fixed['stable']<=.8*zero['stable'] and fixed['changed']<=.8*zero['changed']):
        result['status']='DEV_FAIL_STRONG_CONTROLS';return result
    rows=[]
    for seed in MODEL_SEEDS:
        model=p87.new_model('hybrid',seed).eval()
        before=p87.fingerprint(model)
        started=time.perf_counter()
        ff=features(model,fit);fv=features(model,val)
        beta=ridge(ff['context_pre'],fit['actions'])
        reconstructed=restored_key(fv['context_pre'],beta)
        options={
          'raw_direct':aa,
          'stem_shared':fv['stem'],
          'context_shared':fv['context'],
          'ridge_recovered_action':reconstructed}
        evals={}
        for arm,qs in options.items():
            pred,w=online(val,qs[:,:48],qs)
            evals[arm]=scores(pred,val)
            assert w.numel()==len(val['actions'])*256
        pred,w=online(val,fv['native_fact'],fv['context'])
        evals['context_mismatched']=scores(pred,val)
        assert before==p87.fingerprint(model)
        rows.append({'seed':seed,'original_model_SHA256':before,
           'action_ridge_SHA256':hashlib.sha256(beta.numpy().tobytes()).hexdigest(),
           'trainable_extra_ridge_coefficients':int(beta.numel()),
           'scores':evals,'cpu_seconds':time.perf_counter()-started})
    result['rows']=rows
    result['distinct_original_models']=len({r['original_model_SHA256'] for r in rows})==3
    result['frozen_signatures']={
      'mismatched_key_failure_three_of_three':all(r['scores']['context_mismatched']['final']>=2*fixed['final']
           and r['scores']['context_shared']['final']<=.85*r['scores']['context_mismatched']['final'] for r in rows),
      'context_mixing_harm_three_of_three':all(r['scores']['context_shared']['final']>=1.5*r['scores']['stem_shared']['final'] for r in rows),
      'FIT_only_action_ridge_recovery_three_of_three':all(r['scores']['ridge_recovered_action']['final']<=1.20*fixed['final'] for r in rows)}
    result['status']='DEV_MECHANISTIC_DIAGNOSTIC_ONLY' if result['distinct_original_models'] and any(result['frozen_signatures'].values()) else 'DEV_INCONCLUSIVE_OR_NEGATIVE'
    result['elapsed_seconds']=time.perf_counter()-t0
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',type=Path,required=True)
    a=parser.parse_args();r=run();a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':r['status'],'signatures':r.get('frozen_signatures'),
         'elapsed_seconds':r.get('elapsed_seconds'),'scientific_pass':False}))

if __name__=='__main__':main()
