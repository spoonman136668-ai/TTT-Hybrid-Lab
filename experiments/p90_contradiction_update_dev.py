"""P90: prospective author-synthetic learned causal update; no OOD or native model changes.

Frozen before this source in GitHub commit a04c67295d43a886b63c0113ece137cdf7d65aa6.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from experiments import p86_portable_causal_preflight as p86
from experiments import p87_native_causal_falsifier as p87

SEEDS={'FIT':900111,'VAL':900113}
COUNTS={'FIT':256,'VAL':128}
MODEL_SEEDS=(900201,900203,900207)
ARMS=('constant','observation','contradiction')
STEPS,BATCH,CHECK=80,16,10
FREEZE='a04c67295d43a886b63c0113ece137cdf7d65aa6'
SOURCE_BLOBS={'model.py':'5d9ea57b827aa1796548447868fa5ee14c251b49',
              'experiments/p86_portable_causal_preflight.py':'008492665ee9090db5791963f2e2336ea283ee62'}


def source(split,n=None):
    if split not in SEEDS:raise ValueError('P90_PROTECTED_OR_UNKNOWN_SOURCE')
    n=COUNTS[split] if n is None else n
    if type(n) is not int or n<8 or n>COUNTS[split] or n%2:raise ValueError('P90_BAD_WORLD_COUNT')
    seed=SEEDS[split]
    changed=set(sorted(range(n),key=lambda w:p86.digest_int(seed,w,'regime',0,0,2**32))[:n//2])
    aa=[];ee=[];reg=[]
    for w in range(n):
        u0,v0=p86.basis(seed,w,'u0'),p86.basis(seed,w,'v0')
        u1,v1=p86.basis(seed,w,'u1'),p86.basis(seed,w,'v1')
        a=[];e=[]
        for t in range(49):
            ai=[p86.digest_int(seed,w,'action',t,j,193)-96 for j in range(16)]
            u,v=(u0,v0) if (t<24 or w not in changed) else (u1,v1)
            a.append([z/128 for z in ai]);e.append(p86.causal_effect(ai,u,v))
        aa.append(a);ee.append(e);reg.append(w in changed)
    return {'actions':torch.tensor(aa,dtype=torch.float32),
            'effects':torch.tensor(ee,dtype=torch.float32),
            'changed_world':torch.tensor(reg,dtype=torch.bool)}


def world_sha(w):
    return hashlib.sha256(b''.join(w[k].contiguous().numpy().tobytes()
                     for k in ('actions','effects','changed_world'))).hexdigest()


def p86_world(w):
    return {'history':torch.cat((w['actions'][:,:48],w['effects'][:,:48]),dim=-1),
            'query':torch.cat((w['actions'][:,48],torch.zeros_like(w['actions'][:,48])),dim=-1),
            'effect':w['effects'][:,48],'changed_world':w['changed_world']}


def initial(seed,arm):
    if arm not in ARMS:raise ValueError('P90_UNKNOWN_ARM')
    g=torch.Generator().manual_seed(seed)
    bias_eta=float(torch.randn((),generator=g)*.02)
    bias_forget=-2.9444389791664403+float(torch.randn((),generator=g)*.02)
    return nn.Parameter(torch.tensor([bias_eta,bias_forget]+([0.,0.] if arm!='constant' else []),dtype=torch.float32))


def rollout(w,arm,params=None,from_step=0,to_step=49,state=None):
    if arm not in (*ARMS,'fixed'):raise ValueError('P90_UNKNOWN_ARM')
    if arm!='fixed' and (params is None or params.numel()!=(2 if arm=='constant' else 4)):
        raise ValueError('P90_BAD_CONTROLLER_PARAMETERS')
    a=F.normalize(w['actions'],dim=-1);effect=w['effects'];batch=len(a)
    W=a.new_zeros((batch,16,16)) if state is None else state
    reads=[]
    for t in range(from_step,to_step):
        key=a[:,t];pred=torch.bmm(W,key.unsqueeze(-1)).squeeze(-1)
        reads.append(pred)
        if t>=48:continue
        residual=effect[:,t]-pred
        if arm=='fixed':
            eta=1.;decay=.99
        else:
            if arm=='constant':
                e_logit=params[0];f_logit=params[1]
            else:
                feature=effect[:,t] if arm=='observation' else residual
                feature=torch.sqrt(feature.square().mean(-1)+1e-9)
                feature=feature/(feature+.1)
                e_logit=params[0]+params[2]*feature
                f_logit=params[1]+params[3]*feature
            eta=2*torch.sigmoid(e_logit)
            decay=1-.2*torch.sigmoid(f_logit)
        if isinstance(eta,torch.Tensor) and eta.ndim:
            eta=eta[:,None,None];decay=decay[:,None,None]
        W=decay*W+eta*residual.unsqueeze(-1)*key.unsqueeze(1)
    return torch.stack(reads,dim=1),W


def mse(pred,w):
    e=(pred-w['effects']).square().mean(dim=-1)
    q=w['changed_world']
    return {'final':float(e[:,-1].mean()),'stable':float(e[~q,-1].mean()),
            'changed':float(e[q,-1].mean()),'late':float(e[:,32:].mean())}


def train(seed,arm,fit,val,indices,deadline):
    params=initial(seed,arm)
    initial_sha=hashlib.sha256(params.detach().numpy().tobytes()).hexdigest()
    optimizer=torch.optim.AdamW([params],lr=.05,weight_decay=0)
    best=float('inf');best_step=0;best_state=None;trace=[];started=time.perf_counter()
    for step in range(1,STEPS+1):
        if time.perf_counter()>deadline:raise TimeoutError('P90_CPU_BUDGET')
        ind=indices[step-1]
        batch={k:v[ind] for k,v in fit.items()}
        optimizer.zero_grad(set_to_none=True)
        pred,_=rollout(batch,arm,params)
        loss=(pred[:,32:]-batch['effects'][:,32:]).square().mean()
        if not torch.isfinite(loss):raise ValueError('P90_NONFINITE_LOSS')
        loss.backward();nn.utils.clip_grad_norm_([params],1.)
        optimizer.step()
        if step%CHECK==0:
            with torch.no_grad():
                pred,_=rollout(val,arm,params)
                s=mse(pred,val)
            if not all(math.isfinite(x) for x in s.values()):raise ValueError('P90_NONFINITE_VAL')
            trace.append({'step':step,'mse':s})
            if s['final']<best:
                best=s['final'];best_step=step;best_state=params.detach().clone()
    selected=best_state
    with torch.no_grad():prediction,_=rollout(val,arm,selected)
    return {'seed':seed,'arm':arm,'parameter_count':selected.numel(),'initial_sha256':initial_sha,
            'selected_sha256':hashlib.sha256(selected.numpy().tobytes()).hexdigest(),
            'learned_logits':[float(x) for x in selected],
            'selected_step':best_step,'validation':mse(prediction,val),'trace':trace,
            'train_seconds':time.perf_counter()-started,'num_updates':STEPS}


def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.perf_counter();deadline=start+150;root=Path(__file__).resolve().parents[1]
    o={'schema':'ttt-hybrid.p90-trained-256-float-causal-update-dev-result.v1','freeze_commit':FREEZE,
       'status':'NOT_STARTED','scientific_pass':False,'OOD_generated':False,'negative_generated':False,
       'original_hybrid_trained':False,'whole_process_compute_matched':False}
    o['custody']={p:{'expected':s,'actual':p86.blob(root/p)} for p,s in SOURCE_BLOBS.items()}
    if any(z['expected']!=z['actual'] for z in o['custody'].values()):
        o['status']='DEV_FAIL_ORIGINAL_SOURCE_CUSTODY';return o
    fit,val=source('FIT'),source('VAL')
    o['source']={split:{'sha256':world_sha(w),'n':len(w['actions']),'audit':p86.audit(p86_world(w))}
                 for split,w in [('FIT',fit),('VAL',val)]}
    if not all(z['audit']['pass'] for z in o['source'].values()):
        o['status']='DEV_FAIL_SOURCE_AUDIT';return o
    o['cold_restore_original_P78']=p87.exact_cold_restore(p86_world(val))
    if not o['cold_restore_original_P78']:
        o['status']='DEV_FAIL_COLD_RESTORE';return o
    fixed,state=rollout(val,'fixed')
    original=p86.online(p86_world(val),1.,.99)[0]
    o['W256_matches_P86']=bool(torch.allclose(fixed[:,-1],original,atol=1e-6,rtol=1e-6) and state.numel()==len(val['actions'])*256)
    zero=mse(torch.zeros_like(fixed),val);baseline=mse(fixed,val)
    wrong={k:v.clone() for k,v in val.items()}
    wrong['effects'][:,:48]=val['effects'][:,:48][:,torch.arange(48).roll(1)]
    corrupted,_=rollout(wrong,'fixed')
    wrong_score=float((corrupted[:,-1]-val['effects'][:,-1]).square().mean())
    o['controls']={'zero':zero,'P86_CAL_locked_fixed_256':baseline,'wrong_credit_final_mse':wrong_score,
       'wrong_credit_pass':bool(wrong_score>=1.5*baseline['final'] and wrong_score>=.01)}
    if not(o['W256_matches_P86'] and o['controls']['wrong_credit_pass']
       and baseline['stable']<=.8*zero['stable'] and baseline['changed']<=.8*zero['changed']):
        o['status']='DEV_FAIL_CONVENTIONAL_CONTROLS';return o
    ids=torch.randint(len(fit['actions']),(STEPS,BATCH),generator=torch.Generator().manual_seed(900299))
    o['sampler_sha256']=hashlib.sha256(ids.numpy().tobytes()).hexdigest()
    rows=[]
    for seed in MODEL_SEEDS:
        rows.append({'seed':seed,'models':{arm:train(seed,arm,fit,val,ids,deadline) for arm in ARMS}})
    o['rows']=rows
    o['frozen_gates']={}
    for r in rows:
        constant=r['models']['constant'];observation=r['models']['observation'];contradiction=r['models']['contradiction']
        def plateau(d):return all(t['mse']['final']<=1.05*d['validation']['final'] for t in d['trace'][-2:])
        r['conventional_qualified']=bool(constant['validation']['final']<=.9*zero['final'] and constant['selected_step']<STEPS and plateau(constant))
        r['contradiction_qualified']=bool(contradiction['validation']['final']<=.9*baseline['final']
            and contradiction['validation']['final']<=.9*constant['validation']['final']
            and contradiction['validation']['final']<=.95*observation['validation']['final']
            and contradiction['validation']['changed']<=.9*baseline['changed']
            and contradiction['validation']['stable']<=1.1*baseline['stable']
            and contradiction['selected_step']<STEPS and plateau(contradiction))
    o['frozen_gates']['all_conventional_qualified']=all(r['conventional_qualified'] for r in rows)
    o['frozen_gates']['all_contradiction_qualified']=all(r['contradiction_qualified'] for r in rows)
    if not o['frozen_gates']['all_conventional_qualified']:
        o['status']='DEV_FAIL_TRAINED_CONVENTIONAL_QUALIFICATION'
    elif not o['frozen_gates']['all_contradiction_qualified']:
        o['status']='DEV_FAIL_CONTRADICTION_VS_STRONG_CONTROLS'
    else:
        o['status']='DEV_FIT_VAL_ONLY_NOT_SCIENCE'
    o['elapsed_seconds']=time.perf_counter()-start
    return o


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();r=run();a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':r['status'],'seconds':r.get('elapsed_seconds'),
         'OOD_generated':r['OOD_generated'],'scientific_pass':r['scientific_pass']}))

if __name__=='__main__':main()
