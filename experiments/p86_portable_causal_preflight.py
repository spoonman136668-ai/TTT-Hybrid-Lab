"""P86 platform-identical integer-action causal source, authored DEV baseline only.

Pre-source freeze commit 2050098d4548fc332b1360af7ff79ed41c692fa3.
The reserved heldout and negative source APIs are intentionally unavailable.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from native_p78_bounded_raw_ttt import BoundedRawTTT

D=16
SEEDS={'FIT':860111,'CAL':860113}
N={'FIT':256,'CAL':128}
RESERVED={'HELDOUT':860117,'NEGATIVE':860119}
RATES=(.25,.5,1.,1.5,2.)
DECAYS=(1.,.99,.97,.95,.90)
FREEZE='2050098d4548fc332b1360af7ff79ed41c692fa3'
ORIGINAL_BLOBS={'model.py':'5d9ea57b827aa1796548447868fa5ee14c251b49',
                'native_p78_bounded_raw_ttt.py':'db948c53bb1a28af5fe0e6a62ef71463db6d95ba'}


def blob(path):
    x=Path(path).read_bytes()
    return hashlib.sha1(b'blob '+str(len(x)).encode()+b'\0'+x).hexdigest()


def digest_int(seed,world,tag,i,j=0,mod=1):
    key=f'p86-portable-v1|{seed}|{world}|{tag}|{i}|{j}'.encode('ascii')
    return int.from_bytes(hashlib.sha256(key).digest()[:4],'little') % mod


def basis(seed,w,tag):
    return [[digest_int(seed,w,tag,j,k,7)-3 for k in range(4)] for j in range(D)]


def causal_effect(action_int,u,v):
    hidden=[sum(action_int[j]*u[j][k] for j in range(D)) for k in range(4)]
    y=[sum(hidden[k]*v[d][k] for k in range(4)) for d in range(D)]
    assert max(abs(z) for z in y)<2**24
    return [z/16384 for z in y]


def source(split,n=None):
    if split not in SEEDS:raise ValueError('P86_PROTECTED_OR_UNKNOWN_SOURCE')
    n=N[split] if n is None else n
    if type(n) is not int or n<8 or n>N[split] or n%2:
        raise ValueError('P86_INVALID_WORLD_COUNT')
    seed=SEEDS[split]
    switched=set(sorted(range(n),key=lambda w:digest_int(seed,w,'regime',0,0,2**32))[:n//2])
    facts=[];query=[];targets=[];regime=[]
    for w in range(n):
        u0,v0=basis(seed,w,'u0'),basis(seed,w,'v0')
        u1,v1=basis(seed,w,'u1'),basis(seed,w,'v1')
        changed=w in switched
        events=[]
        for t in range(49):
            ai=[digest_int(seed,w,'action',t,j,193)-96 for j in range(D)]
            a=[z/128 for z in ai]
            if t<24 or not changed: u,v=u0,v0
            else: u,v=u1,v1
            effects=causal_effect(ai,u,v)
            if t<48:events.append(a+effects)
            else:
                query.append(a+[0.]*D)
                targets.append(effects)
        facts.append(events)
        regime.append(changed)
    return {'history':torch.tensor(facts,dtype=torch.float32),
            'query':torch.tensor(query,dtype=torch.float32),
            'effect':torch.tensor(targets,dtype=torch.float32),
            'changed_world':torch.tensor(regime,dtype=torch.bool)}


def world_sha(w):
    return hashlib.sha256(b''.join(w[k].contiguous().cpu().numpy().tobytes()
            for k in ('history','query','effect','changed_world'))).hexdigest()


def audit(w):
    b=len(w['history'])
    unique=((F.normalize(w['history'][:,:,:D],dim=-1)*
          F.normalize(w['query'][:,None,:D],dim=-1)).sum(-1)).abs().max().item()
    no_future=bool(torch.count_nonzero(w['query'][:,D:])==0)
    finite=all(bool(torch.isfinite(w[k]).all()) for k in ('history','query','effect'))
    shape=w['history'].shape==(b,48,32) and w['query'].shape==(b,32)
    half=bool(int(w['changed_world'].sum())==b//2)
    nonzero=bool((w['history'][:,:,:D].norm(dim=-1)>1e-8).all())
    return {'pass':bool(unique<.999 and no_future and finite and shape and half and nonzero),
            'n':b,'changed':int(w['changed_world'].sum()),'max_abs_query_history_cosine':float(unique),
            'future_effect_hidden':no_future,'finite':finite,'shape':shape}


def online(w,eta,decay):
    x=F.normalize(w['history'][:,:,:D],dim=-1)
    y=w['history'][:,:,D:]
    W=torch.zeros(len(x),D,D,dtype=torch.float32)
    for t in range(48):
        k=x[:,t]
        error=y[:,t]-torch.bmm(W,k[:,:,None]).squeeze(-1)
        W=decay*W+eta*error[:,:,None]*k[:,None,:]
    result=torch.bmm(W,F.normalize(w['query'][:,:D],dim=-1)[:,:,None]).squeeze(-1)
    return result,W


def p78_original(w):
    m=BoundedRawTTT(max_events=48)
    z,state=m.read_from_episode(w['history'],w['query'][:,:D])
    return z,state.fast_weights


def last24_ols(w):
    x=w['history'][:,-24:,:D].double()
    y=w['history'][:,-24:,D:].double()
    ridge=torch.eye(D,dtype=torch.float64)[None]*1e-4
    beta=torch.linalg.solve(x.transpose(1,2)@x+ridge,x.transpose(1,2)@y)
    return torch.bmm(w['query'][:,None,:D].double(),beta).squeeze(1).float()


def nearest(w):
    x=w['history'][:,:,:D]
    q=w['query'][:,:D]
    scores=(F.normalize(x,dim=-1)*F.normalize(q[:,None,:],dim=-1)).sum(-1)
    best=scores.argmax(-1)
    return w['history'][torch.arange(len(best)),best,D:]


def error(pred,w):
    e=(pred-w['effect']).square().mean(-1)
    return {'combined':float(e.mean()),'stable':float(e[~w['changed_world']].mean()),
            'changed':float(e[w['changed_world']].mean())}


def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    t0=time.perf_counter()
    out={'schema':'ttt-hybrid.p86-portable-source-admission-dev.v1','freeze_commit':FREEZE,
         'status':'NOT_RUN','heldout_generated':False,'negative_generated':False,
         'original_hybrid_trained':False,'science_pass':False}
    root=Path(__file__).resolve().parents[1]
    out['custody']={k:{'actual':blob(root/k),'expected':v} for k,v in ORIGINAL_BLOBS.items()}
    if any(x['actual']!=x['expected'] for x in out['custody'].values()):
        out['status']='DEV_FAIL_SOURCE_CUSTODY';return out
    fit,cal=source('FIT'),source('CAL')
    out['source']={'sha256':{'FIT':world_sha(fit),'CAL':world_sha(cal)},
                   'audit':{'FIT':audit(fit),'CAL':audit(cal)},'seeds':SEEDS,'n':N}
    if not all(z['pass'] for z in out['source']['audit'].values()):
        out['status']='DEV_FAIL_SOURCE_AUDIT';return out
    zero=error(torch.zeros_like(cal['effect']),cal)
    original,origin_state=p78_original(cal)
    original_metrics=error(original,cal)
    rows=[];winner=None
    for eta in RATES:
        for decay in DECAYS:
            pred,W=online(cal,eta,decay)
            if W.numel()!=len(cal['history'])*256:raise ValueError('P86_PERSISTENT_MEMORY_BUDGET')
            row={'eta':eta,'decay':decay,'mse':error(pred,cal),'memory_floats':256}
            rows.append(row)
            if winner is None or row['mse']['combined']<winner['mse']['combined']:
                winner=row
    matched,control=online(cal,.5,1.)
    parity=bool(torch.allclose(original,matched,atol=1e-7,rtol=1e-7) and
                torch.allclose(origin_state,control,atol=1e-7,rtol=1e-7))
    ols=error(last24_ols(cal),cal)
    nn=error(nearest(cal),cal)
    out['CAL']={'zero':zero,'original_p78':original_metrics,'grid':rows,'selected':winner,
                'P78_exact_state_parity':parity,'OLS_last24_unmatched_768_history':ols,
                'nearest_unmatched_1536_history':nn}
    passed=bool(parity and len(rows)==25 and
        winner['mse']['stable']<=.75*zero['stable'] and
        winner['mse']['changed']<=.80*zero['changed'] and
        winner['mse']['combined']<=.80*zero['combined'] and
        ols['stable']<=.75*zero['stable'] and ols['changed']<=.75*zero['changed'])
    out['gates']={'admission_pass':passed,'selected_stable_pass':winner['mse']['stable']<=.75*zero['stable'],
        'selected_changed_pass':winner['mse']['changed']<=.80*zero['changed'],
        'OLS_identifiable':ols['stable']<=.75*zero['stable'] and ols['changed']<=.75*zero['changed'],
        'source_exact_hash_crossplatform':'REQUIRED_FINAL_G4'}
    out['status']='PREFLIGHT_PASS_PORTABLE_CAL_BASELINE_ONLY' if passed else 'PREFLIGHT_FAIL_CONVENTIONAL_OR_IDENTIFIABILITY'
    out['elapsed_seconds']=time.perf_counter()-t0
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',required=True,type=Path)
    args=parser.parse_args();r=run();args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':r['status'],'selected':r.get('CAL',{}).get('selected'),
        'science_pass':r['science_pass'],'heldout_generated':r['heldout_generated']}))

if __name__=='__main__':main()
