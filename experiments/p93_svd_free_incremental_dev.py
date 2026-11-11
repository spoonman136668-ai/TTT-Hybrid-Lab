"""P93 frozen SVD-free 256-float two-timescale factual causal factors.

Prospective pre-data frozen commit: 0740cf629b1c775ff404851ded1411a41581258c.
All worlds are author-synthetic development; no protected OOD/NEGATIVE use.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from experiments import p92_lowrank_twospeed_dev as p92

FREEZE = '0740cf629b1c775ff404851ded1411a41581258c'
FAMILIES = {'A':{'FIT':930111,'VAL':930113}, 'B':{'FIT':930121,'VAL':930123}}
SIZES = {'FIT':96,'VAL':64}
FACTOR_GRID = tuple((u, v, d) for u in (.10,.35,.75) for v in (.025,.10) for d in (.95,.99))
FULL_GRID = p92.SINGLE_GRID
SOURCE_BLOBS = {
    'experiments/p92_lowrank_twospeed_dev.py':'09b18e9432e0808eb130084bb7be734cc8ab58a1',
    'model.py':'5d9ea57b827aa1796548447868fa5ee14c251b49',
}


def source(seed,n):
    legal={v:SIZES[k] for family in FAMILIES.values() for k,v in family.items()}
    if seed not in legal or n!=legal[seed]:
        raise ValueError('P93_PROTECTED_OR_INVALID_SOURCE')
    def u32(w,tag,t,k=0,mod=1):
        key=f'p86-portable-v1|{seed}|{w}|{tag}|{t}|{k}'.encode('ascii')
        return int.from_bytes(hashlib.sha256(key).digest()[:4],'little') % mod
    def basis(w,tag):
        return [[u32(w,tag,j,k,7)-3 for k in range(4)] for j in range(16)]
    changed=set(sorted(range(n),key=lambda w:u32(w,'regime',0,0,2**32))[:n//2])
    aa=[];ee=[];rr=[]
    for w in range(n):
        u0,v0=basis(w,'u0'),basis(w,'v0')
        u1,v1=basis(w,'u1'),basis(w,'v1')
        actions=[];effects=[]
        for t in range(49):
            aint=[u32(w,'action',t,j,193)-96 for j in range(16)]
            u,v=(u0,v0) if (t<24 or w not in changed) else (u1,v1)
            latent=[sum(aint[j]*u[j][k] for j in range(16)) for k in range(4)]
            yint=[sum(latent[k]*v[d][k] for k in range(4)) for d in range(16)]
            if max(abs(z) for z in yint)>=2**24: raise ValueError('P93_DYADIC_OVERFLOW')
            actions.append([a/128 for a in aint]);effects.append([z/16384 for z in yint])
        aa.append(actions);ee.append(effects);rr.append(w in changed)
    return {'actions':torch.tensor(aa,dtype=torch.float32),
            'effects':torch.tensor(ee,dtype=torch.float32),
            'changed_world':torch.tensor(rr,dtype=torch.bool)}


def checksum(w):return p92.checksum(w)


def walsh():
    H=torch.ones((1,1),dtype=torch.float32)
    for _ in range(4):
        H=torch.cat((torch.cat((H,H),dim=1),torch.cat((H,-H),dim=1)),dim=0)
    return H/4


def initial_state(batch,slots=2):
    if slots not in (1,2):raise ValueError('P93_INVALID_SLOT_COUNT')
    H=walsh().view(4,4,16) if False else walsh()
    V=torch.stack((H[:4],H[4:8]),dim=0)[:slots].unsqueeze(0).expand(batch,-1,-1,-1).clone()
    U=V.new_zeros((batch,slots,16,4))
    return (U,V)


def factor_rollout(world,params,mode='dual',start=0,stop=49,state=None,observed=None):
    if mode not in ('dual','single','fixedV'):raise ValueError('P93_INVALID_MODE')
    eta_u,eta_v,decay_fast=params
    slots=1 if mode=='single' else 2
    if state is None: U,V=initial_state(len(world['actions']),slots)
    else: U,V=state
    if U.shape!=(len(world['actions']),slots,16,4) or V.shape!=(len(world['actions']),slots,4,16):
        raise ValueError('P93_INVALID_FACTOR_STATE')
    a=F.normalize(world['actions'],dim=-1)
    y=world['effects'] if observed is None else observed
    sqrt_decay=torch.tensor([decay_fast,.999][:slots],dtype=a.dtype).sqrt().view(1,slots,1,1)
    stepU=torch.tensor([eta_u,.25*eta_u][:slots],dtype=a.dtype).view(1,slots,1,1)
    stepV=torch.tensor([eta_v,.25*eta_v][:slots],dtype=a.dtype).view(1,slots,1,1)
    reads=[]
    for t in range(start,stop):
        k=a[:,t]
        h=torch.matmul(V,k[:,None,:,None]).squeeze(-1)
        pred=torch.matmul(U,h[:,:,:,None]).squeeze(-1).sum(dim=1)
        reads.append(pred)
        if t==48:continue
        r=y[:,t]-pred
        ug=(r[:,None,:,None]*h[:,:,None,:])/(h.square().sum(-1)[:,:,None,None]+.25)
        gp=torch.matmul(U.transpose(-1,-2),r[:,None,:,None]).squeeze(-1)
        vg=(gp[:,:,:,None]*k[:,None,None,:])/(U.square().sum(dim=(-2,-1))[:,:,None,None]+.25)
        Unew=sqrt_decay*U+stepU*ug
        Vnew=sqrt_decay*V+(0. if mode=='fixedV' else stepV)*vg
        U,V=Unew,Vnew
    return torch.stack(reads,dim=1),(U,V)


def cold(world,params):
    with torch.no_grad():
        pred,end=factor_rollout(world,params)
        _,checkpoint=factor_rollout(world,params,stop=24)
        tail,resumed=factor_rollout(world,params,start=24,state=tuple(z.clone() for z in checkpoint))
    return (all(torch.equal(x,y) for x,y in zip(end,resumed)) and
            torch.equal(pred[:,-1],tail[:,-1]) and
            sum(x.numel() for x in end)==len(world['effects'])*256)


def blob(path):
    data=Path(path).read_bytes()
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def benchmark(fn,warmups=5,repeats=30):
    with torch.no_grad():
        for _ in range(warmups):fn()
        values=[]
        for _ in range(repeats):
            started=time.perf_counter();fn();values.append((time.perf_counter()-started)*1000)
    return sorted(values)[len(values)//2]


def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.perf_counter();root=Path(__file__).resolve().parents[1]
    result={'schema':'ttt-hybrid.p93-svd-free-online-factors-dev-result.v1',
      'freeze_commit':FREEZE,'scientific_pass':False,'OOD_generated':False,
      'NEGATIVE_generated':False,'full_hybrid_trained':False,'status':'NOT_RUN'}
    result['custody']={path:{'expected':sha,'actual':blob(root/path)} for path,sha in SOURCE_BLOBS.items() if (root/path).exists()}
    if any(z['actual']!=z['expected'] for z in result['custody'].values()):
        result['status']='DEV_FAIL_CODE_CUSTODY';return result
    # FIT selection completed without accessing any VAL scores.
    fits={name:source(seeds['FIT'],SIZES['FIT']) for name,seeds in FAMILIES.items()}
    result['source']={name:{'FIT':{'seed':FAMILIES[name]['FIT'],'hash':checksum(w),'audit':p92.audit(w)}} for name,w in fits.items()}
    if not all(z['FIT']['audit']['pass'] for z in result['source'].values()):
        result['status']='DEV_FAIL_SOURCE';return result
    full_grid=[]
    for eta,decay in FULL_GRID:
        scores={name:p92.metric(p92.full(w,eta,decay)[0],w)['combined'] for name,w in fits.items()}
        full_grid.append({'eta':eta,'decay':decay,'FIT':scores,'mean':sum(scores.values())/2})
    factor_grid=[]
    for params in FACTOR_GRID:
        scores={name:p92.metric(factor_rollout(w,params)[0],w)['combined'] for name,w in fits.items()}
        factor_grid.append({'params':list(params),'FIT':scores,'mean':sum(scores.values())/2})
    best_full=min(full_grid,key=lambda x:x['mean'])
    best_factor=min(factor_grid,key=lambda x:x['mean'])
    result['FIT']={'full_grid':full_grid,'factor_grid':factor_grid,'full_selected':best_full,'factor_selected':best_factor}
    params=tuple(best_factor['params'])
    vals={name:source(family['VAL'],SIZES['VAL']) for name,family in FAMILIES.items()}
    result['VAL']={}
    for name,w in vals.items():
        result['source'][name]['VAL']={'seed':FAMILIES[name]['VAL'],'hash':checksum(w),'audit':p92.audit(w)}
        if not result['source'][name]['VAL']['audit']['pass']:
            result['status']='DEV_FAIL_SOURCE';return result
        fixed,fstate=p92.full(w)
        fitted,_=p92.full(w,best_full['eta'],best_full['decay'])
        candidate,state=factor_rollout(w,params)
        ab1,_=factor_rollout(w,params,mode='single')
        ab2,_=factor_rollout(w,params,mode='fixedV')
        zero=torch.zeros_like(fixed)
        rotated=w['effects'].clone();rotated[:,:48]=w['effects'][:,:48][:,torch.arange(48).roll(1)]
        wrong,_=p92.full(w,observed=rotated)
        wrong_error=float((wrong[:,-1]-w['effects'][:,-1]).square().mean())
        fixed_score=p92.metric(fixed,w)
        score={'fixed':fixed_score,'full_fit':p92.metric(fitted,w),'factor':p92.metric(candidate,w),
               'single':p92.metric(ab1,w),'fixed_V':p92.metric(ab2,w),'zero':p92.metric(zero,w),
               'state_float32':sum(q.numel() for q in state)//len(w['actions']),
               'wrong_credit_mse':wrong_error,
               'wrong_credit_pass':wrong_error>=1.5*fixed_score['combined'] and wrong_error>=.01,
               'cold_bitwise':cold(w,params)}
        result['VAL'][name]=score
    if not all(x['state_float32']==256 and x['wrong_credit_pass'] and x['cold_bitwise'] and
               x['fixed']['stable']<=.8*x['zero']['stable'] and x['fixed']['changed']<=.8*x['zero']['changed']
               for x in result['VAL'].values()):
        result['status']='DEV_FAIL_CONTROL';return result
    result['gates']={}
    for name,x in result['VAL'].items():
        f=x['factor']; ref=x['full_fit'];fixed=x['fixed']
        result['gates'][name]={
          'combined':f['combined']<=1.05*min(ref['combined'],fixed['combined']),
          'stable':f['stable']<=1.10*min(ref['stable'],fixed['stable']),
          'changed':f['changed']<=1.10*min(ref['changed'],fixed['changed']),
          'single':f['combined']<=.95*x['single']['combined'],
          'fixed_V':f['combined']<=.95*x['fixed_V']['combined']}
    result['latency_ms']={}
    for name,w in vals.items():
        standard=benchmark(lambda:p92.full(w,best_full['eta'],best_full['decay']))
        incremental=benchmark(lambda:factor_rollout(w,params))
        result['latency_ms'][name]={'full':standard,'factors':incremental,
                                    'ratio':incremental/standard,'within_3x':incremental<=3*standard}
    quality=all(all(g.values()) for g in result['gates'].values())
    cost=all(x['within_3x'] for x in result['latency_ms'].values())
    result['status']='DEV_FAIL_QUALITY' if not quality else 'DEV_FAIL_COST' if not cost else 'DEV_FIT_VAL_ONLY_NOT_SCIENCE'
    result['elapsed_seconds']=time.perf_counter()-start
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();r=run();a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':r['status'],'selected':r.get('FIT',{}).get('factor_selected'),
                      'VAL':r.get('VAL'), 'latency_ms':r.get('latency_ms'),
                      'elapsed_seconds':r.get('elapsed_seconds')}))

if __name__=='__main__':main()
