"""P92: prospective two-timescale factorized causal state vs full W256 control.

Frozen before source/score: 3af59fa35a180f52d4f827216a51c0c5d5b31b63.
Author synthetic DEV only; reserved OOD/NEGATIVE sources inaccessible.
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

FREEZE = '3af59fa35a180f52d4f827216a51c0c5d5b31b63'
FAMILIES = {'A': {'FIT': 920111, 'VAL': 920113}, 'B': {'FIT': 920121, 'VAL': 920123}}
SIZES = {'FIT': 96, 'VAL': 64}
SINGLE_GRID = ((.5,.95),(.5,.99),(1.,.95),(1.,.99),(1.5,.95),(1.5,.99))
DUAL_GRID = tuple((e,d,s,.99) for e in (.5,1.) for d in (.90,.97) for s in (.10,.25))
RANK = 4


def source(seed: int, n: int) -> dict:
    valid = {s: SIZES[split] for family in FAMILIES.values() for split,s in family.items()}
    if seed not in valid or n != valid[seed]:
        raise ValueError('P92_PROTECTED_OR_INVALID_SOURCE')
    def u32(w,tag,t,k=0,mod=1):
        key=f'p86-portable-v1|{seed}|{w}|{tag}|{t}|{k}'.encode('ascii')
        return int.from_bytes(hashlib.sha256(key).digest()[:4],'little') % mod
    def basis(w,tag):
        return [[u32(w,tag,j,k,7)-3 for k in range(4)] for j in range(16)]
    change=set(sorted(range(n),key=lambda w:u32(w,'regime',0,0,2**32))[:n//2])
    aa=[];ee=[];rr=[]
    for w in range(n):
        u0,v0=basis(w,'u0'),basis(w,'v0')
        u1,v1=basis(w,'u1'),basis(w,'v1')
        actions=[];effects=[]
        for t in range(49):
            ai=[u32(w,'action',t,j,193)-96 for j in range(16)]
            u,v=(u0,v0) if (t<24 or w not in change) else (u1,v1)
            h=[sum(ai[j]*u[j][k] for j in range(16)) for k in range(4)]
            yi=[sum(h[k]*v[d][k] for k in range(4)) for d in range(16)]
            if max(abs(y) for y in yi)>=2**24: raise ValueError('P92_DYADIC_OVERFLOW')
            actions.append([a/128 for a in ai])
            effects.append([y/16384 for y in yi])
        aa.append(actions);ee.append(effects);rr.append(w in change)
    return {'actions':torch.tensor(aa,dtype=torch.float32),
            'effects':torch.tensor(ee,dtype=torch.float32),
            'changed_world':torch.tensor(rr,dtype=torch.bool)}


def checksum(w):
    return hashlib.sha256(b''.join(w[k].contiguous().cpu().numpy().tobytes()
        for k in ('actions','effects','changed_world'))).hexdigest()


def audit(w):
    a=w['actions'];e=w['effects'];b=len(a)
    cosine=(F.normalize(a[:,:48],dim=-1)*F.normalize(a[:,48,None],dim=-1)).sum(-1).abs().max().item()
    finite=bool(torch.isfinite(a).all() and torch.isfinite(e).all())
    dyadic=bool(torch.equal(a*128,(a*128).round()) and torch.equal(e*16384,(e*16384).round()))
    return {'pass':finite and dyadic and cosine<.999 and int(w['changed_world'].sum())==b//2,
            'n':b,'changed':int(w['changed_world'].sum()),'max_abs_query_history_cosine':float(cosine),
            'source_exact_dyadic':dyadic,'finite':finite}


def full(w,eta=1.,decay=.99,start=0,stop=49,state=None,observed=None):
    k=F.normalize(w['actions'],dim=-1)
    y=w['effects'] if observed is None else observed
    W=k.new_zeros((len(k),16,16)) if state is None else state
    reads=[]
    for t in range(start,stop):
        a=k[:,t];predict=torch.bmm(W,a[:,:,None]).squeeze(-1)
        reads.append(predict)
        if t<48:
            r=y[:,t]-predict
            W=decay*W+eta*r[:,:,None]*a[:,None,:]
    return torch.stack(reads,1),W


def project4(W):
    # Working matrix temporary is 16x16; only top rank-4 factors persist.
    U,S,Vh=torch.linalg.svd(W,full_matrices=False)
    sqrt=torch.sqrt(S[:,:RANK])
    return U[:,:,:RANK]*sqrt[:,None,:],sqrt[:,:,None]*Vh[:,:RANK,:]


def dual(w,par,start=0,stop=49,state=None):
    ef,df,es,ds=par
    a=F.normalize(w['actions'],dim=-1);y=w['effects'];b=len(a)
    if state is None:
        state=(a.new_zeros((b,16,4)),a.new_zeros((b,4,16)),a.new_zeros((b,16,4)),a.new_zeros((b,4,16)))
    Uf,Vf,Us,Vs=state
    reads=[]
    for t in range(start,stop):
        k=a[:,t]
        Wf=Uf@Vf; Ws=Us@Vs
        pred=torch.bmm(Wf+Ws,k[:,:,None]).squeeze(-1)
        reads.append(pred)
        if t<48:
            r=y[:,t]-pred
            outer=r[:,:,None]*k[:,None,:]
            Uf,Vf=project4(df*Wf+ef*outer)
            Us,Vs=project4(ds*Ws+es*outer)
    return torch.stack(reads,1),(Uf,Vf,Us,Vs)


def single_fast(w,par):
    ef,df,_,_=par
    a=F.normalize(w['actions'],dim=-1); y=w['effects'];b=len(a)
    Uf=a.new_zeros((b,16,4));Vf=a.new_zeros((b,4,16))
    reads=[]
    for t in range(49):
        k=a[:,t];W=Uf@Vf
        pred=torch.bmm(W,k[:,:,None]).squeeze(-1);reads.append(pred)
        if t<48:
            Uf,Vf=project4(df*W+ef*(y[:,t]-pred)[:,:,None]*k[:,None,:])
    return torch.stack(reads,1)


def metric(pred,w):
    error=(pred-w['effects']).square().mean(-1)
    flag=w['changed_world']
    return {'combined':float(error[:,-1].mean()),'stable':float(error[~flag,-1].mean()),
            'changed':float(error[flag,-1].mean()),'stable_late':float(error[~flag,32:].mean()),
            'change_early':float(error[flag,24:32].mean()),'change_late':float(error[flag,40:].mean())}


def cold(w,par):
    with torch.no_grad():
        all_out,st=dual(w,par)
        _,half=dual(w,par,stop=24)
        tail,replay=dual(w,par,start=24,state=tuple(x.clone() for x in half))
    return bool(all(torch.equal(x,y) for x,y in zip(st,replay)) and
                torch.equal(all_out[:,-1],tail[:,-1]) and sum(x.numel() for x in st)==len(w['actions'])*256)


def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    t0=time.perf_counter()
    result={'schema':'ttt-hybrid.p92-lowrank-two-speed-causal-dev-result.v1','freeze_commit':FREEZE,
            'status':'NOT_RUN','scientific_pass':False,'OOD_generated':False,'NEGATIVE_generated':False,
            'full_native_hybrid_trained':False,'whole_process_compute_matched':False}
    worlds={name:{split:source(seed,SIZES[split]) for split,seed in fam.items()}
             for name,fam in FAMILIES.items()}
    result['source']={name:{split:{'seed':FAMILIES[name][split],'sha256':checksum(w),'audit':audit(w)}
                          for split,w in family.items()} for name,family in worlds.items()}
    if not all(y['audit']['pass'] for x in result['source'].values() for y in x.values()):
        result['status']='DEV_FAIL_SOURCE';return result
    fit={name:w['FIT'] for name,w in worlds.items()}
    val={name:w['VAL'] for name,w in worlds.items()}
    # Frozen FIT-only selection; no VAL data consulted during choices.
    gridfull=[]
    for eta,decay in SINGLE_GRID:
        score={name:metric(full(w,eta,decay)[0],w)['combined'] for name,w in fit.items()}
        gridfull.append({'eta':eta,'decay':decay,'FIT_mse_by_family':score,'FIT_mean':sum(score.values())/len(score)})
    fullwinner=min(gridfull,key=lambda z:z['FIT_mean'])
    dualgrid=[]
    for par in DUAL_GRID:
        score={name:metric(dual(w,par)[0],w)['combined'] for name,w in fit.items()}
        dualgrid.append({'params':list(par),'FIT_mse_by_family':score,'FIT_mean':sum(score.values())/len(score)})
    dualwinner=min(dualgrid,key=lambda z:z['FIT_mean'])
    result['FIT_selection']={'full_grid':gridfull,'dual_grid':dualgrid,'full_selected':fullwinner,'dual_selected':dualwinner}
    chosen=tuple(dualwinner['params'])
    scores={}
    for name,w in val.items():
        fixed,fixW=full(w)
        control,_=full(w,fullwinner['eta'],fullwinner['decay'])
        factored,factors=dual(w,chosen)
        ablation=single_fast(w,chosen)
        zero=torch.zeros_like(fixed)
        altered=w['effects'].clone()
        altered[:,:48]=w['effects'][:,:48][:,torch.arange(48).roll(1)]
        wrong,_=full(w,observed=altered)
        wrong_final=float((wrong[:,-1]-w['effects'][:,-1]).square().mean())
        reference=metric(fixed,w)
        report={'fixed':reference,'full_fit':metric(control,w),'dual':metric(factored,w),
                'single_rank4':metric(ablation,w),'zero':metric(zero,w),
                'wrong_credit_final_mse':wrong_final,
                'wrong_credit_pass':wrong_final>=1.5*reference['combined'] and wrong_final>=.01,
                'cold_restore_bitwise':cold(w,chosen),
                'dual_predictive_state_floats_per_world':sum(v.numel() for v in factors)//len(w['actions']),
                'full_predictive_state_floats_per_world':fixW.numel()//len(w['actions'])}
        scores[name]=report
    result['VAL']=scores
    if not all(v['wrong_credit_pass'] and v['cold_restore_bitwise'] and
               v['dual_predictive_state_floats_per_world']==256 and
               v['full_predictive_state_floats_per_world']==256 and
               v['fixed']['stable']<=.8*v['zero']['stable'] and
               v['fixed']['changed']<=.8*v['zero']['changed'] for v in scores.values()):
        result['status']='DEV_FAIL_CONTROLS';return result
    qualities={}
    for name,v in scores.items():
        cmp=min(v['fixed']['combined'],v['full_fit']['combined'])
        change_cmp=min(v['fixed']['changed'],v['full_fit']['changed'])
        stable_cmp=min(v['fixed']['stable'],v['full_fit']['stable'])
        qualities[name]={'overall':v['dual']['combined']<=.9*cmp,'changed':v['dual']['changed']<=.85*change_cmp,
                         'stable':v['dual']['stable']<=1.05*stable_cmp,
                         'two_slots':v['dual']['combined']<=.9*v['single_rank4']['combined']}
    result['performance_gates']=qualities
    # Frozen local CPU complete forward timing (not system-wide FLOP/memory parity).
    costs={}
    for name,w in val.items():
        costs[name]={}
        for key,fn in [('full',lambda:full(w,fullwinner['eta'],fullwinner['decay'])),
                       ('dual',lambda:dual(w,chosen))]:
            for _ in range(5):fn()
            samples=[]
            for _ in range(20):
                start=time.perf_counter();fn();samples.append((time.perf_counter()-start)*1000)
            costs[name][key]=sorted(samples)[len(samples)//2]
        costs[name]['ratio']=costs[name]['dual']/costs[name]['full']
        costs[name]['within_3x']=costs[name]['ratio']<=3
    result['costs_median_ms']=costs
    all_perf=all(all(v.values()) for v in qualities.values())
    all_cost=all(x['within_3x'] for x in costs.values())
    result['status']='DEV_FAIL_QUALITY' if not all_perf else 'DEV_FAIL_COST' if not all_cost else 'DEV_FIT_VAL_ONLY_NOT_SCIENCE'
    result['elapsed_seconds']=time.perf_counter()-t0
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();r=run();a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':r['status'],'elapsed_seconds':r.get('elapsed_seconds'),
          'selection':r.get('FIT_selection',{}).get('dual_selected'),
          'costs':r.get('costs_median_ms',{})}))

if __name__=='__main__':main()
