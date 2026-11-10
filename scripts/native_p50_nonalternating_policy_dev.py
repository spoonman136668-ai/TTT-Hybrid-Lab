"""P50 authored DEV: factual causal changes under observation-conditioned actions.

No protected source, owner first-use, model promotion, or FORGE dispatch.
"""
from __future__ import annotations
import argparse, hashlib, io, json, math, time
from pathlib import Path
import numpy as np

N=8
T=np.arange(N,dtype=np.float64)/8
SPLITS=(3,4,5)
COHORTS={790001:(4096,(.35,2.),'calibration'),790101:(1536,(.35,.9),'evaluation'),790103:(1536,(.95,1.35),'evaluation'),790107:(1536,(1.4,2.),'evaluation')}

def authored(seed):
    """Actual policy can see only prior physical sensor x, not hidden worlds."""
    if type(seed) is not int or seed not in COHORTS: raise ValueError('P50_UNREGISTERED_SOURCE')
    n,band,role=COHORTS[seed];g=np.random.default_rng(seed)
    if role=='calibration':kind=np.full(n,2,dtype=np.int64)
    else:
        kind=np.repeat(np.arange(3,dtype=np.int64),[768,384,384]);g.shuffle(kind)
    policy=np.tile(np.arange(4,dtype=np.int64),math.ceil(n/4))[:n];g.shuffle(policy)
    k=g.choice(SPLITS,n);u=lambda lo,hi:g.uniform(lo,hi,(n,1))
    b0,b1,b2=u(-.2,.2),u(-.4,.4),u(-.1,.1)
    amp,phase,freq=u(0,.08),u(-math.pi,math.pi),u(*band)
    beta=g.choice(np.array([-1.,1.]),(n,1))*u(.25,.45)
    jump=g.choice(np.array([-1.,1.]),(n,1))*u(.4,.7)
    x=u(-.3,.3);initial=g.choice(np.array([-1.,1.]),n)
    a=np.empty((n,N));y=np.empty((n,N));old=initial.copy()
    for i,t in enumerate(T):
        random_action=g.choice(np.array([-1.,1.]),n)
        feedback=np.where(x[:,0]>0,-1.,1.)
        choose=g.random(n)
        now=np.where(policy==0,initial,
             np.where(policy==1,np.where(choose<.2,random_action,feedback),
               np.where(policy==2,np.where(choose<.8,old,-old),random_action)))
        effect=beta[:,0]*np.where((kind==0)&(i>=k),-1.,1.)
        step=jump[:,0]*((kind==1)&(i>=k))
        nuisance=b0[:,0]+b1[:,0]*t+b2[:,0]*t*t+amp[:,0]*np.sin(2*math.pi*freq[:,0]*t+phase[:,0])+step
        delta=nuisance+effect*now+g.normal(0,.02,n)
        a[:,i]=now;y[:,i]=delta;x[:,0]+=delta;old=now
    truth=beta[:,0]*np.where(kind==0,-1.,1.)
    trace=hashlib.sha256(b''.join(z.tobytes() for z in (a,y,truth,kind,k,policy))).hexdigest()
    return dict(a=a,y=y,beta=truth,kind=kind,k=k,policy=policy,sha=trace,seed=seed,role=role)


def design(actions,model='',split=None):
    if actions.shape!=(N,): raise ValueError('P50_ACTION_HISTORY_SHAPE')
    cols=[np.ones(N),T,T*T,actions]
    if model=='causal':cols.append(actions*(np.arange(N)>=split))
    elif model=='nuisance':cols.append((np.arange(N)>=split).astype(float))
    elif model:raise ValueError('P50_UNKNOWN_MODEL')
    return np.column_stack(cols)


def least(y,x,weights=None):
    if weights is None:weights=np.ones(len(y))
    h=x*weights[:,None];v=y*weights
    singular=np.linalg.svd(h,compute_uv=False)
    if not singular.size or singular[-1]<=1e-10*singular[0] or singular[0]/singular[-1]>1e6:return None
    coeff=np.linalg.lstsq(h,v,rcond=1e-10)[0]
    sse=float(np.square((y-x@coeff)*weights).sum())
    bic=len(y)*math.log(max(sse,1e-10)/len(y))+x.shape[1]*math.log(len(y))
    return coeff,bic

def fit_one(action,delta):
    if action.shape!=(N,) or delta.shape!=(N,) or not np.isfinite(action).all() or not np.isfinite(delta).all() or not np.all(np.isin(action,(-1.,1.))):
        raise ValueError('P50_BAD_FACTUAL_EVENT_SEQUENCE')
    x=design(action);base=least(delta,x)
    fields=dict(valid=False,full=np.nan,weighted=np.nan,trailing5=np.nan,last3=np.nan,
                causal=np.nan,nuisance=np.nan,causal_margin=-np.inf,nuisance_margin=-np.inf)
    if base is None:return fields
    coeff,bic=base;fields['valid']=True;fields['full']=float(coeff[3]);best_null=bic;best_causal=np.inf;best_nuisance=np.inf
    for k in SPLITS:
        for mode in ('causal','nuisance'):
            f=least(delta,design(action,mode,k))
            if f is None:continue
            c,score=f;score+=math.log(len(SPLITS))
            if mode=='causal' and score<best_causal:
                best_causal=score;fields['causal']=float(c[3]+c[4])
            if mode=='nuisance' and score<best_nuisance:
                best_nuisance=score;fields['nuisance']=float(c[3])
            if mode=='nuisance':fields['nuisance_margin']=max(fields['nuisance_margin'],bic-score)
    best_null=min(best_null,best_nuisance)
    fields['causal_margin']=best_null-best_causal
    weights=np.exp(-.70*(7-np.arange(N)))**.5
    f=least(delta,x,weights=weights)
    if f is not None:fields['weighted']=float(f[0][3])
    f=least(delta[3:],x[3:])
    if f is not None:fields['trailing5']=float(f[0][3])
    tt=T[-3:];f=least(delta[-3:],np.column_stack([np.ones(3),tt,action[-3:]]))
    if f is not None:fields['last3']=float(f[0][2])
    return fields


def fitted(a,y):
    if a.shape!=y.shape or a.ndim!=2 or a.shape[1]!=N:raise ValueError('P50_BAD_FACTUAL_BATCH')
    return [fit_one(a[i],y[i]) for i in range(len(y))]


def calibration(src):
    if src['seed']!=790001 or src['role']!='calibration' or not np.all(src['kind']==2):raise ValueError('P50_BAD_CALIBRATION_CUSTODY')
    rows=fitted(src['a'],src['y']);valid=[x for x in rows if x['valid']]
    def threshold(field):
        v=np.array([x[field] for x in valid if np.isfinite(x[field])]);return float(np.quantile(v,.99,method='higher'))
    return dict(causal=threshold('causal_margin'),nuisance=threshold('nuisance_margin'),valid_calibration=len(valid))


def inferred(rows,threshold,mode):
    n=len(rows);out=np.full(n,np.nan);alarm=np.zeros(n,dtype=bool)
    for i,x in enumerate(rows):
        if not x['valid']:continue
        causal=np.isfinite(x['causal']) and x['causal_margin']>max(0.,threshold['causal'])
        alarm[i]=causal
        if causal:out[i]=x['causal'];continue
        full=x['full'];nu=x['nuisance']
        if not np.isfinite(nu):out[i]=full;continue
        if mode=='single':out[i]=nu if x['nuisance_margin']>0 else full
        elif mode=='dual':out[i]=nu if x['nuisance_margin']>max(0.,threshold['nuisance']) else full
        elif mode=='soft':
            advantage=np.clip(x['nuisance_margin']*.5,-30,30)
            p=1/(1+np.exp(-advantage));out[i]=(1-p)*full+p*nu
        else:raise ValueError('P50_UNKNOWN_INFERENCE')
    return out,alarm


def scores(src,rows,threshold):
    valid=np.array([r['valid'] for r in rows]);kind=src['kind'];truth=src['beta']
    variants={z:inferred(rows,threshold,z) for z in ('single','dual','soft')}
    controls={z:np.array([r[z] for r in rows]) for z in ('full','weighted','trailing5','last3')}
    groups={z:((kind==j)&valid) for z,j in (('causal',0),('exogenous',1),('stable',2))}
    def error(pred):
        return {k:float(np.abs(pred[v & np.isfinite(pred)]-truth[v & np.isfinite(pred)]).mean()) if (v & np.isfinite(pred)).any() else None for k,v in groups.items()}
    results={name:dict(mae=error(pred),alarm={z:float(alarm[group].mean()) if group.any() else None for z,group in groups.items()}) for name,(pred,alarm) in variants.items()}
    reference={name:error(pred) for name,pred in controls.items()}
    bypolicy={label:dict(n=int((src['policy']==j).sum()),identifiable=int(((src['policy']==j)&valid).sum())) for j,label in enumerate(('constant','feedback','sticky','random'))}
    a=results['soft']['alarm'];e=results['soft']['mae'];single=results['single']['mae']
    def le(x,y,m):return x is not None and y is not None and x<=m*y
    checks=dict(coverage=valid.mean()>=.65,causal_recall=a['causal'] is not None and a['causal']>=.70,
       stable_false_alarm=a['stable'] is not None and a['stable']<=.10,
       exogenous_false_alarm=a['exogenous'] is not None and a['exogenous']<=.10,
       stable_MAE=le(e['stable'],reference['full']['stable'],1.10),
       exogenous_MAE=le(e['exogenous'],single['exogenous'],1.10),
       causal_MAE=le(e['causal'],min(x for x in (reference['weighted']['causal'],reference['trailing5']['causal']) if x is not None),.85))
    return dict(valid_n=int(valid.sum()),fraction=float(valid.mean()),per_policy_coverage=bypolicy,
       methods=results,controls=reference,checks=checks,pass_all=all(checks.values()))

def study():
    started=time.perf_counter();cal=authored(790001);threshold=calibration(cal);reports=[]
    for seed in (790101,790103,790107):
        world=authored(seed);rows=fitted(world['a'],world['y']);s=scores(world,rows,threshold)
        # Cold restore includes ONLY factual executed actions and observed deltas.
        buf=io.BytesIO();np.savez_compressed(buf,a=world['a'],y=world['y']);buf.seek(0)
        with np.load(buf,allow_pickle=False) as f:cold=fitted(f['a'],f['y'])
        r1,a1=inferred(rows,threshold,'soft');r2,a2=inferred(cold,threshold,'soft')
        s['cold_bit_exact']=bool(np.array_equal(r1,r2,equal_nan=True) and np.array_equal(a1,a2))
        s['seed']=seed;s['source_sha256']=world['sha'];reports.append(s)
    return dict(schema='ttt-hybrid.p50.policy-factual-action-dev-result.v1',
        source_type='AUTHORED_EXPLORATORY_DEV_ONLY',calibration_seed=790001,
        calibration_source_sha256=cal['sha'],calibration_thresholds=threshold,reports=reports,
        frozen_all_cohort_pass=all(x['pass_all'] and x['cold_bit_exact'] for x in reports),
        wall_seconds=time.perf_counter()-started,scientific_first_use='NOT_RUN',
        full_original_windows_ubuntu_regression='NOT_RUN',FORGE='UNTOUCHED')


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);args=p.parse_args()
    z=study();args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(z,indent=2,sort_keys=True,default=lambda x:x.item() if isinstance(x,np.generic) else repr(x))+'\n')
    print(json.dumps({'threshold':z['calibration_thresholds'],
        'cohorts':[{'seed':x['seed'],'coverage':x['fraction'],'per_policy':x['per_policy_coverage'],
            'metrics':x['methods'],'checks':x['checks'],'controls':x['controls'],
            'cold':x['cold_bit_exact']} for x in z['reports']],
        'pass':z['frozen_all_cohort_pass'],'seconds':z['wall_seconds']},indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else repr(x)))

if __name__=='__main__':main()
