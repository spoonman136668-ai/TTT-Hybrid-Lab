"""P47 frozen DEV: distinguish factual causal switches from exogenous steps."""
import argparse, hashlib, io, json, math, time
from pathlib import Path
import numpy as np

COHORTS=((760111,1536,(.35,.90)),(760117,1536,(.95,1.35)),(760123,1536,(1.40,2.0)))
N=8; T=np.arange(N)/8; SPLITS=(3,4,5)

def source(seed,n,band):
    if (seed,n,band) not in COHORTS:raise ValueError('P47_UNFROZEN_SOURCE')
    g=np.random.default_rng(seed)
    kind=np.repeat([0,1,2],[768,384,384]);g.shuffle(kind)
    k=g.choice(SPLITS,n);sgn=g.choice(np.array([-1.,1.]),n)
    a=sgn[:,None]*np.power(-1.,np.arange(N))[None,:]
    u=lambda low,high:g.uniform(low,high,(n,1))
    b0=u(-.2,.2);b1=u(-.4,.4);b2=u(-.1,.1)
    amp=u(0,.08);phase=u(-math.pi,math.pi);freq=u(*band)
    beta=g.choice(np.array([-1.,1.]),(n,1))*u(.25,.45)
    change=(kind[:,None]==0)&(np.arange(N)[None,:]>=k[:,None])
    beta_seq=beta*np.where(change,-1.,1.)
    jump=g.choice(np.array([-1.,1.]),(n,1))*u(.4,.7)
    shock=(kind[:,None]==1)&(np.arange(N)[None,:]>=k[:,None])
    background=b0+b1*T+b2*T*T+amp*np.sin(2*math.pi*freq*T+phase)+jump*shock
    y=background+beta_seq*a+g.normal(0,.02,(n,N))
    true_beta=beta_seq[:,-1].copy()
    sha=hashlib.sha256(b''.join(z.tobytes() for z in (a,y,true_beta,kind,k))).hexdigest()
    return dict(a=a,y=y,kind=kind,k=k,beta=true_beta,sha=sha,seed=seed)

def design(sign,mode='',k=None):
    a=sign*np.power(-1.,np.arange(N))
    cols=[np.ones(N),T,T*T,a]
    if mode=='causal':cols.append(a*(np.arange(N)>=k))
    if mode=='nuisance':cols.append((np.arange(N)>=k).astype(float))
    return np.stack(cols,-1)

def fit(y,sign,mode='',k=None):
    x=design(sign,mode,k)
    if np.linalg.matrix_rank(x,tol=1e-11)<x.shape[-1]:raise ValueError('P47_UNIDENTIFIABLE')
    pin=np.linalg.lstsq(x,np.eye(N),rcond=1e-11)[0]
    coeff=y@pin.T
    sse=np.square(y-coeff@x.T).sum(1)
    effect=coeff[:,3]+(coeff[:,4] if mode=='causal' else 0.)
    bic=N*np.log(np.maximum(sse,1e-10)/N)+x.shape[1]*math.log(N)
    if mode:bic+=math.log(3)
    return effect,bic

def estimate(a,y,causal_only=False):
    if a.shape!=y.shape or a.ndim!=2 or a.shape[1]!=N or not np.isfinite(a).all() or not np.isfinite(y).all():
        raise ValueError('P47_BAD_FACTUAL_HISTORY')
    n=len(y); pred=np.zeros(n); full=np.zeros(n);weight=np.zeros(n);tail=np.zeros(n)
    mode=np.full(n,-1,dtype=int); chosen=np.full(n,-1,dtype=int)
    best=np.full(n,np.inf)
    for sign in (-1.,1.):
        ids=np.flatnonzero(a[:,0]==sign)
        if not len(ids):continue
        if not np.array_equal(a[ids],np.broadcast_to(sign*np.power(-1.,np.arange(N)),(len(ids),N))):
            raise ValueError('P47_BAD_ACTION_SEQUENCE')
        data=y[ids]; eff,bic=fit(data,sign)
        pred[ids]=eff;full[ids]=eff;best[ids]=bic
        for k in SPLITS:
            for name,tag in [('causal',0),('nuisance',1)]:
                if causal_only and tag==1:continue
                candidate,score=fit(data,sign,name,k)
                chosen_mask=score<best[ids];idx=ids[chosen_mask]
                pred[idx]=candidate[chosen_mask];best[idx]=score[chosen_mask]
                mode[idx]=tag;chosen[idx]=k
        x=design(sign)
        w=np.sqrt(np.exp(-.70*(7-np.arange(N))))
        wp=np.linalg.lstsq(w[:,None]*x,np.eye(N)*w[None,:],rcond=1e-11)[0]
        weight[ids]=(data@wp.T)[:,3]
        tp=np.linalg.lstsq(x[3:],np.eye(5),rcond=1e-11)[0]
        tail[ids]=(data[:,3:]@tp.T)[:,3]
    last5=a[:,3]*(y[:,3]-4*y[:,4]+6*y[:,5]-4*y[:,6]+y[:,7])/16
    last3=a[:,5]*(y[:,5]-2*y[:,6]+y[:,7])/4
    return dict(candidate=pred,model_kind=mode,chosen_k=chosen,full=full,
                weighted=weight,trailing5=tail,last5=last5,last3=last3)

def metrics(d,est):
    masks={'causal':d['kind']==0,'exogenous':d['kind']==1,'stable':d['kind']==2,'all':d['kind']>=0}
    results={name:{g:dict(n=int(mask.sum()),mae=float(np.abs(val[mask]-d['beta'][mask]).mean()),
                        sign_accuracy=float((np.sign(val[mask])==np.sign(d['beta'][mask])).mean()))
                   for g,mask in masks.items()} for name,val in est.items() if name not in ('model_kind','chosen_k')}
    alarm=est['model_kind']==0
    det=dict(causal_recall=float(alarm[masks['causal']].mean()),
             exogenous_false_alarm=float(alarm[masks['exogenous']].mean()),
             stable_false_alarm=float(alarm[masks['stable']].mean()),
             causal_alarm_count=int(alarm.sum()),
             exact_causal_switch_location_rate=float((est['chosen_k'][masks['causal']]==d['k'][masks['causal']]).mean()))
    c=results['candidate'];g=results
    gates=dict(causal_MAE_15pct_strongest=c['causal']['mae']<=.85*min(g[k]['causal']['mae'] for k in ('trailing5','weighted','last5')),
        exogenous_false_alarm_10pct=det['exogenous_false_alarm']<=.10,
        stable_false_alarm_10pct=det['stable_false_alarm']<=.10,
        causal_recall_70pct=det['causal_recall']>=.70,
        stable_MAE_110pct_full=c['stable']['mae']<=1.10*g['full']['stable']['mae'])
    return dict(seed=d['seed'],source_sha256=d['sha'],results=results,detection=det,gates=gates,
                frozen_dev_pass=all(gates.values()))

def run():
    t0=time.perf_counter();rows=[]
    for seed,n,band in COHORTS:
        d=source(seed,n,band);start=time.perf_counter()
        est=estimate(d['a'],d['y'])
        row=metrics(d,est);row['estimation_seconds']=time.perf_counter()-start
        buf=io.BytesIO();np.savez_compressed(buf,a=d['a'],y=d['y']);buf.seek(0)
        with np.load(buf) as restored: cold=estimate(restored['a'],restored['y'])
        row['cold_exact']=all(np.array_equal(est[key],cold[key]) for key in est)
        rows.append(row)
    return dict(schema='ttt-hybrid.p47.causal-vs-nuisance-dev-receipt.v1',
                classification='AUTHORED_DEV_NOT_EXTERNAL_SCIENTIFIC_FIRST_USE',
                rows=rows,all_cohort_gates_pass=all(x['frozen_dev_pass'] and x['cold_exact'] for x in rows),
                local_elapsed_seconds=time.perf_counter()-t0,
                original_source_qualification='NOT_RUN',windows_ubuntu_full_regression='NOT_RUN',
                trained_neural_controls='NOT_RUN',FORGE='UNTOUCHED')

def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args();out=run();args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'cohorts':[{k:v for k,v in row.items() if k!='results'} for row in out['rows']],
         'causal_MAE':[{k:round(v['causal']['mae'],5) for k,v in row['results'].items()} for row in out['rows']],
         'all_gates':out['all_cohort_gates_pass'],'wall':out['local_elapsed_seconds']},indent=2))

if __name__=='__main__':main()
