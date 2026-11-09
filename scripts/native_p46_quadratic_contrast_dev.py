"""P46: frozen source-free five-real-intervention causal credit under nonlinear drift.

Original P43 + conventional P44 recurrent arms; no heldout source feedback.
Run from repo root: python -m scripts.native_p46_quadratic_contrast_dev --receipt PATH.
"""
from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
from scripts import native_p44_compare_dev as p44
from scripts import native_p45_intervention_contrast_dev as p45

SEEDS=(9503,9511,9521)
SOURCE_SEEDS={'train':750001,'val':750101,'ood':750201}
COUNTS={'train':3072,'val':384,'ood':384}
INDEX_SEED=751001
UPDATES=1200
BATCH=128
DT=1/8


def source(cohort):
    if cohort not in SOURCE_SEEDS:raise ValueError('P46_UNKNOWN_COHORT')
    n=COUNTS[cohort];rng=np.random.default_rng(SOURCE_SEEDS[cohort]);ood=cohort=='ood'
    u=lambda lo,hi:rng.uniform(lo,hi,(n,1))
    b0=u(-.1,.1);b1=u(-.25,.25);b2=u(-.35,.35);b3=u(-.4,.4)
    amp=u(-.3,.3);phase=u(-np.pi,np.pi)
    freq=u(*( (1.2,1.8) if ood else (.25,.65) ))
    beta=rng.choice(np.array([-1.,1.]),(n,1))*u(.2,.45)
    switched=np.zeros((n,),dtype=bool)
    if ood:switched[:96]=True
    aa=rng.choice(np.array([-1.,1.]),(n,8));first=rng.choice(np.array([-1.,1.]),(n,1))
    aa[:,3:]=first*np.array([1,-1,1,-1,1])[None,:]
    qa=rng.choice(np.array([-1.,1.]),(n,1));x=u(-.3,.3)
    hist=np.zeros((n,8,4),dtype=np.float64)
    for i in range(8):
        t=i*DT;eff=beta*np.where(switched[:,None] & (i>=4),-1.,1.)
        nuisance=b0+b1*t+b2*t*t+b3*t**3+amp*np.sin(2*np.pi*freq*t+phase)
        delta=nuisance+eff*aa[:,i:i+1]+rng.normal(0,.01,(n,1))
        hist[:,i,:]=np.concatenate((x,delta,aa[:,i:i+1],np.full((n,1),np.log1p(DT))),axis=1)
        x=x+delta
    t=1.;effect=beta*np.where(switched[:,None],-1.,1.)
    nuisance=b0+b1*t+b2*t*t+b3*t**3+amp*np.sin(2*np.pi*freq*t+phase)
    fq=x+nuisance+effect*qa+rng.normal(0,.01,(n,1))
    ev=x+nuisance+effect*np.array([[-1.,1.]])+rng.normal(0,.01,(n,2))
    d={k:torch.tensor(v,dtype=torch.float32) for k,v in [('history',hist),('query_sensor',x),('query_action',qa),('factual_query_y',fq),('evaluator_both',ev),('effect',effect)]}
    d['switched']=torch.tensor(switched)
    d['source_sha256']=hashlib.sha256(b''.join(d[k].numpy().tobytes() for k in ('history','query_sensor','query_action','factual_query_y','evaluator_both','effect','switched'))).hexdigest()
    return d


def degree_two_contrast(data):
    """Untrained 5 completed actual interventions; remove quadratic nuisance."""
    h=data['history'];act=h[:,3:8,2];A=act[:,0:1]
    alternating=torch.tensor([1.,-1.,1.,-1.,1.],dtype=h.dtype,device=h.device)
    if not bool(torch.equal(act,A*alternating)):
        raise ValueError('P46_NOT_FIVE_ALTERNATING_FACTS')
    # Analytically pre-solved residualized action contrast: fourth difference.
    # Exact rational weights remove degree <=3 polynomial nuisance; equivalent
    # to rank-checked residualization against [1,t,t^2] on frozen equally
    # spaced five-action design. Avoid a redundant tiny LAPACK solve per batch.
    w=h.new_tensor([1.,-4.,6.,-4.,1.])/16
    return A*(h[:,3:8,1]*w).sum(1,keepdim=True)


def classical(cohort,degree=3,latest=8,reg=.001):
    """Strong same-history conventional polynomial nuisance/action ridge."""
    h=cohort['history'].double();n=h.shape[0]
    ts=torch.arange(8-latest,8,dtype=h.dtype)[None,:].expand(n,-1)/8
    features=[ts**i for i in range(degree+1)]+[h[:,-latest:,2]]
    X=torch.stack(features,dim=-1)
    y=h[:,-latest:,1,None]
    ridge=reg*torch.eye(X.shape[-1],dtype=h.dtype).expand(n,-1,-1)
    theta=torch.linalg.solve(X.transpose(1,2)@X+ridge,X.transpose(1,2)@y).squeeze(-1)
    rank=int((torch.linalg.matrix_rank(X)==X.shape[-1]).sum())
    background=theta[:,:degree+1].sum(-1)
    s=cohort['query_sensor'][:,0].double()
    pred=torch.stack((s+background-theta[:,-1],s+background+theta[:,-1]),dim=1).float()
    truth=cohort['evaluator_both']; eff=cohort['effect'][:,0]
    return {'mse':float((pred-truth).square().mean()),
            'effect_mae':float((theta[:,-1].float()-eff).abs().mean()),
            'rank':rank,'n':n}


def training(model,cohorts,draws,arm):
    opt=torch.optim.AdamW(model.parameters(),lr=.003);curve=[];start=time.perf_counter()
    for i in range(UPDATES):
        model.train();batch=p44.batch_data(cohorts['train'],draws[i]);n=len(batch['history'])
        can=torch.tensor([[[-1.],[1.]]],dtype=torch.float32).expand(n,-1,-1)
        pred=model(batch,can)
        pick=((batch['query_action'][:,0]+1)/2).long()
        actual_pred=pred[torch.arange(n),pick,:]
        loss=(actual_pred-batch['factual_query_y']).square().mean()
        if arm!='none':
            label=degree_two_contrast(batch) if arm in ('five','wrong','gru_five') else p45.aba_from_completed_facts(batch)
            if arm=='wrong':label=-label
            beta_hat=(pred[:,1,:]-pred[:,0,:])*.5
            loss=loss+(beta_hat-label).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        if (i+1)%200==0:
            model.eval();v=cohorts['val']
            with torch.no_grad():
                y=model(v,v['query_action'])[:,0,:]
                curve.append(float((y-v['factual_query_y']).square().mean()))
    return {'optimizer_updates':UPDATES,'validation_curve':curve,'train_wall_seconds':time.perf_counter()-start,
            'last_two_relative_change':abs(curve[-1]-curve[-2])/max(1e-12,curve[-2])}


def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.perf_counter();cohorts={x:source(x) for x in COUNTS}
    draws=torch.randint(0,COUNTS['train'],(UPDATES,BATCH),generator=torch.Generator().manual_seed(INDEX_SEED))
    drawhash=hashlib.sha256(draws.numpy().tobytes()).hexdigest()
    five=degree_two_contrast(cohorts['ood']);three=p45.aba_from_completed_facts(cohorts['ood'])
    beta=cohorts['ood']['effect'];s=cohorts['ood']['switched']
    controls={'five_effect_mae':float((five-beta).abs().mean()),
              'three_effect_mae':float((three-beta).abs().mean()),
              'five_switch_effect_mae':float((five[s]-beta[s]).abs().mean()),
              'five_nonswitch_effect_mae':float((five[~s]-beta[~s]).abs().mean()),
              'ridge8_degree3':classical(cohorts['ood'],degree=3,latest=8),
              'ridge5_degree2':classical(cohorts['ood'],degree=2,latest=5)}
    arms=[('P43_five',p44.OriginalP43,'five'),('P43_three',p44.OriginalP43,'three'),
          ('P43_none',p44.OriginalP43,'none'),('P43_wrong',p44.OriginalP43,'wrong'),
          ('GRU_five',p44.GRUConventional,'gru_five')]
    records=[]
    for seed in SEEDS:
        models={}
        for name,factory,mode in arms:
            torch.manual_seed(seed);model=factory()
            fit=training(model,cohorts,draws,mode)
            out=p44.evaluate(model,cohorts['ood'])
            val=p44.evaluate(model,cohorts['val'])
            info={'train':fit,'ood':out,'val':val,'params':p44.params(model)}
            if isinstance(model,p44.OriginalP43):
                good,digest=p44.cold(model,cohorts['ood']);info['cold_exact']=good;info['cold_sha256']=digest
            models[name]=info
        c=models['P43_five']['ood'];best_mse=min(models[k]['ood']['mse'] for k in ('P43_three','P43_none','GRU_five'))
        best_effect=min(models['GRU_five']['ood']['effect_mae'],controls['five_effect_mae'],controls['ridge8_degree3']['effect_mae'])
        gates={'mse_vs_strong_5pct':c['mse']<=.95*min(best_mse,controls['ridge8_degree3']['mse']),
               'effect_vs_strong_5pct':c['effect_mae']<=.95*best_effect,
               'wrong_sign_effect_margin_20pct':c['effect_mae']<=.8*models['P43_wrong']['ood']['effect_mae'],
               'cold_exact':models['P43_five']['cold_exact'],
               'neural_plateau_all_1pct':all(x['train']['last_two_relative_change']<=.01 for x in models.values())}
        records.append({'seed':seed,'arms':models,'gates':gates})
    return {'schema':'ttt-hybrid.P46.quadratic-factual-contrast-dev-result.v1',
            'source_kind':'AUTHORED_DEVELOPMENT_NOT_OWNER_QUALIFIED_FIRST_USE',
            'source_hashes':{k:v['source_sha256'] for k,v in cohorts.items()},
            'shared_train_draw_sha256':drawhash,'source_seeds':SOURCE_SEEDS,
            'model_seeds':SEEDS,'fit_episode_count':COUNTS,'updates_each':UPDATES,
            'controls':controls,'records':records,'all_3_seed_positive':all(all(v for v in row['gates'].values()) for row in records),
            'wall_seconds':time.perf_counter()-start,'original_repo_full_windows_and_ubuntu':'NOT_RUN','FORGE':'UNTOUCHED'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);args=p.parse_args()
    o=run();args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(o,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'source_hashes':o['source_hashes'],'controls':o['controls'],
        'per_seed':{x['seed']:{'gates':x['gates'],'mse':{k:v['ood']['mse'] for k,v in x['arms'].items()},
                             'effect_mae':{k:v['ood']['effect_mae'] for k,v in x['arms'].items()}} for x in o['records']},
        'all_positive':o['all_3_seed_positive'],'wall_seconds':o['wall_seconds']},indent=2))

if __name__=='__main__':main()
