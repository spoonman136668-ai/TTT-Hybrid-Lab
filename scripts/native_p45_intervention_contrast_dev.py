"""P45 DEV: three *factual* alternating interventions as causal-credit supervision.

No unseen physical source first use. No alternative query-outcome training labels.
The exact original P43 module and P44 fully parameter-matched GRU are imported.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from scripts import native_p44_compare_dev as prev

SEEDS=(9403,9413,9421)
SOURCE_SEEDS={'fit':740001,'validation':740101,'ood_dev':740201}
COUNTS={'fit':3072,'validation':384,'ood_dev':384}
INDEX_SEED=741001
BATCH=128
UPDATES=1000
DT=1/8
LAMBDA=1.0


def source(cohort: str):
    if cohort not in COUNTS:
        raise ValueError('P45_UNREGISTERED_DEV_SOURCE')
    n=COUNTS[cohort]
    g=np.random.default_rng(SOURCE_SEEDS[cohort])
    u=lambda lo,hi:g.uniform(lo,hi,(n,1))
    b0=u(-.1,.1); b1=u(-.25,.25); b2=u(-.15,.15);amp=u(-.15,.15)
    phase=u(-math.pi,math.pi)
    f=u(*( (1.25,1.8) if cohort=='ood_dev' else (.45,1.0) ))
    beta=g.choice(np.array([-1.,1.]),(n,1))*u(.2,.45)
    switch=np.zeros(n,dtype=bool)
    if cohort=='ood_dev': switch[:96]=True
    actions=g.choice(np.array([-1.,1.]),(n,8))
    a=g.choice(np.array([-1.,1.]),(n,1))
    actions[:,5]=a[:,0];actions[:,6]=-a[:,0];actions[:,7]=a[:,0]
    factual_query_action=g.choice(np.array([-1.,1.]),(n,1))
    x=u(-.3,.3)
    hist=np.zeros((n,8,4),dtype=np.float64)
    for i in range(8):
        t=i*DT
        effective_beta=beta*np.where(switch[:,None] & (i>=4),-1.,1.)
        nuisance=b0+b1*t+b2*t*t+amp*np.sin(2*np.pi*f*t+phase)
        delta=nuisance+effective_beta*actions[:,i:i+1]+g.normal(0,.01,(n,1))
        hist[:,i,:]=np.concatenate((x,delta,actions[:,i:i+1],np.full((n,1),np.log1p(DT))),axis=1)
        x=x+delta
    t=1.0;current_effect=beta*np.where(switch[:,None],-1.,1.)
    nuisance=b0+b1*t+b2*t*t+amp*np.sin(2*np.pi*f*t+phase)
    realized_query=x+nuisance+current_effect*factual_query_action+g.normal(0,.01,(n,1))
    # For heldout evaluator ONLY: physically run both actions on independent copies.
    evaluator_both=x+nuisance+current_effect*np.array([[-1.,1.]])+g.normal(0,.01,(n,2))
    tensors={key:torch.tensor(val,dtype=torch.float32) for key,val in (
       ('history',hist),('query_sensor',x),('query_action',factual_query_action),
       ('factual_query_y',realized_query),('evaluator_both',evaluator_both),('effect',current_effect))}
    tensors['switched']=torch.tensor(switch)
    tensors['source_sha256']=hashlib.sha256(b''.join(tensors[k].numpy().tobytes() for k in (
        'history','query_sensor','query_action','factual_query_y','evaluator_both','effect','switched'))).hexdigest()
    return tensors


def aba_from_completed_facts(cohort):
    """ABA label uses real deltas/actions ONLY; never true beta or query outcomes."""
    h=cohort['history']
    a0=h[:,5,2:3]
    assert bool((h[:,7,2:3]==a0).all()) and bool((h[:,6,2:3]==-a0).all())
    return .5*a0*(.5*(h[:,5,1:2]+h[:,7,1:2])-h[:,6,1:2])


def fit(model, cohorts, indices, *, auxiliary):
    opt=torch.optim.AdamW(model.parameters(),lr=.003)
    stats=[];start=time.perf_counter()
    for i in range(UPDATES):
        model.train();sub=prev.batch_data(cohorts['fit'],indices[i])
        n=sub['history'].shape[0]
        candidates=torch.tensor([[[-1.],[1.]]],dtype=torch.float32).expand(n,-1,-1)
        pred=model(sub,candidates)
        idx=((sub['query_action'][:,0]+1)/2).long()
        factual_pred=pred[torch.arange(n),idx,:]
        loss=(factual_pred-sub['factual_query_y']).square().mean()
        if auxiliary in ('true','wrong_sign'):
            estimated_beta=(pred[:,1,:]-pred[:,0,:])*.5
            actual_factual_contrast=aba_from_completed_facts(sub)
            if auxiliary=='wrong_sign':actual_factual_contrast=-actual_factual_contrast
            loss=loss+LAMBDA*(estimated_beta-actual_factual_contrast).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        if (i+1)%200==0:
            model.eval();valid=cohorts['validation']
            with torch.no_grad():
                vp=model(valid,valid['query_action'])[:,0,:]
                stats.append(float((vp-valid['factual_query_y']).square().mean()))
    return {'updates':UPDATES,'validation_mse_curve':stats,'last_relative_improvement':
            (stats[-2]-stats[-1])/max(stats[-2],1e-12),'fit_wall_seconds':time.perf_counter()-start}


def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    cohorts={cohort:source(cohort) for cohort in COUNTS}
    draws=torch.randint(0,COUNTS['fit'],(UPDATES,BATCH),generator=torch.Generator().manual_seed(INDEX_SEED))
    drawsha=hashlib.sha256(draws.numpy().tobytes()).hexdigest()
    fitted_ridge=prev.ridge(cohorts['ood_dev'])
    heuristic=aba_from_completed_facts(cohorts['ood_dev'])
    aba_ood_effect_mae=float((heuristic-cohorts['ood_dev']['effect']).abs().mean())
    result=[]
    factories={'true_aux':prev.OriginalP43,'no_aux':prev.OriginalP43,
               'wrong_sign':prev.OriginalP43,'conventional_GRU':prev.GRUConventional}
    for seed in SEEDS:
        arms={}
        for name,factory in factories.items():
            torch.manual_seed(seed)
            model=factory()
            training=fit(model,cohorts,draws,auxiliary=
                {'true_aux':'true','wrong_sign':'wrong_sign'}.get(name,'none'))
            training['parameters']=prev.params(model)
            arms[name]={'train':training,
                'validation':prev.evaluate(model,cohorts['validation']),
                'ood':prev.evaluate(model,cohorts['ood_dev'])}
            if isinstance(model,prev.OriginalP43):
                exact,digest=prev.cold(model,cohorts['ood_dev'])
                arms[name]['cold_exact']=exact
                arms[name]['cold_sha256']=digest
        x=arms['true_aux']['ood'];a=arms['no_aux']['ood'];g=arms['conventional_GRU']['ood'];w=arms['wrong_sign']['ood']
        flags={
            'ood_mse_5pct_better_than_no_aux_and_GRU':x['mse']<=.95*min(a['mse'],g['mse']),
            'ood_causal_mae_5pct_better_than_no_aux_and_ABA':x['effect_mae']<=.95*min(a['effect_mae'],aba_ood_effect_mae),
            'ood_effect_mae_20pct_better_than_wrong_sign':x['effect_mae']<=.8*w['effect_mae'],
            'cold_exact':bool(arms['true_aux']['cold_exact']),
            'baseline_plateau_1pct':all(abs(v['train']['last_relative_improvement'])<=.01 for v in arms.values())}
        result.append({'seed':seed,'arms':arms,'gates':flags})
    return {'schema':'ttt-hybrid.p45-factual-contrast-development-receipt.v1',
            'science_first_use':'NOT_RUN','source_sha256':{k:v['source_sha256'] for k,v in cohorts.items()},
            'source_seeds':SOURCE_SEEDS,'sample_index_sha256':drawsha,
            'factual_triple_aba_ood_effect_mae':aba_ood_effect_mae,
            'conventional_quadratic_ridge_ood':fitted_ridge,
            'seeds':result,'positive_all_seeds':all(all(s['gates'].values()) for s in result),
            'total_wall_seconds':time.perf_counter()-start,
            'model_first_use':'AUTHORED_DEVELOPMENT_ONLY','full_original_repo_regression':'NOT_RUN'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args();out=run();args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'source_sha256':out['source_sha256'],'sample_index_sha256':out['sample_index_sha256'],
        'aba_ood_effect_mae':out['factual_triple_aba_ood_effect_mae'],
        'ridge':out['conventional_quadratic_ridge_ood'],
        'arms':{s['seed']:{k:{'ood_mse':v['ood']['mse'],'effect_mae':v['ood']['effect_mae'],
                         'training_seconds':v['train']['fit_wall_seconds']} for k,v in s['arms'].items()}
                 for s in out['seeds']},
        'gates':{s['seed']:s['gates'] for s in out['seeds']},
        'wall_seconds':out['total_wall_seconds']},indent=2))

if __name__=='__main__':main()
