"""P53 frozen, authored-DEV action-credit contrast memory experiment.

Uses only completed executed actions and their actual outcomes in the model.
Two hypothetical query outcomes are evaluator-only. No new external science.
"""
from __future__ import annotations
import argparse, hashlib, io, json, math, platform, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scripts import native_p51_factual_hybrid_learning_dev as p51
from scripts import native_p52_action_factorized_learning_dev as p52

SOURCE_SEEDS = {'fit':871101,'validation':871103,'unfamiliar_dev':871107}
SOURCE_N = {'fit':1536,'validation':256,'unfamiliar_dev':384}
MODEL_SEEDS=(881101,881103,881107)
INDEX_SEED=881199
STEPS=300
BATCH=64
VAL_EVERY=50


def source(name):
    if name not in SOURCE_SEEDS:
        raise ValueError('P53_UNREGISTERED_DEV_SOURCE')
    n=SOURCE_N[name]
    rng=np.random.default_rng(SOURCE_SEEDS[name]);ood=name=='unfamiliar_dev'
    u=lambda lo,hi:rng.uniform(lo,hi,(n,1))
    x=u(-.3,.3);intercept=u(-.12,.12);linear=u(-.25,.25);quad=u(-.12,.12)
    amp=u(-.18,.18) if ood else u(-.09,.09)
    freq=u(.9,1.6) if ood else u(.25,.70)
    phase=u(-math.pi,math.pi)
    beta=rng.choice(np.array([-1.,1.]),(n,1))*u(.20,.45)
    sw=np.zeros(n,dtype=bool)
    sw[:round(n*.35)]=True;rng.shuffle(sw)
    k=rng.integers(3,6,(n,1))
    a_hist=rng.choice(np.array([-1.,1.]),(n,8))
    history=np.empty((n,8,5),dtype=np.float32)
    for i in range(8):
        t=i*.125; effect=beta*np.where(sw[:,None]&(i>=k),-1.,1.)
        nuisance=intercept+linear*t+quad*t*t+amp*np.sin(2*math.pi*freq*t+phase)
        a=a_hist[:,i:i+1]
        delta=nuisance+effect*a+rng.normal(0,.01,(n,1))
        history[:,i,:]=np.concatenate((x,delta,a,np.full((n,1),.125),np.zeros((n,1))),axis=1)
        x=x+delta
    actual_action=rng.choice(np.array([-1.,1.]),(n,1))
    nuisance=(intercept+linear+quad+amp*np.sin(2*math.pi*freq+phase))
    true_beta=beta*np.where(sw[:,None],-1.,1.)
    observed_y=x+nuisance+true_beta*actual_action+rng.normal(0,.01,(n,1))
    paired=None
    if name!='fit':
        paired=x+nuisance+true_beta*np.array([[-1.,1.]])+rng.normal(0,.01,(n,2))
    raw={'history':history,'sensor':x.astype(np.float32),'query_a':actual_action.astype(np.float32),
         'factual_y':observed_y.astype(np.float32),'effect':true_beta.astype(np.float32),
         'switch':sw,'paired':None if paired is None else paired.astype(np.float32)}
    digest=hashlib.sha256(b''.join(raw[k].tobytes() for k in
        ('history','sensor','query_a','factual_y','effect','switch')
        if raw[k] is not None)+ (paired.astype(np.float32).tobytes() if paired is not None else b'')).hexdigest()
    return {k:torch.tensor(v) for k,v in raw.items() if v is not None},digest


def last_opposite_effect(history):
    """Exactly one latest opposite executed action pair per episode. No oracle.

    Returns effect, age/8, valid. Identical action histories have no estimate.
    """
    if history.ndim!=3 or history.shape[1:]!=(8,5) or not bool(torch.isfinite(history).all()):
        raise ValueError('P53_BAD_FACTUAL_HISTORY')
    a=history[:,:,2]; y=history[:,:,1]
    ids=torch.arange(8,device=a.device)[None,:].expand(len(a),-1)
    eligible=(a[:,:-1]!=a[:,-1,None])
    idx=torch.where(eligible,ids[:,:-1],-1).max(1).values
    valid=idx>=0
    safe=idx.clamp(min=0)
    opp_a=a.gather(1,safe[:,None])[:,0]
    opp_y=y.gather(1,safe[:,None])[:,0]
    denom=a[:,-1]-opp_a
    safe_denom=torch.where(valid,denom,torch.ones_like(denom))
    eff=torch.where(valid,(y[:,-1]-opp_y)/safe_denom,torch.zeros_like(denom))
    age=torch.where(valid,(7-safe).to(y.dtype)/8,torch.ones_like(eff))
    return eff,age,valid


def opposite_baseline(data):
    h=data['history'];sensor=data['sensor'][:,0]
    eff,age,valid=last_opposite_effect(h)
    # Unknown action contrast -> simple original P51 last-effect fallback.
    last=h[:,-1,1]/h[:,-1,2]
    eff=torch.where(valid,eff,last)
    background=h[:,-1,1]-eff*h[:,-1,2]
    both=torch.stack((sensor+background-eff,sensor+background+eff),1)
    return p51.scoring(both,data)


class ContradictionGatedP53(nn.Module):
    """P52 exact unchanged trainable trunk plus 3 learned factual fast weights."""
    def __init__(self):
        super().__init__()
        self.base=p52.FactorizedFactualP50()
        self.contradiction_gate=nn.Parameter(torch.tensor([0.,0.,0.]))

    def forward(self,data,candidates,wrong_credit=False):
        history=p51.history_with_credit(data,wrong_credit)
        query=p51.query_features(data['sensor'],candidates)
        b,k=query.shape[:2]
        action=query[:,:,2:3].clone()
        query=query.clone();query[:,:,2:3]=0.
        seq=torch.cat((history[:,None,:,:].expand(-1,k,-1,-1),query[:,:,None,:]),dim=2)
        outputs=self.base.native.pilot(seq.reshape(b*k,9,5),ttt_update=True)[:,-1,:].reshape(b,k,2)
        nuisance=outputs[:,:,:1]
        learned=outputs[:,:,1:]
        fast,age,valid=last_opposite_effect(history)
        theta=self.contradiction_gate
        g=torch.sigmoid(theta[0]+theta[1]*torch.abs(learned-fast[:,None,None])
                        +theta[2]*age[:,None,None])
        blended=torch.where(valid[:,None,None],g*fast[:,None,None]+(1-g)*learned,learned)
        return data['sensor'][:,None,:]+nuisance+action*blended


def train(model, cohorts, sampler, wrong_credit=False):
    opt=torch.optim.AdamW(model.parameters(),lr=.003)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=STEPS,eta_min=.0001)
    curve=[];start=time.perf_counter()
    for step in range(STEPS):
        model.train()
        x=p51.take(cohorts['fit'],sampler[step])
        pred=model(x,x['query_a'],wrong_credit=wrong_credit)[:,0,:]
        loss=(pred-x['factual_y']).square().mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError('P53_NONFINITE_TRAIN_LOSS')
        opt.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step();scheduler.step()
        if (step+1)%VAL_EVERY==0:
            model.eval()
            with torch.no_grad():
                d=cohorts['validation']
                guess=model(d,d['query_a'],wrong_credit=wrong_credit)[:,0,:]
                val=float((guess-d['factual_y']).square().mean())
                if not math.isfinite(val):raise ValueError('P53_NONFINITE_VALIDATION')
                curve.append(val)
    return dict(wall_seconds=time.perf_counter()-start,loss=float(loss.detach()),
                validation_curve=curve,plateau_last50=abs(curve[-1]-curve[-2])/max(curve[-2],1e-9)<=.01)


def cold_restore(model,source):
    data=p51.take(source,torch.arange(3));c=torch.tensor([[[-1.],[1.]]]).expand(3,-1,-1)
    model.eval()
    with torch.no_grad():before=model(data,c)
    buf=io.BytesIO();torch.save({'model':model.state_dict()},buf);buf.seek(0)
    fresh=ContradictionGatedP53().eval()
    fresh.load_state_dict(torch.load(buf,weights_only=True)['model'])
    with torch.no_grad():after=fresh(data,c)
    return bool(torch.equal(before,after))


def run():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    src={k:source(k) for k in SOURCE_SEEDS}
    coh={k:x[0] for k,x in src.items()};hashes={k:x[1] for k,x in src.items()}
    generator=torch.Generator().manual_seed(INDEX_SEED)
    batches=torch.randint(0,len(coh['fit']['history']),(STEPS,BATCH),generator=generator)
    batch_sha=hashlib.sha256(batches.numpy().tobytes()).hexdigest()
    arms={'P53':ContradictionGatedP53,'P52':p52.FactorizedFactualP50,
          'GRU':p51.GRUControl,'wrong_credit_P53':ContradictionGatedP53}
    params={name:p51.params(ctor()) for name,ctor in arms.items()}
    if params['P53']-params['P52']!=3:
        raise ValueError('P53_ADDITIONAL_PARAMETER_BUDGET_DRIFT')
    baselines={'last_effect':p51.last_effect(coh['unfamiliar_dev']),
               'opposite_pair':opposite_baseline(coh['unfamiliar_dev']),
               'ridge':p51.polynomial_ridge(coh['unfamiliar_dev'])}
    rows=[]
    for seed in MODEL_SEEDS:
        results={}
        for name,cls in arms.items():
            torch.manual_seed(seed);model=cls()
            wrong=name=='wrong_credit_P53'
            receipt=train(model,coh,batches,wrong_credit=wrong)
            unfamiliar=p51.evaluate(model,coh['unfamiliar_dev'],wrong_credit=wrong)
            validation=p51.evaluate(model,coh['validation'],wrong_credit=wrong)
            results[name]={'training':receipt,'unfamiliar':unfamiliar,'validation':validation}
            if name=='P53':
                results[name]['cold_exact']=cold_restore(model,coh['unfamiliar_dev'])
                results[name]['gate_params']=model.contradiction_gate.detach().tolist()
        m=lambda name,group,key:results[name]['unfamiliar']['metrics'][group][key]
        baseline=lambda name,group,key:baselines[name]['metrics'][group][key]
        best_eff=min([baseline(n,'all','effect_mae') for n in baselines]+[m(n,'all','effect_mae') for n in ('P52','GRU')])
        best_switched=min([baseline(n,'switched','effect_mae') for n in ('last_effect','opposite_pair')])
        best_mse=min([baseline(n,'all','mse') for n in baselines]+[m(n,'all','mse') for n in ('P52','GRU')])
        gates={
         'effect_MAE_5pct_strongest':m('P53','all','effect_mae')<=.95*best_eff,
         'switched_effect_MAE_10pct_fast':m('P53','switched','effect_mae')<=.90*best_switched,
         'prediction_MSE_5pct_strongest':m('P53','all','mse')<=.95*best_mse,
         'wrong_credit_prediction_20pct':m('P53','all','mse')<=.8*m('wrong_credit_P53','all','mse'),
         'cold_exact':results['P53']['cold_exact'],
         'all_neural_validation_plateau':all(z['training']['plateau_last50'] for z in results.values())}
        rows.append({'seed':seed,'arms':results,'gates':gates,'all_gates':all(gates.values())})
    return {'schema':'ttt-hybrid.p53-frozen-contradiction-dev-result.v1',
      'classification':'AUTHORED_DEV_NOT_INDEPENDENT_SCIENCE',
      'source_seeds':SOURCE_SEEDS,'source_hashes':hashes,'sample_index_sha256':batch_sha,
      'model_seeds':MODEL_SEEDS,'updates_per_neural_arm':STEPS,'params':params,
      'strong_conventional_controls':baselines,'rows':rows,
      'all_frozen_gates':all(x['all_gates'] for x in rows),
      'wall_seconds':time.perf_counter()-start,'python':platform.python_version(),'torch':torch.__version__,
      'external_science':'NOT_RUN','Windows_Linux_full_regression':'NOT_RUN',
      'FORGE':'UNTOUCHED','promotion':'NONE'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args();res=run();args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(res,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'all_gates':res['all_frozen_gates'],'params':res['params'],
      'strong_controls':{k:v['metrics'] for k,v in res['strong_conventional_controls'].items()},
      'seeds':[{'seed':x['seed'],'gates':x['gates'],
               'all_MSE':{k:v['unfamiliar']['metrics']['all']['mse'] for k,v in x['arms'].items()},
               'switched_effect_MAE':{k:v['unfamiliar']['metrics']['switched']['effect_mae'] for k,v in x['arms'].items()}}
              for x in res['rows']]},indent=2))

if __name__=='__main__':main()
