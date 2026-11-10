"""P52 prospective, source-free development comparison on native factual action streams.

Only the *completed* physical observations and the one executed training query
are supervised. Factorization forces action effect into a separate continuous
readout rather than inferring the query-action dependence through token inputs.
This is a conventional inductive bias, NOT a scientific causal-learning proof.
Do not replay the new evaluation seeds as a second independent experiment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from model import Configuration
from native_factual_hybrid_stream import NativeFactualHybridStream
from scripts import native_p51_factual_hybrid_learning_dev as p51

FRESH_SEEDS = {"fit": 851101, "validation": 851103, "unfamiliar_dev": 851107}
MODEL_SEEDS = (861101, 861103, 861107)
SAMPLER_SEED = 861199
UPDATES = 800
BATCH = 64
VALIDATION_EVERY = 40
LR = .003
LR_MIN = .0001


class FactorizedFactualP50(nn.Module):
    """Same P50 SSM/attention/TTT/depth trunk; explicit effect readout.

    The hypothetical query action is masked from the trunk, then multiplied
    against a learned causal-effect factor in the final continuous prediction.
    The factual history retains *executed* action identity and real outcomes.
    This does not prove identifiability against nonlinear hidden confounding.
    """
    def __init__(self):
        super().__init__()
        self.native = NativeFactualHybridStream(1, capacity=8,
                cfg=Configuration(width=16, heads=2), min_facts=3)
        self.native.pilot.head = nn.Linear(16, 2)

    def forward(self, data, candidates, wrong_credit=False):
        history = p51.history_with_credit(data, wrong_credit)
        query = p51.query_features(data['sensor'], candidates)
        b, k = query.shape[:2]
        # Mask the proposed action from SSM, attention and TTT. This ensures
        # exactly antisymmetric differences: y(+a)-y(-a)=2*a*effect(history).
        query = query.clone()
        actions = query[:, :, 2:3].clone()
        query[:, :, 2:3] = 0.
        seq = torch.cat((history[:, None, :, :].expand(-1,k,-1,-1),
                         query[:, :, None, :]), dim=2).reshape(b*k, p51.HISTORY+1, 5)
        out = self.native.pilot(seq, ttt_update=True)[:, -1, :].reshape(b,k,2)
        return data['sensor'][:,None,:] + out[:,:,:1] + actions*out[:,:,1:]


def new_source():
    """Reuses byte-pinned P51 authored physics with separate prospective seeds."""
    previous = p51.SEEDS
    try:
        p51.SEEDS = dict(FRESH_SEEDS)
        return {name: p51.source(name) for name in FRESH_SEEDS}
    finally:
        p51.SEEDS = previous


def train(model, cohorts, sampler, *, wrong_credit=False):
    opt = torch.optim.AdamW(model.parameters(),lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=UPDATES,eta_min=LR_MIN)
    curve=[]
    start=time.perf_counter()
    for step in range(UPDATES):
        model.train()
        d=p51.take(cohorts['fit'],sampler[step])
        predicted=model(d,d['query_a'],wrong_credit=wrong_credit)[:,0,:]
        loss=(predicted-d['factual_y']).square().mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError('P52_NONFINITE_TRAIN_LOSS')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        scheduler.step()
        if (step+1)%VALIDATION_EVERY==0:
            model.eval()
            with torch.no_grad():
                d=cohorts['validation']
                pred=model(d,d['query_a'],wrong_credit=wrong_credit)[:,0,:]
                mse=float(((pred-d['factual_y'])**2).mean())
                if not math.isfinite(mse):
                    raise ValueError('P52_NONFINITE_VALIDATION')
                curve.append(mse)
    plateau=abs(curve[-2]-curve[-1])/max(curve[-2],1e-9)<=.01
    return {'wall_seconds':time.perf_counter()-start,
            'last_train_loss':float(loss.detach()),
            'validation_curve':curve,
            'plateau_last40':bool(plateau),
            'last40_relative_variation':abs(curve[-2]-curve[-1])/max(curve[-2],1e-9)}


def exact_cold_restore(model,data):
    model.eval()
    sample=p51.take(data,torch.arange(3))
    a=torch.tensor([[[-1.],[1.]]]).expand(3,-1,-1)
    with torch.no_grad():
        first=model(sample,a)
    import io
    buf=io.BytesIO()
    torch.save({'weights':model.state_dict()},buf)
    buf.seek(0)
    other=FactorizedFactualP50().eval()
    other.load_state_dict(torch.load(buf,weights_only=True)['weights'])
    with torch.no_grad():
        second=other(sample,a)
    return bool(torch.equal(first,second))


def evaluate(model,data,wrong_credit=False):
    return p51.evaluate(model,data,wrong_credit=wrong_credit)


def run():
    started=time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source=new_source()
    cohorts={name:v[0] for name,v in source.items()}
    hashes={name:v[1] for name,v in source.items()}
    sampler=torch.randint(0,len(cohorts['fit'] ['history']),(UPDATES,BATCH),
          generator=torch.Generator().manual_seed(SAMPLER_SEED))
    sampler_sha=hashlib.sha256(sampler.numpy().tobytes()).hexdigest()
    arms={'original_P50':p51.P50Original,
          'factorized_P50':FactorizedFactualP50,
          'GRU':p51.GRUControl,
          'MLP':p51.MLPControl,
          'wrong_credit_factorized':FactorizedFactualP50}
    params={name:p51.params(factory()) for name,factory in arms.items()}
    if params['factorized_P50']>1.02*params['original_P50']:
        raise ValueError('P52_PARAMETER_BUDGET_EXCEEDED')
    ridge=p51.polynomial_ridge(cohorts['unfamiliar_dev'])
    last=p51.last_effect(cohorts['unfamiliar_dev'])
    records=[]
    for seed in MODEL_SEEDS:
        record={}
        for name,factory in arms.items():
            torch.manual_seed(seed)
            model=factory()
            wrong=name=='wrong_credit_factorized'
            t=train(model,cohorts,sampler,wrong_credit=wrong)
            unfamiliar=evaluate(model,cohorts['unfamiliar_dev'],wrong_credit=wrong)
            validation=evaluate(model,cohorts['validation'],wrong_credit=wrong)
            record[name]={'training':t,'unfamiliar':unfamiliar,
                          'validation':validation}
            if name=='factorized_P50':
                record[name]['cold_exact']=exact_cold_restore(model,cohorts['unfamiliar_dev'])
                # Direct algebraic check on exactly matched input; no future
                # sensor outcomes are supplied to this read-only prediction.
                sample=p51.take(cohorts['unfamiliar_dev'],torch.arange(4))
                plus=torch.ones((4,1))
                minus=-plus
                with torch.no_grad():
                    yp=model(sample,plus)
                    ym=model(sample,minus)
                    yb=model(sample,torch.stack((minus,plus),dim=1))
                record[name]['action_antisymmetry']=bool(torch.allclose(
                    (yp[:,0,:]+ym[:,0,:])/2.,
                    (yb[:,0,:]+yb[:,1,:])/2.,atol=1e-6,rtol=0))
        def M(arm,metric,group='all'):
            return record[arm]['unfamiliar']['metrics'][group][metric]
        fact=record['factorized_P50']
        base_MSE=min([ridge['metrics']['all']['mse'],
                      last['metrics']['all']['mse']]+[
                      M(arm,'mse') for arm in ('original_P50','GRU','MLP')])
        base_effect=min([ridge['metrics']['all']['effect_mae'],
                         last['metrics']['all']['effect_mae']]+[
                         M(arm,'effect_mae') for arm in ('original_P50','GRU','MLP')])
        gates={
          'OOD_prediction_5pct_strongest': M('factorized_P50','mse')<=.95*base_MSE,
          'OOD_effect_5pct_strongest': M('factorized_P50','effect_mae')<=.95*base_effect,
          'wrong_credit_20pct': M('factorized_P50','mse')<=.8*M('wrong_credit_factorized','mse'),
          'switched_effect_10pct_last': M('factorized_P50','effect_mae','switched') <= .90*last['metrics']['switched']['effect_mae'],
          'action_antisymmetry':fact['action_antisymmetry'],
          'cold_exact':fact['cold_exact'],
          'all_neural_validation_plateau':all(v['training']['plateau_last40'] for v in record.values())
        }
        records.append({'seed':seed,'arms':record,'gates':gates,'all_gates':all(gates.values())})
    return {
      'schema':'ttt-hybrid.native-p52.factorized-action-P50-convergence-dev-result.v1',
      'classification':'FROZEN_AUTHORED_SYNTHETIC_DEV_NOT_INDEPENDENT_SCIENCE',
      'model_sha':'7d0c253fc884b1a8979cf762518d8b1bbfedbcbc',
      'p51_parent_sha':'57bb854ca2342f5d20b94c963bb62445c5401d84',
      'source_sha256':hashes,'sampler_indices_sha256':sampler_sha,
      'model_seeds':list(MODEL_SEEDS),'parameters':params,
      'training_budget_per_arm':UPDATES,'total_neural_updates':len(MODEL_SEEDS)*len(arms)*UPDATES,
      'baseline':{'ridge':ridge,'last_effect':last},'records':records,
      'all_frozen_gates':all(r['all_gates'] for r in records),
      'total_wall_seconds':time.perf_counter()-started,
      'python':platform.python_version(),'torch':torch.__version__,
      'original_windows_linux_full_regression':'NOT_RUN',
      'real_external_source_first_use':'NOT_RUN',
      'strong_baseline_full_FLOPs_and_final_convergence':'UNVERIFIED',
      'FORGE':'UNCHANGED','deployment':'NONE'
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--receipt',type=Path,required=True)
    args=ap.parse_args()
    result=run()
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    summary={'seeds':[], 'parameters':result['parameters'],
             'source_SHA256':result['source_sha256'],
             'all_gates':result['all_frozen_gates'],
             'total_wall_seconds':result['total_wall_seconds']}
    for row in result['records']:
        summary['seeds'].append({'seed':row['seed'],'gates':row['gates'],
            'MSE':{name:v['unfamiliar']['metrics']['all']['mse'] for name,v in row['arms'].items()},
            'effect_MAE':{name:v['unfamiliar']['metrics']['all']['effect_mae'] for name,v in row['arms'].items()},
            'switched_effect_MAE':{name:v['unfamiliar']['metrics']['switched']['effect_mae'] for name,v in row['arms'].items()},
            'plateau':{name:v['training']['plateau_last40'] for name,v in row['arms'].items()}})
    print(json.dumps(summary,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
