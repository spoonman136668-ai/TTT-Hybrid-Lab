"""P61 DEV: reward-only learning from six completed noisy motor probes with aliased current sensor.

New authored scalar dynamics, NOT official Gymnasium, not owner-qualified science.
No latent polarity or unexecuted final rewards are given to any model.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from model import Configuration, HybridPilot

SPLITS = {'fit': (991101,4096,.50,.24), 'val': (991103,512,.50,.24),
          'unfamiliar_dev': (991107,1024,.75,.30)}
MODEL_SEEDS = (992101,992103,992107)
INDEX_SEED = 992199
UPDATES = 480
BATCH = 128
VAL_EVERY = 80
PROBES = 6
FEATURES = 5


def source(split):
    if split not in SPLITS:
        raise ValueError('P61_UNREGISTERED_SOURCE')
    seed,n,switch_p,noise = SPLITS[split]
    rng = np.random.default_rng(seed)
    initial = rng.choice(np.array([-1,1],dtype=np.int8),size=n)
    switched = rng.random(n) < switch_p
    actions = rng.choice(np.array([-1.,1.],dtype=np.float32),size=(n,PROBES))
    factual_mode = initial[:,None] * np.where(switched[:,None] &
                               (np.arange(PROBES)[None,:]>=3),-1,1)
    observed = (0.18*factual_mode*actions+rng.normal(0,noise,(n,PROBES))).astype(np.float32)
    h = np.zeros((n,PROBES,FEATURES),dtype=np.float32)
    h[:,:,0]=actions
    h[:,:,1]=observed
    h[:,:,2]=(np.arange(PROBES)+1)/PROBES
    final_sign = (initial*np.where(switched,-1,1)).astype(np.float32)
    query_action = rng.choice(np.array([-1.,1.],dtype=np.float32),size=n)
    # Revealed only AFTER real query action is executed; one factual reward, no synthetic counterfactual FIT labels.
    final_reward = np.where(query_action==final_sign,1.,-1.).astype(np.float32)
    raw={'history':h, 'actual_query_action':query_action[:,None],
         'actual_observed_reward':final_reward[:,None],
         'evaluator_hidden_final_mode':final_sign[:,None],
         'evaluator_switched':switched}
    sha=hashlib.sha256(b''.join(raw[k].tobytes() for k in raw)).hexdigest()
    return {k:torch.from_numpy(v) for k,v in raw.items()}, sha


def factual_input(data,proposed_actions,wrong_credit=False,drop_history=False):
    h=data['history']
    b=h.shape[0]
    if h.shape != (b,PROBES,FEATURES) or proposed_actions.ndim!=3 or proposed_actions.shape[0]!=b or proposed_actions.shape[2]!=1:
        raise ValueError('P61_BAD_FACTUAL_HISTORY_OR_QUERY')
    if not bool(torch.isfinite(h).all() and torch.isfinite(proposed_actions).all()):
        raise ValueError('P61_NONFINITE_ACTUAL_FACT')
    if not bool((proposed_actions.abs()==1).all()):
        raise ValueError('P61_QUERY_MUST_BE_REAL_MOTOR_CHOICE')
    k=proposed_actions.shape[1]
    facts=h.clone()
    if drop_history: facts.zero_()
    if wrong_credit: facts[:,:,0]=torch.roll(facts[:,:,0],shifts=1,dims=1)
    query=h.new_zeros((b,k,FEATURES))
    query[:,:,2]=1
    query[:,:,3]=1
    query[:,:,4]=proposed_actions[:,:,0]
    return torch.cat((facts[:,None,:,:].expand(-1,k,-1,-1),query[:,:,None,:]),dim=2)


class HybridQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=HybridPilot(FEATURES,1,Configuration(width=8,heads=2,ssm=True,attention=True,ttt=True,depth=True))
    def forward(self,d,acts,wrong_credit=False,ttt_update=True):
        z=factual_input(d,acts,wrong_credit)
        b,k=z.shape[:2]
        return self.body(z.reshape(b*k,PROBES+1,FEATURES),ttt_update=ttt_update)[:,-1,0].reshape(b,k)


class GRUQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.GRU(FEATURES,17,batch_first=True)
        self.head=nn.Linear(17,1)
    def forward(self,d,acts,wrong_credit=False,ttt_update=True):
        z=factual_input(d,acts,wrong_credit)
        b,k=z.shape[:2]
        _,h=self.body(z.reshape(b*k,PROBES+1,FEATURES))
        return self.head(h[-1]).reshape(b,k)


class HistoryMLPQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Linear((PROBES+1)*FEATURES,22),nn.Tanh(),nn.Linear(22,18),nn.Tanh(),nn.Linear(18,1))
    def forward(self,d,acts,wrong_credit=False,ttt_update=True):
        z=factual_input(d,acts,wrong_credit)
        b,k=z.shape[:2]
        return self.body(z.reshape(b*k,-1)).reshape(b,k)


class MemorylessQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(FEATURES,20),nn.Tanh(),nn.Linear(20,20),nn.Tanh(),nn.Linear(20,1))
    def forward(self,d,acts,wrong_credit=False,ttt_update=True):
        z=factual_input(d,acts,wrong_credit)
        return self.body(z[:,:,-1,:]).squeeze(-1)


FACTORIES={'Hybrid':HybridQ,'GRU':GRUQ,'history_MLP':HistoryMLPQ,
           'memoryless':MemorylessQ,'wrong_credit_Hybrid':HybridQ}


def public_data(d,idx=None):
    allowed=('history','actual_query_action','actual_observed_reward')
    return {k:(d[k] if idx is None else d[k][idx]) for k in allowed}


def score_model(model,source_data,wrong_credit=False,ttt_update=True):
    data=public_data(source_data)
    b=len(data['history'])
    candidates=torch.tensor([[-1.],[1.]],dtype=torch.float32).unsqueeze(0).expand(b,-1,-1)
    model.eval()
    begin=time.perf_counter()
    with torch.no_grad():
        q=model(data,candidates,wrong_credit,ttt_update=ttt_update)
    action=torch.where(q[:,1]>q[:,0],torch.ones(b),-torch.ones(b))
    # Only selected actual physical action has reward revealed. This is not training.
    realized_reward=torch.where(action==source_data['evaluator_hidden_final_mode'][:,0],1.,-1.)
    switched=source_data['evaluator_switched']
    return {
        'success':float((realized_reward==1.).float().mean()),
        'mean_realized_reward':float(realized_reward.mean()),
        'success_switched':float((realized_reward[switched]==1.).float().mean()),
        'success_stable':float((realized_reward[~switched]==1.).float().mean()),
        'actual_final_motor_actions':b,
        'inference_seconds':time.perf_counter()-begin
    }


def score_analytical(d,n):
    actual=d['history'][:,-n:,:]
    s=(actual[:,:,0]*actual[:,:,1]).sum(1)
    chosen=torch.where(s>=0,torch.ones_like(s),-torch.ones_like(s))
    z=d['evaluator_hidden_final_mode'][:,0]
    success=chosen==z
    sw=d['evaluator_switched']
    return {'success':float(success.float().mean()),'mean_realized_reward':float((2.*success.float()-1.).mean()),
            'success_switched':float(success[sw].float().mean()),
            'success_stable':float(success[~sw].float().mean()),'actual_final_motor_actions':len(chosen)}


def param_count(model):return sum(p.numel() for p in model.parameters())


def train(model,fit,val,indexes,wrong_credit=False):
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
    curve=[];start=time.perf_counter()
    for step in range(UPDATES):
        model.train()
        d=public_data(fit,indexes[step])
        act=d['actual_query_action'][:,None,:]
        q=model(d,act,wrong_credit=wrong_credit)[:,0]
        loss=(q-d['actual_observed_reward'][:,0]).square().mean()
        if not bool(torch.isfinite(loss)):raise ValueError('P61_NONFINITE_FIT_LOSS')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if (step+1)%VAL_EVERY==0:
            v=score_model(model,val,wrong_credit=wrong_credit)
            curve.append(v['success'])
    return {'validation_accuracy_curve':curve,
            'plateau_last80_abs_accuracy_delta_le_1pp':abs(curve[-1]-curve[-2])<=.01,
            'last_factual_fit_loss':float(loss.detach()),
            'fit_seconds':time.perf_counter()-start}


def cold_restore(model,data,wrong_credit=False):
    d=public_data(data,torch.arange(5))
    candidates=torch.tensor([[[-1.],[1.]]],dtype=torch.float32).expand(5,-1,-1)
    model.eval()
    with torch.no_grad():before=model(d,candidates,wrong_credit)
    buf=io.BytesIO()
    torch.save({'weights':model.state_dict(),'factual_history':d['history']},buf)
    buf.seek(0)
    obj=torch.load(buf,weights_only=True)
    restored=type(model)().eval()
    restored.load_state_dict(obj['weights'])
    with torch.no_grad():after=restored(d,candidates,wrong_credit)
    return bool(torch.equal(before,after) and torch.equal(obj['factual_history'],d['history'])),hashlib.sha256(buf.getvalue()).hexdigest()


def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    wall=time.perf_counter()
    parts={name:source(name) for name in SPLITS}
    data={key:pair[0] for key,pair in parts.items()}
    hashes={key:pair[1] for key,pair in parts.items()}
    torch_generator=torch.Generator().manual_seed(INDEX_SEED)
    indices=torch.randint(len(data['fit']['history']),(UPDATES,BATCH),generator=torch_generator)
    index_sha=hashlib.sha256(indices.numpy().tobytes()).hexdigest()
    analytic={n:score_analytical(data['unfamiliar_dev'],k) for n,k in [('recent_3',3),('all_6',6)]}
    params={name:param_count(factory()) for name,factory in FACTORIES.items()}
    rows=[]
    for seed in MODEL_SEEDS:
        outcome={}
        for name,factory in FACTORIES.items():
            torch.manual_seed(seed)
            model=factory()
            wrong=name=='wrong_credit_Hybrid'
            training=train(model,data['fit'],data['val'],indices,wrong_credit=wrong)
            metric=score_model(model,data['unfamiliar_dev'],wrong_credit=wrong)
            exact,weights_sha=cold_restore(model,data['val'],wrong_credit=wrong)
            outcome[name]={'train':training,'OOD':metric,'cold_exact':exact,'weights_sha256':weights_sha}
            if name=='Hybrid':
                outcome['TTT_disabled']={'OOD':score_model(model,data['unfamiliar_dev'],ttt_update=False)}
            print(f'{seed} {name} success={metric["success"]:.4f} elapsed_train={training["fit_seconds"]:.2f}',flush=True)
        h=outcome['Hybrid']['OOD']['success']
        strongest=max(outcome['GRU']['OOD']['success'],outcome['history_MLP']['OOD']['success'],
                      analytic['recent_3']['success'],analytic['all_6']['success'])
        gates={
            'hybrid_success_ge_75pct':h>=.75,
            'hybrid_5pp_over_strongest':h>=strongest+.05,
            'hybrid_10pp_over_wrong_credit':h>=outcome['wrong_credit_Hybrid']['OOD']['success']+.10,
            'hybrid_5pp_over_TTT_off':h>=outcome['TTT_disabled']['OOD']['success']+.05,
            'GRU_15pp_over_memoryless':outcome['GRU']['OOD']['success']>=outcome['memoryless']['OOD']['success']+.15,
            'all_neural_validation_plateau':all(outcome[name]['train']['plateau_last80_abs_accuracy_delta_le_1pp'] for name in FACTORIES),
            'all_cold_exact':all(outcome[name]['cold_exact'] for name in FACTORIES)
        }
        rows.append({'seed':seed,'arms':outcome,'gates':gates,'all_gates_pass':all(gates.values())})
    return {
      'schema':'ttt-hybrid.p61-hidden-motor-polarity-factual-Q-DEV-result.v1',
      'qualification':'AUTHORED_DEVELOPMENT_ONLY_NOT_EXTERNAL_SCIENCE',
      'source_sha256':hashes,'sampler_indices_sha256':index_sha,'source_seeds':SPLITS,
      'model_seeds':MODEL_SEEDS,'params':params,'fit_steps_per_arm':UPDATES,
      'actual_physical_probe_actions_fit':SPLITS['fit'][1]*PROBES,
      'actual_final_actions_fit':SPLITS['fit'][1],
      'analytical_controls':analytic,'rows':rows,'combined_pass':all(r['all_gates_pass'] for r in rows),
      'total_elapsed_seconds':time.perf_counter()-wall,
      'python':platform.python_version(),'torch':torch.__version__,
      'original_model_git_blob':'5d9ea57b827aa1796548447868fa5ee14c251b49',
      'official_Gymnasium':'NOT_APPLICABLE_CUSTOM_POMDP',
      'source_first_use':'NOT_AUTHORIZED',
      'exact_windows_linux_full_regression':'NOT_EXECUTED',
      'FORGE':'UNCHANGED','model_promotion':'NONE'
    }


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--receipt',type=Path,required=True)
    args=a.parse_args()
    result=run();args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'strong_controls':result['analytical_controls'],
        'seeds':[{'seed':r['seed'],'gates':r['gates'],
        'success':{n:z['OOD']['success'] for n,z in r['arms'].items()}} for r in result['rows']],
        'overall':result['combined_pass'],'seconds':result['total_elapsed_seconds']},indent=2))

if __name__=='__main__':main()
