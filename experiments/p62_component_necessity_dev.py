"""P62: separately trained native ablations of actual factual reward learning.

Frozen authored development ONLY. Read original PyTorch HybridPilot, no new
model family, independent science, original runner, GPU, scheduler, or FORGE.
The trained Q heads observe one actually executed final action/reward per episode;
they never receive the latent true polarity or alternative action reward.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from model import Configuration, HybridPilot

SPLITS = {
    'fit': (993201, 4096, .50, .24, .30, .12, 993221),
    'val': (993203, 512, .50, .24, .30, .12, 993223),
    'unfamiliar_dev': (993207, 1024, .75, .30, .45, .18, 993227),
}
MODEL_SEEDS = (994101, 994103, 994107)
SAMPLER_SEED = 994199
UPDATES, BATCH, VAL_EVERY = 480, 128, 80
PROBES, FEATURES = 6, 5
CANDIDATES = torch.tensor([-1.,1.]).reshape(1,2,1)


def source(split):
    if split not in SPLITS:
        raise ValueError('P62_UNREGISTERED_SOURCE')
    seed, n, flip_probability, sigma, max_offset, max_trend, fake_seed = SPLITS[split]
    rng = np.random.default_rng(seed)
    initial = rng.choice(np.array([-1,1],np.int8), n)
    flipped = rng.random(n) < flip_probability
    actions = rng.choice(np.array([-1.,1.],np.float32),(n,PROBES))
    modes = initial[:,None] * np.where(flipped[:,None] & (np.arange(PROBES)[None,:]>=3),-1,1)
    offset = rng.uniform(-max_offset,max_offset,n)
    trend = rng.uniform(-max_trend,max_trend,n)
    index = np.arange(PROBES)
    factual = (0.18*modes*actions+offset[:,None]+trend[:,None]*(index[None,:]-2.5)/3
               +rng.normal(0,sigma,(n,PROBES))).astype(np.float32)
    h = np.zeros((n,PROBES,FEATURES),np.float32)
    h[:,:,0],h[:,:,1],h[:,:,2] = actions,factual,(index+1)/PROBES
    final_mode = (initial*np.where(flipped,-1,1)).astype(np.float32)
    actual_query = rng.choice(np.array([-1.,1.],np.float32),n).reshape(-1,1)
    actual_reward = np.where(actual_query[:,0]==final_mode,1.,-1.).astype(np.float32).reshape(-1,1)
    # Deliberately wrong-credit input is a non-invertible independent draw.
    corrupted = np.random.default_rng(fake_seed).choice(np.array([-1.,1.],np.float32),(n,PROBES))
    items = {'history':h, 'actual_query_action':actual_query, 'actual_observed_reward':actual_reward,
             'destroyed_credit_actions':corrupted, 'evaluator_final_polarity':final_mode,
             'evaluator_switched':flipped, 'evaluator_hidden_offset':offset,
             'evaluator_hidden_trend':trend}
    sha = hashlib.sha256(b''.join(np.ascontiguousarray(items[k]).tobytes() for k in items)).hexdigest()
    return {k:torch.from_numpy(v) for k,v in items.items()},sha


def factual_only(data,ids=None):
    keys=('history','actual_query_action','actual_observed_reward','destroyed_credit_actions')
    return {k:(data[k] if ids is None else data[k][ids]) for k in keys}


def inputs(d,actions,wrong_credit=False):
    h=d['history']
    b=h.shape[0]
    if h.ndim != 3 or tuple(h.shape[1:])!=(PROBES,FEATURES) or actions.shape[0]!=b or actions.ndim!=3 or actions.shape[2]!=1 or not 1<=actions.shape[1]<=16:
        raise ValueError('P62_INVALID_FACTUAL_INPUT')
    if not bool(torch.isfinite(h).all() and torch.isfinite(actions).all()):
        raise ValueError('P62_NONFINITE_FACTUAL_INPUT')
    if not bool((actions.abs()==1).all()):
        raise ValueError('P62_INVALID_ACTUAL_ACTION')
    facts=h.clone()
    if wrong_credit:
        if d['destroyed_credit_actions'].shape != (b,PROBES):
            raise ValueError('P62_BAD_WRONG_CREDIT_SHAPE')
        facts[:,:,0]=d['destroyed_credit_actions']
    k=actions.shape[1]
    query=h.new_zeros((b,k,FEATURES))
    query[:,:,2]=1
    query[:,:,3]=1
    query[:,:,4]=actions[:,:,0]
    return torch.cat((facts[:,None,:,:].expand(-1,k,-1,-1),query[:,:,None,:]),dim=2)


class HybridQ(nn.Module):
    def __init__(self, missing=None):
        super().__init__()
        cfg = Configuration(width=8,heads=2,ssm=missing!='SSM',attention=missing!='attention',
                            ttt=missing!='TTT',depth=missing!='depth')
        self.body=HybridPilot(FEATURES,1,cfg)
    def forward(self,d,acts,wrong_credit=False,ttt_update=True):
        z=inputs(d,acts,wrong_credit)
        b,k=z.shape[:2]
        return self.body(z.reshape(b*k,PROBES+1,FEATURES),ttt_update=ttt_update)[:,-1,0].reshape(b,k)


class GRUQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.GRU(FEATURES,17,batch_first=True)
        self.head=nn.Linear(17,1)
    def forward(self,d,acts,wrong_credit=False,ttt_update=True):
        z=inputs(d,acts,wrong_credit)
        b,k=z.shape[:2]
        _,h=self.body(z.reshape(b*k,PROBES+1,FEATURES))
        return self.head(h[-1]).reshape(b,k)


class MLPQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Linear((PROBES+1)*FEATURES,22),nn.Tanh(),
                                nn.Linear(22,18),nn.Tanh(),nn.Linear(18,1))
    def forward(self,d,acts,wrong_credit=False,ttt_update=True):
        z=inputs(d,acts,wrong_credit)
        b,k=z.shape[:2]
        return self.body(z.reshape(b*k,-1)).reshape(b,k)


FACTORIES={
    'Hybrid':lambda:HybridQ(),
    'no_TTT':lambda:HybridQ('TTT'),
    'no_SSM':lambda:HybridQ('SSM'),
    'no_attention':lambda:HybridQ('attention'),
    'no_depth':lambda:HybridQ('depth'),
    'GRU':lambda:GRUQ(),
    'history_MLP':lambda:MLPQ(),
    'destroyed_credit_Hybrid':lambda:HybridQ(),
}


def scores_from_action(chosen,data):
    final=data['evaluator_final_polarity']
    success=chosen==final
    flipped=data['evaluator_switched']
    return {'success':float(success.float().mean()),
            'mean_reward':float((2*success.float()-1).mean()),
            'success_flipped':float(success[flipped].float().mean()),
            'success_stable':float(success[~flipped].float().mean()),
            'number_executed_final_actions':len(chosen)}


def score_model(model,data,wrong_credit=False,ttt_update=True):
    x=factual_only(data)
    acts=CANDIDATES.expand(x['history'].shape[0],-1,-1)
    model.eval()
    start=time.perf_counter()
    with torch.no_grad():
        pred=model(x,acts,wrong_credit=wrong_credit,ttt_update=ttt_update)
    action=torch.where(pred[:,1]>pred[:,0],torch.ones(len(pred)),-torch.ones(len(pred)))
    record=scores_from_action(action,data)
    record['inference_seconds']=time.perf_counter()-start
    return record


def analytical(data,n):
    history=data['history'][:,-n:]
    s=(history[:,:,0]*history[:,:,1]).sum(1)
    return scores_from_action(torch.where(s>=0,torch.ones_like(s),-torch.ones_like(s)),data)


def fitted_affine(data):
    '''Same six realized facts, rank-3 nuisance + actual action least-squares.'''
    x=data['history']
    actions=x[:,:,0].double()
    outcomes=x[:,:,1].double()
    clock=(torch.arange(PROBES,dtype=torch.float64)-2.5)/3
    design=torch.stack((torch.ones_like(actions), clock[None,:].expand_as(actions),actions),-1)
    rank=torch.linalg.matrix_rank(design)
    good=rank==3
    # Batch least squares via pinv, including rank-deficient rows that use a
    # prospectively specified zero-training recent-3 fallback.
    coef=torch.linalg.pinv(design)@outcomes.unsqueeze(-1)
    estim=coef[:,2,0]
    recent=(x[:,-3:,0]*x[:,-3:,1]).sum(1).double()
    sign=torch.where(good,estim,recent)
    choice=torch.where(sign>=0,torch.ones_like(sign),-torch.ones_like(sign)).float()
    record=scores_from_action(choice,data)
    record['fraction_full_rank']=float(good.float().mean())
    record['fallback']='recent_three_if_rank_deficient'
    return record


def train(model,fit,val,batches,wrong_credit=False):
    optim=torch.optim.AdamW(model.parameters(),lr=.003)
    curve=[];losses=[]; start=time.perf_counter()
    for t,idx in enumerate(batches):
        model.train()
        d=factual_only(fit,idx)
        factual_action=d['actual_query_action'].unsqueeze(1)
        predicted=model(d,factual_action,wrong_credit=wrong_credit)[:,0]
        loss=(predicted-d['actual_observed_reward'][:,0]).square().mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError('P62_NONFINITE_LOSS')
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optim.step()
        if (t+1)%VAL_EVERY==0:
            v=score_model(model,val,wrong_credit=wrong_credit)
            curve.append(v['success']);losses.append(float(loss.detach()))
    return {'val_success_every_80':curve,'fit_loss_at_checkpoints':losses,
            'plateau_last_80_abs_accuracy_delta_le_1pp':abs(curve[-1]-curve[-2])<=.01,
            'last_fit_loss':float(loss.detach()),'train_seconds':time.perf_counter()-start}


def cold_restore(model,factory,data,wrong_credit=False):
    d=factual_only(data,torch.arange(5))
    a=CANDIDATES.expand(5,-1,-1)
    model.eval()
    with torch.no_grad(): p=model(d,a,wrong_credit)
    buff=io.BytesIO()
    torch.save({'weights':model.state_dict(), 'facts':d},buff)
    buff.seek(0)
    cold=factory().eval()
    recovered=torch.load(buff,weights_only=True)
    cold.load_state_dict(recovered['weights'])
    with torch.no_grad():q=cold(recovered['facts'],a,wrong_credit)
    return bool(torch.equal(p,q) and all(torch.equal(d[k],recovered['facts'][k]) for k in d)),hashlib.sha256(buff.getvalue()).hexdigest()


def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    raw={k:source(k) for k in SPLITS}
    streams={k:z[0] for k,z in raw.items()}
    hashes={k:z[1] for k,z in raw.items()}
    batches=torch.randint(len(streams['fit']['history']),(UPDATES,BATCH),
                          generator=torch.Generator().manual_seed(SAMPLER_SEED))
    index_sha=hashlib.sha256(batches.numpy().tobytes()).hexdigest()
    baseline={name:analytical(streams['unfamiliar_dev'],k) for name,k in [('recent_three',3),('all_six',6)]}
    baseline['affine_OLS']=fitted_affine(streams['unfamiliar_dev'])
    params={name:sum(x.numel() for x in fac().parameters()) for name,fac in FACTORIES.items()}
    rows=[]
    for seed in MODEL_SEEDS:
        arms={}
        for name,factory in FACTORIES.items():
            torch.manual_seed(seed)
            m=factory();wrong=name=='destroyed_credit_Hybrid'
            learning=train(m,streams['fit'],streams['val'],batches,wrong_credit=wrong)
            perf=score_model(m,streams['unfamiliar_dev'],wrong_credit=wrong)
            exact,cold_hash=cold_restore(m,factory,streams['val'],wrong_credit=wrong)
            arms[name]={'training':learning,'OOD':perf,'cold_exact':exact,'cold_bytes_sha256':cold_hash}
            if name=='Hybrid':
                arms['post_train_TTT_disabled']={'OOD':score_model(m,streams['unfamiliar_dev'],ttt_update=False)}
            print(f'P62 seed={seed} arm={name} OOD={perf["success"]:.5f} train={learning["train_seconds"]:.2f}s',flush=True)
        h=arms['Hybrid']['OOD']['success']
        best=max(arms['GRU']['OOD']['success'],arms['history_MLP']['OOD']['success'],
                 *[v['success'] for v in baseline.values()])
        gates={
            'Hybrid_success_at_least_75pct':h>=.75,
            'Hybrid_at_least_3pp_over_strongest_conventional':h>=best+.03,
            'Hybrid_at_least_2pp_over_separately_trained_no_TTT':h>=arms['no_TTT']['OOD']['success']+.02,
            'Hybrid_at_least_10pp_over_destroyed_credit':h>=arms['destroyed_credit_Hybrid']['OOD']['success']+.10,
            'all_train_arms_validation_plateau':all(arms[k]['training']['plateau_last_80_abs_accuracy_delta_le_1pp'] for k in FACTORIES),
            'all_cold_exact':all(arms[k]['cold_exact'] for k in FACTORIES),
        }
        rows.append({'seed':seed,'arms':arms,'strongest_conventional_OOD':best,
                     'gates':gates,'all_gates_pass':all(gates.values())})
    return {'schema':'ttt-hybrid.p62-separately-trained-component-necessity-dev-result.v1',
            'source_qualification':'AUTHORED_DEV_NOT_EXTERNAL_SCIENCE',
            'predeclared_freeze_sha':'8dc66c413a672500c84a6fb85fcb5e9dad7e983c',
            'original_model_blob':'5d9ea57b827aa1796548447868fa5ee14c251b49',
            'source_sha256':hashes,'sampler_sha256':index_sha,
            'source_seeds_and_sizes':SPLITS,'model_seeds':MODEL_SEEDS,'fit_updates_per_arm':UPDATES,
            'fit_probe_actions':4096*PROBES,'fit_final_executed_reward_actions':4096,
            'params':params,'analytical_controls':baseline,'rows':rows,
            'registered_all_seed_PASS':all(z['all_gates_pass'] for z in rows),
            'total_wall_seconds':time.perf_counter()-start,
            'python':platform.python_version(),'torch':torch.__version__,
            'full_original_windows_and_linux_final_regression':'NOT_EXECUTED',
            'independent_science':'NOT_EXECUTED', 'promotion':'NONE','FORGE':'UNCHANGED'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--receipt',type=Path,required=True)
    args=ap.parse_args()
    result=run();args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print('P62_REGISTERED_SUMMARY',json.dumps({
        'seeds':[{'seed':x['seed'],'gates':x['gates'],
                  'ood':{k:y['OOD']['success'] for k,y in x['arms'].items()}} for x in result['rows']],
        'analytical':result['analytical_controls'],'all_pass':result['registered_all_seed_PASS'],
        'seconds':result['total_wall_seconds']}),flush=True)

if __name__=='__main__':main()
