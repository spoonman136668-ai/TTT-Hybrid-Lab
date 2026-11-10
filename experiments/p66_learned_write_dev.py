"""P66 authored DEV: relevance-admission vs factual action-consequence controls.

Only actually executed action and measured consequence enter any learner. Query
identity is unavailable to the write policy until the complete factual stream.
The original model is UNCHANGED; its normalized-key fast update is reproduced.
This is not independent scientific first-use or a neural HybridPilot experiment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

DIM, NTARGET, REPEATS, NDIST, ETA = 16, 8, 4, 32, .5
STEPS = NTARGET * REPEATS + NDIST
SOURCE = {'fit': (966101, 1024), 'val': (966103, 256), 'ood': (966107, 256)}
SEEDS = (966201, 966203, 966207)
TRAIN_UPDATES, BATCH, INDEX_SEED = 400, 64, 966299
VALIDATE_EVERY = 80


def source(split: str):
    if split not in SOURCE:
        raise ValueError('P66_UNKNOWN_FROZEN_SPLIT')
    seed, n = SOURCE[split]
    g = torch.Generator(device='cpu').manual_seed(seed)
    bases = torch.linalg.qr(torch.randn((n,DIM,DIM),generator=g)).Q
    targets = bases[:,:,:NTARGET].transpose(1,2).contiguous()
    effects = torch.tanh(torch.randn((n,NTARGET,DIM),generator=g))
    distkeys = torch.randn((n,NDIST,DIM),generator=g)
    if split == 'ood':
        projection = torch.einsum('bkd,btd->btk',targets,distkeys)
        distkeys = distkeys + 2*torch.einsum('btk,bkd->btd',projection,targets)
    distkeys = F.normalize(distkeys,dim=-1)
    dist_effects = torch.tanh(torch.randn((n,NDIST,DIM),generator=g))
    targetkeys = targets.repeat(1,REPEATS,1)
    targeteffects = effects.repeat(1,REPEATS,1)
    keys = torch.cat((targetkeys,distkeys),dim=1)
    trueeffects = torch.cat((targeteffects,dist_effects),dim=1)
    actions = torch.randint(0,2,(n,STEPS,1),generator=g).float()*2-1
    observations = actions * trueeffects + torch.randn((n,STEPS,DIM),generator=g)*.03
    query_idx = torch.randint(0,NTARGET,(n,),generator=g)
    rows = torch.arange(n)
    qkey = targets[rows,query_idx].contiguous()
    qeffect = effects[rows,query_idx]
    qa = torch.randint(0,2,(n,1),generator=g).float()*2-1
    qobs = qa*qeffect + torch.randn((n,DIM),generator=g)*.03
    # The hidden physical effects and never-acted counterfactuals do not enter
    # the returned learner data or supervision; only one actual query is seen.
    data={'keys':keys.contiguous(),'executed':actions.contiguous(),
          'observed':observations.contiguous(),'query_context':qkey,
          'query_action':qa,'query_observed':qobs}
    sha=hashlib.sha256(b''.join(v.numpy().tobytes() for v in data.values())).hexdigest()
    return data, sha


def select(data, ids):
    return {k:v[ids] for k,v in data.items()}


def replace_actual_credit(d,split):
    seed,n=SOURCE[split]
    generator=torch.Generator().manual_seed(seed+909)
    independent = torch.randint(0,2,(n,STEPS,1),generator=generator).float()*2-1
    other=dict(d)
    other['executed']=independent
    return other


def fast_update(W, k, v, gate=1.0):
    pred=torch.bmm(W,k.unsqueeze(-1)).squeeze(-1)
    return W + ETA*(v-pred).unsqueeze(-1)*k.unsqueeze(1)*gate


class LearnedGate(nn.Module):
    """4 trainable scalars, original TTT update; gate has factual inputs only."""
    def __init__(self):
        super().__init__()
        self.weights=nn.Parameter(torch.randn(4)*.05)

    def forward(self,d,return_gate=False):
        keys=d['keys']; acts=d['executed']; obs=d['observed']
        W=keys.new_zeros((len(keys),DIM,DIM))
        gates=[]
        for t in range(STEPS):
            k=keys[:,t]; v=obs[:,t]*acts[:,t]
            previous=torch.bmm(W,k.unsqueeze(-1)).squeeze(-1)
            norm=previous.norm(dim=-1)
            surprise=1-F.cosine_similarity(previous,v,dim=-1,eps=1e-6)
            w=self.weights
            g=torch.sigmoid(w[0]+w[1]*norm+w[2]*surprise+w[3]*(t/STEPS))
            W=fast_update(W,k,v,g[:,None,None]);gates.append(g)
        q=d['query_context']
        effect=torch.bmm(W,q.unsqueeze(-1)).squeeze(-1)
        physical=d['query_action']*effect
        if return_gate:
            return physical, torch.stack(gates,dim=1)
        return physical


class ConventionalGRU(nn.Module):
    """Fully exposed factual recurrent control; no future query during writing."""
    def __init__(self):
        super().__init__()
        self.rnn=nn.GRU(33,16,batch_first=True)
        self.head=nn.Sequential(nn.Linear(32,48),nn.Tanh(),nn.Linear(48,DIM))

    def forward(self,d):
        history=torch.cat((d['keys'],d['executed'],d['observed']),dim=-1)
        _,h=self.rnn(history)
        predicted_effect=self.head(torch.cat((h[-1],d['query_context']),dim=-1))
        return d['query_action']*predicted_effect


def fixed_memory(d,mode,*,threshold=0.2,seed=0,write_rate=0.5):
    W=d['keys'].new_zeros((len(d['keys']),DIM,DIM))
    acts=d['executed'];obs=d['observed'];keys=d['keys']
    gen=torch.Generator().manual_seed(seed)
    if mode=='fifo':
        fifo_keys=keys[:,-NTARGET:]
        fifo_vals=(acts*obs)[:,-NTARGET:]
        cos=torch.einsum('bkd,bd->bk',fifo_keys,d['query_context'])
        chosen=cos.argmax(1)
        estimate=fifo_vals[torch.arange(len(keys)),chosen]
        return d['query_action']*estimate
    if mode=='zero':
        return d['query_observed'].new_zeros(d['query_observed'].shape)
    for t in range(STEPS):
        k=keys[:,t];v=obs[:,t]*acts[:,t]
        pred=torch.bmm(W,k.unsqueeze(-1)).squeeze(-1)
        if mode=='all':
            gate=1.0
        elif mode=='oracle_phase':
            gate=1.0 if t<NTARGET*REPEATS else 0.0
        elif mode=='surprise':
            gate=((v-pred).norm(dim=1)>threshold).float()[:,None,None]
        elif mode=='norm':
            gate=(pred.norm(dim=1)>threshold).float()[:,None,None]
        elif mode=='random':
            gate=(torch.rand((len(keys),1,1),generator=gen)<write_rate).float()
        else:raise ValueError('P66_BAD_CONTROL')
        W=fast_update(W,k,v,gate)
    effect=torch.bmm(W,d['query_context'].unsqueeze(-1)).squeeze(-1)
    return d['query_action']*effect


def source_train_threshold(data):
    """FIT-only factual residual norm median under the original all-write rule."""
    keys=data['keys'];a=data['executed'];y=data['observed']
    W=keys.new_zeros((len(keys),DIM,DIM));r=[]
    with torch.no_grad():
        for t in range(STEPS):
            k=keys[:,t];v=y[:,t]*a[:,t]
            prev=torch.bmm(W,k.unsqueeze(-1)).squeeze(-1)
            r.append((v-prev).norm(dim=-1))
            W=fast_update(W,k,v)
    return float(torch.stack(r,1).median().item())


def train(model,fit,val,indices):
    optim=torch.optim.AdamW(model.parameters(),lr=.003)
    trajectory=[]
    start=time.perf_counter()
    for j in range(TRAIN_UPDATES):
        model.train()
        d=select(fit,indices[j])
        p=model(d)
        loss=(p-d['query_observed']).square().mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('P66_NONFINITE_TRAINING')
        optim.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optim.step()
        if (j+1)%VALIDATE_EVERY==0:
            model.eval()
            with torch.no_grad():
                q=model(val)
            trajectory.append(float((q-val['query_observed']).square().mean()))
    return {'trajectory':trajectory,'last_training_loss':float(loss.detach()),
            'plateau_last80_delta_lt_0.005':abs(trajectory[-1]-trajectory[-2])<=.005,
            'training_seconds':time.perf_counter()-start}


def mse(pred,d):
    return float((pred-d['query_observed']).square().mean().item())


def eval_model(model,data):
    model.eval()
    with torch.no_grad():
        return mse(model(data),data)


def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    beginning=time.perf_counter()
    datasets={split:source(split) for split in SOURCE}
    data={split:pair[0] for split,pair in datasets.items()}
    hashes={split:pair[1] for split,pair in datasets.items()}
    wrong={split:replace_actual_credit(data[split],split) for split in SOURCE}
    indices=torch.randint(0,SOURCE['fit'][1],(TRAIN_UPDATES,BATCH),generator=torch.Generator().manual_seed(INDEX_SEED))
    idx_sha=hashlib.sha256(indices.numpy().tobytes()).hexdigest()
    cut=source_train_threshold(select(data['fit'],torch.arange(128)))
    normvals=[]
    for threshold in (0.05,.10,.2,.3,.5):
        with torch.no_grad():
            normvals.append({'threshold':threshold,'fit128_mse':mse(fixed_memory(select(data['fit'],torch.arange(128)),'norm',threshold=threshold),select(data['fit'],torch.arange(128)))})
    best_norm=min(normvals,key=lambda row:row['fit128_mse'])['threshold']
    basic={}
    with torch.no_grad():
        for name in ('all','oracle_phase','surprise','norm','fifo','zero'):
            p=fixed_memory(data['ood'],name,threshold=cut if name=='surprise' else best_norm)
            basic[name]=mse(p,data['ood'])
    rows=[]
    for seed in SEEDS:
        arms={}
        for name,cls,train_data in (('learned_gate',LearnedGate,data),('wrong_credit_gate',LearnedGate,wrong),('GRU',ConventionalGRU,data)):
            torch.manual_seed(seed)
            model=cls()
            training=train(model,train_data['fit'],train_data['val'],indices)
            metric=eval_model(model,train_data['ood'])
            arms[name]={'mse':metric,'training':training,'params':sum(p.numel() for p in model.parameters())}
            if name=='learned_gate':
                with torch.no_grad():
                    _,gates=model(data['fit'],return_gate=True)
                arms[name]['fit_write_rate']=float(gates.mean())
                arms[name]['fit_early_write_rate']=float(gates[:,:32].mean())
                arms[name]['fit_late_write_rate']=float(gates[:,32:].mean())
        # Equal acceptance frequency random write, independent deterministic stream.
        with torch.no_grad():
            rand=fixed_memory(data['ood'],'random',seed=seed+909,write_rate=arms['learned_gate']['fit_write_rate'])
        arms['random_write']={'mse':mse(rand,data['ood']),'rate_from_fit':arms['learned_gate']['fit_write_rate']}
        c=arms['learned_gate']['mse']
        strongest=min(arms['GRU']['mse'],basic['all'],basic['surprise'],basic['norm'],basic['fifo'])
        thresholds={'ge_10pct_beat_strongest':c<=.90*strongest,
                    'ge_20pct_beat_wrong_credit':c<=.8*arms['wrong_credit_gate']['mse'],
                    'ge_15pct_beat_random_write':c<=.85*arms['random_write']['mse'],
                    'GRU_validation_plateau':arms['GRU']['training']['plateau_last80_delta_lt_0.005']}
        rows.append({'seed':seed,'arms':arms,'frozen_gates':thresholds,'all_pass':all(thresholds.values())})
        print(f'SEED {seed}: learned={c:.6f}, GRU={arms["GRU"]["mse"]:.6f}, wrong={arms["wrong_credit_gate"]["mse"]:.6f}, pass={all(thresholds.values())}',flush=True)
    return {'schema':'ttt-hybrid.p66-authored-relevance-write-dev-result.v1',
            'classification':'AUTHORED_DEV_NOT_SCIENTIFIC_FIRST_USE',
            'original_model_git_blob':'5d9ea57b827aa1796548447868fa5ee14c251b49',
            'source_sha256':hashes,'train_index_sha256':idx_sha,
            'source_seeds':SOURCE,'model_seeds':SEEDS,'train_updates_per_model':TRAIN_UPDATES,
            'new_actual_final_query_results_in_FIT':SOURCE['fit'][1],
            'train_actions_observed_before_each_query':STEPS,
            'classical_threshold_from_fit':{'surprise':cut,'norm':best_norm,'norm_grid':normvals},
            'classical_same_OOD_query_baselines':basic,'trained_arms_by_seed':rows,
            'strict_all_seed_qualification':all(row['all_pass'] for row in rows),
            'physical_source':'AUTHOR_SIMULATED_NOT_EXTERNAL',
            'phase_known_oracle_privileged':True,
            'single_fixed_phase_can_create_shortcut':True,
            'compute_parity':'NOT_PROVEN',
            'scientific_first_use':'NOT_RUN','full_windows_linux_final_sha_regression':'UNVERIFIED',
            'full_original_HybridPilot_training':'NOT_RUN',
            'no_forge_owner_or_controller_change':True,
            'wall_seconds':time.perf_counter()-beginning}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args()
    out=run()
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'all_pass':out['strict_all_seed_qualification'],'baselines':out['classical_same_OOD_query_baselines'],'elapsed':out['wall_seconds']},indent=2))


if __name__=='__main__':main()
