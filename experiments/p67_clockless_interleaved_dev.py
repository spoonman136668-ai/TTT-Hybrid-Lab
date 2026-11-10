"""P67 frozen authored DEV: clockless factual write admission under random interleaving.

Not an independently witnessed scientific first use. Uses P66's exactly pinned
original fast-matrix update and control implementations WITHOUT changing them.
Never exposes query identity or latent outcomes to the write gate in advance.
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
from experiments import p66_learned_write_dev as p66

SOURCE={'fit':(967101,768),'val':(967103,192),'ood':(967107,256)}
SEEDS=(967201,967203,967207)
TRAIN_UPDATES=600
BATCH=32
VAL_EVERY=100
INDEX_SEED=967299


def source(split):
    if split not in SOURCE: raise ValueError('P67_UNREGISTERED_SOURCE')
    old=p66.SOURCE
    try:
        p66.SOURCE=SOURCE
        d,_=p66.source(split)
    finally:
        p66.SOURCE=old
    g=torch.Generator(device='cpu').manual_seed(SOURCE[split][0]+907)
    n=len(d['keys'])
    # Each world has an independent, hidden random permutation of its causal
    # repeats and distractors, NOT a global chronological target phase.
    perms=torch.stack([torch.randperm(p66.STEPS,generator=g) for _ in range(n)])
    inds=torch.arange(n)[:,None]
    for name in ('keys','executed','observed'):
        d[name]=d[name][inds,perms].contiguous()
    assert d['keys'].shape==(n,64,16)
    digest=hashlib.sha256(b''.join(v.numpy().tobytes() for v in d.values())).hexdigest()
    return d,digest


def wrong_credit(d,split):
    gen=torch.Generator(device='cpu').manual_seed(SOURCE[split][0]+909)
    altered=dict(d)
    altered['executed']=torch.randint(0,2,d['executed'].shape,generator=gen).float()*2-1
    return altered


class ClocklessGate(nn.Module):
    """Only actual observed response, current key and 16x16 factual fast state."""
    def __init__(self):
        super().__init__()
        self.weights=nn.Parameter(torch.randn(3)*.05)

    def forward(self,d,return_gate=False):
        keys,actions,observed=d['keys'],d['executed'],d['observed']
        fast=keys.new_zeros((len(keys),16,16))
        rates=[]
        for i in range(p66.STEPS):
            k=keys[:,i]
            v=actions[:,i]*observed[:,i]
            prediction=torch.bmm(fast,k.unsqueeze(-1)).squeeze(-1)
            norm=prediction.norm(dim=1)
            contrast=1-F.cosine_similarity(prediction,v,dim=-1,eps=1e-6)
            w=self.weights
            rate=torch.sigmoid(w[0]+w[1]*norm+w[2]*contrast)
            fast=p66.fast_update(fast,k,v,rate[:,None,None])
            rates.append(rate)
        prediction=torch.bmm(fast,d['query_context'].unsqueeze(-1)).squeeze(-1)
        out=d['query_action']*prediction
        if return_gate:return out,torch.stack(rates,1)
        return out


class StrongGRU(nn.Module):
    """Fully observed conventional recurrent baseline, larger than gate."""
    def __init__(self):
        super().__init__()
        self.rnn=nn.GRU(33,32,batch_first=True)
        self.head=nn.Sequential(nn.Linear(48,64),nn.Tanh(),nn.Linear(64,16))

    def forward(self,d):
        history=torch.cat((d['keys'],d['executed'],d['observed']),dim=-1)
        _,last=self.rnn(history)
        effect=self.head(torch.cat((last[-1],d['query_context']),dim=-1))
        return d['query_action']*effect


def all_history_exact_context(d):
    """Strong factual classical benchmark; >256 floats, NOT memory matched."""
    cosine=torch.einsum('btd,bd->bt',d['keys'],d['query_context'])
    equivalent=(cosine>0.9999).to(d['observed'].dtype)
    counted=equivalent.sum(dim=1,keepdim=True).clamp_min(1.)
    assoc=(d['observed']*d['executed']*equivalent[:,:,None]).sum(1)/counted
    return d['query_action']*assoc


def trained(model,fit,val,ids):
    optim=torch.optim.AdamW(model.parameters(),lr=.003)
    validation=[]
    t0=time.perf_counter()
    for j in range(TRAIN_UPDATES):
        model.train()
        batch=p66.select(fit,ids[j])
        pred=model(batch)
        loss=(pred-batch['query_observed']).square().mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError('P67_NONFINITE_ACTUAL_TRAIN_LOSS')
        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        optim.step()
        if (j+1)%VAL_EVERY==0:
            model.eval()
            with torch.no_grad():
                v=model(val)
            validation.append(float((v-val['query_observed']).square().mean().item()))
    return {'validation_curve':validation,
            'plateau_last100_le_0p005':abs(validation[-1]-validation[-2])<=.005,
            'training_seconds':time.perf_counter()-t0,
            'final_fit_loss':float(loss.detach())}


def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    pairs={k:source(k) for k in SOURCE}
    data={k:v[0] for k,v in pairs.items()}
    sha={k:v[1] for k,v in pairs.items()}
    bad={k:wrong_credit(data[k],k) for k in SOURCE}
    idx=torch.randint(0,len(data['fit']['keys']),(TRAIN_UPDATES,BATCH),
                      generator=torch.Generator().manual_seed(INDEX_SEED))
    idx_hash=hashlib.sha256(idx.numpy().tobytes()).hexdigest()
    sample=p66.select(data['fit'],torch.arange(128))
    threshold=p66.source_train_threshold(sample)
    norm_grid=[{'cut':c,'fit_mse':p66.mse(p66.fixed_memory(sample,'norm',threshold=c),sample)}
               for c in (0.05,.1,.2,.3,.5)]
    cutoff=min(norm_grid,key=lambda z:z['fit_mse'])['cut']
    rows=[]
    for seed in SEEDS:
        arms={}
        for name,typ,dataset in (('clockless_gate',ClocklessGate,data),
                                 ('wrong_credit',ClocklessGate,bad),
                                 ('GRU',StrongGRU,data)):
            torch.manual_seed(seed)
            model=typ()
            tr=trained(model,dataset['fit'],dataset['val'],idx)
            model.eval()
            with torch.no_grad():
                prediction=model(dataset['ood'])
            mse=p66.mse(prediction,dataset['ood'])
            arms[name]={'mse':mse,'training':tr,
                        'parameters':sum(p.numel() for p in model.parameters())}
            if name=='clockless_gate':
                with torch.no_grad():_,rates=model(data['fit'],return_gate=True)
                arms[name]['fit_mean_write_rate']=float(rates.mean())
            print(f'{seed} {name} OOD_DEV_mse={mse:.6f} plateau={tr["plateau_last100_le_0p005"]}',flush=True)
        # Strong fixed baselines evaluated on the same single new OOD source,
        # never used to tune model, learning rate, selected checkpoint or threshold.
        with torch.no_grad():
            fixed={n:p66.mse(p66.fixed_memory(data['ood'],n,
                    threshold=threshold if n=='surprise' else cutoff),data['ood'])
                   for n in ('all','surprise','norm','fifo','zero')}
            fixed['all_history_unmatched_capacity']=p66.mse(all_history_exact_context(data['ood']),data['ood'])
            random=p66.mse(p66.fixed_memory(data['ood'],'random',seed=seed+909,
                          write_rate=arms['clockless_gate']['fit_mean_write_rate']),data['ood'])
        c=arms['clockless_gate']['mse']
        best=min(arms['GRU']['mse'],*[fixed[k] for k in ('all','surprise','norm','fifo')])
        gates={'gain_10pct_vs_best_memory_matched_or_neural':c<=.9*best,
               'gain_20pct_vs_wrong_credit':c<=.8*arms['wrong_credit']['mse'],
               'gain_15pct_vs_random':c<=.85*random,
               'GRU_val_last100_plateau':arms['GRU']['training']['plateau_last100_le_0p005']}
        rows.append({'seed':seed,'arms':arms,'fixed_controls':fixed,
                     'random_write_mse':random,'frozen_gates':gates,
                     'all_gates_pass':all(gates.values())})
        print(f'seed={seed} all_frozen_gates={all(gates.values())}',flush=True)
    return {'schema':'ttt-hybrid.p67-clockless-random-interleaving-factual-dev.v1',
            'source':'FRESH_AUTHORED_DEV_NOT_INDEPENDENT_SCIENCE',
            'sources_sha256':sha,'sampler_sha256':idx_hash,
            'model_seeds':SEEDS,'data_source':SOURCE,'neural_steps_each':TRAIN_UPDATES,
            'batch':BATCH,'training_arms':['clockless_gate','wrong_credit','GRU'],
            'total_actual_optimizer_steps':len(SEEDS)*3*TRAIN_UPDATES,
            'frozen_control_thresholds':{'surprise':threshold,'norm':cutoff,'norm_grid':norm_grid},
            'models':rows,'strict_joint_all_3_seeds_pass':all(row['all_gates_pass'] for row in rows),
            'all_history_exact_control_not_capacity_matched':True,
            'real_world_physical_actions':'NOT_RUN','science_first_use':'NOT_RUN',
            'original_HybridPilot_full_train':'NOT_RUN',
            'full_Windows_Linux_regression':'NOT_RUN','FORGE':'UNCHANGED',
            'elapsed_seconds':time.perf_counter()-start}


def main():
    args=argparse.ArgumentParser()
    args.add_argument('--receipt',required=True,type=Path)
    a=args.parse_args()
    out=run()
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'joint_pass':out['strict_joint_all_3_seeds_pass'],
          'source_sha256':out['sources_sha256'],'elapsed':out['elapsed_seconds']},indent=2))

if __name__=='__main__':main()
