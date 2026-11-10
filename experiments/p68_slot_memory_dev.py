"""P68 pre-registered synthetic DEV memory comparison. No original Hybrid edits.

P68 source seeds and rules were frozen in Git commit 7ae6adda4671aae50f212445bd86e6bd43c8492b
before writing or executing this module. Learners see factual executed simulator
(action, observed effect) pairs only; final actual query is withheld until readout.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import platform
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

D, SLOTS, ETA = 16, 8, 0.5
SEEDS = {'fit': 980111, 'val': 980113, 'ood_dev': 980117, 'orthogonal': 980119}
N = {'fit': 512, 'val': 128, 'ood_dev': 128, 'orthogonal': 64}
MODEL_SEEDS = (980201, 980203, 980207)
FIT_UPDATES, BATCH, CHECK = 180, 32, 30


def digest(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def make_world(split, n=None):
    if split not in SEEDS:
        raise ValueError('P68_UNKNOWN_SPLIT')
    n = N[split] if n is None else n
    if n <= 0 or n > N[split]:
        raise ValueError('P68_BAD_WORLD_COUNT')
    g = torch.Generator().manual_seed(SEEDS[split])
    ood = split == 'ood_dev'
    reps, noise = (4, 48) if ood else (6, 16)
    q = torch.linalg.qr(torch.randn(n, D, D, generator=g)).Q
    # Vary mappings independently across the never-reused source families.
    base_keys = q[:, :, :SLOTS].transpose(1, 2).contiguous()
    base_effects = torch.tanh(torch.randn(n, SLOTS, D, generator=g) * (1.2 if ood else 1.0))
    event_keys = base_keys.repeat(1, reps, 1)
    event_effects = base_effects.repeat(1, reps, 1)
    distractor_keys = F.normalize(torch.randn(n, noise, D, generator=g), dim=-1)
    if split == 'orthogonal':
        distractor_keys = F.normalize(torch.bmm(torch.randn(n, noise, 8, generator=g), q[:, :, 8:].transpose(1, 2)), dim=-1)
    distractor_effects = torch.tanh(torch.randn(n, noise, D, generator=g))
    keys = torch.cat((event_keys, distractor_keys), 1)
    effects = torch.cat((event_effects, distractor_effects), 1)
    roles = torch.cat((torch.ones(n, SLOTS * reps, dtype=torch.bool),
                       torch.zeros(n, noise, dtype=torch.bool)), 1)
    order = torch.rand(n, keys.shape[1], generator=g).argsort(1)
    keys = keys.gather(1, order[:, :, None].expand_as(keys))
    effects = effects.gather(1, order[:, :, None].expand_as(effects))
    roles = roles.gather(1, order)
    chosen = torch.randint(SLOTS, (n,), generator=g)
    query = base_keys[torch.arange(n), chosen]
    true_effect = base_effects[torch.arange(n), chosen]
    wrong = F.normalize(torch.randn(n, keys.shape[1], D, generator=g), dim=-1)
    return {'action':keys, 'effect':effects, 'query_action':query,
            'observed_query_effect':true_effect, 'role_audit':roles,
            'wrong_credit_key':wrong}


def factual(w):
    return {'action': w['action'], 'effect': w['effect']}


def subset(w, ids):
    return {k: x[ids] for k, x in w.items()}


def source_hash(w):
    return hashlib.sha256(b''.join(w[k].contiguous().numpy().tobytes() for k in
       ('action','effect','query_action','observed_query_effect','role_audit','wrong_credit_key'))).hexdigest()


def positional_audit(roles):
    n, length = roles.shape
    assert length % 4 == 0 and n >= 4
    frac = roles.float().reshape(n,4,length//4).mean((0,2))
    imbalance = float((frac-roles.float().mean()).abs().max())
    training, held = roles[:n//2], roles[n//2:]
    pred = (training.float().mean(0) >= .5)[None].expand_as(held)
    tpr = ((pred & held).sum().float() / held.sum().clamp_min(1)).item()
    tnr = (((~pred) & (~held)).sum().float() / (~held).sum().clamp_min(1)).item()
    return {'max_quartile_imbalance':imbalance,'heldout_index_only_balanced_accuracy':.5*(tpr+tnr),
            'quartile_relevant_fractions':frac.tolist(),
            'pass':bool(imbalance<=.05 and .5*(tpr+tnr)<=.55)}


class WriteGate(nn.Module):
    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.theta = nn.Parameter((torch.rand(4, generator=g)-.5)*.1)

    def forward(self, events, carry=None, lo=0, hi=None, *, mode='learned', wrong=None):
        a, effects = events['action'], events['effect']
        if a.shape != effects.shape or a.ndim != 3 or a.shape[-1] != D:
            raise ValueError('P68_BAD_FACTS')
        if not bool(torch.isfinite(a).all() and torch.isfinite(effects).all()):
            raise ValueError('P68_NONFINITE_FACTS')
        count, steps, _ = a.shape
        hi = steps if hi is None else hi
        if not (0 <= lo <= hi <= steps): raise ValueError('P68_BAD_RANGE')
        w = a.new_zeros((count,D,D)) if carry is None else carry
        if w.shape != (count,D,D): raise ValueError('P68_BAD_CARRY')
        for step in range(lo,hi):
            key = a[:,step] if wrong is None else wrong[:,step]
            val = effects[:,step]
            pred = (w@key[:,:,None]).squeeze(-1)
            residual = val-pred
            error = residual.square().mean(-1)
            if mode=='learned':
                features = torch.stack((torch.ones_like(error), torch.log(error+1e-6),
                           torch.log(pred.square().mean(-1)+1e-6),
                           F.cosine_similarity(pred,val,dim=-1,eps=1e-6)), -1)
                weight = torch.sigmoid(features@self.theta)
            elif mode=='all': weight = torch.ones_like(error)
            else: raise ValueError('P68_BAD_MODE')
            w = w + ETA*weight[:,None,None]*residual[:,:,None]*key[:,None,:]
        return w


def fast_predict(w, query):
    return (w@query[:,:,None]).squeeze(-1)


def cache_stream(events, *, frequency=True):
    """Eight 16+16 float slots; key magnitude also encodes hit count.

    No role labels, extra carried counters, future queries or hidden source data.
    Frequencies are retrieved from norms of the keys already in the 256-float
    persistent cache. Deterministic eviction/skip when min count > 1.
    """
    a, y = events['action'], events['effect']
    if a.shape != y.shape or a.ndim!=3 or a.shape[-1] != D:
        raise ValueError('P68_BAD_FACTS')
    batch, steps, _ = a.shape
    keys = torch.zeros(batch,SLOTS,D,dtype=a.dtype)
    values = torch.zeros_like(keys)
    row = torch.arange(batch)
    for t in range(steps):
        k, v = a[:,t], y[:,t]
        norm = keys.norm(dim=-1)
        sim = (F.normalize(keys,dim=-1) * k[:,None,:]).sum(-1)
        match_score, match_at = sim.max(1)
        matched = match_score >= .98
        hit_count = norm[row,match_at]
        if frequency:
            new_key = (hit_count+1).clamp(max=48)[:,None] * k
            new_val = (values[row,match_at]*hit_count[:,None]+v)/(hit_count[:,None]+1)
        else:
            new_key, new_val = k,v
        # Miss insertion only for unused/weak slots. A confirmed key (count>1)
        # is protected from an unrelated single-exposure distractor.
        min_count, weakest = norm.min(1)
        place = torch.where(matched, match_at, weakest)
        allow_miss = (min_count<=1.01) if frequency else torch.ones_like(matched)
        place_ok = matched | allow_miss
        row_write = row[place_ok]
        col_write = place[place_ok]
        keys[row_write,col_write] = torch.where(matched[place_ok,None], new_key[place_ok], k[place_ok])
        values[row_write,col_write] = torch.where(matched[place_ok,None], new_val[place_ok], v[place_ok])
    return (keys, values)


def cache_predict(cache, query):
    keys,values=cache
    sim=(F.normalize(keys,dim=-1)*query[:,None,:]).sum(-1)
    confidence, ix = sim.max(1)
    return values[torch.arange(len(query)),ix] * (confidence>=.98)[:,None]


def errors(pred, w):
    return (pred-w['observed_query_effect']).square().mean(-1)


def baseline_eval(w):
    zero = errors(torch.zeros_like(w['observed_query_effect']),w)
    full = WriteGate(980201)(factual(w),mode='all')
    allw = errors(fast_predict(full,w['query_action']),w)
    frequency = errors(cache_predict(cache_stream(factual(w)),w['query_action']),w)
    cyclic = errors(cache_predict(cache_stream(factual(w),frequency=False),w['query_action']),w)
    return {'zero':zero,'all_write':allw,'frequency_cache':frequency,'cyclic_cache':cyclic}


def gate_eval(model,w,wrong=False):
    with torch.no_grad():
        mat=model(factual(w),wrong=w['wrong_credit_key'] if wrong else None)
        return errors(fast_predict(mat,w['query_action']),w)


def train(model, fit, val, indices, updates=FIT_UPDATES, wrong=False):
    opt=torch.optim.AdamW(model.parameters(),lr=.01)
    best, state=float('inf'),None
    checks=[]
    for i in range(updates):
        sample=subset(fit, indices[i])
        opt.zero_grad(set_to_none=True)
        mat=model(factual(sample),wrong=sample['wrong_credit_key'] if wrong else None)
        loss=errors(fast_predict(mat,sample['query_action']),sample).mean()
        if not bool(torch.isfinite(loss)):raise ValueError('P68_NONFINITE_TRAIN')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if (i+1)%CHECK==0:
            with torch.no_grad():
                value=float(gate_eval(model,val,wrong=wrong).mean())
            checks.append({'step':i+1,'mse':value})
            if value<best:
                best,state=value,copy.deepcopy(model.state_dict())
    if state is not None: model.load_state_dict(state)
    return {'best_validation_mse':best,'checkpoints':checks,'steps':updates,'params':4}


def exact_restore(model, val):
    with torch.no_grad():
        facts=factual(val)
        full=model(facts)
        half=facts['action'].shape[1]//2
        before=model(facts,hi=half)
        new=WriteGate(980201)
        new.load_state_dict(copy.deepcopy(model.state_dict()))
        restored=new(facts,carry=before,lo=half)
        return bool(torch.equal(full,restored))


def paired_lower_relative(gate, comparator, seed=980401, repeats=2000):
    rng=torch.Generator().manual_seed(seed)
    idx=torch.randint(len(gate),(repeats,len(gate)),generator=rng)
    lhs=gate[idx].mean(-1)
    rhs=comparator[idx].mean(-1).clamp_min(1e-12)
    return float(torch.quantile(1-lhs/rhs,.025))


def study():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    fit,val=make_world('fit'),make_world('val')
    audits={k:positional_audit(w['role_audit']) for k,w in [('fit',fit),('val',val)]}
    if not all(v['pass'] for v in audits.values()):
        return {'status':'DEV_FAIL_SOURCE_SHORTCUT','audits':audits}
    val_b=baseline_eval(val)
    competency=float(val_b['frequency_cache'].mean())<=.75*float(val_b['zero'].mean())
    record={'schema':'ttt-hybrid.p68-slot-memory-dev-result.v1', 'level':'AUTHORED_DEV_NOT_INDEPENDENT_SCIENCE',
        'status':'NOT_FINISHED', 'source_hashes':{'fit':source_hash(fit),'val':source_hash(val)},
        'source_audits':audits,'zero_val':float(val_b['zero'].mean()),
        'frequency_cache_val':float(val_b['frequency_cache'].mean()),
        'baseline_competence':competency,'new_source_seeds':SEEDS,
        'cache_memory_floats':2*SLOTS*D,'gate_memory_floats':D*D,
        'original_hybrid_unchanged':True,'forge':'UNCHANGED','science_pass':False,'full_cross_platform':'NOT_RUN','gpu':'NOT_RUN'}
    if not competency:
        record['status']='DEV_FAIL_BASELINE_COMPETENCE'
        record['ood_scored']=False
        return record
    rng=torch.Generator().manual_seed(980299)
    indices=torch.randint(N['fit'],(FIT_UPDATES,BATCH),generator=rng)
    record['sample_indices_sha256']=digest(indices)
    rows=[]
    for seed in MODEL_SEEDS:
        gate=WriteGate(seed)
        init_hash=digest(gate.theta)
        meta=train(gate,fit,val,indices)
        rows.append({'seed':seed,'init_sha':init_hash,'gate_validation':meta,
                     'qualified_gate':meta['best_validation_mse']<=.85*record['zero_val'],
                     'cold_restored_bitwise':exact_restore(gate,subset(val,torch.arange(8))),
                     'frozen_theta':gate.theta.detach().tolist()})
    record['trained_rows']=rows
    record['distinct_initialization_sha256']=len(set(x['init_sha'] for x in rows))==3
    if not record['distinct_initialization_sha256'] or not all(x['qualified_gate'] and x['cold_restored_bitwise'] for x in rows):
        record['status']='DEV_FAIL_GATE_VALIDATION';record['ood_scored']=False
        record['seconds']=time.perf_counter()-start
        return record
    ood=make_world('ood_dev')
    record['source_hashes']['ood_dev']=source_hash(ood)
    record['source_audits']['ood_dev']=positional_audit(ood['role_audit'])
    if not record['source_audits']['ood_dev']['pass']:
        record['status']='DEV_FAIL_OOD_SOURCE_SHORTCUT';record['ood_scored']=False;return record
    bo=baseline_eval(ood)
    report={key:float(val.mean()) for key,val in bo.items()}
    record['ood_controls']=report
    ref_name=min(bo,key=lambda n:float(bo[n].mean()))
    record['best_nonprivileged_baseline']=ref_name
    # OOD once, after freezing all validation decisions and models.
    for row in rows:
        gate=WriteGate(row['seed'])
        with torch.no_grad():gate.theta.copy_(torch.tensor(row['frozen_theta']))
        e=gate_eval(gate,ood)
        row['ood_mse']=float(e.mean())
        row['best_baseline_ood_mse']=float(bo[ref_name].mean())
        row['bootstrap_lower_relative_gain']=paired_lower_relative(e,bo[ref_name])
        row['candidate_10pct_better']=row['ood_mse']<=.9*row['best_baseline_ood_mse']
        row['passed_primary']=row['candidate_10pct_better'] and row['bootstrap_lower_relative_gain']>0
    record['ood_scored']=True
    record['status']='DEV_PASS_PRIMARY_ONLY' if all(r['passed_primary'] for r in rows) else 'DEV_FAIL_STRONG_BASELINE'
    record['secondary_wrong_credit_full_train']='NOT_RUN'
    record['whole_process_compute_matched']=False
    record['seconds']=time.perf_counter()-start
    return record


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['preflight','study'],required=True)
    ap.add_argument('--receipt',type=Path,required=True)
    args=ap.parse_args()
    if args.mode=='preflight':
        torch.set_num_threads(1)
        fit,val=make_world('fit',32),make_world('val',32)
        r={'mode':'engineering_preflight','source':source_hash(fit),
           'source_deterministic':source_hash(fit)==source_hash(make_world('fit',32)),
           'baseline_budget':2*SLOTS*D,
           'val_baselines':{k:float(v.mean()) for k,v in baseline_eval(val).items()},
           'restore':exact_restore(WriteGate(980201),val),
           'science_pass':False}
    else:r=study()
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(r,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'mode':args.mode,'status':r.get('status'),'competence':r.get('baseline_competence'),
                      'science_pass':False},sort_keys=True),flush=True)


if __name__=='__main__':main()
