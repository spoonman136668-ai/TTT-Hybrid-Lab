"""P69 preregistered corrected-cycle 256-float control. Authored synthetic DEV only.

Frozen source+gates: experiments/p69_genuine_cyclic_freeze.json, Git commit
ff9b8ce153db330339af0cf984ad332424595847 (before P69 data birth).
Original P68 experiment intentionally unchanged; only selected reusable methods
are imported, NEVER its incorrectly labeled cyclic baseline.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

BASE = Path(__file__).with_name('p68_slot_memory_dev.py')
_spec = importlib.util.spec_from_file_location('p68_reference_p69', BASE)
p68 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p68)

D, SLOTS = 16, 8
SEEDS = {'fit': 990111, 'val': 990113, 'ood_dev': 990117, 'orthogonal': 990119}
N = {'fit': 512, 'val': 128, 'ood_dev': 128, 'orthogonal': 64}
MODEL_SEEDS = (990201, 990203, 990207)
WRONG_SEEDS = (991001, 991003, 991007)
FIT_UPDATES, BATCH = 180, 32
MAX_SECONDS = 480


def make_world(split, count=None):
    if split not in SEEDS:
        raise ValueError('P69_INVALID_SPLIT')
    count = N[split] if count is None else count
    if count <= 0 or count > N[split]:
        raise ValueError('P69_INVALID_COUNT')
    gen = torch.Generator().manual_seed(SEEDS[split])
    ood = split == 'ood_dev'
    repetitions, noise = (4, 48) if ood else (6, 16)
    basis = torch.linalg.qr(torch.randn(count, D, D, generator=gen)).Q
    targets = basis[:, :, :SLOTS].transpose(1, 2).contiguous()
    effects = torch.tanh(torch.randn(count, SLOTS, D, generator=gen) * (1.2 if ood else 1.0))
    key = targets.repeat(1, repetitions, 1)
    consequence = effects.repeat(1, repetitions, 1)
    nuisance = F.normalize(torch.randn(count, noise, D, generator=gen), dim=-1)
    if split == 'orthogonal':
        nuisance = F.normalize(torch.bmm(torch.randn(count, noise, 8, generator=gen),
                                       basis[:, :, 8:].transpose(1, 2)), dim=-1)
    nuisance_effect = torch.tanh(torch.randn(count, noise, D, generator=gen))
    key = torch.cat((key, nuisance), dim=1)
    consequence = torch.cat((consequence, nuisance_effect), dim=1)
    role = torch.cat((torch.ones(count, SLOTS * repetitions, dtype=torch.bool),
                      torch.zeros(count, noise, dtype=torch.bool)), dim=1)
    order = torch.rand(count, key.shape[1], generator=gen).argsort(dim=1)
    key = key.gather(1, order[:, :, None].expand_as(key))
    consequence = consequence.gather(1, order[:, :, None].expand_as(consequence))
    role = role.gather(1, order)
    chosen = torch.randint(SLOTS, (count,), generator=gen)
    observed_query = effects[torch.arange(count), chosen]
    query_key = targets[torch.arange(count), chosen]
    wrong_keys = F.normalize(torch.randn(count, key.shape[1], D, generator=gen), dim=-1)
    return {'action': key, 'effect': consequence,
            'query_action': query_key, 'observed_query_effect': observed_query,
            'role_audit': role, 'wrong_credit_key': wrong_keys}


def validate_facts(events):
    if set(events) != {'action', 'effect'}:
        raise ValueError('P69_NON_FACTUAL_FIELDS')
    a, y = events['action'], events['effect']
    if a.shape != y.shape or a.ndim != 3 or a.shape[-1] != D or a.dtype != torch.float32:
        raise ValueError('P69_INVALID_FACT_SHAPE')
    if not bool(torch.isfinite(a).all() and torch.isfinite(y).all()):
        raise ValueError('P69_NONFINITE_FACT')
    if not bool((a.norm(dim=-1) > 1e-7).all()):
        raise ValueError('P69_ZERO_ACTION_KEY')
    return a, y


def cache_stream(events, carry=None, lo=0, hi=None):
    """Persist 8*(16-key+16-value)=256 floats; pointer encoded in key norm 2.

    Exactly one stored key has norm 2 (the most recently INSERTED key),
    the other occupied keys have norm 1; zeros indicate vacant slots.
    The next miss replaces (last_inserted+1)%8. A hit changes only value.
    No extra pointer, clock, counter or step number is retained between calls.
    """
    a, y = validate_facts(events)
    batch, length, _ = a.shape
    hi = length if hi is None else hi
    if not (0 <= lo <= hi <= length):
        raise ValueError('P69_INVALID_RANGE')
    if carry is None:
        keys = a.new_zeros(batch, SLOTS, D)
        values = a.new_zeros(batch, SLOTS, D)
    else:
        if not isinstance(carry, tuple) or len(carry) != 2:
            raise ValueError('P69_INVALID_STATE')
        keys, values = (x.clone() for x in carry)
        if keys.shape != (batch, SLOTS, D) or values.shape != keys.shape:
            raise ValueError('P69_INVALID_STATE')
        norms = keys.norm(dim=-1)
        marker = norms > 1.5
        if not bool((marker.sum(-1) <= 1).all()):
            raise ValueError('P69_MULTIPLE_POINTERS')
        if not bool(((norms < 1e-6) | ((norms - 1).abs() < 1e-5) |
                     ((norms - 2).abs() < 1e-5)).all()):
            raise ValueError('P69_CORRUPTED_STATE')
        if not bool(torch.isfinite(values).all()):
            raise ValueError('P69_NONFINITE_STATE')
    row = torch.arange(batch)
    for t in range(lo, hi):
        normalized_action = F.normalize(a[:, t], dim=-1)
        old_norm = keys.norm(dim=-1)
        marker = old_norm > 1.5
        last = torch.where(marker.any(dim=1), marker.float().argmax(dim=1),
                           torch.full((batch,), -1, dtype=torch.long))
        similarity = (F.normalize(keys, dim=-1) * normalized_action[:, None, :]).sum(-1)
        nearest_score, nearest = similarity.max(dim=1)
        is_hit = nearest_score >= .98
        if bool(is_hit.any()):
            values[row[is_hit], nearest[is_hit]] = y[is_hit, t]
        miss = ~is_hit
        if bool(miss.any()):
            mr = row[miss]
            prior = last[miss]
            prior_exists = prior >= 0
            if bool(prior_exists.any()):
                rr = mr[prior_exists]
                pp = prior[prior_exists]
                keys[rr, pp] = F.normalize(keys[rr, pp], dim=-1)
            position = (prior + 1) % SLOTS
            keys[mr, position] = normalized_action[miss] * 2
            values[mr, position] = y[miss, t]
    return keys, values


def cold_restore(world):
    facts = p68.factual(world)
    mid = facts['action'].shape[1] // 2
    full = cache_stream(facts)
    part = cache_stream(facts, hi=mid)
    after = cache_stream(facts, carry=copy.deepcopy(part), lo=mid)
    query = world['query_action']
    return (all(torch.equal(x, z) for x, z in zip(full, after)) and
            torch.equal(p68.cache_predict(full, query), p68.cache_predict(after, query)))


def baseline_errors(world):
    facts = p68.factual(world)
    zero = p68.errors(torch.zeros_like(world['observed_query_effect']), world)
    allw = p68.WriteGate(990201)(facts, mode='all')
    allwrite = p68.errors(p68.fast_predict(allw, world['query_action']), world)
    frequency = p68.errors(p68.cache_predict(p68.cache_stream(facts), world['query_action']), world)
    true_cyclic = p68.errors(p68.cache_predict(cache_stream(facts), world['query_action']), world)
    return {'zero': zero, 'all_write': allwrite,
            'frequency_cache': frequency, 'true_cyclic_cache': true_cyclic}


def ortho_negative():
    w = make_world('orthogonal')
    facts = p68.factual(w)
    full = p68.WriteGate(990201)(facts, mode='all')
    role = w['role_audit']
    positions = (~role).long().argsort(dim=1, stable=True)
    n = int(role[0].sum())
    targets = {key: value.gather(1, positions[:, :, None].expand_as(value))[:, :n]
               for key, value in facts.items()}
    clean = p68.WriteGate(990201)(targets, mode='all')
    return float((p68.fast_predict(full, w['query_action']) -
                  p68.fast_predict(clean, w['query_action'])).abs().max())


def strict_study():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    fit, val = make_world('fit'), make_world('val')
    audits = {name: p68.positional_audit(w['role_audit']) for name,w in [('fit',fit),('val',val)]}
    receipt = {'schema':'ttt-hybrid.p69-strong-baselines-dev-result.v1',
               'level':'AUTHORED_DEV_NOT_SCIENCE','status':'NOT_EVALUATED',
               'protocol_freeze_commit':'ff9b8ce153db330339af0cf984ad332424595847',
               'source_seeds':SEEDS,'source_hashes':{'fit':p68.source_hash(fit),'val':p68.source_hash(val)},
               'audits':audits,'memory_floats':2*SLOTS*D,'ood_scored':False,
               'scientific_pass':False,'full_original_regression':'NOT_RUN','native_gpu':'NOT_RUN',
               'original_hybrid':'UNCHANGED','FORGE':'UNCHANGED','whole_process_compute_matched':False}
    if not all(x['pass'] for x in audits.values()) or not cold_restore(p68.subset(val,torch.arange(8))):
        receipt['status']='DEV_FAIL_SOURCE_OR_COLD_STATE'
        return receipt
    val_err = baseline_errors(val)
    val_means = {k:float(z.mean()) for k,z in val_err.items()}
    receipt['val_baselines'] = val_means
    competent = {k: val_means[k] <= .75*val_means['zero'] for k in
                 ('frequency_cache','true_cyclic_cache','all_write')}
    receipt['baseline_competence'] = competent
    if not all(competent.values()):
        receipt['status']='DEV_FAIL_CONVENTIONAL_CONTROL_COMPETENCE'
        receipt['elapsed_seconds']=time.perf_counter()-start
        return receipt
    samples = torch.randint(N['fit'],(FIT_UPDATES,BATCH),generator=torch.Generator().manual_seed(990299))
    receipt['sampler_sha256'] = p68.digest(samples)
    trained=[]
    for seed,wrong_seed in zip(MODEL_SEEDS,WRONG_SEEDS):
        model=p68.WriteGate(seed)
        init=p68.digest(model.theta)
        mark=time.perf_counter()
        meta=p68.train(model,fit,val,samples)
        msec=time.perf_counter()-mark
        trained.append((model,{'seed':seed,'init_sha256':init,
                     'best_gate_validation_mse':meta['best_validation_mse'],
                     'gate_train_seconds':msec,'train_steps':meta['steps'],
                     'initialization_parameters':4,'cache_floats':256,
                     'cold_restore':p68.exact_restore(model,p68.subset(val,torch.arange(8)))}))
    receipt['model_rows']=[r for _,r in trained]
    receipt['independent_model_init']=len({r['init_sha256'] for _,r in trained})==3
    if (not receipt['independent_model_init'] or
        not all(r['best_gate_validation_mse']<=.85*val_means['zero'] and r['cold_restore']
                for _,r in trained)):
        receipt['status']='DEV_FAIL_CANDIDATE_VALIDATION';receipt['elapsed_seconds']=time.perf_counter()-start
        return receipt
    wrong=[]
    for wrong_seed in WRONG_SEEDS:
        model=p68.WriteGate(wrong_seed)
        mark=time.perf_counter()
        meta=p68.train(model,fit,val,samples,wrong=True)
        wrong.append((model,{'seed':wrong_seed,'validation_mse':meta['best_validation_mse'],
                             'train_seconds':time.perf_counter()-mark,'steps':meta['steps']}))
    receipt['wrong_credit_rows']=[r for _,r in wrong]
    if time.perf_counter()-start > MAX_SECONDS:
        receipt['status']='DEV_FAIL_CPU_BUDGET';return receipt
    # Single evaluation after all arms/validation decisions are fixed.
    ood=make_world('ood_dev')
    receipt['source_hashes']['ood_dev']=p68.source_hash(ood)
    receipt['audits']['ood_dev']=p68.positional_audit(ood['role_audit'])
    if not receipt['audits']['ood_dev']['pass']:
        receipt['status']='DEV_FAIL_OOD_SHORTCUT';return receipt
    controls=baseline_errors(ood)
    receipt['ood_baselines']={k:float(z.mean()) for k,z in controls.items()}
    strongest=min(controls, key=lambda k:float(controls[k].mean()))
    receipt['strongest_baseline']=strongest
    negative=ortho_negative()
    receipt['orthogonal_negative_max_abs_diff']=negative
    receipt['orthogonal_control_pass']=negative<=1e-5
    for (m,row),(wrong_m,wrong_row) in zip(trained,wrong):
        err=p68.gate_eval(m,ood)
        bad=p68.gate_eval(wrong_m,ood,wrong=True)
        row['ood_mse']=float(err.mean())
        row['wrong_credit_ood_mse']=float(bad.mean())
        row['bootstrap_lower_95_relative']=p68.paired_lower_relative(err,controls[strongest],seed=990401)
        row['relative_reduction']=1-row['ood_mse']/receipt['ood_baselines'][strongest]
        row['primary_pass']=(row['relative_reduction']>=.10 and row['bootstrap_lower_95_relative']>0)
        row['wrong_credit_negative_pass']=row['wrong_credit_ood_mse']>=1.2*row['ood_mse']
        wrong_row['ood_mse']=float(bad.mean())
    receipt['ood_scored']=True
    receipt['status']='DEV_PASS_ALL_PREDECLARED_CONTROLS' if (receipt['orthogonal_control_pass']
       and all(r['primary_pass'] and r['wrong_credit_negative_pass'] for _,r in trained)) else 'DEV_FAIL_STRICT_CONTROLS'
    receipt['elapsed_seconds']=time.perf_counter()-start
    receipt['scientific_pass']=False
    return receipt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    result=strict_study()
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'val_baselines':result.get('val_baselines'),
                      'ood_scored':result['ood_scored'],'scientific_pass':False}),flush=True)

if __name__=='__main__':
    main()
