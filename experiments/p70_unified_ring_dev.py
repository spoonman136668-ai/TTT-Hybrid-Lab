"""P70 pre-frozen synthetic DEV: exact PR177 ring comparator as the sole ring.

Prospective P70 protocol Git commit 1eea9f21bfbb3b29a261e7fa1ec6e5a26151a11b
precedes creation or evaluation of all P70 development data. This uses ONE
original P69 ring implementation (PR177 blob bf12bd...) and the unchanged P68
learned-gate/frequency/all-write controls, never P68's broken cyclic arm.
Do not use as independent scientific first-use or original Hybrid validation.
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

RING_SOURCE = Path(__file__).with_name('p69_ring_cache.py')

def git_blob_bytes(b):
    return hashlib.sha1(b'blob ' + str(len(b)).encode('ascii') + b'\0' + b).hexdigest()

assert git_blob_bytes(BASE.read_bytes()) == '7db52d3a150903b2feff661c9562017ee24c0dfe', 'P70_P68_SOURCE_DRIFT'
assert git_blob_bytes(RING_SOURCE.read_bytes()) == 'bf12bd3525230ecf58ecc43837f67fdf23578f35', 'P70_PR177_RING_SOURCE_DRIFT'
_ring_spec = importlib.util.spec_from_file_location('p69_ring_exact_for_p70', RING_SOURCE)
ring = importlib.util.module_from_spec(_ring_spec)
_ring_spec.loader.exec_module(ring)

D, SLOTS = 16, 8
SEEDS = {'fit': 1000111, 'val': 1000113, 'ood_dev': 1000117, 'negative': 1000119}
N = {'fit': 512, 'val': 128, 'ood_dev': 128, 'negative': 64}
MODEL_SEEDS = (1000201, 1000203, 1000207)
WRONG_SEEDS = (1001001, 1001003, 1001007)
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
    if split == 'negative':
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


def cache_stream(events, carry=None, lo=0, hi=None):
    """Adapt ranged P68 factual event views to exact PR177 ring.stream.

    No replacement logic here. The sole ring algorithm lives in p69_ring_cache.
    Carry consists only of two (batch,8,16) float tensors, validated by ring.
    """
    if set(events) != {'action','effect'}:
        raise ValueError('P70_FACTUAL_ONLY_REQUIRED')
    a, y = events['action'],events['effect']
    if a.ndim != 3 or a.shape != y.shape:
        raise ValueError('P70_BAD_FACTUAL_SHAPE')
    hi=a.shape[1] if hi is None else hi
    if not (0<=lo<=hi<=a.shape[1]):
        raise ValueError('P70_BAD_RANGE')
    return ring.stream({'action':a[:,lo:hi], 'effect':y[:,lo:hi]},carry)


def cold_restore(world):
    facts = p68.factual(world)
    mid = facts['action'].shape[1] // 2
    full = cache_stream(facts)
    part = cache_stream(facts, hi=mid)
    after = cache_stream(facts, carry=copy.deepcopy(part), lo=mid)
    query = world['query_action']
    return (all(torch.equal(x, z) for x, z in zip(full, after)) and
            torch.equal(ring.predict(full, query), ring.predict(after, query)))


def baseline_errors(world):
    facts = p68.factual(world)
    zero = p68.errors(torch.zeros_like(world['observed_query_effect']), world)
    allw = p68.WriteGate(1000201)(facts, mode='all')
    allwrite = p68.errors(p68.fast_predict(allw, world['query_action']), world)
    frequency = p68.errors(p68.cache_predict(p68.cache_stream(facts), world['query_action']), world)
    true_cyclic = p68.errors(ring.predict(cache_stream(facts), world['query_action']), world)
    return {'zero': zero, 'all_write': allwrite,
            'frequency_cache': frequency, 'pr177_ring_cache': true_cyclic}


def ortho_negative():
    w = make_world('negative')
    facts = p68.factual(w)
    full = p68.WriteGate(1000201)(facts, mode='all')
    role = w['role_audit']
    positions = (~role).long().argsort(dim=1, stable=True)
    n = int(role[0].sum())
    targets = {key: value.gather(1, positions[:, :, None].expand_as(value))[:, :n]
               for key, value in facts.items()}
    clean = p68.WriteGate(1000201)(targets, mode='all')
    return float((p68.fast_predict(full, w['query_action']) -
                  p68.fast_predict(clean, w['query_action'])).abs().max())


def strict_study():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    fit, val = make_world('fit'), make_world('val')
    audits = {name: p68.positional_audit(w['role_audit']) for name,w in [('fit',fit),('val',val)]}
    receipt = {'schema':'ttt-hybrid.p70-pr177-exact-ring-dev-result.v1',
               'level':'AUTHORED_DEV_NOT_SCIENCE','status':'NOT_EVALUATED',
               'protocol_freeze_commit':'1eea9f21bfbb3b29a261e7fa1ec6e5a26151a11b',
               'source_seeds':SEEDS,'source_hashes':{'fit':p68.source_hash(fit),'val':p68.source_hash(val)},
               'audits':audits,'memory_floats':2*SLOTS*D,'ood_scored':False,
               'scientific_pass':False,'full_original_regression':'NOT_RUN','native_gpu':'NOT_RUN',
               'original_hybrid':'UNCHANGED','FORGE':'UNCHANGED','whole_process_compute_matched':False, 'ring_pr':177,'ring_blob':'bf12bd3525230ecf58ecc43837f67fdf23578f35', 'p68_blob':'7db52d3a150903b2feff661c9562017ee24c0dfe'}
    if not all(x['pass'] for x in audits.values()) or not cold_restore(p68.subset(val,torch.arange(8))):
        receipt['status']='DEV_FAIL_SOURCE_OR_COLD_STATE'
        return receipt
    val_err = baseline_errors(val)
    val_means = {k:float(z.mean()) for k,z in val_err.items()}
    receipt['val_baselines'] = val_means
    competent = {k: val_means[k] <= .75*val_means['zero'] for k in
                 ('frequency_cache','pr177_ring_cache','all_write')}
    receipt['baseline_competence'] = competent
    if not all(competent.values()):
        receipt['status']='DEV_FAIL_CONVENTIONAL_CONTROL_COMPETENCE'
        receipt['elapsed_seconds']=time.perf_counter()-start
        return receipt
    samples = torch.randint(N['fit'],(FIT_UPDATES,BATCH),generator=torch.Generator().manual_seed(1000299))
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
        row['bootstrap_lower_95_relative']=p68.paired_lower_relative(err,controls[strongest],seed=1000401)
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
