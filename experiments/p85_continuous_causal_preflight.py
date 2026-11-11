"""Frozen P85 DEV: continuous action-consequence and within-world causal shift.

Freeze commit df086a69514576f2afbe61e2ad79656810cd25ec.
Only source FIT/CAL is admitted. No external/witnessed scientific first-use.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from native_p78_bounded_raw_ttt import BoundedRawTTT

DIM = 16
FIT_CAL_SEEDS = {'fit': 850111, 'calibration': 850113}
FIT_CAL_N = {'fit': 256, 'calibration': 128}
RESERVED_SEEDS = {'heldout': 850117, 'negative': 850119}
RATES = (0.25, 0.5, 1., 1.5, 2.)
DECAYS = (1., .99, .97, .95, .90)
FREEZE_COMMIT = 'df086a69514576f2afbe61e2ad79656810cd25ec'
RAW_BLOB = 'db948c53bb1a28af5fe0e6a62ef71463db6d95ba'
MODEL_BLOB = '5d9ea57b827aa1796548447868fa5ee14c251b49'


def git_blob(path: Path) -> str:
    b = path.read_bytes()
    return hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\x00' + b).hexdigest()


def _orthonormal(batch: int, generator: torch.Generator) -> torch.Tensor:
    return torch.linalg.qr(torch.randn(batch, DIM, 4, generator=generator)).Q


def make_worlds(split: str, n: int | None = None) -> dict:
    """Never materializes heldout or negative data at this preflight gate."""
    if split not in FIT_CAL_SEEDS:
        raise ValueError('P85_PROTECTED_OR_UNKNOWN_SOURCE')
    n = FIT_CAL_N[split] if n is None else n
    if type(n) is not int or not 8 <= n <= FIT_CAL_N[split] or n % 2:
        raise ValueError('P85_INVALID_COUNT')
    g = torch.Generator().manual_seed(FIT_CAL_SEEDS[split])
    u0 = _orthonormal(n, g)
    v0 = _orthonormal(n, g)
    u1 = _orthonormal(n, g)
    v1 = _orthonormal(n, g)
    regimes = torch.zeros(n, dtype=torch.bool)
    regimes[torch.randperm(n, generator=g)[:n // 2]] = True
    act = F.normalize(torch.randn(n, 49, DIM, generator=g), dim=-1)
    past = act[:, :48]
    query = act[:, 48]
    pre = torch.einsum('btd,bdk->btk', past, u0)
    effect0 = torch.tanh(3.0 * torch.einsum('btk,bdk->btd', pre, v0))
    late_basis = torch.where(regimes[:, None, None], u1, u0)
    late_impact = torch.where(regimes[:, None, None], v1, v0)
    late = torch.einsum('btd,bdk->btk', past[:, 24:], late_basis)
    effect1 = torch.tanh(3.0 * torch.einsum('btk,bdk->btd', late, late_impact))
    effects = torch.cat((effect0[:, :24], effect1), dim=1)
    qcomponents = torch.einsum('bd,bdk->bk', query, late_basis)
    target = torch.tanh(3.0 * torch.einsum('bk,bdk->bd', qcomponents, late_impact))
    return {'history': torch.cat((past, effects), dim=-1).contiguous(),
            'query': torch.cat((query, torch.zeros_like(query)), dim=-1).contiguous(),
            'effect': target.contiguous(), 'changed_world': regimes.contiguous()}


def world_sha(w: dict) -> str:
    return hashlib.sha256(b''.join(w[k].contiguous().numpy().tobytes()
        for k in ('history', 'query', 'effect', 'changed_world'))).hexdigest()


def audit(w: dict) -> dict:
    count = len(w['history'])
    max_similarity = float((F.normalize(w['history'][:, :, :DIM], dim=-1) *
                            F.normalize(w['query'][:, None, :DIM], dim=-1)).sum(dim=-1).max())
    err = float(w['query'][:, DIM:].abs().max())
    all_finite = all(bool(torch.isfinite(w[key]).all()) for key in ('history','query','effect'))
    equal_halves = int(w['changed_world'].sum()) == count // 2
    shape = w['history'].shape == (count,48,32) and w['query'].shape == (count,32)
    actual_all_unique = bool((w['history'][:, :, :DIM].norm(dim=-1) > .9999).all())
    qualifies = bool(equal_halves and err == 0 and max_similarity < .999 and all_finite
                     and shape and actual_all_unique)
    return {'n':count, 'changed_count':int(w['changed_world'].sum()),
            'max_query_history_cosine':max_similarity, 'future_effect_abs_max':err,
            'finite':all_finite, 'shape_correct':shape, 'pass':qualifies}


def sgd_256(w: dict, eta: float, decay: float) -> tuple[torch.Tensor, torch.Tensor]:
    facts = w['history']; q = w['query'][:, :DIM]
    b = len(facts)
    state = torch.zeros(b,DIM,DIM,dtype=facts.dtype)
    keys = F.normalize(facts[:, :, :DIM],dim=-1)
    values = facts[:, :, DIM:]
    for i in range(48):
        k = keys[:,i]
        delta = values[:,i] - torch.bmm(state, k.unsqueeze(-1)).squeeze(-1)
        state = decay*state + eta * delta.unsqueeze(-1)*k.unsqueeze(1)
    prediction = torch.bmm(state,F.normalize(q,dim=-1).unsqueeze(-1)).squeeze(-1)
    return prediction,state


def original_p78(w: dict) -> tuple[torch.Tensor, torch.Tensor]:
    model = BoundedRawTTT(max_events=48)
    prediction,state=model.read_from_episode(w['history'],w['query'][:,:DIM])
    return prediction,state.fast_weights


def last24_ols(w: dict) -> torch.Tensor:
    # Diagnostic only; retains 24x32 observations and temporary matrix workspace.
    x = w['history'][:, -24:, :DIM].double()
    y = w['history'][:, -24:, DIM:].double()
    reg = torch.eye(DIM,dtype=torch.float64)[None] * 1e-4
    beta = torch.linalg.solve(x.transpose(1,2)@x+reg, x.transpose(1,2)@y)
    return torch.bmm(w['query'][:,None,:DIM].double(),beta).squeeze(1).float()


def last_neighbor(w: dict) -> torch.Tensor:
    keys = w['history'][:, :, :DIM]
    q = w['query'][:, :DIM]
    index = torch.bmm(keys,q[:,:,None]).squeeze(-1).argmax(-1)
    return w['history'][torch.arange(len(index)),index,DIM:]


def mse_by_regime(pred: torch.Tensor,w: dict) -> dict:
    each = (pred-w['effect']).square().mean(dim=-1)
    return {'combined':float(each.mean()), 'stable':float(each[~w['changed_world']].mean()),
            'changed':float(each[w['changed_world']].mean())}


def run() -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    t0=time.perf_counter()
    out={'schema':'ttt-hybrid.p85-continuous-causal-admission-preflight-result.v1',
         'freeze_commit':FREEZE_COMMIT, 'status':'NOT_RUN', 'scientific_pass':False,
         'protected_heldout_generated':False,'negative_generated':False,
         'full_hybrid_trained':False,'whole_process_cost_matched':False}
    root=Path(__file__).resolve().parents[1]
    out['original_blob_custody']={
        name:{'expected':blob,'actual':git_blob(root/name)}
        for name,blob in [('model.py',MODEL_BLOB),('native_p78_bounded_raw_ttt.py',RAW_BLOB)]}
    if not all(x['expected']==x['actual'] for x in out['original_blob_custody'].values()):
        out['status']='PRE_FLIGHT_FAIL_CUSTODY';return out
    fit,cal=make_worlds('fit'),make_worlds('calibration')
    out['source']={'seeds':FIT_CAL_SEEDS,'sizes':FIT_CAL_N,'hashes':{'fit':world_sha(fit),'calibration':world_sha(cal)},
                   'audits':{'fit':audit(fit),'calibration':audit(cal)}}
    if not all(z['pass'] for z in out['source']['audits'].values()):
        out['status']='PRE_FLIGHT_FAIL_SOURCE';return out
    zero=mse_by_regime(torch.zeros_like(cal['effect']),cal)
    original, original_state=original_p78(cal)
    direct=mse_by_regime(original,cal)
    grid=[];selected=None;best=float('inf')
    for eta in RATES:
        for decay in DECAYS:
            pred,state=sgd_256(cal,eta,decay)
            assert state.numel()==128*256
            metrics=mse_by_regime(pred,cal)
            row={'eta':eta,'decay':decay,'mse':metrics,'state_floats_per_world':256}
            grid.append(row)
            if metrics['combined']<best:
                selected=row;best=metrics['combined']
    matched,_=sgd_256(cal,.5,1.)
    original_equal=bool(torch.allclose(original,matched,atol=1e-7,rtol=1e-7)
                        and torch.allclose(original_state,sgd_256(cal,.5,1.)[1],atol=1e-7,rtol=1e-7))
    reference=mse_by_regime(last24_ols(cal),cal)
    neighbor=mse_by_regime(last_neighbor(cal),cal)
    out['calibration']={'zero':zero,'original_p78':direct,'grid':grid,'selected':selected,
                        'p78_exact_grid_equivalence':original_equal,
                        'ols_last24_unmatched_768_float_history':reference,
                        'nearest_neighbor_unmatched_1536_float_history':neighbor,
                        'selected_frozen_before_any_heldout':True}
    qualifies=bool(original_equal and len(grid)==25 and
        selected['mse']['stable']<=.75*zero['stable'] and
        selected['mse']['changed']<=.80*zero['changed'] and
        selected['mse']['combined']<=.80*zero['combined'] and
        reference['stable']<=.75*zero['stable'] and reference['changed']<=.75*zero['changed'])
    out['admission_controls']={'pass':qualifies,'selected_beat_stable_zero_25pct':selected['mse']['stable']<=.75*zero['stable'],
        'selected_beat_changed_zero_20pct':selected['mse']['changed']<=.80*zero['changed'],
        'selected_beat_combined_zero_20pct':selected['mse']['combined']<=.80*zero['combined'],
        'OLS_late24_identifiable_both':reference['stable']<=.75*zero['stable'] and reference['changed']<=.75*zero['changed'],
        'same_P78_original_256_state':original_equal}
    out['status']='PREFLIGHT_PASS_BASELINES_ONLY' if qualifies else 'PREFLIGHT_FAIL_CONVENTIONAL_OR_IDENTIFIABILITY'
    out['elapsed_seconds']=time.perf_counter()-t0
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args();o=run();args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(o,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'status':o['status'],'seconds':o.get('elapsed_seconds'),
          'selected':o.get('calibration',{}).get('selected'),
          'heldout_generated':o['protected_heldout_generated'],'scientific_pass':False}))

if __name__=='__main__':main()
