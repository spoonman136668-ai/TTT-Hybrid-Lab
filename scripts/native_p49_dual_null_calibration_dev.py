"""P49 authored DEV: dual stable-null calibration of causal and nuisance changes.

Uses exact original P47 factual design/BIC, no new learned architecture, no
external scientific first use, no hidden world labels or unexecuted outcomes.
The new source seeds and acceptance metrics are precommitted separately.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import time
from pathlib import Path
import numpy as np
from scripts import native_p47_change_vs_nuisance_dev as p47

COHORTS = {
    780001: (4096, (.35, 2.0), 'calibration'),
    780101: (1536, (.35, .90), 'evaluation'),
    780103: (1536, (.95, 1.35), 'evaluation'),
    780107: (1536, (1.40, 2.0), 'evaluation'),
}
QUANTILE = .99


def fresh_source(seed):
    if type(seed) is not int or seed not in COHORTS:
        raise ValueError('P49_UNREGISTERED_SOURCE')
    n, band, role = COHORTS[seed]
    rng = np.random.default_rng(seed)
    if role == 'calibration':
        kind = np.full(n, 2, dtype=np.int64)
    else:
        kind = np.repeat(np.array([0, 1, 2], dtype=np.int64), [768, 384, 384])
        rng.shuffle(kind)
    k = rng.choice(np.array(p47.SPLITS, dtype=np.int64), n)
    initial = rng.choice(np.array([-1., 1.]), n)
    a = initial[:, None] * np.power(-1., np.arange(p47.N))[None, :]
    u = lambda low, high: rng.uniform(low, high, (n, 1))
    b0, b1, b2 = u(-.2, .2), u(-.4, .4), u(-.1, .1)
    amp = u(0, .08)
    phase = u(-math.pi, math.pi)
    freq = u(*band)
    beta = rng.choice(np.array([-1., 1.]), (n, 1)) * u(.25, .45)
    changed = (kind[:, None] == 0) & (np.arange(p47.N)[None, :] >= k[:, None])
    beta_seq = beta * np.where(changed, -1., 1.)
    jump = rng.choice(np.array([-1., 1.]), (n, 1)) * u(.40, .70)
    shock = (kind[:, None] == 1) & (np.arange(p47.N)[None, :] >= k[:, None])
    nuisance = (b0 + b1 * p47.T + b2 * p47.T**2
                + amp * np.sin(2*math.pi*freq*p47.T + phase) + jump*shock)
    y = nuisance + beta_seq*a + rng.normal(0, .02, (n, p47.N))
    truth = beta_seq[:, -1].copy()
    source_sha256 = hashlib.sha256(b''.join(arr.tobytes() for arr in
        (a, y, truth, kind, k))).hexdigest()
    return dict(a=a, y=y, beta=truth, kind=kind, k=k, seed=seed,
                role=role, sha=source_sha256)


def evidence(a, y):
    # This validation and all unchanged benchmark arms are the exact P47 code.
    controls = p47.estimate(a, y)
    n = a.shape[0]
    full_bic = np.full(n, np.inf)
    full_effect = np.zeros(n)
    nuisance_bic = np.full(n, np.inf)
    nuisance_effect = np.zeros(n)
    nuisance_k = np.full(n, -1, dtype=np.int64)
    causal_bic = np.full(n, np.inf)
    causal_effect = np.zeros(n)
    causal_k = np.full(n, -1, dtype=np.int64)
    for sign in (-1., 1.):
        ix = np.flatnonzero(a[:, 0] == sign)
        if not len(ix):
            continue
        obs = y[ix]
        f_eff, f_bic = p47.fit(obs, sign)
        full_bic[ix], full_effect[ix] = f_bic, f_eff
        for k in p47.SPLITS:
            n_eff, n_bic = p47.fit(obs, sign, 'nuisance', k)
            selected = n_bic < nuisance_bic[ix]
            ids = ix[selected]
            nuisance_bic[ids], nuisance_effect[ids], nuisance_k[ids] = (n_bic[selected], n_eff[selected], k)
            c_eff, c_bic = p47.fit(obs, sign, 'causal', k)
            selected = c_bic < causal_bic[ix]
            ids = ix[selected]
            causal_bic[ids], causal_effect[ids], causal_k[ids] = (c_bic[selected], c_eff[selected], k)
    null_bic = np.minimum(full_bic, nuisance_bic)
    return dict(causal_margin=null_bic-causal_bic,
                nuisance_margin=full_bic-nuisance_bic,
                full_effect=full_effect, nuisance_effect=nuisance_effect,
                causal_effect=causal_effect, nuisance_k=nuisance_k, causal_k=causal_k,
                nuisance_best=nuisance_bic<full_bic, controls=controls)


def thresholds(calibration):
    if calibration['seed'] != 780001 or calibration['role'] != 'calibration':
        raise ValueError('P49_INVALID_CALIBRATION_CUSTODY')
    if not bool(np.all(calibration['kind']==2)):
        raise ValueError('P49_CALIBRATION_MIXED_SOURCE_REFUSED')
    fit = evidence(calibration['a'], calibration['y'])
    ac = fit['causal_margin']
    an = fit['nuisance_margin']
    if not np.isfinite(ac).all() or not np.isfinite(an).all():
        raise ValueError('P49_NONFINITE_CALIBRATION')
    return {'causal':float(np.quantile(ac, QUANTILE, method='higher')),
            'nuisance':float(np.quantile(an, QUANTILE, method='higher')),
            'causal_margin_sha256':hashlib.sha256(ac.tobytes()).hexdigest(),
            'nuisance_margin_sha256':hashlib.sha256(an.tobytes()).hexdigest()}


def predict(a, y, calibration, dual=True):
    if any(not np.isfinite(calibration[k]) for k in ('causal','nuisance')):
        raise ValueError('P49_INVALID_THRESHOLDS')
    fit = evidence(a,y)
    causal = fit['causal_margin'] > max(0.,calibration['causal'])
    # P48 baseline selects nuisance whenever nuisance BIC is below full BIC;
    # P49 requires independent stable-null evidence for this second decision.
    nuisance = ((fit['nuisance_margin']>max(0.,calibration['nuisance'])) if dual
                else fit['nuisance_best']) & ~causal
    pred = np.where(causal,fit['causal_effect'],
                    np.where(nuisance,fit['nuisance_effect'],fit['full_effect']))
    out = {k:v.copy() for k,v in fit['controls'].items()}
    out['candidate'] = pred
    out['model_kind'] = np.where(causal, 0, np.where(nuisance, 1, -1))
    out['chosen_k'] = np.where(causal, fit['causal_k'],np.where(nuisance,fit['nuisance_k'],-1))
    return out


def study():
    started=time.perf_counter()
    source_cal=fresh_source(780001)
    cutoff=thresholds(source_cal)
    reports=[]
    for seed in (780101, 780103, 780107):
        data=fresh_source(seed)
        original=p47.metrics(data,p47.estimate(data['a'],data['y']))
        single=p47.metrics(data,predict(data['a'],data['y'],cutoff,dual=False))
        output=predict(data['a'],data['y'],cutoff,dual=True)
        dual=p47.metrics(data,output)
        nmask=data['kind']==1
        change_limit=dual['results']['candidate']['exogenous']['mae'] <= 1.10*single['results']['candidate']['exogenous']['mae']
        # Source persistence checks actual facts only, no labels in recovery.
        buf=io.BytesIO();np.savez_compressed(buf,a=data['a'],y=data['y']);buf.seek(0)
        with np.load(buf,allow_pickle=False) as obj:
            replay=predict(obj['a'],obj['y'],cutoff,dual=True)
        exact=all(np.array_equal(output[k],replay[k]) for k in output)
        reports.append({'seed':seed,'sha256':data['sha'], 'original':original,
                        'p48_control':single, 'p49_dual':dual,
                        'exogenous_effect_noninferiority':bool(change_limit),
                        'cold_exact':bool(exact),
                        'nuisance_model_selected':int((output['model_kind']==1).sum()),
                        'nuisance_selected_stable':int(((output['model_kind']==1)&(data['kind']==2)).sum()),
                        'nuisance_selected_exogenous':int(((output['model_kind']==1)&nmask).sum())})
    return {'schema':'ttt-hybrid.p49.dual-stable-null-calibration-dev-receipt.v1',
            'source_type':'AUTHORED_DEV_NO_EXTERNAL_SCIENCE_FIRST_USE',
            'calibration_sha256':source_cal['sha'],'thresholds':cutoff,
            'reports':reports,
            'frozen_all_cohort_positive':all(r['p49_dual']['frozen_dev_pass'] and r['exogenous_effect_noninferiority'] and r['cold_exact'] for r in reports),
            'wall_seconds':time.perf_counter()-started,
            'original_windows_ubuntu_full_regression':'NOT_RUN','FORGE':'UNTOUCHED'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    result=study()
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'calibration':result['thresholds'],'outcomes':[
        {'seed':r['seed'],'source_sha256':r['sha256'],
         'P48_false_stable':r['p48_control']['detection']['stable_false_alarm'],
         'P49_false_stable':r['p49_dual']['detection']['stable_false_alarm'],
         'P48_stable_MAE':r['p48_control']['results']['candidate']['stable']['mae'],
         'P49_stable_MAE':r['p49_dual']['results']['candidate']['stable']['mae'],
         'full_stable_MAE':r['p49_dual']['results']['full']['stable']['mae'],
         'causal_recall':r['p49_dual']['detection']['causal_recall'],
         'gates':r['p49_dual']['gates'],'nuisance_noninferiority':r['exogenous_effect_noninferiority'],
         'cold':r['cold_exact']}
        for r in result['reports']], 'frozen_all_cohort_positive':result['frozen_all_cohort_positive'],
        'wall_seconds':result['wall_seconds']},indent=2))


if __name__ == '__main__':
    main()
