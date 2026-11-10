"""P55 DEV: non-Gaussian factual intervention value; same real logged probes.

The Laplace decision policy assumes a Laplace noise family but estimates prior,
cue/probe scales and variances from actually executed probes and noisy rewards.
No hidden theta enters the fitted interface; no unchosen outcome is observed.
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
from scipy.special import expit
from scripts import native_p54_noisy_factual_reward_dev as p54

COHORTS = {
    850001: (49152, 'fit'),
    850101: (2048, 'validation'),
    850201: (8192, 'ood_a'),
    850301: (8192, 'ood_b'),
    850401: (4096, 'sham'),
}
LAPLACE_SCALE = 1 / math.sqrt(2)
BOOTSTRAP_SEED = 850901


def source(seed):
    if type(seed) is not int or seed not in COHORTS:
        raise ValueError('P55_SOURCE_NOT_FROZEN')
    n, cohort = COHORTS[seed]
    rng = np.random.default_rng(seed)
    if cohort in ('fit', 'validation'):
        ranges = ((.1, 1.2), (.1, 1.2), (.55, .95), (.55, .95), (.05, .4), (.5, 1.1))
    elif cohort in ('ood_a', 'sham'):
        ranges = ((0, 1.5), (0, 1.6), (.95, 1.25), (.95, 1.25), (.05, .65), (1.1, 1.6))
    else:
        ranges = ((0, 2), (0, 2), (1.1, 1.4), (1.1, 1.4), (.01, .75), (1.4, 2.0))
    cg, pg, cs, ps, price, rs = [rng.uniform(lo, hi, n).astype(np.float32) for lo, hi in ranges]
    if cohort == 'sham':
        pg.fill(0)
    theta = rng.choice(np.array([-1., 1.], dtype=np.float32), n)
    ec = rng.laplace(scale=LAPLACE_SCALE, size=n).astype(np.float32)
    ep = rng.laplace(scale=LAPLACE_SCALE, size=n).astype(np.float32)
    er = rng.laplace(scale=LAPLACE_SCALE, size=n).astype(np.float32)
    prior = np.stack((theta*cg+cs*ec, cg, cs, pg, ps, price), -1)
    probe = (rng.random(n) < .5).astype(np.float32)
    action = rng.choice(np.array([-1., 1.], dtype=np.float32), n)
    measured = np.where(probe > 0, theta*pg+ps*ep, 0).astype(np.float32)
    reward_noise = rs*er
    realized_reward = (theta*action+reward_noise-price*probe).astype(np.float32)
    # Only these five physically observed quantities enter fitting.
    log = dict(prior=prior, actual_probe=probe, actual_measurement=measured,
               actual_final_action=action, observed_net_reward=realized_reward)
    evaluator = dict(prior=prior, theta=theta, eps_probe=ep, reward_noise=reward_noise)
    raw = b''.join(arr.tobytes() for arr in (prior, theta, ep, reward_noise, probe, action, measured, realized_reward))
    sha = hashlib.sha256(raw).hexdigest()
    log_sha = hashlib.sha256(b''.join(x.tobytes() for x in log.values())).hexdigest()
    return dict(seed=seed, role=cohort, n=n, log=log, eval=evaluator,
                source_sha256=sha, log_sha256=log_sha)


def laplace_log_ratio(observed, signal_mu, b):
    return (np.abs(observed + signal_mu) - np.abs(observed - signal_mu)) / b


def laplace_cdf(observed, mu, b):
    d = (observed-mu) / b
    return np.where(d < 0, .5*np.exp(np.minimum(d, 0)), 1-.5*np.exp(-np.maximum(d, 0)))


def expected_laplace_value(prior_log_odds, mu, b):
    """Analytic E[correct action after observation] without realizing a probe."""
    p = expit(prior_log_odds)
    limit = 2*mu/b
    threshold = np.clip(-prior_log_odds*b/2, -mu, mu)
    correct = p*(1-laplace_cdf(threshold, mu, b))+(1-p)*laplace_cdf(threshold,-mu,b)
    correct = np.where(prior_log_odds >= limit, p, correct)
    correct = np.where(prior_log_odds <= -limit, 1-p, correct)
    return np.maximum(0, 2*correct-1-np.abs(2*p-1))


def choose(tape, model, policy='laplace'):
    if policy not in ('laplace', 'oracle', 'never', 'always'):
        raise ValueError('P55_UNKNOWN_POLICY')
    c = np.asarray(tape['prior'], dtype=np.float64)
    theta = np.asarray(tape['theta'], dtype=np.float64)
    noise_p = np.asarray(tape['eps_probe'], dtype=np.float64)
    noise_r = np.asarray(tape['reward_noise'], dtype=np.float64)
    cue, cg, cs, pg, ps, cost = c.T
    if policy == 'oracle':
        prior, sg, vg, sp, vp = .5, 1., 1., 1., 1.
    else:
        prior = model['prior_positive']
        sg, vg, sp, vp = (model[k] for k in ('cue_scale', 'cue_variance', 'probe_scale', 'probe_variance'))
    if min(vg, vp) <= 0 or not 0 < prior < 1:
        raise ValueError('P55_INVALID_FITTED_LIKELIHOOD')
    prior_logodds = math.log(prior/(1-prior)) + laplace_log_ratio(cue/cs, sg*cg/cs, math.sqrt(vg/2))
    gain = expected_laplace_value(prior_logodds, sp*pg/ps, math.sqrt(vp/2))
    if policy == 'always':
        actual_probe = np.ones(len(c), dtype=np.float64)
    elif policy == 'never':
        actual_probe = np.zeros(len(c), dtype=np.float64)
    else:
        actual_probe = (gain > cost+1e-12).astype(np.float64)
    measurement = np.zeros(len(c), dtype=np.float64)
    ix = actual_probe > 0
    measurement[ix] = theta[ix]*pg[ix] + ps[ix]*noise_p[ix]
    final_logodds = prior_logodds + actual_probe*laplace_log_ratio(
        measurement/ps, sp*pg/ps, math.sqrt(vp/2))
    executed_action = np.where(final_logodds >= 0, 1., -1.)
    actual_reward = theta*executed_action + noise_r - cost*actual_probe
    weak = (np.abs(cue)/cs < .5) & (pg > .8)
    low, high = weak & (cost < .2), weak & (cost >= .4)
    return dict(net_reward=float(actual_reward.mean()), probe_fraction=float(actual_probe.mean()),
                actual_probe=actual_probe, actual_action=executed_action,
                actual_measurement=measurement, reward=actual_reward,
                actual_executed_probes=int(actual_probe.sum()),
                low_cost_probe=float(actual_probe[low].mean()) if low.any() else None,
                high_cost_probe=float(actual_probe[high].mean()) if high.any() else None,
                low_group_n=int(low.sum()), high_group_n=int(high.sum()),
                decisions_sha256=hashlib.sha256(np.column_stack((actual_probe, executed_action, actual_reward)).tobytes()).hexdigest())


def bootstrap_paired(x, y, seed):
    difference = np.asarray(x)-np.asarray(y)
    rng = np.random.default_rng(seed)
    boot = np.empty(1000)
    for i in range(1000):
        boot[i] = difference[rng.integers(0, len(difference), size=len(difference))].mean()
    return dict(gain=float(difference.mean()), lower_95=float(np.percentile(boot, 2.5)),
                upper_95=float(np.percentile(boot, 97.5)), resamples=1000, seed=seed)


def run():
    start = time.perf_counter()
    worlds = {seed: source(seed) for seed in COHORTS}
    fitted = p54.fit_moments(worlds[850001]['log'])
    wrong = p54.fit_moments(p54.shuffled_credit_log(worlds[850001]['log']))
    result = {}
    for seed in (850201, 850301, 850401):
        tape = worlds[seed]['eval']
        chosen = choose(tape, fitted, 'laplace')
        never = choose(tape, fitted, 'never')
        always = choose(tape, fitted, 'always')
        oracle = choose(tape, fitted, 'oracle')
        gaussian = p54.decisions(tape, fitted, 'moment')
        wrong_action = choose(tape, wrong, 'laplace')
        actions = dict(laplace=chosen, never=never, always=always, oracle=oracle, gaussian=gaussian, wrong_credit=wrong_action)
        ci = bootstrap_paired(chosen['reward'], never['reward'], BOOTSTRAP_SEED+seed)
        metrics = {name:{k:v for k,v in a.items() if k not in ('reward','actual_probe','actual_action','actual_measurement')} for name,a in actions.items()}
        if seed == 850401:
            gates = dict(sham_zero_probes=(chosen['actual_executed_probes']==0),
                         model_aware_sham_zero=(oracle['actual_executed_probes']==0))
        else:
            gates = dict(
                better_than_never_point02=chosen['net_reward'] >= never['net_reward']+.02,
                beats_gaussian_point01=chosen['net_reward'] >= gaussian['net_reward']+.01,
                within_oracle_point025=chosen['net_reward'] >= oracle['net_reward']-.025,
                probes_nontrivial=.05 <= chosen['probe_fraction'] <= .95,
                paired_ci_positive=ci['lower_95']>0,
                cost_sensitivity=(chosen['low_group_n']>=30 and chosen['high_group_n']>=30 and
                    chosen['low_cost_probe']>=chosen['high_cost_probe']+.15),
                wrong_credit_margin_point025=chosen['net_reward']>=wrong_action['net_reward']+.025)
        result[str(seed)] = dict(arms=metrics, paired_ci=ci, gates=gates)
    # Only learned fit scalars survive cold reconstruction; no evaluator latents.
    restored = json.loads(json.dumps(fitted, sort_keys=True))
    cold = {str(seed):choose(worlds[seed]['eval'], fitted)['decisions_sha256']==choose(worlds[seed]['eval'], restored)['decisions_sha256']
            for seed in (850201,850301,850401)}
    return dict(schema='ttt-hybrid.p55.laplace-factual-probe-dev.v1',
                classification='NEW_AUTHORED_DEV_ONLY_NOT_EXTERNAL_SCIENTIFIC_FIRST_USE',
                sources={str(seed):z['source_sha256'] for seed,z in worlds.items()},
                logged_sources={str(seed):z['log_sha256'] for seed,z in worlds.items()},
                fit=fitted, wrong_credit_fit=wrong, coeval_models_share_exact_fit_log=True,
                results=result, cold_exact=cold,
                frozen_dev_all_pass=all(all(row['gates'].values()) for row in result.values()) and all(cold.values()),
                wall_seconds=time.perf_counter()-start, 
                full_windows_ubuntu_regression='NOT_RUN',
                independent_scientific_first_use='NOT_RUN', FORGE='UNTOUCHED')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--receipt',type=Path,required=True)
    args=ap.parse_args()
    output=run()
    args.receipt.parent.mkdir(exist_ok=True,parents=True)
    args.receipt.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'frozen_all':output['frozen_dev_all_pass'], 'fit':output['fit'],
                      'OOD':{k:{'metrics':{n:{'reward':z['net_reward'], 'probes':z['probe_fraction']} for n,z in v['arms'].items()},
                                'checks':v['gates'],'paired':v['paired_ci']} for k,v in output['results'].items()},
                      'cold':output['cold_exact'],'time':output['wall_seconds']},indent=2))


if __name__=='__main__':
    main()
