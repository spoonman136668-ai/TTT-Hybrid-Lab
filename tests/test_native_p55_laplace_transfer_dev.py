"""P55 prospectively frozen, independent no-oracle source and negative controls."""
from pathlib import Path
import hashlib
import io
import json
import math
import numpy as np
import pytest
from scripts import native_p55_laplace_transfer_dev as p55
from scripts import native_p54_noisy_factual_reward_dev as p54

HASHES = {
    850001:'98c7e68fb33947b9dbe0eba794b269a12ddc3a54b717f091aad9f6db87d0d280',
    850101:'1fe0b9b0878c459ea27d8672ea80addd867e27254c216a251330420d9f8b2d28',
    850201:'e747d93d550bd2c65dbf80db0b0d934819a9ccc70c2f451ed06162fc1467a702',
    850301:'5284cc83603e9e873c14ad7e8c4584e04605533bb3676024a76bc67630ef1534',
    850401:'50d300b72957153d7204385c9156e7d9c85512473ba3261e4d8057092a47755b',
}


def test_exact_original_p54_and_frozen_sources():
    root=Path(__file__).resolve().parents[1]
    raw=(root/'scripts/native_p54_noisy_factual_reward_dev.py').read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\x00'+raw).hexdigest()
    assert blob=='c4239b125e3d5928807738aac117be89f202d7a1'
    for seed,digest in HASHES.items():
        obj=p55.source(seed)
        assert obj['source_sha256']==digest
        assert len(obj['log']['prior'])==p55.COHORTS[seed][0]
        log=obj['log']
        assert set(log)=={'prior','actual_probe','actual_measurement','actual_final_action','observed_net_reward'}
        assert np.all(log['actual_measurement'][log['actual_probe']==0]==0)
        assert np.all(np.isin(log['actual_final_action'],[-1,1]))
        assert np.isfinite(log['observed_net_reward']).all()
    with pytest.raises(ValueError,match='P55_SOURCE_NOT_FROZEN'):
        p55.source(840001)


def test_no_oracle_theta_or_unexecuted_probe_in_fitting():
    log=p55.source(850001)['log']
    trained=p54.fit_moments(log)
    assert trained['physical_training_facts']==49152
    assert trained['executed_training_probes']==24557
    assert 0.97 < trained['cue_scale'] < 1.04
    assert 0.97 < trained['probe_scale'] < 1.04
    assert not np.isclose(trained['reward_proxy_variance'],0)
    with pytest.raises(ValueError,match='P54_UNAUTHORIZED_FIT_FIELD'):
        p54.fit_moments({**log, 'theta':np.ones(49152)})
    bad={**log,'actual_measurement':log['actual_measurement'].copy()}
    bad['actual_measurement'][int(np.flatnonzero(log['actual_probe']==0)[0])]=1
    with pytest.raises(ValueError,match='P54_UNEXECUTED_MEASUREMENT'):
        p54.fit_moments(bad)


def test_laplace_likelihood_and_value_of_information_analytic_limits():
    mus=np.array([0.,.2,.5,1.,2.])
    b=np.ones_like(mus)
    zero=np.zeros_like(mus)
    gain=p55.expected_laplace_value(zero,mus,b)
    np.testing.assert_allclose(gain,1-np.exp(-mus),atol=1e-12,rtol=1e-12)
    # Uninformative diagnostic cannot improve decisions even under nonzero priors.
    np.testing.assert_allclose(p55.expected_laplace_value(np.array([-3.,0.,3.]),np.zeros(3),np.ones(3)),0,atol=1e-12)
    # In very confident states, even an observed Laplace measurement cannot flip decision.
    np.testing.assert_allclose(p55.expected_laplace_value(np.array([10.,-10.]),np.array([.5,.5]),np.ones(2)),0,atol=1e-12)
    ratios=p55.laplace_log_ratio(np.array([-2.,0.,2.]),np.ones(3),np.ones(3))
    np.testing.assert_allclose(ratios,[-2.,0.,2.],atol=1e-12)


def test_fitted_gaussian_and_laplace_share_actual_logs_and_worlds():
    fit=p55.source(850001)['log']
    gaussian=p54.fit_moments(fit)
    learned=p54.fit_moments(fit)
    assert gaussian==learned
    w=p55.source(850201)['eval']
    a=p55.choose(w,learned)
    g=p54.decisions(w,gaussian,'moment')
    assert 0<a['probe_fraction']<1 and 0<g['probe_fraction']<1
    # Actual physical reward is a real action's signed outcome plus reward noise less executed probe cost.
    for arm in (a,g):
        expected=w['theta']*arm['actual_action'] + w['reward_noise']-w['prior'][:,5]*arm['actual_probe']
        np.testing.assert_allclose(arm['reward'],expected,atol=1e-12)
        assert np.all(arm['actual_measurement'][arm['actual_probe']==0]==0)
    assert a['decisions_sha256']!=g['decisions_sha256']


def test_sham_and_cold_likelihood_state_are_deterministic():
    fit=p54.fit_moments(p55.source(850001)['log'])
    w=p55.source(850401)['eval']
    a=p55.choose(w,fit)
    assert a['actual_executed_probes']==0
    b=p55.choose(w,json.loads(json.dumps(fit,sort_keys=True)))
    assert a['decisions_sha256']==b['decisions_sha256']
    with pytest.raises(ValueError,match='P55_INVALID_FITTED_LIKELIHOOD'):
        p55.choose(w,{**fit,'probe_variance':0})


def test_frozen_partial_gain_and_missing_strict_margin_preserved():
    z=p55.run()
    assert z['sources']=={str(s):v for s,v in HASHES.items()}
    assert z['coeval_models_share_exact_fit_log']
    assert not z['frozen_dev_all_pass']
    assert all(z['cold_exact'].values())
    for seed in ('850201','850301'):
        q=z['results'][seed]
        assert not q['gates']['beats_gaussian_point01']
        assert all(value for k,value in q['gates'].items() if k!='beats_gaussian_point01')
        assert q['paired_ci']['lower_95']>0
        assert q['arms']['laplace']['net_reward']>q['arms']['gaussian']['net_reward']
    assert z['results']['850401']['gates']['sham_zero_probes']
