"""P53 precommitted-source DEV controls and exact negative science receipt."""
from pathlib import Path
import hashlib
import inspect
import json
import io
import numpy as np
import pytest
import torch
from scripts import native_p53_factual_likelihood_dev as p53

EXPECTED={
 830101:'da7fc882f4f660bedad276c02e59e8db29d406f5f53498af25cef00071718ab9',
 830103:'54352c1423a3fbf07b862a9a36829d950ac25b5db1277f80171263032a39bbcf',
 830107:'87245b03b022de8857c159a9c8aaff49721e19c87458b2b4f7585a19430e85d8',
 830109:'5161498e050bc1e935f9f01578b00f101c79b449ffb7651da6944754ae6663b9',
}


def test_source_pins_and_unchanged_original_P52():
    source=Path(__file__).resolve().parents[1]/'scripts/native_p52_probe_value_dev.py'
    raw=source.read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    assert blob=='0f6e1ab12132b062fa9d447d0f2014d045520eb8'
    for seed, digest in EXPECTED.items():
        d=p53.source(seed)
        assert d['source_sha256']==digest
        assert d['log']['prior'].shape==(p53.SOURCE[seed][0],6)
        assert int((d['log']['actual_probe']>0).sum())>0
    with pytest.raises(ValueError,match='P53_UNREGISTERED_SOURCE'):
        p53.source(820101)


def test_real_reward_reconstructs_binary_sign_only_after_executed_action():
    d=p53.source(830101)
    observed=d['log']
    recovered=(observed['observed_net_reward']+observed['prior'][:,5]*observed['actual_probe'])/observed['actual_final_action']
    np.testing.assert_allclose(recovered,d['data']['theta'].numpy(),rtol=0,atol=1e-6)
    assert np.all(observed['actual_measurement'][observed['actual_probe']==0]==0)
    model=p53.fitted_likelihood(observed)
    assert model['physical_training_facts']==89600
    assert model['executed_training_probes']>44000
    assert model['executed_training_probes']<46000
    assert model['reward_sign_max_residual']<1e-6
    assert all(.95<model[k]<1.05 for k in ('cue_scale','probe_scale','cue_variance','probe_variance'))


def test_invalid_or_forged_factual_observations_refused():
    d=p53.source(830103)
    log=d['log']; p=p53.fitted_likelihood(log)
    assert 0<p['prior_positive']<1
    with pytest.raises(ValueError,match='P53_LOG_SCHEMA_REFUSED'):
        p53.fitted_likelihood({**log,'theta':d['data']['theta'].numpy()})
    corrupt={**log,'actual_measurement':log['actual_measurement'].copy()}
    ix=int(np.flatnonzero(log['actual_probe']==0)[0]);corrupt['actual_measurement'][ix]=1000.
    with pytest.raises(ValueError,match='P53_UNEXECUTED_MEASUREMENT_LEAK'):
        p53.fitted_likelihood(corrupt)
    corrupt={**log,'observed_net_reward':log['observed_net_reward'].copy()}
    corrupt['observed_net_reward'][0]=.123
    with pytest.raises(ValueError,match='P53_REWARD_NOT_BINARY_EFFECT_IDENTIFIABLE'):
        p53.fitted_likelihood(corrupt)


def test_no_latent_labels_enter_factual_fitting_api_or_action_choice():
    fit=inspect.getsource(p53.fitted_likelihood)
    assert 'log: dict' in fit and 'source(' not in fit
    assert "['theta']" not in fit and "['eps_probe']" not in fit
    # Evaluate a parameter fit without passing any hidden sign to the fitted model.
    p=p53.fitted_likelihood(p53.source(830101)['log'])
    d=p53.source(830107)
    a=p53.decision(d['data'],p,'learned')
    b=p53.decision(d['data'],p,'never')
    assert a['net_reward']>b['net_reward']
    assert a['probe_rate']>.05
    assert a['actual_executed_probes']==round(a['probe_rate']*2048)
    assert a['low_probe_fraction']>=a['high_probe_fraction']+.15


def test_fitted_parameters_cold_JSON_roundtrip_exact_physical_replay():
    fit=p53.fitted_likelihood(p53.source(830101)['log'])
    d=p53.source(830109)['data']
    fresh=json.loads(json.dumps(fit,sort_keys=True))
    a=p53.decision(d,fit,'learned')
    b=p53.decision(d,fresh,'learned')
    assert a['decisions_sha256']==b['decisions_sha256']
    np.testing.assert_array_equal(a['reward'],b['reward'])
    assert p53.paired_ci(a['reward'],p53.decision(d,fit,'never')['reward'])['lower_95']>0


def test_complete_frozen_DEV_result_retained_and_no_unsupported_science_claim():
    # A persisted receipt is a DEV execution record, not new external first use.
    receipt=Path(__file__).resolve().parents[1]/'experiments/native_p53_factual_likelihood_dev_receipt.json'
    d=json.loads(receipt.read_text())
    assert d['source_hashes']=={str(k):v for k,v in EXPECTED.items()}
    assert d['frozen_positive']
    assert d['independent_scientific_first_use']=='NOT_RUN'
    assert len(d['original_P52_RL_on_new_source'])==3
    for s in (830107,830109):
        o=d['ood'][str(s)]
        assert all(o['gate'].values())
        assert o['paired_bootstrap']['lower_95']>0
        assert o['learned']['net_reward']>o['learned_never']['net_reward']
        for b in d['original_P52_RL_on_new_source']:
            assert b['arms']['factorized']['eval'][str(s)]['probe_rate']==0
            assert b['arms']['gru']['eval'][str(s)]['probe_rate']==0
