"""P56 preregistered model-family evidence, failed mixed-world gain and custody."""
from pathlib import Path
import hashlib
import inspect
import json
import numpy as np
import pytest
from scripts import native_p56_family_selection_dev as p56

ROOT=Path(__file__).resolve().parents[1]
PARENTS={
 'scripts/native_p54_noisy_factual_reward_dev.py':'c4239b125e3d5928807738aac117be89f202d7a1',
 'scripts/native_p55_laplace_transfer_dev.py':'fe211b950c34377768047f5f78e38007b1de0e18'
}


def test_pinned_parent_source_and_no_prior_consumed_seeds():
    for path,sha in PARENTS.items():
        raw=(ROOT/path).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==sha
    assert set(p56.FIT_SEEDS)=={860001,860101,860201}
    assert set(p56.OOD_SEEDS)=={860003,860103,860203}
    with pytest.raises(ValueError,match='P56_UNREGISTERED_SOURCE'):
        p56.source(850201)


def test_fresh_factual_log_masks_unexecuted_probe_and_keeps_evaluator_separate():
    for seed in p56.FIT_SEEDS:
        s=p56.source(seed)
        log=s['log']
        assert set(log)=={'prior','actual_probe','actual_measurement','actual_final_action','observed_net_reward'}
        assert len(log['prior'])==p56.FIT_N
        assert np.all(log['actual_measurement'][log['actual_probe']==0]==0)
        assert np.all(np.isin(log['actual_final_action'],(-1,1)))
        assert np.isfinite(log['observed_net_reward']).all()
        assert 'theta' not in log and 'probe_noise' not in log
        assert len(s['sha256'])==64 and len(s['log_sha256'])==64


def test_disjoint_factual_fit_and_selection_selection_pins():
    for seed,expected in [(860001,{'cue':'G','probe':'G','reward':'G'}),
                          (860101,{'cue':'L','probe':'L','reward':'L'}),
                          (860201,{'cue':'G','probe':'L','reward':'G'})]:
        log=p56.source(seed)['log']
        first={k:v[:-p56.SELECT_N] for k,v in log.items()}
        other={k:v[-p56.SELECT_N:] for k,v in log.items()}
        result=p56.fit_model(first,other)
        assert result['family']==expected
        assert result['fit_records']==16384
        assert result['selection_records']==4096
        assert result['fit_executed_probes']>7500
        assert result['selection_executed_probes']>1900
        assert all(np.isfinite(v) for scores in result['selection_scores'].values() for v in scores.values())
    bad={k:v.copy() for k,v in other.items()}
    bad['actual_measurement'][np.flatnonzero(bad['actual_probe']==0)[0]]=999.
    with pytest.raises(ValueError,match='P56_UNEXECUTED_MEASUREMENT_LEAK'):
        p56.fit_model(first,bad)
    with pytest.raises(ValueError,match='P56_NONFACTUAL_LOG_SCHEMA'):
        p56.fit_model({**first,'theta':np.ones(len(first['prior']))},other)


def test_gaussian_and_laplace_likelihood_and_probe_value_physical_limits():
    x=np.array([-.5,0,.5],dtype=float)
    assert np.all(np.isfinite(p56.logpdf(x,0,1,'G')))
    assert np.all(np.isfinite(p56.logpdf(x,0,1,'L')))
    assert np.allclose(p56.posterior_logratio(x,0,1,'L'),0)
    assert np.allclose(p56.diagnostic_gain(np.array([-2.,0.,2.]),np.zeros(3),1,'G'),0,atol=1e-10)
    assert np.allclose(p56.diagnostic_gain(np.array([-2.,0.,2.]),np.zeros(3),1,'L'),0,atol=1e-10)
    with pytest.raises(ValueError,match='P56_UNSUPPORTED_FAMILY'):
        p56.logpdf(x,0,1,'Cauchy')


def test_cold_model_family_and_actual_zero_information_sham():
    data=p56.source(860201)
    model=p56.fit_model({k:v[:-p56.SELECT_N] for k,v in data['log'].items()},
                        {k:v[-p56.SELECT_N:] for k,v in data['log'].items()})
    sham=p56.source(p56.SHAM_SEED)
    actual=p56.decide(sham['eval'],model)
    assert actual['actual_probe_count']==0
    reloaded=json.loads(json.dumps(model,sort_keys=True))
    cold=p56.decide(sham['eval'],reloaded)
    assert actual['decisions_sha256']==cold['decisions_sha256']
    assert np.array_equal(actual['reward'],cold['reward'])


def test_frozen_mixed_environment_failure_is_preserved_not_tuned():
    result=p56.experiment()
    assert not result['frozen_all_gates']
    assert result['sham']['actual_probe_count']==0
    assert [result['rows'][str(x)]['fitted_model']['family']['probe'] for x in p56.OOD_SEEDS]==['G','L','L']
    for seed in p56.OOD_SEEDS:
        row=result['rows'][str(seed)]
        assert row['gates']['correct_family']
        assert row['gates']['cold_exact']
        assert row['gates']['positive_paired_CI']
        assert row['gates']['learns_positive_value']
    mixed=result['rows']['860203']
    assert not mixed['gates']['near_stronger_fixed']
    assert not mixed['gates']['mixed_strong_gain']
    assert mixed['arms']['gaussian']['mean_reward']>mixed['arms']['selected']['mean_reward']
    assert mixed['arms']['selected']['mean_reward']>mixed['arms']['laplace']['mean_reward']


def test_receipt_is_dev_only_and_contains_all_frozen_sources():
    path=ROOT/'experiments/native_p56_family_selection_dev_receipt.json'
    if not path.exists():
        pytest.skip('Receipt not yet committed during source-code authoring')
    receipt=json.loads(path.read_text())
    assert receipt['classification']=='FROZEN_DEV_FAIL_NOT_INDEPENDENT_SCIENCE'
    assert receipt['scientific_first_use']=='NOT_RUN'
    assert receipt['overall_frozen_gate']=='FAIL'
    assert len(receipt['source_shas'])==7
