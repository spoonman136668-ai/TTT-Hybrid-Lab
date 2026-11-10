"""P54 bounded no-secret source, noisy reward, controls and custody tests."""
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from scripts import native_p54_noisy_factual_reward_dev as p54

ROOT=Path(__file__).resolve().parents[1]
SOURCE_SHA={
 840001:'1a94e605bdd23877b13e8e563b027cc2fe970beb0a133bd6954e3307a15114eb',
 840101:'b3eb1d26c74e56d23b933cb86f87b5b980b5f2cfa3a0102960d2b9cb7b9671d0',
 840201:'40c898637273978a6cbed8ea3dba4f99c6ce09cd1fa28a48c8a3c960ffa678a1',
 840301:'6302fa3dbee7d7f14324c359f699a5058f183d299d0449cc3f29b786544390f3',
 840401:'7ecdf74cc915cec7fa062f9ce5aab0a305a149f2d395112680407d9b0e379913',
}

def test_pinned_original_P53_and_fresh_source_custody():
    raw=(ROOT/'scripts/native_p53_factual_likelihood_dev.py').read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    assert blob=='e514261e0923c6db9274e5bf4fc91d15106376ef'
    for seed,sha in SOURCE_SHA.items():
        w=p54.source(seed)
        assert w['source_sha256']==sha
        assert w['n']==p54.SOURCES[seed][0]
        assert np.isfinite(w['log']['observed_net_reward']).all()
        assert np.all(w['log']['actual_measurement'][w['log']['actual_probe']==0]==0)
    with pytest.raises(ValueError,match='P54_UNFROZEN_SOURCE'):
        p54.source(830101)


def test_no_single_episode_reward_label_reveals_hidden_sign():
    w=p54.source(840001)
    log=w['log']
    z=(log['observed_net_reward']+log['prior'][:,5]*log['actual_probe'])/log['actual_final_action']
    assert float(np.var(z))>1.3
    assert float(np.mean(np.abs(np.abs(z)-1.)<1e-4))<.01
    with pytest.raises(ValueError,match='P53_REWARD_NOT_BINARY_EFFECT_IDENTIFIABLE'):
        p54.p53.fitted_likelihood(log)
    assert not np.allclose(z,w['evaluation']['theta'])


def test_only_executed_probe_and_actual_reward_enter_fit():
    w=p54.source(840001)
    log=w['log']
    assert set(log)=={'prior','actual_probe','actual_measurement','actual_final_action','observed_net_reward'}
    source_code=inspect.getsource(p54.fit_moments)
    assert "['theta']" not in source_code and 'eps_probe' not in source_code
    with pytest.raises(ValueError,match='P54_UNAUTHORIZED_FIT_FIELD'):
        p54.fit_moments({**log,'theta':w['evaluation']['theta']})
    bad={**log,'actual_measurement':log['actual_measurement'].copy()}
    bad['actual_measurement'][np.flatnonzero(log['actual_probe']==0)[0]]=42
    with pytest.raises(ValueError,match='P54_UNEXECUTED_MEASUREMENT'):
        p54.fit_moments(bad)
    wrong={**log,'actual_final_action':np.zeros(len(log['actual_final_action']))}
    with pytest.raises(ValueError,match='P54_UNEXECUTED_ACTION'):
        p54.fit_moments(wrong)
    fit=p54.fit_moments(log)
    assert fit['executed_training_probes']==24436
    assert fit['exact_sign_fraction']<.01
    assert all(.94<fit[k]<1.06 for k in ('cue_scale','cue_variance','probe_scale','probe_variance'))


def test_credit_permutation_is_matched_source_and_destroys_signal():
    w=p54.source(840001)
    bad=p54.shuffled_credit_log(w['log'])
    for name in ('prior','actual_probe','actual_measurement','actual_final_action'):
        np.testing.assert_array_equal(bad[name],w['log'][name])
    good=p54.fit_moments(w['log']);wrong=p54.fit_moments(bad)
    assert wrong['executed_training_probes']==good['executed_training_probes']
    assert max(wrong['cue_scale'],wrong['probe_scale'])<.08
    x=w['evaluation']
    assert p54.decisions(x,good)['net_reward']>p54.decisions(x,wrong)['net_reward']+.15


def test_same_actual_probe_actions_for_both_neural_control_families():
    torch.manual_seed(55)
    fit=p54.source(840001)['log']
    ix=torch.arange(128)
    prior=torch.from_numpy(fit['prior'][ix].copy())
    flag=torch.from_numpy(fit['actual_probe'][ix].copy())
    obs=torch.from_numpy(fit['actual_measurement'][ix].copy())
    target=torch.from_numpy(((fit['observed_net_reward']+fit['prior'][:,5]*fit['actual_probe'])/fit['actual_final_action'])[ix].copy())
    counts=[]
    for cls in (p54.MLPRegressor,p54.GRURegressor):
        m=cls();counts.append(sum(t.numel() for t in m.parameters()))
        a,b=m(prior,flag,obs)
        assert a.shape==b.shape==target.shape
        loss=((a-target)**2+(b-target)**2).mean();loss.backward()
        assert all(t.grad is not None and bool(torch.isfinite(t.grad).all()) for t in m.parameters())
    assert counts==[560,562]
    assert max(counts)/min(counts)<1.10


def test_physical_sham_and_frozen_heldout_positive_with_cold_replay():
    result=p54.run()
    assert result['frozen_dev_positive']
    assert result['moment_cold_exact']
    assert result['exact_P53_sign_fitter_refused']
    assert result['source_sha256']=={str(s):sha for s,sha in SOURCE_SHA.items()}
    assert result['actual_training_probe_count']==24436
    for seed in (840201,840301):
        r=result['results'][str(seed)]
        assert all(r['gates'].values())
        assert r['paired_ci']['lower_95']>0
        assert r['arms']['moment']['probe_fraction']>.20
    sham=result['results']['840401']
    assert sham['gates']['zero_sham_probe']
    assert sham['arms']['moment']['actual_executed_probes']==0
    assert all(v['cold_bit_exact'] for row in result['neural'] for v in row['arms'].values())
    assert result['original_full_Windows_Ubuntu_regression']=='NOT_RUN'
