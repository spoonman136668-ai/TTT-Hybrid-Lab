"""P50 prospective authored DEV controls; not external scientific qualification."""
import hashlib
from pathlib import Path
import io
import numpy as np
import pytest
from scripts import native_p50_nonalternating_policy_dev as p50
from scripts import native_p47_change_vs_nuisance_dev as p47

SHA={790001:'878678fe12b771e4ae2461963d878ba942bed8b94db6edb49179b4b5968a44da',
     790101:'59ab65b4a1a0f0f83fec6c5cad4d95ce5e263758a1e23fc4dfc5cb32e2e9a6c8',
     790103:'e2849e95616549923b3255162a4c2bb27b22e3d220f8fa30d6a6407838071b5a',
     790107:'4fe6f7475ddb5af4203411a9ec066ed60b9fa88e647f1708c470e18dba81c167'}

def blob(path):
    raw=Path(path).read_bytes()
    return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()

def test_original_p47_and_exact_new_source_custody():
    root=Path(__file__).resolve().parents[1]
    assert blob(root/'scripts/native_p47_change_vs_nuisance_dev.py')=='9be334264131db3bae79f19bed3271e060a1a903'
    assert blob(root/'scripts/native_p50_nonalternating_policy_dev.py')=='973624acca10cb0de7d6d3c9b9dc07a78ccdfc2b'

def test_four_new_frozen_cohorts_and_no_replayed_first_use():
    for seed,expect in SHA.items():
        world=p50.authored(seed)
        assert world['sha']==expect
        n=p50.COHORTS[seed][0]
        assert world['a'].shape==world['y'].shape==(n,8)
        assert [int((world['policy']==k).sum()) for k in range(4)]==[n//4]*4
        assert [int((world['kind']==k).sum()) for k in range(3)]==([0,0,4096] if seed==790001 else [768,384,384])
        assert np.isfinite(world['y']).all()
        assert np.all(np.isin(world['a'],(-1,1)))
    with pytest.raises(ValueError, match='P50_UNREGISTERED_SOURCE'):
        p50.authored(780101)

def test_original_alternating_only_refusal_and_non_alternating_rank():
    world=p50.authored(790101)
    ix=next(i for i in range(len(world['a'])) if world['policy'][i]==3 and
        np.any(world['a'][i,1:]!=-world['a'][i,:-1]) and
        p50.fit_one(world['a'][i],world['y'][i])['valid'])
    with pytest.raises(ValueError, match='P47_BAD_ACTION_SEQUENCE'):
        p47.estimate(world['a'][ix:ix+1],world['y'][ix:ix+1])
    p=p50.fit_one(world['a'][ix],world['y'][ix])
    assert p['valid'] and np.isfinite(p['full'])

def test_constant_action_cannot_identify_effect_and_is_not_invented():
    a=np.ones(8);t=p50.T;delta=.2+.3*t+.25*a
    out=p50.fit_one(a,delta)
    assert not out['valid'] and np.isnan(out['full'])
    outcome,alarm=p50.inferred([out],dict(causal=3.,nuisance=3.),'soft')
    assert np.isnan(outcome[0]) and not alarm[0]
    with pytest.raises(ValueError,match='P50_BAD_FACTUAL_EVENT_SEQUENCE'):
        p50.fit_one(np.zeros(8),delta)

def test_realized_only_replay_and_frozen_negative_gates():
    result=p50.study()
    assert not result['frozen_all_cohort_pass']
    assert [x['seed'] for x in result['reports']]==[790101,790103,790107]
    assert all(x['cold_bit_exact'] for x in result['reports'])
    assert all(not x['checks']['causal_recall'] for x in result['reports'])
    assert all(not x['checks']['causal_MAE'] for x in result['reports'])
    assert all(x['per_policy_coverage']['constant']['identifiable']==0 for x in result['reports'])
    assert [x['source_sha256'] for x in result['reports']]==[SHA[k] for k in (790101,790103,790107)]
