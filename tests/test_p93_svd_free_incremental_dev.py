"""P93 source-exact SVD-free incremental factor and frozen negative tests."""
from pathlib import Path
import torch
import pytest
from experiments import p93_svd_free_incremental_dev as p93

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def test_original_parent_custody_and_freeze():
    assert p93.FREEZE=='0740cf629b1c775ff404851ded1411a41581258c'
    assert p93.blob(ROOT/'experiments/p92_lowrank_twospeed_dev.py')==p93.SOURCE_BLOBS['experiments/p92_lowrank_twospeed_dev.py']
    if (ROOT/'model.py').exists():
        assert p93.blob(ROOT/'model.py')==p93.SOURCE_BLOBS['model.py']


def test_seed_families_and_frozen_selection_grids():
    assert p93.FAMILIES=={'A':{'FIT':930111,'VAL':930113},'B':{'FIT':930121,'VAL':930123}}
    assert len(p93.FACTOR_GRID)==12 and len(p93.FULL_GRID)==6
    assert p93.FACTOR_GRID[0]==(.10,.025,.95)


def test_protected_source_fails_closed():
    for seed in (930117,930119,920113,930999):
        with pytest.raises(ValueError,match='P93_PROTECTED_OR_INVALID_SOURCE'):p93.source(seed,64)
    with pytest.raises(ValueError,match='P93_PROTECTED_OR_INVALID_SOURCE'):p93.source(930113,96)


def test_exact_crossplatform_new_source_sha256():
    expected={(930111,96):'05e1a3020f81452159531f614694c1db3e833c71c18f5f96ee08c3ca4c962141',
              (930113,64):'7edbfccc6801e144dfc0e24fc79eeb151955dbac5bee41c59f21832def5fc6dc',
              (930121,96):'ecd2c9dcdf3c2e3e7b470186554c97e0ddaae0521be69551fc1ae2778f8520a1',
              (930123,64):'a021b71a347e9295decc033571c5fc6637473d04562b2c011c1554b57b55ec87'}
    for (seed,n),hash_value in expected.items():
        w=p93.source(seed,n)
        assert p93.checksum(w)==hash_value
        assert p93.p92.audit(w)['pass']
        assert p93.checksum(w)==p93.checksum(p93.source(seed,n))


def test_initial_factors_are_256_float_no_SVD_state():
    u,v=p93.initial_state(8,2)
    assert u.shape==(8,2,16,4) and v.shape==(8,2,4,16)
    assert (u.numel()+v.numel())//8==256
    assert torch.equal(u,torch.zeros_like(u))
    assert p93.walsh().shape==(16,16)
    torch.testing.assert_close(p93.walsh()@p93.walsh().T,torch.eye(16),atol=0,rtol=0)


def test_extra_model_ablation_is_lower_capacity():
    w=p93.source(930113,64)
    _,(u,v)=p93.factor_rollout(w,(.75,.1,.99),mode='single')
    assert (u.numel()+v.numel())//len(w['effects'])==128


def test_svd_free_factors_cold_restart_bitwise():
    assert p93.cold(p93.source(930113,64),(.75,.10,.99))
    assert p93.cold(p93.source(930123,64),(.75,.10,.99))


def test_query_future_outcome_is_not_read():
    w=p93.source(930113,64)
    q={k:v.clone() for k,v in w.items()}
    q['effects'][:,-1]*=-1
    for mode in ('dual','single','fixedV'):
        a,_=p93.factor_rollout(w,(.75,.1,.99),mode=mode)
        b,_=p93.factor_rollout(q,(.75,.1,.99),mode=mode)
        assert torch.equal(a,b)


def test_invalid_extra_state_is_rejected():
    w=p93.source(930113,64)
    with pytest.raises(ValueError,match='P93_INVALID_MODE'):p93.factor_rollout(w,(.75,.1,.99),mode='x')
    with pytest.raises(ValueError,match='P93_INVALID_SLOT_COUNT'):p93.initial_state(8,3)
    u,v=p93.initial_state(64,2)
    with pytest.raises(ValueError,match='P93_INVALID_FACTOR_STATE'):
        p93.factor_rollout(w,(.75,.1,.99),state=(u[:,:1],v))


def test_frozen_complete_negative_no_holdout():
    r=p93.run()
    assert r['status']=='DEV_FAIL_QUALITY'
    assert not r['scientific_pass'] and not r['OOD_generated'] and not r['NEGATIVE_generated']
    assert not r['full_hybrid_trained']
    assert r['FIT']['full_selected']['eta']==1 and r['FIT']['full_selected']['decay']==.99
    assert r['FIT']['factor_selected']['params']==[.75,.10,.99]
    assert len(r['FIT']['factor_grid'])==12 and len(r['FIT']['full_grid'])==6
    for name in ('A','B'):
        assert r['VAL'][name]['wrong_credit_pass'] and r['VAL'][name]['cold_bitwise']
        assert r['VAL'][name]['state_float32']==256
        assert not r['gates'][name]['combined'] and not r['gates'][name]['stable'] and not r['gates'][name]['changed']
        assert r['VAL'][name]['factor']['combined']>r['VAL'][name]['full_fit']['combined']
