"""P91 prospective two-family stability/forgetting qualification and negative controls."""
import json
from pathlib import Path
import pytest
import torch
from experiments import p91_stability_forgetting_dev as p91

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)

def test_freeze_prior_and_original_blobs():
    assert p91.FREEZE=='96fe23932f87b04dd689bd147277a4cea8b250f3'
    assert all(p91.p86.blob(ROOT/p)==sha for p,sha in p91.BLOBS.items())

def test_two_families_distinct_source_and_frozen_sizes():
    assert p91.FAMILIES=={'A':{'FIT':910111,'VAL':910113},'B':{'FIT':910121,'VAL':910123}}
    assert (p91.STEPS,p91.BATCH,p91.CHECK)==(80,16,10)
    assert p91.SEEDS==(910201,910203,910207)

def test_reserved_source_first_use_is_unavailable():
    for s in [910117,910119,900111,900113,0]:
        with pytest.raises(ValueError,match='P91_SOURCE_FENCE'):p91.source(s,64)
    with pytest.raises(ValueError,match='P91_SOURCE_FENCE'):p91.source(910113,128)

def test_new_source_sha256_exact_cross_platform():
    expected={(910111,128):'13374e0ba586148fc1e13a1cf5846c5de122acd569e66a02817c569394a8940e',
              (910113,64):'8f72576cc1e4da6d121e83409259cc7015c1ef9181740375e9ad486b63443c4b',
              (910121,128):'214d1e415c0a2a786fe6f841fb87f798c162f9e8f307caadebe0c227c2ff6384',
              (910123,64):'6a90abe7783788f2eb171a11169130018084d76eca1d8add23293d816bdb7221'}

    assert all(p91.signature(p91.source(seed,n))==sha for (seed,n),sha in expected.items())

def test_missing_outcome_and_causal_novelty():
    for family in p91.FAMILIES.values():
        for split,seed in family.items():
            w=p91.source(seed,128 if split=='FIT' else 64)
            audit=p91.p86.audit(p91.p90.p86_world(w))
            assert audit['pass']
            assert w['changed_world'].sum().item()==len(w['effects'])//2
            assert torch.count_nonzero(p91.p90.p86_world(w)['query'][:,16:])==0

def test_conventional_fixed_recursions_match_original_P86():
    w=p91.source(910113,64)
    pred,state=p91.predict(w,'fixed')
    base=p91.p86.online(p91.p90.p86_world(w),1,.99)[0]
    assert state.shape==(64,16,16)
    torch.testing.assert_close(pred[:,-1],base,atol=1e-6,rtol=1e-6)

def test_cold_resume_all_arms_without_new_state():
    w=p91.source(910123,64)
    for arm in (*p91.ARMS,'fixed'):
        x=None if arm=='fixed' else p91.init(910201,arm).detach()
        assert p91.replay_cold(w,arm,x)

def test_future_query_outcome_is_never_observed():
    w=p91.source(910113,64)
    changed={k:v.clone() for k,v in w.items()}
    changed['effects'][:,-1]*=-1
    for arm in ('fixed','contradiction','protected'):
        theta=None if arm=='fixed' else p91.init(910201,arm).detach()
        with torch.no_grad():
            first,_=p91.predict(w,arm,theta)
            second,_=p91.predict(changed,arm,theta)
        assert torch.equal(first,second)

def test_controller_starts_equal_and_does_not_have_hidden_per_world_state():
    for seed in p91.SEEDS:
        c=p91.init(seed,'constant')
        for arm in ('observation','contradiction','protected'):
            theta=p91.init(seed,arm)
            assert torch.equal(theta[:2],c)
            assert torch.count_nonzero(theta[2:])==0
            assert theta.numel()==4

def test_complete_frozen_two_family_result_rejects_unmet_changed_gate():
    r=p91.study()
    assert r['status']=='DEV_FAIL_STABILITY_OR_ADAPTATION'
    assert r['all_conventional_qualified'] and not r['all_protected_qualified']
    assert not r['science_pass'] and not r['OOD_generated'] and not r['NEGATIVE_generated']
    assert len(r['rows'])==3 and len(r['controls'])==2
    for name in ('A','B'):
        assert r['controls'][name]['parity'] and r['controls'][name]['wrong_credit_pass']
    for row in r['rows']:
        assert row['trained_constant_qualified'] and not row['protected_qualified']
        assert all(row['cold_restores'].values())
        assert len(row['arms'])==4
        assert all(v['updates']==80 and len(v['trace'])==8 for v in row['arms'].values())
        for name in ('A','B'):
            p=row['arms']['protected']['families'][name]
            f=r['controls'][name]['fixed']
            assert p['final']<f['final'] and p['stable']<f['stable']
            assert p['changed']>0.85*f['changed']
