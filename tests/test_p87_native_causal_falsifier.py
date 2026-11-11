"""P87 source-exact, causal custody and trained-result negative controls.

The 3x3x400-update authored study is executed once as a separate dev study;
this regression checks its immutable evidence without spending a second budget.
"""
import json
from pathlib import Path
import torch
import pytest
from experiments import p87_native_causal_falsifier as p87
from experiments import p86_portable_causal_preflight as p86

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def test_original_git_blobs_and_frozen_protocol():
    assert all(p86.blob(ROOT/k)==v for k,v in p87.EXPECTED_BLOBS.items())
    assert p87.FREEZE=='ab92a50e4c27261b0aa02aa31fd4263b17980ae7'


def test_fresh_seeds_and_exact_training_budget():
    assert p87.SEEDS=={'FIT':870111,'VAL':870113}
    assert p87.RESERVED=={'OOD':870117,'NEGATIVE':870119}
    assert p87.MODEL_SEEDS==(870201,870203,870207)
    assert (p87.STEPS,p87.BATCH,p87.CHECK)==(400,16,40)


def test_reservations_denied_at_source_boundary():
    for split in ('OOD','NEGATIVE','P86_CAL'):
        with pytest.raises(ValueError,match='P87_PROTECTED_OR_UNKNOWN_SOURCE'):
            p87.source(split)
    for count in (0,7,9,129,385):
        with pytest.raises(ValueError,match='P87_INVALID_SOURCE_COUNT'):
            p87.source('VAL',count)


def test_fresh_portable_cohort_byte_hash_custody():
    assert p87.checksum(p87.source('FIT'))=='1b7d00e764a4e6f38307439e8ca66905d72851907cd0e554db5fa835a39518a9'
    assert p87.checksum(p87.source('VAL'))=='ed76f5e93fa8af3ca36e087d90106f5f13bd96094dc2aa2fdfaac3cf48d9612c'


def test_shape_hidden_outcome_and_source_determinism():
    for split in p87.SEEDS:
        x=p87.source(split,8)
        assert x['history'].shape==(8,48,32)
        assert x['query'].shape==(8,32)
        assert x['effect'].shape==(8,16)
        assert x['changed_world'].sum().item()==4
        assert not torch.count_nonzero(x['query'][:,16:])
        assert p87.checksum(x)==p87.checksum(p87.source(split,8))
        assert p86.audit(x)['pass']


def test_future_effect_leak_fails_closed():
    x=p87.source('VAL',8)
    x['query'][:,16:]=x['effect']
    assert not p86.audit(x)['pass']


def test_cold_restore_bitwise_original_256_float_ttt():
    assert p87.exact_cold_restore(p87.source('VAL',8))


def test_original_and_trained_baselines_have_honest_parameter_counts():
    expected={'hybrid':5297,'gru':2672,'ssm':1936}
    for arm,n in expected.items():
        z=p87.new_model(arm,870201)
        assert sum(q.numel() for q in z.parameters() if q.requires_grad)==n
        assert p87.fingerprint(z)==p87.fingerprint(p87.new_model(arm,870201))
    with pytest.raises(ValueError,match='P87_UNKNOWN_MODEL'):
        p87.new_model('oracle',870201)


def test_all_three_models_accept_identical_causal_observation_streams():
    w=p87.source('VAL',8)
    x=p87.input_tensor(w)
    for arm in p87.ARMS:
        m=p87.new_model(arm,870201).eval()
        with torch.no_grad():
            y=m(x)
            if y.ndim==3:y=y[:,-1]
        assert y.shape==(8,16) and torch.isfinite(y).all()


def test_native_hybrid_prefix_is_causal_to_future_query():
    w=p87.source('FIT',8)
    x=p87.input_tensor(w)
    changed=x.clone();changed[:,-1,:16]*=-1
    m=p87.new_model('hybrid',870201).eval()
    with torch.no_grad():
        a=m(x);b=m(changed)
    torch.testing.assert_close(a[:,:-1],b[:,:-1],rtol=1e-6,atol=1e-6)


def test_p86_cal_selected_fixed_control_remains_competent_on_fresh_VAL():
    w=p87.source('VAL',16)
    fixed=p86.error(p86.online(w,1.,.99)[0],w)
    original=p86.error(p86.p78_original(w)[0],w)
    zero=p87.mse_by_regime(torch.zeros_like(w['effect']),w['effect'],w['changed_world'])
    assert fixed['combined']<zero['combined']
    assert original['combined']<zero['combined']


def test_frozen_completed_dev_negative_result_no_heldout():
    r=json.loads((ROOT/'experiments/p87_native_causal_falsifier_dev_result.json').read_text())
    assert r['status']=='DEV_FAIL_TRAINED_CONVENTIONAL_UNQUALIFIED'
    assert not r['scientific_pass'] and not r['OOD_generated'] and not r['negative_generated']
    assert r['full_hybrid_trained'] and r['cold_restore_bitwise_P78']
    assert r['cohorts']['FIT']['sha256']=='1b7d00e764a4e6f38307439e8ca66905d72851907cd0e554db5fa835a39518a9'
    assert r['cohorts']['VAL']['sha256']=='ed76f5e93fa8af3ca36e087d90106f5f13bd96094dc2aa2fdfaac3cf48d9612c'
    assert len(r['rows'])==3
    assert all(len(row['models'])==3 for row in r['rows'])
    assert all(all(m['num_updates']==400 and len(m['validation_trace'])==10
            for m in row['models'].values()) for row in r['rows'])
    assert all(not row['eligible']['gru']['pass'] and not row['eligible']['ssm']['pass']
               and not row['eligible']['hybrid']['pass'] for row in r['rows'])
