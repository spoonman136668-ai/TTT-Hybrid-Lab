"""P92 exact source, causal state, cold restore and strict negative controls."""
from pathlib import Path
import hashlib
import torch
import pytest
from experiments import p92_lowrank_twospeed_dev as p92

ROOT=Path(__file__).resolve().parent
torch.set_num_threads(1)


def test_frozen_protocol_and_rule_count():
    assert p92.FREEZE=='3af59fa35a180f52d4f827216a51c0c5d5b31b63'
    assert len(p92.SINGLE_GRID)==6 and len(p92.DUAL_GRID)==8
    assert p92.FAMILIES=={'A':{'FIT':920111,'VAL':920113},'B':{'FIT':920121,'VAL':920123}}


def test_protected_ood_negative_and_wrong_shape_refused():
    for seed in (920117,920119,900111,0):
        with pytest.raises(ValueError,match='P92_PROTECTED_OR_INVALID_SOURCE'):p92.source(seed,64)
    with pytest.raises(ValueError,match='P92_PROTECTED_OR_INVALID_SOURCE'):p92.source(920113,96)


def test_cross_platform_exact_dyadic_source_sha():
    expected={(920111,96):'c4db80417b60230878a30b12d6b49d922e1b3ad11ccd98eda3178c930998f1e5',
              (920113,64):'86254f9f817f60523e864b718a4a175dfafcb5fafaa450739b9a446a46523904',
              (920121,96):'25dd4c9f92500991a7f1b72b8da1c533eeeb86412b64f101fdac5a1b241a9976',
              (920123,64):'e30a07fc7ca1e90e986823bed7682a64692c95b6a3bf09202cafa72790929826'}
    for (seed,n),sha in expected.items():
        w=p92.source(seed,n)
        assert p92.checksum(w)==sha
        assert p92.audit(w)['pass']
        assert p92.checksum(w)==p92.checksum(p92.source(seed,n))


def test_exact_256_float_persistent_state_and_factor_shapes():
    w=p92.source(920113,64)
    _,fw=p92.full(w)
    _,parts=p92.dual(w,(1.,.97,.25,.99))
    assert fw.shape==(64,16,16)
    assert [x.shape for x in parts]==[(64,16,4),(64,4,16),(64,16,4),(64,4,16)]
    assert sum(x.numel() for x in parts)//len(w['actions'])==256
    assert torch.isfinite(fw).all() and all(torch.isfinite(p).all() for p in parts)


def test_query_effect_is_withheld_even_under_counterfactual_change():
    w=p92.source(920113,64)
    counter={k:v.clone() for k,v in w.items()}
    counter['effects'][:,48]*=-1
    f0,_=p92.full(w);f1,_=p92.full(counter)
    d0,_=p92.dual(w,(1.,.97,.25,.99));d1,_=p92.dual(counter,(1.,.97,.25,.99))
    assert torch.equal(f0,f1) and torch.equal(d0,d1)


def test_factor_storage_bitwise_cold_resume():
    assert p92.cold(p92.source(920113,64),(1.,.97,.25,.99))
    assert p92.cold(p92.source(920123,64),(1.,.97,.25,.99))


def test_original_source_custody_when_full_repo_is_present():
    original=ROOT/'model.py'
    p86=ROOT/'experiments/p86_portable_causal_preflight.py'
    for path,sha in ((original,'5d9ea57b827aa1796548447868fa5ee14c251b49'),
                     (p86,'008492665ee9090db5791963f2e2336ea283ee62')):
        if path.exists():
            bits=path.read_bytes()
            blob=hashlib.sha1(b'blob '+str(len(bits)).encode()+b'\0'+bits).hexdigest()
            assert blob==sha


def test_full_frozen_negative_suppresses_any_promotion():
    out=p92.run()
    assert out['status']=='DEV_FAIL_QUALITY'
    assert not out['scientific_pass'] and not out['OOD_generated'] and not out['NEGATIVE_generated']
    assert not out['full_native_hybrid_trained']
    assert out['FIT_selection']['full_selected']['eta']==1.
    assert out['FIT_selection']['full_selected']['decay']==.99
    assert out['FIT_selection']['dual_selected']['params']==[1.,.97,.25,.99]
    for name in ('A','B'):
        score=out['VAL'][name];gates=out['performance_gates'][name]
        assert score['wrong_credit_pass'] and score['cold_restore_bitwise']
        assert not any(gates.values())
        assert score['dual']['combined']>score['fixed']['combined']
        assert out['costs_median_ms'][name]['ratio']>3
        assert not out['costs_median_ms'][name]['within_3x']
