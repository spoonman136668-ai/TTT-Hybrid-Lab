"""P49 source-free engineering and frozen negative-result controls."""
from pathlib import Path
import hashlib
import io
import numpy as np
import pytest
from scripts import native_p49_dual_null_calibration_dev as p49

EXPECTED = {
    780101:'6e634b1e940cf58751c51a8fd73ab924e61da6043ba26272f68866c7e69cec65',
    780103:'7379c3eed8af0ef1266d38c61761aed6745a2577abebf2606674d2c4e7d8aee2',
    780107:'baf5c889e352ad17390c19f8e79ac171116a790c300659944921e14cbd0261a6',
}

def test_original_p47_blob_still_pinned_and_all_sources_new():
    original=(Path(__file__).resolve().parents[1]/'scripts/native_p47_change_vs_nuisance_dev.py').read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(original)).encode()+b'\x00'+original).hexdigest()
    assert blob=='9be334264131db3bae79f19bed3271e060a1a903'
    assert set(p49.COHORTS)=={780001,780101,780103,780107}
    for consumed in (760111,770001,770101,999999):
        with pytest.raises(ValueError,match='P49_UNREGISTERED_SOURCE'):
            p49.fresh_source(consumed)


def test_new_source_custody_and_group_balance():
    for seed,sha in EXPECTED.items():
        w=p49.fresh_source(seed)
        assert w['sha']==sha
        assert w['a'].shape==w['y'].shape==(1536,8)
        assert [int((w['kind']==x).sum()) for x in (0,1,2)]==[768,384,384]
        assert np.isfinite(w['y']).all()
        assert np.array_equal(w['a'][:,1:],-w['a'][:,:-1])


def test_two_stable_null_thresholds_are_source_guarded():
    calibrator=p49.fresh_source(780001)
    t=p49.thresholds(calibrator)
    assert t['causal']==pytest.approx(17.908385472250835,abs=1e-9)
    assert t['nuisance']==pytest.approx(25.566222596495976,abs=1e-9)
    assert t['causal']>0 and t['nuisance']>0
    forged={**calibrator,'kind':calibrator['kind'].copy()}
    forged['kind'][0]=1
    with pytest.raises(ValueError,match='P49_CALIBRATION_MIXED_SOURCE_REFUSED'):
        p49.thresholds(forged)
    with pytest.raises(ValueError,match='P49_INVALID_CALIBRATION_CUSTODY'):
        p49.thresholds(p49.fresh_source(780101))


def test_read_only_candidate_and_cold_factual_replay():
    c=p49.thresholds(p49.fresh_source(780001))
    w=p49.fresh_source(780101)
    factual_a=w['a'].copy();factual_y=w['y'].copy()
    a=p49.predict(w['a'],w['y'],c,dual=True)
    b=p49.predict(w['a'],w['y'],c,dual=False)
    assert np.array_equal(factual_a,w['a']) and np.array_equal(factual_y,w['y'])
    assert set(a)==set(b)=={'candidate','model_kind','chosen_k','full','weighted','trailing5','last5','last3'}
    stream=io.BytesIO()
    np.savez_compressed(stream, a=w['a'],y=w['y']);stream.seek(0)
    with np.load(stream,allow_pickle=False) as inp:
        replay=p49.predict(inp['a'],inp['y'],c,dual=True)
    assert all(np.array_equal(a[key],replay[key]) for key in a)
    assert not np.array_equal(a['candidate'],b['candidate'])
    with pytest.raises(ValueError,match='P49_INVALID_THRESHOLDS'):
        p49.predict(w['a'],w['y'],{'causal':float('nan'),'nuisance':1.},dual=True)


def test_traded_off_stable_accuracy_and_exogenous_recall_preserve_negative():
    result=p49.study()
    assert not result['frozen_all_cohort_positive']
    assert [x['seed'] for x in result['reports']]==[780101,780103,780107]
    for x in result['reports']:
        assert x['cold_exact']
        assert x['p49_dual']['frozen_dev_pass']
        assert x['p49_dual']['detection']['causal_recall']==1.0
        assert x['p49_dual']['detection']['stable_false_alarm']<=.10
        assert x['p49_dual']['results']['candidate']['stable']['mae'] < x['p48_control']['results']['candidate']['stable']['mae']
        assert not x['exogenous_effect_noninferiority']
        assert x['p49_dual']['results']['candidate']['exogenous']['mae'] > x['p48_control']['results']['candidate']['exogenous']['mae']
