"""P48 source-free engineering checks; DO NOT consume protected P48 cohorts here."""
import hashlib
from pathlib import Path
import numpy as np
import pytest

from scripts import native_p47_change_vs_nuisance_dev as p47
from scripts.native_p48_calibrated_causal_switch_dev import (
    COHORTS, calibrated_threshold, fitted_hypotheses, predict, fresh_source)


def original_blob(path):
    b=Path(path).read_bytes()
    return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()


def manual_rows():
    a=np.broadcast_to(np.power(-1.,np.arange(8)),(3,8)).copy()
    t=np.arange(8)/8
    y=np.stack([.10+.05*t+.3*a[0],-.15+.4*t-.3*a[1],
                .10+.1*t+.2*a[2]+.5*(np.arange(8)>=4)],axis=0)
    return a,y


def test_exact_inherited_p47_original_source_immutable():
    root=Path(__file__).resolve().parents[1]
    assert original_blob(root/'scripts/native_p47_change_vs_nuisance_dev.py')=='9be334264131db3bae79f19bed3271e060a1a903'


def test_null_margin_and_choice_preserve_original_factual_design():
    a,y=manual_rows()
    fit=fitted_hypotheses(a,y)
    assert fit['margin'].shape==(3,)
    assert np.isfinite(fit['margin']).all()
    for k,v in fit.items():
        assert v.shape==(3,)
    original=p47.estimate(a,y)
    proposal=predict(a,y,1e12)
    assert not np.any(proposal['model_kind']==0)
    np.testing.assert_array_equal(proposal['full'],original['full'])
    np.testing.assert_array_equal(proposal['weighted'],original['weighted'])
    np.testing.assert_array_equal(proposal['trailing5'],original['trailing5'])


def test_negative_action_sequence_and_shapes_fail_original_validator():
    a,y=manual_rows()
    with pytest.raises(ValueError,match='P47_BAD_ACTION_SEQUENCE'):
        fitted_hypotheses(np.ones_like(a),y)
    with pytest.raises(ValueError,match='P47_BAD_FACTUAL_HISTORY'):
        predict(a[:,:4],y[:,:4],1.)
    with pytest.raises(ValueError,match='P48_INVALID_FROZEN_THRESHOLD'):
        predict(a,y,float('nan'))


def test_stable_only_calibration_custody_guards():
    a,y=manual_rows()
    forged=dict(seed=770001,role='calibration',a=a,y=y,kind=np.array([2,0,2]))
    with pytest.raises(ValueError,match='P48_CALIBRATION_CONTAINS_NONSTABLE_WORLD'):
        calibrated_threshold(forged)
    forged['kind'][:]=2
    forged['role']='evaluation'
    with pytest.raises(ValueError,match='P48_CALIBRATION_SOURCE_CUSTODY_REFUSED'):
        calibrated_threshold(forged)


def test_new_source_protocol_rejects_consumed_p47_and_unregistered_seeds():
    assert set(COHORTS)=={770001,770101,770103,770107}
    for seed in (760111,760117,760123,999999):
        with pytest.raises(ValueError,match='P48_UNREGISTERED_SOURCE_FIRST_USE_REFUSED'):
            fresh_source(seed)


def test_no_unobserved_outcomes_or_hidden_labels_passed_to_estimator():
    a,y=manual_rows()
    original_a=a.copy();original_y=y.copy()
    before=fitted_hypotheses(a,y)
    after=predict(a,y,1000.)
    assert np.array_equal(a,original_a) and np.array_equal(y,original_y)
    assert before['margin'].shape==(len(y),)
    assert not np.any(after['model_kind']==0)


def test_repeated_factual_prediction_is_bit_exact_and_calibration_is_positive():
    a,y=manual_rows()
    first=predict(a,y,.5)
    second=predict(a,y,.5)
    assert all(np.array_equal(first[k],second[k]) for k in first)
    # Margin is a model-comparison score, not a source-hidden causal truth.
    assert np.isfinite(fitted_hypotheses(a,y)['margin']).all()
