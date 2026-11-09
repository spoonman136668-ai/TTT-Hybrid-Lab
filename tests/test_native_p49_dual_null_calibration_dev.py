"""P49 no-new-source focused engineering controls, exact pinned source custody."""
import hashlib
from pathlib import Path
import numpy as np
import pytest
from scripts import native_p47_change_vs_nuisance_dev as p47
from scripts import native_p48_calibrated_causal_switch_dev as p48
from scripts import native_p49_dual_null_calibration_dev as p49


def git_blob(path):
    b=Path(path).read_bytes()
    return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()


def toy():
    a=np.array([[1.,-1.,1.,-1.,1.,-1.,1.,-1.],
                [-1.,1.,-1.,1.,-1.,1.,-1.,1.]])
    t=p47.T
    y=np.array([.1+.2*t+.3*a[0],-.2+.15*t-.3*a[1]])
    return a,y


def test_p47_and_p48_original_exact_source_unmodified():
    root=Path(__file__).resolve().parents[1]
    assert git_blob(root/'scripts/native_p47_change_vs_nuisance_dev.py')=='9be334264131db3bae79f19bed3271e060a1a903'
    assert git_blob(root/'scripts/native_p48_calibrated_causal_switch_dev.py')=='2509ad213e2ebf102d3d78aecd19cf6d67e9654a'


def test_no_accepted_nuisance_branch_matches_original_full_window():
    a,y=toy()
    very_high=(1e9,1e9)
    out=p49.predict(a,y,very_high)
    original=p47.estimate(a,y)
    assert (out['model_kind']==-1).all()
    np.testing.assert_array_equal(out['candidate'],original['full'])
    np.testing.assert_array_equal(out['full'],original['full'])
    np.testing.assert_array_equal(out['weighted'],original['weighted'])
    np.testing.assert_array_equal(out['last5'],original['last5'])


def test_p48_causal_selection_unchanged_by_second_null_threshold():
    a,y=toy()
    # Deterministic artificial switch in the physically authored sequence.
    y[0]+=.8*a[0]*(np.arange(8)>=4)
    parent=p48.predict(a,y,0.)
    child=p49.predict(a,y,(0.,1e9))
    mask=parent['model_kind']==0
    assert mask.any()
    np.testing.assert_array_equal(child['candidate'][mask],parent['candidate'][mask])
    np.testing.assert_array_equal(child['chosen_k'][mask],parent['chosen_k'][mask])


def test_nuisance_fit_selection_can_be_abstained_on_stable_history():
    a,y=toy()
    data=p49.null_evidence(a,y)
    assert set(data)=={'margin','full_effect','nuisance_effect','nuisance_k'}
    assert all(v.shape==(2,) for v in data.values())
    assert np.isfinite(data['margin']).all()
    out=p49.predict(a,y,(1e9,1e9))
    assert (out['model_kind']==-1).all()


def test_bad_data_and_source_seeds_fail_without_fallback():
    a,y=toy()
    for seed in (760111,770001,770101,780099,0):
        with pytest.raises(ValueError,match='P49_UNREGISTERED_SOURCE'):
            p49.source(seed)
    with pytest.raises(ValueError,match='P47_BAD_ACTION_SEQUENCE'):
        p49.predict(np.ones_like(a),y,(1.,1.))
    with pytest.raises(ValueError,match='P49_INVALID_FROZEN_THRESHOLDS'):
        p49.predict(a,y,(float('nan'),1.))
    with pytest.raises(ValueError,match='P47_BAD_FACTUAL_HISTORY'):
        p49.predict(a,y[:,:4],(1.,1.))


def test_calibration_only_stable_and_exact_seed_required():
    a,y=toy()
    bad=dict(seed=780001,role='calibration',a=a,y=y,kind=np.array([2,0]))
    with pytest.raises(ValueError,match='P49_NONSTABLE_CALIBRATION'):
        p49.calibrate(bad)
    bad['kind']=np.array([2,2]);bad['seed']=780103
    with pytest.raises(ValueError,match='P49_CALIBRATION_CUSTODY'):
        p49.calibrate(bad)


def test_prediction_determinism_no_writes_to_factual_histories():
    a,y=toy();a0=a.copy();y0=y.copy()
    v=p49.predict(a,y,(1.,2.));w=p49.predict(a,y,(1.,2.))
    assert all(np.array_equal(v[k],w[k]) for k in v)
    assert np.array_equal(a,a0) and np.array_equal(y,y0)
