"""P47 source-free, frozen 3-cohort development/negative controls."""
import hashlib
import io
import numpy as np
import pytest
from scripts import native_p47_change_vs_nuisance_dev as p

EXPECTED = {
    760111: 'ea5e1470fa2a86677c951f8eeefc41c8b393bd3f6ab0eaa065f76ef9b39b747f',
    760117: 'a6bd914640d984d3e8cde0db4127e37410bc9924bf710e96c1faf7a75da418ef',
    760123: '6f0e58e8396338a640bb7fae2b4861724b3a918709d753afd02b86a1d75033aa',
}

def test_frozen_source_sha_and_factual_action_custody():
    assert len(p.COHORTS) == 3
    for seed,n,band in p.COHORTS:
        d=p.source(seed,n,band)
        assert d['sha'] == EXPECTED[seed]
        assert d['a'].shape == d['y'].shape == (1536,8)
        assert [int(np.sum(d['kind']==k)) for k in (0,1,2)] == [768,384,384]
        assert np.isfinite(d['y']).all()
        assert np.all(np.abs(d['a'])==1)
        assert np.array_equal(d['a'][:,1:],-d['a'][:,:-1])
    with pytest.raises(ValueError,match='P47_UNFROZEN_SOURCE'):
        p.source(750001,1536,(.35,.9))

def test_full_rank_observational_models_and_rank_deficient_sequence_rejection():
    for sign in (-1.,1.):
        for mode in ('','causal','nuisance'):
            for split in ((None,) if not mode else p.SPLITS):
                x=p.design(sign,mode,split)
                assert np.linalg.matrix_rank(x,tol=1e-11)==x.shape[1]
    d=p.source(*p.COHORTS[0]); bad=d['a'][:1].copy()
    bad[:]=1
    with pytest.raises(ValueError,match='P47_BAD_ACTION_SEQUENCE'):
        p.estimate(bad,d['y'][:1])
    with pytest.raises(ValueError,match='P47_BAD_FACTUAL_HISTORY'):
        p.estimate(d['a'][:1],np.full((1,8),np.nan))

def test_known_factual_causal_flip_vs_exogenous_jump():
    a=np.array([[1,-1,1,-1,1,-1,1,-1]],dtype=float)
    x=p.T
    stable=.1+.2*x+.05*x*x+.35*a[0]
    flip=.1+.2*x+.05*x*x+.35*a[0]*np.where(np.arange(8)>=4,-1.,1.)
    nuisance=stable+.6*(np.arange(8)>=4)
    s=p.estimate(np.repeat(a,3,axis=0),np.stack((stable,flip,nuisance)))
    assert s['model_kind'].tolist()==[-1,0,1]
    assert abs(s['candidate'][0]-.35)<1e-7
    assert abs(s['candidate'][1]+.35)<1e-7
    assert abs(s['candidate'][2]-.35)<1e-7

def test_cold_replay_never_uses_evaluator_labels():
    d=p.source(*p.COHORTS[1]); one=p.estimate(d['a'],d['y'])
    f=io.BytesIO();np.savez_compressed(f,a=d['a'],y=d['y']);f.seek(0)
    with np.load(f) as data: replay=p.estimate(data['a'],data['y'])
    for key in one:
        assert np.array_equal(one[key],replay[key])
    assert one.keys()=={'candidate','model_kind','chosen_k','full','weighted','trailing5','last5','last3'}

def test_consumer_cannot_forge_missing_actions_or_source_shape():
    d=p.source(*p.COHORTS[2])
    with pytest.raises(ValueError,match='P47_BAD_FACTUAL_HISTORY'):
        p.estimate(d['a'][:2,:5],d['y'][:2,:5])
    with pytest.raises(ValueError,match='P47_BAD_FACTUAL_HISTORY'):
        p.estimate(d['a'][:2],d['y'][:1])

def test_preserve_frozen_false_alarm_failure_without_result_tuning():
    out=p.run()
    assert not out['all_cohort_gates_pass']
    assert [row['seed'] for row in out['rows']]==[760111,760117,760123]
    assert all(row['cold_exact'] for row in out['rows'])
    assert all(row['detection']['causal_recall']==1.0 for row in out['rows'])
    assert all(row['detection']['exogenous_false_alarm']==0.0 for row in out['rows'])
    assert all(row['detection']['stable_false_alarm']>0.10 for row in out['rows'])
    assert all(not row['gates']['stable_MAE_110pct_full'] for row in out['rows'])
