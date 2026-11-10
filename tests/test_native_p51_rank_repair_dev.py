"""P51 local source-free DEV falsification and custody tests.

Does not consume any externally sealed or original research science sources.
"""
import hashlib
import inspect
import itertools
from pathlib import Path
import numpy as np
import pytest
from scripts import native_p51_rank_repair_dev as p51

EXPECTED = {
    800101: 'fd3ea4e69e4d309e78c83eab7f2b1613414ff6092f70be41e5be01389b43508b',
    800103: '906f3d412b0f9efa590f4d9bbc0a1b77f540cd98109e2bb0cecd6fe816e220b5',
    800107: '7ec6ac68b597e654c7d0afcdc595c6a2b9920cc328bdbaad2bc6ecd72d8efa8f',
}


def test_new_source_custody_and_no_prior_seeds():
    assert set(p51.SOURCE_SEEDS) == set(EXPECTED)
    for seed,sha in EXPECTED.items():
        w=p51.authored(seed)
        assert w['source_sha256']==sha
        assert w['initial_a'].shape==(1536,6)
        assert w['initial_y'].shape==(1536,6)
        assert np.isfinite(w['initial_y']).all()
        assert all(int((w['kind']==i).sum())==384 for i in range(4))
        assert all(int((w['policy']==i).sum())==384 for i in range(4))
    with pytest.raises(ValueError,match='P51_UNREGISTERED_DEVELOPMENT_SOURCE'):
        p51.authored(790101)


def test_all_6_and_7_action_history_choices_equal_Doptimal():
    for length in (6,7):
        histories=np.array(list(itertools.product([-1.,1.],repeat=length)))
        a=p51.score_actions(histories,length,'orthogonal')
        b=p51.score_actions(histories,length,'doptimal')
        assert np.array_equal(a,b)
        assert set(np.unique(a))<=set((-1.,1.))
    with pytest.raises(ValueError,match='P51_INVALID_PROSPECTIVE_ACTION_DESIGN'):
        p51.score_actions(np.ones((1,3)),3,'orthogonal')


def test_equal_real_6_step_history_and_exactly_2_new_interventions():
    world=p51.authored(800101)
    original_action=world['initial_a'].copy()
    original_delta=world['initial_y'].copy()
    for name in p51.ARMS:
        x=p51.executed_arm(world,name)
        assert x['a'].shape==x['y'].shape==(1536,8)
        np.testing.assert_array_equal(x['a'][:,:6],original_action)
        np.testing.assert_array_equal(x['y'][:,:6],original_delta)
        assert np.all(np.isin(x['a'],(-1.,1.)))
        for t in (6,7):
            # This is the realized outcome for the executed action, not an oracle
            # observation of the action not chosen.
            np.testing.assert_array_equal(x['y'][:,t],world['base'][:,t] + world['effect'][:,t]*x['a'][:,t])
    np.testing.assert_array_equal(world['initial_a'],original_action)
    np.testing.assert_array_equal(world['initial_y'],original_delta)


def test_rank_deficient_constant_actions_abstain_not_invent_effect():
    a=np.ones((4,8));y=np.ones((4,8))*.3
    estimate,ident=p51.estimate(a,y,0,'full')
    assert not ident.any() and np.isnan(estimate).all()
    recent,rv=p51.estimate(a,y,5,'recent')
    assert not rv.any() and np.isnan(recent).all()


def test_probe_receives_no_future_consequences_or_latent_labels():
    code=inspect.getsource(p51.score_actions)
    assert 'source' not in code and 'effect' not in code and 'beta' not in code
    assert 'future_outcome' not in code
    action=np.array([[1.,1.,1.,1.,1.,1.]])
    before=action.copy()
    p51.score_actions(action,6,'orthogonal')
    np.testing.assert_array_equal(before,action)


def test_frozen_dev_negative_and_cold_replay_three_cohorts():
    result=p51.run()
    assert not result['all_three_positive']
    assert [r['seed'] for r in result['rows']]==list(EXPECTED)
    for row in result['rows']:
        assert row['world_sha256']==EXPECTED[row['seed']]
        assert all(row['cold_exact'].values())
        assert not row['checks']['causal_MAE_better_strong_10pct']
        assert row['checks']['constant_full_rank_95pct']
        assert row['checks']['constant_coverage_gain_vs_random_10pp']
        assert row['checks']['sham_effect_MAE_under_point06']
        assert row['arms']['orthogonal']['causal_switch_penalized_MAE'] == row['arms']['doptimal']['causal_switch_penalized_MAE']
        assert row['arms']['flip']['causal_switch_penalized_MAE'] < row['arms']['orthogonal']['causal_switch_penalized_MAE']
        assert row['arms']['passive']['policy']['constant']['full_valid']==0.0
