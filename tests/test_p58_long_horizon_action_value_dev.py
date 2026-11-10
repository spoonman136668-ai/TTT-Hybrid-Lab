"""P58 source-free controls and deterministic single-episode replay; no model training."""
from __future__ import annotations

import ast
import inspect
from pathlib import Path
import numpy as np
import pytest

from experiments import p58_long_horizon_action_value_dev as p58
from native_p54_mountaincar_factual_bridge import published_reset, published_step


def test_exact_existing_reference_git_source_and_registrations():
    p58.verify_source()
    assert p58.FIRST_SEED == 985100 and p58.LAST_SEED == 985195
    assert p58.HORIZON == 256
    assert p58.POLICIES == ('neutral','random3','always_right','momentum_bangbang')


def test_policy_decisions_depend_only_on_actual_current_state():
    rng=np.random.default_rng(99)
    assert p58.action_for('neutral',np.array([-.5,0.],np.float32),None)==0.
    assert p58.action_for('always_right',np.array([-.5,0.],np.float32),None)==1.
    assert p58.action_for('momentum_bangbang',np.array([-.5,0.],np.float32),None)==-1.
    assert p58.action_for('momentum_bangbang',np.array([-.5,.003],np.float32),None)==1.
    assert p58.action_for('momentum_bangbang',np.array([-.5,-.003],np.float32),None)==-1.
    assert p58.action_for('random3',np.array([-.5,.003],np.float32),rng) in (-1.,0.,1.)
    with pytest.raises(ValueError,match='P58_RANDOM_SOURCE_REQUIRED'):
        p58.action_for('random3',np.array([-.5,0.],np.float32),None)


def test_unregistered_policy_and_seeds_fail_closed():
    with pytest.raises(ValueError,match='P58_UNREGISTERED_POLICY'):
        p58.episode(985100,'oracle')
    for seed in (985099,985196,985100.0):
        with pytest.raises(ValueError,match='P58_UNREGISTERED_SEED'):
            p58.episode(seed,'neutral')


def test_single_factual_episode_has_only_real_physical_transitions():
    state=published_reset(985100)
    original=state.copy()
    d=p58.episode(985100,'neutral')
    assert len(d['first_actions'])==8
    assert d['steps']==256
    assert not d['success']
    assert d['energy']==0 and d['cumulative_reward']==0
    assert np.array_equal(state,original)
    assert len(d['trace_sha256'])==64


def test_bangbang_factual_controller_goal_reward_and_natural_termination():
    d=p58.episode(985100,'momentum_bangbang')
    assert d['success']
    assert 1 <= d['steps'] <= 256
    assert d['cumulative_reward']==pytest.approx(100.-0.1*d['energy'],abs=1e-8)
    assert d['energy']==d['steps']
    assert d['final_position']>=0.45


def test_episode_randomness_replays_from_independent_fixed_seed():
    x=p58.episode(985100,'random3')
    y=p58.episode(985100,'random3')
    assert x==y
    assert x['steps']==256
    assert x['energy']>0


def test_no_future_source_or_lookahead_in_actual_policy():
    tree=ast.parse(inspect.getsource(p58.action_for))
    called={node.func.id for node in ast.walk(tree)
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)}
    assert 'published_step' not in called
    assert not {'true_beta','oracle','future_state','counterfactual'} & set(dir(p58))


def test_physical_reference_is_disclosed_as_source_transcription_not_native_gymnasium():
    src=Path(inspect.getsourcefile(p58)).read_text()
    assert 'not an installed official Gymnasium' in src
    source=published_step(published_reset(985101),0.0)
    assert source.reward==0.0
    assert not source.terminated
