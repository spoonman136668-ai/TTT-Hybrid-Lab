"""P60-POL controls: real physical reward and memory-dependent search; DEV only."""
from pathlib import Path
import ast
import inspect
import json
import numpy as np
import pytest
from p60_reward_only_dev import (
  TRAIN_SEEDS,VALID_SEEDS,OOD_A,OOD_B,MODEL_SEEDS,GENERATIONS,CANDIDATES,
  ARMS,P58_MOMENTUM_TRACE,physical_episode,actual_action,check_original_p58_trace,
  fit_search,trial,git_blob,
)
ROOT=Path(__file__).resolve().parents[1]


def test_original_source_trace_reproduced_and_frozen_seeds_disjoint():
    assert check_original_p58_trace()==P58_MOMENTUM_TRACE
    assert set(TRAIN_SEEDS).isdisjoint(VALID_SEEDS)
    assert set(TRAIN_SEEDS).isdisjoint(OOD_A)
    assert set(TRAIN_SEEDS).isdisjoint(OOD_B)
    assert set(OOD_A).isdisjoint(OOD_B)
    assert all(seed>=989000 for seed in TRAIN_SEEDS+VALID_SEEDS+OOD_A+OOD_B)
    assert all(seed not in range(988000,988008) for seed in TRAIN_SEEDS+OOD_A+OOD_B)


def test_policy_requires_only_current_position_and_actual_completed_position_change():
    theta=np.array([1.,0.,0.])
    assert actual_action(theta,-.5,-.49,'history')==-1.
    assert actual_action(theta,-.5,-.51,'history')==1.
    assert actual_action(theta,-.5,None,'history')==1.
    assert actual_action(theta,-.5,-.49,'zero_history')==1.
    assert actual_action(theta,-.5,-.49,'memoryless')==1.
    with pytest.raises(ValueError,match='P60_UNKNOWN_SEARCH_POLICY'):
        actual_action(theta,-.5,None,'full_state_oracle')
    source=inspect.getsource(actual_action)
    assert 'velocity' not in source and 'published_step' not in source


def test_factually_executed_steps_and_exact_terminal_reward():
    for kind,theta in [('history',[1.,0.,0.]),('memoryless',[1.,0.,0.]),
                       ('position_delta_sign',None),('full_state_momentum',None),('random_actions',None)]:
        result=physical_episode(989100,kind,np.array(theta) if theta else None)
        assert 1<=result['steps']<=256
        assert result['physical_action_energy']==result['steps']
        assert result['net_return']==pytest.approx((100. if result['success'] else 0.)-.1*result['steps'],abs=1e-6)
        assert len(result['trace_sha256'])==64
        assert result==physical_episode(989100,kind,np.array(theta) if theta else None)


def test_reject_unknown_fitting_and_bad_physical_policy():
    with pytest.raises(ValueError,match='P60_FROZEN_FIT_SOURCE_REQUIRED'):
        fit_search(8888,'history_cem')
    with pytest.raises(ValueError,match='P60_FROZEN_FIT_SOURCE_REQUIRED'):
        fit_search(9741,'oracle')
    with pytest.raises(ValueError,match='P60_BAD_PHYSICAL_SEED'):
        physical_episode(989100.5,'random_actions')
    with pytest.raises(ValueError,match='P60_UNKNOWN_PHYSICAL_POLICY'):
        physical_episode(989100,'oracle')
    with pytest.raises(ValueError,match='P60_BAD_REAL_REWARD_CONTROLLER'):
        physical_episode(989100,'history',np.ones(4))


def test_each_fitted_method_uses_identical_actual_episode_budget_and_cold_actions():
    for arm in ARMS:
        fit=fit_search(MODEL_SEEDS[0],arm)
        assert fit['train_episode_count']==GENERATIONS*CANDIDATES*len(TRAIN_SEEDS)==960
        assert fit['physical_train_steps']>0
        assert len(fit['generation_history'])==GENERATIONS
        assert np.isfinite(fit['theta']).all()
        assert 0<fit['train_wall_seconds']
        assert fit['true_training_reward_at_selection']<=100.
        if arm=='wrong_credit_cem':
            assert fit['training_return_used_for_selection']<=100.


def test_full_frozen_result_preserves_strong_random_search_negative():
    receipt=ROOT/'experiments/p60_pol_reward_only_hidden_velocity_dev_receipt.json'
    if not receipt.exists():pytest.skip('Receipt checked in after source authoring')
    x=json.loads(receipt.read_text())
    assert x['classification']=='FROZEN_DEV_FAIL_ARCHITECTURE_OR_SEARCH_SUPERIORITY'
    assert x['frozen_overall']=='FAIL_0_OF_3'
    assert x['scientific_first_use']=='NOT_RUN'
    assert x['tested_source_sha256']
    for row in x['seeds']:
        for cohort in ('oodA','oodB'):
            v=row[cohort]
            assert v['history_cem']['success_count']==64
            assert v['history_random']['success_count']==64
            assert v['memoryless_cem']['success_count']==0
            assert v['history_zero']['success_count']==0
            assert v['position_delta_sign']['success_count']==64
            assert not row['frozen_gates'][cohort]['strict_beats_equal_budget_random_search_5pp']


def test_actual_source_does_not_consult_teacher_actions_or_future_lookahead():
    body=ast.parse(inspect.getsource(fit_search))
    called={q.func.id for q in ast.walk(body) if isinstance(q,ast.Call) and isinstance(q.func,ast.Name)}
    assert 'teacher_action' not in called
    assert 'published_step' not in called
    assert 'physical_episode' in called
    assert 'future_reward' not in inspect.getsource(physical_episode)
