"""P59 source-bound frozen readout and negative-claim custody tests.

Local isolated harness mirrors P54 published equations, independently verified
against the exact original 96-episode P58 momentum trace. Repository tests use
its unchanged actual P54 physics source, and full platform CI remains required.
"""
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments import p59_goal_policy_readout_dev as p59
from native_p54_mountaincar_factual_bridge import published_reset,published_step

ROOT=Path(__file__).resolve().parents[1]
EXPECTED={
    'train_executed_trajectories':'fa101629292fafb9013d63f4c3626f5cde184039e86db41b17dcd3b77ff61c8f',
    'validation_executed_trajectories':'f12d268371eca7c5f4ce05dc46422f58a16e9cf3dd9ad04febc24b06a041b214',
    'OOD_executed_trajectories':'99ea2fb450fe2121a7f198dd936ea3a581f0af836575b747b9a7d78be6c35151'}


def test_exact_original_hybrid_blob_and_source_custody():
    original=(ROOT/'model.py').read_bytes()
    assert p59.git_blob(original)==p59.MODEL_GIT_BLOB=='5d9ea57b827aa1796548447868fa5ee14c251b49'
    physical=(ROOT/'native_p54_mountaincar_factual_bridge.py').read_bytes()
    # The published-equation-only local harness is not the exact original file.
    # In an original checkout, the P54 source MUST be the specified exact Git blob.
    if b'Local source-equation mirror' not in physical:
        assert p59.git_blob(physical)==p59.PHYSICS_GIT_BLOB
    else:
        assert b'NOT' not in physical[:10]


def test_mirror_matches_original_p58_complete_episode_trace():
    hashes=[]
    for seed in range(985100,985196):
        s=published_reset(seed);h=hashlib.sha256();h.update(s.tobytes())
        for t in range(256):
            a=p59.teacher_action(s)
            d=published_step(s,a);s=d.next_sensor
            h.update(np.asarray([a],np.float32).tobytes())
            h.update(s.tobytes())
            h.update(np.asarray([d.reward],np.float64).tobytes())
            h.update(bytes([int(d.terminated)]))
            if d.terminated:break
        hashes.append(h.hexdigest())
    assert hashlib.sha256(''.join(hashes).encode()).hexdigest()=='1686baec1451b442b818a37f35843f75a3f0c8c42ae4a677c3b4f13d07652853'


def test_disjoint_seed_families_and_source_sealed_receipt():
    assert len(set(p59.TRAIN_SEEDS+p59.VALID_SEEDS+p59.OOD_SEEDS))==256
    assert set(range(985100,985196)).isdisjoint(p59.TRAIN_SEEDS+p59.VALID_SEEDS+p59.OOD_SEEDS)
    data=json.loads((ROOT/'experiments/p59_goal_policy_readout_dev_receipt.json').read_text())
    assert data['schema']=='ttt-hybrid.p59.goal-reaching-factual-imitation-readout-dev-receipt.v1'
    # Read the original frozen receipt's actual nested fields; never rewrite it.
    assert data['authored_experiment']['source_SHA256']=={
        'train':EXPECTED['train_executed_trajectories'],
        'validation':EXPECTED['validation_executed_trajectories'],
        'OOD':EXPECTED['OOD_executed_trajectories']}
    assert (data['authored_experiment']['fit_executed_episodes'],
            data['authored_experiment']['fit_executed_actions'],
            data['authored_experiment']['validation_episodes'],
            data['authored_experiment']['OOD_episodes'])==(128,10195,32,96)
    assert [row['seed'] for row in data['per_seed']]==[9511,9523,9531]
    assert data['parent']['original_reference_model_git_blob']==p59.MODEL_GIT_BLOB
    assert data['parent']['local_physics_git_blob']==p59.PHYSICS_GIT_BLOB


def test_exact_param_budget_and_reference_unchanged():
    assert p59.parameter_counts()=={'Hybrid':1282,'GRU':1242,'MLP':1249}
    assert 'HybridPilot' in inspect.getsource(p59.HybridPolicy)
    assert p59.TRAIN_UPDATES==400 and p59.BATCH==96


def test_query_contains_no_future_and_observation_only_after_completed_step():
    sensor=published_reset(986000)
    observed=published_step(sensor,-1.)
    one=p59.row(sensor,observed.next_sensor,-1.)
    q=p59.feature([one],observed.next_sensor)
    assert q.shape==(5,5)
    assert np.array_equal(q[:3],np.zeros((3,5),np.float32))
    assert np.array_equal(q[-2],one)
    assert q[-1,2:].tolist()==[0.,0.,0.]
    # The candidate stores float32; compare against the same explicit cast.
    assert np.array_equal(q[-1,:2],np.array(
        [observed.next_sensor[0],observed.next_sensor[1]*100.],dtype=np.float32))
    with pytest.raises(ValueError,match='P59_FACTUAL_CAPACITY_OVERFLOW'):
        p59.feature([one]*5,sensor)
    src=inspect.getsource(p59.batch_rollout)
    assert 'teacher_action' not in src and 'source_oracle' not in src


def test_all_models_execute_legal_actual_actions_and_reward_accounting():
    torch.set_num_threads(1)
    seeds=[986400,986401,986402]
    for name,cls in p59.FACTORIES.items():
        model=cls()
        x=p59.batch_rollout(model,seeds,-.75,-.25)
        assert x['worlds']==len(seeds)
        assert 0<=x['successes']<=len(seeds)
        assert 3<=x['total_actual_steps']<=3*256
        assert x['mean_return']==pytest.approx(100*x['success']-.1*x['mean_steps'],abs=1e-8)
        assert len(x['trace_sha256'])==64


def test_cold_weights_unchanged_for_all_factory_classes():
    for cls in p59.FACTORIES.values():
        m=cls()
        cold,sha=p59.cold_restore(m,np.zeros((10,5,5),np.float32))
        assert cold and len(sha)==64


def test_frozen_outcomes_preserve_strong_GRU_and_imitation_only():
    data=json.loads((ROOT/'experiments/p59_goal_policy_readout_dev_receipt.json').read_text())
    # Exact outcome class and strong conventional-control negatives are frozen.
    assert data['classification']=='FROZEN_AUTHOR_GENERATED_IMITATION_DEV_PASS_NOT_CAUSAL_ACTION_CONSEQUENCE_SCIENCE'
    gates=data['frozen_gates']
    assert gates['overall']=='DEV_PASS_3_OF_3'
    assert gates['OOD_success_noninferior_vs_trained_GRU_and_MLP_5ppt']=='PASS_3_OF_3'
    assert gates['model_weights_cold_restore_exact']=='PASS_ALL'
    assert data['source_fidelity']['installed_official_gymnasium_runtime']=='NOT_AVAILABLE'
    limits=data['scientific_limits']
    assert limits['SCIENTIFIC_PASS']=='UNVERIFIED'
    assert limits['causal_action_consequence_learning'].startswith('NOT_TESTED')
    assert limits['strong_GRU_control']=='288/288 success; Hybrid has no demonstrated gain'
    assert data['teacher_controls']['unfamiliar_OOD']['successes']==96
    rows=data['per_seed']
    assert [row['seed'] for row in rows]==[9511,9523,9531]
    assert [row['hybrid']['OOD_goals'] for row in rows]==[96,96,95]
    assert all(row['GRU']['OOD_goals']==96 for row in rows)
    assert all(all(row[arm]['cold'] for arm in ('hybrid','GRU','MLP')) for row in rows)
    assert all(all(row[arm]['train_s']>0 for arm in ('hybrid','GRU','MLP')) for row in rows)
