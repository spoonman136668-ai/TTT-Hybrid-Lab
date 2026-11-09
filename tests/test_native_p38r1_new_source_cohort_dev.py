"""No-source DEV tests for P38-R1 new frozen cohort, replay and strict gates."""
import json, torch, pytest
from pathlib import Path
import native_p38_conditional_action_choice_dev as base
import native_p38r1_new_source_cohort_dev as x

def test_new_frozen_cohort_disjoint_source_and_model_initialization(monkeypatch):
    monkeypatch.setenv('NATIVE_P38R1_AUTHORED_DEV_ONLY','1')
    p=x.freeze()
    assert p['cohort']['fresh_model_seeds']==list(x.FRESH)
    assert set(x.FRESH).isdisjoint(set(base.SEEDS))
    assert {x.TRAIN_SEED,x.VAL_SEED,x.HELDOUT_SEED}.isdisjoint({base.TRAIN_SEED,base.VALIDATION_SEED,base.HELDOUT_SEED})
    assert p['parent']['old_original_source_first_use_consumed'] is True
    assert x.original_gitblob(x.ROOT/'native_p38_conditional_action_choice_dev.py')==x.ORIGINAL_BLOB
    assert p['pass']['all_three_model_seeds'] is True
    monkeypatch.delenv('NATIVE_P38R1_AUTHORED_DEV_ONLY')
    with pytest.raises(RuntimeError,match='ORIGINAL_SCIENCE_FIRST_USE_NOT_AUTHORIZED'):
        x.freeze()

def test_scoped_new_seed_fit_restores_prior_p38_frozen_seed_set():
    sensors=torch.tensor([[.2,.7],[-.2,-.7]]*512,dtype=torch.float32)
    actions=torch.tensor([[-1.],[1.]]*512)
    actual=sensors.clone();actual[:,0]+=actions[:,0]*sensors[:,1]*.25
    before=base.SEEDS
    for arm in base.ARMS:
        model,receipt=x.fit_neural(arm,x.FRESH[0],sensors,actions,actual)
        assert base.SEEDS==before
        assert receipt['optimizer_minibatches']==256
        assert receipt['real_source_train_examples']==len(sensors)
        assert all(torch.isfinite(v).all() for v in model.parameters())
    assert base.SEEDS==before

def test_sham_and_conventional_science_thresholds_are_not_trivially_passed():
    frozen={'pass':{'native_optimal_min_every_goal_every_cue':.9,
        'improvement_over_sham_min':.3,'improvement_over_best_fixed_min':.3,
        'within_fully_trained_MLP':.05,'within_fully_fitted_ridge':.05}}
    scores={}
    for goal in ('positive_goal','negative_goal'):
        for arm in (*base.ARMS,'fully_trained_conventional_ridge','fixed_action_minus_one','fixed_action_plus_one'):
            scores[arm+':'+goal]={'fraction_optimal':.5,
                'positive_cue_optimal':.5,'negative_cue_optimal':.5}
    assert not all(all(v.values()) for v in x.metrics_gate(scores,frozen).values())
    for goal in ('positive_goal','negative_goal'):
        scores[base.ARMS[0]+':'+goal].update(
            fraction_optimal=1.,positive_cue_optimal=1.,negative_cue_optimal=1.)
        scores[base.ARMS[2]+':'+goal]['fraction_optimal']=1.
        scores['fully_trained_conventional_ridge:'+goal]['fraction_optimal']=1.
    assert all(all(v.values()) for v in x.metrics_gate(scores,frozen).values())
    scores[base.ARMS[1]+':positive_goal']['fraction_optimal']=.8
    assert not x.metrics_gate(scores,frozen)['positive_goal']['higher_than_wrong_action_credit']

def test_wrong_state_or_frozen_source_output_blocked(monkeypatch,tmp_path):
    monkeypatch.setenv('NATIVE_P38R1_AUTHORED_DEV_ONLY','1')
    f=tmp_path/'receipt.json'
    with pytest.raises(FileNotFoundError):
        x.cold(f)
    with pytest.raises(RuntimeError,match='SOURCE_CUSTODY_INCOMPLETE'):
        f.write_text(json.dumps({'schema':x.SCHEMA,'seeds':[], 'real_authored_train_source_steps':1024}))
        x.cold(f)
