"""P38 synthetic-only focused native conditional-action DEV regression tests.

No P38 sealed 238000/238500/239000-source cohort or physical science used.
"""
import math
import torch
import pytest

import native_p38_conditional_action_choice_dev as p38
from native_intervention_choice import preview_interventions


def test_authored_actual_source_action_effect_and_goal_both_reverse_by_cue():
    for position in (-.4,0.,.4):
        for cue in (-.8,.8):
            outcomes=[p38.AuthoredConditionalWorld(position,cue).step(a)[2][0]
                      for a in (-1.,1.)]
            assert (outcomes[1]-outcomes[0])*cue>0
            positive_target=position+.32
            negative_target=position-.32
            best_pos=min(range(2),key=lambda i:abs(outcomes[i]-positive_target))
            best_neg=min(range(2),key=lambda i:abs(outcomes[i]-negative_target))
            assert best_pos != best_neg
        with pytest.raises(ValueError,match='ACTION_NOT_PRECOMMITTED'):
            p38.AuthoredConditionalWorld(position,.8).step(0.)


def test_true_native_and_conventional_model_parameter_counts_and_shapes():
    native=p38.native()
    conventional=p38.ConventionalMLP()
    assert sum(p.numel() for p in native.parameters())==734
    assert sum(p.numel() for p in conventional.parameters())==746
    sensors=torch.randn(4,2);actions=torch.ones(4,1)
    assert native.step(sensors,actions,torch.ones(4,1))[0].shape==(4,2)
    assert conventional(sensors,actions).shape==(4,2)
    branches=preview_interventions(native,sensors,
            torch.tensor([-1.,1.]).reshape(1,2,1).expand(4,2,1),torch.ones(4,1))
    assert branches.shape==(4,2,2)
    assert not torch.equal(branches[:,0,:],branches[:,1,:])


def test_factual_sham_train_identical_initial_native_and_source_sample_budget():
    sensors=torch.tensor([[.2,-.7],[-.2,.7],[.3,.8],[-.3,-.8]]*8,dtype=torch.float32)
    actions=torch.tensor([[-1.],[1.],[1.],[-1.]]*8,dtype=torch.float32)
    targets=sensors.clone();targets[:,0]=sensors[:,0]+.3*actions[:,0]*sensors[:,1]
    torch.manual_seed(p38.SEEDS[0])
    original_sha={k:v.detach().clone() for k,v in p38.native().state_dict().items()}
    for arm in p38.ARMS[:2]:
        torch.manual_seed(p38.SEEDS[0])
        initial={k:v.detach().clone() for k,v in p38.native().state_dict().items()}
        assert all(torch.equal(original_sha[k],v) for k,v in initial.items())
        model,receipt=p38.train_one(arm,p38.SEEDS[0],sensors,actions,targets,epochs=1)
        assert receipt['optimizer_minibatches']==1
        assert receipt['real_source_train_examples']==len(sensors)
        assert receipt['trainable_parameters']==734
        assert all(torch.isfinite(p).all() for p in model.parameters())
        if arm==p38.ARMS[1]:
            assert receipt['wrong_action_aux_samples']==len(sensors)
            assert 0.15<receipt['wrong_action_credit_fraction']<=1.


def test_sham_action_permutation_does_not_change_factual_minibatch_order(monkeypatch):
    sensors=torch.tensor([[.2,-.7],[-.2,.7],[.3,.8],[-.3,-.8]]*64,dtype=torch.float32)
    actions=torch.tensor([[-1.],[1.],[1.],[-1.]]*64,dtype=torch.float32)
    targets=sensors.clone();targets[:,0]+=actions[:,0]*sensors[:,1]*.2
    original_randperm=torch.randperm
    recorded=[]
    def spy_randperm(n,*args,**kw):
        output=original_randperm(n,*args,**kw)
        recorded.append((n,tuple(output.tolist())))
        return output
    monkeypatch.setattr(torch,"randperm",spy_randperm)
    p38.train_one(p38.ARMS[0],p38.SEEDS[0],sensors,actions,targets,epochs=2)
    original_batches=[x for x in recorded if x[0]==256]
    recorded.clear()
    p38.train_one(p38.ARMS[1],p38.SEEDS[0],sensors,actions,targets,epochs=2)
    sham_batches=[x for x in recorded if x[0]==256]
    assert original_batches==sham_batches
    assert len(original_batches)==2


def test_precommitted_science_gate_rejects_equal_failed_policies(monkeypatch):
    # Use the immutable frozen-science thresholds, not authored source examples.
    criteria={'every_native_seed_goal_action_optimal_fraction_at_least':.9,
      'native_minus_sham_goal_action_optimal_fraction_at_least':.3,
      'native_minus_best_fixed_action_optimal_fraction_at_least':.3,
      'native_no_more_than_below_mlp_optimal_fraction':.05,
      'native_no_more_than_below_ridge_optimal_fraction':.05}
    monkeypatch.setattr(p38,'frozen',lambda:{'science_dev_success':criteria})
    values={}
    for goal in ('positive_goal','negative_goal'):
        for arm in (*p38.ARMS,'fully_trained_conventional_ridge','fixed_action_minus_one','fixed_action_plus_one'):
            values[arm+':'+goal]={'fraction_optimal':.5,
                'positive_cue_optimal':.5,'negative_cue_optimal':.5}
    results=p38.gates(values)
    assert not any(all(v.values()) for v in results.values())
    for goal in ('positive_goal','negative_goal'):
        values[p38.ARMS[0]+':'+goal].update(
             fraction_optimal=1.,positive_cue_optimal=1.,negative_cue_optimal=1.)
    assert all(all(v.values()) for v in p38.gates(values).values())


def test_authored_factual_source_rejects_invalid_inputs_without_execution():
    with pytest.raises(ValueError,match='INVALID_SOURCE_INITIAL_SENSOR'):
        p38.AuthoredConditionalWorld(0.,0.)
    with pytest.raises(ValueError,match='INVALID_SOURCE_INITIAL_SENSOR'):
        p38.AuthoredConditionalWorld(float('nan'),.6)
    with pytest.raises(ValueError,match='UNREGISTERED_TRAINING_ARM'):
        p38.train_one('unknown',p38.SEEDS[0],torch.zeros(1,2),
                      torch.zeros(1,1),torch.zeros(1,2),epochs=0)
