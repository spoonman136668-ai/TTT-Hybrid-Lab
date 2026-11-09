"""P37 native interface tests. Synthetic DEV fixtures only; no science first-use."""
import ast
import io
from pathlib import Path
import pytest
import torch

from native_continuous_state import NativeContinuousHybrid, ContinuousState
from native_intervention_choice import (
    MAX_CANDIDATES, preview_interventions, select_goal_action, commit_factual_outcome)


def case():
    torch.manual_seed(3709)
    net=NativeContinuousHybrid(sensor_dim=3,action_dim=2,width=12,history_limit=4).eval()
    sensor=torch.tensor([[.2,-.3,.7],[-.4,.5,.1]],dtype=torch.float32)
    actions=torch.tensor([[[1.,0.],[-1.,0.],[0.,1.]],
                          [[0.,-1.],[1.,1.],[-1.,1.]]],dtype=torch.float32)
    dt=torch.tensor([[.1],[.2]],dtype=torch.float32)
    return net,sensor,actions,dt


def matured(net,sensor,actions,dt):
    state=net.initial_state(sensor)
    with torch.no_grad():
        for i in range(3):
            _,state=net.step(sensor,actions[:,i],dt,state)
    return state


def snap(state):
    return (state.recurrent.clone(),state.fast_weights.clone(),state.history.clone())


def test_every_candidate_uses_exact_same_factual_prior_state_and_observation():
    net,sensor,candidates,dt=case()
    state=matured(net,sensor,candidates,dt)
    state_before=snap(state)
    model_before={k:v.detach().clone() for k,v in net.state_dict().items()}
    batch=preview_interventions(net,sensor,candidates,dt,state)
    assert batch.shape==(2,3,3)
    for k in range(3):
        single,_=net.step(sensor,candidates[:,k],dt,state)
        torch.testing.assert_close(batch[:,k,:],single,atol=1e-7,rtol=0)
    for a,b in zip(state_before,snap(state)):
        assert torch.equal(a,b)
    assert all(torch.equal(v,model_before[k]) for k,v in net.state_dict().items())
    assert torch.count_nonzero(state.fast_weights)==0


def test_candidate_order_permutation_and_goal_conditioned_action_reversal():
    net,sensor,actions,dt=case()
    original=preview_interventions(net,sensor,actions,dt)
    order=[2,0,1]
    reordered=preview_interventions(net,sensor,actions[:,order],dt)
    torch.testing.assert_close(reordered,original[:,order],atol=1e-7,rtol=0)
    assert not torch.equal(original[:,0],original[:,2])
    left=select_goal_action(net,sensor,actions,dt,original[:,0].detach())
    right=select_goal_action(net,sensor,actions,dt,original[:,2].detach())
    assert left.chosen_indices.tolist()==[0,0]
    assert right.chosen_indices.tolist()==[2,2]
    assert torch.equal(left.chosen_actions,actions[:,0])
    assert torch.equal(right.chosen_actions,actions[:,2])
    assert torch.count_nonzero(left.squared_target_errors[:,0])==0


def test_factual_update_once_after_observation_no_cross_candidate_memory_contamination():
    net,sensor,actions,dt=case()
    source_state=matured(net,sensor,actions,dt)
    old=snap(source_state)
    speculative=preview_interventions(net,sensor,actions,dt,source_state)
    observed=speculative[:,1].detach()+.33  # stand-in for physically observed consequence
    with torch.no_grad():
        prediction,learned=commit_factual_outcome(
            net,sensor,actions[:,1],dt,observed,source_state)
    torch.testing.assert_close(prediction,speculative[:,1],atol=1e-7,rtol=0)
    assert torch.count_nonzero(learned.fast_weights-source_state.fast_weights)>0
    for a,b in zip(old,snap(source_state)):
        assert torch.equal(a,b)
    after=preview_interventions(net,sensor,actions,dt,learned)
    before=preview_interventions(net,sensor,actions,dt,source_state)
    assert not torch.allclose(after,before)
    assert torch.count_nonzero(source_state.fast_weights)==0


def test_no_outcome_no_commit_and_actual_only_writes_factual_memory():
    net,sensor,actions,dt=case()
    source=net.initial_state(sensor)
    previews=preview_interventions(net,sensor,actions,dt,source)
    assert previews.shape==(2,3,3)
    assert torch.count_nonzero(source.fast_weights)==0
    with pytest.raises(TypeError):
        commit_factual_outcome(net,sensor,actions[:,0],dt)
    with pytest.raises(ValueError,match="INVALID_OBSERVED_OUTCOME_VALUES"):
        bad=torch.full_like(sensor,float('nan'))
        commit_factual_outcome(net,sensor,actions[:,0],dt,bad,source)
    assert torch.count_nonzero(source.fast_weights)==0


def test_cold_model_and_native_state_restore_stable_selection():
    net,sensor,actions,dt=case()
    state=matured(net,sensor,actions,dt)
    target=torch.tensor([[.4,.2,.2],[-.2,.4,-.1]])
    first=select_goal_action(net,sensor,actions,dt,target,state)
    mem=io.BytesIO()
    torch.save({'model':net.state_dict(),'recurrent':state.recurrent,
                'fast_weights':state.fast_weights,'history':state.history},mem)
    mem.seek(0)
    data=torch.load(mem,weights_only=True)
    restored=NativeContinuousHybrid(3,2,width=12,history_limit=4).eval()
    restored.load_state_dict(data['model'])
    cold=ContinuousState(data['recurrent'],data['fast_weights'],data['history'])
    second=select_goal_action(restored,sensor,actions,dt,target,cold)
    torch.testing.assert_close(first.predictions,second.predictions,atol=0,rtol=0)
    torch.testing.assert_close(first.squared_target_errors,
                               second.squared_target_errors,atol=0,rtol=0)
    assert torch.equal(first.chosen_indices,second.chosen_indices)


def test_preview_remains_differentiable_not_a_hidden_training_step():
    net,sensor,actions,dt=case()
    pred=preview_interventions(net,sensor,actions,dt)
    pred.square().mean().backward()
    assert net.dynamics.weight.grad is not None
    assert torch.isfinite(net.dynamics.weight.grad).all()


def test_reject_lexical_and_malformed_candidate_state_goal_and_time():
    net,sensor,actions,dt=case()
    with pytest.raises(ValueError,match='NATIVE_NONCONTINUOUS_CANDIDATE_ACTIONS'):
        preview_interventions(net,sensor,actions.long(),dt)
    with pytest.raises(ValueError,match='NATIVE_INVALID_CANDIDATE_ACTIONS_SHAPE'):
        preview_interventions(net,sensor,actions[:1],dt)
    with pytest.raises(ValueError,match='NATIVE_INVALID_CANDIDATE_COUNT'):
        preview_interventions(net,sensor,actions[:,0:0],dt)
    with pytest.raises(ValueError,match='NATIVE_INVALID_CANDIDATE_COUNT'):
        preview_interventions(net,sensor,actions[:,0:1].expand(2,MAX_CANDIDATES+1,2),dt)
    with pytest.raises(ValueError,match='NATIVE_NONPOSITIVE_ELAPSED_TIME'):
        preview_interventions(net,sensor,actions,torch.zeros_like(dt))
    with pytest.raises(ValueError,match='NATIVE_INVALID_EXPLICIT_GOAL_VALUES'):
        goal=sensor.clone();goal[1,0]=float('nan')
        select_goal_action(net,sensor,actions,dt,goal)
    with pytest.raises(ValueError,match='NATIVE_STATE_SHAPE_DRIFT'):
        preview_interventions(net,sensor,actions,dt,ContinuousState(sensor,sensor,sensor))
    with pytest.raises(ValueError,match='NATIVE_INTERVENTION_DEVICE_OR_DTYPE_DRIFT'):
        preview_interventions(net,sensor,actions.double(),dt)


def test_no_tokenizer_or_next_token_objective_in_intervention_component():
    tree=ast.parse((Path(__file__).resolve().parents[1]/'native_intervention_choice.py').read_text())
    attrs=[n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)]
    for x in ('Embedding','CrossEntropyLoss','tokenizer','vocabulary','token_ids'):
        assert x not in attrs
