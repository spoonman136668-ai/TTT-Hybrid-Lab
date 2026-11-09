"""P43 source-free controls: no protected first use or accepted science."""
import ast
import io
from pathlib import Path
import torch
import pytest
from native_trainable_causal_belief import (NativeTrainableCausalBelief,
    NativeCausalBeliefState)


def case(batch=2):
    torch.manual_seed(4319)
    model=NativeTrainableCausalBelief(sensor_dim=2,width=12,min_facts=2).eval()
    sensor=torch.tensor([[.2,.3],[-.2,.4]])[:batch]
    action=torch.tensor([[[1.],[-1.]],[[1.],[-1.]]])[:batch]
    dt=torch.ones((batch,1),dtype=torch.float32)
    return model,sensor,action,dt,model.initial_state(sensor)


def test_no_source_no_action_and_pure_same_state_candidate_previews():
    model,sensor,actions,dt,state=case()
    snapshot=(state.hidden.clone(),state.elapsed.clone(),state.observations.clone())
    weights={k:v.detach().clone() for k,v in model.state_dict().items()}
    original=model.preview(sensor,actions,dt,state)
    switched=model.preview(sensor,actions.flip(1),dt,state)
    torch.testing.assert_close(original.predictions.flip(1),switched.predictions,atol=0,rtol=0)
    assert not bool(original.factual_ready.any())
    assert all(torch.equal(x,y) for x,y in zip(snapshot,(state.hidden,state.elapsed,state.observations)))
    assert all(torch.equal(x,weights[k]) for k,x in model.state_dict().items())
    with pytest.raises(ValueError,match='P43_INSUFFICIENT_FACTUAL_HISTORY'):
        model.select(sensor,actions,dt,sensor,state)


def test_update_requires_actual_outcome_and_changes_only_returned_state():
    model,sensor,actions,dt,state=case()
    nxt=sensor+torch.tensor([[.1,-.1],[.2,-.2]])
    updated=model.observe(sensor,actions[:,0],dt,nxt,state)
    assert updated.observations.tolist()==[1,1]
    assert state.observations.tolist()==[0,0]
    assert not torch.equal(updated.hidden,state.hidden)
    assert torch.equal(state.elapsed,torch.zeros_like(state.elapsed))
    with pytest.raises(TypeError):
        model.observe(sensor,actions[:,0],dt,state=state)
    with pytest.raises(ValueError,match='P43_INVALID_FACTUAL_OUTCOME'):
        model.observe(sensor,actions[:,0],dt,torch.full_like(sensor,float('nan')),state)


def test_hidden_contexts_with_different_real_consequences_separate_beliefs():
    model,sensor,actions,dt,state=case()
    next_a=sensor+torch.tensor([[.25,0.],[.25,0.]])
    next_b=sensor+torch.tensor([[-.25,0.],[-.25,0.]])
    first=model.observe(sensor,actions[:,0],dt,next_a,state)
    second=model.observe(sensor,actions[:,0],dt,next_b,state)
    assert not torch.equal(first.hidden,second.hidden)
    first2=model.observe(sensor,actions[:,1],dt,next_a,first)
    second2=model.observe(sensor,actions[:,1],dt,next_b,second)
    assert (first2.observations==2).all()
    assert not torch.equal(model.preview(sensor,actions,dt,first2).predictions,
                           model.preview(sensor,actions,dt,second2).predictions)
    goal=sensor+.1
    assert model.select(sensor,actions,dt,goal,first2).shape==(2,)


def test_factual_gradient_path_reaches_recurrent_learning_and_both_heads():
    model,sensor,actions,dt,state=case()
    current=state
    for j in range(3):
        actual_next=sensor+(.2 if j%2==0 else -.2)*actions[:,0]
        current=model.observe(sensor,actions[:,0],dt,actual_next,current)
    pred=model.preview(sensor,actions,dt,current).predictions
    # Train against factual candidate only, never counterfactual oracle.
    truth=sensor+.3*actions[:,0]
    loss=(pred[:,0]-truth).square().mean()
    loss.backward()
    assert model.memory_update.weight_ih.grad is not None
    assert float(model.memory_update.weight_ih.grad.abs().sum())>0
    assert float(model.exogenous_head.weight.grad.abs().sum())>0
    assert float(model.action_effect_head.weight.grad.abs().sum())>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_cold_checkpoint_and_state_restore_exact_predictions():
    model,sensor,actions,dt,state=case()
    for i in range(2):
        state=model.observe(sensor,actions[:,i],dt,sensor+.2*actions[:,i],state)
    p=model.preview(sensor,actions,dt,state)
    buffer=io.BytesIO()
    torch.save({'model':model.state_dict(),'state':{
        'hidden':state.hidden,'observations':state.observations,'elapsed':state.elapsed}},buffer)
    buffer.seek(0)
    payload=torch.load(buffer,weights_only=True)
    cold=NativeTrainableCausalBelief(2,12,2).eval()
    cold.load_state_dict(payload['model'])
    restored=NativeCausalBeliefState(**payload['state'])
    q=cold.preview(sensor,actions,dt,restored)
    assert torch.equal(p.predictions,q.predictions)
    assert torch.equal(p.factual_ready,q.factual_ready)


def test_fails_closed_on_invalid_actions_dtype_device_and_state():
    model,sensor,actions,dt,state=case()
    with pytest.raises(ValueError,match='P43_INVALID_CANDIDATE_ACTIONS'):
        model.preview(sensor,actions.long(),dt,state)
    with pytest.raises(ValueError,match='P43_ZERO_ACTION_UNIDENTIFIABLE'):
        model.observe(sensor,torch.zeros_like(actions[:,0]),dt,sensor,state)
    with pytest.raises(ValueError,match='P43_NONPOSITIVE_TIME'):
        model.observe(sensor,actions[:,0],torch.zeros_like(dt),sensor,state)
    invalid=NativeCausalBeliefState(state.hidden[:,:2],state.observations,state.elapsed)
    with pytest.raises(ValueError,match='P43_STATE_SHAPE_DRIFT'):
        model.preview(sensor,actions,dt,invalid)


def test_token_free_component_no_lexical_objective():
    f=Path(__file__).resolve().parents[1]/'native_trainable_causal_belief.py'
    tree=ast.parse(f.read_text())
    names={x.attr for x in ast.walk(tree) if isinstance(x,ast.Attribute)}
    assert not names.intersection({'Embedding','CrossEntropyLoss','tokenizer','vocabulary','token_ids'})


def test_bounded_factual_training_smoke_reduces_same_fixture_loss():
    """Development trainability only, not sealed transfer/scientific qualification."""
    torch.manual_seed(43011)
    torch.set_num_threads(1)
    model=NativeTrainableCausalBelief(sensor_dim=1,width=16,min_facts=2)
    sensor=torch.zeros((12,1))
    mode=torch.tensor([[1.],[-1.]]).repeat(6,1)
    dt=torch.ones_like(sensor)
    candidate=torch.tensor([[[1.],[-1.]]]).expand(12,-1,-1)
    optimizer=torch.optim.Adam(model.parameters(),lr=.02)
    losses=[]
    for i in range(80):
        state=model.initial_state(sensor)
        for factual_action in (1.,-1.,1.):
            action=torch.full_like(sensor,factual_action)
            observed_next=sensor+0.10+0.30*mode*action
            state=model.observe(sensor,action,dt,observed_next,state)
        predicted=model.preview(sensor,candidate,dt,state)
        truth=sensor+0.10+0.30*mode
        loss=(predicted.predictions[:,0]-truth).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * .1
    assert losses[-1] < .002
