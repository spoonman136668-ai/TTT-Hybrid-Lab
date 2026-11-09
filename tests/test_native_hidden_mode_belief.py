"""P40 no-source engineering controls: hidden mode is never an API argument."""
import io
import torch
import pytest
from native_hidden_mode_belief import FactualModeInference, HiddenModeBelief


def fixture():
    model=FactualModeInference(1,capacity=2,novelty=.1,noise=.05)
    sensor=torch.tensor([[0.2]],dtype=torch.float32)
    candidates=torch.tensor([[[-1.],[1.]]])
    dt=torch.ones((1,1))
    return model,sensor,candidates,dt


def actual(sensor,action,mode):
    # Mode is only visible to the *source* that supplies the factual outcome.
    return sensor+0.30*mode*action


def test_no_observation_no_hidden_oracle_and_read_only_candidates():
    model,sensor,candidates,dt=fixture()
    state=model.initial_state(sensor)
    prediction=model.preview(sensor,candidates,dt,state)
    torch.testing.assert_close(prediction,sensor.unsqueeze(1).expand(-1,2,-1))
    assert state.factual_steps.item()==0
    assert state.observations.sum().item()==0
    for _ in range(3):model.preview(sensor,candidates.flip(1),dt,state)
    assert state.factual_steps.item()==0


def test_learns_factual_effect_then_reverses_unobserved_mode_after_real_feedback():
    model,sensor,candidates,dt=fixture()
    initial=model.initial_state(sensor)
    plus=candidates[:,1]
    one=model.observe(sensor,plus,dt,actual(sensor,plus,+1),initial)
    predicted=model.preview(sensor,candidates,dt,one)
    assert predicted[0,1,0].item()>sensor[0,0].item()+.29
    assert model.choose(sensor,candidates,dt,sensor+.3,one).tolist()==[1]
    # Under a hidden reversal, the old belief cannot know the new mode before feedback.
    assert model.choose(sensor,candidates,dt,sensor+.3,one).tolist()==[1]
    two=model.observe(sensor,plus,dt,actual(sensor,plus,-1),one)
    assert two.observations.sum().item()==2
    assert two.probabilities[0,1]>.99
    assert model.choose(sensor,candidates,dt,sensor+.3,two).tolist()==[0]
    assert model.choose(sensor,candidates,dt,sensor-.3,two).tolist()==[1]
    assert initial.observations.sum().item()==0
    assert one.observations.sum().item()==1


def test_mode_recall_after_interference_and_cold_tensors():
    model,sensor,candidates,dt=fixture()
    state=model.initial_state(sensor)
    for mode in (+1,-1,-1,+1):
        state=model.observe(sensor,candidates[:,1],dt,actual(sensor,candidates[:,1],mode),state)
    assert state.observations.sum().item()==4
    assert (state.observations[0]>0).sum().item()==2
    assert model.choose(sensor,candidates,dt,sensor+.3,state).tolist()==[1]
    f=io.BytesIO()
    torch.save({'effects':state.effects,'observations':state.observations,
        'probabilities':state.probabilities,'factual_steps':state.factual_steps},f)
    f.seek(0)
    cold=HiddenModeBelief(**torch.load(f,weights_only=True))
    assert torch.equal(model.preview(sensor,candidates,dt,state),model.preview(sensor,candidates,dt,cold))
    assert torch.equal(state.probabilities,cold.probabilities)


def test_explicit_negative_unidentifiability_before_feedback():
    model,sensor,candidates,dt=fixture()
    # Identical visible observations + the same factual history must produce
    # the same action; hidden mode cannot be inferred without information.
    state=model.observe(sensor,candidates[:,1],dt,actual(sensor,candidates[:,1],+1),model.initial_state(sensor))
    first=model.choose(sensor,candidates,dt,sensor+.3,state)
    alternative=model.choose(sensor,candidates,dt,sensor+.3,state)
    assert torch.equal(first,alternative)
    # Correct response under mode=-1 cannot be guaranteed before new evidence.
    assert first.tolist()==[1]
    assert (actual(sensor,candidates[:,1],-1)<sensor).all()


def test_fail_closed_invalid_observations_capacity_and_device_shapes():
    model,sensor,candidates,dt=fixture()
    state=model.initial_state(sensor)
    with pytest.raises(ValueError,match='INVALID_OBSERVED_NEXT'):
        model.observe(sensor,candidates[:,1],dt,torch.full_like(sensor,float('nan')),state)
    with pytest.raises(ValueError,match='INVALID_ACTION'):
        model.observe(sensor,torch.zeros_like(candidates[:,1]),dt,sensor,state)
    with pytest.raises(ValueError,match='INVALID_ELAPSED_TIME'):
        model.preview(sensor,candidates,torch.zeros_like(dt),state)
    with pytest.raises(ValueError,match='INVALID_ACTION'):
        model.preview(sensor,candidates.long(),dt,state)
    for mode in (+1,-1):
        state=model.observe(sensor,candidates[:,1],dt,actual(sensor,candidates[:,1],mode),state)
    old=state.effects.clone()
    with pytest.raises(ValueError,match='P40_UNMODELED_DYNAMICS_CAPACITY'):
        model.observe(sensor,candidates[:,1],dt,sensor,state)
    assert torch.equal(state.effects,old)
    with pytest.raises(ValueError,match='P40_BELIEF_SHAPE_DRIFT'):
        bad=HiddenModeBelief(state.effects[:,:1],state.observations,state.probabilities,state.factual_steps)
        model.preview(sensor,candidates,dt,bad)
