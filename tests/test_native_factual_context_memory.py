"""P39 development-only controls: no consumed source, owner action or training."""
import ast
import io
from pathlib import Path

import pytest
import torch

from native_continuous_state import NativeContinuousHybrid, ContinuousState
from native_factual_context_memory import FactualContextMemory, FactualContextState


def case(slots=4):
    net = NativeContinuousHybrid(2, 1, width=12, history_limit=3).eval()
    with torch.no_grad():
        # Known diagnostic baseline, not a trained model: predicts identity.
        for parameter in net.parameters():
            parameter.zero_()
    sensor = torch.tensor([[0.2, 0.8]], dtype=torch.float32)
    action = torch.tensor([[[1.], [-1.]]], dtype=torch.float32)
    dt = torch.ones((1, 1), dtype=torch.float32)
    memory = FactualContextMemory(net, slots=slots, radius=0.2)
    return memory, sensor, action, dt


def snap(state):
    native = state.native
    return [v.clone() for v in (native.recurrent, native.fast_weights,
            native.history, state.contexts, state.residuals, state.occupied,
            state.next_slot, state.observations)]


def test_preview_is_pure_and_candidate_order_independent():
    memory, sensor, candidates, dt = case()
    state = memory.initial_state(sensor)
    before = snap(state)
    params = {k: v.detach().clone() for k,v in memory.model.state_dict().items()}
    first = memory.preview(sensor, candidates, dt, state)
    swapped = memory.preview(sensor, candidates.flip(1), dt, state)
    torch.testing.assert_close(first.flip(1), swapped, atol=0, rtol=0)
    for a,b in zip(before, snap(state)):
        assert torch.equal(a,b)
    assert all(torch.equal(value, params[key]) for key,value in memory.model.state_dict().items())
    assert state.observations.item() == 0


def test_real_factual_context_retains_action_and_cue_conditioning():
    memory, sensor, candidates, dt = case()
    state = memory.initial_state(sensor)
    original = memory.preview(sensor, candidates, dt, state)
    observed = sensor + torch.tensor([[0.32, 0.]])
    before, state1 = memory.commit_factual(sensor, candidates[:,0], dt, observed, state)
    torch.testing.assert_close(before, sensor, atol=0, rtol=0)
    after = memory.preview(sensor, candidates, dt, state1)
    torch.testing.assert_close(after[:,0], observed, atol=1e-7, rtol=0)
    torch.testing.assert_close(after[:,1], original[:,1], atol=0, rtol=0)
    opposite_context = sensor.clone(); opposite_context[:,1] *= -1
    unrelated = memory.preview(opposite_context, candidates, dt, state1)
    torch.testing.assert_close(unrelated, opposite_context[:,None,:].expand(-1,2,-1), atol=0, rtol=0)
    assert state.observations.item()==0 and state1.observations.item()==1
    assert torch.count_nonzero(state.contexts)==0


def test_sequential_factual_write_and_cold_restore_preserve_choices():
    memory, sensor, candidates, dt = case(slots=3)
    initial = memory.initial_state(sensor)
    _, one = memory.commit_factual(sensor,candidates[:,0],dt,sensor+torch.tensor([[0.3,0.]]),initial)
    opposite = sensor.clone(); opposite[:,1] = -0.8
    _, two = memory.commit_factual(opposite,candidates[:,0],dt,opposite+torch.tensor([[-0.3,0.]]),one)
    positive = memory.preview(sensor,candidates,dt,two)
    negative = memory.preview(opposite,candidates,dt,two)
    assert positive[0,0,0] > sensor[0,0]
    assert negative[0,0,0] < opposite[0,0]
    target = sensor.clone(); target[:,0] += 0.3
    picked,cost = memory.choose(sensor,candidates,dt,target,two)
    assert picked.tolist()==[0]
    payload = io.BytesIO()
    torch.save({'native':{'recurrent':two.native.recurrent,'fast_weights':two.native.fast_weights,
                          'history':two.native.history},'contexts':two.contexts,
                'residuals':two.residuals,'occupied':two.occupied,
                'next_slot':two.next_slot,'observations':two.observations},payload)
    payload.seek(0)
    frozen = torch.load(payload, weights_only=True)
    cold = FactualContextState(ContinuousState(**frozen['native']),
            frozen['contexts'],frozen['residuals'],frozen['occupied'],
            frozen['next_slot'],frozen['observations'])
    cold_picked,cold_cost = memory.choose(sensor,candidates,dt,target,cold)
    assert torch.equal(cold_picked,picked)
    torch.testing.assert_close(cold_cost,cost,atol=0,rtol=0)


def test_deterministic_bounded_eviction_and_negative_controls():
    memory, sensor, candidates, dt = case(slots=2)
    state = memory.initial_state(sensor)
    for i in range(3):
        changed = sensor.clone(); changed[:,0] += i * 0.5
        _,state = memory.commit_factual(changed,candidates[:,0],dt,changed+0.2,state)
    assert state.occupied.sum().item()==2
    assert state.observations.item()==3
    assert state.next_slot.item()==1
    assert not any(torch.allclose(slot,sensor[0]) for slot in state.contexts[0,:,:2])
    with pytest.raises(TypeError):
        memory.commit_factual(sensor,candidates[:,0],dt,state=state)
    with pytest.raises(ValueError,match='INVALID_OBSERVED_OUTCOME_VALUES'):
        memory.commit_factual(sensor,candidates[:,0],dt,torch.full_like(sensor,float('nan')),state)
    assert state.observations.item()==3
    with pytest.raises(ValueError,match='P39_STATE_SHAPE_OR_DEVICE_DRIFT'):
        bad = FactualContextState(state.native,state.contexts[:,:1],state.residuals,
                                 state.occupied,state.next_slot,state.observations)
        memory.preview(sensor,candidates,dt,bad)
    with pytest.raises(ValueError,match='NONCONTINUOUS_CANDIDATE_ACTIONS'):
        memory.preview(sensor,candidates.long(),dt,state)


def test_component_does_not_depend_on_tokenized_cognition():
    path=Path(__file__).resolve().parents[1]/'native_factual_context_memory.py'
    tree=ast.parse(path.read_text())
    attr={node.attr for node in ast.walk(tree) if isinstance(node,ast.Attribute)}
    assert not attr.intersection({'Embedding','tokenizer','CrossEntropyLoss','vocabulary','token_ids'})
