"""P55 engineering only: zero motor command is completed physical evidence, not proof of causality."""
from __future__ import annotations
import io
import torch
import numpy as np
import pytest
from model import Configuration
from native_factual_hybrid_stream import NativeFactualHybridStream,FactualStreamState
from native_p54_mountaincar_factual_bridge import NativeMountainCarFactualBridge,FactualPhysicalSession,published_step

torch.set_num_threads(1)


def context(batch=2,capacity=8):
    torch.manual_seed(55019)
    native=NativeFactualHybridStream(2,capacity=capacity,cfg=Configuration(width=16,heads=2)).eval()
    bridge=NativeMountainCarFactualBridge(native,max_steps=999)
    state=bridge.start(tuple(5501+i for i in range(batch)))
    return bridge,state


def candidate(batch):
    return torch.tensor([[[-1.],[0.],[1.]]],dtype=torch.float32).expand(batch,-1,-1).clone()


def snapshot(state):
    return [state.physical.clone(),state.steps.clone(),state.rewards.clone(),state.terminated.clone(),state.truncated.clone()]+[getattr(state.memory,k).clone() for k in FactualStreamState.__dataclass_fields__]


def test_zero_is_completed_action_and_upstream_published_step_exact():
    bridge,state=context()
    old=snapshot(state)
    results=[published_step(x.numpy(),0.) for x in state.physical]
    next_state=bridge.actual_step(state,torch.zeros((2,1)))
    for i,physical in enumerate(results):
        np.testing.assert_array_equal(next_state.physical[i].numpy(),physical.next_sensor)
    assert next_state.rewards.tolist()==[0.,0.]
    assert next_state.steps.tolist()==[1,1]
    assert next_state.memory.observations.tolist()==[1,1]
    assert next_state.memory.actions[:,0,0].tolist()==[0.,0.]
    assert all(torch.equal(x,y) for x,y in zip(old,snapshot(state)))


def test_eight_actual_neutrals_remain_unidentifiable_even_with_elapsed_time():
    bridge,state=context()
    for _ in range(8):
        state=bridge.actual_step(state,torch.zeros((2,1)))
    assert state.memory.observations.tolist()==[8,8]
    assert bridge.preview(state,candidate(2)).identified.tolist()==[False,False]
    with pytest.raises(ValueError,match='P50_FACTUAL_ACTION_EFFECT_NOT_IDENTIFIABLE'):
        bridge.native.select(state.physical,candidate(2),torch.ones((2,1)),state.physical,state.memory)


def test_neutral_mixed_real_actions_reach_rank_three_without_fictitious_actions():
    bridge,state=context()
    for a in (0.,1.,-1.,0.):
        state=bridge.actual_step(state,torch.full((2,1),a))
    assert bridge.preview(state,candidate(2)).identified.tolist()==[True,True]
    assert state.memory.actions[0,:4,0].tolist()==[0.,1.,-1.,0.]
    assert state.memory.elapsed_steps[0,:4,0].tolist()==[1.,1.,1.,1.]


def test_zero_is_valid_hypothetical_candidate_and_never_written():
    bridge,state=context()
    for a in (0.,1.,-1.,0.):
        state=bridge.actual_step(state,torch.full((2,1),a))
    old=snapshot(state)
    weights={k:v.detach().clone() for k,v in bridge.native.state_dict().items()}
    c=candidate(2)
    first=bridge.preview(state,c)
    reversed_output=bridge.preview(state,c.flip(1))
    torch.testing.assert_close(first.predictions.flip(1),reversed_output.predictions,atol=0,rtol=0)
    assert first.predictions.shape==(2,3,2)
    assert all(torch.equal(x,y) for x,y in zip(old,snapshot(state)))
    assert all(torch.equal(v,weights[k]) for k,v in bridge.native.state_dict().items())


def test_mixed_batch_zero_versus_nonzero_keeps_physical_and_history_independent():
    bridge,state=context()
    a=torch.tensor([[0.],[1.]])
    future=bridge.actual_step(state,a)
    assert future.memory.actions[:,0,0].tolist()==[0.,1.]
    assert future.rewards.tolist()==pytest.approx([0.,-0.1],abs=1e-7)
    for i in range(2):
        np.testing.assert_array_equal(future.physical[i].numpy(),published_step(state.physical[i].numpy(),float(a[i])).next_sensor)


def test_cold_tensor_checkpoint_with_neutral_facts_and_zero_candidate_is_bit_exact():
    bridge,state=context()
    for a in (0.,-1.,1.,0.):
        state=bridge.actual_step(state,torch.full((2,1),a))
    pred=bridge.preview(state,candidate(2));before=snapshot(state)
    raw={'model':bridge.native.state_dict(),'state':{k:getattr(state.memory,k) for k in FactualStreamState.__dataclass_fields__},'physical':state.physical,'rewards':state.rewards,'steps':state.steps,'terminated':state.terminated,'truncated':state.truncated}
    buf=io.BytesIO();torch.save(raw,buf);buf.seek(0);data=torch.load(buf,weights_only=True)
    restored_native=NativeFactualHybridStream(2,capacity=8,cfg=Configuration(width=16,heads=2)).eval()
    restored_native.load_state_dict(data['model'])
    cold=FactualPhysicalSession(data['physical'],FactualStreamState(**data['state']),data['steps'],data['rewards'],data['terminated'],data['truncated'])
    result=NativeMountainCarFactualBridge(restored_native).preview(cold,candidate(2))
    assert torch.equal(pred.predictions,result.predictions)
    assert torch.equal(pred.identified,result.identified)
    assert all(torch.equal(x,y) for x,y in zip(before,snapshot(cold)))


def test_invalid_motor_and_invalid_time_still_fail_before_mutation():
    bridge,state=context()
    before=snapshot(state)
    for bad,expected in [ (torch.full((2,1),float('nan')),'P50_INVALID_EXECUTED_ACTION'),(torch.tensor([[0.],[1.00001]]),'P54_ACTION_OUTSIDE_PHYSICAL_BOUNDS'),(torch.ones((2,2)),'P50_INVALID_EXECUTED_ACTION') ]:
        with pytest.raises(ValueError,match=expected):bridge.actual_step(state,bad)
    with pytest.raises(ValueError,match='P50_CANDIDATE_LIMIT'):
        bridge.preview(state,torch.zeros((2,17,1)))
    assert all(torch.equal(x,y) for x,y in zip(before,snapshot(state)))


def test_ring_wrap_with_neutral_actions_retains_exact_completed_fact_counts():
    bridge,state=context()
    for a in [0.,0.,1.,0.,-1.,0.,1.,0.,0.]:
        state=bridge.actual_step(state,torch.full((2,1),a))
    assert state.memory.observations.tolist()==[9,9]
    assert state.memory.cursor.tolist()==[1,1]
    assert state.memory.elapsed_total[:,0].tolist()==[9.,9.]
    assert bridge.preview(state,candidate(2)).facts_used.tolist()==[8,8]
    assert bridge.preview(state,candidate(2)).identified.tolist()==[True,True]
