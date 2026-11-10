"""P64-K R2 local tensor-only parameter-revision custody controls.

No original source or protected scientific cohort is used by these fixtures.
A new model revision MUST NOT reuse the old SSM/KV/TTT factual cache.
"""
from __future__ import annotations
import torch
import pytest
from model import HybridPilot, Configuration
from native_p64k_incremental_hybrid_stream import StreamingHybridPilot


def sample(seed=1649):
    torch.manual_seed(seed)
    core=HybridPilot(5,2,Configuration(width=8,heads=2,ssm=True,attention=True,ttt=True,depth=True)).eval()
    adapter=StreamingHybridPilot(core,max_confirmed_events=8)
    facts=torch.randn((2,3,5))
    query=torch.randn((2,2,5))
    zero=adapter.initial_state(facts[:,0])
    with torch.no_grad():
        _,state=adapter.ingest_confirmed(facts,zero,torch.ones((2,3),dtype=torch.bool))
    return core,adapter,facts,query,state


def assert_expired(adapter,query,state):
    with pytest.raises(ValueError,match='P64K_MODEL_CHANGED_REBUILD_FACTUAL_CACHE'):
        adapter.preview(query,state)
    with pytest.raises(ValueError,match='P64K_MODEL_CHANGED_REBUILD_FACTUAL_CACHE'):
        adapter.ingest_confirmed(query[:,:1],state,torch.ones((2,1),dtype=torch.bool))
    with pytest.raises(ValueError,match='P64K_MODEL_CHANGED_REBUILD_FACTUAL_CACHE'):
        adapter.initial_state(query[:,0])


def test_unchanged_reference_model_retains_same_state_preview_purity():
    core,adapter,_,query,state=sample()
    weights={k:v.detach().clone() for k,v in core.state_dict().items()}
    with torch.no_grad():
        a=adapter.preview(query,state)
        b=adapter.preview(query.flip(1),state).flip(1)
    torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
    assert state.ttt.confirmed_count.tolist()==[3,3]
    assert all(torch.equal(v,weights[k]) for k,v in core.state_dict().items())


def test_optimizer_parameter_step_invalidates_active_factual_cache():
    core,adapter,facts,query,state=sample(1651)
    optimizer=torch.optim.SGD(core.parameters(),lr=0.04)
    optimizer.zero_grad(set_to_none=True)
    core.head.weight.grad=torch.ones_like(core.head.weight)
    optimizer.step()
    assert_expired(adapter,query,state)
    # No inference of old facts against newly fitted weights: a new adapter
    # must start with an empty, source-reconfirmed factual memory.
    fresh=StreamingHybridPilot(core,max_confirmed_events=8)
    empty=fresh.initial_state(facts[:,0])
    assert empty.ttt.confirmed_count.tolist()==[0,0]
    with torch.no_grad():
        _,new=fresh.ingest_confirmed(facts,empty,torch.ones((2,3),dtype=torch.bool))
        predicted=fresh.preview(query,new)
    assert predicted.shape==(2,2,2) and torch.isfinite(predicted).all()


def test_in_place_weight_change_invalidates_all_cached_calls():
    core,adapter,_,query,state=sample(1653)
    with torch.no_grad():
        core.stem.weight.add_(.125)
    assert_expired(adapter,query,state)


def test_load_state_dict_same_values_still_refuses_stale_cache():
    core,adapter,_,query,state=sample(1655)
    snapshot={name:v.clone() for name,v in core.state_dict().items()}
    core.load_state_dict(snapshot)
    assert_expired(adapter,query,state)


def test_parameter_object_replacement_refuses_old_cache():
    core,adapter,_,query,state=sample(1657)
    old=core.head.weight
    core.head.weight=torch.nn.Parameter(old.detach().clone())
    assert_expired(adapter,query,state)


def test_original_TTT_fast_learning_rate_change_refuses_old_fast_weights():
    core,adapter,_,query,state=sample(1659)
    core.ttt.eta=float(core.ttt.eta)*.5
    assert_expired(adapter,query,state)


def test_grad_storage_only_without_weight_update_keeps_cache_valid():
    core,adapter,_,query,state=sample(1661)
    core.stem.weight.grad=torch.ones_like(core.stem.weight)
    with torch.no_grad():
        a=adapter.preview(query,state)
        b=adapter.preview(query,state)
    assert torch.equal(a,b)


def test_fresh_adapter_after_explicit_model_revision_load_is_usable():
    core,old,facts,query,state=sample(1663)
    core.load_state_dict({k:v.clone() for k,v in core.state_dict().items()})
    with pytest.raises(ValueError,match='P64K_MODEL_CHANGED_REBUILD_FACTUAL_CACHE'):
        old.preview(query,state)
    new=StreamingHybridPilot(core,max_confirmed_events=8)
    init=new.initial_state(facts[:,0])
    with torch.no_grad():
        _,confirmed=new.ingest_confirmed(facts,init,torch.ones((2,3),dtype=torch.bool))
        prediction=new.preview(query,confirmed)
        ref=core(torch.cat((facts,query[:,:1]),dim=1))[:,-1]
    torch.testing.assert_close(prediction[:,0],ref,atol=1e-6,rtol=1e-6)
