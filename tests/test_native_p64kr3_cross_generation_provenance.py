"""P64-K R3 frozen generation-tag regression; manual tensors, engineering DEV only.

The state provenance tag is an exact-model-content identity, not a cryptographic
signature of physical source facts. No scientific first use is exercised here.
"""
from __future__ import annotations
import io
import torch
import pytest
from model import HybridPilot, Configuration
from native_p62_streaming_ttt import TTTFactualState
from native_p64k_incremental_hybrid_stream import StreamingHybridPilot, HybridFactualState


def sample(seed=1691):
    torch.manual_seed(seed)
    core=HybridPilot(5,2,Configuration(width=8,heads=2,ssm=True,attention=True,ttt=True,depth=True)).eval()
    stream=StreamingHybridPilot(core,max_confirmed_events=8)
    facts=torch.randn((2,3,5))
    queries=torch.randn((2,2,5))
    with torch.no_grad():
        zero=stream.initial_state(facts[:,0])
        _,state=stream.ingest_confirmed(facts,zero,torch.ones((2,3),dtype=torch.bool))
    return core,stream,facts,queries,state


def test_stamp_persists_every_confirmed_step_and_untouched_previews():
    core,stream,facts,query,state=sample()
    assert state.model_fingerprint==stream._model_fingerprint
    assert isinstance(state.model_fingerprint,str) and len(state.model_fingerprint)==64
    assert torch.isfinite(stream.preview(query,state)).all()
    with torch.no_grad():
        more,state2=stream.ingest_confirmed(facts[:,:1],state,torch.ones((2,1),dtype=torch.bool))
    assert more.shape==(2,1,2) and state2.model_fingerprint==state.model_fingerprint
    assert state2.ttt.confirmed_count.tolist()==[4,4]


def test_cross_adapter_rejects_old_state_after_in_place_weight_change():
    core,old,facts,query,state=sample(1693)
    with torch.no_grad():
        core.head.weight.add_(0.25)
    newer=StreamingHybridPilot(core,max_confirmed_events=8)
    assert newer._model_fingerprint!=state.model_fingerprint
    with pytest.raises(ValueError,match='P64K_STALE_OR_UNTAGGED_FACTUAL_STATE'):
        newer.preview(query,state)
    with pytest.raises(ValueError,match='P64K_STALE_OR_UNTAGGED_FACTUAL_STATE'):
        newer.ingest_confirmed(facts[:,:1],state,torch.ones((2,1),dtype=torch.bool))
    with pytest.raises(ValueError,match='P64K_MODEL_CHANGED_REBUILD_FACTUAL_CACHE'):
        old.preview(query,state)
    with torch.no_grad():
        initialized=newer.initial_state(facts[:,0])
        _,refreshed=newer.ingest_confirmed(facts,initialized,torch.ones((2,3),dtype=torch.bool))
        assert torch.isfinite(newer.preview(query,refreshed)).all()


def test_same_shape_unrelated_model_cannot_inherit_foreign_state():
    _,_,facts,query,state=sample(1695)
    torch.manual_seed(5001)
    independent=HybridPilot(5,2,Configuration(width=8,heads=2,ssm=True,attention=True,ttt=True,depth=True)).eval()
    new=StreamingHybridPilot(independent,max_confirmed_events=8)
    assert new._model_fingerprint!=state.model_fingerprint
    with pytest.raises(ValueError,match='P64K_STALE_OR_UNTAGGED_FACTUAL_STATE'):
        new.preview(query,state)


def test_exact_weight_cold_restore_keeps_original_preview_bitwise():
    core,old,facts,queries,state=sample(1697)
    with torch.no_grad():
        expected=old.preview(queries,state)
    buffer=io.BytesIO()
    torch.save({'weights':core.state_dict(),'factual':{
        'ssm':state.ssm_hidden,'ak':state.attention_k,'av':state.attention_v,
        'tw':state.ttt.fast_weights,'count':state.ttt.confirmed_count,
        'source_model_sha256':state.model_fingerprint}},buffer)
    buffer.seek(0)
    restored_data=torch.load(buffer,weights_only=True)
    other=HybridPilot(5,2,Configuration(width=8,heads=2,ssm=True,attention=True,ttt=True,depth=True)).eval()
    other.load_state_dict(restored_data['weights'])
    new=StreamingHybridPilot(other,max_confirmed_events=8)
    f=restored_data['factual']
    cold=HybridFactualState(f['ssm'],f['ak'],f['av'],
        TTTFactualState(f['tw'],f['count']),f['source_model_sha256'])
    assert new._model_fingerprint==state.model_fingerprint
    with torch.no_grad():
        actual=new.preview(queries,cold)
    assert torch.equal(expected,actual)


def test_old_untagged_manual_cold_state_fails_closed():
    _,stream,facts,query,state=sample(1699)
    legacy=HybridFactualState(state.ssm_hidden,state.attention_k,state.attention_v,state.ttt)
    assert legacy.model_fingerprint is None
    with pytest.raises(ValueError,match='P64K_STALE_OR_UNTAGGED_FACTUAL_STATE'):
        stream.preview(query,legacy)


def test_eta_change_makes_old_tag_incompatible_with_new_adapter():
    core,old,_,query,state=sample(1701)
    core.ttt.eta=float(core.ttt.eta)*0.5
    newer=StreamingHybridPilot(core,max_confirmed_events=8)
    assert newer._model_fingerprint!=state.model_fingerprint
    with pytest.raises(ValueError,match='P64K_STALE_OR_UNTAGGED_FACTUAL_STATE'):
        newer.preview(query,state)
    with pytest.raises(ValueError,match='P64K_MODEL_CHANGED_REBUILD_FACTUAL_CACHE'):
        old.preview(query,state)


def test_optimizer_step_new_adapter_cannot_consume_old_training_state():
    core,old,facts,query,state=sample(1703)
    opt=torch.optim.SGD(core.parameters(),lr=0.05)
    opt.zero_grad(set_to_none=True)
    core.stem.weight.grad=torch.ones_like(core.stem.weight)
    opt.step()
    new=StreamingHybridPilot(core,max_confirmed_events=8)
    with pytest.raises(ValueError,match='P64K_STALE_OR_UNTAGGED_FACTUAL_STATE'):
        new.preview(query,state)
    init=new.initial_state(facts[:,0])
    assert init.ttt.confirmed_count.tolist()==[0,0]
    with torch.no_grad():
        _,state2=new.ingest_confirmed(facts,init,torch.ones((2,3),dtype=torch.bool))
        actual=new.preview(query,state2)
        original=torch.cat([core(torch.cat((facts,query[:,j:j+1]),dim=1))[:,-1:, :]
                              for j in range(2)],dim=1)
    torch.testing.assert_close(actual,original,atol=1e-6,rtol=1e-6)


def test_same_weight_load_state_dict_new_adapter_preserves_provenance():
    core,old,_,query,state=sample(1705)
    snapshot={k:v.detach().clone() for k,v in core.state_dict().items()}
    core.load_state_dict(snapshot)
    newer=StreamingHybridPilot(core,max_confirmed_events=8)
    # Old adapter must reject its own PyTorch parameter revision, but a newly
    # restored exact-weight model may reuse identically sourced factual tensors.
    assert newer._model_fingerprint==state.model_fingerprint
    with pytest.raises(ValueError,match='P64K_MODEL_CHANGED_REBUILD_FACTUAL_CACHE'):
        old.preview(query,state)
    with torch.no_grad():
        assert torch.isfinite(newer.preview(query,state)).all()
