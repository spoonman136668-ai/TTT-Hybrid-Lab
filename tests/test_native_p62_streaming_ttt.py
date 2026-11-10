"""P62 preregistered manual-tensor continuity, purity, error, and cold controls."""
from __future__ import annotations
import hashlib
import io
from pathlib import Path
import pytest
import torch
from model import TTTLinearMemory
from native_p62_streaming_ttt import StreamingTTT, TTTFactualState


def core_and_chunks(seed=61,batch=2):
    torch.manual_seed(seed)
    core=TTTLinearMemory(6).eval()
    facts=torch.randn(batch,7,6)
    query=torch.randn(batch,2,6)
    return core,facts,query


def full_and_streamed(core, facts, query):
    fresh=StreamingTTT(core, max_confirmed_events=16)
    s0=fresh.initial_state(facts[:,0,:])
    a,s1=fresh.ingest_confirmed(facts[:,:3],s0,torch.ones((len(facts),3),dtype=torch.bool))
    b,s2=fresh.ingest_confirmed(facts[:,3:5],s1,torch.ones((len(facts),2),dtype=torch.bool))
    c,s3=fresh.ingest_confirmed(facts[:,5:],s2,torch.ones((len(facts),2),dtype=torch.bool))
    q=fresh.preview(query,s3)
    return torch.cat((a,b,c),dim=1),q,s0,s3


def test_original_source_git_blob_and_zero_new_trainable_params():
    source=(Path(__file__).resolve().parents[1]/'model.py').read_bytes()
    sha=hashlib.sha1(b'blob '+str(len(source)).encode()+b'\0'+source).hexdigest()
    assert sha=='5d9ea57b827aa1796548447868fa5ee14c251b49'
    core,_,_=core_and_chunks()
    streaming=StreamingTTT(core)
    assert streaming.core is core
    assert not hasattr(streaming,'state_dict')


def test_segmented_factual_stream_matches_original_full_sequence_two_seeds():
    for seed,batch in ((61,2),(83,3)):
        core,facts,query=core_and_chunks(seed,batch)
        with torch.no_grad():
            original=core(torch.cat((facts,query[:,:1]),dim=1),update=True)
            seen,preview,s0,s3=full_and_streamed(core,facts,query[:,:1])
        torch.testing.assert_close(seen,original[:,:facts.shape[1]],atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(preview,original[:,facts.shape[1]:],atol=1e-6,rtol=1e-6)
        assert torch.equal(s0.confirmed_count,torch.zeros_like(s0.confirmed_count))
        assert s3.confirmed_count.tolist()==[facts.shape[1]]*batch


def test_preview_is_pure_reorder_independent_read_before_write():
    core,facts,q=core_and_chunks()
    with torch.no_grad():
        _,f,s0,s=full_and_streamed(core,facts,q)
        before=s.fast_weights.detach().clone();count=s.confirmed_count.clone()
        forward=StreamingTTT(core).preview(q,s)
        reverse=StreamingTTT(core).preview(q.flip(1),s)
        another=StreamingTTT(core).preview(q,s)
    assert torch.equal(forward,another)
    torch.testing.assert_close(forward,reverse.flip(1),atol=1e-6,rtol=1e-6)
    assert torch.equal(s.fast_weights,before) and torch.equal(s.confirmed_count,count)
    assert torch.equal(s0.confirmed_count,torch.zeros_like(s0.confirmed_count))


def test_reset_between_chunks_is_not_equivalent_to_factual_carry():
    core,facts,query=core_and_chunks(61)
    with torch.no_grad():
        _,streamed,_,s=full_and_streamed(core,facts,query)
        reset=core(torch.cat((facts[:,5:],query),dim=1),update=True)[:,-2:]
    assert float((streamed-reset).abs().max())>1e-5
    assert float(s.fast_weights.abs().max())>0


def test_fail_closed_unconfirmed_shape_nonfinite_and_memory_budget_are_atomic():
    core,facts,query=core_and_chunks()
    model=StreamingTTT(core,max_confirmed_events=7)
    initial=model.initial_state(facts[:,0,:])
    confirmed=torch.ones((2,7),dtype=torch.bool)
    with torch.no_grad():
        _,s=model.ingest_confirmed(facts,initial,confirmed)
    weights=s.fast_weights.clone();counts=s.confirmed_count.clone()
    for bad,mask,err in [
       (facts[:,:1],torch.zeros((2,1),dtype=torch.bool),'P62_UNCONFIRMED_FACT_MUST_NOT_COMMIT'),
       (facts[:,:1],torch.ones((2,1),dtype=torch.bool),'P62_BOUNDED_FACTUAL_MEMORY_FULL'),
       (facts[:,:1].clone().fill_(float('nan')),torch.ones((2,1),dtype=torch.bool),'P62_NONFINITE_FACT_OR_STATE'),
       (facts[:,:1].double(),torch.ones((2,1),dtype=torch.bool),'P62_CORE_DEVICE_DTYPE_MISMATCH'),
    ]:
        with pytest.raises(ValueError,match=err):model.ingest_confirmed(bad,s,mask)
    with pytest.raises(ValueError,match='P62_BAD_SEQUENCE_SHAPE'):model.preview(query[0],s)
    assert torch.equal(s.fast_weights,weights) and torch.equal(s.confirmed_count,counts)
    assert torch.equal(initial.confirmed_count,torch.zeros_like(initial.confirmed_count))


def test_cold_replay_tensor_weights_and_fast_state_bit_exact():
    core,facts,query=core_and_chunks(83)
    model=StreamingTTT(core)
    s0=model.initial_state(facts[:,0,:])
    with torch.no_grad():
        _,s=model.ingest_confirmed(facts,s0,torch.ones((len(facts),7),dtype=torch.bool))
        before=model.preview(query,s)
    b=io.BytesIO()
    torch.save({'core':core.state_dict(),'fast_weights':s.fast_weights,'confirmed_count':s.confirmed_count},b)
    b.seek(0)
    d=torch.load(b,weights_only=True)
    replica=TTTLinearMemory(6).eval()
    replica.load_state_dict(d['core'])
    state=TTTFactualState(d['fast_weights'],d['confirmed_count'])
    with torch.no_grad():after=StreamingTTT(replica).preview(query,state)
    assert torch.equal(before,after) and torch.equal(s.fast_weights,state.fast_weights)


def test_factual_training_remains_differentiable_with_no_preview_update():
    core,facts,query=core_and_chunks(61)
    model=StreamingTTT(core)
    s=model.initial_state(facts[:,0,:])
    _,s2=model.ingest_confirmed(facts[:,:4],s,torch.ones((2,4),dtype=torch.bool))
    pred=model.preview(query,s2)
    pred.square().mean().backward()
    assert core.key.weight.grad is not None
    assert bool(torch.isfinite(core.key.weight.grad).all())
    assert float(core.key.weight.grad.abs().sum())>0
