"""P64-S exact-source full-Hybrid streaming engineering fixtures; no science first use."""
import hashlib
import io
from pathlib import Path
import torch
import pytest
from model import HybridPilot, Configuration
from native_p64s_full_hybrid_stream import FullHybridFactualStream, FullHybridFactualState
from native_p62_streaming_ttt import TTTFactualState


def specimen(seed=64, batch=2, width=8):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    original=HybridPilot(5,2,Configuration(width=width,heads=2,ssm=True,attention=True,ttt=True,depth=True)).eval()
    stream=FullHybridFactualStream(original,max_confirmed=8)
    facts=torch.randn(batch,6,5)*0.3
    queries=torch.randn(batch,3,5)*0.3
    return original,stream,facts,queries


def ingest_segments(stream,facts,chunks):
    state=stream.initial_state(facts[:,0])
    outputs=[];off=0
    for n in chunks:
        block=facts[:,off:off+n]
        out,state=stream.ingest_confirmed(block,state,torch.ones(block.shape[:2],dtype=torch.bool))
        outputs.append(out);off+=n
    assert off==facts.shape[1]
    return torch.cat(outputs,dim=1),state


def git_blob(file):
    b=Path(file).read_bytes()
    return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()


def test_exact_parent_source_custody_no_extra_parameters():
    root=Path(__file__).resolve().parents[1]
    assert git_blob(root/'model.py')=='5d9ea57b827aa1796548447868fa5ee14c251b49'
    assert git_blob(root/'native_p62_streaming_ttt.py')=='776e63a0ff9474a860e87088718acb36637a7bec'
    model,stream,_,_=specimen()
    assert stream.core is model and stream.ttt.core is model.ttt
    assert not hasattr(stream,'parameters')


@pytest.mark.parametrize('seed,batch,chunks',[(64,2,[1,2,3]),(86,3,[3,3])])
def test_completed_chunk_outputs_match_original_full_sequence(seed,batch,chunks):
    original,stream,facts,queries=specimen(seed,batch)
    with torch.no_grad():
        continuous=original(facts)
        chunks_out,state=ingest_segments(stream,facts,chunks)
        torch.testing.assert_close(chunks_out,continuous,atol=1e-6,rtol=1e-6)
        assert torch.equal(state.history,facts)
        assert state.ttt.confirmed_count.tolist()==[6]*batch
        for k in range(queries.shape[1]):
            expected=original(torch.cat((facts,queries[:,k:k+1]),dim=1))[:,-1]
            got=stream.preview(queries,state)[:,k]
            torch.testing.assert_close(got,expected,atol=1e-6,rtol=1e-6)


def test_same_state_candidate_order_pure_model_weights_and_telemetry():
    original,stream,facts,q=specimen()
    with torch.no_grad():
        _,state=ingest_segments(stream,facts,[1,2,3])
        before={k:v.clone() for k,v in original.state_dict().items()}
        old_history=state.history.clone(); old_fast=state.ttt.fast_weights.clone()
        old_count=state.ttt.confirmed_count.clone(); old_util=original.depth.last_utilization
        direct=stream.preview(q,state)
        reversed_q=stream.preview(q.flip(1),state)
        again=stream.preview(q,state)
    assert torch.equal(direct,again)
    torch.testing.assert_close(direct,reversed_q.flip(1),atol=1e-6,rtol=1e-6)
    assert torch.equal(old_history,state.history)
    assert torch.equal(old_fast,state.ttt.fast_weights)
    assert torch.equal(old_count,state.ttt.confirmed_count)
    assert original.depth.last_utilization==old_util
    assert all(torch.equal(v,before[k]) for k,v in original.state_dict().items())


def test_unconfirmed_and_invalid_facts_fail_before_mutation():
    original,stream,facts,q=specimen()
    with torch.no_grad():_,state=ingest_segments(stream,facts[:,:3],[1,2])
    old=state.ttt.fast_weights.clone(); hist=state.history.clone()
    utilization=original.depth.last_utilization
    illegal=[(facts[:,3:4],torch.zeros((2,1),dtype=torch.bool),'P64S_UNCONFIRMED_EVENT'),
             (facts[:,3:4].clone().fill_(float('nan')),torch.ones((2,1),dtype=torch.bool),'P64S_INVALID_ANALOG'),
             (facts[:,3:4].double(),torch.ones((2,1),dtype=torch.bool),'P64S_INVALID_ANALOG'),
             (facts[:,3:4],torch.ones((1,1),dtype=torch.bool),'P64S_UNCONFIRMED_EVENT')]
    for f,flag,error in illegal:
        with pytest.raises(ValueError,match=error):stream.ingest_confirmed(f,state,flag)
    with pytest.raises(ValueError,match='P64S_INVALID_ANALOG'):
        stream.preview(q.clone().fill_(float('inf')),state)
    with pytest.raises(ValueError,match='P64S_INVALID_ANALOG'):
        stream.preview(q.double(),state)
    assert torch.equal(old,state.ttt.fast_weights) and torch.equal(hist,state.history)
    assert original.depth.last_utilization==utilization


def test_no_facts_refuses_speculation_and_full_capacity_refuses_updates():
    original,stream,facts,q=specimen()
    s0=stream.initial_state(facts[:,0])
    with pytest.raises(ValueError,match='P64S_NO_CONFIRMED_HISTORY'):
        stream.preview(q,s0)
    with torch.no_grad():_,s6=ingest_segments(stream,facts,[3,3])
    _,s8=stream.ingest_confirmed(q[:,:2],s6,torch.ones((2,2),dtype=torch.bool))
    with pytest.raises(ValueError,match='P64S_FACTUAL_HISTORY_FULL'):
        stream.ingest_confirmed(q[:,2:3],s8,torch.ones((2,1),dtype=torch.bool))
    with pytest.raises(ValueError,match='P64S_CORRUPTED_HISTORY'):
        stream.preview(q,FullHybridFactualState(s6.history[:,:3,:2],s6.ttt))
    with pytest.raises(ValueError,match='P64S_COUNT_MISMATCH'):
        stream.preview(q,FullHybridFactualState(s6.history,
            TTTFactualState(s6.ttt.fast_weights,s6.ttt.confirmed_count+1)))


def test_midstream_cold_model_and_factual_state_restore_exact_continuation():
    original,stream,facts,q=specimen(86,3)
    with torch.no_grad():
        _,s3=ingest_segments(stream,facts[:,:3],[1,2])
        buffer=io.BytesIO()
        torch.save({'model':original.state_dict(),'history':s3.history,
                    'fast_weights':s3.ttt.fast_weights,'count':s3.ttt.confirmed_count},buffer)
        buffer.seek(0)
        payload=torch.load(buffer,weights_only=True)
        cold=HybridPilot(5,2,Configuration(width=8,heads=2,ssm=True,attention=True,ttt=True,depth=True)).eval()
        cold.load_state_dict(payload['model'])
        alternate=FullHybridFactualStream(cold,max_confirmed=8)
        s3restored=FullHybridFactualState(payload['history'],
            TTTFactualState(payload['fast_weights'],payload['count']))
        assert torch.equal(stream.preview(q,s3),alternate.preview(q,s3restored))
        confirmation=torch.ones((3,3),dtype=torch.bool)
        u,s6=stream.ingest_confirmed(facts[:,3:],s3,confirmation)
        v,s6cold=alternate.ingest_confirmed(facts[:,3:],s3restored,confirmation)
        assert torch.equal(u,v)
        assert torch.equal(s6.ttt.fast_weights,s6cold.ttt.fast_weights)
        assert torch.equal(stream.preview(q,s6),alternate.preview(q,s6cold))


def test_end_to_end_gradient_through_original_ssm_attention_ttt_and_head():
    original,stream,facts,q=specimen()
    state=stream.initial_state(facts[:,0])
    _,s=stream.ingest_confirmed(facts[:,:4],state,torch.ones((2,4),dtype=torch.bool))
    pred=stream.preview(q,s)
    pred.square().sum().backward()
    for name in ['stem.weight','ssm.delta.weight','attn.qkv.weight','ttt.key.weight','head.weight']:
        grad=dict(original.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all(), name
        assert float(grad.abs().sum())>1e-10,name


def test_original_reference_path_has_no_new_factual_model_weights():
    original,stream,facts,q=specimen()
    import torch.nn as nn
    assert isinstance(original,nn.Module)
    assert stream.core is original
    assert len(list(original.parameters()))>0
    assert 'Embedding' not in Path(__file__).resolve().parents[1].joinpath('native_p64s_full_hybrid_stream.py').read_text()
