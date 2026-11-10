"""P64-S preregistered exact-original four-block stream parity and state controls."""
from __future__ import annotations
import hashlib
import io
from pathlib import Path
import pytest
import torch
from model import HybridPilot, Configuration
from native_p64k_incremental_hybrid_stream import StreamingHybridPilot, HybridFactualState
from native_p62_streaming_ttt import TTTFactualState


def setup(seed=641,batch=1):
    torch.manual_seed(seed)
    core=HybridPilot(5,2,Configuration(width=8,heads=2,ssm=True,attention=True,ttt=True,depth=True)).eval()
    stream=StreamingHybridPilot(core,max_confirmed_events=12)
    events=torch.randn(batch,9,5)
    candidates=torch.randn(batch,2,5)
    state=stream.initial_state(events[:,0,:])
    return core,stream,events,candidates,state


def ingest_in_chunks(stream,events,state):
    ys=[]
    off=0
    for n in (2,3,4):
        out,state=stream.ingest_confirmed(events[:,off:off+n,:],state,
                                           torch.ones((len(events),n),dtype=torch.bool))
        ys.append(out)
        off+=n
    return torch.cat(ys,dim=1),state


def state_tensors(s):
    return (s.ssm_hidden,s.attention_k,s.attention_v,s.ttt.fast_weights,s.ttt.confirmed_count)


def test_original_custody_and_parameter_identity():
    root=Path(__file__).resolve().parents[1]
    for filename,expected in [('model.py','5d9ea57b827aa1796548447868fa5ee14c251b49'),
                               ('native_p62_streaming_ttt.py','776e63a0ff9474a860e87088718acb36637a7bec')]:
        blob=(root/filename).read_bytes()
        sha=hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()
        assert sha==expected
    core,stream,_,_,_=setup()
    assert stream.core is core and stream.ttt.core is core.ttt
    assert len(list(core.parameters()))==len(list(stream.core.parameters()))


def test_full_stream_all_four_original_blocks_parity_three_seeds():
    torch.set_num_threads(1)
    for seed,batch in ((641,1),(643,2),(647,3)):
        core,stream,events,candidates,initial=setup(seed,batch)
        with torch.no_grad():
            original=core(events)
            original_candidates=torch.cat([core(torch.cat((events,candidates[:,i:i+1,:]),dim=1))[:,-1:,:]
                                          for i in range(2)],dim=1)
            core.depth.last_utilization=.234
            streamed,state=ingest_in_chunks(stream,events,initial)
            forecast=stream.preview(candidates,state)
        torch.testing.assert_close(streamed,original,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(forecast,original_candidates,atol=1e-6,rtol=1e-6)
        assert core.depth.last_utilization==.234
        assert state.ttt.confirmed_count.tolist()==[9]*batch
        assert state.attention_k.shape==(batch,2,9,4)
        assert state.attention_v.shape==(batch,2,9,4)


def test_candidate_permutation_and_same_state_preview_exact():
    _,stream,events,candidates,state=setup(643,2)
    with torch.no_grad():
        _,s=ingest_in_chunks(stream,events,state)
        before=tuple(z.clone() for z in state_tensors(s))
        u=stream.preview(candidates,s)
        v=stream.preview(candidates.flip(1),s).flip(1)
        w=stream.preview(candidates,s)
    assert torch.equal(u,v) and torch.equal(u,w)
    assert all(torch.equal(x,y) for x,y in zip(state_tensors(s),before))


def test_atomic_invalid_unconfirmed_nonfinite_and_capacity():
    _,stream,events,candidates,state=setup()
    old=tuple(z.clone() for z in state_tensors(state))
    with pytest.raises(ValueError,match='UNCONFIRMED'):
        stream.ingest_confirmed(events,state,torch.zeros((1,9),dtype=torch.bool))
    bad=events.clone();bad[0,8,2]=float('nan')
    with pytest.raises(ValueError,match='BAD_SEQUENCE'):
        stream.ingest_confirmed(bad,state,torch.ones((1,9),dtype=torch.bool))
    assert all(torch.equal(x,y) for x,y in zip(state_tensors(state),old))
    _,full=ingest_in_chunks(stream,events,state)
    with pytest.raises(ValueError,match='BOUNDED'):
        stream.ingest_confirmed(events[:,:4],full,torch.ones((1,4),dtype=torch.bool))
    assert full.attention_k.shape[2]==9 and full.ttt.confirmed_count.item()==9
    with pytest.raises(ValueError,match='BAD_SEQUENCE'):
        stream.preview(candidates.expand(1,2,-1).repeat(1,9,1),full)


def test_cold_tensor_only_restore_does_not_regenerate_factual_prefix():
    core,stream,events,candidates,state=setup(647,3)
    with torch.no_grad():
        _,state=ingest_in_chunks(stream,events,state)
        original=stream.preview(candidates,state)
    io_buf=io.BytesIO()
    torch.save({'weights':core.state_dict(), 'state':{'ssm':state.ssm_hidden,
        'ak':state.attention_k,'av':state.attention_v,
        'tw':state.ttt.fast_weights,'n':state.ttt.confirmed_count}},io_buf)
    io_buf.seek(0)
    loaded=torch.load(io_buf,weights_only=True)
    replacement=HybridPilot(5,2,Configuration(width=8,heads=2)).eval()
    replacement.load_state_dict(loaded['weights'])
    adapter=StreamingHybridPilot(replacement,max_confirmed_events=12)
    s=loaded['state']
    restored=HybridFactualState(s['ssm'],s['ak'],s['av'],TTTFactualState(s['tw'],s['n']))
    with torch.no_grad():
        after=adapter.preview(candidates,restored)
    assert torch.equal(original,after)
    assert torch.equal(state.ttt.fast_weights,restored.ttt.fast_weights)


def test_real_factual_gradient_through_all_four_blocks():
    core,stream,events,candidates,state=setup(641,1)
    with torch.no_grad():
        core.depth.router.bias.fill_(2.)
    yes=torch.ones((1,9),dtype=torch.bool)
    _,new_state=stream.ingest_confirmed(events,state,yes)
    predicted=stream.preview(candidates,new_state)
    predicted.square().sum().backward()
    for name,p in [('stem',core.stem.weight),('ssm',core.ssm.candidate.weight),
                   ('attn',core.attn.qkv.weight),('ttt',core.ttt.key.weight),
                   ('depth',core.depth.ffn[0].weight),('head',core.head.weight)]:
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().sum())>0,name


def test_state_already_cold_has_no_learning_or_science_claim():
    _,stream,events,candidates,state=setup()
    assert not isinstance(stream,torch.nn.Module)
    with torch.no_grad():
        _,s=ingest_in_chunks(stream,events,state)
    assert s.ttt.confirmed_count.item()==9
    assert torch.equal(state.ttt.confirmed_count,torch.zeros_like(state.ttt.confirmed_count))
    assert s.ssm_hidden.shape==(1,8)
