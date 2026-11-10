"""P64K-R1: preserve the parent's no-factual-evidence preview refusal."""
import hashlib
from pathlib import Path
import pytest
import torch
from model import HybridPilot, Configuration
from native_p64s_full_hybrid_stream import FullHybridFactualStream
from native_p64k_incremental_hybrid_stream import StreamingHybridPilot, HybridFactualState
from native_p62_streaming_ttt import TTTFactualState


def fixture(batch=2):
    torch.manual_seed(681)
    core = HybridPilot(5, 2, Configuration(width=8, heads=2, ssm=True,
                        attention=True, ttt=True, depth=True)).eval()
    parent = FullHybridFactualStream(core, max_confirmed=10)
    candidate = StreamingHybridPilot(core, max_confirmed_events=10)
    facts = torch.randn(batch, 4, 5)
    queries = torch.randn(batch, 2, 5)
    return core, parent, candidate, facts, queries


def snapshot(state):
    return tuple(t.clone() for t in (state.ssm_hidden, state.attention_k,
                        state.attention_v, state.ttt.fast_weights, state.ttt.confirmed_count))


def test_parent_and_incremental_both_fail_closed_before_any_confirmed_fact():
    core, parent, candidate, facts, queries = fixture()
    s0 = parent.initial_state(facts[:, 0])
    t0 = candidate.initial_state(facts[:, 0])
    old = snapshot(t0)
    weights = {k:v.detach().clone() for k,v in core.state_dict().items()}
    with pytest.raises(ValueError,match='P64S_NO_CONFIRMED_HISTORY'):
        parent.preview(queries, s0)
    with pytest.raises(ValueError,match='P64S_NO_CONFIRMED_HISTORY'):
        candidate.preview(queries, t0)
    assert all(torch.equal(x,y) for x,y in zip(snapshot(t0),old))
    assert all(torch.equal(v,weights[k]) for k,v in core.state_dict().items())


def test_confirmed_single_actual_event_allows_same_state_preview_with_original_parity():
    core, _, candidate, facts, queries=fixture()
    initial = candidate.initial_state(facts[:, 0])
    with torch.no_grad():
        _, state=candidate.ingest_confirmed(facts[:,:1], initial,
                                            torch.ones((2,1),dtype=torch.bool))
        old=snapshot(state)
        expected=torch.cat([core(torch.cat((facts[:,:1],queries[:,i:i+1]),dim=1))[:,-1:]
                            for i in range(2)],dim=1)
        output=candidate.preview(queries,state)
        reverse=candidate.preview(queries.flip(1),state).flip(1)
    torch.testing.assert_close(output,expected,atol=1e-6,rtol=1e-6)
    assert torch.equal(output,reverse)
    assert all(torch.equal(x,y) for x,y in zip(snapshot(state),old))
    assert torch.equal(initial.ttt.confirmed_count,torch.zeros_like(initial.ttt.confirmed_count))


def test_zero_history_branch_preserves_parent_model_source_and_cold_state_atomicity():
    root=Path(__file__).resolve().parents[1]
    for path,sha in [('model.py','5d9ea57b827aa1796548447868fa5ee14c251b49'),
        ('native_p62_streaming_ttt.py','776e63a0ff9474a860e87088718acb36637a7bec'),
        ('native_p64s_full_hybrid_stream.py','56f4d566db9375e4b27e7d9eabcaf67498c25d94')]:
        b=(root/path).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==sha
    _,_,candidate,facts,queries=fixture()
    zero=candidate.initial_state(facts[:,0])
    assert all(torch.equal(z,z.clone()) for z in snapshot(zero))
    corrupt=HybridFactualState(zero.ssm_hidden,zero.attention_k,zero.attention_v,
        TTTFactualState(zero.ttt.fast_weights,torch.tensor([0,1],dtype=torch.long)))
    with pytest.raises(ValueError,match='BAD_CONFIRMED_COUNT'):
        candidate.preview(queries,corrupt)
    with pytest.raises(ValueError,match='P64S_NO_CONFIRMED_HISTORY'):
        candidate.preview(queries,zero)
