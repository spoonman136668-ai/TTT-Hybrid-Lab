"""Focused P69 protocol-implementation checks; do not score consumed P68 OOD."""
import pytest
import torch
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location(
    "p69_ring_cache", Path(__file__).resolve().parents[1] / "experiments" / "p69_ring_cache.py"
)
p69 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p69)
stream, predict = p69.stream, p69.predict


def facts(keys):
    return {'action': keys, 'effect': keys.clone()}


def ids(state):
    return sorted(int(torch.argmax(k)) for k in state[0][0])


def test_true_round_robin_beyond_capacity_and_no_extra_state():
    keys=torch.eye(16)[:12][None]
    state=stream(facts(keys[:,:10]))
    assert ids(state) == list(range(2,10))
    assert sum(a.numel() for a in state) == 256
    assert ids(stream(facts(keys[:,10:]), state)) == list(range(4,12))


def test_hits_do_not_advance_cursor_or_reinsert_key():
    keys=torch.eye(16)[:10][None]
    state=stream(facts(keys[:,:8]))
    updated=stream(facts(keys[:,0:1]),state)
    assert ids(updated) == list(range(8))
    state=stream(facts(keys[:,8:9]),updated)
    assert ids(state) == list(range(1,9))
    state=stream(facts(keys[:,9:10]),state)
    assert ids(state) == list(range(2,10))


def test_continuous_equals_chunked_and_cold_restored_bitwise():
    x=torch.eye(16)[:12][None].repeat(3,1,1)
    direct=stream(facts(x))
    interim=stream(facts(x[:,:7]))
    restored=tuple(t.detach().clone() for t in interim)
    chunked=stream(facts(x[:,7:]), restored)
    assert all(torch.equal(a,b) for a,b in zip(direct,chunked))
    assert torch.equal(predict(direct,x[:,5]),predict(chunked,x[:,5]))
    assert all(torch.equal(a,b) for a,b in zip(interim,restored))  # no in-place carry mutation


def test_read_only_and_factual_only_no_query_leak():
    x=torch.eye(16)[:8][None]
    state=stream(facts(x))
    before=tuple(t.clone() for t in state)
    assert torch.allclose(predict(state,x[:,2]),x[:,2])
    assert torch.equal(predict(state,torch.eye(16)[8:9]),torch.zeros(1,16))
    assert all(torch.equal(a,b) for a,b in zip(state,before))
    with pytest.raises(ValueError,match='P69_FACTUAL_ONLY_REQUIRED'):
        stream({'action':x,'effect':x,'future_query':x[:,0]})


def test_bad_carry_rejected_without_fallback():
    x=torch.eye(16)[:8][None]
    state=stream(facts(x))
    corrupted=(state[0].clone(), state[1].clone())
    corrupted[0][0,0] *= 2
    with pytest.raises(ValueError,match='P69_CORRUPT_CARRY'):
        stream(facts(x), corrupted)
    with pytest.raises(ValueError,match='P69_ZERO_ACTION'):
        stream({'action':torch.zeros(1,2,16),'effect':torch.zeros(1,2,16)})


def test_two_streams_independent_and_still_256_each():
    x=torch.eye(16)[:12][None]
    y=torch.flip(x,dims=[1])
    all_events={'action':torch.cat((x,y),0),'effect':torch.cat((x,y),0)}
    state=stream(all_events)
    assert sum(t.numel() for t in state)==512
    assert ids((state[0][:1],state[1][:1]))==list(range(4,12))
    assert ids((state[0][1:],state[1][1:]))==list(range(8))


def test_reference_fifo_randomized_hits_and_misses():
    """Independent toy FIFO reference; not a scored experiment or P68 reuse."""
    g=torch.Generator().manual_seed(306911)
    keys=torch.eye(16)[:14]
    for _ in range(7):
        seq=torch.randint(0,14,(93,),generator=g).tolist()
        slots=[None]*8
        values=[None]*8
        last=None
        state=None
        for action_id in seq:
            x=keys[action_id:action_id+1][None]
            state=stream(facts(x),state)
            if action_id in slots:
                at=slots.index(action_id)
                values[at]=action_id
            else:
                if None in slots: at=slots.index(None)
                else: at=(last+1)%8
                slots[at]=action_id
                values[at]=action_id
                last=at
            actual=[int(torch.argmax(k)) if torch.count_nonzero(k) else None for k in state[0][0]]
            assert actual==slots
            actual_values=[int(torch.argmax(v)) if torch.count_nonzero(v) else None for v in state[1][0]]
            assert actual_values==values
            assert sum(t.numel() for t in state)==256
            markers=(state[0][0].norm(dim=-1)>1.5).nonzero().view(-1).tolist()
            assert markers==[last]
