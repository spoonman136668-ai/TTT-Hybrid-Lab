"""P64-K R1: preserve original P64-S no-fact abstention; only local manual tensor DEV."""
import torch
import pytest
from model import Configuration, HybridPilot
from native_p64s_full_hybrid_stream import FullHybridFactualStream
from native_p64k_incremental_hybrid_stream import StreamingHybridPilot


def two_streams():
    torch.manual_seed(64091)
    core=HybridPilot(5,2,Configuration(width=8,heads=2,ssm=True,attention=True,ttt=True,depth=True)).eval()
    return core, FullHybridFactualStream(core,max_confirmed=8),StreamingHybridPilot(core,max_confirmed_events=8)


def test_zero_confirmed_facts_abstains_identically_to_parent():
    core,parent,incremental=two_streams()
    template=torch.zeros((2,5))
    candidates=torch.randn((2,2,5))
    original_weights={k:v.detach().clone() for k,v in core.state_dict().items()}
    original_util=core.depth.last_utilization
    a=parent.initial_state(template)
    b=incremental.initial_state(template)
    with pytest.raises(ValueError,match='P64S_NO_CONFIRMED_HISTORY'):
        parent.preview(candidates,a)
    with pytest.raises(ValueError,match='P64K_NO_CONFIRMED_FACTUAL_HISTORY'):
        incremental.preview(candidates,b)
    assert torch.equal(b.ttt.confirmed_count,torch.zeros_like(b.ttt.confirmed_count))
    assert b.attention_k.shape[2]==0
    assert core.depth.last_utilization==original_util
    assert all(torch.equal(v,original_weights[k]) for k,v in core.state_dict().items())


def test_one_authenticated_fact_allows_pure_equal_original_preview():
    core,parent,incremental=two_streams()
    template=torch.zeros((2,5))
    fact=torch.randn((2,1,5))
    candidates=torch.randn((2,3,5))
    good=torch.ones((2,1),dtype=torch.bool)
    with torch.no_grad():
        _,a=parent.ingest_confirmed(fact,parent.initial_state(template),good)
        _,b=incremental.ingest_confirmed(fact,incremental.initial_state(template),good)
        before=(b.ssm_hidden.clone(),b.attention_k.clone(),b.attention_v.clone(),
                b.ttt.fast_weights.clone(),b.ttt.confirmed_count.clone())
        x=parent.preview(candidates,a)
        y=incremental.preview(candidates,b)
        z=incremental.preview(candidates.flip(1),b).flip(1)
    torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(y,z,atol=1e-6,rtol=1e-6)
    assert all(torch.equal(old,new) for old,new in zip(before,(b.ssm_hidden,b.attention_k,
         b.attention_v,b.ttt.fast_weights,b.ttt.confirmed_count)))


def test_empty_batch_cannot_gain_readiness_from_hypothetical_previews():
    _,_,incremental=two_streams()
    template=torch.zeros((1,5))
    initial=incremental.initial_state(template)
    for amount in (1,2,8,16):
        candidate=torch.randn((1,amount,5))
        with pytest.raises(ValueError,match='P64K_NO_CONFIRMED_FACTUAL_HISTORY'):
            incremental.preview(candidate,initial)
    assert initial.ttt.confirmed_count.item()==0
