"""P94 engineering-only native factual credit and cold-custody invariants."""
from pathlib import Path
import pytest
import torch
from native_p94_confirmed_causal_memory import ConfirmedCausalMemory, AnchoredHybridPilot
from experiments import p86_portable_causal_preflight as p86

torch.set_num_threads(1)

def fixture(batch=4,n=48,dim=16):
    g=torch.Generator().manual_seed(940111)
    actions=torch.randn(batch,n+1,dim,generator=g)
    true_w=torch.randn(batch,dim,dim,generator=g)*0.02
    effects=torch.bmm(true_w,actions[:,:n].transpose(1,2)).transpose(1,2)
    return torch.cat((actions[:,:n],effects),dim=-1),actions[:,-1]

def reference(h,q):
    from torch.nn.functional import normalize
    b,t,doubled=h.shape
    d=doubled//2
    w=torch.zeros(b,d,d)
    for t in range(t):
        k=normalize(h[:,t,:d],dim=-1)
        residual=h[:,t,d:]-torch.bmm(w,k.unsqueeze(-1)).squeeze(-1)
        w=.99*w+residual.unsqueeze(-1)*k.unsqueeze(1)
    return torch.bmm(w,normalize(q,dim=-1).unsqueeze(-1)).squeeze(-1),w

def test_original_p86_custody():
    root=Path(__file__).resolve().parents[1]
    assert p86.blob(root/"model.py")=="5d9ea57b827aa1796548447868fa5ee14c251b49"
    assert p86.blob(root/"experiments/p86_portable_causal_preflight.py")=="008492665ee9090db5791963f2e2336ea283ee62"

def test_direct_reference_parity():
    h,q=fixture()
    got,state=ConfirmedCausalMemory().forward_episode(h,q)
    expected,w=reference(h,q)
    assert torch.equal(got,expected)
    assert torch.equal(state.fast_weights,w)
    assert torch.equal(state.confirmed_count,torch.full((4,),48,dtype=torch.int64))

def test_exact_p86_existing_cal_fixture_parity():
    # Existing consumed CAL used strictly as engineering fixture, never as novel VAL.
    old=p86.source("CAL",n=8)
    got,state=ConfirmedCausalMemory().forward_episode(old["history"],old["query"][:,:16])
    ref,refstate=p86.online(old,1.,.99)
    assert torch.equal(got,ref)
    assert torch.equal(state.fast_weights,refstate)
    assert state.fast_weights.numel()==8*256

def test_bitwise_cold_restoration_and_immutable_input():
    h,q=fixture();m=ConfirmedCausalMemory()
    start=m.initial_state(q)
    first=m.ingest_confirmed(h[:,:24,:16],h[:,:24,16:],
                              torch.ones(4,24,dtype=torch.bool),start)
    snapshot=first.snapshot()
    final=m.ingest_confirmed(h[:,24:,:16],h[:,24:,16:],
                              torch.ones(4,24,dtype=torch.bool),snapshot)
    whole=m.forward_episode(h,q)[1]
    assert torch.equal(final.fast_weights,whole.fast_weights)
    assert torch.equal(final.confirmed_count,whole.confirmed_count)
    assert torch.equal(m.read_only(q,final),m.read_only(q,whole))
    assert not start.fast_weights.any()
    assert torch.equal(first.fast_weights,snapshot.fast_weights)

def test_unconfirmed_outcomes_nan_sentinel_never_leak():
    h,q=fixture();m=ConfirmedCausalMemory()
    mask=torch.ones(4,48,dtype=torch.bool);mask[:,::3]=False
    altered=h.clone();altered[:,::3,16:]=float("nan")
    a,sa=m.forward_episode(h,q,mask)
    b,sb=m.forward_episode(altered,q,mask)
    assert torch.equal(a,b) and torch.equal(sa.fast_weights,sb.fast_weights)
    assert torch.equal(sa.confirmed_count,torch.full((4,),32,dtype=torch.int64))
    zeros=torch.zeros_like(mask)
    pred,state=m.forward_episode(altered,q,zeros)
    assert not pred.any() and not state.fast_weights.any()
    assert not state.confirmed_count.any()

def test_queries_do_not_write_and_no_cross_episode_leak():
    h,q=fixture();m=ConfirmedCausalMemory()
    _,a=m.forward_episode(h,q)
    _,b=m.forward_episode(h,q+1.25)
    assert torch.equal(a.fast_weights,b.fast_weights)
    one,st=m.forward_episode(h,q)
    _=m.forward_episode(h*2,q*2)
    again,st2=m.forward_episode(h,q)
    assert torch.equal(one,again) and torch.equal(st.fast_weights,st2.fast_weights)

def test_wrong_action_credit_changes_prediction():
    h,q=fixture(batch=8);m=ConfirmedCausalMemory()
    good,_=m.forward_episode(h,q)
    changed=h.clone();changed[:,:,16:]=h[:,:,16:].roll(11,dims=1)
    wrong,_=m.forward_episode(changed,q)
    assert float((good-wrong).square().mean())>1e-4

def test_native_hybrid_residual_initializes_to_exact_baseline():
    h,q=fixture(batch=2,n=6)
    model=AnchoredHybridPilot().eval()
    with torch.no_grad():
        raw,_=model.factual.forward_episode(h,q)
        default=model(h,q)
        gated=model(h,q,use_residual=True)
    assert torch.equal(default,raw)
    assert torch.equal(gated,raw)
    assert model.residual_scale.item()==0.
    # The residual is a separately trainable hypothesis, not an accepted gain.
    result=model(h,q,use_residual=True).square().mean()
    result.backward()
    assert model.residual_scale.grad is not None

def test_fail_closed_invalid_shapes_and_unverified_facts():
    h,q=fixture();m=ConfirmedCausalMemory()
    mask=torch.ones(4,48,dtype=torch.bool)
    nan=h.clone();nan[:,0,16]=float("nan")
    with pytest.raises(ValueError,match="P94_NONFINITE_CONFIRMED_EFFECT"):
        m.forward_episode(nan,q,mask)
    with pytest.raises(ValueError,match="P94_INVALID_ACTIONS"):
        m.forward_episode(h,q.double())
    with pytest.raises(ValueError,match="P94_INVALID_FACTS"):
        m.forward_episode(h,q,mask.int())
    with pytest.raises(ValueError,match="P94_INVALID_HISTORY"):
        m.forward_episode(h[:,:,:30],q)
