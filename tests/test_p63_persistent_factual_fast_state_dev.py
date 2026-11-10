"""P63 isolated factual-only fast-state DEV invariants."""
import hashlib
import io
from pathlib import Path
import torch
import pytest
from experiments import p63_persistent_factual_fast_state_dev as p63


def test_source_separation_and_hidden_inputs_excluded():
    with pytest.raises(ValueError,match='P63_UNKNOWN_SOURCE'):
        p63.source('anything_else')
    a,sha=p63.source('fit')
    b,sha2=p63.source('ood')
    assert sha!=sha2
    assert a['executed_action'].shape==(512,32)
    assert b['real_observed_delta'].shape==(128,32)
    assert 'hidden_evaluator_sign' in a
    assert 'hidden_evaluator_sign' not in p63.allowed(a)


def test_fast_read_pure_and_only_actual_observation_updates():
    model=p63.FastState()
    state=torch.zeros((2,2))
    actions=torch.tensor([1.,-1.])
    before=state.clone()
    assert model.read(state,actions).tolist()==[0.,0.]
    assert torch.equal(state,before)
    nxt=model.observe(state,actions,torch.tensor([.3,-.2]))
    assert not torch.equal(nxt,state)
    assert torch.equal(state,before)


def test_invalid_fact_denied():
    model=p63.FastState()
    s=torch.zeros((2,2))
    with pytest.raises(ValueError,match='P63_INVALID_EXECUTED_ACTION'):
        model.observe(s,torch.tensor([0.,1.]),torch.ones(2))
    with pytest.raises(ValueError,match='P63_NONFINITE_FACT'):
        model.observe(s,torch.ones(2),torch.tensor([float('nan'),1.]))


def test_cold_state_restore_matches_full_replay_without_past_facts():
    model=p63.FastState()
    source,_=p63.source('val')
    ok,sha=p63.cold_resume(model,source)
    assert ok and len(sha)==64


def test_wrong_credit_independent_not_invertible():
    d,_=p63.source('fit')
    a=d['executed_action']; fake=d['wrong_action_credit']
    agreement=float((a==fake).float().mean())
    assert .45<agreement<.55
    assert not any(torch.equal(a,torch.roll(fake,j,1)) for j in range(32))


def test_gradient_through_real_factual_online_update():
    torch.manual_seed(5)
    d=p63.allowed(p63.source('val')[0],torch.arange(4))
    m=p63.FastState()
    p,_=m(d)
    loss=(p[:,4:]-d['real_observed_delta'][:,4:]).square().mean()
    loss.backward()
    assert m.logit_lr.grad is not None and abs(float(m.logit_lr.grad))>0
    assert m.logit_decay.grad is not None and torch.isfinite(m.logit_decay.grad)


def test_one_strong_regression_uses_past_only():
    d=p63.allowed(p63.source('val')[0],torch.arange(8))
    original=p63.sliding_ridge(d)
    changed={k:v.clone() for k,v in d.items()}
    changed['real_observed_delta'][:,-1]+=1000
    later=p63.sliding_ridge(changed)
    assert torch.equal(original,later)


def test_no_retraining_for_hypothetical_read():
    m=p63.FastState()
    state=torch.tensor([[.25,.10]])
    w={k:v.clone() for k,v in m.state_dict().items()}
    p=m.read(state,torch.tensor([1.]))
    q=m.read(state,torch.tensor([-1.]))
    assert p.item()>.3 and q.item()<.2
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in w.items())
