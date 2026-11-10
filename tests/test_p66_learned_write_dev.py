"""P66 no-source regression; no protected authentic first use."""
import hashlib
from pathlib import Path
import pytest
import torch
from experiments import p66_learned_write_dev as p66


def test_source_exact_sizes_independent_seeds_and_one_physical_query():
    allhash=[]
    for name, (seed,n) in p66.SOURCE.items():
        d,sha=p66.source(name)
        assert d['keys'].shape==(n,64,16)
        assert d['executed'].shape==(n,64,1)
        assert d['observed'].shape==(n,64,16)
        assert d['query_action'].shape==(n,1)
        assert d['query_observed'].shape==(n,16)
        assert set(d)=={'keys','executed','observed','query_context','query_action','query_observed'}
        assert len(sha)==64
        allhash.append(sha)
    assert len(set(allhash))==3
    assert p66.source('fit')[1]==allhash[0]
    with pytest.raises(ValueError,match='P66_UNKNOWN_FROZEN_SPLIT'):
        p66.source('unknown')


def test_exact_original_update_one_factual_association():
    fast=torch.zeros((1,16,16))
    key=torch.zeros((1,16));key[0,3]=1
    physical_effect=torch.linspace(-.5,.5,16)[None,:]
    executed=torch.tensor([[-1.]])
    observed=physical_effect*executed
    v=executed*observed
    nxt=p66.fast_update(fast,key,v)
    expected=.5*physical_effect.unsqueeze(-1)*key.unsqueeze(1)
    assert torch.equal(nxt,expected)


def test_physical_action_credit_sham_noninvertible_not_prediction_of_latent():
    d,sha=p66.source('val')
    wrong=p66.replace_actual_credit(d,'val')
    assert torch.equal(wrong['observed'],d['observed'])
    assert torch.equal(wrong['query_action'],d['query_action'])
    assert torch.equal(wrong['query_observed'],d['query_observed'])
    agreement=float((wrong['executed']==d['executed']).float().mean())
    assert .45<agreement<.55
    assert not torch.equal(wrong['executed'],d['executed'])


def test_relevance_gate_learns_only_from_factual_query_outcome_and_has_gradient():
    d,_=p66.source('val')
    x=p66.select(d,torch.arange(4))
    torch.manual_seed(1)
    gate=p66.LearnedGate()
    p,g=gate(x,return_gate=True)
    assert p.shape==(4,16) and g.shape==(4,64)
    assert float(g.min())>0 and float(g.max())<1
    loss=(p-x['query_observed']).square().mean()
    loss.backward()
    assert gate.weights.grad is not None
    assert float(gate.weights.grad.abs().sum())>0
    assert len(list(gate.parameters()))==1 and gate.weights.numel()==4


def test_classical_controls_share_actual_facts_and_query():
    d,_=p66.source('val')
    x=p66.select(d,torch.arange(5))
    for mode in ('all','surprise','norm','fifo','oracle_phase','random','zero'):
        p=p66.fixed_memory(x,mode,threshold=.2,seed=1)
        assert p.shape==(5,16) and torch.isfinite(p).all()
    assert p66.ConventionalGRU()(x).shape==(5,16)
    with pytest.raises(ValueError,match='P66_BAD_CONTROL'):
        p66.fixed_memory(x,'bad')


def test_no_future_query_is_visible_in_writes_and_query_action_sign_flips_readout():
    d,_=p66.source('val')
    a=p66.select(d,torch.arange(5))
    b=dict(a)
    b['query_action']=-a['query_action']
    b['query_observed']=-a['query_observed']
    torch.manual_seed(10)
    m=p66.LearnedGate().eval()
    with torch.no_grad():
        first,ga=m(a,return_gate=True)
        second,gb=m(b,return_gate=True)
    assert torch.equal(ga,gb)
    assert torch.equal(first,-second)
    assert torch.equal(a['executed'],b['executed'])
