"""Source-free and FIT/VAL-only checks, no protected first-use; P67 exact-source."""
import torch
import pytest
from experiments import p67_clockless_interleaved_dev as p
from experiments import p66_learned_write_dev as old


def test_fresh_source_custody_and_no_oracle_metadata():
    a,sha=p.source('fit'); b,other=p.source('val')
    assert sha!=other and sha==p.source('fit')[1]
    assert a['keys'].shape==(768,64,16)
    assert b['keys'].shape==(192,64,16)
    assert set(a)=={'keys','executed','observed','query_context','query_action','query_observed'}
    with pytest.raises(ValueError,match='P67_UNREGISTERED_SOURCE'):
        p.source('future')


def test_random_interleaving_genuinely_breaks_global_clock_and_keeps_factual_alignment():
    d,_=p.source('val')
    # The first 32 positions no longer monopolize repeated target keys.
    # Every target query has four exact context matches after interleaving.
    qsim=torch.einsum('btd,bd->bt',d['keys'],d['query_context'])
    hits=(qsim>0.9999)
    assert hits.sum(1).tolist()==[4]*len(qsim)
    assert bool(hits[:,:32].any()) and bool(hits[:,32:].any())
    # At least some worlds have their final query-key visit early AND late.
    assert bool((hits[:,:16].sum(1)==0).any())
    assert bool((hits[:,48:].sum(1)==0).any())


def test_no_clock_feature_and_query_is_pure_read_only():
    x,_=p.source('val'); d=old.select(x,torch.arange(4))
    torch.manual_seed(331)
    m=p.ClocklessGate()
    a,ga=m(d,return_gate=True)
    changed=dict(d); changed['query_action']=-d['query_action']
    changed['query_observed']=-d['query_observed']
    b,gb=m(changed,return_gate=True)
    assert torch.equal(ga,gb)
    assert torch.equal(a,-b)
    assert m.weights.numel()==3 and ga.shape==(4,64)
    loss=(a-d['query_observed']).square().mean()
    loss.backward()
    assert m.weights.grad is not None and float(m.weights.grad.abs().sum())>0


def test_historical_credit_destroyed_not_cyclic_shift():
    x,_=p.source('val')
    fake=p.wrong_credit(x,'val')
    assert torch.equal(fake['keys'],x['keys'])
    assert torch.equal(fake['observed'],x['observed'])
    assert torch.equal(fake['query_observed'],x['query_observed'])
    fraction=float((fake['executed']==x['executed']).float().mean())
    assert .45<fraction<.55
    assert not any(torch.equal(fake['executed'],x['executed'].roll(j,1)) for j in range(64))


def test_explicit_unmatched_capacity_classical_baseline_and_conventional_fit():
    x,_=p.source('val'); d=old.select(x,torch.arange(4))
    v=p.all_history_exact_context(d)
    assert v.shape==(4,16) and torch.isfinite(v).all()
    classical=old.fixed_memory(d,'fifo')
    assert classical.shape==(4,16)
    assert p.StrongGRU()(d).shape==(4,16)
    assert 64*(16+16)>256


def test_original_p66_source_not_modified_by_p67_generation():
    import hashlib
    from pathlib import Path
    raw=(Path(__file__).resolve().parents[1]/'experiments/p66_learned_write_dev.py').read_bytes()
    sha=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    assert sha=='c26a100d139f5e8409156fbebaba81b4fbef4f1c'
    assert old.SOURCE['fit'][0]==966101
    p.source('fit')
    assert old.SOURCE['fit'][0]==966101
