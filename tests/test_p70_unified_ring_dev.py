"""P70 source-exact integration checks, not scientific or OOD outcome qualification."""
from pathlib import Path
import importlib.util
import torch
import pytest

SOURCE=Path(__file__).resolve().parents[1]/'experiments'/'p70_unified_ring_dev.py'
spec=importlib.util.spec_from_file_location('p70',SOURCE)
p70=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p70)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def test_parent_exact_blobs_and_one_ring_backend():
    assert p70.git_blob_bytes(p70.BASE.read_bytes())=='7db52d3a150903b2feff661c9562017ee24c0dfe'
    assert p70.git_blob_bytes(p70.RING_SOURCE.read_bytes())=='bf12bd3525230ecf58ecc43837f67fdf23578f35'
    assert p70.cache_stream.__code__.co_names.count('ring') == 1


def test_factual_only_and_independent_source_family():
    a=p70.make_world('fit',8)
    assert set(p70.p68.factual(a))=={'action','effect'}
    assert p70.p68.source_hash(a)==p70.p68.source_hash(p70.make_world('fit',8))
    assert p70.p68.source_hash(a)!=p70.p68.source_hash(p70.make_world('val',8))
    with pytest.raises(ValueError,match='P70_FACTUAL_ONLY_REQUIRED'):
        p70.cache_stream({**p70.p68.factual(a),'query_action':a['query_action']})


def test_ring_true_256_float_and_10_unique_misses():
    k=torch.eye(16)[:10][None]
    e={'action':k,'effect':k.clone()}
    state=p70.cache_stream(e)
    assert sum(x.numel() for x in state)==256
    assert sorted(int(x.argmax()) for x in state[0][0])==list(range(2,10))
    assert int((state[0][0].norm(dim=-1)>1.5).sum())==1
    assert torch.equal(p70.ring.predict(state,k[:,0]),torch.zeros(1,16))


def test_ring_hit_no_advance_exact_cold_resume():
    w=p70.make_world('fit',8)
    f=p70.p68.factual(w)
    assert p70.cold_restore(w)
    continuous=p70.cache_stream(f)
    midpoint=p70.cache_stream(f,hi=32)
    again=p70.cache_stream(f,carry=tuple(x.detach().clone() for x in midpoint),lo=32)
    assert all(torch.equal(x,y) for x,y in zip(again,continuous))
    before=tuple(x.clone() for x in again)
    _=p70.ring.predict(again,w['query_action'])
    assert all(torch.equal(x,y) for x,y in zip(before,again))


def test_shortcut_checks_fit_val_before_ood_scoring():
    for name in ('fit','val'):
        world=p70.make_world(name)
        audit=p70.p68.positional_audit(world['role_audit'])
        assert audit['pass'], audit


def test_zero_allwrite_frequency_ring_are_finite_and_same_world():
    w=p70.make_world('val',8)
    control=p70.baseline_errors(w)
    assert set(control)=={'zero','all_write','frequency_cache','pr177_ring_cache'}
    assert all(torch.isfinite(v).all() and v.shape==(8,) for v in control.values())
    f=p70.p68.factual(w)
    freq=p70.p68.cache_stream(f)
    assert sum(t[0].numel() for t in freq)==256
    assert p70.p68.WriteGate(1000201)(f).shape==(8,16,16)


def test_three_distinct_seeds_and_gradients():
    hashes=[p70.p68.digest(p70.p68.WriteGate(s).theta) for s in p70.MODEL_SEEDS]
    assert len(set(hashes))==3
    w=p70.make_world('fit',8)
    m=p70.p68.WriteGate(p70.MODEL_SEEDS[0])
    pred=p70.p68.fast_predict(m(p70.p68.factual(w)),w['query_action'])
    p70.p68.errors(pred,w).mean().backward()
    assert bool(torch.isfinite(m.theta.grad).all())


def test_wrong_action_credit_and_negative_sanity():
    w=p70.make_world('negative',4)
    m=p70.p68.WriteGate(p70.MODEL_SEEDS[0])
    good=m(p70.p68.factual(w))
    wrong=m(p70.p68.factual(w),wrong=w['wrong_credit_key'])
    assert not torch.equal(good,wrong)


def test_failure_closed_invalid_carry_no_external_cursor():
    w=p70.make_world('val',4)
    f=p70.p68.factual(w)
    state=p70.cache_stream(f)
    damaged=(state[0].clone(),state[1].clone())
    damaged[0][0,0]*=2
    with pytest.raises(ValueError,match='P69_CORRUPT_CARRY'):
        p70.cache_stream(f,carry=damaged)
    with pytest.raises(ValueError,match='P70_BAD_RANGE'):
        p70.cache_stream(f,lo=-1)
