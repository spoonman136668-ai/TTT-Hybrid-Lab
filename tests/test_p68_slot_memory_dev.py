"""P68 engineering tests; no sealed scientific source or first use."""
import importlib.util
from pathlib import Path
import torch
import pytest

spec=importlib.util.spec_from_file_location('p68',Path(__file__).resolve().parents[1]/'experiments'/'p68_slot_memory_dev.py')
p68=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p68)
torch.set_num_threads(1)


def test_source_repeatable_factual_only():
    a=p68.make_world('fit',8)
    assert p68.source_hash(a)==p68.source_hash(p68.make_world('fit',8))
    assert set(p68.factual(a))=={'action','effect'}
    assert 'query_action' not in p68.factual(a)


def test_256_state_and_query_unavailable():
    a=p68.make_world('val',8)
    c=p68.cache_stream(p68.factual(a))
    assert sum(v[0].numel() for v in c)==256
    assert p68.WriteGate(980201)(p68.factual(a)).shape==(8,16,16)
    assert torch.isfinite(p68.cache_predict(c,a['query_action'])).all()


def test_positives_are_interleaved():
    for k in ['fit','val','ood_dev']:
        a=p68.make_world(k,32)
        assert p68.positional_audit(a['role_audit'])['pass']


def test_cache_hit_updates_and_reads():
    k=torch.eye(16)[:8]
    facts={'action':k[None].repeat(2,1,1).repeat(1,3,1),
           'effect':(k*.5)[None].repeat(2,1,1).repeat(1,3,1)}
    c=p68.cache_stream(facts)
    z=p68.cache_predict(c,torch.eye(16)[:2])
    assert torch.allclose(z,torch.eye(16)[:2]*.5,atol=1e-6)


def test_new_singletons_not_mistaken_for_hits():
    facts={'action':torch.eye(16)[None].repeat(2,1,1),
           'effect':torch.eye(16)[None].repeat(2,1,1)}
    c=p68.cache_stream(facts)
    z=p68.cache_predict(c,torch.eye(16)[12:14])
    assert torch.equal(z,torch.zeros_like(z))


def test_gate_gradient_and_distinct_seeds():
    a=p68.make_world('fit',8)
    m=p68.WriteGate(980201)
    pred=p68.fast_predict(m(p68.factual(a)),a['query_action'])
    p68.errors(pred,a).mean().backward()
    assert torch.isfinite(m.theta.grad).all()
    assert len({p68.digest(p68.WriteGate(s).theta) for s in p68.MODEL_SEEDS})==3


def test_wrong_credit_input_guard():
    a=p68.make_world('fit',4)
    m=p68.WriteGate(980201)
    incorrect=m(p68.factual(a),wrong=a['wrong_credit_key'])
    assert not torch.equal(incorrect,m(p68.factual(a)))
    with pytest.raises(ValueError):m({'action':torch.ones(4,3,17),'effect':torch.ones(4,3,17)})


def test_cold_restore_bitwise():
    a=p68.make_world('val',8)
    assert p68.exact_restore(p68.WriteGate(980201),a)


def test_preflight_budget_and_baseline_finite():
    a=p68.make_world('fit',8)
    b=p68.baseline_eval(a)
    assert set(b)=={'zero','all_write','frequency_cache','cyclic_cache'}
    assert all(torch.isfinite(x).all() for x in b.values())
