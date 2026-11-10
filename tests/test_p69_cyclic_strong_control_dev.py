"""P69 engineering controls for genuinely cyclic 256-float cache."""
import importlib.util
from pathlib import Path
import pytest
import torch

path=Path(__file__).resolve().parents[1]/'experiments'/'p69_cyclic_strong_control_dev.py'
spec=importlib.util.spec_from_file_location('p69',path)
p69=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p69)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def facts(indices):
    keys=torch.eye(16)[indices].unsqueeze(0)
    return {'action':keys,'effect':keys.clone()*.75}


def test_frozen_source_is_distinct_and_deterministic():
    for s in ['fit','val','ood_dev']:
        w=p69.make_world(s,8)
        assert p69.p68.source_hash(w)==p69.p68.source_hash(p69.make_world(s,8))
        assert set(p69.p68.factual(w))=={'action','effect'}
    assert p69.p68.source_hash(p69.make_world('fit',8))!=p69.p68.source_hash(p69.make_world('val',8))


def test_eight_distinct_inserts_and_ninth_wraps():
    key,val=p69.cache_stream(facts(list(range(8))))
    assert key.numel()+val.numel()==256
    assert (key.norm(dim=-1)>1.5).sum().item()==1
    assert torch.isclose(key[0,7].norm(),torch.tensor(2.))
    key,val=p69.cache_stream(facts(list(range(9))))
    assert torch.isclose(key[0,0].norm(),torch.tensor(2.))
    assert torch.isclose(key[0,7].norm(),torch.tensor(1.))
    assert torch.allclose(p69.p68.cache_predict((key,val),torch.eye(16)[8:9]),torch.eye(16)[8:9]*.75)


def test_hit_does_not_move_pointer_or_evict():
    indices=list(range(8))+[3,4]
    k,v=p69.cache_stream(facts(indices))
    assert torch.isclose(k[0,7].norm(),torch.tensor(2.))
    assert torch.equal(p69.p68.cache_predict((k,v),torch.eye(16)[3:4]),torch.eye(16)[3:4]*.75)


def test_restore_across_wrap_no_hidden_counter():
    w=p69.make_world('val',8)
    assert p69.cold_restore(w)
    w2=facts(list(range(16)))
    direct=p69.cache_stream(w2)
    resumed=p69.cache_stream(w2,p69.cache_stream(w2,hi=6),lo=6)
    assert all(torch.equal(x,y) for x,y in zip(direct,resumed))


def test_invalid_state_and_future_query_refused():
    a=facts([0,1])
    with pytest.raises(ValueError):p69.cache_stream({**a,'future_query':torch.zeros(1,16)})
    with pytest.raises(ValueError):p69.cache_stream(a,(torch.ones(1,8,16),torch.ones(1,8,16)))


def test_index_only_shortcut_and_256_budget():
    w=p69.make_world('fit',32)
    audit=p69.p68.positional_audit(w['role_audit'])
    assert audit['pass']
    k,v=p69.cache_stream(p69.p68.factual(w))
    assert k.numel()+v.numel()==32*256


def test_cache_prediction_not_using_target_label():
    w=p69.make_world('val',8)
    c=p69.cache_stream(p69.p68.factual(w))
    w['observed_query_effect']=-w['observed_query_effect']
    c2=p69.cache_stream(p69.p68.factual(w))
    assert all(torch.equal(a,b) for a,b in zip(c,c2))


def test_wrong_action_raises_error_against_true_action():
    w=p69.make_world('fit',8)
    model=p69.p68.WriteGate(990201)
    assert not torch.equal(model(p69.p68.factual(w)),model(p69.p68.factual(w),wrong=w['wrong_credit_key']))


def test_orthogonal_nuisance_allwrite_preserves_readout():
    assert p69.ortho_negative()<=1e-5
