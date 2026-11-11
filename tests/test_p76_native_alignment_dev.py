"""P76 source-exact focused engineering; not scientific first-use evidence."""
from __future__ import annotations
import pytest
import torch
from experiments import p76_native_alignment_dev as p76

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def test_fresh_seeded_determinism_and_no_ood_before_qualification():
    for name in ('fit','val','negative'):
        a=p76.make_world(name,8)
        assert p76.source_sha(a)==p76.source_sha(p76.make_world(name,8))
        assert a['facts'].shape==(8,24,32)
        assert a['query'].shape==(8,1,32)
        assert a['label'].shape==(8,16)
        assert torch.equal(a['query'][:,:,16:],torch.zeros_like(a['query'][:,:,16:]))


def test_randomized_roles_and_no_index_shortcut():
    assert p76.audit(p76.make_world('fit'))['pass']
    assert p76.audit(p76.make_world('val'))['pass']


def test_raw_strong_control_exact_retrieval():
    w=p76.make_world('val',16)
    m=p76.original(760201)
    f=p76.features(m,w)
    r=p76.fixed(f,'raw')
    assert r['hit']==1.0 and r['mse']<1e-6
    assert f['positive'].sum(1).tolist()==[2]*16


def test_no_future_effect_or_final_query_in_historical_representation():
    w=p76.make_world('fit',8)
    model=p76.original(760201)
    a=p76.features(model,w)
    changed={k:v.clone() for k,v in w.items()}
    changed['label'].add_(torch.randn_like(changed['label']))
    b=p76.features(model,changed)
    assert torch.equal(a['native_keys'],b['native_keys'])
    # Change the future query; causal historical native keys must remain exact.
    changed['query'][:,:,:16]=changed['facts'][:,0:1,:16]
    c=p76.features(model,changed)
    assert torch.equal(a['native_keys'],c['native_keys'])


def test_source_fails_closed_on_extra_fields_and_bad_count():
    with pytest.raises(ValueError,match='P76_UNKNOWN_SOURCE'):
        p76.make_world('p74_consumed')
    with pytest.raises(ValueError,match='P76_BAD_COUNT'):
        p76.make_world('fit',0)
    w=p76.make_world('fit',8)
    with pytest.raises(ValueError,match='P76_WORLD_SCHEMA'):
        p76.factual({**w,'future_target':w['label']})
    w['query']=torch.ones(8,1,32)
    with pytest.raises(ValueError,match='P76_FACTUAL_SHAPE'):
        p76.factual(w)


def test_distinct_core_seeds_frozen_after_features():
    before=[]
    for seed in p76.MODEL_SEEDS:
        core=p76.original(seed)
        h=p76.model_hash(core)
        p76.features(core,p76.make_world('val',8))
        assert h==p76.model_hash(core)
        before.append(h)
    assert len(set(before))==3


def test_learned_alignment_gradient_and_exact_restore():
    f=p76.features(p76.original(760201),p76.make_world('fit',8))
    m=p76.Alignment(760201)
    k,q=p76.pairs(f,'native')
    loss=p76.contrastive(m.logits(k,q),f['positive'])
    loss.backward()
    assert m.key_map.grad is not None and m.query_map.grad is not None
    assert torch.isfinite(m.key_map.grad).all() and torch.isfinite(m.query_map.grad).all()
    restore=p76.Alignment(760203)
    restore.load_state_dict({k:v.detach().clone() for k,v in m.state_dict().items()})
    assert torch.equal(m.logits(k,q),restore.logits(k,q))
    assert sum(p.numel() for p in m.parameters())==512


def test_wrong_credit_destabilizes_nonprivileged_raw_effect_readout():
    f=p76.features(p76.original(760201),p76.make_world('negative',8))
    good=p76.fixed(f,'raw')
    wrong_effects=f['observed_effects'].flip(1)
    i=(f['raw_keys']*f['raw_query'][:,None]).sum(-1).argmax(1)
    bad=wrong_effects[torch.arange(len(i)),i]
    assert good['mse']<1e-6
    assert float((bad-f['labels']).square().mean())>.01


def test_bad_shapes_and_empty_pos_mask_refused():
    f=p76.features(p76.original(760201),p76.make_world('fit',8))
    m=p76.Alignment(760201)
    with pytest.raises(ValueError,match='P76_BAD_PROJECTION_INPUT'):
        m.logits(f['native_keys'][:,:,:15],f['native_query'])
    k,q=p76.pairs(f,'native')
    with pytest.raises(ValueError,match='P76_BAD_POSITIVE_MASK'):
        p76.contrastive(m.logits(k,q),torch.zeros_like(f['positive']))
