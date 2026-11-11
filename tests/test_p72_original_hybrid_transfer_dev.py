"""P72 source/engineering controls; OOD scoring and authentic science excluded."""
from __future__ import annotations
import hashlib
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import p72_original_hybrid_transfer_dev as p


def _blob(path):
    b=Path(path).read_bytes()
    return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()


def test_exact_original_and_gate_model_blobs():
    root=Path(__file__).resolve().parents[1]
    expected={'model.py':'5d9ea57b827aa1796548447868fa5ee14c251b49',
      'native_p62_streaming_ttt.py':'776e63a0ff9474a860e87088718acb36637a7bec',
      'native_p64k_incremental_hybrid_stream.py':'89c3ef5819ac594ad687c67c579a559754109d4c',
      'native_p71_gated_hybrid_stream.py':'e93ffe29806ac15705cbfd002c5725b9bc4b5e45'}
    for name,sha in expected.items(): assert _blob(root/name)==sha


def test_frozen_source_reproducible_and_roles_hidden():
    w=p.make_world('fit',8)
    assert p.source_hash(w)==p.source_hash(p.make_world('fit',8))
    assert w['facts'].shape==(8,24,32)
    assert w['query'].shape==(8,1,32)
    assert not w['query'].requires_grad
    assert torch.equal(w['query'][:,:,16:],torch.zeros_like(w['query'][:,:,16:]))
    assert w['role_audit'].sum(dim=1).tolist()==[16]*8
    assert p.position_audit(p.make_world('fit'))['pass']
    assert p.position_audit(p.make_world('val'))['pass']
    with pytest.raises(ValueError,match='P72_UNKNOWN_SOURCE'):
        p.make_world('legacy_consumed')


def test_factual_history_contains_measured_effect_but_query_does_not():
    w=p.make_world('val',8)
    target_keys=w['query'][:,0,:16]
    keys=w['facts'][:,:,:16]
    effects=w['facts'][:,:,16:]
    for i in range(8):
        score=((keys[i]-target_keys[i]).abs().amax(dim=-1)<1e-7)
        assert int(score.sum())==2
        assert torch.allclose(effects[i,score],w['label'][i].expand(2,16))


def test_conventional_cache_strict_state_and_finite_output():
    w=p.make_world('val',8)
    z=p.cache_prediction(w)
    assert z.shape==(8,16)
    assert bool(torch.isfinite(z).all())
    assert w['facts'].shape[-1]==32


def test_three_distinct_paired_model_initializations():
    hashes=[]
    for seed in p.MODEL_SEEDS:
        a=p.new_core(seed);b=p.new_core(seed)
        assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))
        hashes.append(p.digest_tensor(a.stem.weight))
    assert len(set(hashes))==3


def test_full_four_block_gradient_and_stable_read_only_preview():
    torch.set_num_threads(1)
    w=p.make_world('fit',2)
    for mode in ('all_write','learned'):
        core=p.new_core(720201)
        z=p.native_predict(core,w,mode)
        assert z.shape==(2,16)
        assert bool(torch.isfinite(z).all())
        p.losses(z,w).mean().backward()
        for name in ('stem','ssm','attn','ttt','depth','head'):
            params=list(getattr(core,name).parameters())
            assert any(q.grad is not None and bool(torch.isfinite(q.grad).all()) for q in params)
