"""P78 synthetic DEV: exact-source P62 native TTT replacement and all negatives."""
import copy, hashlib
from pathlib import Path
import pytest
import torch
from native_p78_bounded_raw_ttt import BoundedRawTTT, RawTTTState, POLICY_TAG
from native_p62_streaming_ttt import TTTFactualState
from experiments import p78_internal_identity_dev as p78

torch.set_num_threads(1)


def blob(path):
    b=path.read_bytes()
    return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\x00'+b).hexdigest()


def test_exact_original_project_source_blobs():
    root=Path(__file__).resolve().parents[1]
    assert blob(root/'model.py')=='5d9ea57b827aa1796548447868fa5ee14c251b49'
    assert blob(root/'native_p62_streaming_ttt.py')=='776e63a0ff9474a860e87088718acb36637a7bec'
    assert blob(root/'native_p64k_incremental_hybrid_stream.py')=='89c3ef5819ac594ad687c67c579a559754109d4c'


def test_new_fit_val_source_deterministic_and_no_query_leak():
    for split in ('fit','val'):
        a=p78.make_world(split,12);b=p78.make_world(split,12)
        assert p78.world_sha(a)==p78.world_sha(b)
        assert a['facts'].shape==(12,24,32) and a['query'].shape==(12,1,32)
        assert bool((a['query'][:,:,16:]==0).all())
        assert bool((a['role_audit'].sum(-1)==16).all())
        assert p78.audit(p78.make_world(split))['pass']


def test_refuse_unknown_source_and_invalid_n():
    with pytest.raises(ValueError,match='P78_UNDECLARED_SPLIT'):
        p78.make_world('future_world')
    with pytest.raises(ValueError,match='P78_BAD_WORLD_COUNT'):
        p78.make_world('val',-1)


def test_256_float_state_and_chunk_restore_read_only():
    w=p78.make_world('fit',4)
    assert p78.cold_equivalence(w)
    t=BoundedRawTTT(48);s=t.initial_state(w['facts'][:,0]);b=s.fast_weights.clone()
    fact=w['facts']
    done=t.ingest_confirmed(fact,s,torch.ones(fact.shape[:2],dtype=torch.bool))
    assert done.fast_weights.numel()==len(fact)*256
    assert torch.equal(s.fast_weights,b)
    before=done.fast_weights.clone()
    first=t.read_only(w['query'][:,0,:16],done)
    second=t.read_only(w['query'][:,0,:16],done)
    assert torch.equal(first,second) and torch.equal(before,done.fast_weights)


def test_read_before_write_and_new_action_requires_facts():
    t=BoundedRawTTT();w=p78.make_world('fit',2)
    s=t.initial_state(w['facts'][:,0]);key=w['facts'][:,0,:16]
    with pytest.raises(ValueError,match='P78_NO_CONFIRMED_HISTORY'):
        t.read_only(key,s)
    s1=t.ingest_confirmed(w['facts'][:,:1],s,torch.ones(2,1,dtype=torch.bool))
    assert torch.isfinite(t.read_only(key,s1)).all()
    assert not torch.equal(s.fast_weights,s1.fast_weights)


def test_unconfirmed_or_bad_fact_rejected_atomically():
    t=BoundedRawTTT();w=p78.make_world('fit',3)
    s=t.initial_state(w['facts'][:,0]);old=s.fast_weights.clone()
    with pytest.raises(ValueError,match='P78_UNCONFIRMED_FACT'):
        t.ingest_confirmed(w['facts'],s,torch.zeros(3,24,dtype=torch.bool))
    assert torch.equal(old,s.fast_weights)
    bad=w['facts'].clone();bad[0,2,2]=float('nan')
    with pytest.raises(ValueError,match='P78_INVALID_FACT_OR_CARRY'):
        t.ingest_confirmed(bad,s,torch.ones(3,24,dtype=torch.bool))
    with pytest.raises(ValueError,match='P78_INVALID_FACT_OR_CARRY'):
        t.ingest_confirmed(w['facts'],TTTFactualState(torch.zeros(3,16,15),s.confirmed_count),torch.ones(3,24,dtype=torch.bool))


def test_capacity_and_no_extra_persistent_float_storage():
    t=BoundedRawTTT(24);w=p78.make_world('fit',2)
    s=t.initial_state(w['facts'][:,0]);done=t.ingest_confirmed(w['facts'],s,torch.ones(2,24,dtype=torch.bool))
    assert done.fast_weights.shape==(2,16,16)
    with pytest.raises(ValueError,match='P78_FACT_CAPACITY_EXCEEDED'):
        t.ingest_confirmed(w['facts'][:,:1],done,torch.ones(2,1,dtype=torch.bool))


def test_native_exact_original_four_block_forward_parity():
    w=p78.make_world('val',3);m=p78.core(780201)
    f=p78.features(m,w)
    pred=p78.output(m,f,'native_original')
    with torch.no_grad():
        exact=m(torch.cat((w['facts'],w['query']),1))[:,-1]
    torch.testing.assert_close(pred,exact,atol=1e-6,rtol=1e-6)


def test_identity_and_original_share_same_trainable_parameter_start():
    a=p78.core(780201);b=p78.core(780201)
    assert p78.core_sha(a)==p78.core_sha(b)
    assert set(k for k,p in a.named_parameters() if p.requires_grad)=={'ttt.proj.weight','ttt.proj.bias','head.weight','head.bias'}
    assert sum(p.numel() for p in a.parameters() if p.requires_grad)==544


def test_zero_state_and_strong_fixed_controls_are_real_facts():
    w=p78.make_world('val',12);m=p78.core(780201);f=p78.features(m,w)
    assert f['raw_state'].shape==(12,16,16)
    assert torch.isfinite(p78.frequency(w)).all()
    assert p78.fixed_raw_fast_mse(f)>=0
    assert torch.isfinite(p78.output(m,f,'no_memory')).all()


def test_native_state_type_and_tampered_policy_tag_rejected():
    w=p78.make_world('fit',2)
    t=BoundedRawTTT();s=t.initial_state(w['facts'][:,0])
    flags=torch.ones(2,24,dtype=torch.bool)
    with pytest.raises(ValueError,match='P78_INVALID_FACT_OR_CARRY'):
        t.ingest_confirmed(w['facts'],TTTFactualState(s.fast_weights,s.confirmed_count),flags)
    with pytest.raises(ValueError,match='P78_INVALID_FACT_OR_CARRY'):
        t.ingest_confirmed(w['facts'],RawTTTState(s.fast_weights,s.confirmed_count,'WRONG_POLICY'),flags)
    assert s.policy_tag==POLICY_TAG
