"""P81 exact original source-bound readout diagnostic engineering tests only."""
import hashlib
import importlib.util
from pathlib import Path
import pytest
import torch

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('p81',root/'experiments/p81_readout_information_dev.py')
p81=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p81)
torch.set_num_threads(1)


def test_parent_exact_original_source_blobs():
    assert p81.git_blob(root/'model.py')=='5d9ea57b827aa1796548447868fa5ee14c251b49'
    assert p81.git_blob(root/'native_p78_bounded_raw_ttt.py')=='db948c53bb1a28af5fe0e6a62ef71463db6d95ba'


def test_fresh_deterministic_source_shapes_query_hidden():
    for name in ('fit','val','negative'):
        a=p81.source(name,8)
        assert a['facts'].shape==(8,24,32)
        assert a['query'].shape==(8,1,32)
        assert a['label'].shape==(8,16)
        assert p81.digest(a)==p81.digest(p81.source(name,8))
        assert torch.all(a['role_audit'].sum(-1)==16)
        assert torch.equal(a['query'][:,:,16:],torch.zeros((8,1,16)))


def test_sealed_ood_and_invalid_world_count_rejected():
    with pytest.raises(ValueError,match='P81_SEALED_OR_UNKNOWN_SPLIT'):
        p81.source('ood_dev')
    with pytest.raises(ValueError,match='P81_BAD_SOURCE_COUNT'):
        p81.source('val',0)


def test_fit_val_position_and_schema_leakage_guards():
    assert p81.audit(p81.source('fit'))['pass']
    assert p81.audit(p81.source('val'))['pass']
    bad=p81.source('fit',8)
    bad['query'][:,:,16:]=3
    assert not p81.audit(bad)['pass']
    with pytest.raises(ValueError,match='P81_BAD_WORLD_SCHEMA'):
        p81.factual_read(bad)


def test_confirmed_only_bounded_memory_and_cold_equivalence():
    w=p81.source('val',8)
    assert p81.cold_restores(w)
    read,state=p81.factual_read(w)
    assert read.shape==(8,16)
    assert state.fast_weights.numel()==8*256
    assert state.confirmed_count.tolist()==[24]*8
    old=state.fast_weights.clone()
    t=p81.BoundedRawTTT(max_events=24)
    with pytest.raises(ValueError,match='P78_FACT_CAPACITY_EXCEEDED'):
        t.ingest_confirmed(w['facts'][:,:1],state,torch.ones(8,1,dtype=torch.bool))
    assert torch.equal(state.fast_weights,old)
    with pytest.raises(ValueError,match='P78_UNCONFIRMED_FACT'):
        t.ingest_confirmed(w['facts'][:,:1],t.initial_state(w['facts'][:,0]),torch.zeros(8,1,dtype=torch.bool))


def test_causal_historical_native_features_ignore_future_query_action():
    w=p81.source('fit',4)
    m=p81.core(810201)
    with torch.no_grad():
        z=m.attn(m.ssm(m.stem(torch.cat((w['facts'],w['query']),dim=1))))
        modified=w['query'].clone()
        modified[:,:,0]*=-1
        z2=m.attn(m.ssm(m.stem(torch.cat((w['facts'],modified),dim=1))))
    torch.testing.assert_close(z[:,:-1],z2[:,:-1],atol=1e-6,rtol=1e-6)


def test_original_model_frozen_and_all_feature_arms_finite():
    m=p81.core(810201)
    before=p81.core_sha(m)
    w=p81.source('fit',8)
    read,_=p81.factual_read(w)
    views=p81.feature_views(m,w,read)
    assert set(views)=={'raw_16','native_context_16','memory_projected_16',
           'memory_projected_after_depth_16','sum_pre_depth_16',
           'sum_post_depth_16','concat_32','native_memory_actual'}
    assert views['concat_32'].shape==(8,32)
    assert all(torch.isfinite(v).all() for v in views.values())
    assert before==p81.core_sha(m)


def test_ridge_only_fit_then_validate_with_no_target_in_features():
    fit,val=p81.source('fit',32),p81.source('val',16)
    fr,_=p81.factual_read(fit)
    vr,_=p81.factual_read(val)
    params=p81.fit_ridge(fr,fit['label'])
    assert params.shape==(17,16)
    assert torch.isfinite(params).all()
    original=params.clone()
    _=p81.read_ridge(vr,params)
    assert torch.equal(original,params)


def test_independent_wrong_credit_keeps_target_and_query_fixed():
    w=p81.source('negative',16)
    bad=p81.negative_credit(w)
    assert torch.equal(bad['query'],w['query'])
    assert torch.equal(bad['label'],w['label'])
    assert torch.equal(bad['facts'][:,:,:16],w['facts'][:,:,:16])
    assert not torch.equal(bad['facts'][:,:,16:],w['facts'][:,:,16:])
    correct,_=p81.factual_read(w)
    wrong,_=p81.factual_read(bad)
    assert p81.mse(wrong,w['label'])>2*p81.mse(correct,w['label'])


def test_original_model_seeds_independent():
    hashes=[p81.core_sha(p81.core(s)) for s in p81.CORE_SEEDS]
    assert len(set(hashes))==3
    assert p81.core_sha(p81.core(p81.CORE_SEEDS[0]))==hashes[0]



def test_frozen_full_dev_diagnostic_reproduces_without_ood():
    receipt=p81.run()
    assert receipt['status']=='DEV_MECHANISTIC_DIAGNOSTIC_PASS_ONLY'
    assert not receipt['OOD_generated'] and not receipt['OOD_scored']
    assert not receipt['science_pass']
    assert receipt['cold_bitwise']
    assert len(receipt['rows'])==3
    assert all(r['projection_decodable'] and not r['depth_decodable']
               and r['additive_fusion_bottleneck'] for r in receipt['rows'])
    assert receipt['negative_credit']['pass']
