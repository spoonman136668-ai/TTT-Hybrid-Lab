"""P74 exact-source focused source/representation tests, not OOD or science."""
from pathlib import Path
import importlib.util
import pytest
import torch

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('p74',root/'experiments/p74_blockwise_address_value_readout.py')
p74=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p74)
torch.set_num_threads(1)


def test_fresh_split_determinism_and_immutable_shapes():
    for split in ('fit','val','negative'):
        a=p74.make_world(split,8)
        assert p74.source_sha(a)==p74.source_sha(p74.make_world(split,8))
        assert a['facts'].shape==(8,24,32)
        assert a['query'].shape==(8,1,32)
        assert a['label'].shape==(8,16)
        assert torch.all(a['role_audit'].sum(-1)==16)
        assert torch.equal(a['query'][:,:,16:],torch.zeros(8,1,16))


def test_source_factual_view_not_query_label_or_roles():
    w=p74.make_world('fit',8)
    assert p74.factual(w).shape==(8,24,32)
    assert 'label' not in {'facts'}
    corrupt={**w,'future_query':w['query']}
    with pytest.raises(ValueError,match='P74_BAD_WORLD_FIELDS'):
        p74.native_features(p74.core(740201),corrupt)


def test_clean_factual_raw_retrieval_and_wrong_credit_loss():
    x=p74.make_world('negative',16)
    r=p74.raw_address(x)
    assert torch.all(r['hit'])
    torch.testing.assert_close(r['pred'],x['label'],atol=1e-6,rtol=0)
    wrong=p74.wrong_credit(x)
    assert p74.mse(p74.raw_address(wrong)['pred'],x['label'])>0.01


def test_invalid_source_rejected():
    with pytest.raises(ValueError,match='P74_UNKNOWN_SPLIT'):
        p74.make_world('ood_dev')
    with pytest.raises(ValueError,match='P74_BAD_WORLD_COUNT'):
        p74.make_world('fit',0)


def test_frozen_ridge_only_uses_fit_and_has_intercept():
    w=p74.make_world('fit',32)
    raw=p74.raw_address(w)
    params=p74.solve_ridge(raw['pred'],w['label'])
    assert params.shape==(17,16)
    assert p74.mse(p74.linear_readout(raw['pred'],params),w['label'])<.01


def test_exact_original_readout_and_native_shapes():
    w=p74.make_world('val',4)
    core=p74.core(740201)
    before=p74.core_sha(core)
    features=p74.native_features(core,w)
    assert features['keys'].shape==(4,24,16)
    assert features['values'].shape==(4,24,16)
    assert features['query_key'].shape==(4,16)
    assert features['final_feature'].shape==(4,16)
    assert before==p74.core_sha(core)


def test_contribution_backpropagation_reaches_factual_stream():
    m=p74.core(740201)
    w=p74.make_world('fit',4)
    contrib=p74.gradient_contribution(m,w)
    assert contrib['mandatory_nonzero_finite']
    assert all(torch.isfinite(torch.tensor(v)).item() for v in contrib['norms'].values() if v is not None)


def test_bad_shapes_and_source_contract_stopped_before_metrics():
    w=p74.make_world('fit',8)
    w['query']=torch.zeros(8,1,31)
    with pytest.raises(ValueError,match='P74_BAD_QUERY'):
        p74.native_features(p74.core(740201),w)


def test_distinct_original_core_seeds_and_not_trained_during_probe():
    hashes=[p74.core_sha(p74.core(s)) for s in p74.CORE_SEEDS]
    assert len(set(hashes))==3
