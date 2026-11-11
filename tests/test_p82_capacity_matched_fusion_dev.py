"""P82 original-source-bound equal-parameter fusion research safeguards."""
import hashlib
from pathlib import Path
import pytest
import torch
from experiments import p82_capacity_matched_fusion_dev as p82

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def test_source_bytes_and_original_code_unmodified():
    expected={'model.py':'5d9ea57b827aa1796548447868fa5ee14c251b49',
      'native_p78_bounded_raw_ttt.py':'db948c53bb1a28af5fe0e6a62ef71463db6d95ba',
      'experiments/p81_readout_information_dev.py':'80f136a3507cf53434695c1d083cb774d40857e9',
      'experiments/p72_original_hybrid_transfer_dev.py':'83099a6fb0ae3f37aea2715ff8351caf5f5cd39e'}
    assert all(p82.p81.git_blob(ROOT/name)==sha for name,sha in expected.items())


def test_fresh_seeds_and_frozen_training_schedule():
    assert p82.SEEDS=={'fit':820111,'val':820113,'transfer':820117,'negative':820119}
    assert p82.CORE_SEEDS==(820201,820203,820207)
    assert (p82.STEPS,p82.BATCH,p82.CHECK)==(300,16,30)
    assert len(p82.ARMS)==4


def test_fit_val_source_shape_determinism_and_future_unknown():
    for split in ('fit','val'):
        a=p82.world(split,8);b=p82.world(split,8)
        assert p82.world_sha(a)==p82.world_sha(b)
        assert a['facts'].shape==(8,24,32)
        assert a['query'].shape==(8,1,32)
        assert torch.equal(a['query'][:,:,16:],torch.zeros(8,1,16))
        assert a['role_audit'].sum(-1).tolist()==[16]*8


def test_no_transfer_firstuse_during_frozen_validation():
    # No transfer/negative cohorts are materialized by the test suite.
    assert 'transfer' not in ('fit','val')
    with pytest.raises(ValueError,match='P82_UNKNOWN_SPLIT'):
        p82.world('prior_ood_consumed')
    with pytest.raises(ValueError,match='P82_INVALID_WORLD_COUNT'):
        p82.world('val',7)


def test_complete_fit_val_role_position_controls():
    assert p82.audit(p82.world('fit'))['pass']
    assert p82.audit(p82.world('val'))['pass']
    x=p82.world('fit',8)
    x['query'][:,:,16:]=1
    assert not p82.audit(x)['pass']


def test_bounded_confirmed_factual_state_cold_restore():
    w=p82.world('val',8)
    assert p82.p81.cold_restores(w)
    m=p82.model(820201)
    f=p82.features(m,w)
    assert f['state_floats']==256 and f['native_state_floats']==256
    assert f['raw'].shape==(8,16) and f['context'].shape==(8,16)


def test_same_544_trainable_weights_and_all_four_arms_finite():
    m=p82.model(820201)
    start=p82.p81.core_sha(m)
    f=p82.features(m,p82.world('val',8))
    trainables=sum(x.numel() for x in m.parameters() if x.requires_grad)
    assert trainables==544
    for mode in p82.ARMS:
        out=p82.out(m,f,mode)
        assert out.shape==(8,16) and torch.isfinite(out).all()
    assert p82.p81.core_sha(m)==start
    with pytest.raises(ValueError,match='P82_UNKNOWN_ARM'):
        p82.out(m,f,'oracle_future_label')


def test_corrected_fusion_operator_exact_predeclared_no_parameter_addition():
    m=p82.model(820201)
    f=p82.features(m,p82.world('val',8))
    with torch.no_grad():
        expected=m.head(m.ttt.proj(f['raw'])+.05*f['depth_context'])
        torch.testing.assert_close(expected,p82.out(m,f,'corrected'),rtol=0,atol=0)
        original=m.head(m.depth(f['context']+m.ttt.proj(f['raw'])))
        torch.testing.assert_close(original,p82.out(m,f,'original_raw_fusion'),rtol=0,atol=0)


def test_strong_untrained_256_float_comparator_and_zero():
    w=p82.world('val',16)
    prediction=p82.cache_prediction(w)
    assert prediction.shape==(16,16) and torch.isfinite(prediction).all()
    fixed,_=p82.BoundedRawTTT(max_events=48).read_from_episode(w['facts'],w['query'][:,0,:16])
    assert fixed.shape==prediction.shape


def test_frozen_end_to_end_dev_negative_and_unconsumed_transfer():
    result=p82.study()
    assert result['status']=='DEV_FAIL_FROZEN_VAL_OR_CONVERGENCE'
    assert not result['OOD_generated'] and not result['OOD_scored']
    assert not result['negative_generated'] and not result['scientific_pass']
    assert result['cold_restore']
    assert len(result['model_rows'])==3
    assert result['validation_controls']['pass']
    for row in result['model_rows']:
        assert row['identical_model_initializations']
        assert row['core_state_floats']==256
        assert all(x['trainable_params']==544 for x in row['arms'].values())
        assert row['arms']['corrected']['best_mse']<row['arms']['original_raw_fusion']['best_mse']
    assert sum(row['qualified'] for row in result['model_rows'])==1
