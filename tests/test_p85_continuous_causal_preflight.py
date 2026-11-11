"""P85 continuous action-consequence preflight, no sealed-source access."""
import math
from pathlib import Path

import pytest
import torch
from experiments import p85_continuous_causal_preflight as p85

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def test_exact_original_reference_custody():
    assert p85.git_blob(ROOT/'model.py')==p85.MODEL_BLOB
    assert p85.git_blob(ROOT/'native_p78_bounded_raw_ttt.py')==p85.RAW_BLOB


def test_frozen_registry_and_comparator_grid():
    assert p85.FIT_CAL_SEEDS=={'fit':850111,'calibration':850113}
    assert p85.RESERVED_SEEDS=={'heldout':850117,'negative':850119}
    assert p85.FIT_CAL_N=={'fit':256,'calibration':128}
    assert len(p85.RATES)*len(p85.DECAYS)==25
    assert p85.FREEZE_COMMIT=='df086a69514576f2afbe61e2ad79656810cd25ec'


def test_sealed_firstuse_denied():
    for name in ('heldout','negative','ood','prior_consumed'):
        with pytest.raises(ValueError,match='P85_PROTECTED_OR_UNKNOWN_SOURCE'):
            p85.make_worlds(name)


def test_invalid_or_unbalanced_world_counts_denied():
    for n in (0,7,9,129,257):
        with pytest.raises(ValueError,match='P85_INVALID_COUNT'):
            p85.make_worlds('calibration',n)


def test_fit_cal_shape_and_bitwise_determinism():
    for name in ('fit','calibration'):
        a,b=p85.make_worlds(name,8),p85.make_worlds(name,8)
        assert p85.world_sha(a)==p85.world_sha(b)
        assert a['history'].shape==(8,48,32)
        assert a['query'].shape==(8,32)
        assert a['effect'].shape==(8,16)
        assert int(a['changed_world'].sum())==4
        assert torch.equal(a['query'][:,16:],torch.zeros((8,16)))


def test_full_fit_cal_novel_unseen_action_and_source_admission():
    for name in ('fit','calibration'):
        a=p85.make_worlds(name)
        audit=p85.audit(a)
        assert audit['pass']
        assert audit['max_query_history_cosine']<.999
        assert audit['n']==p85.FIT_CAL_N[name]
        assert audit['changed_count']==p85.FIT_CAL_N[name]//2


def test_hidden_future_effect_guard_rejects_spoofed_observation():
    a=p85.make_worlds('fit',8)
    a['query'][:,16:]=a['effect']
    assert not p85.audit(a)['pass']


def test_original_p78_exact_256_float_recurrence():
    a=p85.make_worlds('calibration',16)
    before,carry=p85.original_p78(a)
    after,w=p85.sgd_256(a,.5,1.)
    assert carry.shape==w.shape==(16,16,16)
    torch.testing.assert_close(carry,w,atol=1e-7,rtol=1e-7)
    torch.testing.assert_close(before,after,atol=1e-7,rtol=1e-7)


def test_counterfactual_query_never_modifies_factual_carry():
    a=p85.make_worlds('calibration',8)
    original,state=p85.sgd_256(a,1.,.99)
    other={k:v.clone() for k,v in a.items()}
    other['query'][:,:16]=-other['query'][:,:16]
    flipped,remains=p85.sgd_256(other,1.,.99)
    assert torch.equal(state,remains)
    torch.testing.assert_close(flipped,-original,atol=1e-6,rtol=1e-6)


def test_untrained_analytical_ols_uses_only_last24_actual_history():
    w=p85.make_worlds('calibration',16)
    a=p85.last24_ols(w)
    assert a.shape==(16,16) and torch.isfinite(a).all()
    # Values earlier than 24 events are unavailable to this diagnostic.
    b={k:v.clone() for k,v in w.items()}
    b['history'][:,:24,16:]*=-1
    assert torch.equal(p85.last24_ols(b),a)


def test_full_preregistered_preflight_result_and_no_heldout():
    result=p85.run()
    assert result['status']=='PREFLIGHT_PASS_BASELINES_ONLY'
    assert result['admission_controls']['pass']
    assert not result['protected_heldout_generated']
    assert not result['negative_generated']
    assert not result['full_hybrid_trained']
    assert not result['scientific_pass']
    assert result['calibration']['p78_exact_grid_equivalence']
    assert len(result['calibration']['grid'])==25
    assert result['calibration']['selected']['state_floats_per_world']==256


def test_source_shift_and_baseline_effect_is_causal_not_nearest_key_lookup():
    result=p85.run()
    z=result['calibration']
    assert z['selected']['mse']['changed']<z['zero']['changed']
    assert z['selected']['mse']['stable']<z['zero']['stable']
    assert z['nearest_neighbor_unmatched_1536_float_history']['combined']>z['selected']['mse']['combined']
