"""P83 frozen author-synthetic DEV tests with first-use and original custody guards."""
from pathlib import Path
import pytest
import torch
from experiments import p83_optimizer_vs_fusion_dev as p83

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def test_source_blob_custody():
    assert p83.p81.git_blob(ROOT/'model.py')=='5d9ea57b827aa1796548447868fa5ee14c251b49'
    assert p83.p81.git_blob(ROOT/'native_p78_bounded_raw_ttt.py')=='db948c53bb1a28af5fe0e6a62ef71463db6d95ba'
    assert p83.p81.git_blob(ROOT/'experiments/p82_capacity_matched_fusion_dev.py')=='c44f77093db501502c3741fdcf70f606f2f202b7'


def test_frozen_seed_registry_and_training_parameters():
    assert p83.SEEDS=={'fit':830111,'val':830113,'transfer':830117,'negative':830119}
    assert p83.COUNTS['fit']==224 and p83.COUNTS['val']==112
    assert p83.MODEL_SEEDS==(830201,830203,830207)
    assert (p83.STEPS,p83.BATCH,p83.CHECK)==(300,16,30)


def test_fit_val_fresh_determinism_no_future_effect():
    for key in ('fit','val'):
        a,b=p83.world(key,8),p83.world(key,8)
        assert p83.digest(a)==p83.digest(b)
        assert a['facts'].shape==(8,24,32)
        assert a['query'].shape==(8,1,32)
        assert a['role_audit'].sum(-1).tolist()==[16]*8
        assert torch.count_nonzero(a['query'][:,:,16:])==0


def test_unqualified_transfer_and_negative_fail_closed():
    for name in ('transfer','negative'):
        with pytest.raises(ValueError,match='P83_PROTECTED_UNQUALIFIED_SOURCE'):
            p83.world(name)
    with pytest.raises(ValueError,match='P83_UNDECLARED_SPLIT'):
        p83.world('prior_consumed')
    with pytest.raises(ValueError,match='P83_BAD_WORLD_COUNT'):
        p83.world('fit',7)


def test_position_leakage_and_bitwise_cold_restore():
    assert p83.p82.audit(p83.world('fit'))['pass']
    assert p83.p82.audit(p83.world('val'))['pass']
    w=p83.world('val',8)
    assert p83.p81.cold_restores(w)
    w['query'][:,:,16:]=1
    assert not p83.p82.audit(w)['pass']


def test_all_original_core_initializations_distinct_and_repeatable():
    shas=[p83.fingerprint(p83.initial(seed)) for seed in p83.MODEL_SEEDS]
    assert len(set(shas))==3
    assert p83.fingerprint(p83.initial(p83.MODEL_SEEDS[0]))==shas[0]


def test_exact_same_544_parameter_three_arms():
    m=p83.initial(830201)
    h=p83.fingerprint(m)
    views=p83.features(m,p83.world('val',8))
    assert views['state_floats']==views['native_state_floats']==256
    assert sum(p.numel() for p in m.parameters() if p.requires_grad)==544
    for arm in p83.ARMS:
        out=p83.output(m,views,arm)
        assert out.shape==(8,16) and torch.isfinite(out).all()
    with pytest.raises(ValueError,match='P83_UNKNOWN_ARM'):
        p83.output(m,views,'oracle_future')
    assert p83.fingerprint(m)==h


def test_ridge_fit_only_head_is_272_coefficients():
    m=p83.initial(830201)
    before=p83.fingerprint(m)
    f=p83.features(m,p83.world('fit',16))
    v=p83.features(m,p83.world('val',8))
    z=p83.ridge_probe(m,f,v)
    assert z['trainable_ridge_coefficients']==272
    assert z['validation_mse']>=0 and len(z['sha256'])==64
    assert p83.fingerprint(m)==before


def test_full_frozen_diagnostic_reproduces_negative_without_transfer():
    z=p83.study()
    assert z['status']=='DEV_DIAGNOSTIC_FROZEN_FEATURE_INSUFFICIENT'
    assert not z['scientific_pass']
    assert not z['OOD_generated'] and not z['OOD_scored'] and not z['negative_generated']
    assert z['cold_bitwise'] and z['fixed_controls']['competent']
    assert z['source_audits']['fit']['pass'] and z['source_audits']['val']['pass']
    assert len(z['rows'])==3 and z['distinct_original_inits']
    assert sum(row['ridge_sufficient'] for row in z['rows'])==1
    assert sum(row['cosine_eligible'] for row in z['rows'])==0
    for row in z['rows']:
        assert row['paired_bitwise_initialization']
        assert all(meta['trained_parameter_count']==544 and meta['updates']==300
            and len(meta['validation_trace'])==10 for meta in row['arms'].values())
        assert row['arms']['cosine']['best_validation_mse'] < z['fixed_controls']['strongest_mse']
