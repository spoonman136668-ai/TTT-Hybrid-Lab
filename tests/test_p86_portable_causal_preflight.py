"""P86 exact cross-platform source custody, first-use fence and causal baseline checks."""
from pathlib import Path
import pytest
import torch
from experiments import p86_portable_causal_preflight as p86

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def test_original_model_and_P78_bytes_unchanged():
    assert all(p86.blob(ROOT/p)==sha for p,sha in p86.ORIGINAL_BLOBS.items())


def test_freeze_and_seed_namespace():
    assert p86.FREEZE=='2050098d4548fc332b1360af7ff79ed41c692fa3'
    assert p86.SEEDS=={'FIT':860111,'CAL':860113}
    assert p86.RESERVED=={'HELDOUT':860117,'NEGATIVE':860119}
    assert len(p86.RATES)*len(p86.DECAYS)==25


def test_reserved_sources_fail_closed():
    for split in ('HELDOUT','NEGATIVE','OOD','P85_CAL'):
        with pytest.raises(ValueError,match='P86_PROTECTED_OR_UNKNOWN_SOURCE'):
            p86.source(split)


def test_bad_or_odd_worlds_fail_closed():
    for n in (0,7,9,129,257):
        with pytest.raises(ValueError,match='P86_INVALID_WORLD_COUNT'):
            p86.source('CAL',n)


def test_integer_source_byte_identical_after_repetition():
    for split in ('FIT','CAL'):
        x,y=p86.source(split,8),p86.source(split,8)
        assert p86.world_sha(x)==p86.world_sha(y)
        assert x['history'].shape==(8,48,32)
        assert x['query'].shape==(8,32)
        assert x['effect'].shape==(8,16)
        assert int(x['changed_world'].sum())==4
        assert torch.count_nonzero(x['query'][:,16:])==0


def test_cross_platform_exact_frozen_fit_cal_digest():
    assert p86.world_sha(p86.source('FIT'))=='116200b48a00131d1e6bf9581ed730cabdbb9a298204058e477efe63a796a060'
    assert p86.world_sha(p86.source('CAL'))=='c090d996946fc113382d0b61e5c83e4d78a1932d27e67a6573ed3afb603eddde'


def test_dyadic_source_generation_is_exact_not_torch_random():
    w=p86.source('CAL',8)
    # Source scalars are exact dyadic fractions, independent of runtime QR/BLAS.
    assert torch.equal(w['history'][:,:,:16]*128,(w['history'][:,:,:16]*128).round())
    assert torch.equal(w['history'][:,:,16:]*16384,(w['history'][:,:,16:]*16384).round())
    assert torch.equal(w['effect']*16384,(w['effect']*16384).round())


def test_no_future_effect_or_query_duplicate():
    for split in ('FIT','CAL'):
        w=p86.source(split)
        assert p86.audit(w)['pass']
        assert p86.audit(w)['max_abs_query_history_cosine']<.999
        assert p86.audit(w)['future_effect_hidden']
    bad=p86.source('CAL',8)
    bad['query'][:,16:]=bad['effect']
    assert not p86.audit(bad)['pass']


def test_original_P78_state_is_exact_same_256_float_matrix():
    w=p86.source('CAL',8)
    original,w0=p86.p78_original(w)
    comparator,w1=p86.online(w,.5,1.)
    assert w0.shape==w1.shape==(8,16,16)
    torch.testing.assert_close(w0,w1,atol=1e-7,rtol=1e-7)
    torch.testing.assert_close(original,comparator,atol=1e-7,rtol=1e-7)


def test_counterfactual_query_does_not_change_factual_state():
    w=p86.source('CAL',8)
    original,s0=p86.online(w,1.,.99)
    revised={k:v.clone() for k,v in w.items()}
    revised['query'][:,:16]*=-1
    pred,s1=p86.online(revised,1.,.99)
    assert torch.equal(s0,s1)
    torch.testing.assert_close(pred,-original,atol=1e-6,rtol=1e-6)


def test_different_executed_action_changes_observed_effect():
    w=p86.source('CAL',8)
    # This is a causal action->effect mapping; not exact-key replay.
    assert not torch.equal(w['history'][:,0,:16],w['history'][:,1,:16])
    assert not torch.equal(w['history'][:,0,16:],w['history'][:,1,16:])


def test_ols_only_uses_observed_last24_history():
    w=p86.source('CAL',8)
    p=p86.last24_ols(w)
    other={k:v.clone() for k,v in w.items()}
    other['history'][:,:24,16:]*=-1
    assert torch.equal(p,p86.last24_ols(other))


def test_preregistered_calibration_admission_no_heldout():
    r=p86.run()
    assert r['status']=='PREFLIGHT_PASS_PORTABLE_CAL_BASELINE_ONLY'
    assert r['gates']['admission_pass']
    assert r['CAL']['P78_exact_state_parity']
    assert len(r['CAL']['grid'])==25
    assert r['CAL']['selected']['memory_floats']==256
    assert not r['heldout_generated'] and not r['negative_generated']
    assert not r['original_hybrid_trained'] and not r['science_pass']


def test_strong_conventional_is_better_than_nearest_neighbor():
    r=p86.run()['CAL']
    assert r['selected']['mse']['combined']<r['nearest_unmatched_1536_history']['combined']
    assert r['selected']['mse']['changed']<r['zero']['changed']
