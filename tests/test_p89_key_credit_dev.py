"""P89 frozen source, causal key association and preregistered negative outcome."""
import pytest
import torch
from pathlib import Path
from experiments import p89_key_credit_dev as p89
from experiments import p87_native_causal_falsifier as p87

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def test_source_and_original_model_git_blobs():
    assert all(p89.p86.blob(ROOT/path)==blob for path,blob in p89.ORIGINAL.items())


def test_seed_registry_and_reserved_OOD_fail_closed():
    assert p89.SEEDS=={'FIT':890111,'VAL':890113}
    assert p89.MODEL_SEEDS==(890201,890203,890207)
    assert p89.FREEZE=='750dd41bbd66e372ae13b4e1b2a1246f6f9a9d9e'
    for split in ('OOD','NEGATIVE','P88_FIT'):
        with pytest.raises(ValueError,match='P89_PROTECTED_OR_UNKNOWN_SOURCE'):
            p89.source(split)
    for n in (0,7,9,65,129):
        with pytest.raises(ValueError,match='P89_BAD_SOURCE_SIZE'):
            p89.source('VAL',n)


def test_exact_portable_fresh_FIT_VAL_cohorts():
    assert p89.digest(p89.source('FIT'))=='d7eaa4d2fbb99be7cc9eff9e13c8a0f36bcc5dc74f558c1ae52901195fa981d7'
    assert p89.digest(p89.source('VAL'))=='c0fd5bfc2c3d0b7ec605e31a3b538353f732376d32e60fbdb6f816fb80fc7cd2'
    for split in ('FIT','VAL'):
        w=p89.source(split,8)
        assert p89.digest(w)==p89.digest(p89.source(split,8))
        assert w['actions'].shape==(8,49,16)
        assert w['effects'].shape==(8,49,16)
        assert p89.p86.audit(p89.p86_world(w))['pass']
        assert w['changed_world'].sum().item()==4


def test_no_future_effect_on_any_query():
    x=p89.paired(p89.source('VAL',8))
    assert x.shape==(8,97,32)
    assert not torch.count_nonzero(x[:,::2,16:])
    w=p89.source('VAL',8)
    assert torch.equal(x[:,1::2,16:],w['effects'][:,:48])


def test_original_full_context_prefix_causality():
    w=p89.source('VAL',8)
    x=p89.paired(w)
    changed=x.clone()
    changed[:,2*19+1,16:]*=-1
    m=p87.new_model('hybrid',890201).eval()
    with torch.no_grad():
        def encode(z):return m.attn(m.ssm(m.stem(z)))[:,::2]
        a,b=encode(x),encode(changed)
    torch.testing.assert_close(a[:,:20],b[:,:20],rtol=1e-6,atol=1e-6)


def test_fast_state_fixed_w_matches_p86_exact_256_float():
    w=p89.source('VAL',8)
    keys=torch.nn.functional.normalize(w['actions'],dim=-1)
    pred,state=p89.online(w,keys[:,:48],keys)
    ref=p89.p86.online(p89.p86_world(w),1.,.99)[0]
    assert state.shape==(8,16,16)
    torch.testing.assert_close(pred[:,-1],ref,rtol=1e-6,atol=1e-6)
    with pytest.raises(ValueError,match='P89_BAD_KEY_QUERY_SHAPE'):
        p89.online(w,keys[:,:47],keys)


def test_positive_wrong_credit_negative_control():
    w=p89.source('VAL',8)
    keys=torch.nn.functional.normalize(w['actions'],dim=-1)
    clean,_=p89.online(w,keys[:,:48],keys)
    shifted=w['effects'][:,:48][:,torch.arange(48).roll(1)]
    wrong,_=p89.online(w,keys[:,:48],keys,values=shifted)
    assert p89.scores(wrong,w)['final']>1.5*p89.scores(clean,w)['final']


def test_source_bound_full_diagnostic_and_negative_outcome():
    out=p89.run()
    assert out['status']=='DEV_INCONCLUSIVE_OR_NEGATIVE'
    assert not out['science_pass'] and not out['OOD_generated']
    assert not out['full_model_trained']
    assert out['cold_restore_P78'] and out['reference_parity']
    assert out['controls']['wrong_credit_pass']
    assert len(out['rows'])==3 and out['distinct_original_models']
    assert not any(out['frozen_signatures'].values())
    assert all(len(row['action_ridge_SHA256'])==64 and row['trainable_extra_ridge_coefficients']==272
               and len(row['scores'])==5 for row in out['rows'])
    fixed=out['controls']['fixed']['final']
    assert all(row['scores']['context_shared']['final']>4*fixed for row in out['rows'])
