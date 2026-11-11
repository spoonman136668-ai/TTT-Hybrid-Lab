"""P78-MR focused pre-FIT/VAL engineering; no cohort outcomes scored."""
import sys
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from p78mr_adaptive_controls import (
    configurations, new_learned, fixed_outputs, select_calibrated, scored_errors,
    STATE_FLOATS, MODEL_SEEDS,
)
from p77_write_policy import WritePolicy
import p78mr_factual_source as source


def toy(batch=3, steps=11):
    g=torch.Generator(device='cpu').manual_seed(778808)
    a=F.normalize(torch.randn(batch,steps,16,generator=g),dim=-1)
    y=torch.randn(batch,steps,16,generator=g)
    q=F.normalize(torch.randn(batch,16,generator=g),dim=-1)
    return {'action':a,'effect':y},q


def test_frozen_control_count_and_calibration_tie_rule():
    cs=configurations()
    assert len(cs)==83 and len(set(cs))==83
    assert sum(x[0]=='direction' for x in cs)==36
    assert sum(x[0]=='residual' for x in cs)==36
    assert sum(x[0]=='constant' for x in cs)==10
    fake={'total':[1.]*83}
    fake['total'][14]=.2;fake['total'][15]=.2
    assert select_calibrated(cs,fake)==(14,cs[14])


def test_vectorized_fixed_constant_matches_prior_original():
    f,q=toy()
    p,state=fixed_outputs(f,q)
    cs=configurations()
    for eta in (0.1,0.5,1.0):
        ix=cs.index(('constant',eta,0.,0.))
        ref=WritePolicy('fixed_eta',eta=eta)
        with torch.no_grad():
            assert torch.allclose(p[ix],ref(f,q),atol=1e-6,rtol=1e-6)
            assert torch.allclose(state[ix],ref.encode(f),atol=1e-6,rtol=1e-6)
    assert state.shape==(83,3,16,16)
    assert state.numel()==83*3*STATE_FLOATS


def test_original_fixed_contradiction_matches():
    f,q=toy()
    cs=configurations();ix=cs.index(('fixed_contradiction',0.,0.,0.))
    p,s=fixed_outputs(f,q)
    with torch.no_grad():
        ref=WritePolicy('fixed_contradiction')
        assert torch.allclose(p[ix],ref(f,q),atol=1e-6,rtol=1e-6)
        assert torch.allclose(s[ix],ref.encode(f),atol=1e-6,rtol=1e-6)


def test_fixed_tensor_only_cold_restore_bitwise():
    f,q=toy()
    _,full=fixed_outputs(f,q)
    left={k:v[:,:4] for k,v in f.items()}
    right={k:v[:,4:] for k,v in f.items()}
    _,first=fixed_outputs(left,q)
    before=first.clone()
    _,after=fixed_outputs(right,q,carry=first)
    assert torch.equal(first,before)
    assert torch.equal(full,after)


def test_independent_new_theta_and_no_rng_perturbation():
    g=torch.get_rng_state().clone()
    models=[new_learned('non_directional3',s) for s in MODEL_SEEDS]
    paired=new_learned('cosine4',MODEL_SEEDS[0])
    assert torch.equal(models[0].theta,paired.theta[:3])
    assert len({m.theta.detach().numpy().tobytes() for m in models})==3
    assert torch.equal(g,torch.get_rng_state())
    with pytest.raises(ValueError,match='P78MR_BAD_LEARNER'):
        new_learned('full5',MODEL_SEEDS[0])


def test_model_gradient_stays_within_state_budget():
    f,q=toy()
    model=new_learned('cosine4',MODEL_SEEDS[0])
    loss=model(f,q).square().mean()
    loss.backward()
    assert model.theta.numel()==4
    assert torch.isfinite(model.theta.grad).all()
    assert model.theta.grad.abs().sum()>0
    assert model.encode(f).shape==(3,16,16)


def test_read_before_write_and_no_oracle_input():
    f,q=toy()
    model=new_learned('non_directional3',MODEL_SEEDS[0])
    f2={k:v.clone() for k,v in f.items()}
    f2['effect'][:,8:]+=10
    assert torch.equal(model.encode({k:v[:,:8] for k,v in f.items()}),
                       model.encode({k:v[:,:8] for k,v in f2.items()}))
    with pytest.raises(ValueError,match='P76_FACTUAL_HISTORY_ONLY'):
        fixed_outputs(dict(f,oracle_role=torch.ones(3,11)),q)
    with pytest.raises(ValueError,match='P78MR_BAD_FIXED_CARRY'):
        fixed_outputs(f,q,carry=torch.zeros(83,3,16,15))


def test_split_disjoint_and_ood_not_admitted():
    a=source.make_world('fit',0)
    assert source.digest_world(a)==source.digest_world(source.make_world('fit',0))
    assert a['action'].shape==(96,16)
    assert source.digest_world(a)!=source.digest_world(source.make_world('cal',0))
    assert source.digest_world(a)!=source.digest_world(source.make_world('val',0))
    for key in ('ood_0','ood_1','ood_2','negative'):
        with pytest.raises(ValueError,match='P78MR_HOLDOUT_NOT_ADMITTED'):
            source.make_world(key,0)
        with pytest.raises(ValueError,match='P78MR_HOLDOUT_NOT_ADMITTED'):
            source.make_batch(key,1)


def test_scoring_stratifies_but_only_evaluator_receives_label():
    target=torch.zeros(4,16)
    pred=torch.ones(2,4,16)
    change=torch.tensor([True,False,True,False])
    r=scored_errors(pred,target,change)
    assert r['total']==[1.,1.]
    assert r['changed']==[1.,1.]
    assert r['stable']==[1.,1.]
    assert set(source.factual(source.make_world('fit',0)))=={'action','effect'}
