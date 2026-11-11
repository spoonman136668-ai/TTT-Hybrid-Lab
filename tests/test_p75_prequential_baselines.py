"""Source-free P75 focused engineering tests: no new research dataset consumed."""
import sys
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
EXPERIMENTS=Path(__file__).resolve().parents[1]/'experiments'
sys.path.insert(0,str(EXPERIMENTS))
from p75_prequential_baselines import (before_write_predictions,historical_match_mask,losses,
                                       model_from_p75_recipe,frozen_sample_indices)


def fixture():
    torch.manual_seed(477)
    a = F.normalize(torch.randn(2, 7, 16),dim=-1)
    a[:,2]=a[:,0]
    a[:,4]=-a[:,0]
    a[:,6]=a[:,1]
    y = torch.randn(2,7,16)
    q = F.normalize(torch.randn(2,16),dim=-1)
    truth = torch.randn(2,16)
    return {'action':a,'effect':y},q,truth


def test_mask_is_past_only_and_antiparallel():
    facts,_,_=fixture()
    mask=historical_match_mask(facts)
    assert mask.dtype == torch.bool and mask.shape==(2,7)
    assert not mask[:,0].any()
    assert mask[:,2].all() and mask[:,4].all() and mask[:,6].all()


def test_future_action_cannot_change_prior_mask():
    f,_,_=fixture()
    mask=historical_match_mask(f)
    new={'action':f['action'].clone(),'effect':f['effect'].clone()}
    new['action'][:,6]=new['action'][:,0]
    assert torch.equal(historical_match_mask(new)[:,:6],mask[:,:6])


@pytest.mark.parametrize('kind',('gru','ssm'))
def test_before_write_agrees_with_independent_prefix(kind):
    f,_,_=fixture()
    model=model_from_p75_recipe(kind,975201).eval()
    with torch.no_grad():
        pre,final=before_write_predictions(model,f)
        for t in (0,1,2,5,6):
            prefix={'action':f['action'][:,:t], 'effect':f['effect'][:,:t]}
            if t == 0:
                state=f['action'].new_zeros(2,256)
            else:
                state=model.encode(prefix)
            reference=model.read(state,f['action'][:,t])
            assert torch.allclose(pre[:,t],reference,atol=1e-6,rtol=1e-5)
        assert torch.allclose(final,model.encode(f),atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize('kind',('gru','ssm'))
def test_future_effect_changes_cannot_affect_current_prediction(kind):
    f,_,_=fixture()
    model=model_from_p75_recipe(kind,975203).eval()
    with torch.no_grad():
        before,_=before_write_predictions(model,f)
        changed={'action':f['action'].clone(),'effect':f['effect'].clone()}
        changed['effect'][:,3:]+=123.5
        after,_=before_write_predictions(model,changed)
    assert torch.allclose(before[:,:4],after[:,:4],atol=1e-6,rtol=1e-5)
    assert not torch.allclose(before[:,4:],after[:,4:])


@pytest.mark.parametrize('kind',('gru','ssm'))
def test_aux_loss_gradient_finite_and_optimizer_step(kind):
    f,q,y=fixture()
    model=model_from_p75_recipe(kind,975207)
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    out=losses(model,f,q,y,'matched_prequential')
    assert out['world_state_floats']==256
    assert out['matched_events']>=6
    assert torch.isfinite(out['loss'])
    assert out['loss']>out['final_loss']
    out['loss'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    opt.step()
    assert any(not torch.equal(v,before[k]) for k,v in model.state_dict().items())


@pytest.mark.parametrize('kind',('gru','ssm'))
def test_final_only_has_no_added_training_loss(kind):
    f,q,y=fixture()
    model=model_from_p75_recipe(kind,975201)
    a=losses(model,f,q,y,'final_only')
    b=losses(model,f,q,y,'matched_prequential')
    assert torch.equal(a['final_loss'],b['final_loss'])
    assert torch.equal(a['loss'],a['final_loss'])
    assert b['loss']>=a['loss']


def test_input_role_metadata_rejected():
    f,q,y=fixture()
    model=model_from_p75_recipe('ssm',975201)
    f['role_label']=torch.ones(2,7)
    with pytest.raises(ValueError,match='P69_HISTORICAL_FACTS_ONLY'):
        losses(model,f,q,y,'matched_prequential')


def test_no_prior_match_has_exact_zero_auxiliary():
    f,q,y=fixture()
    f['action']=torch.eye(16)[:7][None].repeat(2,1,1)
    model=model_from_p75_recipe('gru',975201)
    out=losses(model,f,q,y,'matched_prequential')
    assert out['matched_events']==0
    assert out['auxiliary_loss']==0
    assert torch.equal(out['loss'],out['final_loss'])


def test_observation_at_t_changes_loss_not_prediction_at_t():
    f,q,y=fixture()
    model=model_from_p75_recipe('gru',975201)
    pre,_=before_write_predictions(model,f)
    g={'action':f['action'].clone(),'effect':f['effect'].clone()}
    g['effect'][:,2]+=10.0
    after,_=before_write_predictions(model,g)
    assert torch.allclose(pre[:,2],after[:,2],atol=1e-6,rtol=1e-5)
    assert ((pre[:,2]-f['effect'][:,2])**2).mean()!=((after[:,2]-g['effect'][:,2])**2).mean()


def test_future_query_does_not_enter_encoder():
    f,q,y=fixture()
    with pytest.raises(ValueError,match='P69_HISTORICAL_FACTS_ONLY'):
        before_write_predictions(model_from_p75_recipe('gru',975201),dict(f,final_query=q))


def test_negative_and_controlled_mode():
    f,q,y=fixture()
    with pytest.raises(ValueError,match='P75_UNKNOWN_MODE'):
        losses(model_from_p75_recipe('ssm',975201),f,q,y,'oracle')

def test_new_model_seed_space_independent_and_no_global_rng_effect():
    state=torch.get_rng_state().clone()
    first=model_from_p75_recipe('gru',975201)
    second=model_from_p75_recipe('gru',975201)
    other=model_from_p75_recipe('gru',975203)
    assert torch.equal(state,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(first.parameters(),second.parameters()))
    assert any(not torch.equal(a,b) for a,b in zip(first.parameters(),other.parameters()))
    with pytest.raises(ValueError,match='P75_BAD_MODEL_SPEC'):
        model_from_p75_recipe('gru',981201)


def test_sampler_fixed_across_two_training_arms_and_no_global_rng_effect():
    state=torch.get_rng_state().clone()
    a=frozen_sample_indices(512,steps:=11)
    b=frozen_sample_indices(512,steps)
    assert a.shape==(11,32) and torch.equal(a,b)
    assert torch.equal(state,torch.get_rng_state())
    with pytest.raises(ValueError,match='P75_BAD_SAMPLER'):
        frozen_sample_indices(512,updates=601)
