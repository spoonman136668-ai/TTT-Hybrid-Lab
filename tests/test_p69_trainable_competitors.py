"""P69 baseline engineering. All fixtures invented in memory; never use holdout."""
import importlib.util
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

source=Path(__file__).resolve().parents[1]/'experiments'/'p69_trainable_competitors.py'
spec=importlib.util.spec_from_file_location('p69_trainable_competitors',source)
p69=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p69)
torch.set_num_threads(1)


def toy(batch=3,steps=6,seed=91):
    gen=torch.Generator().manual_seed(seed)
    a=F.normalize(torch.randn(batch,steps,16,generator=gen),dim=-1)
    y=torch.randn(batch,steps,16,generator=gen)*.12
    q=F.normalize(torch.randn(batch,16,generator=gen),dim=-1)
    return {'action':a,'effect':y,'query_action':q,'observed_query_effect':torch.randn(batch,16,generator=gen)*.14}


def facts(t):
    return {'action':t['action'],'effect':t['effect']}


@pytest.mark.parametrize('kind',['learned_gate','gru','ssm'])
def test_persistent_state_256_and_prediction_shape(kind):
    m=p69.model_from_recipe(kind,981201)
    data=toy()
    state=m.encode(facts(data))
    assert state.shape==(3,16,16) if kind=='learned_gate' else state.shape==(3,256)
    assert state[0].numel()==256
    assert m.read(state,data['query_action']).shape==(3,16)
    assert torch.isfinite(m(facts(data),data['query_action'])).all()


@pytest.mark.parametrize('kind',['learned_gate','gru','ssm'])
def test_backward_gradient_finite_and_nonzero(kind):
    m=p69.model_from_recipe(kind,981201)
    data=toy(2,5)
    loss=(m(facts(data),data['query_action'])-data['observed_query_effect']).square().mean()
    loss.backward()
    grads=[p.grad for p in m.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(torch.any(g.abs()>0) for g in grads)


@pytest.mark.parametrize('kind',['learned_gate','gru','ssm'])
def test_reproducible_initializations_and_distinct_seeds(kind):
    a=p69.model_from_recipe(kind,981201)
    b=p69.model_from_recipe(kind,981201)
    c=p69.model_from_recipe(kind,981203)
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))
    assert any(not torch.equal(x,y) for x,y in zip(a.parameters(),c.parameters()))
    assert p69.parameter_count(a)==p69.parameter_count(b)
    assert (p69.parameter_count(a)==4) if kind=='learned_gate' else p69.parameter_count(a)>1000


@pytest.mark.parametrize('kind',['learned_gate','gru','ssm'])
def test_no_query_or_privileged_labels_during_writes(kind):
    data=toy()
    m=p69.model_from_recipe(kind,981201)
    with pytest.raises(ValueError,match='P69_HISTORICAL_FACTS_ONLY'):
        m.encode({**facts(data),'query_action':data['query_action']})
    with pytest.raises(ValueError,match='P69_HISTORICAL_FACTS_ONLY'):
        m.encode({**facts(data),'role_changed_audit':torch.ones(3,6)})
    with pytest.raises(ValueError,match='P69_ZERO_ACTION'):
        m.encode({'action':torch.zeros(3,6,16),'effect':data['effect']})


def test_gate_bitwise_cold_restore_and_original_equation():
    data=toy(3,8)
    m=p69.model_from_recipe('learned_gate',981201)
    full=m.encode(facts(data))
    mid=m.encode({k:v[:,:4] for k,v in facts(data).items()})
    resumed=m.encode({k:v[:,4:] for k,v in facts(data).items()}, mid.clone())
    assert torch.equal(full,resumed)
    with torch.no_grad():
        w=torch.zeros(3,16,16)
        a,y=facts(data).values()
        for i in range(8):
            x=a[:,i]
            p=(w@x.unsqueeze(-1)).squeeze(-1)
            e=y[:,i]-p
            z=torch.stack([torch.ones(3),torch.log(e.square().mean(-1)+1e-6),
                torch.log(p.square().mean(-1)+1e-6),F.cosine_similarity(p,y[:,i],dim=-1,eps=1e-6)],dim=-1)
            w=w+.5*torch.sigmoid(z@m.theta)[:,None,None]*e[:,:,None]*x[:,None,:]
    assert torch.equal(full,w)


@pytest.mark.parametrize('kind',['gru','ssm'])
def test_conventional_models_chunked_state_matches_uninterrupted(kind):
    data=toy(2,6)
    m=p69.model_from_recipe(kind,981201).eval()
    with torch.no_grad():
        full=m.encode(facts(data))
        part=m.encode({k:v[:,:3] for k,v in facts(data).items()})
        restored=m.encode({k:v[:,3:] for k,v in facts(data).items()},part.clone())
    assert torch.allclose(full,restored,rtol=1e-5,atol=1e-6)


def test_registered_source_seeds_and_indices():
    a=p69.sample_indices(512,steps=2)
    assert a.shape==(2,32)
    assert torch.equal(a,p69.sample_indices(512,steps=2))
    with pytest.raises(ValueError,match='P69_BAD_TRAIN_SPEC'):
        p69.sample_indices(512,steps=601)
    with pytest.raises(ValueError,match='P69_UNREGISTERED_MODEL_SEED'):
        p69.model_from_recipe('gru',980201)


def test_competency_is_not_claimed_without_training_and_plateau():
    empty=p69.check_competency([],1.0,'gru')
    assert not empty['competent'] and not empty['converged']
    flat=[{'mse':0.4} for _ in range(3)]
    assert p69.check_competency(flat,1.0,'gru')['competent']
    diverging=[{'mse':0.4},{'mse':0.5},{'mse':0.6}]
    assert not p69.check_competency(diverging,1.0,'gru')['competent']


def test_no_study_entrypoint_no_ood_source_or_p68_data_access():
    s=source.read_text()
    assert 'if __name__' not in s
    assert 'make_world(' not in s
    assert 'ood_0' not in s
    assert 'p68_slot_memory_dev' not in s
    assert 'torch.save' not in s


def test_smoke_one_toy_step_per_arm_not_frozen_training():
    d=toy(3,4)
    for kind in ('learned_gate','gru','ssm'):
        model=p69.model_from_recipe(kind,981201)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        optimizer.zero_grad()
        loss=(model(facts(d),d['query_action'])-d['observed_query_effect']).square().mean()
        loss.backward()
        optimizer.step()
        assert bool(torch.isfinite(loss))
