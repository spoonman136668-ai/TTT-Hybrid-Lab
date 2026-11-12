"""Only toy factual streams, never sealed P76 source/score."""
import sys
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from p76_memory_repair import FactualRepair,MODES,STATE_FLOATS,MODEL_SEEDS


def toy(b=3,t=9):
    generator=torch.Generator().manual_seed(760900)
    a=F.normalize(torch.randn((b,t,16),generator=generator),dim=-1)
    y=torch.randn((b,t,16),generator=generator)
    q=F.normalize(torch.randn((b,16),generator=generator),dim=-1)
    return {'action':a,'effect':y},q


@pytest.mark.parametrize('mode',MODES)
def test_factual_only_state_budget_and_unchanged_read(mode):
    facts,q=toy()
    model=FactualRepair(mode,976201 if mode=='learned_contradiction_repair' else None)
    with torch.no_grad():
        state=model.encode(facts)
        frozen=state.clone()
        result=model.read(state,q)
        again=model.read(state,q)
    assert state.shape==(3,16,16) and state.numel()==3*STATE_FLOATS
    assert result.shape==(3,16) and torch.equal(result,again)
    assert torch.equal(state,frozen)
    assert model.theta.numel()==5 if mode=='learned_contradiction_repair' else sum(p.numel() for p in model.parameters())==0


@pytest.mark.parametrize('mode',MODES)
def test_bitwise_tensor_only_cold_resume(mode):
    facts,q=toy()
    model=FactualRepair(mode,976201 if mode=='learned_contradiction_repair' else None)
    with torch.no_grad():
        full=model.encode(facts)
        first=model.encode({k:v[:,:4] for k,v in facts.items()})
        frozen=first.clone()
        after=model.encode({k:v[:,4:] for k,v in facts.items()},carry=first)
        assert torch.equal(frozen,first)
        assert torch.equal(full,after)
        assert torch.equal(model.read(full,q),model.read(after,q))


@pytest.mark.parametrize('eta,mode',[(.5,'all_write_eta_0.5'),(1.0,'all_write_eta_1.0')])
def test_constant_gain_independent_reference(eta,mode):
    facts,q=toy(b=2,t=7)
    model=FactualRepair(mode)
    w=torch.zeros(2,16,16)
    for t in range(7):
        a=facts['action'][:,t]
        y=facts['effect'][:,t]
        prediction=torch.bmm(w,a.unsqueeze(-1)).squeeze(-1)
        w=w+eta*torch.einsum('bi,bj->bij',y-prediction,a)
    assert torch.allclose(w,model.encode(facts),atol=0,rtol=0)


def test_contradiction_repairs_stale_action_more_aggressively():
    a=torch.zeros(1,2,16);a[:,:,0]=1
    y=torch.zeros(1,2,16);y[:,0,0]=1;y[:,1,0]=-1
    facts={'action':a,'effect':y}
    fixed=FactualRepair('fixed_contradiction_repair')
    reference=FactualRepair('all_write_eta_0.5')
    # Initial .5 write gives pred=0.5; second observed value contradicts it.
    repaired=fixed.encode(facts)[0,0,0]
    baseline=reference.encode(facts)[0,0,0]
    assert repaired.item()==-1.0
    assert baseline.item()==-0.25
    assert repaired<baseline


def test_new_independent_theta_seeds_and_finite_gradient():
    f,q=toy(b=2)
    initial=[FactualRepair('learned_contradiction_repair',seed) for seed in MODEL_SEEDS]
    assert len({m.theta.detach().numpy().tobytes() for m in initial})==3
    model=initial[0]
    prediction=model(f,q)
    loss=prediction.square().mean()
    loss.backward()
    assert torch.isfinite(model.theta.grad).all()
    assert model.theta.grad.abs().sum()>0


def test_bad_factual_and_nonfinite_fail_closed():
    f,q=toy()
    model=FactualRepair('all_write_eta_0.5')
    for key in ('future_query','role','phase','time_index'):
        with pytest.raises(ValueError,match='P76_FACTUAL_HISTORY_ONLY'):
            model.encode(dict(f,**{key:q}))
    bad={k:v.clone() for k,v in f.items()};bad['effect'][0,0,0]=float('nan')
    with pytest.raises(ValueError,match='P76_NONFINITE_FACTS'):
        model.encode(bad)
    with pytest.raises(ValueError,match='P76_BAD_CARRY'):
        model.encode(f,carry=torch.zeros(3,12,12))
    with pytest.raises(ValueError,match='P76_ZERO_QUERY'):
        model.read(torch.zeros(3,16,16),torch.zeros(3,16))


def test_future_effect_does_not_change_prior_state():
    f,q=toy()
    x={k:v[:,:5].clone() for k,v in f.items()}
    changed={k:v.clone() for k,v in f.items()}
    changed['effect'][:,5:]+=111
    model=FactualRepair('learned_contradiction_repair',976201)
    assert torch.equal(model.encode(x),model.encode({k:v[:,:5] for k,v in changed.items()}))
    assert not torch.equal(model.encode(f),model.encode(changed))


def test_wrong_action_credit_is_detectable_on_toy():
    f,q=toy()
    g={'action':f['action'].flip(1),'effect':f['effect']}
    model=FactualRepair('all_write_eta_0.5')
    assert not torch.equal(model.encode(f),model.encode(g))


def test_frozen_modes_and_seeds_rejected():
    with pytest.raises(ValueError,match='P76_UNREGISTERED_MODE'):
        FactualRepair('oracle')
    with pytest.raises(ValueError,match='P76_UNREGISTERED_MODEL_SEED'):
        FactualRepair('learned_contradiction_repair',975201)
    with pytest.raises(ValueError,match='P76_FIXED_MODE_HAS_NO_SEED'):
        FactualRepair('all_write_eta_0.5',976201)
