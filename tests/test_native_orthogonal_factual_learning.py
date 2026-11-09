"""P42 source-free, frozen DEV: actual actions vs nuisance trend, no science first use."""
import io
import pytest
import torch
from native_orthogonal_factual_learning import OrthogonalFactualLearner, OrthogonalState

ACTIONS = [1.,-1.,1.,-1.,-1.,1.,1.,-1.]


def fit_authored(intercept,trend,effect):
    m=OrthogonalFactualLearner(8)
    sensor=torch.tensor([[0.]],dtype=torch.float64)
    dt=torch.ones_like(sensor)
    s=m.initial_state(sensor)
    receipts=[]
    for i,action in enumerate(ACTIONS,1):
        a=torch.tensor([[action]],dtype=sensor.dtype)
        pred,valid=m.preview(sensor,a.reshape(1,1,1),dt,s)
        assert s.observations.item()==i-1
        physical_next=sensor+intercept+trend*i+effect*action
        s=m.observe(sensor,a,dt,physical_next,s)
        c,good=m.coefficients(sensor,s)
        receipts.append((c,good,pred,valid))
    return m,sensor,dt,s,receipts


def test_frozen_affine_world_learns_effect_after_three_independent_facts():
    for b,t,e in ((.15,.4,.3),(.4,.05,0.),(-.25,-.2,-.3)):
        m,sensor,dt,state,rows=fit_authored(b,t,e)
        assert all(not bool(rec[1].item()) for rec in rows[:2])
        for c,good,_,_ in rows[2:]:
            assert bool(good.item())
            assert float(c[0,2])==pytest.approx(e,abs=1e-5)
        assert state.observations.item()==len(ACTIONS)
        assert int(state.occupied.sum())==len(ACTIONS)
        assert float(rows[-1][0][0,1])==pytest.approx(t,abs=1e-5)


def test_sham_outcome_has_no_spurious_action_effect_and_readonly_preview():
    m,sensor,dt,state,_=fit_authored(.4,.05,0.)
    before=[x.clone() for x in vars(state).values()]
    pred,valid=m.preview(sensor,torch.tensor([[[-1.],[1.]]],dtype=sensor.dtype),dt,state)
    assert bool(valid.item())
    assert float((pred[:,0]-pred[:,1]).abs().max())<1e-5
    assert all(torch.equal(x,y) for x,y in zip(before,vars(state).values()))


def test_cold_state_replay_is_bit_exact_and_invalid_does_not_mutate():
    m,sensor,dt,state,_=fit_authored(.15,.4,.3)
    buf=io.BytesIO()
    torch.save(vars(state),buf)
    buf.seek(0)
    cold=OrthogonalState(**torch.load(buf,weights_only=True))
    can=torch.tensor([[[-1.],[1.]]],dtype=sensor.dtype)
    a,va=m.preview(sensor,can,dt,state)
    b,vb=m.preview(sensor,can,dt,cold)
    assert torch.equal(a,b) and torch.equal(va,vb)
    with pytest.raises(ValueError,match='P42_INVALID_ACTION'):
        m.observe(sensor,torch.zeros_like(sensor),dt,sensor,state)
    assert state.observations.item()==8
    assert torch.equal(state.times,cold.times)


def test_quadratic_nuisance_is_outside_affine_model_and_preserved_negative():
    m=OrthogonalFactualLearner(8)
    sensor=torch.zeros((1,1),dtype=torch.float64)
    dt=torch.ones_like(sensor)
    state=m.initial_state(sensor)
    for i,a in enumerate(ACTIONS[:3],1):
        action=torch.tensor([[a]],dtype=sensor.dtype)
        actual=sensor + .3*a - float(i*i)
        state=m.observe(sensor,action,dt,actual,state)
    coeff,valid=m.coefficients(sensor,state)
    assert bool(valid.item())
    assert float(coeff[0,2])==pytest.approx(-.2,abs=1e-5) # true effect +.3


def test_rank_deficient_history_refuses_to_invent_action_effect():
    m=OrthogonalFactualLearner(4)
    x=torch.zeros((1,1),dtype=torch.float32)
    dt=torch.ones_like(x)
    state=m.initial_state(x)
    for t in (1,2,3,4):
        a=torch.ones_like(x)
        state=m.observe(x,a,dt,x+.3,state)
    coeff,valid=m.coefficients(x,state)
    assert not bool(valid.item())
    pred,mask=m.preview(x,torch.tensor([[[-1.],[1.]]]),dt,state)
    assert not bool(mask.item())
    assert torch.equal(pred,torch.zeros_like(pred))
