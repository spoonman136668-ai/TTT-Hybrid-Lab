"""P45 DEV fixtures only; no protected source first use or owner dispatch."""
from __future__ import annotations
import ast
import io
from pathlib import Path
import torch
import pytest

from native_trainable_causal_belief import NativeTrainableCausalBelief
from native_p43_identifiability import CertifiedCausalState
from native_trainable_causal_belief import NativeCausalBeliefState
from native_p45_aba_fast_adapter import NativeABAFastAdapter, ABAFastState


def case(batch=1):
    torch.manual_seed(4519)
    native=NativeTrainableCausalBelief(sensor_dim=1,width=12,min_facts=2).eval()
    adapter=NativeABAFastAdapter(native)
    sensor=torch.zeros((batch,1))
    return adapter,sensor,adapter.initial_state(sensor)


def events(adapter,sensor,state,actions=(1.,-1.,1.),times=(1.,2.,3.),
           *,offset=.4,trend=.2,beta=.3,curve=0.):
    prior=0.
    for i,(a,t) in enumerate(zip(actions,times)):
        physical=torch.full_like(sensor,float(a))
        timestep=torch.full_like(sensor,float(t-prior))
        shift=torch.tensor([beta]*sensor.shape[0])[:,None].to(sensor)
        factual=sensor+float(offset+trend*t+curve*t*t)+physical*shift
        state=adapter.observe(sensor,physical,timestep,factual,state)
        prior=t
    return state


def candidate(sensor):
    return torch.tensor([[[-1.],[1.]]],dtype=sensor.dtype).expand(sensor.shape[0],-1,-1)


def snap(state):
    assert isinstance(state,ABAFastState)
    factual=state.certified.factual
    return [x.detach().clone() for x in (factual.hidden,factual.elapsed,
       factual.observations,state.certified.design_gram,state.actions,
       state.deltas,state.times,state.recent_count,state.fast_effect,
       state.factual_contrasts)]


def test_affine_factual_aba_estimates_true_action_effect_and_ties_strong_aba():
    adapter,sensor,state=case()
    state=events(adapter,sensor,state)
    assert state.factual_contrasts.item()==1
    assert state.fast_effect.item()==pytest.approx(.3,abs=1e-6)
    # Exactly the same zero-training ABA estimate is a strong conventional control.
    y=state.deltas[0,:,0]
    conventional_aba=float((.5*(y[0]+y[2])-y[1])/2)
    assert state.fast_effect.item()==pytest.approx(conventional_aba,abs=1e-7)
    assert adapter.certifier.evidence_ready(sensor,state.certified).item()


def test_irregular_completion_times_preserve_linear_nuisance_cancellation():
    adapter,sensor,state=case()
    state=events(adapter,sensor,state,times=(1.,2.,4.))
    assert state.fast_effect.item()==pytest.approx(.3,abs=1e-6)
    assert adapter.certifier.evidence_ready(sensor,state.certified).item()


def test_no_factual_contrast_equals_original_p43_rank_checked_predictions():
    adapter,sensor,state=case()
    before=adapter.certifier.preview(sensor,candidate(sensor),torch.ones_like(sensor),state.certified)
    out=adapter.preview(sensor,candidate(sensor),torch.ones_like(sensor),state)
    torch.testing.assert_close(out.predictions,before.predictions,atol=0,rtol=0)
    assert not out.factual_ready.item()
    with pytest.raises(ValueError,match='P45_FACTUAL_ACTION_UNIDENTIFIABLE'):
        adapter.select(sensor,candidate(sensor),torch.ones_like(sensor),sensor+.3,state)


def test_fast_blend_changes_predictions_without_committing_hypothetical_results():
    adapter,sensor,state=case()
    state=events(adapter,sensor,state)
    native=adapter.certifier.preview(sensor,candidate(sensor),torch.ones_like(sensor),state.certified)
    old=snap(state)
    param={k:v.detach().clone() for k,v in adapter.state_dict().items()}
    pred=adapter.preview(sensor,candidate(sensor),torch.ones_like(sensor),state)
    slope_native=(native.predictions[0,1,0]-native.predictions[0,0,0])/2
    slope_adapted=(pred.predictions[0,1,0]-pred.predictions[0,0,0])/2
    expected=(slope_native+state.fast_effect[0,0])/2
    assert slope_adapted.item()==pytest.approx(expected.item(),abs=1e-6)
    reversed_order=adapter.preview(sensor,candidate(sensor).flip(1),torch.ones_like(sensor),state)
    torch.testing.assert_close(pred.predictions.flip(1),reversed_order.predictions,atol=0,rtol=0)
    for a,b in zip(old,snap(state)):
        assert torch.equal(a,b)
    assert all(torch.equal(v,param[k]) for k,v in adapter.state_dict().items())


def test_nonaba_intervening_fact_retains_last_factual_fast_effect():
    adapter,sensor,state=case()
    state=events(adapter,sensor,state)
    baseline=state.fast_effect.clone(); count=state.factual_contrasts.clone()
    t=torch.ones_like(sensor)
    # This fourth fact destroys the last-three ABA pattern but must not erase memory.
    fact=sensor+.4+.2*4+.3
    state=adapter.observe(sensor,t,t,fact,state)
    assert torch.equal(baseline,state.fast_effect)
    assert torch.equal(count,state.factual_contrasts)
    assert state.certified.factual.observations.item()==4


def test_quad_nuisance_falsifies_aba_assumption_and_is_not_hidden():
    adapter,sensor,state=case()
    state=events(adapter,sensor,state,curve=-1.)
    # q*t^2 creates half-q false action attribution at equally spaced times.
    assert state.fast_effect.item()==pytest.approx(-.2,abs=1e-6)
    assert abs(state.fast_effect.item()-.3)>.4


def test_only_real_completed_actions_write_factual_memory():
    adapter,sensor,state=case()
    old=snap(state)
    for _ in range(5):
        adapter.preview(sensor,candidate(sensor),torch.ones_like(sensor),state)
    assert state.certified.factual.observations.item()==0
    assert state.factual_contrasts.item()==0
    assert all(torch.equal(a,b) for a,b in zip(old,snap(state)))
    with pytest.raises(TypeError):
        adapter.observe(sensor,candidate(sensor)[:,0],torch.ones_like(sensor),state=state)


def test_learnable_gate_has_gradient_without_rewriting_actual_observations():
    adapter,sensor,state=case()
    state=events(adapter,sensor,state)
    pred=adapter.preview(sensor,candidate(sensor),torch.ones_like(sensor),state)
    loss=(pred.predictions[:,1,:]-(sensor+.1)).square().mean()
    loss.backward()
    assert adapter.blend_logit.grad is not None
    assert adapter.blend_logit.grad.abs().item()>1e-10
    assert torch.isfinite(adapter.blend_logit.grad).all()


def test_cold_tensor_checkpoint_restores_model_and_factual_memory_bit_exact():
    adapter,sensor,state=case()
    state=events(adapter,sensor,state)
    inp=candidate(sensor);dt=torch.ones_like(sensor)
    first=adapter.preview(sensor,inp,dt,state)
    old=snap(state)
    buf=io.BytesIO()
    factual=state.certified.factual
    torch.save({'model':adapter.state_dict(),
        'state':{'hidden':factual.hidden,'observations':factual.observations,
                 'elapsed':factual.elapsed,'gram':state.certified.design_gram,
                 'actions':state.actions,'deltas':state.deltas,'times':state.times,
                 'recent_count':state.recent_count,'fast_effect':state.fast_effect,
                 'factual_contrasts':state.factual_contrasts}},buf)
    buf.seek(0); data=torch.load(buf,weights_only=True)
    cold=NativeABAFastAdapter(NativeTrainableCausalBelief(1,width=12,min_facts=2)).eval()
    cold.load_state_dict(data['model'])
    st=data['state']
    restored=ABAFastState(CertifiedCausalState(NativeCausalBeliefState(st['hidden'],st['observations'],st['elapsed']),st['gram']),
        st['actions'],st['deltas'],st['times'],st['recent_count'],st['fast_effect'],st['factual_contrasts'])
    replay=cold.preview(sensor,inp,dt,restored)
    torch.testing.assert_close(first.predictions,replay.predictions,atol=0,rtol=0)
    assert torch.equal(first.factual_ready,replay.factual_ready)
    assert all(torch.equal(a,b) for a,b in zip(old,snap(restored)))


def test_reject_invalid_actual_outcome_without_partial_commit():
    adapter,sensor,state=case(batch=2)
    before=snap(state)
    with pytest.raises(ValueError,match='P43_INVALID_FACTUAL_OUTCOME'):
        adapter.observe(sensor,torch.ones_like(sensor),torch.ones_like(sensor),
                        torch.tensor([[.5],[float('nan')]]),state)
    assert all(torch.equal(a,b) for a,b in zip(before,snap(state)))
    with pytest.raises(ValueError,match='P43_ZERO_ACTION_UNIDENTIFIABLE'):
        adapter.observe(sensor,torch.zeros_like(sensor),torch.ones_like(sensor),sensor,state)


def test_zero_trained_aba_is_an_explicit_strong_comparator_and_no_tokenized_path():
    adapter,sensor,state=case()
    assert sum(p.numel() for p in adapter.parameters())-sum(p.numel() for p in adapter.native.parameters())==1
    file=Path(__file__).resolve().parents[1]/'native_p45_aba_fast_adapter.py'
    attrs={n.attr for n in ast.walk(ast.parse(file.read_text())) if isinstance(n,ast.Attribute)}
    assert not attrs.intersection({'Embedding','CrossEntropyLoss','tokenizer','vocabulary','token_ids'})
    assert state.factual_contrasts.item()==0


def test_real_action_effect_reversal_requires_new_factual_contrast_before_fast_update():
    adapter,sensor,state=case()
    state=events(adapter,sensor,state,beta=.3)
    assert state.fast_effect.item()==pytest.approx(.3,abs=1e-6)
    old_effect=state.fast_effect.clone()
    # A mode flip is not announced to the model. Only later physical effects
    # can change this evidence state, not hypothetical action previews.
    adapter.preview(sensor,candidate(sensor),torch.ones_like(sensor),state)
    assert torch.equal(old_effect,state.fast_effect)
    for t,a in ((4.,-1.),(5.,1.),(6.,-1.)):
        actual=sensor+(.4+.2*t-.3*a)
        state=adapter.observe(sensor,torch.full_like(sensor,a),
                              torch.ones_like(sensor),actual,state)
    assert state.factual_contrasts.item()==4
    assert state.fast_effect.item()==pytest.approx(-.3,abs=1e-6)
    assert state.certified.factual.observations.item()==6


def test_two_independent_rows_do_not_exchange_their_factual_fast_effects():
    adapter,sensor,state=case(batch=2)
    for t,a in ((1.,1.),(2.,-1.),(3.,1.)):
        action=torch.full_like(sensor,a)
        beta=torch.tensor([[.3],[-.3]])
        factual=sensor+(.4+.2*t)+beta*action
        state=adapter.observe(sensor,action,torch.ones_like(sensor),factual,state)
    torch.testing.assert_close(state.fast_effect[:,0],torch.tensor([.3,-.3]),atol=1e-6,rtol=0)
    assert state.factual_contrasts.tolist()==[1,1]
