"""No-source P45-I R1 factual freshness controls; no science first use."""
from __future__ import annotations
import ast
import io
from pathlib import Path

import pytest
import torch

from native_trainable_causal_belief import NativeTrainableCausalBelief, NativeCausalBeliefState
from native_p43_identifiability import CertifiedCausalState
from native_p45_aba_fast_adapter import NativeABAFastAdapter, ABAFastState
from native_p45i_freshness_gate import NativeABAFreshnessGate, FreshABACausalState


def scenario(batch=1):
    torch.manual_seed(4517)
    native = NativeTrainableCausalBelief(1, width=12, min_facts=2).eval()
    gate = NativeABAFreshnessGate(NativeABAFastAdapter(native))
    sensor = torch.zeros((batch, 1))
    return gate, sensor, gate.initial_state(sensor)


def choices(sensor):
    return torch.tensor([[[-1.], [1.]]], dtype=sensor.dtype).expand(sensor.shape[0], -1, -1)


def event(gate, sensor, state, action, t, beta=.3, nonlinear=0.):
    action = torch.full_like(sensor, float(action))
    dt = torch.ones_like(sensor)
    observed = sensor + (.4 + .2*t + nonlinear*t*t) + beta*action
    return gate.observe(sensor, action, dt, observed, state)


def mature(gate, sensor, state, beta=.3, nonlinear=0.):
    for t, a in ((1, 1), (2, -1), (3, 1)):
        state = event(gate, sensor, state, a, t, beta, nonlinear)
    return state


def snapshot(state):
    s = state.fast
    f = s.certified.factual
    return [x.detach().clone() for x in (
        f.hidden, f.elapsed, f.observations, s.certified.design_gram,
        s.actions, s.deltas, s.times, s.recent_count,
        s.fast_effect, s.factual_contrasts, state.last_valid_contrast_step)]


def test_original_adapter_wrongly_remains_ready_after_real_unannounced_reversal():
    gate, sensor, state = scenario()
    state = mature(gate, sensor, state)
    assert state.fast.fast_effect[0,0].item() == pytest.approx(.3, abs=1e-6)
    changed = event(gate, sensor, state, 1, 4, beta=-.3)
    assert changed.fast.fast_effect[0,0].item() == pytest.approx(.3, abs=1e-6)
    old = gate.adapter.preview(sensor, choices(sensor), torch.ones_like(sensor), changed.fast)
    assert old.factual_ready.item()  # proven real defect in original readiness
    guarded = gate.preview(sensor, choices(sensor), torch.ones_like(sensor), changed)
    assert not guarded.factual_ready.item()
    torch.testing.assert_close(guarded.predictions,
        gate.adapter.certifier.preview(sensor, choices(sensor), torch.ones_like(sensor),
                                       changed.fast.certified).predictions, atol=0, rtol=0)
    with pytest.raises(ValueError, match='P45R1_NO_FRESH_IDENTIFIABLE_CONTRAST'):
        gate.select(sensor, choices(sensor), torch.ones_like(sensor), sensor+.25, changed)


def test_fresh_completed_aba_allows_exact_original_fast_prediction_and_selection():
    gate, sensor, state = scenario()
    state = mature(gate, sensor, state)
    assert gate.freshness(sensor,state).tolist() == [True]
    prediction = gate.preview(sensor, choices(sensor), torch.ones_like(sensor), state)
    assert prediction.factual_ready.item()
    torch.testing.assert_close(prediction.predictions,
        gate.adapter.preview(sensor,choices(sensor),torch.ones_like(sensor),state.fast).predictions,
        atol=0,rtol=0)
    assert gate.select(sensor,choices(sensor),torch.ones_like(sensor),sensor+.3,state).shape==(1,)
    assert state.last_valid_contrast_step.item()==3


def test_no_history_or_unsupported_action_histories_must_abstain():
    gate,sensor,state=scenario()
    for t,a in ((1,1),(2,1),(3,1),(4,1)):
        state=event(gate,sensor,state,a,t)
    assert state.fast.certified.factual.observations.item()==4
    assert state.fast.factual_contrasts.item()==0
    assert not gate.freshness(sensor,state).item()
    assert not gate.preview(sensor,choices(sensor),torch.ones_like(sensor),state).factual_ready.item()
    with pytest.raises(ValueError,match='P45R1_NO_FRESH_IDENTIFIABLE_CONTRAST'):
        gate.select(sensor,choices(sensor),torch.ones_like(sensor),sensor+.2,state)


def test_new_valid_three_actual_post_switch_facts_reenable_fresh_effect():
    gate,sensor,state=scenario()
    state=mature(gate,sensor,state)
    for t,a in ((4,1),(5,-1)):
        state=event(gate,sensor,state,a,t,beta=-.3)
        assert not gate.freshness(sensor,state).item()
    state=event(gate,sensor,state,1,6,beta=-.3)
    assert gate.freshness(sensor,state).item()
    assert state.last_valid_contrast_step.item()==6
    assert state.fast.fast_effect.item()==pytest.approx(-.3,abs=1e-6)
    assert gate.preview(sensor,choices(sensor),torch.ones_like(sensor),state).factual_ready.item()


def test_nonABA_factual_event_is_only_expiry_not_memory_erase():
    gate,sensor,state=scenario()
    state=mature(gate,sensor,state)
    before=state.fast.fast_effect.clone()
    after=event(gate,sensor,state,1,4)
    assert after.last_valid_contrast_step.item()==3
    assert after.fast.certified.factual.observations.item()==4
    assert torch.equal(after.fast.fast_effect,before)
    assert not gate.freshness(sensor,after).item()
    assert gate.freshness(sensor,state).item()


def test_candidate_preview_read_only_and_conserved_weights():
    gate,sensor,state=scenario()
    state=mature(gate,sensor,state)
    before=snapshot(state)
    params={k:v.detach().clone() for k,v in gate.state_dict().items()}
    for _ in range(4):
        forward=gate.preview(sensor,choices(sensor),torch.ones_like(sensor),state)
        reversed=gate.preview(sensor,choices(sensor).flip(1),torch.ones_like(sensor),state)
        torch.testing.assert_close(forward.predictions.flip(1),reversed.predictions,atol=0,rtol=0)
    assert all(torch.equal(x,y) for x,y in zip(before,snapshot(state)))
    assert all(torch.equal(x,params[k]) for k,x in gate.state_dict().items())


def test_batch_rows_have_distinct_freshness_and_no_leakage():
    gate,sensor,state=scenario(batch=2)
    state=mature(gate,sensor,state)
    a=torch.tensor([[1.],[-1.]])
    fact=sensor + .4 + .2*4 + .3*a
    state=gate.observe(sensor,a,torch.ones_like(sensor),fact,state)
    # Row 0: [-1,+1,+1] not ABA => stale. Row 1: [-1,+1,-1] ABA => fresh.
    assert gate.freshness(sensor,state).tolist()==[False,True]
    predicted=gate.preview(sensor,choices(sensor),torch.ones_like(sensor),state)
    assert predicted.factual_ready.tolist()==[False,True]
    with pytest.raises(ValueError,match='NO_FRESH_IDENTIFIABLE_CONTRAST'):
        gate.select(sensor,choices(sensor),torch.ones_like(sensor),sensor+.2,state)


def test_invalid_factual_outcome_preserves_all_old_state():
    gate,sensor,state=scenario(batch=2)
    state=mature(gate,sensor,state)
    before=snapshot(state)
    with pytest.raises(ValueError,match='P43_INVALID_FACTUAL_OUTCOME'):
        gate.observe(sensor,torch.ones_like(sensor),torch.ones_like(sensor),
                     torch.tensor([[.3],[float('nan')]]),state)
    assert all(torch.equal(x,y) for x,y in zip(before,snapshot(state)))
    with pytest.raises(ValueError,match='P43_ZERO_ACTION_UNIDENTIFIABLE'):
        gate.observe(sensor,torch.zeros_like(sensor),torch.ones_like(sensor),sensor,state)


def test_tampered_freshness_state_and_counter_drift_rejected():
    gate,sensor,state=scenario()
    state=mature(gate,sensor,state)
    bad=FreshABACausalState(state.fast,torch.tensor([5],dtype=torch.long))
    with pytest.raises(ValueError,match='P45R1_FRESHNESS_CUSTODY_DRIFT'):
        gate.preview(sensor,choices(sensor),torch.ones_like(sensor),bad)
    bad=FreshABACausalState(state.fast,state.last_valid_contrast_step.float())
    with pytest.raises(ValueError,match='P45R1_FRESHNESS_SHAPE_OR_DTYPE_DRIFT'):
        gate.freshness(sensor,bad)
    forged=FreshABACausalState(state.fast,state.last_valid_contrast_step.clone())
    forged.fast.actions[0,0,0]=-1 # breaks ABA in forged local snapshot
    with pytest.raises(ValueError,match='P45R1_CONTRAST_SOURCE_MISMATCH'):
        gate.preview(sensor,choices(sensor),torch.ones_like(sensor),forged)


def test_checkpoint_model_and_factual_freshness_restore_bit_exact():
    gate,sensor,state=scenario()
    state=mature(gate,sensor,state)
    state=event(gate,sensor,state,1,4)
    original=gate.preview(sensor,choices(sensor),torch.ones_like(sensor),state)
    x=state.fast; y=x.certified.factual
    pack={'model':gate.state_dict(),'hidden':y.hidden,'elapsed':y.elapsed,
        'observations':y.observations,'gram':x.certified.design_gram,'actions':x.actions,
        'deltas':x.deltas,'times':x.times,'recent_count':x.recent_count,
        'fast_effect':x.fast_effect,'factual_contrasts':x.factual_contrasts,
        'last':state.last_valid_contrast_step}
    buf=io.BytesIO();torch.save(pack,buf);buf.seek(0)
    cold=torch.load(buf,weights_only=True)
    new=NativeABAFreshnessGate(NativeABAFastAdapter(
        NativeTrainableCausalBelief(1,width=12,min_facts=2).eval()))
    new.load_state_dict(cold['model'])
    restored=FreshABACausalState(ABAFastState(CertifiedCausalState(
        NativeCausalBeliefState(cold['hidden'],cold['observations'],cold['elapsed']),
        cold['gram']),cold['actions'],cold['deltas'],cold['times'],
        cold['recent_count'],cold['fast_effect'],cold['factual_contrasts']),cold['last'])
    replay=new.preview(sensor,choices(sensor),torch.ones_like(sensor),restored)
    assert torch.equal(original.predictions,replay.predictions)
    assert torch.equal(original.factual_ready,replay.factual_ready)


def test_nonlinear_nuisance_makes_even_fresh_aba_wrong_explicit_negative():
    gate,sensor,state=scenario()
    state=mature(gate,sensor,state,nonlinear=-1.)
    assert gate.freshness(sensor,state).item()
    assert state.fast.fast_effect.item()==pytest.approx(-.2,abs=1e-6)
    assert abs(state.fast.fast_effect.item()-.3)>.4
    # Freshness does not equal causal truth; no scientific improvement claim.


def test_token_free_and_no_new_trainable_parameters():
    gate,sensor,state=scenario()
    assert sum(p.numel() for p in gate.parameters())==sum(p.numel() for p in gate.adapter.parameters())
    source=Path(__file__).resolve().parents[1]/'native_p45i_freshness_gate.py'
    names={n.attr for n in ast.walk(ast.parse(source.read_text())) if isinstance(n,ast.Attribute)}
    assert not names.intersection({'Embedding','CrossEntropyLoss','tokenizer','vocabulary','token_ids'})
