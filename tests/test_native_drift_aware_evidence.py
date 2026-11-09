"""P42 source-free engineering tests: no owner action, training or external source."""
import io
import math
import ast
from pathlib import Path
import pytest
import torch
from native_drift_aware_evidence import DriftAwareCausalEvidence,CausalEvidenceState


def setup(window=6,batch=1):
    m=DriftAwareCausalEvidence(1,window=window,min_samples=4)
    sensor=torch.zeros((batch,1),dtype=torch.float32)
    state=m.initial_state(sensor)
    return m,sensor,state


def record(m,sensor,state,t,action,delta):
    executed=torch.full_like(sensor,float(action))
    stamp=torch.full_like(sensor,float(t))
    actual=sensor+float(delta)
    return m.observe(sensor,executed,stamp,actual,state)


def candidates(sensor):
    return torch.tensor([[[-1.],[1.]]],dtype=sensor.dtype).expand(sensor.shape[0],-1,-1)


def test_identifies_causal_effect_separate_from_affine_drift_after_factual_actions():
    m,sensor,state=setup()
    for t,a in enumerate((1,-1,1,-1)):
        state=record(m,sensor,state,t,a,.4*t+.3*a)
    assert state.identifiable.tolist()==[True]
    assert state.coefficients[0,2,0].item()==pytest.approx(.3,abs=1e-6)
    assert state.coefficients[0,1,0].item()==pytest.approx(.4,abs=1e-6)
    out=m.preview(sensor,candidates(sensor),sensor+4,state)
    torch.testing.assert_close(out,torch.tensor([[[1.3],[1.9]]]),atol=1e-6,rtol=0)
    assert m.choose(sensor,candidates(sensor),sensor+4,sensor+1.9,state).tolist()==[1]
    # P40-style last action-normalized effect misattributes nuisance drift.
    last_naive=(.4*3-.3)/(-1.)
    assert last_naive<0 and state.coefficients[0,2,0].item()>0


def test_zero_effect_nuisance_not_misattributed_to_action_after_contrast():
    m,sensor,state=setup()
    for t,a in enumerate((1,-1,1,-1)):
        state=record(m,sensor,state,t,a,.4+.2*t)
    assert state.identifiable.item()
    assert abs(state.coefficients[0,2,0].item())<1e-6
    pred=m.preview(sensor,candidates(sensor),sensor+4,state)
    assert abs((pred[0,0,0]-pred[0,1,0]).item())<1e-6
    assert pred[0,0,0].item()==pytest.approx(1.2,abs=1e-6)


def test_noisy_history_requires_multiple_actions_and_reduces_last_effect_error():
    m,sensor,state=setup()
    act=(1,-1,1,-1,1,-1)
    noise=(.02,-.02,.01,-.01,.02,-.02)
    for t,(a,n) in enumerate(zip(act,noise)):
        state=record(m,sensor,state,t,a,.2*t+.3*a+n)
    estimate=state.coefficients[0,2,0].item()
    last_effect=(.2*5-.3-.02)/(-1.)
    assert abs(estimate-.3)<.03
    assert abs(estimate-.3)<abs(last_effect-.3)
    assert state.observations.item()==6
    assert state.count.item()==6


def test_rank_deficient_and_nonmonotone_observations_fail_closed_without_mutation():
    m,sensor,state=setup()
    for t in range(4):
        state=record(m,sensor,state,t,1,.4*t+.3)
    assert not state.identifiable.item()
    with pytest.raises(ValueError,match='INSUFFICIENT_FACTUAL_ACTION_IDENTIFIABILITY'):
        m.preview(sensor,candidates(sensor),sensor+4,state)
    original=state.observations.clone()
    with pytest.raises(ValueError,match='NONMONOTONE_FACTUAL_TIMESTAMP'):
        record(m,sensor,state,3,-1,.5)
    assert torch.equal(original,state.observations)
    with pytest.raises(ValueError,match='P42_INVALID_OBSERVED_NEXT'):
        m.observe(sensor,sensor+1,sensor+5,torch.full_like(sensor,float('nan')),state)
    assert torch.equal(original,state.observations)


def test_window_eviction_and_tensor_only_cold_restore_bit_exact():
    m,sensor,state=setup(window=4)
    for t,a in enumerate((1,-1,1,-1,1,-1)):
        state=record(m,sensor,state,t,a,.4*t+.3*a)
    assert state.count.item()==4
    assert state.observations.item()==6
    assert state.times[0,:].tolist()==[2.,3.,4.,5.]
    assert state.identifiable.item()
    before=m.preview(sensor,candidates(sensor),sensor+6,state)
    f=io.BytesIO()
    torch.save({name:getattr(state,name) for name in CausalEvidenceState.__dataclass_fields__},f)
    f.seek(0)
    cold=CausalEvidenceState(**torch.load(f,weights_only=True))
    after=m.preview(sensor,candidates(sensor),sensor+6,cold)
    torch.testing.assert_close(before,after,atol=0,rtol=0)


def test_robustness_limit_quadratic_nuisance_breaks_affine_assumption():
    m,sensor,state=setup()
    for t,a in enumerate((1,-1,1,-1,1,-1)):
        state=record(m,sensor,state,t,a,.10*(t**2)+.3*a)
    predicted=m.preview(sensor,candidates(sensor),sensor+6,state)
    true=torch.tensor([[[3.6-.3],[3.6+.3]]])
    assert (predicted-true).abs().max().item()>.05  # preserve a negative


def test_reject_noncontinuous_action_and_state_shapes():
    m,sensor,state=setup()
    with pytest.raises(ValueError,match='P42_INVALID_ACTION'):
        m.observe(sensor,torch.ones_like(sensor,dtype=torch.long),sensor,sensor,state)
    with pytest.raises(ValueError,match='P42_STATE_SHAPE_OR_DEVICE_DRIFT'):
        invalid=CausalEvidenceState(state.times[:,:1],state.actions,state.deltas,
                    state.count,state.observations,state.coefficients,state.identifiable)
        m.preview(sensor,candidates(sensor),sensor+1,invalid)


def test_component_native_continuous_no_tokenizer_dependency():
    tree=ast.parse((Path(__file__).resolve().parents[1]/'native_drift_aware_evidence.py').read_text())
    names={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}
    assert not names.intersection({'Embedding','CrossEntropyLoss','tokenizer','vocabulary','token_ids'})


def test_hidden_reversal_cannot_be_foreseen_but_new_factual_window_adapts():
    m,sensor,state=setup(window=6)
    for t in range(6):
        a=1 if t%2==0 else -1
        state=record(m,sensor,state,t,a,.2*t+.3*a)
    assert state.coefficients[0,2,0].item()==pytest.approx(.3,abs=1e-6)
    # Before the first new-mode consequence, only old facts are available.
    assert state.coefficients[0,2,0].item()>0
    for t in range(6,12):
        a=1 if t%2==0 else -1
        state=record(m,sensor,state,t,a,.2*t-.3*a)
    assert state.count.item()==6
    assert state.coefficients[0,2,0].item()==pytest.approx(-.3,abs=1e-6)


def test_two_batch_members_receive_only_their_own_factual_evidence():
    m,sensor,state=setup(batch=2)
    for t in range(4):
        a=1. if t%2==0 else -1.
        executed=torch.full_like(sensor,a)
        stamp=torch.full_like(sensor,float(t))
        true_effect=torch.tensor([[.3],[-.3]])
        drift=torch.tensor([[.2*t],[.4*t]])
        state=m.observe(sensor,executed,stamp,sensor+drift+executed*true_effect,state)
    torch.testing.assert_close(state.coefficients[:,2,0],torch.tensor([.3,-.3]),atol=1e-6,rtol=0)
    assert state.identifiable.tolist()==[True,True]
    before=state.observations.clone()
    with pytest.raises(ValueError,match='NONMONOTONE_FACTUAL_TIMESTAMP'):
        m.observe(sensor,torch.ones_like(sensor),torch.tensor([[4.],[3.]]),sensor,state)
    assert torch.equal(state.observations,before)
