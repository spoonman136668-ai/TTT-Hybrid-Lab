"""No-source P43-R1 tests; synthetic completed transition fixtures only."""
import io
import ast
from pathlib import Path
import torch
import pytest
from native_trainable_causal_belief import NativeTrainableCausalBelief
from native_p43_identifiability import CertifiedFactualCausalBelief,CertifiedCausalState


def setup(batch=1):
    torch.manual_seed(43191)
    m=NativeTrainableCausalBelief(1,width=12,min_facts=2).eval()
    gate=CertifiedFactualCausalBelief(m)
    sensor=torch.zeros((batch,1))
    return gate,sensor,gate.initial_state(sensor)


def execute(gate,sensor,state,actions):
    for a in actions:
        physical=torch.full_like(sensor,float(a))
        elapsed=state.factual.elapsed+1
        actual=sensor+.1+.05*elapsed+.2*physical
        state=gate.observe(sensor,physical,torch.ones_like(sensor),actual,state)
    return state


def candidates(sensor):
    return torch.tensor([[[-1.],[1.]]],dtype=sensor.dtype).expand(sensor.shape[0],-1,-1)


def test_p43_unqualified_fact_count_no_longer_authorizes_false_action_readiness():
    gate,sensor,state=setup()
    state=execute(gate,sensor,state,[1,1,1,1])
    assert gate.model.preview(sensor,candidates(sensor),torch.ones_like(sensor),state.factual).factual_ready.item()
    assert not gate.preview(sensor,candidates(sensor),torch.ones_like(sensor),state).factual_ready.item()
    with pytest.raises(ValueError,match='P43R1_FACTUAL_ACTION_EFFECT_UNIDENTIFIABLE'):
        gate.select(sensor,candidates(sensor),torch.ones_like(sensor),sensor+.2,state)
    assert state.factual.observations.item()==4


def test_action_diversity_alone_does_not_resolve_perfect_time_confound():
    gate,sensor,state=setup()
    state=execute(gate,sensor,state,[1,2,3,4])
    assert not gate.evidence_ready(sensor,state).item()
    assert state.factual.observations.item()==4


def test_rank_valid_completed_interventions_enable_bounded_selection():
    gate,sensor,state=setup()
    state=execute(gate,sensor,state,[1,-1,1,-1])
    assert gate.evidence_ready(sensor,state).item()
    before={k:v.detach().clone() for k,v in gate.model.state_dict().items()}
    h=state.factual.hidden.clone()
    gram=state.design_gram.clone()
    result=gate.preview(sensor,candidates(sensor),torch.ones_like(sensor),state)
    flip=gate.preview(sensor,candidates(sensor).flip(1),torch.ones_like(sensor),state)
    assert result.factual_ready.item()
    torch.testing.assert_close(result.predictions.flip(1),flip.predictions,atol=0,rtol=0)
    assert gate.select(sensor,candidates(sensor),torch.ones_like(sensor),sensor+.2,state).shape==(1,)
    assert torch.equal(state.design_gram,gram) and torch.equal(state.factual.hidden,h)
    assert all(torch.equal(v,before[k]) for k,v in gate.model.state_dict().items())


def test_batch_fail_closed_and_atomic_invalid_source_update():
    gate,sensor,state=setup(batch=2)
    actions=([1.,1.,1.,1.],[1.,-1.,1.,-1.])
    for t in range(4):
        a=torch.tensor([[actions[0][t]],[actions[1][t]]])
        dt=torch.ones_like(sensor)
        actual=sensor+.1+.05*(state.factual.elapsed+dt)+.2*a
        state=gate.observe(sensor,a,dt,actual,state)
    assert gate.evidence_ready(sensor,state).tolist()==[False,True]
    with pytest.raises(ValueError,match='UNIDENTIFIABLE'):
        gate.select(sensor,candidates(sensor),torch.ones_like(sensor),sensor+.2,state)
    saved=state.design_gram.clone();steps=state.factual.observations.clone()
    with pytest.raises(ValueError,match='P43_INVALID_FACTUAL_OUTCOME'):
        gate.observe(sensor,torch.ones_like(sensor),torch.ones_like(sensor),torch.full_like(sensor,float('nan')),state)
    assert torch.equal(saved,state.design_gram) and torch.equal(steps,state.factual.observations)


def test_tensor_only_cold_restore_and_no_token_dependency():
    gate,sensor,state=setup()
    state=execute(gate,sensor,state,[1,-1,-1,1])
    buf=io.BytesIO()
    torch.save({'model':gate.model.state_dict(),'hidden':state.factual.hidden,
        'elapsed':state.factual.elapsed,'count':state.factual.observations,'gram':state.design_gram},buf)
    buf.seek(0);raw=torch.load(buf,weights_only=True)
    cold_model=NativeTrainableCausalBelief(1,12,2).eval();cold_model.load_state_dict(raw['model'])
    from native_trainable_causal_belief import NativeCausalBeliefState
    restored=CertifiedCausalState(NativeCausalBeliefState(raw['hidden'],raw['count'],raw['elapsed']),raw['gram'])
    cold=CertifiedFactualCausalBelief(cold_model)
    a=gate.preview(sensor,candidates(sensor),torch.ones_like(sensor),state)
    b=cold.preview(sensor,candidates(sensor),torch.ones_like(sensor),restored)
    assert torch.equal(a.predictions,b.predictions) and torch.equal(a.factual_ready,b.factual_ready)
    path=Path(__file__).resolve().parent/'native_p43_identifiability.py'
    attrs={n.attr for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.Attribute)}
    assert not attrs.intersection({'Embedding','CrossEntropyLoss','tokenizer','vocabulary','token_ids'})


def test_reject_forged_certificate_or_undercounted_history():
    gate,sensor,state=setup()
    state=execute(gate,sensor,state,[1,-1,1,-1])
    forged=CertifiedCausalState(state.factual,state.design_gram[:,:2,:2])
    with pytest.raises(ValueError,match='CERTIFICATE_SHAPE'):
        gate.preview(sensor,candidates(sensor),torch.ones_like(sensor),forged)
    tampered=CertifiedCausalState(state.factual,state.design_gram.clone())
    tampered.design_gram[0,0,0]=3
    with pytest.raises(ValueError,match='CERTIFICATE_FACTUAL_COUNT_DRIFT'):
        gate.evidence_ready(sensor,tampered)
