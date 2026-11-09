"""P50 isolated no-source architecture engineering; all outcomes are authored facts."""
import ast
import io
from pathlib import Path
import pytest
import torch
from model import Configuration
from native_factual_hybrid_stream import NativeFactualHybridStream, FactualStreamState


torch.set_num_threads(1)


def case(batch=1, cap=5, **flags):
    torch.manual_seed(5019)
    cfg=Configuration(width=16,heads=2,**flags)
    model=NativeFactualHybridStream(1,capacity=cap,cfg=cfg,min_facts=3).eval()
    sensor=torch.full((batch,1),.2)
    actions=torch.tensor([[[-1.],[1.]]]).expand(batch,-1,-1).clone()
    dt=torch.ones((batch,1))
    return model,sensor,actions,dt,model.initial_state(sensor)


def event(model,sensor,state,action,step,effect=.3):
    a=torch.full_like(sensor,float(action))
    dt=torch.ones_like(sensor)
    # Actual completed physical transition, not an oracle alternative action.
    actual=sensor+.2+.05*step+effect*a
    return model.observe(sensor,a,dt,actual,state)


def trained_history(model,sensor,state):
    for step,a in enumerate((1,-1,1,-1),1):
        state=event(model,sensor,state,a,step)
    return state


def snapshot(state):
    return [getattr(state,name).clone() for name in FactualStreamState.__dataclass_fields__]


def test_all_four_original_reference_components_receive_raw_continuous_events():
    model,sensor,actions,dt,state=case()
    assert model.pilot.ssm is not None
    assert model.pilot.attn is not None
    assert model.pilot.ttt is not None
    assert model.pilot.depth is not None
    p=model.preview(sensor,actions,dt,state)
    assert p.predictions.shape==(1,2,1)
    assert torch.isfinite(p.predictions).all()
    assert p.identified.tolist()==[False]
    assert p.facts_used.tolist()==[0]


def test_rank_identifiability_and_no_source_lookahead():
    model,sensor,actions,dt,state=case()
    with pytest.raises(ValueError,match='P50_FACTUAL_ACTION_EFFECT_NOT_IDENTIFIABLE'):
        model.select(sensor,actions,dt,sensor+.5,state)
    for i,a in enumerate((1,1,1,1),1):
        state=event(model,sensor,state,a,i)
    assert model.preview(sensor,actions,dt,state).identified.tolist()==[False]
    # A different model path from the same factual state must not change history.
    alternate=event(model,sensor,state,-1,5)
    assert alternate.observations.item()==5
    assert state.observations.item()==4
    assert model.preview(sensor,actions,dt,alternate).identified.tolist()==[True]


def test_linear_action_time_confound_refuses_even_with_many_actual_facts():
    model,sensor,actions,dt,state=case()
    for i,a in enumerate((1,2,3,4),1):
        state=event(model,sensor,state,a,i)
    p=model.preview(sensor,actions,dt,state)
    assert p.facts_used.tolist()==[4]
    assert not p.identified.item()
    with pytest.raises(ValueError,match='NOT_IDENTIFIABLE'):
        model.select(sensor,actions,dt,sensor,state)


def test_completed_fact_rank_ready_and_candidate_order_invariance():
    model,sensor,actions,dt,state=case()
    state=trained_history(model,sensor,state)
    assert model.preview(sensor,actions,dt,state).identified.tolist()==[True]
    a=model.preview(sensor,actions,dt,state)
    b=model.preview(sensor,actions.flip(1),dt,state)
    torch.testing.assert_close(a.predictions.flip(1),b.predictions,atol=0,rtol=0)
    assert model.select(sensor,actions,dt,sensor+.1,state).shape==(1,)


def test_probe_does_not_modify_factual_memory_or_trained_weights():
    model,sensor,actions,dt,state=case()
    state=trained_history(model,sensor,state)
    tensors=snapshot(state)
    weights={k:v.detach().clone() for k,v in model.state_dict().items()}
    for _ in range(3):
        model.preview(sensor,actions,dt,state)
        model.preview(sensor,actions.flip(1),dt,state)
    assert all(torch.equal(a,b) for a,b in zip(tensors,snapshot(state)))
    assert all(torch.equal(v,weights[k]) for k,v in model.state_dict().items())
    assert state.observations.item()==4


def test_bounded_ring_persists_only_last_real_actions_and_total_count():
    model,sensor,actions,dt,state=case(cap=3)
    for i,a in enumerate((1,-1,1,-1,1),1):
        state=event(model,sensor,state,a,i)
    assert state.observations.item()==5
    assert state.cursor.item()==2
    assert model.preview(sensor,actions,dt,state).facts_used.item()==3
    assert state.elapsed_total.item()==5
    assert model.preview(sensor,actions,dt,state).identified.item()


def test_gradient_flows_to_ssm_attention_ttt_depth_and_raw_output():
    model,sensor,actions,dt,state=case()
    with torch.no_grad():
        model.pilot.depth.router.weight.zero_()
        model.pilot.depth.router.bias.fill_(2.)
    state=trained_history(model,sensor,state)
    model.train()
    predicted=model.preview(sensor,actions,dt,state).predictions
    loss=(predicted[:,0,:]-(sensor-.2)).square().mean()
    loss.backward()
    for name in ('stem','ssm','attn','ttt','depth','head'):
        part=getattr(model.pilot,name)
        assert any(p.grad is not None and (p.grad.abs()>0).any().item()
                   for p in part.parameters()), name
    assert all(torch.isfinite(p.grad).all().item() for p in model.parameters()
               if p.grad is not None)


def test_cold_model_and_factual_tensor_memory_bit_exact():
    model,sensor,actions,dt,state=case()
    state=trained_history(model,sensor,state)
    model.eval()
    a=model.preview(sensor,actions,dt,state)
    pack={'weights':model.state_dict(),
          'state':{k:getattr(state,k) for k in FactualStreamState.__dataclass_fields__}}
    buf=io.BytesIO();torch.save(pack,buf);buf.seek(0)
    raw=torch.load(buf,weights_only=True)
    fresh=NativeFactualHybridStream(1,capacity=5,cfg=Configuration(width=16,heads=2))
    fresh.load_state_dict(raw['weights']);fresh.eval()
    restored=FactualStreamState(**raw['state'])
    b=fresh.preview(sensor,actions,dt,restored)
    assert torch.equal(a.predictions,b.predictions)
    assert torch.equal(a.identified,b.identified)
    assert torch.equal(a.facts_used,b.facts_used)


def test_invalid_factual_outcome_and_wrong_action_atomic_rejection():
    model,sensor,actions,dt,state=case(batch=2)
    before=snapshot(state)
    with pytest.raises(ValueError,match='P50_INVALID_FACTUAL_NEXT'):
        model.observe(sensor,torch.ones_like(sensor),dt,
                      torch.tensor([[.1],[float('nan')]]),state)
    with pytest.raises(ValueError,match='P50_ZERO_ACTION_UNIDENTIFIABLE'):
        model.observe(sensor,torch.zeros_like(sensor),dt,sensor,state)
    with pytest.raises(ValueError,match='P50_NONPOSITIVE_ELAPSED_TIME'):
        model.observe(sensor,torch.ones_like(sensor),torch.zeros_like(dt),sensor,state)
    assert all(torch.equal(a,b) for a,b in zip(before,snapshot(state)))


def test_corrupt_state_or_candidate_must_fail_closed():
    model,sensor,actions,dt,state=case()
    invalid=FactualStreamState(state.sensors[:,:1],state.deltas,state.actions,
                              state.elapsed_steps,state.observations,state.cursor,
                              state.elapsed_total)
    with pytest.raises(ValueError,match='P50_STATE_SHAPE_OR_DEVICE_DRIFT'):
        model.preview(sensor,actions,dt,invalid)
    with pytest.raises(ValueError,match='P50_INVALID_CANDIDATES'):
        model.preview(sensor,actions.long(),dt,state)
    with pytest.raises(ValueError,match='P50_CANDIDATE_LIMIT'):
        model.preview(sensor,actions[:,:1,:].expand(-1,17,-1),dt,state)


def test_two_rows_keep_independent_actions_and_next_observations():
    model,sensor,actions,dt,state=case(batch=2)
    for i in range(1,5):
        a=torch.tensor([[1. if i%2 else -1.],[-1. if i%2 else 1.]])
        next_sensor=sensor+.2+.05*i+a*torch.tensor([[.3],[-.3]])
        state=model.observe(sensor,a,dt,next_sensor,state)
    p=model.preview(sensor,actions,dt,state)
    assert p.identified.tolist()==[True,True]
    assert p.predictions.shape==(2,2,1)
    # In a row-separated causal model, another row's observations do not enter.
    only=FactualStreamState(*[x[:1].clone() for x in snapshot(state)])
    solo=model.preview(sensor[:1],actions[:1],dt[:1],only)
    torch.testing.assert_close(p.predictions[:1],solo.predictions,atol=0,rtol=0)


def test_no_tokenizer_or_lexical_objective_in_native_composed_path():
    root=Path(__file__).resolve().parents[1]
    for file in ('native_factual_hybrid_stream.py','model.py'):
        tree=ast.parse((root/file).read_text())
        names={x.attr for x in ast.walk(tree) if isinstance(x,ast.Attribute)}
        assert not names.intersection({'Embedding','CrossEntropyLoss','tokenizer',
                                        'vocabulary','token_ids'})
    model,sensor,actions,dt,state=case(ttt=False,attention=False,depth=False)
    assert torch.isfinite(model.preview(sensor,actions,dt,state).predictions).all()
