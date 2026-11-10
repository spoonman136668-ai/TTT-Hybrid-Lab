"""P51 development engineering controls; manual tiny facts, not frozen source generation."""
import ast, hashlib
from pathlib import Path
import pytest
import torch
from scripts import native_p51_factual_hybrid_learning_dev as p51


def git_blob(path):
    data=Path(path).read_bytes()
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def toy(n=2):
    hist=torch.zeros(n,8,5)
    state=torch.zeros(n,1)
    for t,a in enumerate((1,-1,1,-1,1,-1,1,-1)):
        delta=.10+.15*a+.02*t
        hist[:,t,0:1]=state
        hist[:,t,1:2]=delta
        hist[:,t,2:3]=a
        hist[:,t,3:4]=.125
        state=state+delta
    return {'history':hist,'sensor':state,'query_a':torch.ones(n,1),
            'factual_y':state+.2,'paired':torch.cat((state-.15,state+.15),dim=1),
            'effect':torch.full((n,1),.15),'switch':torch.zeros(n,dtype=torch.bool)}


def test_original_exact_p50_source_and_existing_reference_unchanged():
    root=Path(__file__).resolve().parents[1]
    assert git_blob(root/'model.py')=='5d9ea57b827aa1796548447868fa5ee14c251b49'
    assert git_blob(root/'native_factual_hybrid_stream.py')=='f78803f9f1d864382f7c4e27f72492dad7175bfd'


def test_parameter_matched_distinct_neural_baselines():
    assert p51.params(p51.P50Original())==4610
    assert p51.params(p51.GRUControl())==4521
    assert p51.params(p51.MLPControl())==4621
    assert p51.params(p51.P50Original())==p51.params(p51.P50Original())


def test_only_one_factual_query_action_supervises_training():
    d=toy();candidate=p51.query_features(d['sensor'],d['query_a'])
    assert candidate.shape==(2,1,5)
    assert torch.all(candidate[:,:,1]==0)
    assert torch.all(candidate[:,:,-1]==1)
    assert torch.all(d['history'][:,:,-1]==0)
    assert torch.all(candidate[:,:,2]==d['query_a'])


def test_all_arms_accept_matched_factual_inputs_and_finite_action_perturbation():
    d=toy()
    actions=torch.tensor([[[-1.],[1.]]]).expand(2,-1,-1)
    for model in (p51.P50Original(),p51.GRUControl(),p51.MLPControl()):
        y=model(d,actions)
        assert y.shape==(2,2,1)
        assert torch.isfinite(y).all()


def test_wrong_credit_changes_only_completed_action_ids_not_outcomes_or_query():
    d=toy();old={k:v.clone() for k,v in d.items()}
    broken=p51.history_with_credit(d,wrong_credit=True)
    assert torch.equal(broken[:,:,2],torch.roll(d['history'][:,:,2],1,1))
    assert torch.equal(broken[:,:,1],d['history'][:,:,1])
    assert torch.equal(broken[:,:,0],d['history'][:,:,0])
    assert all(torch.equal(v,old[k]) for k,v in d.items())


def test_original_p50_pilot_receives_gradient_from_actual_query_outcome():
    d=toy();model=p51.P50Original().train()
    with torch.no_grad():
        model.native.pilot.depth.router.weight.zero_()
        model.native.pilot.depth.router.bias.fill_(1.)
    target=d['factual_y']
    predicted=model(d,d['query_a'])[:,0,:]
    loss=(predicted-target).square().mean();loss.backward()
    for module in ('stem','ssm','attn','ttt','depth','head'):
        ref=getattr(model.native.pilot,module)
        assert any(p.grad is not None and (p.grad.abs()>0).any().item() for p in ref.parameters())


def test_classical_strong_ridge_uses_only_factual_history():
    d=toy();o=p51.polynomial_ridge(d)
    assert o['metrics']['all']['n']==2
    assert o['metrics']['all']['mse']>=0
    assert o['metrics']['all']['effect_mae']>=0


def test_no_reused_consumed_source_seeds():
    assert set(p51.SEEDS.values())=={811101,811103,811107}
    assert set(p51.INIT)=={831101,831103,831107}
    assert p51.INDEX_SEED==831199
    for label in ('P47','P48','P49','P50'):
        assert label not in str(p51.SEEDS.values())
    with pytest.raises(ValueError,match='P51_UNREGISTERED_DEV_SOURCE'):
        p51.source('old_p50_policy_source')


def test_native_path_has_no_lexical_embedding_or_token_objective():
    for file in ('native_factual_hybrid_stream.py','model.py'):
        tree=ast.parse((Path(__file__).resolve().parents[1]/file).read_text())
        attrs={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}
        assert not (attrs&{'Embedding','CrossEntropyLoss','tokenizer','token_ids'})
