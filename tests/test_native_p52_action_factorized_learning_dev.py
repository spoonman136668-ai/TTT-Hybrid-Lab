"""P52 source-free engineering controls. Never create or consume P52 cohorts."""
import ast
import hashlib
from pathlib import Path
import pytest
import torch
from scripts import native_p51_factual_hybrid_learning_dev as p51
from scripts import native_p52_action_factorized_learning_dev as p52


def fixture():
    torch.manual_seed(5200)
    model=p52.FactorizedFactualP50().eval()
    batch=3
    data={'history':torch.randn(batch,8,5),
          'sensor':torch.randn(batch,1),
          'query_a':torch.ones(batch,1)}
    data['history'][:,:,2]=torch.tensor([1.,-1.,1.,1.,-1.,-1.,1.,-1.])
    data['history'][:,:,3]=.125
    data['history'][:,:,4]=0.
    return model,data


def githash(path):
    b=Path(path).read_bytes()
    return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()


def test_original_neural_source_custody_unchanged():
    root=Path(__file__).resolve().parents[1]
    assert githash(root/'model.py')=='5d9ea57b827aa1796548447868fa5ee14c251b49'
    assert githash(root/'native_factual_hybrid_stream.py')=='f78803f9f1d864382f7c4e27f72492dad7175bfd'
    assert githash(root/'scripts/native_p51_factual_hybrid_learning_dev.py')=='d7dbf31f5ee46167775eef7b0c273cb18ba0d606'


def test_parameter_budget_and_all_original_recurrent_components():
    m,d=fixture()
    assert p51.params(m)==4627
    assert p51.params(m) <= 1.02*p51.params(p51.P50Original())
    assert all(getattr(m.native.pilot,k) is not None for k in ('ssm','attn','ttt','depth'))


def test_strict_action_antisymmetric_effect_readout_without_query_leak():
    m,d=fixture()
    candidate=torch.tensor([[[-1.],[1.]]]).expand(3,-1,-1)
    a=m(d,candidate)
    b=m(d,candidate.flip(1))
    torch.testing.assert_close(a.flip(1),b,atol=0,rtol=0)
    # An action cannot affect the nuisance trunk: 
    y0=m(d,torch.zeros(3,1))[:,0,:]
    torch.testing.assert_close((a[:,0,:]+a[:,1,:])/2.,y0,atol=1e-6,rtol=0)


def test_wrong_credit_changes_only_executed_history_action_identity():
    m,d=fixture()
    original=d['history'].clone()
    wrong=p51.history_with_credit(d,wrong_credit=True)
    torch.testing.assert_close(wrong[:,:,2],torch.roll(original[:,:,2],1,dims=1),atol=0,rtol=0)
    for column in (0,1,3,4):
        torch.testing.assert_close(wrong[:,:,column],original[:,:,column],atol=0,rtol=0)
    assert torch.equal(d['history'],original)


def test_read_only_candidates_no_parameters_or_factual_input_writes():
    m,d=fixture()
    before=d['history'].clone()
    weights={k:v.detach().clone() for k,v in m.state_dict().items()}
    a=torch.tensor([[[-1.],[1.]]]).expand(3,-1,-1)
    for _ in range(3):m(d,a)
    assert torch.equal(before,d['history'])
    assert all(torch.equal(v,weights[k]) for k,v in m.state_dict().items())


def test_training_gradient_reaches_sensory_trunk_and_causal_effect_head():
    m,d=fixture()
    c=d['query_a']
    loss=(m(d,c)-.1).square().mean();loss.backward()
    assert m.native.pilot.head.weight.grad is not None
    assert m.native.pilot.head.weight.grad[1].abs().sum().item()>0
    for key in ('stem','ssm','attn','ttt'):
        component=getattr(m.native.pilot,key)
        assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in component.parameters()),key


def test_failed_source_reuse_has_deterministic_seed_identity_controls():
    assert p52.FRESH_SEEDS=={'fit':851101,'validation':851103,'unfamiliar_dev':851107}
    assert p52.MODEL_SEEDS==(861101,861103,861107)
    assert p52.UPDATES==800 and p52.BATCH==64
    assert p52.FRESH_SEEDS!=p51.SEEDS
    assert len(set(p52.FRESH_SEEDS.values()))==3


def test_free_tokenless_path_not_embedding_or_token_objective():
    code=Path(p52.__file__).read_text()
    tree=ast.parse(code)
    attrs={node.attr for node in ast.walk(tree) if isinstance(node,ast.Attribute)}
    assert not attrs.intersection({'Embedding','CrossEntropyLoss','tokenizer','vocabulary','token_ids'})
