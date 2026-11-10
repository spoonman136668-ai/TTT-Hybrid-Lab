"""Focused P62 DEV custody and negative controls; no protected source first use."""
import hashlib
from pathlib import Path
import torch
import pytest
from experiments import p62_component_necessity_dev as p62


def test_original_reference_blob():
    src=Path(__file__).resolve().parents[1]/'model.py'
    content=src.read_bytes()
    assert hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()=='5d9ea57b827aa1796548447868fa5ee14c251b49'


def test_frozen_split_only_and_disjoint_rng():
    with pytest.raises(ValueError,match='P62_UNREGISTERED_SOURCE'):
        p62.source('test')
    train,sha=p62.source('fit')
    test,sha2=p62.source('unfamiliar_dev')
    assert sha!=sha2 and len(train['history'])==4096 and len(test['history'])==1024
    assert train['history'].shape==(4096,6,5)


def test_wrong_credit_independence_and_not_reversible_roll():
    data,_=p62.source('val')
    factual=data['history'][:,:,0]
    wrong=data['destroyed_credit_actions']
    assert float((factual==wrong).float().mean())>.44
    assert float((factual==wrong).float().mean())<.56
    assert not any(torch.equal(wrong,torch.roll(factual,i,dims=1)) for i in range(6))
    assert not torch.equal(factual,wrong)


def test_hypothetical_action_query_cannot_write_history():
    d=p62.factual_only(p62.source('val')[0],torch.arange(2))
    frozen={k:v.clone() for k,v in d.items()}
    actions=p62.CANDIDATES.expand(2,-1,-1)
    a=p62.inputs(d,actions,False)
    b=p62.inputs(d,actions.flip(1),False)
    torch.testing.assert_close(a.flip(1),b,rtol=0,atol=0)
    assert all(torch.equal(d[k],v) for k,v in frozen.items())


def test_original_all_four_and_train_alone_toggle():
    true=p62.FACTORIES['Hybrid']()
    assert all(getattr(true.body,k) is not None for k in ('ssm','attn','ttt','depth'))
    names={'no_TTT':'ttt','no_SSM':'ssm','no_attention':'attn','no_depth':'depth'}
    for label,member in names.items():
        m=p62.FACTORIES[label]()
        assert getattr(m.body,member) is None
        assert all(getattr(m.body,k) is not None for k in ('ssm','attn','ttt','depth') if k!=member)


def test_gradient_from_factual_reward_only():
    torch.manual_seed(13)
    m=p62.FACTORIES['Hybrid']()
    d=p62.factual_only(p62.source('val')[0],torch.arange(3))
    score=m(d,d['actual_query_action'][:,None,:])[:,0]
    loss=(score-d['actual_observed_reward'][:,0]).square().mean()
    loss.backward()
    for name in ('stem','ssm','attn','ttt','head'):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in getattr(m.body,name).parameters())


def test_factual_reward_query_is_only_actual_executed_action():
    data,_=p62.source('val')
    d=p62.factual_only(data,torch.arange(3))
    assert 'evaluator_final_polarity' not in d
    assert 'evaluator_hidden_offset' not in d
    assert 'evaluator_hidden_trend' not in d
    assert all(k in d for k in ('actual_query_action','actual_observed_reward'))


def test_cold_exact_without_hidden_hardware_or_controller():
    torch.manual_seed(9)
    d,_=p62.source('val')
    for key in ('Hybrid','no_TTT','GRU','history_MLP','destroyed_credit_Hybrid'):
        m=p62.FACTORIES[key]()
        exact,_=p62.cold_restore(m,p62.FACTORIES[key],d,wrong_credit=key=='destroyed_credit_Hybrid')
        assert exact


def test_affine_nuisance_analytic_baseline_is_actual_facts_only():
    d,_=p62.source('val')
    score=p62.fitted_affine(d)
    assert 0 <= score['success'] <= 1
    assert 0 < score['fraction_full_rank'] <= 1
    assert score['number_executed_final_actions']==512
