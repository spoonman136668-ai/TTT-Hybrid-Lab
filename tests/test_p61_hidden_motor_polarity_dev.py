"""P61 no-first-use manual algebra fixtures, exact model custody and negative controls."""
from __future__ import annotations
import io
import hashlib
from pathlib import Path

import pytest
import torch
from model import HybridPilot
from experiments import p61_hidden_motor_polarity_dev as p


def gitblob(path):
    data=Path(path).read_bytes()
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def manual():
    h=torch.zeros((2,6,5))
    h[:,:,0]=torch.tensor([[1.,-1.,1.,-1.,1.,1.],[-1.,-1.,1.,1.,-1.,1.]])
    h[:,:,1]=h[:,:,0]*torch.tensor([[-.18], [.18]])
    h[:,:,2]=torch.arange(1,7)/6
    return {'history':h,'actual_query_action':torch.tensor([[1.],[-1.]]),
            'actual_observed_reward':torch.tensor([[-1.],[-1.]])}


def test_original_model_byte_custody_and_all_four_arch_components():
    assert gitblob(Path(__file__).resolve().parents[1]/'model.py')=='5d9ea57b827aa1796548447868fa5ee14c251b49'
    m=p.HybridQ()
    assert isinstance(m.body,HybridPilot)
    assert all(getattr(m.body,z) is not None for z in ('stem','ssm','attn','ttt','depth','head'))
    assert m.body.cfg.width==8 and m.body.cfg.heads==2


def test_candidate_query_has_no_future_outcome_or_hidden_polarity():
    d=manual();q=torch.tensor([[[-1.],[1.]]]).expand(2,-1,-1)
    observed=p.factual_input(d,q)
    assert observed.shape==(2,2,7,5)
    assert torch.equal(observed[:,:,:6,3:],torch.zeros_like(observed[:,:,:6,3:]))
    assert torch.equal(observed[:,:,-1,0:2],torch.zeros_like(observed[:,:,-1,0:2]))
    assert (observed[:,:,-1,3]==1).all()
    assert observed[0,:,-1,4].tolist()==[-1.,1.]
    assert 'evaluator_hidden_final_mode' not in p.public_data(dict(d,evaluator_hidden_final_mode=torch.ones((2,1))))


def test_wrong_credit_changes_only_completed_actual_action_fields():
    d=manual();q=torch.ones((2,1,1))
    a=p.factual_input(d,q)
    b=p.factual_input(d,q,wrong_credit=True)
    assert not torch.equal(a[:,:,0:6,0],b[:,:,0:6,0])
    assert torch.equal(a[:,:,0:6,1:],b[:,:,0:6,1:])
    assert torch.equal(a[:,:,-1,:],b[:,:,-1,:])
    assert torch.equal(d['history'][:,:,0],a[:,0,:6,0])


def test_causal_baseline_only_uses_actual_executed_probes():
    d=manual()
    d['evaluator_hidden_final_mode']=torch.tensor([[-1.],[1.]])
    d['evaluator_switched']=torch.tensor([True,False])
    assert p.score_analytical(d,3)['success']==1.
    d2={k:v.clone() for k,v in d.items()}
    d2['history'][:,:,1]=0.
    assert p.score_analytical(d2,3)['success']==.5


def test_original_hybrid_query_order_and_weights_unchanged():
    torch.manual_seed(71)
    d=manual();m=p.HybridQ().eval()
    q=torch.tensor([[[-1.],[1.]]]).expand(2,-1,-1)
    weights={k:v.detach().clone() for k,v in m.state_dict().items()}
    first=m(d,q)
    reverse=m(d,q.flip(1))
    torch.testing.assert_close(first.flip(1),reverse,atol=1e-6,rtol=0)
    assert all(torch.equal(v,weights[k]) for k,v in m.state_dict().items())


def test_cold_restoration_bit_identical_on_manual_history():
    torch.manual_seed(18)
    d=manual();m=p.HybridQ().eval()
    buf=io.BytesIO();torch.save(m.state_dict(),buf);buf.seek(0)
    cold=p.HybridQ().eval();cold.load_state_dict(torch.load(buf,weights_only=True))
    a=torch.tensor([[[-1.],[1.]]]).expand(2,-1,-1)
    assert torch.equal(m(d,a),cold(d,a))
    assert torch.equal(d['history'],d['history'].clone())


def test_invalid_factual_inputs_and_unregistered_cohort_fail_closed():
    d=manual();q=torch.ones((2,1,1))
    with pytest.raises(ValueError,match='P61_BAD_FACTUAL_HISTORY_OR_QUERY'):
        p.factual_input(dict(d,history=d['history'][:,:5]),q)
    with pytest.raises(ValueError,match='P61_NONFINITE_ACTUAL_FACT'):
        bad=d['history'].clone();bad[0,0,1]=float('nan');p.factual_input(dict(d,history=bad),q)
    with pytest.raises(ValueError,match='P61_QUERY_MUST_BE_REAL_MOTOR_CHOICE'):
        p.factual_input(d,torch.zeros_like(q))
    with pytest.raises(ValueError,match='P61_UNREGISTERED_SOURCE'):
        p.source('protected_unseen_science')


def test_frozen_seeds_and_budgets_without_generating_any_cohort():
    assert set(p.SPLITS)=={'fit','val','unfamiliar_dev'}
    assert p.SPLITS['fit']==(991101,4096,.5,.24)
    assert p.SPLITS['unfamiliar_dev']==(991107,1024,.75,.30)
    assert p.MODEL_SEEDS==(992101,992103,992107)
    assert p.UPDATES==480 and p.BATCH==128 and p.PROBES==6
    assert p.param_count(p.HybridQ())==1282


def test_wrong_credit_frozen_cyclic_shift_is_reversible_diagnostic_limitation():
    # This predeclared wrong-credit arm is a *weak* control because a fixed
    # rotation of observed historical action identities is exactly invertible.
    # Do not infer lack of causal use from its preserved performance.
    d=manual();q=torch.ones((2,1,1))
    z=p.factual_input(d,q,wrong_credit=True)
    recovered=torch.roll(z[:,0,:6,0],shifts=-1,dims=1)
    assert torch.equal(recovered,d['history'][:,:,0])


def test_consumed_authored_source_replay_hash_and_true_current_aliasing():
    d, digest=p.source('unfamiliar_dev')
    assert digest=='a151ee31fae9814a2c7e5bba8c199a1f065bf5f237365f5b46bae429c857763b'
    assert d['history'].shape==(1024,6,5)
    assert 'evaluator_hidden_final_mode' not in p.public_data(d)
    assert torch.unique(d['evaluator_hidden_final_mode']).tolist()==[-1.,1.]
    assert (d['evaluator_switched'].float().mean()>.70)
    assert torch.equal(d['actual_observed_reward'].abs(),torch.ones_like(d['actual_observed_reward']))
