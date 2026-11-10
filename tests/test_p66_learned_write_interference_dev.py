"""P66 source-isolated engineering invariants; no unfamiliar DEV source consumed."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
import torch
import pytest
from experiments import p66_learned_write_interference_dev as p66


def tiny():
    actions=torch.zeros(2,4,16);actions[:,:,0]=1.
    effects=torch.zeros_like(actions);effects[:,:,1]=.4
    q=actions[:,:8].clone() if False else actions[:,:1].expand(-1,8,-1).clone()
    v=effects[:,:1].expand(-1,8,-1).clone()
    wrong=torch.zeros_like(actions);wrong[:,:,2]=1.
    return dict(action=actions,effect=effects,target_key=q,target_value=v,
                wrong_action=wrong,evaluator_target=torch.tensor([[1,1,0,0],[1,0,1,0]],dtype=torch.bool))


def test_reference_equation_and_learned_gate():
    d=tiny();m=p66.LearnedGate()
    w,_,_=m(d,hi=1,mode='all')
    expected=0.5*d['effect'][:,0,:,None]*d['action'][:,0,None,:]
    assert torch.equal(w,expected)
    learned,_,_=m(d,hi=1)
    assert torch.equal(learned,expected*.5)


def test_read_is_pure_and_invalid_range_denied():
    d=tiny();m=p66.LearnedGate();w,_,_=m(d,mode='all')
    saved=w.clone();q=p66.read(w,d['target_key'])
    assert torch.equal(w,saved) and q.shape==(2,8,16)
    with pytest.raises(ValueError,match='P66_INVALID_RANGE'):
        m(d,lo=3,hi=2)


def test_cold_resume_exact_and_oracle_not_hidden():
    d=tiny();m=p66.LearnedGate()
    yes,sha=p66.cold_exact(m,d)
    assert yes and len(sha)==64
    w,frac,_=m(d,mode='oracle')
    assert 0.<frac<1.
    bad={k:v.clone() for k,v in d.items()}
    bad['evaluator_target'][:]=False
    w0,_,_=m(bad,mode='oracle')
    assert torch.count_nonzero(w0)==0
    assert torch.count_nonzero(w)>0


def test_deterministic_fit_source_and_action_credit_corruption():
    a,ha=p66.source('fit');b,hb=p66.source('fit')
    assert ha==hb and torch.equal(a['action'],b['action'])
    assert not torch.equal(a['action'],a['wrong_action'])
    assert a['evaluator_target'].shape==(192,96)
    assert a['evaluator_target'].sum(dim=1).unique().tolist()==[32]
    with pytest.raises(ValueError,match='P66_UNKNOWN_SOURCE'):
        p66.source('protected')


def test_trainable_gradient_and_separate_gru():
    d=tiny();m=p66.LearnedGate();loss=p66.objective(m,d,'gate')
    loss.backward();assert torch.isfinite(m.theta.grad).all()
    gru=p66.TrainedGRU();loss=p66.objective(gru,d,'gru')
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in gru.parameters())
