"""Source-free no-training controls for P56. Registered heldout worlds are not generated here."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pytest
import torch
from model import Configuration, HybridPilot
from native_factual_hybrid_stream import NativeFactualHybridStream
from native_p54_mountaincar_factual_bridge import published_step
from scripts import native_p56_mountaincar_factual_learning_dev as p56

torch.set_num_threads(1)


def gitblob(path):
    a=Path(path).read_bytes()
    return hashlib.sha1(b'blob '+str(len(a)).encode()+b'\0'+a).hexdigest()


def tiny(n=2):
    s=np.array([-.5,0.],np.float32)
    data=[]
    for _ in range(n):
        h=[]
        for a in (0.,1.,-1.,0.,1.,0.,-1.,0.):
            obs=published_step(s,a).next_sensor
            h.append(np.concatenate((s,obs-s,np.array([a,1.,0.],np.float32))))
            s=obs
        data.append(dict(history=torch.from_numpy(np.stack(h)),sensor=torch.from_numpy(s.copy())))
    return {'history':torch.stack([x['history'] for x in data]), 'sensor':torch.stack([x['sensor'] for x in data])}


def test_existing_exact_original_p55_source_custody():
    root=Path(__file__).resolve().parents[1]
    assert gitblob(root/'model.py')=='5d9ea57b827aa1796548447868fa5ee14c251b49'
    assert gitblob(root/'native_factual_hybrid_stream.py')=='731ac511090f1c3ca83c2bf784591e604cfc1a87'
    assert gitblob(root/'native_p54_mountaincar_factual_bridge.py')=='9ecfc5822989336b3bbb14f29b8ebf84296c7039'


def test_original_reference_pilot_is_identical_in_p56_not_new_architecture():
    torch.manual_seed(561)
    model=p56.Hybrid()
    assert isinstance(model.pilot,HybridPilot)
    assert all(getattr(model.pilot,n) is not None for n in ['stem','ssm','attn','ttt','depth','head'])
    assert model.pilot.head.out_features==2
    assert len(model.state_dict())>20


def test_history_contains_only_completed_physical_actions_and_no_future_outcome():
    d=tiny()
    candidates=torch.tensor([[[-1.],[0.],[1.]]]).expand(2,-1,-1).clone()
    z=p56.form(d['history'],d['sensor'],candidates)
    assert z.shape==(2,3,9,7)
    assert z[:,:,-1,2:4].count_nonzero()==0
    assert (z[:,:,-1,6]==1).all()
    assert (z[:,:,:8,6]==0).all()
    assert z[0,:,4,4].tolist()==[1.,1.,1.]
    assert z[0,:,-1,4].tolist()==[-1.,0.,1.]


def test_candidate_order_does_not_change_actual_observed_history_or_weights():
    torch.manual_seed(8)
    model=p56.Hybrid().eval();d=tiny()
    q=torch.tensor([[[-1.],[0.],[1.]]]).expand(2,-1,-1)
    before=d['history'].clone();weights={k:v.clone() for k,v in model.state_dict().items()}
    f=model(d,q);r=model(d,q.flip(1))
    torch.testing.assert_close(f.flip(1),r,atol=0,rtol=0)
    assert torch.equal(d['history'],before)
    assert all(torch.equal(v,weights[k]) for k,v in model.state_dict().items())


def test_wrong_credit_corrupts_only_historical_executed_action_not_realized_outcomes():
    d=tiny();q=torch.ones((2,1,1))
    a=p56.form(d['history'],d['sensor'],q,False)
    b=p56.form(d['history'],d['sensor'],q,True)
    assert not torch.equal(a[:,:,0:8,4],b[:,:,0:8,4])
    assert torch.equal(a[:,:,0:8,2:4],b[:,:,0:8,2:4])
    assert torch.equal(a[:,:,-1,:],b[:,:,-1,:])


def test_zero_action_is_a_legitimate_completed_physical_fact():
    d=tiny();assert (d['history'][:,:,4]==0).any()
    query=torch.zeros((2,1,1))
    out=p56.Hybrid().eval()(d,query)
    assert out.shape==(2,1,2) and bool(torch.isfinite(out).all())


def test_invalid_shape_and_nonfinite_factual_source_refuse():
    d=tiny();q=torch.zeros((2,1,1))
    with pytest.raises(ValueError,match='P56_BAD_COMPLETED_FACTUAL_HISTORY_OR_QUERY'):
        p56.form(d['history'][:,:6],d['sensor'],q)
    bad=d['history'].clone();bad[1,5,3]=float('nan')
    with pytest.raises(ValueError,match='P56_NONFINITE_ACTUAL_SOURCE'):
        p56.form(bad,d['sensor'],q)


def test_manual_factual_cold_weights_and_history_bit_exact():
    torch.manual_seed(4)
    d=tiny();m=p56.Hybrid().eval()
    assert p56.cold(m,dict(d,executed_query_action=torch.zeros((2,1))))


def test_strong_original_published_mechanics_is_zero_error_for_own_source():
    p=np.array([-.5,0.],np.float32)
    q=[published_step(p,float(a)).next_sensor for a in p56.CHOICES]
    y=torch.from_numpy(np.stack(q)[None,:,:])
    a=p56.metrics(y,y)
    assert a=={'mse':0.,'effect_mae':0.,'neutral_mse':0.}


def test_frozen_registered_sources_and_optimization_budget():
    assert p56.SPLITS=={'fit':(966101,768),'val':(966103,192),'ood':(966107,256)}
    assert p56.MODEL_SEEDS==(976101,976103,976107)
    assert p56.SAMPLE_SEED==976199
    assert p56.UPDATES==200 and p56.BATCH==64
    for bad in ('other','heldout','train','966107'):
        with pytest.raises(ValueError,match='P56_UNREGISTERED_SOURCE_FIRST_USE'):
            p56.source(bad)
