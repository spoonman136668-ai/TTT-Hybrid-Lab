"""DEV-only source/test interface controls; not original repo acceptance."""
from pathlib import Path
import hashlib
import torch
import pytest
from scripts import native_p44_compare_dev as q
from native_trainable_causal_belief import NativeTrainableCausalBelief


def test_p43_exact_source_blob_and_capacity_parity():
    original=(Path(__file__).resolve().parents[1]/'native_trainable_causal_belief.py').read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(original)).encode()+b'\0'+original).hexdigest()
    assert blob=='49e8d1e84ae2ba02956830b55c8c9480a37c509f'
    p=q.params(q.OriginalP43());g=q.params(q.GRUConventional());m=q.params(q.MLPConventional())
    assert (p,g,m)==(1394,1393,1281)
    assert max(p,g,m)/min(p,g,m) < 1.10


def test_deterministic_disjoint_development_cohorts_and_no_extra_query():
    fixed={'fit':'e43f58a7c63c3e5b77f4d8a16c9fe05cc1b35a7e6d66b9a5380e96f636951580',
           'validation':'8bcdb9ce30ce2f558df002008cc064854f0af5cd636ae6757194fde1bdb5d091',
           'unfamiliar_dev':'ad3c019a772f5585adccc056415ff198f68b8341c725f47738eb4ff66ad41d29'}
    for cohort,digest in fixed.items():
        s=q.source(cohort)
        assert s['source_sha256']==digest
        assert s['query_action'].shape==(q.COUNTS[cohort],1)
        assert s['factual_query_y'].shape==(q.COUNTS[cohort],1)
        assert s['evaluator_both'].shape==(q.COUNTS[cohort],2)
        assert (s['history'][:,:,2].abs()==1).all()
        assert torch.isfinite(s['history']).all()
        if cohort=='unfamiliar_dev':
            assert s['switched'].sum().item()==128
        else:
            assert s['switched'].sum().item()==0
    with pytest.raises(ValueError,match='P44_NONREGISTERED_SOURCE_COHORT'):
        q.source('original_science')


def test_only_factual_observed_queries_train_original_p43_and_gru_control():
    # Hand-authored two-step calibration: not one of the sealed/DEV source seeds.
    torch.manual_seed(44)
    for family in (q.OriginalP43,q.GRUConventional,q.MLPConventional):
        model=family()
        history=torch.tensor([[[0., .12, 1.,.1173],[.12,-.08,-1.,.1173],
              [.04,.14,1.,.1173],[.18,-.06,-1.,.1173],[.12,.12,1.,.1173],
              [.24,-.07,-1.,.1173],[.17,.13,1.,.1173],[.30,-.06,-1.,.1173]]])
        demo={'history':history,'query_sensor':torch.tensor([[.24]]),
              'query_action':torch.tensor([[1.]]),'factual_query_y':torch.tensor([[.39]])}
        pred=model(demo,demo['query_action'])[:,0,:]
        loss=(pred-demo['factual_query_y']).square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert pred.shape==(1,1)
        pre=[p.clone().detach() for p in model.parameters()]
        optim=torch.optim.SGD(model.parameters(),lr=.001);optim.step()
        assert any(not torch.equal(x,p) for x,p in zip(pre,model.parameters()))


def test_read_only_candidate_order_and_original_exact_cold_state():
    torch.manual_seed(42)
    model=q.OriginalP43().eval()
    world=q.source('validation'); s=q.batch_data(world,torch.arange(3))
    initial=model.factual_state(s)
    frozen=(initial.hidden.clone(),initial.elapsed.clone(),initial.observations.clone())
    a=torch.tensor([[[-1.],[1.]]]).expand(3,-1,-1)
    pred=model.native.preview(s['query_sensor'],a,torch.full((3,1),q.DT),initial).predictions
    flipped=model.native.preview(s['query_sensor'],a.flip(1),torch.full((3,1),q.DT),initial).predictions
    torch.testing.assert_close(pred,flipped.flip(1),atol=0,rtol=0)
    assert all(torch.equal(x,y) for x,y in zip(frozen,(initial.hidden,initial.elapsed,initial.observations)))
    ok,sha=q.cold(model,world)
    assert ok and len(sha)==64


def test_causal_null_control_and_ridge_interference_receipt():
    cohort=q.source('unfamiliar_dev')
    x=q.ridge(cohort)
    assert x['rank_full']==507 and x['switch']['n']==128 and x['no_switch']['n']==384
    assert x['switch']['effect_mae']>x['no_switch']['effect_mae']
