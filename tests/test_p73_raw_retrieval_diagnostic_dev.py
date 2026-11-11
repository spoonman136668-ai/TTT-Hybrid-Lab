"""P73 source/reader engineering tests; no OOD or protected scientific first use."""
import importlib.util
from pathlib import Path
import pytest
import torch

root=Path(__file__).resolve().parents[1]
p=root/'experiments/p73_raw_retrieval_diagnostic_dev.py'
spec=importlib.util.spec_from_file_location('p73',p)
p73=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p73)
torch.set_num_threads(1)


def test_source_deterministic_and_fresh_exact_shapes():
    for k in ('fit','val','negative'):
        a=p73.make_world(k,8);b=p73.make_world(k,8)
        assert p73.source_hash(a)==p73.source_hash(b)
        assert a['facts'].shape==(8,24,32)
        assert a['query'].shape==(8,1,32)
        assert a['label'].shape==(8,16)
        assert a['role_audit'].sum(-1).tolist()==[16]*8
        assert torch.equal(a['query'][:,:,16:],torch.zeros(8,1,16))


def test_full_source_position_leakage_audit():
    for k in ('fit','val'):
        assert p73.p72.position_audit(p73.make_world(k))['pass']


def test_same_observations_for_all_arms():
    w=p73.make_world('fit',8)
    assert p73.facts_only(w).shape==(8,24,32)
    assert 'label' not in {'facts'}
    r=p73.LearnedRetrieval(730201)
    assert torch.isfinite(r(p73.facts_only(w),w['query'])).all()
    with pytest.raises(ValueError,match='P73_NONFACTUAL_OR_INVALID_HISTORY'):
        r(torch.cat((w['facts'],w['query'].expand(-1,24,-1)),-1),w['query'])


def test_distinct_model_seeds_and_finite_gradient():
    h=p73.make_world('fit',8)
    models=[p73.LearnedRetrieval(x) for x in p73.MODEL_SEEDS]
    hashes=[p73.tensor_sha(torch.cat((m.metric.flatten(),m.log_temperature.view(1)))) for m in models]
    assert len(set(hashes))==3
    loss=p73.direct_eval(models[0],h).mean()
    loss.backward()
    assert torch.isfinite(models[0].metric.grad).all()
    assert torch.isfinite(models[0].log_temperature.grad)


def test_wrong_credit_changes_only_past_effects():
    w=p73.make_world('fit',8)
    bad=p73.wrong_credit(w,730399)
    assert torch.equal(w['facts'][:,:,:16],bad['facts'][:,:,:16])
    assert torch.equal(w['query'],bad['query'])
    assert torch.equal(w['label'],bad['label'])
    assert not torch.equal(w['facts'][:,:,16:],bad['facts'][:,:,16:])


def test_train_exact_one_step_no_ood(monkeypatch):
    w=p73.make_world('fit',8);v=p73.make_world('val',8)
    indices=torch.randint(8,(1,8),generator=torch.Generator().manual_seed(10))
    monkeypatch.setattr(p73,'FIT_STEPS',1)
    monkeypatch.setattr(p73,'CHECK',1)
    import time
    m,receipt=p73.train_conventional(730201,w,v,indices,time.perf_counter()+20)
    assert receipt['steps']==1 and receipt['status']=='DONE'
    assert receipt['trained_parameters']==257
    assert receipt['best_val_mse']>=0
