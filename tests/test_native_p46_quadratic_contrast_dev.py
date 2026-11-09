"""P46 no-source DEV controls, not independent scientific acceptance."""
import hashlib
import inspect
from pathlib import Path
import pytest
import torch
from scripts import native_p46_quadratic_contrast_dev as p46

ROOT=Path(__file__).resolve().parents[1]
EXPECTED={
    'train':'4f942b4b8535d8c5804aba7d0747e0141bcfe85a3c4fda0f0f2ed31d1f55a395',
    'val':'c83797d051c884178b65db4ccdaf2fd2a2dfd31ac6d8b1caf9bc1c59bb6d0d9a',
    'ood':'18cb491b482fb527e752e75307f304e22e6386326ec25336703ebe118ace9f85',
}

def test_original_unchanged_p43_p44_p45_git_blobs():
    blobs=[('native_trainable_causal_belief.py','49e8d1e84ae2ba02956830b55c8c9480a37c509f'),
           ('scripts/native_p44_compare_dev.py','aafc9b44e761431f4213985baf05a237fa14f4f6'),
           ('scripts/native_p45_intervention_contrast_dev.py','363996307bd90b4effc2ba7acae81c241e6492da')]
    for path,sha in blobs:
        raw=(ROOT/path).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==sha


def test_frozen_seeds_exact_factual_source_action_and_mode_switch():
    assert p46.SOURCE_SEEDS=={'train':750001,'val':750101,'ood':750201}
    assert p46.COUNTS=={'train':3072,'val':384,'ood':384}
    for cohort,digest in EXPECTED.items():
        d=p46.source(cohort)
        assert d['source_sha256']==digest
        n=p46.COUNTS[cohort]
        assert d['history'].shape==(n,8,4)
        assert d['factual_query_y'].shape==(n,1)
        assert d['evaluator_both'].shape==(n,2)
        assert d['switched'].sum().item()==(96 if cohort=='ood' else 0)
        assert bool(torch.isfinite(d['history']).all())
        a=d['history'][:,3:,2]
        assert torch.equal(a[:,0],a[:,2]) and torch.equal(a[:,2],a[:,4])
        assert torch.equal(a[:,0],-a[:,1]) and torch.equal(a[:,1],a[:,3])
    with pytest.raises(ValueError,match='P46_UNKNOWN_COHORT'):
        p46.source('previous_original_science')


def test_five_factual_interventions_cancel_cubic_but_not_quartic():
    n=3
    h=torch.zeros((n,8,4),dtype=torch.float32)
    t=torch.arange(8,dtype=torch.float32)/8
    aa=torch.tensor([1.,-1.,1.,-1.,1.])
    h[:,3:8,2]=aa
    beta=.31
    h[:,:,1]=.2-.1*t+.6*t*t+beta*h[:,:,2]
    assert p46.degree_two_contrast({'history':h}).flatten().tolist()==pytest.approx([beta]*n,abs=2e-6)
    # Symmetric alternating pattern also cancels cubic (odd component).
    h[:,:,1] += .85*t**3
    assert p46.degree_two_contrast({'history':h}).flatten().tolist()==pytest.approx([beta]*n,abs=2e-6)
    h[:,:,1] += .85*t**4
    assert float((p46.degree_two_contrast({'history':h})-beta).abs().mean())>0.00001


def test_no_heldout_oracle_in_training_objective_and_equal_budget():
    fn=inspect.getsource(p46.training)
    assert 'factual_query_y' in fn
    assert 'evaluator_both' not in fn
    assert "['effect']" not in fn
    assert p46.SEEDS==(9503,9511,9521)
    assert p46.UPDATES==1200 and p46.BATCH==128 and p46.INDEX_SEED==751001
    assert p46.p44.params(p46.p44.OriginalP43())==1394
    assert p46.p44.params(p46.p44.GRUConventional())==1393


def test_five_contrast_refuses_non_alternating_action_credit():
    d=p46.source('val');c={**d,'history':d['history'].clone()}
    c['history'][:,4,2]=c['history'][:,3,2]
    with pytest.raises(ValueError,match='P46_NOT_FIVE_ALTERNATING_FACTS'):
        p46.degree_two_contrast(c)


def test_p43_cold_state_replay_exact_no_speculative_commit():
    torch.manual_seed(46)
    model=p46.p44.OriginalP43().eval()
    d=p46.source('val')
    ok,digest=p46.p44.cold(model,d)
    assert ok and len(digest)==64
