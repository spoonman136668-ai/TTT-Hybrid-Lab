"""P45 exact-source development controls; no independent source first use."""
import hashlib
import inspect
from pathlib import Path
import numpy as np
import pytest
import torch
from scripts import native_p45_intervention_contrast_dev as p45

PINNED_RUNNER_BLOB = '363996307bd90b4effc2ba7acae81c241e6492da'
EXPECTED_SHA = {
    'fit': '8dd6100cde77f76a0cf265a5705a1e7eb2e0926c65dc7b73fde22249ba60fc99',
    'validation': '9b219ee2ae2a2fecb9f24b545d0a706cbb9b378ddeefc26cbd072b9a829f204a',
    'ood_dev': 'fa62560b3adc5096d7c0f902be9d132ef53f215611cd2574694f8a2f6c495c39',
}


def test_original_p43_and_p44_ancestor_source_pinned():
    repo = Path(__file__).resolve().parents[1]
    for path,expected in [
        ('native_trainable_causal_belief.py','49e8d1e84ae2ba02956830b55c8c9480a37c509f'),
        ('scripts/native_p44_compare_dev.py','aafc9b44e761431f4213985baf05a237fa14f4f6'),
        ('scripts/native_p45_intervention_contrast_dev.py',PINNED_RUNNER_BLOB),
    ]:
        raw=(repo/path).read_bytes()
        actual=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\x00'+raw).hexdigest()
        assert actual==expected


def test_frozen_cohorts_are_disjoint_and_source_actions_factual():
    assert (p45.COUNTS,p45.SOURCE_SEEDS) == (
        {'fit':3072,'validation':384,'ood_dev':384},
        {'fit':740001,'validation':740101,'ood_dev':740201})
    assert len(set(p45.SOURCE_SEEDS.values()))==3
    for cohort,digest in EXPECTED_SHA.items():
        data=p45.source(cohort)
        assert data['source_sha256']==digest
        n=p45.COUNTS[cohort]
        assert data['history'].shape==(n,8,4)
        assert data['factual_query_y'].shape==(n,1)
        assert data['evaluator_both'].shape==(n,2)
        assert ((data['history'][:,:,2].abs()==1).all())
        assert ((data['query_action'].abs()==1).all())
        assert torch.isfinite(data['history']).all()
        assert data['switched'].sum().item()==(96 if cohort=='ood_dev' else 0)
    with pytest.raises(ValueError,match='P45_UNREGISTERED_DEV_SOURCE'):
        p45.source('original_protected_science')


def test_strong_factual_ABA_only_consumes_completed_history():
    data=p45.source('ood_dev')
    first=data['history'][:,5,2]
    assert torch.equal(first,-data['history'][:,6,2])
    assert torch.equal(first,data['history'][:,7,2])
    aba=p45.aba_from_completed_facts(data)
    # The teacher is computed from actual observed intervention outcomes;
    # it is not directly supplied the evaluator-only true causal effect.
    alternative={**data, 'effect':-data['effect'],
                 'evaluator_both':data['evaluator_both'].flip(1)}
    assert torch.equal(aba,p45.aba_from_completed_facts(alternative))
    assert float((aba-data['effect']).abs().mean())==pytest.approx(.0162818376,abs=1e-6)
    corrupt={**data,'history':data['history'].clone()}
    corrupt['history'][:,6,2]=corrupt['history'][:,5,2]
    with pytest.raises(AssertionError):
        p45.aba_from_completed_facts(corrupt)


def test_only_actual_query_outcome_is_used_as_supervised_target():
    train_source=inspect.getsource(p45.fit)
    assert "sub['factual_query_y']" in train_source
    assert 'evaluator_both' not in train_source
    assert 'sub[\'effect\']' not in train_source
    assert p45.UPDATES==1000 and p45.BATCH==128 and p45.LAMBDA==1.0
    assert p45.SEEDS==(9403,9413,9421)
    assert p45.INDEX_SEED==741001


def test_no_new_model_parameters_and_original_cold_source():
    for name,model in [('p43',p45.prev.OriginalP43()),
                       ('gru',p45.prev.GRUConventional())]:
        expected=1394 if name=='p43' else 1393
        assert p45.prev.params(model)==expected
    torch.manual_seed(45)
    model=p45.prev.OriginalP43().eval()
    state=p45.prev.batch_data(p45.source('validation'),torch.arange(3))
    ok,sha=p45.prev.cold(model,state)
    assert ok and len(sha)==64
