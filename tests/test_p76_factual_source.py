"""P76 source admission engineering checks; no OOD cohort generation."""
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import p76_factual_source as new
import p69_causal_shift_source as old


def test_fresh_p76_seed_namespace_and_parent_no_mutation():
    assert new.PARENT_BLOB=='51d4c4f16e18e1fb8265a92eee3d5f70b860893b'
    assert old.SEEDS['fit']==981101 and new.SPLIT_SEEDS['fit']==976101
    a=new.make_world('fit',0)
    assert new.digest_world(a)==new.digest_world(new.make_world('fit',0))
    assert new.digest_world(a)!=old.digest_world(old.make_world('fit',0))
    assert a['action'].shape==(96,16)


def test_never_admit_holdout_sources():
    for split in ('ood_0','ood_1','ood_2','negative'):
        with pytest.raises(ValueError,match='P76_HOLDOUT_NOT_ADMITTED'):
            new.make_world(split,0)
        with pytest.raises(ValueError,match='P76_HOLDOUT_NOT_ADMITTED'):
            new.make_batch(split,8)


def test_fit_val_independent_and_privileged_labels_removed():
    fit=new.make_batch('fit',8)
    val=new.make_batch('val',8)
    assert not torch.equal(fit['action'],val['action'])
    assert fit['action'].shape==(8,96,16)
    assert set(new.factual(fit))=={'action','effect'}
    assert fit['query_changed_audit'].dtype==torch.bool


def test_source_code_contains_no_training_or_scoring_entrypoint():
    raw=(Path(__file__).resolve().parents[1]/'experiments'/'p76_factual_source.py').read_text()
    assert 'if __name__' not in raw and 'optimizer' not in raw and 'study(' not in raw
