"""P75 FIT/VAL ENGINEERING samples only; OOD never called."""
import sys
from pathlib import Path
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]/'experiments'
sys.path.insert(0,str(ROOT))
import p69_causal_shift_source as old
import p75_factual_source as fresh


def test_source_parent_blob_pinned_and_seeds_independent():
    assert fresh.SOURCE_BLOB=='51d4c4f16e18e1fb8265a92eee3d5f70b860893b'
    assert fresh.SEEDS['fit']==975101
    assert old.SEEDS['fit']==981101  # no module-global mutation
    assert old._seed('fit',0)!=fresh._source._seed('fit',0)


def test_source_is_fresh_and_deterministic_for_fit_and_validation():
    for split in ('fit','val'):
        a=fresh.make_world(split,0)
        b=fresh.make_world(split,0)
        assert fresh.digest_world(a)==fresh.digest_world(b)
        assert fresh.digest_world(a)!=old.digest_world(old.make_world(split,0))
        assert a['action'].shape==(96,16)
        assert len(a['changed_roles_audit'])==4
        assert set(fresh.factual_history(a))=={'action','effect'}


def test_small_engineering_split_disjoint_and_audited():
    fit=fresh.make_batch('fit',8)
    val=fresh.make_batch('val',8)
    assert fit['action'].shape==val['action'].shape==(8,96,16)
    assert not torch.equal(fit['action'],val['action'])
    assert fit['query_action'].shape==(8,16)
    assert set(fresh.factual_history(fit))=={'action','effect'}


def test_bad_selectors_do_not_leak_sealed_data():
    with pytest.raises(ValueError,match='P75_UNADMITTED_SOURCE'):
        fresh.make_world('train',0)
    for sealed in ('ood_0','ood_1','ood_2','negative'):
        with pytest.raises(ValueError,match='P75_UNADMITTED_SOURCE'):
            fresh.make_world(sealed,0)
        with pytest.raises(ValueError,match='P75_UNADMITTED_SOURCE'):
            fresh.make_batch(sealed,8)
    with pytest.raises(ValueError,match='P69_BAD_BATCH'):
        fresh.make_batch('val',193)


def test_unqualified_science_runner_is_absent():
    path=ROOT/'p75_factual_source.py'
    text=path.read_text()
    assert 'if __name__' not in text
    assert 'torch.save' not in text
    assert 'study(' not in text
