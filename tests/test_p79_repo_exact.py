"""P79 frozen source/protocol tests; no OOD or negative cohort generation."""
import hashlib
from pathlib import Path
import importlib.util
import pytest
import torch

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('p79_reproducible',root/'experiments/p79_reproducible_from_p78.py')
p79=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p79)
torch.set_num_threads(1)


def git_blob(path):
    b=path.read_bytes()
    return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()


def test_exact_original_reference_custody():
    assert git_blob(root/'model.py')=='5d9ea57b827aa1796548447868fa5ee14c251b49'
    assert git_blob(root/'native_p78_bounded_raw_ttt.py')=='db948c53bb1a28af5fe0e6a62ef71463db6d95ba'


def test_new_frozen_seed_registry_and_training_schedule():
    assert p79.SEEDS=={'fit':790111,'val':790113,'ood_dev':790117,'negative':790119}
    assert p79.N=={'fit':256,'val':128,'ood_dev':128,'negative':64}
    assert p79.CORE_SEEDS==(790201,790203,790207)
    assert (p79.UPDATES,p79.BATCH,p79.CHECK)==(420,16,30)


def test_fit_val_determinism_without_ood_generation():
    for kind in ('fit','val'):
        a=p79.make_world(kind,8)
        b=p79.make_world(kind,8)
        assert p79.world_sha(a)==p79.world_sha(b)
        assert a['facts'].shape==(8,24,32)
        assert a['query'].shape==(8,1,32)
        assert a['role_audit'].sum(-1).tolist()==[16]*8
        assert torch.equal(a['query'][:,:,16:],torch.zeros(8,1,16))


def test_full_fit_val_source_position_controls():
    for kind in ('fit','val'):
        assert p79.audit(p79.make_world(kind))['pass']


def test_invalid_source_fails_closed():
    with pytest.raises(ValueError,match='P78_UNDECLARED_SPLIT'):
        p79.make_world('invalid_ood')
    with pytest.raises(ValueError,match='P78_BAD_WORLD_COUNT'):
        p79.make_world('val',0)


def test_256_floats_per_world_and_bitwise_cold_restore():
    w=p79.make_world('val',8)
    assert p79.cold_equivalence(w)
    model=p79.core(790201)
    x=p79.features(model,w)
    assert x['raw_state'].numel()==8*256
    assert x['native_state'].numel()==8*256


def test_all_three_original_initial_weights_distinct_but_repeatable():
    assert len({p79.core_sha(p79.core(s)) for s in p79.CORE_SEEDS})==3
    assert p79.core_sha(p79.core(790201))==p79.core_sha(p79.core(790201))


def test_frozen_trainable_seam_only():
    m=p79.core(790201)
    assert all(not p.requires_grad for p in m.stem.parameters())
    assert all(not p.requires_grad for p in m.ttt.key.parameters())
    assert all(p.requires_grad for p in m.head.parameters())
    assert all(p.requires_grad for p in m.ttt.proj.parameters())
