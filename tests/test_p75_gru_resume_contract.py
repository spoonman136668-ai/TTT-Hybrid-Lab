"""Read-only/resume-fence tests; no experimental dataset or training accessed."""
import sys
from pathlib import Path
import pytest
EXPERIMENTS=Path(__file__).resolve().parents[1]/'experiments'
sys.path.insert(0,str(EXPERIMENTS))
import p75_gru_resume as resume


def test_invalid_seed_mode_fails_before_world_birth(tmp_path,monkeypatch):
    def unexpected(*a,**kw):
        raise AssertionError('ILLEGAL_SOURCE_BIRTH')
    monkeypatch.setattr(resume,'make_batch',unexpected)
    for seed,mode in ((999999,'final_only'),(975201,'oracle'),(975203,'random_mask')):
        with pytest.raises(ValueError,match='P75_UNREGISTERED_TRAINING'):
            resume.run_chunk(seed,mode,tmp_path)
    assert not list(tmp_path.iterdir())


def test_code_blob_drift_fails_before_training(tmp_path,monkeypatch):
    def unexpected(*a,**kw):
        raise AssertionError('ILLEGAL_SOURCE_BIRTH')
    monkeypatch.setattr(resume,'make_batch',unexpected)
    monkeypatch.setitem(resume.LOCKED,'p75_prequential_baselines.py','badsha')
    with pytest.raises(RuntimeError,match='P75_SOURCE_HASH_DRIFT'):
        resume.run_chunk(975201,'final_only',tmp_path)
    assert not list(tmp_path.iterdir())


def test_resume_has_fixed_factual_worlds_and_no_external_runner():
    text=(EXPERIMENTS/'p75_gru_resume.py').read_text()
    assert "make_batch('fit',512),make_batch('val',192)" in text
    assert 'ood_0' not in text and 'ood_1' not in text and 'ood_2' not in text
    assert 'torch.load(statefile,map_location=' in text
    assert "c.get('sample_indices_sha256')!=sample_hash" in text
    assert 'subprocess' not in text and 'requests' not in text
