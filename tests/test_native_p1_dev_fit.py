"""Small no-science negative tests; full DEV training runs separately."""
import ast
from pathlib import Path
import pytest
import torch
from scripts.native_p1_dev_fit import (
    ARMS, DEV_INITIALIZATION, MAX_UPDATES, synthetic_pairs, train_arm, run)

def test_paired_analytic_fixture_is_factual_and_disjoint():
    fit=synthetic_pairs(fit=True)
    val=synthetic_pairs(fit=False)
    assert fit[0].shape==(126,4,4) and val[0].shape==(96,4,4)
    assert torch.equal(fit[1][::2],fit[1][1::2])
    assert torch.equal(fit[0][::2],fit[0][1::2])
    assert torch.equal(fit[3][::2],fit[3][1::2])
    assert torch.equal(fit[2][::2],-fit[2][1::2])
    assert (fit[4][1::2,0]-fit[4][::2,0]).min() > .2
    assert all(not torch.equal(val[1][0],fit[1][i]) for i in range(126))
    assert len(ARMS)==3 and MAX_UPDATES==1600 and DEV_INITIALIZATION==46109

def test_scientific_imports_excluded():
    source=(Path(__file__).resolve().parents[1]/"scripts/native_p1_dev_fit.py").read_text()
    tree=ast.parse(source)
    imports=[n.module or "" for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert not any(any(k in v for k in ("g7_s20","g7_s25","g7_s26","g7_s27")) for v in imports)

def test_disabled_dev_science_fence_blocks_before_any_training(monkeypatch):
    monkeypatch.setenv("NATIVE_P1_DEV_FORBID_SCIENCE","0")
    with pytest.raises(RuntimeError,match="P1_DEV_SCIENCE_FENCE_INVALID"):
        run()

def test_no_unregistered_arm():
    with pytest.raises(ValueError,match="P1_DEV_UNREGISTERED_ARM"):
        train_arm("hidden_model",synthetic_pairs(fit=True),synthetic_pairs(fit=False))
