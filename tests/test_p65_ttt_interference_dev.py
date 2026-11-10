"""P65 independent source-free deterministic mechanism control tests."""
import hashlib
import torch
from experiments.p65_long_horizon_ttt_interference_dev import run, update, read


def test_reference_fast_matrix_equation_one_exact_write():
    base=torch.zeros((1,16,16),dtype=torch.float32)
    key=torch.zeros((1,16));key[:,3]=1.0
    value=torch.linspace(-0.5,0.5,16).reshape(1,16)
    advanced=update(base,key,value)
    expected=0.5*value.unsqueeze(-1)*key.unsqueeze(1)
    assert torch.equal(advanced,expected)
    actual=read(advanced,key.unsqueeze(1))
    assert torch.equal(actual[:,0],0.5*value)


def test_frozen_negative_control_and_failure_are_not_hidden():
    result=run()
    assert result['strict_candidate_success_all_seeds'] is False
    assert result['all_orthogonal_controls_preserved'] is True
    for row in result['rows']:
        assert row['cold_restore_bitwise'] is True
        assert row['equal_budget_indexed_cache_mse']==0.0
        if row['world']=='overlap':
            assert row['post_mse'] > row['pre_mse']
            assert row['candidate_success'] is False
        else:
            assert abs(row['post_mse']-row['pre_mse'])<=1e-5


def test_frozen_exact_reproducibility():
    first=run();second=run()
    assert first==second
    assert all(len(x['data_sha256'])==64 for x in first['rows'])
