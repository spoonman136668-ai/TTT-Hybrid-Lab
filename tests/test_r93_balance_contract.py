"""R93 index-only balance safety regression; cannot open any TRAIN/DEV_TEST world.

R92's exact-action/timing confound is negative precedent. Exercise the
prospectively corrected R93 action rule on every frozen index, both seeds.
"""
import pytest

from experiments import r93_headcap_exposure_diagnostic as r93


@pytest.mark.parametrize("seed",r93.SEEDS)
def test_r93_original_action_assignment_balances_time_law_and_action(seed):
    freeze=r93.frozen()
    spec=freeze["source"]["balance_contract"]
    actual=r93.balance_preflight()
    assert freeze["source"]["supersedes_preimplementation_freeze_blob"]=="9f51f6c6850a43cf353d1555505c2360fd49071d"
    assert spec["must_fail_closed_on_any_imbalance"] is True
    assert spec["train_worlds256_each_cell"]==32
    assert spec["train_worlds16_each_cell"]==2
    assert spec["dev_test_worlds64_each_cell"]==8
    for name,indices,count in (
        ("TRAIN_N256",range(256),32),
        ("TRAIN_N16",range(16),2),
        ("DEV_TEST_N64",range(1000,1064),8)):
        buckets={}
        for index in indices:
            early_cold=index%2
            balanced_hidden_law=((index%r93.r85.N)//2)%2
            selected=r93._chosen_action(seed,index)
            key=(early_cold,balanced_hidden_law,selected)
            buckets[key]=buckets.get(key,0)+1
        assert set(buckets)=={
            (timing,law,action) for timing in (0,1) for law in (0,1)
            for action in (r93.r85.CALIBRATE,r93.r85.EXPLOIT_POS)
        }
        assert set(buckets.values())=={count}
        assert actual[str(seed)][name]=={"strata_count":8,"each_stratum_count":count}


def test_r93_balance_does_not_require_world_generation(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError("R93_BALANCE_ATTEMPTED_UNOPENED_PHYSICAL_PANEL")
    monkeypatch.setattr(r93.r85,"_make_synthetic_world",forbidden)
    monkeypatch.setattr(r93.r89,"_world",forbidden)
    monkeypatch.setattr(r93.r89,"_test_world",forbidden)
    assert r93.balance_preflight()[str(r93.SEEDS[0])]["TRAIN_N256"]["each_stratum_count"]==32
    with pytest.raises(PermissionError,match="R93_UNREGISTERED_BALANCED_ACTION_SEED_OR_INDEX"):
        r93._chosen_action(0,0)
