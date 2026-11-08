"""G2: deterministic randomized order, paired CI, no GPU first-use in pytest."""
import json
import math
from pathlib import Path
import pytest
import torch

from scripts.native_p9_gpu_randomized_dev import (
    ARMS,REPETITIONS,WARMUPS,SEEDS,round_schedule,
    paired_bootstrap,rollout,_all_state_continuous
)
from scripts.native_p9_parallel_refinement_dev import (
    ContinuousDualStream,trace,exercise
)


def test_random_arm_order_exhaustive_balanced_and_reproducible():
    for seed in SEEDS:
        a=round_schedule(seed)
        assert a==round_schedule(seed)
        assert len(a)==REPETITIONS==24
        assert all(set(x)==set(ARMS) and len(x)==4 for x in a)
        assert len(set(a))>=8
    with pytest.raises(ValueError,match="UNREGISTERED_ROUND"):
        round_schedule(7207)
    with pytest.raises(ValueError,match="UNREGISTERED_ROUND"):
        round_schedule(SEEDS[0],25)


def test_bootstrap_preserves_negative_slowdown_and_uses_paired_rounds():
    result=paired_bootstrap([1.2]*24,SEEDS[0])
    assert math.isclose(result["median_ratio"],1.2)
    assert result["bootstrap_95pct_median_CI_lower"]>1.
    assert result["bootstrap_95pct_median_CI_upper"]>1.
    assert len(result["all_paired_ratios"])==24
    assert result==paired_bootstrap([1.2]*24,SEEDS[0])
    result=paired_bootstrap([.70]*24,SEEDS[0])
    assert result["bootstrap_95pct_median_CI_upper"]<.75
    with pytest.raises(ValueError,match="INVALID_PAIRED_MEASUREMENTS"):
        paired_bootstrap([float("nan")],SEEDS[0])


def test_preexisting_P9_semantics_preserved_without_gpu():
    torch.manual_seed(613)
    m=ContinuousDualStream().eval()
    obs=trace(613,16,2)
    variants={
        "parallel_dense":("parallel",[True]*16),
        "parallel_gate":("parallel",obs[4]),
        "parallel_random":("parallel",obs[5]),
        "sequential_dense":("sequential",[True]*16)}
    with torch.no_grad():
        output={k:rollout(m,obs,*v) for k,v in variants.items()}
        prior_out,prior_state,_=exercise(m,obs,"parallel",[True]*16)
    assert torch.equal(prior_out,output["parallel_dense"][0])
    assert torch.equal(prior_state.h,output["parallel_dense"][1].h)
    assert all(_all_state_continuous(output,16).values())
    assert output["parallel_gate"][1].events==16


def test_frozen_budgets_and_original_science_fences():
    root=Path(__file__).resolve().parents[1]
    p=json.loads((root/"experiments/native_p9_gpu_randomized_dev_frozen.json").read_text())
    assert p["schema"]=="ttt-hybrid.native-p9.g2.randomized-cuda-frozen.v1"
    assert p["source"]["parent_g1_sha"]=="cda3885e15d3ecee91b6a55931849002543ffb15"
    assert p["science"]["authorized"] is False
    assert p["science"]["world_seeds"]=={}
    assert p["science"]["first_use_consumed"] is False
    assert p["design"]["rounds"]==REPETITIONS==24
    assert p["design"]["warmups"]==WARMUPS==3
    assert p["design"]["bootstrap_resamples"]==1000
    assert p["design"]["speed_ratio_max"]==.75
    assert p["design"]["untrained_MSE_ratio_max"]==1.01
    assert p["design"]["source_p9_unchanged"] is True
