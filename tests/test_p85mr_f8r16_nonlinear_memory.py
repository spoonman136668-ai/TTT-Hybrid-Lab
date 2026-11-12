"""F8-R16: source-neutral nonlinear-vs-memory negative controls and custody."""
from __future__ import annotations
import inspect
from pathlib import Path

import numpy as np
import pytest

from experiments import p85mr_f8r16_nonlinear_memory as v


def fixture(*, correlated: bool) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(990201 if correlated else 990202)
    x = rng.normal(size=(128,95,7))
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    q = v.r2.features(x, "quadratic35")
    weights = rng.normal(size=(35,4)) * .05
    innovation = rng.normal(size=(128,95,4)) * .008
    if correlated:
        for step in range(1,95):
            innovation[:,step] += .83*innovation[:,step-1]
    y = q@weights + innovation
    return x.astype(np.float64), y.astype(np.float64)


def test_original_source_and_10pct_threshold_cannot_be_overridden():
    f = v.locked()
    assert f["schema"] == v.SCHEMA
    assert f["split"]["old_evaluated_test_worlds"] == 32
    assert f["thresholds"]["bootstrap_seed"] == 1825990128
    assert f["source"]["original_effects_already_consumed"] == 12288
    assert f["source"]["new_source_effects"] == 0
    assert f["native_owner_requirements"]["firstuse"] is False
    with pytest.raises(PermissionError, match="NOT_AUTHORIZED"):
        v.refuse_original_firstuse()


def test_positive_causal_autocorrelation_fixture_detects_past_error():
    x,y=fixture(correlated=True)
    report=v.evaluate_spent_arrays(x,y)
    assert report["new_original_source_effects"] == 0
    assert report["already_inspected_test_worlds"] == 32
    assert report["independent_scientific_pass"] is False
    arm=report["arms"]["quadratic35"]
    assert arm["base_parameters"] == 140
    assert arm["memory_additional_parameters"] == 16
    assert arm["relative_memory_improvement"] >= .10
    assert arm["bootstrap_lower_absolute_gain"] > 0
    assert arm["memory_old_F7_mse"] < arm["lag8_negative_old_F7_mse"]
    assert arm["frozen_criterion_pass"] is True


def test_independent_noise_does_not_invent_10pct_memory_benefit():
    x,y=fixture(correlated=False)
    report=v.evaluate_spent_arrays(x,y)
    arm=report["arms"]["quadratic35"]
    assert arm["relative_memory_improvement"] < .10
    assert arm["frozen_criterion_pass"] is False
    assert report["independent_scientific_pass"] is False


@pytest.mark.parametrize("mutation",[
    "missing_world", "missing_channel", "nan", "inf",
    "feature_norm", "wrong_precision"])
def test_corrupted_physical_arrays_fail_closed(mutation):
    x,y=fixture(correlated=False)
    if mutation=="missing_world":x=x[:-1]
    elif mutation=="missing_channel":y=y[...,:3]
    elif mutation=="nan":x[0,0,2]=np.nan
    elif mutation=="inf":y[0,0,3]=float("inf")
    elif mutation=="feature_norm":x*=3
    elif mutation=="wrong_precision":y=y.astype(np.float32)
    with pytest.raises(ValueError,match="UNTRUSTED_OR_INCOMPLETE"):
        v.evaluate_spent_arrays(x,y)


def test_fake_archives_cannot_invent_old_external_receipts(tmp_path):
    x=tmp_path/"old.zip";y=tmp_path/"trained.zip"
    x.write_bytes(b"FABRICATED");y.write_bytes(b"FABRICATED")
    with pytest.raises(ValueError,match="ORIGINAL_ARTIFACT_DRIFT"):
        v.evaluate_spent_archives(x,y)


def test_source_module_is_not_new_simulator_or_runner():
    source=inspect.getsource(v)
    for forbidden in ("from causal_gym","import causal_gym",
                      "env.reset(", "env.step(", "env.do(",
                      "workflow_dispatch(","subprocess.run(",
                      "requests.get(", "urlopen(", "torch.save(",
                      "optimizer.step("):
        assert forbidden not in source
    assert v.locked()["native_owner_requirements"]["runner_dispatch"] is False
