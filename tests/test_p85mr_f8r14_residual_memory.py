"""F8-R14 negative and positive synthetic controls, no original F7 source access."""
from __future__ import annotations

import copy
import inspect
from pathlib import Path

import numpy as np
import pytest

from experiments import p85mr_f8r14_residual_memory as m


def panel(*, correlated: bool) -> tuple[np.ndarray, np.ndarray]:
    rng=np.random.default_rng(92480 if correlated else 92481)
    x=rng.normal(size=(128,95,7))
    x=x/np.linalg.norm(x,axis=-1,keepdims=True)
    weights=rng.normal(size=(7,4))*0.11
    eps=rng.normal(size=(128,95,4))*0.008
    residual=eps.copy()
    if correlated:
        for t in range(1,95):
            residual[:,t]+=0.80*residual[:,t-1]
    y=x@weights+residual
    return x.astype(np.float64),y.astype(np.float64)


def test_freeze_precedes_analysis_and_denies_external_action():
    f=m.locked()
    assert f["stage"]=="FROZEN_BEFORE_F8R14_RETROSPECTIVE_NUMERICAL_RESULT"
    assert f["fitting"]["memory_params"]==16
    assert f["fitting"]["baseline_params"]==28
    assert f["authority"]["new_CartPoleWind_source_actions"] is False
    assert f["authority"]["owner_firstuse"] is False
    with pytest.raises(PermissionError,match="NOT_AUTHORIZED"):
        m.refuse_original_firstuse()


def test_factual_lag_t_never_observes_y_t_or_future_y():
    r=np.arange(2*6*4,dtype=np.float64).reshape(2,6,4)
    a=m.previous_factual_residual(r)
    b=m.previous_factual_residual(r,7)
    assert np.array_equal(a[:,0],np.zeros((2,4)))
    assert np.array_equal(a[:,1:],r[:,:-1])
    assert np.array_equal(b,np.zeros_like(r))
    changed=r.copy()
    changed[:,3:]+=11111
    recomputed=m.previous_factual_residual(changed)
    assert np.array_equal(a[:,:4],recomputed[:,:4])
    assert not np.array_equal(a[:,4],recomputed[:,4])


def test_past_residual_is_useful_when_physical_memory_exists_in_fixture():
    x,y=panel(correlated=True)
    result=m.evaluate_arrays(x,y)
    assert result["baseline_parameters"]==28
    assert result["additional_memory_parameters"]==16
    assert result["old_holdout_worlds_NOT_NEW"]==32
    assert result["old_holdout_factual_events"]==3040
    assert result["real_original_source_actions"]==0
    assert result["independent_scientific_pass"] is False
    assert result["relative_memory_improvement"]>.10
    assert result["paired_world_bootstrap_2p5_absolute_gain"]>0
    assert result["memory_heldout_mse"]<result["delay7_heldout_mse"]
    assert result["passed_frozen_retrospective_criterion"] is True


def test_no_persistent_signal_is_not_a_scientific_positive():
    x,y=panel(correlated=False)
    result=m.evaluate_arrays(x,y)
    assert result["relative_memory_improvement"]<.10
    assert result["passed_frozen_retrospective_criterion"] is False
    assert result["independent_scientific_pass"] is False


@pytest.mark.parametrize("fault",["wrong_worlds","wrong_channels","nans","unscaled_keys","float32","inf"])
def test_poisoned_input_fails_closed(fault):
    x,y=panel(correlated=False)
    if fault=="wrong_worlds":x=x[:-1]
    elif fault=="wrong_channels":y=y[...,:3]
    elif fault=="nans":x[0,0,1]=np.nan
    elif fault=="inf":y[0,0,0]=np.inf
    elif fault=="unscaled_keys":x*=2.
    elif fault=="float32":y=y.astype(np.float32)
    with pytest.raises(ValueError,match="UNTRUSTED_OR_INCOMPLETE"):
        m.evaluate_arrays(x,y)


def test_wrong_lag_policy_and_raw_archive_drift_are_rejected(tmp_path):
    with pytest.raises(ValueError,match="INVALID_LAG"):
        m.previous_factual_residual(np.ones((2,95,4),dtype=np.float64),-1)
    with pytest.raises(ValueError,match="INVALID_LAG"):
        m.previous_factual_residual(np.ones((2,95,4),dtype=np.float64),8)
    a=tmp_path/"f7.zip";b=tmp_path/"trained.zip"
    a.write_bytes(b"spoofed");b.write_bytes(b"spoofed")
    with pytest.raises(ValueError,match="ORIGINAL_ARTIFACT_DRIFT"):
        m.evaluate_spent_archives(a,b)


def test_no_external_actuation_and_no_future_input_to_fitting():
    source=inspect.getsource(m)
    for forbidden in ("import causal_gym","from causal_gym",
                      "env.reset(","env.do(","workflow_dispatch(",
                      "subprocess.run(","requests.get(","urlopen(",
                      "torch.save(","optimizer.step("):
        assert forbidden not in source
    assert m.locked()["authority"]["runner_dispatch"] is False
    assert m.locked()["authority"]["original_scientific_pass"] is False


def test_archived_negative_result_remains_visible_on_repo():
    import json
    evidence=json.loads((Path(__file__).resolve().parents[1] /
        "experiments/p85mr_f8r14_residual_memory_result.json").read_text())
    assert evidence["freeze_ancestor"] == "13b5ed7e650711a81dbdc36998bc89500cd91a89"
    assert evidence["source_zip_sha256"] == m.r2.F7_ZIP
    assert evidence["capture_sha256"] == m.r2.F7_CAPTURE
    assert evidence["old_F7_holdout_factual_effects"] == 32*95
    assert evidence["baseline_heldout_mse"] == pytest.approx(2.7510204527871957e-6,abs=1e-16)
    assert evidence["memory_heldout_mse"] < evidence["baseline_heldout_mse"]
    assert evidence["relative_memory_improvement"] < .10
    assert evidence["paired_world_bootstrap_2p5_absolute_gain"] > 0
    assert evidence["passed_frozen_retrospective_criterion"] is False
    assert evidence["independent_scientific_pass"] is False
    assert evidence["new_source_effects"] == 0
