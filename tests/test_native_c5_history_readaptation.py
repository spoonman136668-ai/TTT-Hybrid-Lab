"""C5 chronological observed-history assay on restored P3/C4 DEV checkpoints."""
import math

import pytest
import torch

from native_continuous_state import NativeContinuousHybrid
from scripts.native_prequential_gate import NativeCausalEpisode
from scripts import native_c5_history_readaptation as c5


def test_factual_prefix_recovery_vs_matched_recent_and_no_feedback(tmp_path):
    torch.set_num_threads(1)
    root = tmp_path / "source"
    root.mkdir()
    result = c5.measure_one(c5._config()["seeds"][0], root)
    assert result["status"] == "PASS_PREQUENTIAL_PREFIX_ASSAY_INTEGRITY_DEV"
    assert result["checkpoint_lineage"]["champion_position"] == 40
    assert result["checkpoint_lineage"]["challenger_position"] == 80
    assert len(result["arms"]) == 9
    assert all(isinstance(v, bool) for v in result["hypotheses"].values())
    for n in (0, 2, 8):
        for arm in c5.ARMS:
            row = result["arms"][f"{arm}_{n}"]
            assert row["samples_old"] == row["samples_recent"] == 16
            assert row["factual_prefix_samples"] == n
            assert len(row["squared_error_vectors"]["old_sq_errors"]) == 16
            assert len(row["squared_error_vectors"]["recent_sq_errors"]) == 16
            assert row["fast_memory_l2_delta"] >= 0
            assert math.isfinite(row["old_mse"])
            assert math.isfinite(row["recent_mse"])
    assert result["arms"]["no_feedback_8"]["fast_memory_l2_delta"] == 0
    assert result["model_and_optimizer_frozen_after_step80"] is True
    assert result["scientific_first_use"] == "NOT_RUN"
    assert result["independent_scientific_acceptance"] is False
    assert result["model_promoted"] is False
    assert not (root / "accepted").exists()
    assert not (root / "latest").exists()


def test_prefix_rejects_wrong_arm_or_undeclared_length():
    torch.manual_seed(100)
    model = NativeContinuousHybrid(1, 1, width=12, history_limit=4)
    state = model.initial_state(torch.zeros(1, 1))
    native = c5._asdict(state)
    with pytest.raises(ValueError, match="UNREGISTERED"):
        c5._prefix(model, native, "oracle_true_context", 8)
    with pytest.raises(ValueError, match="UNREGISTERED"):
        c5._prefix(model, native, "old_factual", 9)


def test_observed_factual_outcome_arrives_after_prediction(monkeypatch):
    torch.manual_seed(101)
    model = NativeContinuousHybrid(1, 1, width=12, history_limit=4)
    state = c5._asdict(model.initial_state(torch.zeros(1, 1)))
    events = []
    before_predict = NativeCausalEpisode.predict
    before_truth = c5.factual.analytic_consequence

    def observed_prediction(self, sensor, action, dt):
        events.append("predict")
        return before_predict(self, sensor, action, dt)

    def observed_truth(action, time_index):
        events.append("reveal")
        return before_truth(action, time_index)

    monkeypatch.setattr(NativeCausalEpisode, "predict", observed_prediction)
    monkeypatch.setattr(c5.factual, "analytic_consequence", observed_truth)
    updated, errors = c5._prefix(model, state, "old_factual", 2)
    assert events == ["predict", "reveal", "predict", "reveal"]
    assert len(errors) == 2
    assert torch.count_nonzero(updated["fast_weights"]).item() > 0


def test_c5_science_first_use_denied(monkeypatch):
    monkeypatch.setenv("NATIVE_C5_ENGINEERING_ONLY", "0")
    with pytest.raises(RuntimeError, match="SCIENCE_FIRST_USE_FORBIDDEN"):
        c5.run()
    monkeypatch.delenv("NATIVE_C5_ENGINEERING_ONLY")
    monkeypatch.setenv("NATIVE_P3_ENGINEERING_ONLY", "0")
    with pytest.raises(RuntimeError, match="SCIENCE_FIRST_USE_FORBIDDEN"):
        c5.run()
