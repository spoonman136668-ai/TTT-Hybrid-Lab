"""C7: P6 observed-history selection of saved old/new native memory, DEV only."""
import ast
import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch

from scripts.native_c7_p6_factual_evidence import (
    FactualSlopeEvidence, SLOPE_SIGN_FLOOR,
)
from scripts import native_c7_history_archive_selection as c7
from scripts import native_p3_continuity as factual


def test_p6_effect_estimator_needs_two_past_realized_action_outcomes():
    ledger = FactualSlopeEvidence()
    assert ledger.choice == 1
    assert ledger.observations == 0
    old_minus = torch.tensor([[-.41]], dtype=torch.float32)
    old_plus = torch.tensor([[.49]], dtype=torch.float32)
    new_minus = torch.tensor([[.49]], dtype=torch.float32)
    new_plus = torch.tensor([[-.41]], dtype=torch.float32)
    ledger.observe(torch.tensor([[-1.]]), old_minus)
    assert ledger.choice == 1 and ledger.observations == 1
    ledger.observe(torch.tensor([[1.]]), old_plus)
    assert ledger.choice == 1
    ledger.observe(torch.tensor([[-1.]]), new_minus)
    ledger.observe(torch.tensor([[1.]]), new_plus)
    assert ledger.choice == -1 and ledger.observations == 4
    assert ledger._change_events == [4]
    assert SLOPE_SIGN_FLOOR == .2


def test_nonoracle_router_cannot_take_hidden_label_or_target():
    ledger = FactualSlopeEvidence()
    anti = FactualSlopeEvidence()
    action = torch.tensor([[1.]])
    assert list(inspect.signature(c7.route_from_history_only).parameters) == [
        "arm", "action", "inferred", "anti"]
    assert c7.route_from_history_only("inferred_p6", action, ledger, anti) == 1
    assert c7.route_from_history_only("always_recent", action, ledger, anti) == -1
    assert c7.route_from_history_only("action_sign", action, ledger, anti) == 1
    with pytest.raises(ValueError, match="NONORACLE"):
        c7.route_from_history_only("oracle_upper_bound", action, ledger, anti)
    with pytest.raises(TypeError):
        c7.route_from_history_only("inferred_p6", action, ledger, anti, "old")


def test_measured_three_step_checkpoint_custody_and_history_route(tmp_path):
    torch.set_num_threads(1)
    cfg = c7._spec()
    seed = cfg["train"]["model_seeds"][0]
    directory = tmp_path / "isolated"
    directory.mkdir()
    result = c7.measure_seed(seed, directory, cfg)
    assert result["factual_train_steps_before_evaluation"] == 80
    assert result["root_sha256"] != result["old_checkpoint_sha256"]
    assert result["old_checkpoint_sha256"] != result["current_checkpoint_sha256"]
    assert len(result["evaluation_orders"]) == 2
    for order in result["evaluation_orders"]:
        assert len(order["order"]) == 4
        assert len(order["raw_event_ledger"]) == 64
        assert order["raw_event_ledger_sha256"] == hashlib.sha256(
            c7._canonical(order["raw_event_ledger"])).hexdigest()
        assert order["prediction_order"].startswith("PREDICTIONS_BEFORE")
        assert order["used_hidden_label_for_inference"] is False
        assert order["used_current_outcome_for_selection"] is False
        assert order["new_fast_memory_writes"] == 0
        assert order["postcheckpoint_optimizer_updates"] == 0
        assert set(order["measurements"]) == set(c7.ARMS)
        assert all(isinstance(x, bool) for x in order["hypotheses"].values())
        for arm in c7.ARMS:
            rec = order["measurements"][arm]
            assert len(rec["all_64_squared_errors"]) == 64
            assert len(rec["pre_outcome_route_sequence"]) == 64
            assert 0 <= rec["late_route_accuracy"] <= 1
        for event in order["raw_event_ledger"]:
            assert "current_action" in event
            assert "pre_outcome_choices" in event
            assert "factual_observed_after_all_predictions" in event
    assert result["checkpoint_and_native_state_unchanged"] is True
    assert result["no_new_model_or_fast_weight_learning"] is True
    assert result["science_first_use"] == "NOT_RUN"
    assert not (directory / "accepted").exists()
    assert not (directory / "latest").exists()


def test_engineering_only_and_protocol_fence(monkeypatch):
    cfg = c7._spec()
    assert cfg["science"]["original_first_use"] is False
    assert cfg["learned_router_predecessor"]["pr"] == 62
    assert cfg["evidence_predecessor"]["source_git_blob"] == (
        "7f21b73d6255f2611512969e87446352c1dee90c")
    monkeypatch.setenv("NATIVE_C7_ENGINEERING_ONLY", "0")
    with pytest.raises(RuntimeError, match="SCIENCE_FIRST_USE_FORBIDDEN"):
        c7.run()
    monkeypatch.delenv("NATIVE_C7_ENGINEERING_ONLY")
    monkeypatch.setenv("NATIVE_P3_ENGINEERING_ONLY", "0")
    with pytest.raises(RuntimeError, match="SCIENCE_FIRST_USE_FORBIDDEN"):
        c7.run()
