"""Native P1 conventional numeric controls and fail-closed competence admission.

ENGINEERING ONLY: no dataset generation, no fit loop, no science seeds, no
world construction, no runner or automatic scientific first-use.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BaselineAdmissionError(ValueError):
    pass


def _check_numeric(x: Tensor, shape: tuple[int | None, ...], name: str) -> None:
    if not isinstance(x, Tensor) or x.ndim != len(shape):
        raise BaselineAdmissionError("P1_INVALID_" + name + "_SHAPE")
    if any(n is not None and x.shape[i] != n for i, n in enumerate(shape)):
        raise BaselineAdmissionError("P1_INVALID_" + name + "_SHAPE")
    if x.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(x).all()):
        raise BaselineAdmissionError("P1_NONCONTINUOUS_OR_NONFINITE_" + name)


class _AnalogControl(nn.Module):
    def __init__(self, sensor_dim: int, action_dim: int, width: int, max_history: int = 32):
        super().__init__()
        if any(type(i) is not int or i <= 0 for i in
               (sensor_dim, action_dim, width, max_history)):
            raise BaselineAdmissionError("P1_INVALID_MODEL_DIMENSIONS")
        self.sensor_dim = sensor_dim
        self.action_dim = action_dim
        self.width = width
        self.max_history = max_history
        self.history_features = sensor_dim + action_dim + 1

    def _inputs(self, history: Tensor, sensor: Tensor, action: Tensor, dt: Tensor) -> None:
        _check_numeric(sensor, (None, self.sensor_dim), "SENSOR")
        batch = sensor.shape[0]
        if batch < 1:
            raise BaselineAdmissionError("P1_EMPTY_BATCH")
        _check_numeric(history, (batch, None, self.history_features), "HISTORY")
        if not 1 <= history.shape[1] <= self.max_history:
            raise BaselineAdmissionError("P1_UNBOUNDED_HISTORY")
        _check_numeric(action, (batch, self.action_dim), "ACTION")
        _check_numeric(dt, (batch, 1), "TIME")
        if not bool((dt > 0).all()) or not bool((history[:, :, -1] > 0).all()):
            raise BaselineAdmissionError("P1_NONPOSITIVE_ELAPSED_TIME")
        if any(x.device != sensor.device or x.dtype != sensor.dtype
               for x in (history, action, dt)) or (
                   sensor.device != next(self.parameters()).device or
                   sensor.dtype != next(self.parameters()).dtype):
            raise BaselineAdmissionError("P1_DEVICE_OR_DTYPE_DRIFT")


class ConventionalAnalogGRU(_AnalogControl):
    """Ordinary fully trainable GRU; no native fast-memory or token interface."""
    def __init__(self, sensor_dim: int, action_dim: int, width: int = 24,
                 max_history: int = 32):
        super().__init__(sensor_dim, action_dim, width, max_history)
        self.recurrence = nn.GRU(self.history_features, width, batch_first=True)
        self.query = nn.Linear(sensor_dim + action_dim + 1, width)
        self.readout = nn.Linear(width * 2, sensor_dim)

    def forward(self, history: Tensor, sensor: Tensor, action: Tensor,
                dt: Tensor) -> Tensor:
        self._inputs(history, sensor, action, dt)
        hidden = self.recurrence(history)[0][:, -1, :]
        query = torch.tanh(self.query(torch.cat(
            (sensor, action, torch.log1p(dt)), dim=-1)))
        return sensor + self.readout(torch.cat((hidden, query), dim=-1))


class ConventionalAnalogSSM(_AnalogControl):
    """Simple continuous-decay state-space recurrent comparator, not Mamba2."""
    def __init__(self, sensor_dim: int, action_dim: int, width: int = 24,
                 max_history: int = 32):
        super().__init__(sensor_dim, action_dim, width, max_history)
        self.input_map = nn.Linear(self.history_features, width)
        self.decay = nn.Parameter(torch.zeros(width))
        self.query = nn.Linear(sensor_dim + action_dim + 1, width)
        self.readout = nn.Linear(width * 2, sensor_dim)

    def forward(self, history: Tensor, sensor: Tensor, action: Tensor,
                dt: Tensor) -> Tensor:
        self._inputs(history, sensor, action, dt)
        h = sensor.new_zeros((sensor.shape[0], self.width))
        for i in range(history.shape[1]):
            elapsed = history[:, i, -1:].contiguous()
            retention = torch.exp(-elapsed * F.softplus(self.decay)[None, :])
            h = retention * h + (1 - retention) * torch.tanh(
                self.input_map(history[:, i, :]))
        query = torch.tanh(self.query(torch.cat(
            (sensor, action, torch.log1p(dt)), dim=-1)))
        return sensor + self.readout(torch.cat((h, query), dim=-1))


def qualify_conventional_record(record: Mapping, *, arm: str) -> dict:
    """Pure audit of *submitted receipts*, not proof that training occurred.

    Requires externally verified source/checkpoint provenance. This only
    determines whether the numeric and declared acceptance gates are satisfied.
    """
    if arm not in ("conventional_analog_gru", "conventional_analog_ssm"):
        raise BaselineAdmissionError("P1_UNREGISTERED_CONTROL")
    if not isinstance(record, Mapping):
        raise BaselineAdmissionError("P1_MISSING_RECEIPT")
    required = ("update_count", "validation_mse_last_five",
                "static_validation_mse", "effect_sign_accuracy",
                "effect_eligible_coverage", "all_controls_same_information",
                "checkpoint_cold_restore", "source_and_seed_custody_verified",
                "training_completed", "compute_budget_verified")
    if any(k not in record for k in required):
        return {"qualified": False, "status": "BASELINE_NOT_QUALIFIED_MISSING_EVIDENCE"}
    window = record["validation_mse_last_five"]
    if not isinstance(window, (list, tuple)) or len(window) != 5 or any(
        type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in window
    ):
        return {"qualified": False, "status": "BASELINE_NOT_QUALIFIED_INVALID_WINDOW"}
    numeric = [record["static_validation_mse"], record["effect_sign_accuracy"],
               record["effect_eligible_coverage"]]
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in numeric):
        return {"qualified": False, "status": "BASELINE_NOT_QUALIFIED_INVALID_METRICS"}
    if (type(record["update_count"]) is not int or
        not 640 <= record["update_count"] <= 1600 or
        record["static_validation_mse"] <= 0 or
        not 0 <= record["effect_sign_accuracy"] <= 1 or
        not 0 <= record["effect_eligible_coverage"] <= 1):
        return {"qualified": False, "status": "BASELINE_NOT_QUALIFIED_INVALID_BUDGET"}
    if not all(record[key] is True for key in
               ("all_controls_same_information", "checkpoint_cold_restore",
                "source_and_seed_custody_verified", "training_completed",
                "compute_budget_verified")):
        return {"qualified": False, "status": "BASELINE_NOT_QUALIFIED_CUSTODY_OR_TRAINING"}
    spread = (max(window)-min(window))/min(window)
    if spread > 0.01:
        return {"qualified": False, "status": "BASELINE_NOT_QUALIFIED_NO_PLATEAU"}
    if (window[-1] >= record["static_validation_mse"] or
        record["effect_sign_accuracy"] < 0.95 or
        record["effect_eligible_coverage"] < 0.95):
        return {"qualified": False, "status": "BASELINE_NOT_QUALIFIED_NOT_COMPETENT"}
    return {"qualified": True, "status": "CONTROL_RECEIPT_NUMERICALLY_ADMISSIBLE",
            "validation_mse": window[-1], "five_check_spread": spread}
