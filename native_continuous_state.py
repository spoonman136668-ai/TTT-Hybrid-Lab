"""Prospective token-free continuous observation/action dynamics reference.

No vocabulary, token embeddings, next-token prediction or scientific first use.
An engineering-only causal interface, not proof of intelligence or efficiency.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ContinuousState:
    recurrent: torch.Tensor       # [batch, width]
    fast_weights: torch.Tensor    # [batch, sensor_dim, width]
    history: torch.Tensor         # [batch, <= history_limit, width]


class NativeContinuousHybrid(nn.Module):
    """Persistent, bounded native-state causal predictor over analog measurements.

    step predicts the next continuous sensor reading from observed sensors,
    the intervention to perform, and elapsed physical time. learn_outcome
    updates fast memory only after the outcome has been observed. It is separate
    from the historical token-indexed Mamba/TTT pilot and all frozen S25-S27 code.
    """

    def __init__(self, sensor_dim: int, action_dim: int, width: int = 24,
                 history_limit: int = 8, fast_rate: float = .25):
        super().__init__()
        if not all(type(v) is int and v > 0 for v in (sensor_dim, action_dim, width, history_limit)):
            raise ValueError('NATIVE_POSITIVE_DIMS_REQUIRED')
        if not math.isfinite(fast_rate) or not (0.0 < fast_rate <= 1.0):
            raise ValueError('NATIVE_INVALID_FAST_RATE')
        self.sensor_dim = sensor_dim
        self.action_dim = action_dim
        self.width = width
        self.history_limit = history_limit
        self.fast_rate = float(fast_rate)
        self.sensor_encoder = nn.Linear(sensor_dim, width)
        self.action_encoder = nn.Linear(action_dim, width)
        self.time_encoder = nn.Linear(1, width)
        self.write_gate = nn.Linear(width, width)
        self.decay = nn.Parameter(torch.zeros(width))
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.dynamics = nn.Linear(width * 2, sensor_dim)

    def initial_state(self, sensors: torch.Tensor) -> ContinuousState:
        self._validate_float(sensors, (None, self.sensor_dim), 'SENSOR')
        batch = sensors.shape[0]
        return ContinuousState(sensors.new_zeros((batch, self.width)),
                               sensors.new_zeros((batch, self.sensor_dim, self.width)),
                               sensors.new_zeros((batch, 0, self.width)))

    @staticmethod
    def _validate_float(x: torch.Tensor, expected: tuple, role: str):
        if not isinstance(x, torch.Tensor) or x.ndim != len(expected) or any(
                size is not None and x.shape[i] != size for i, size in enumerate(expected)):
            raise ValueError('NATIVE_INVALID_' + role + '_SHAPE')
        if not x.is_floating_point() or x.dtype not in (torch.float32, torch.float64):
            raise ValueError('NATIVE_NONCONTINUOUS_' + role)
        if x.shape[0] < 1 or not bool(torch.isfinite(x).all()):
            raise ValueError('NATIVE_INVALID_' + role + '_VALUES')

    def _validate_state(self, state: ContinuousState, sensor: torch.Tensor):
        if not isinstance(state, ContinuousState):
            raise ValueError('NATIVE_STATE_REQUIRED')
        b = sensor.shape[0]
        if (state.recurrent.shape != (b, self.width)
            or state.fast_weights.shape != (b, self.sensor_dim, self.width)
            or state.history.ndim != 3
            or state.history.shape[0] != b
            or state.history.shape[2] != self.width
            or state.history.shape[1] > self.history_limit):
            raise ValueError('NATIVE_STATE_SHAPE_DRIFT')
        if any(v.device != sensor.device or v.dtype != sensor.dtype
               or not bool(torch.isfinite(v).all()) for v in
               (state.recurrent, state.fast_weights, state.history)):
            raise ValueError('NATIVE_STATE_DEVICE_DTYPE_OR_NONFINITE')

    def step(self, sensor: torch.Tensor, action: torch.Tensor, dt: torch.Tensor,
             state: ContinuousState | None = None) -> tuple[torch.Tensor, ContinuousState]:
        self._validate_float(sensor, (None, self.sensor_dim), 'SENSOR')
        self._validate_float(action, (sensor.shape[0], self.action_dim), 'ACTION')
        self._validate_float(dt, (sensor.shape[0], 1), 'ELAPSED_TIME')
        if not bool(torch.all(dt > 0)):
            raise ValueError('NATIVE_NONPOSITIVE_ELAPSED_TIME')
        if (action.device != sensor.device or dt.device != sensor.device
            or action.dtype != sensor.dtype or dt.dtype != sensor.dtype
            or sensor.device != self.decay.device or sensor.dtype != self.decay.dtype):
            raise ValueError('NATIVE_DEVICE_OR_DTYPE_DRIFT')
        if state is None:
            state = self.initial_state(sensor)
        self._validate_state(state, sensor)
        candidate = torch.tanh(self.sensor_encoder(sensor) + self.action_encoder(action)
                               + self.time_encoder(torch.log1p(dt)))
        retention = torch.exp(-dt * F.softplus(self.decay)[None, :])
        gate = torch.sigmoid(self.write_gate(candidate))
        hidden = retention * state.recurrent + (1-retention) * gate * candidate
        if state.history.shape[1]:
            q = self.query(hidden).unsqueeze(1)
            keys = self.key(state.history)
            values = self.value(state.history)
            weights = torch.softmax(torch.bmm(q, keys.transpose(1,2))/math.sqrt(self.width), dim=-1)
            context = torch.bmm(weights, values).squeeze(1)
        else:
            context = torch.zeros_like(hidden)
        fast_key = F.normalize(self.action_encoder(action), dim=-1)
        correction = torch.bmm(state.fast_weights, fast_key.unsqueeze(-1)).squeeze(-1)
        prediction = sensor + self.dynamics(torch.cat((hidden, context), dim=-1)) + correction
        history = torch.cat((state.history, hidden.unsqueeze(1)), dim=1)[:, -self.history_limit:, :]
        return prediction, ContinuousState(hidden, state.fast_weights, history)

    def learn_outcome(self, state: ContinuousState, action: torch.Tensor,
                      prediction: torch.Tensor, observed_next: torch.Tensor) -> ContinuousState:
        """Causal fast-weight correction: only call after observing the consequence."""
        self._validate_float(prediction, (None, self.sensor_dim), 'PREDICTION')
        self._validate_float(observed_next, tuple(prediction.shape), 'OBSERVED_OUTCOME')
        self._validate_float(action, (prediction.shape[0], self.action_dim), 'ACTION')
        self._validate_state(state, prediction)
        if any(v.device != prediction.device or v.dtype != prediction.dtype
               for v in (action, observed_next)):
            raise ValueError('NATIVE_DEVICE_OR_DTYPE_DRIFT')
        key = F.normalize(self.action_encoder(action), dim=-1)
        error = (observed_next - prediction).unsqueeze(-1)
        weights = state.fast_weights + self.fast_rate * error * key.unsqueeze(1)
        return ContinuousState(state.recurrent, weights, state.history)

    def transition_loss(self, prediction: torch.Tensor, observed_next: torch.Tensor) -> torch.Tensor:
        self._validate_float(prediction, (None, self.sensor_dim), 'PREDICTION')
        self._validate_float(observed_next, tuple(prediction.shape), 'OBSERVED_OUTCOME')
        if prediction.device != observed_next.device or prediction.dtype != observed_next.dtype:
            raise ValueError('NATIVE_DEVICE_OR_DTYPE_DRIFT')
        return (prediction - observed_next).square().mean()
