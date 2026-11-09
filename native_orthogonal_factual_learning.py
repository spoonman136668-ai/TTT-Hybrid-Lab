"""P42 DEV: factual-only, bounded token-free nuisance-separated causal learner.

Fits intercept, elapsed-time trend, and executed-action effect jointly from
completed transitions. This is classical online least squares over a fixed
basis, NOT discovered neural reasoning, independent scientific first use, or
superiority over a strong fitted conventional model. Candidate previews are
read-only and underdetermined histories abstain from causal attribution.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class OrthogonalState:
    elapsed: torch.Tensor      # [batch], cumulative factual elapsed time
    times: torch.Tensor        # [batch, capacity], factual event timestamps
    actions: torch.Tensor      # [batch, capacity], executed scalar actions
    changes: torch.Tensor      # [batch, capacity], realized sensor deltas
    occupied: torch.Tensor     # [batch, capacity], boolean
    cursor: torch.Tensor       # [batch], next ring slot
    observations: torch.Tensor # [batch], cumulative factual count


class OrthogonalFactualLearner:
    """Limited two-source claim: affine nuisance plus scalar physical action.

    A 3-column ordinary least-squares fit is one conventional baseline for
    later trained native mechanisms. It cannot identify arbitrary nuisance
    trends, unobserved switches or causality from non-randomized action logs.
    """
    def __init__(self, capacity: int = 8):
        if type(capacity) is not int or capacity < 3 or capacity > 64:
            raise ValueError('P42_BOUNDED_CAPACITY_REQUIRED')
        self.capacity = capacity

    @staticmethod
    def _observation(x, batch=None, label='SENSOR'):
        if (not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != 1
            or x.shape[0] < 1 or (batch is not None and x.shape[0] != batch)
            or x.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(x).all())):
            raise ValueError('P42_INVALID_' + label)

    def initial_state(self, sensor: torch.Tensor) -> OrthogonalState:
        self._observation(sensor)
        b = sensor.shape[0]
        z = sensor.new_zeros((b, self.capacity))
        return OrthogonalState(sensor.new_zeros((b,)), z.clone(), z.clone(), z.clone(),
                               torch.zeros((b, self.capacity), device=sensor.device, dtype=torch.bool),
                               torch.zeros((b,), device=sensor.device, dtype=torch.int64),
                               torch.zeros((b,), device=sensor.device, dtype=torch.int64))

    def _check(self, sensor, state):
        self._observation(sensor)
        if not isinstance(state, OrthogonalState):
            raise ValueError('P42_STATE_REQUIRED')
        b = sensor.shape[0]
        sizes = ((b,), (b, self.capacity), (b, self.capacity), (b, self.capacity),
                 (b, self.capacity), (b,), (b,))
        vals = (state.elapsed, state.times, state.actions, state.changes,
                state.occupied, state.cursor, state.observations)
        if any(not isinstance(v, torch.Tensor) or tuple(v.shape) != s or v.device != sensor.device
               for v,s in zip(vals,sizes)):
            raise ValueError('P42_STATE_SHAPE_OR_DEVICE_DRIFT')
        if (any(v.dtype != sensor.dtype for v in vals[:4])
            or state.occupied.dtype != torch.bool
            or state.cursor.dtype != torch.int64 or state.observations.dtype != torch.int64):
            raise ValueError('P42_STATE_DTYPE_DRIFT')
        if (any(not bool(torch.isfinite(v).all()) for v in vals[:4])
            or not bool((state.elapsed >= 0).all())
            or not bool(((state.cursor >= 0) & (state.cursor < self.capacity)).all())
            or not bool((state.observations >= 0).all())
            or not bool((state.occupied.sum(1) == state.observations.clamp(max=self.capacity)).all())):
            raise ValueError('P42_STATE_VALUES_INVALID')

    def _time(self, sensor, dt):
        self._observation(dt, sensor.shape[0], 'ELAPSED_TIME')
        if dt.device != sensor.device or dt.dtype != sensor.dtype or not bool((dt > 0).all()):
            raise ValueError('P42_INVALID_ELAPSED_TIME')

    def _action(self, sensor, action, *, candidates=False):
        self._observation(sensor)
        if (not isinstance(action, torch.Tensor)
            or action.ndim != (3 if candidates else 2)
            or action.shape[0] != sensor.shape[0]
            or action.shape[-1] != 1
            or (candidates and not 1 <= action.shape[1] <= 16)
            or action.dtype != sensor.dtype or action.device != sensor.device
            or not bool(torch.isfinite(action).all())
            or not bool((action.abs() > 1e-8).all())):
            raise ValueError('P42_INVALID_ACTION')

    def coefficients(self, sensor: torch.Tensor, state: OrthogonalState):
        """Return factual (intercept, slope, action effect), and rank-valid mask.

        All math uses float64 for stable fit. No future data, no other source
        observations, no hidden mode IDs, and no optimizer updates.
        """
        self._check(sensor,state)
        output = sensor.new_zeros((sensor.shape[0], 3))
        valid = torch.zeros((sensor.shape[0],), device=sensor.device, dtype=torch.bool)
        with torch.no_grad():
            for b in range(sensor.shape[0]):
                mask = state.occupied[b]
                if int(mask.sum()) < 3:
                    continue
                ts = state.times[b, mask].double()
                aa = state.actions[b, mask].double()
                yy = state.changes[b, mask].double()
                features = torch.stack((torch.ones_like(ts), ts, aa), dim=-1)
                if int(torch.linalg.matrix_rank(features)) != 3:
                    continue
                solved = torch.linalg.lstsq(features, yy).solution
                if not bool(torch.isfinite(solved).all()):
                    continue
                output[b] = solved.to(sensor.dtype)
                valid[b] = True
        return output, valid

    def preview(self, sensor: torch.Tensor, candidates: torch.Tensor,
                dt: torch.Tensor, state: OrthogonalState):
        """Read-only predictions. Underidentification is explicit via valid mask."""
        self._check(sensor,state)
        self._action(sensor,candidates,candidates=True)
        self._time(sensor,dt)
        coeff, valid = self.coefficients(sensor,state)
        future_time = state.elapsed + dt[:,0]
        drift = coeff[:,0,None] + coeff[:,1,None] * future_time[:,None]
        predicted = sensor[:,None,:] + (drift[:,None,:] + coeff[:,2,None,None] * candidates)
        predicted = torch.where(valid[:,None,None],predicted,sensor[:,None,:].expand_as(predicted))
        return predicted,valid

    def observe(self, sensor: torch.Tensor, executed_action: torch.Tensor,
                dt: torch.Tensor, actual_next: torch.Tensor, state: OrthogonalState):
        """Commit only after externally authenticated actual action consequence."""
        self._check(sensor,state)
        self._action(sensor,executed_action)
        self._time(sensor,dt)
        self._observation(actual_next,sensor.shape[0],'ACTUAL_NEXT')
        if actual_next.device != sensor.device or actual_next.dtype != sensor.dtype:
            raise ValueError('P42_ACTUAL_OUTCOME_DEVICE_DTYPE_DRIFT')
        ts = state.elapsed + dt[:,0]
        if not bool(torch.isfinite(ts).all()):
            raise ValueError('P42_NONFINITE_CLOCK')
        times,actions,changes,occupied = (state.times.clone(),state.actions.clone(),
                                           state.changes.clone(),state.occupied.clone())
        row = torch.arange(sensor.shape[0], device=sensor.device)
        col = state.cursor
        times[row,col] = ts
        actions[row,col] = executed_action[:,0]
        changes[row,col] = (actual_next-sensor)[:,0]
        occupied[row,col] = True
        return OrthogonalState(ts,times,actions,changes,occupied,
                               (col+1)%self.capacity,state.observations+1)
