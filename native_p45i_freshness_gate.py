"""P45-I R1: freshness gate for a factual ABA action-effect estimate.

The P45-I runtime adapter retains a fast effect indefinitely, even after an
unannounced causal reversal and a new but non-ABA actual outcome. That stored
estimate may be valuable evidence, but is not automatically current evidence.
This sidecar abstains from selecting unless the latest factual event completed
a valid ABA contrast. It never claims to detect a hidden change in advance.

Engineering-only: an affine ABA contrast, however fresh, may be wrong under
nonlinear nuisance or an effect reversal inside the factual triplet.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import nn

from native_p45_aba_fast_adapter import NativeABAFastAdapter, ABAFastState
from native_trainable_causal_belief import NativeCausalPreview


@dataclass(frozen=True)
class FreshABACausalState:
    fast: ABAFastState
    last_valid_contrast_step: torch.Tensor  # [batch] int64, 0 if not identified


class NativeABAFreshnessGate(nn.Module):
    """Only current, witnessed ABA facts may authorize fast-effect action choice.

    Keeps stale fast memory available for audit and later learning, but returns
    recurrent-only diagnostic predictions and factual_ready=False when stale.
    A mode switch cannot be known until real outcomes carry enough information.
    """

    def __init__(self, adapter: NativeABAFastAdapter):
        super().__init__()
        if not isinstance(adapter, NativeABAFastAdapter):
            raise TypeError('P45R1_ABA_ADAPTER_REQUIRED')
        self.adapter = adapter

    def initial_state(self, sensor: torch.Tensor) -> FreshABACausalState:
        fast = self.adapter.initial_state(sensor)
        return FreshABACausalState(fast, torch.zeros((sensor.shape[0],),
                   device=sensor.device, dtype=torch.int64))

    def _check(self, sensor: torch.Tensor, state: FreshABACausalState) -> None:
        if not isinstance(state, FreshABACausalState):
            raise ValueError('P45R1_FRESH_STATE_REQUIRED')
        self.adapter._check(sensor, state.fast)
        last = state.last_valid_contrast_step
        total = state.fast.certified.factual.observations
        actual_contrasts = state.fast.factual_contrasts
        if (not isinstance(last, torch.Tensor) or last.shape != (sensor.shape[0],)
            or last.dtype != torch.int64 or last.device != sensor.device):
            raise ValueError('P45R1_FRESHNESS_SHAPE_OR_DTYPE_DRIFT')
        if (not bool((last >= 0).all()) or not bool((last <= total).all())
            or not bool(((last == 0) == (actual_contrasts == 0)).all())
            or not bool(((last == 0) | (last >= 3)).all())):
            raise ValueError('P45R1_FRESHNESS_CUSTODY_DRIFT')
        # An asserted contrast at the latest step must be backed by the last
        # three completed factual actions. This does not authenticate an
        # external source; that remains the source owner's responsibility.
        current = (last > 0) & (last == total)
        if bool(current.any()):
            actions = state.fast.actions[:, :, 0]
            valid = (torch.isclose(actions[:, 0], actions[:, 2], atol=1e-6, rtol=0)
                 & torch.isclose(actions[:, 0], -actions[:, 1], atol=1e-6, rtol=0)
                 & (actions[:, 0].abs() > 1e-8))
            if not bool((valid | ~current).all()):
                raise ValueError('P45R1_CONTRAST_SOURCE_MISMATCH')

    def freshness(self, sensor: torch.Tensor, state: FreshABACausalState) -> torch.Tensor:
        self._check(sensor, state)
        total = state.fast.certified.factual.observations
        last = state.last_valid_contrast_step
        return (last > 0) & (last == total)

    def observe(self, sensor: torch.Tensor, executed_action: torch.Tensor,
                dt: torch.Tensor, observed_next: torch.Tensor,
                state: FreshABACausalState) -> FreshABACausalState:
        """Advance only after the caller witnesses the factual action outcome."""
        self._check(sensor, state)
        newer = self.adapter.observe(sensor, executed_action, dt,
                                     observed_next, state.fast)
        completed = newer.factual_contrasts > state.fast.factual_contrasts
        step = newer.certified.factual.observations
        last = torch.where(completed, step, state.last_valid_contrast_step)
        result = FreshABACausalState(newer, last)
        self._check(sensor, result)
        return result

    def preview(self, sensor: torch.Tensor, candidates: torch.Tensor,
                dt: torch.Tensor, state: FreshABACausalState) -> NativeCausalPreview:
        """Never treat stale predictions as authorization or write memory."""
        fresh = self.freshness(sensor, state)
        with_fast = self.adapter.preview(sensor, candidates, dt, state.fast)
        recurrent = self.adapter.certifier.preview(sensor, candidates, dt,
                                                   state.fast.certified)
        prediction = torch.where(fresh[:, None, None], with_fast.predictions,
                                 recurrent.predictions)
        return NativeCausalPreview(prediction, with_fast.factual_ready & fresh)

    def select(self, sensor: torch.Tensor, candidates: torch.Tensor,
               dt: torch.Tensor, external_goal: torch.Tensor,
               state: FreshABACausalState) -> torch.Tensor:
        result = self.preview(sensor, candidates, dt, state)
        self.adapter.native._analog(external_goal, tuple(sensor.shape), 'EXTERNAL_GOAL')
        self.adapter.native._on_model(external_goal)
        if not bool(result.factual_ready.all()):
            raise ValueError('P45R1_NO_FRESH_IDENTIFIABLE_CONTRAST')
        return ((result.predictions-external_goal[:, None, :])
                .square().sum(-1)).argmin(-1)
