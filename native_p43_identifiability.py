"""P43-R1 DEV: factual intervention-rank certificate for trainable causal state.

The P43 model's factual count is not evidence that its action effect is
identified: four identical actions, or actions confounded with time, can leave
[intercept, time, action] rank deficient. This optional sidecar abstains on
such histories while preserving P43 weights and recurrent state unchanged.

Rank is a necessary engineering prerequisite under a locally affine nuisance,
NOT proof of causal identification or robust neural OOD generalization.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from native_trainable_causal_belief import (NativeTrainableCausalBelief,
    NativeCausalBeliefState, NativeCausalPreview)


@dataclass(frozen=True)
class CertifiedCausalState:
    factual: NativeCausalBeliefState
    design_gram: torch.Tensor  # [batch,3,3] from completed [1,elapsed_time,action]


class CertifiedFactualCausalBelief:
    """No new parameters, no speculative state commits, no source oracle."""

    def __init__(self, model: NativeTrainableCausalBelief, rcond: float = 1e-6):
        if not isinstance(model, NativeTrainableCausalBelief):
            raise TypeError('P43R1_NATIVE_MODEL_REQUIRED')
        if type(rcond) is not float or not 0 < rcond <= 1e-3:
            raise ValueError('P43R1_INVALID_RANK_TOLERANCE')
        self.model = model
        self.rcond = rcond

    def initial_state(self, sensor: torch.Tensor) -> CertifiedCausalState:
        factual = self.model.initial_state(sensor)
        return CertifiedCausalState(factual, torch.zeros((sensor.shape[0],3,3),
             dtype=torch.float64,device=sensor.device))

    def _check(self, sensor: torch.Tensor, state: CertifiedCausalState):
        if not isinstance(state, CertifiedCausalState):
            raise ValueError('P43R1_CERTIFIED_STATE_REQUIRED')
        self.model._state(sensor, state.factual)
        g=state.design_gram
        if (not isinstance(g,torch.Tensor) or g.shape!=(sensor.shape[0],3,3)
            or g.dtype!=torch.float64 or g.device!=sensor.device
            or not bool(torch.isfinite(g).all())
            or not bool(torch.allclose(g,g.transpose(-1,-2),atol=1e-8,rtol=0))):
            raise ValueError('P43R1_CERTIFICATE_SHAPE_DTYPE_OR_VALUES')
        if not bool((g[:,0,0]-state.factual.observations.double()).abs().le(1e-8).all()):
            raise ValueError('P43R1_CERTIFICATE_FACTUAL_COUNT_DRIFT')
        if not bool((torch.linalg.eigvalsh(g)>=-1e-8).all()):
            raise ValueError('P43R1_NONPSD_CERTIFICATE')

    def evidence_ready(self, sensor: torch.Tensor, state: CertifiedCausalState) -> torch.Tensor:
        self._check(sensor,state)
        vals=torch.linalg.eigvalsh(state.design_gram)
        threshold=vals[:,-1].clamp_min(1.0)*self.rcond
        return (state.factual.observations>=max(3,self.model.min_facts)) & (vals[:,0]>threshold)

    def observe(self,sensor: torch.Tensor, executed_action: torch.Tensor,
                dt: torch.Tensor, factual_next: torch.Tensor,
                state: CertifiedCausalState) -> CertifiedCausalState:
        """Only a caller-authenticated factual action/outcome updates state."""
        self._check(sensor,state)
        # All data validation and model update occur before computing new gram.
        factual=self.model.observe(sensor,executed_action,dt,factual_next,state.factual)
        t=factual.elapsed[:,0].double()
        a=executed_action[:,0].double()
        x=torch.stack((torch.ones_like(t),t,a),dim=-1)
        gram=state.design_gram+torch.einsum('bi,bj->bij',x,x)
        if not bool(torch.isfinite(gram).all()):
            raise ValueError('P43R1_CERTIFICATE_OVERFLOW')
        return CertifiedCausalState(factual,gram)

    def preview(self,sensor: torch.Tensor,candidates: torch.Tensor,
                dt: torch.Tensor,state: CertifiedCausalState) -> NativeCausalPreview:
        self._check(sensor,state)
        raw=self.model.preview(sensor,candidates,dt,state.factual)
        return NativeCausalPreview(raw.predictions,
            raw.factual_ready & self.evidence_ready(sensor,state))

    def select(self,sensor: torch.Tensor,candidates: torch.Tensor,dt: torch.Tensor,
               external_goal: torch.Tensor,state: CertifiedCausalState) -> torch.Tensor:
        result=self.preview(sensor,candidates,dt,state)
        self.model._analog(external_goal,tuple(sensor.shape),'EXTERNAL_GOAL')
        self.model._on_model(external_goal)
        if not bool(result.factual_ready.all()):
            raise ValueError('P43R1_FACTUAL_ACTION_EFFECT_UNIDENTIFIABLE')
        return ((result.predictions-external_goal.unsqueeze(1)).square().sum(-1)).argmin(-1)
