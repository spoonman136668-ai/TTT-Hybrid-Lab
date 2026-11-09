"""P45 DEV: causal fast-effect memory from completed ABA interventions.

A learned scalar blends the original P43 recurrent action effect with a bounded
physical action contrast. The contrast requires three *completed* actions
A,-A,A; timestamp interpolation cancels locally affine nuisance drift.
No future observation or unexecuted alternative is observed by the mechanism.
An ABA contrast is itself a strong conventional baseline, not novel science.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import nn
from native_trainable_causal_belief import NativeTrainableCausalBelief, NativeCausalPreview
from native_p43_identifiability import CertifiedCausalState, CertifiedFactualCausalBelief


@dataclass(frozen=True)
class ABAFastState:
    certified: CertifiedCausalState
    actions: torch.Tensor          # [batch, 3, 1]; trailing actual actions
    deltas: torch.Tensor           # [batch, 3, sensor_dim]; trailing actual changes
    times: torch.Tensor            # [batch, 3, 1]; trailing factual elapsed times
    recent_count: torch.Tensor     # [batch], 0..3
    fast_effect: torch.Tensor      # [batch, sensor_dim], latest factual ABA contrast
    factual_contrasts: torch.Tensor  # [batch], successful ABA updates


class NativeABAFastAdapter(nn.Module):
    """Optional factual adapter; never replaces P43 learned state or its rank gate.

    The trainable gate can be fit using only observed outcomes; it must also
    be compared against a zero-trained ABA estimate on any science claim.
    """
    def __init__(self, native: NativeTrainableCausalBelief):
        super().__init__()
        if not isinstance(native, NativeTrainableCausalBelief):
            raise TypeError('P45_P43_MODEL_REQUIRED')
        if native.sensor_dim < 1:
            raise ValueError('P45_POSITIVE_SENSOR_DIM_REQUIRED')
        self.native = native
        self.certifier = CertifiedFactualCausalBelief(native)
        self.blend_logit = nn.Parameter(torch.tensor(0.0, dtype=next(native.parameters()).dtype,
                                                    device=next(native.parameters()).device))

    def initial_state(self, sensor: torch.Tensor) -> ABAFastState:
        factual = self.certifier.initial_state(sensor)
        b = sensor.shape[0]
        z = sensor.new_zeros
        i = lambda: torch.zeros((b,), dtype=torch.int64, device=sensor.device)
        return ABAFastState(factual, z((b,3,1)),z((b,3,self.native.sensor_dim)),
                            z((b,3,1)),i(),z((b,self.native.sensor_dim)),i())

    def _check(self, sensor: torch.Tensor, state: ABAFastState) -> None:
        if not isinstance(state,ABAFastState):
            raise ValueError('P45_FAST_STATE_REQUIRED')
        self.certifier._check(sensor,state.certified)
        b=sensor.shape[0]
        values=(state.actions,state.deltas,state.times,state.recent_count,
                state.fast_effect,state.factual_contrasts)
        shapes=((b,3,1),(b,3,self.native.sensor_dim),(b,3,1),
                (b,),(b,self.native.sensor_dim),(b,))
        if any(not isinstance(v,torch.Tensor) or v.shape!=shape or v.device!=sensor.device
               for v,shape in zip(values,shapes)):
            raise ValueError('P45_STATE_SHAPE_DEVICE_DRIFT')
        if (any(v.dtype!=sensor.dtype for v in (state.actions,state.deltas,state.times,state.fast_effect))
            or state.recent_count.dtype!=torch.int64 or state.factual_contrasts.dtype!=torch.int64):
            raise ValueError('P45_STATE_DTYPE_DRIFT')
        if any(not bool(torch.isfinite(v).all()) for v in values[:3]+(state.fast_effect,)):
            raise ValueError('P45_NONFINITE_FAST_STATE')
        seen=state.certified.factual.observations
        if (not bool(((state.recent_count>=0)&(state.recent_count<=3)).all())
            or not bool((state.recent_count==seen.clamp(max=3)).all())
            or not bool(((state.factual_contrasts>=0)&
                          (state.factual_contrasts<=(seen-2).clamp(min=0))).all())):
            raise ValueError('P45_FACTUAL_COUNT_DRIFT')
        for row in range(b):
            n=int(state.recent_count[row])
            if n and not bool((state.actions[row,-n:,0].abs()>1e-8).all()):
                raise ValueError('P45_ZERO_FACTUAL_ACTION_IN_STATE')
            if n>1 and not bool((state.times[row,-n+1:,0]>state.times[row,-n:-1,0]).all()):
                raise ValueError('P45_FACTUAL_TIME_ORDER_DRIFT')

    def observe(self, sensor: torch.Tensor, executed_action: torch.Tensor,
                dt: torch.Tensor, observed_next: torch.Tensor,
                state: ABAFastState) -> ABAFastState:
        """One caller-authenticated physical action/outcome; no speculative writes."""
        self._check(sensor,state)
        # P43-R1 validates real action, source outcome and dt before any return.
        certified=self.certifier.observe(sensor,executed_action,dt,observed_next,state.certified)
        actions=torch.cat((state.actions[:,1:,:],executed_action.unsqueeze(1)),dim=1)
        deltas=torch.cat((state.deltas[:,1:,:],(observed_next-sensor).unsqueeze(1)),dim=1)
        times=torch.cat((state.times[:,1:,:],certified.factual.elapsed.unsqueeze(1)),dim=1)
        count=(state.recent_count+1).clamp(max=3)
        a0,a1,a2=[actions[:,i,0] for i in range(3)]
        t0,t1,t2=[times[:,i,0] for i in range(3)]
        valid=(count==3) & torch.isclose(a0,a2,rtol=0,atol=1e-6) & (
            torch.isclose(a0,-a1,rtol=0,atol=1e-6)) & (a0.abs()>1e-8)
        valid=valid & (t0<t1) & (t1<t2)
        # Avoid divide by zero even for rows whose evidence has not matured.
        fraction=(t1-t0)/(t2-t0).clamp_min(1e-12)
        interpolation=deltas[:,0,:]+fraction[:,None]*(deltas[:,2,:]-deltas[:,0,:])
        estimate=(interpolation-deltas[:,1,:])/(2*a0.clamp(min=1e-8).where(a0>=0,a0.clamp(max=-1e-8)))[:,None]
        new_fast=torch.where(valid[:,None],estimate,state.fast_effect)
        if not bool(torch.isfinite(new_fast).all()):
            raise ValueError('P45_NONFINITE_CONTRAST')
        result=ABAFastState(certified,actions,deltas,times,count,new_fast,
                            state.factual_contrasts+valid.long())
        self._check(sensor,result)
        return result

    def preview(self,sensor: torch.Tensor,candidates: torch.Tensor,dt: torch.Tensor,
                state: ABAFastState) -> NativeCausalPreview:
        """Re-evaluate one factual prior; do not advance recurrent or fast state."""
        self._check(sensor,state)
        original=self.certifier.preview(sensor,candidates,dt,state.certified)
        factual=state.certified.factual
        common=self.native.common(torch.cat((factual.hidden,sensor,torch.log1p(dt)),dim=-1))
        original_effect=self.native.action_effect_head(common)
        has_evidence=(state.factual_contrasts>0).to(sensor.dtype)
        mix=torch.sigmoid(self.blend_logit)*has_evidence
        correction=(state.fast_effect-original_effect)*mix[:,None]
        new_predictions=original.predictions+candidates*correction.unsqueeze(1)
        if not bool(torch.isfinite(new_predictions).all()):
            raise ValueError('P45_NONFINITE_PREVIEW')
        return NativeCausalPreview(new_predictions,original.factual_ready)

    def select(self,sensor: torch.Tensor,candidates: torch.Tensor,dt: torch.Tensor,
               external_goal: torch.Tensor,state: ABAFastState) -> torch.Tensor:
        pred=self.preview(sensor,candidates,dt,state)
        self.native._analog(external_goal,tuple(sensor.shape),'EXTERNAL_GOAL')
        self.native._on_model(external_goal)
        if not bool(pred.factual_ready.all()):
            raise ValueError('P45_FACTUAL_ACTION_UNIDENTIFIABLE')
        return (pred.predictions-external_goal[:,None,:]).square().sum(-1).argmin(-1)
