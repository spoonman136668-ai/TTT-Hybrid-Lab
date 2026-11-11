"""P78 opt-in 256-float factual raw-action TTT fast state (engineering DEV).

Writes only executed actions and their observed effects, before any future query.
This replaces the *TTT read/write memory operand* inside a four-block reference
HybridPilot continuation; the original SSM/attention/depth/head source is not
modified. The model is not pretrained, physically attested, or production ready.
"""
from __future__ import annotations
import torch
from dataclasses import dataclass
from torch.nn import functional as F

D = 16
ETA = 0.5
POLICY_TAG = "P78_BOUND_RAW_ACTION_EFFECT_ETA_0.5_V1"

@dataclass(frozen=True)
class RawTTTState:
    fast_weights: torch.Tensor
    confirmed_count: torch.Tensor
    policy_tag: str = POLICY_TAG

class BoundedRawTTT:
    def __init__(self, max_events=48):
        if type(max_events) is not int or not 1<=max_events<=128:
            raise ValueError('P78_INVALID_BOUND')
        self.max_events=max_events

    def initial_state(self, template):
        if (not isinstance(template,torch.Tensor) or template.ndim!=2 or
                template.shape[1]!=32 or template.dtype!=torch.float32 or
                not bool(torch.isfinite(template).all())):
            raise ValueError('P78_BAD_TEMPLATE')
        b=len(template)
        return RawTTTState(template.new_zeros((b,D,D)),torch.zeros(b,dtype=torch.long,device=template.device))

    def _check(self,x,state):
        if (not isinstance(x,torch.Tensor) or x.ndim!=3 or
                x.shape[-1]!=32 or x.shape[1]<1 or x.dtype!=torch.float32 or
                not bool(torch.isfinite(x).all()) or
                not isinstance(state,RawTTTState) or
                state.policy_tag!=POLICY_TAG or
                state.fast_weights.shape!=(len(x),D,D) or
                state.fast_weights.dtype!=x.dtype or state.fast_weights.device!=x.device or
                not bool(torch.isfinite(state.fast_weights).all()) or
                state.confirmed_count.shape!=(len(x),) or
                state.confirmed_count.dtype!=torch.long or state.confirmed_count.device!=x.device or
                not bool((state.confirmed_count>=0).all()) or
                not bool((state.confirmed_count<=self.max_events).all())):
            raise ValueError('P78_INVALID_FACT_OR_CARRY')
        if not bool((x[:,:,:D].norm(dim=-1)>1e-8).all()):
            raise ValueError('P78_ZERO_ACTION')

    def ingest_confirmed(self, facts, state, confirmed):
        self._check(facts,state)
        if (not isinstance(confirmed,torch.Tensor) or confirmed.shape!=facts.shape[:2] or
                confirmed.dtype!=torch.bool or confirmed.device!=facts.device or
                not bool(confirmed.all())):
            raise ValueError('P78_UNCONFIRMED_FACT')
        if not bool((state.confirmed_count+facts.shape[1] <= self.max_events).all()):
            raise ValueError('P78_FACT_CAPACITY_EXCEEDED')
        keys=F.normalize(facts[:,:,:D],dim=-1)
        obs=facts[:,:,D:]
        fast=state.fast_weights
        for i in range(facts.shape[1]):
            key=keys[:,i]
            estimated=torch.bmm(fast,key.unsqueeze(-1)).squeeze(-1)
            fast=fast+ETA*((obs[:,i]-estimated).unsqueeze(-1)*key.unsqueeze(1))
        return RawTTTState(fast,state.confirmed_count+facts.shape[1])

    def read_only(self, query, state):
        if (not isinstance(query,torch.Tensor) or query.ndim!=2 or
                query.shape[1]!=D or query.dtype!=torch.float32 or
                not bool(torch.isfinite(query).all()) or
                not bool((query.norm(dim=-1)>1e-8).all())):
            raise ValueError('P78_BAD_QUERY')
        # Validate carry without needing another factual event.
        self._check(torch.cat((query,torch.zeros_like(query)),-1)[:,None],state)
        if not bool((state.confirmed_count>0).all()):
            raise ValueError('P78_NO_CONFIRMED_HISTORY')
        return torch.bmm(state.fast_weights,F.normalize(query,dim=-1).unsqueeze(-1)).squeeze(-1)

    def read_from_episode(self,facts,query):
        initial=self.initial_state(facts[:,0])
        updated=self.ingest_confirmed(facts,initial,torch.ones(facts.shape[:2],dtype=torch.bool,device=facts.device))
        return self.read_only(query,updated),updated
