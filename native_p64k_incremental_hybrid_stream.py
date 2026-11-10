"""P64-S engineering DEV: exact-original HybridPilot streamed across confirmed facts.

This is a bounded, state-explicit *reference* implementation, not physical source
attestation, scientific learning, fused Mamba2, TTT-E2E, or a deployment route.
No original model parameter is modified. Hypothetical candidates only read state.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from model import HybridPilot
from native_p62_streaming_ttt import StreamingTTT, TTTFactualState


@dataclass(frozen=True)
class HybridFactualState:
    ssm_hidden: torch.Tensor        # [B, W]
    attention_k: torch.Tensor       # [B, heads, confirmed_events, head_dim]
    attention_v: torch.Tensor       # [B, heads, confirmed_events, head_dim]
    ttt: TTTFactualState            # fast weights and confirmed count


class StreamingHybridPilot:
    """Incrementally evaluate all four original reference blocks.

    State has explicit diagonal SSM hidden, causal attention K/V prefix and
    original TTT fast weights. Candidate query branches are read-only and
    never become facts. This adapter has zero separately trainable parameters.
    """

    def __init__(self, core: HybridPilot, max_confirmed_events: int = 128):
        if not isinstance(core, HybridPilot) or any(
            getattr(core, name) is None for name in ('ssm','attn','ttt','depth')):
            raise TypeError('P64S_REQUIRES_ORIGINAL_ALL_FOUR_HYBRID')
        if type(max_confirmed_events) is not int or not 1 <= max_confirmed_events <= 128:
            raise ValueError('P64S_BAD_MAX_FACTS')
        self.core=core
        self.input_dim=core.stem.in_features
        self.output_dim=core.head.out_features
        self.width=core.stem.out_features
        self.heads=core.attn.heads
        self.head_dim=core.attn.dim
        if not 1<=self.input_dim<=128 or not 1<=self.width<=128 or self.width!=self.heads*self.head_dim:
            raise ValueError('P64S_BAD_MODEL_DIMENSIONS')
        self.max_confirmed_events=max_confirmed_events
        self.ttt=StreamingTTT(core.ttt,max_confirmed_events)

    def initial_state(self, template: torch.Tensor) -> HybridFactualState:
        self._validate_template(template)
        b=template.shape[0]
        blank=template.new_zeros((b,self.width))
        empty=template.new_zeros((b,self.heads,0,self.head_dim))
        return HybridFactualState(blank,empty,empty.clone(),self.ttt.initial_state(blank))

    def _validate_template(self, x: torch.Tensor):
        if not isinstance(x,torch.Tensor) or x.ndim!=2 or not 1<=x.shape[0]<=256 or x.shape[1]!=self.input_dim:
            raise ValueError('P64S_BAD_INPUT_SHAPE')
        if x.dtype not in (torch.float32,torch.float64) or x.dtype!=self.core.stem.weight.dtype or x.device!=self.core.stem.weight.device:
            raise ValueError('P64S_DEVICE_DTYPE_MISMATCH')
        if not bool(torch.isfinite(x).all()):
            raise ValueError('P64S_NONFINITE_INPUT')

    def _validate_state(self,state: HybridFactualState,template:torch.Tensor):
        if not isinstance(state,HybridFactualState) or not isinstance(state.ttt,TTTFactualState):
            raise ValueError('P64S_BAD_COLD_STATE')
        b=template.shape[0]
        if not isinstance(state.ttt.confirmed_count,torch.Tensor) or state.ttt.confirmed_count.shape!=(b,) or state.ttt.confirmed_count.dtype!=torch.long or state.ttt.confirmed_count.device!=template.device:
            raise ValueError('P64S_BAD_CONFIRMED_COUNT')
        counts=state.ttt.confirmed_count
        if not bool(((counts>=0)&(counts<=self.max_confirmed_events)).all()) or not bool((counts==counts[0]).all()):
            raise ValueError('P64S_BAD_CONFIRMED_COUNT')
        n=int(counts[0].item())
        for field,tensor,shape in (
            ('ssm',state.ssm_hidden,(b,self.width)),
            ('k',state.attention_k,(b,self.heads,n,self.head_dim)),
            ('v',state.attention_v,(b,self.heads,n,self.head_dim)),
            ('ttt',state.ttt.fast_weights,(b,self.width,self.width))):
            if not isinstance(tensor,torch.Tensor) or tensor.shape!=shape or tensor.dtype!=template.dtype or tensor.device!=template.device or not bool(torch.isfinite(tensor).all()):
                raise ValueError('P64S_BAD_'+field.upper()+'_COLD_STATE')

    def _validate_sequence(self,x:torch.Tensor,state:HybridFactualState,cap:int):
        if not isinstance(x,torch.Tensor) or x.ndim!=3 or x.shape[1]<1 or x.shape[1]>cap:
            raise ValueError('P64S_BAD_SEQUENCE')
        self._validate_template(x[:,0,:])
        if x.shape[2]!=self.input_dim or not bool(torch.isfinite(x).all()):
            raise ValueError('P64S_BAD_SEQUENCE')
        self._validate_state(state,x[:,0,:])

    def _depth_without_telemetry_mutation(self,z:torch.Tensor)->torch.Tensor:
        """Equivalent to original router forward, but leaves last_utilization pure."""
        d=self.core.depth
        normalized=d.norm(z)
        p=torch.sigmoid(d.router(normalized))
        selected=(p>=.5).squeeze(-1)
        routed=torch.zeros_like(z)
        if bool(selected.any()):
            q=d.ffn(normalized[selected])
            g=p[selected]
            ste=1.+g-g.detach()
            routed[selected]=q*ste
        return z+routed

    def _step(self,x:torch.Tensor,state:HybridFactualState,commit:bool):
        """A single causal original-prefix step, [B,input_dim] -> [B,output_dim]."""
        core=self.core
        x=x.contiguous()
        stem=core.stem(x[:,None,:])
        s=core.ssm
        xt=s.norm(stem)[:,0,:]
        rate=F.softplus(s.delta(xt))
        retention=torch.exp(-rate*F.softplus(s.log_decay))
        hidden=retention*state.ssm_hidden+(1.-retention)*torch.tanh(s.candidate(xt))
        z=stem+s.out((torch.sigmoid(s.read_gate(xt))*hidden)[:,None,:])
        a=core.attn
        query,key,value=a.qkv(a.norm(z)).chunk(3,dim=-1)
        def heads(inp):
            return inp.view(len(x),1,self.heads,self.head_dim).transpose(1,2)
        q,k,v=map(heads,(query,key,value))
        all_k=torch.cat((state.attention_k,k),dim=2)
        all_v=torch.cat((state.attention_v,v),dim=2)
        attended=F.scaled_dot_product_attention(q,all_k,all_v,is_causal=False)
        z=z+a.proj(attended.transpose(1,2).contiguous().view(len(x),1,self.width))
        if commit:
            z,fast=self.ttt.ingest_confirmed(z,state.ttt,
                        torch.ones((len(x),1),device=x.device,dtype=torch.bool))
        else:
            z=self.ttt.preview(z,state.ttt)
            fast=state.ttt
        z=self._depth_without_telemetry_mutation(z)
        out=core.head(z)[:,0,:]
        # Do not stage or return unobserved candidate prefixes as committed state.
        next_state=HybridFactualState(hidden,all_k,all_v,fast) if commit else state
        return out,next_state

    def ingest_confirmed(self,events:torch.Tensor,state:HybridFactualState,confirmed:torch.Tensor):
        """Append physically confirmed input features, never imagined outcomes."""
        self._validate_sequence(events,state,self.max_confirmed_events)
        if not isinstance(confirmed,torch.Tensor) or confirmed.shape!=events.shape[:2] or confirmed.dtype!=torch.bool or confirmed.device!=events.device or not bool(confirmed.all()):
            raise ValueError('P64S_UNCONFIRMED_FACT_REFUSED')
        if not bool((state.ttt.confirmed_count+events.shape[1]<=self.max_confirmed_events).all()):
            raise ValueError('P64S_BOUNDED_HISTORY_FULL')
        outputs=[]
        current=state
        for i in range(events.shape[1]):
            y,current=self._step(events[:,i,:],current,True)
            outputs.append(y)
        return torch.stack(outputs,dim=1),current

    def preview(self,candidates:torch.Tensor,state:HybridFactualState)->torch.Tensor:
        """Separate one-step same-state candidate branches; no predicted fact writes."""
        self._validate_sequence(candidates,state,16)
        if not bool((state.ttt.confirmed_count > 0).all()):
            raise ValueError('P64S_NO_CONFIRMED_HISTORY')
        outputs=[]
        for i in range(candidates.shape[1]):
            output,_=self._step(candidates[:,i,:],state,False)
            outputs.append(output)
        return torch.stack(outputs,dim=1)
