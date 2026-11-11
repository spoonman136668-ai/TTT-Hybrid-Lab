"""P76 256-float causal fast-memory repair mechanisms, developer research only.

Frozen protocol: experiments/p76_contradiction_repair_freeze.json.
Every write uses ONLY the executed action and its measured consequence, after
prediction. No hidden world labels, future query, event index, or extra state.
No dataset loaders, OOD references, runner, or workflow dispatch in this module.
"""
import torch
from torch import nn
from torch.nn import functional as F

D = 16
STATE_FLOATS = D * D
MODEL_SEEDS = (976201, 976203, 976207)
MODES = ('all_write_eta_0.5','all_write_eta_1.0',
         'fixed_contradiction_repair','learned_contradiction_repair')


def _facts(facts):
    if not isinstance(facts,dict) or set(facts) != {'action','effect'}:
        raise ValueError('P76_FACTUAL_HISTORY_ONLY')
    a, y = facts['action'], facts['effect']
    if not isinstance(a,torch.Tensor) or not isinstance(y,torch.Tensor):
        raise ValueError('P76_BAD_FACTS')
    if (a.ndim!=3 or y.shape!=a.shape or a.shape[-1]!=D or a.dtype!=y.dtype or
        a.device!=y.device or not a.is_floating_point() or a.shape[1]<1):
        raise ValueError('P76_BAD_FACTS')
    if not bool(torch.isfinite(a).all() and torch.isfinite(y).all()):
        raise ValueError('P76_NONFINITE_FACTS')
    if not bool((a.norm(dim=-1)>1e-8).all()):
        raise ValueError('P76_ZERO_ACTION')
    return a,y


def _query(q,b,dtype,device):
    if not isinstance(q,torch.Tensor) or q.shape!=(b,D) or q.dtype!=dtype or q.device!=device or not bool(torch.isfinite(q).all()):
        raise ValueError('P76_BAD_QUERY')
    if not bool((q.norm(dim=-1)>1e-8).all()):
        raise ValueError('P76_ZERO_QUERY')
    return q


class FactualRepair(nn.Module):
    def __init__(self, mode, seed=None):
        super().__init__()
        if mode not in MODES:
            raise ValueError('P76_UNREGISTERED_MODE')
        self.mode=mode
        if mode=='learned_contradiction_repair':
            if seed not in MODEL_SEEDS:
                raise ValueError('P76_UNREGISTERED_MODEL_SEED')
            gen=torch.Generator(device='cpu').manual_seed(seed)
            self.theta=nn.Parameter((torch.rand(5,generator=gen)-.5)*.1)
        elif seed is not None:
            raise ValueError('P76_FIXED_MODE_HAS_NO_SEED')

    def encode(self,facts,carry=None):
        a,y=_facts(facts)
        b=a.shape[0]
        if carry is None:
            w=a.new_zeros((b,D,D))
        else:
            if (not isinstance(carry,torch.Tensor) or carry.shape!=(b,D,D) or
                    carry.dtype!=a.dtype or carry.device!=a.device or not bool(torch.isfinite(carry).all())):
                raise ValueError('P76_BAD_CARRY')
            w=carry.clone()
        for t in range(a.shape[1]):
            key=a[:,t]
            observed=y[:,t]
            prediction=(w@key[:,:,None]).squeeze(-1)
            residual=observed-prediction
            cosine=F.cosine_similarity(prediction,observed,dim=-1,eps=1e-6)
            confidence=(prediction.norm(dim=-1)/(observed.norm(dim=-1)+1e-6))
            if self.mode=='all_write_eta_0.5':
                gain=torch.full((b,),.5,dtype=a.dtype,device=a.device)
            elif self.mode=='all_write_eta_1.0':
                gain=torch.ones(b,dtype=a.dtype,device=a.device)
            elif self.mode=='fixed_contradiction_repair':
                contradiction=(cosine<0) & (confidence>.30)
                gain=.5+.5*contradiction.to(a.dtype)
            else:
                f=torch.stack((
                    torch.ones_like(cosine),
                    torch.log(residual.square().mean(-1)+1e-6),
                    torch.log(prediction.square().mean(-1)+1e-6),
                    cosine,
                    torch.relu(-cosine)*confidence.clamp(0,1)
                ),dim=-1)
                gain=torch.sigmoid(f@self.theta)
            w=w+gain[:,None,None]*residual[:,:,None]*key[:,None,:]
        return w

    def read(self,state,query):
        if (not isinstance(state,torch.Tensor) or state.ndim!=3 or state.shape[1:]!=(D,D) or
                not bool(torch.isfinite(state).all())):
            raise ValueError('P76_BAD_CARRY')
        query=_query(query,len(state),state.dtype,state.device)
        return (state@query[:,:,None]).squeeze(-1)

    def forward(self,facts,query,carry=None):
        return self.read(self.encode(facts,carry),query)
