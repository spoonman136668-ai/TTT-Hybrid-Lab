"""P77 preregistered 256-float factual memory controls (FIT/VAL engineering only).

All modes update a 16x16 state using historical executed (action, effect)
only, with read-before-write. No labels or future query enter a memory write.
Source and threshold freeze: experiments/p77_directional_write_freeze.json.
"""
import torch
from torch import nn
from torch.nn import functional as F
from p76_memory_repair import _facts, _query, D

MODEL_SEEDS = (977201, 977203, 977207)
MODES = ('full5', 'cosine4', 'non_directional3', 'p70_four_param',
         'fixed_eta', 'fixed_contradiction')
ETA_GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
N_FEATURES = {'full5': 5, 'cosine4': 4, 'non_directional3': 3,
              'p70_four_param': 4}


class WritePolicy(nn.Module):
    def __init__(self, mode, seed=None, eta=None):
        super().__init__()
        if mode not in MODES:
            raise ValueError('P77_BAD_MODE')
        self.mode = mode
        if mode == 'fixed_eta':
            if eta is None or not any(abs(float(eta)-x)<1e-10 for x in ETA_GRID) or seed is not None:
                raise ValueError('P77_BAD_ETA')
            self.eta = float(eta)
        elif eta is not None:
            raise ValueError('P77_ETA_ONLY_FOR_FIXED')
        if mode in N_FEATURES:
            if seed not in MODEL_SEEDS:
                raise ValueError('P77_UNREGISTERED_SEED')
            g = torch.Generator(device='cpu').manual_seed(seed)
            self.theta = nn.Parameter((torch.rand(N_FEATURES[mode], generator=g)-0.5)*0.1)
        elif seed is not None:
            raise ValueError('P77_FIXED_SEED')

    def encode(self, facts, carry=None):
        a, y = _facts(facts)
        b = len(a)
        if carry is None:
            w = a.new_zeros((b, D, D))
        else:
            if (not isinstance(carry, torch.Tensor) or carry.shape!=(b,D,D)
                    or carry.dtype!=a.dtype or carry.device!=a.device
                    or not bool(torch.isfinite(carry).all())):
                raise ValueError('P77_BAD_CARRY')
            w = carry.clone()
        for t in range(a.shape[1]):
            key = a[:, t]
            observed = y[:, t]
            pred = torch.bmm(w, key.unsqueeze(-1)).squeeze(-1)
            residual = observed-pred
            if self.mode=='fixed_eta':
                gain = pred.new_full((b,), self.eta)
            else:
                cosine = F.cosine_similarity(pred, observed, dim=-1, eps=1e-6)
                if self.mode=='fixed_contradiction':
                    confidence=pred.norm(dim=-1)/(observed.norm(dim=-1)+1e-6)
                    gain = 0.5 + 0.5*((cosine<0)&(confidence>0.30)).to(a.dtype)
                else:
                    features = [torch.ones_like(cosine),
                                torch.log(residual.square().mean(-1)+1e-6),
                                torch.log(pred.square().mean(-1)+1e-6)]
                    if self.mode!='non_directional3':
                        features.append(cosine)
                    if self.mode=='full5':
                        confidence = (pred.norm(dim=-1)/(observed.norm(dim=-1)+1e-6)).clamp(0,1)
                        features.append(torch.relu(-cosine)*confidence)
                    gain = torch.sigmoid(torch.stack(features,-1)@self.theta)
                    if self.mode=='p70_four_param':
                        gain = 0.5*gain
            w = w + gain[:,None,None]*residual[:,:,None]*key[:,None,:]
        return w

    def read(self, state, query):
        if not isinstance(state,torch.Tensor) or state.ndim!=3 or state.shape[1:]!=(D,D) or not bool(torch.isfinite(state).all()):
            raise ValueError('P77_BAD_STATE')
        q = _query(query,len(state),state.dtype,state.device)
        return torch.bmm(state,q.unsqueeze(-1)).squeeze(-1)

    def forward(self, facts, query, carry=None):
        return self.read(self.encode(facts,carry), query)
