"""Compact *reference* hybrid recurrent-attention-TTT model.

This deliberately uses a selective diagonal state-space recurrence rather than
Mamba's official fused kernels, and an online threshold depth router rather than
Mixture-of-Depths' global top-k token selection. The former is causal and easy
to audit; the latter keeps autoregressive prefix invariance.

No pretrained weights, model quality, or production speed claims are made.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

class SelectiveStateBlock(nn.Module):
    """Mamba-inspired content-selective diagonal recurrence: no fused scan."""
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.delta = nn.Linear(width, width)
        self.candidate = nn.Linear(width, width)
        self.read_gate = nn.Linear(width, width)
        self.log_decay = nn.Parameter(torch.zeros(width))
        self.out = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, steps, width = x.shape
        h = x.new_zeros((b, width))
        ys = []
        z = self.norm(x)
        for t in range(steps):
            xt = z[:, t, :]
            rate = F.softplus(self.delta(xt))
            retention = torch.exp(-rate * F.softplus(self.log_decay))
            h = retention * h + (1.0 - retention) * torch.tanh(self.candidate(xt))
            ys.append(torch.sigmoid(self.read_gate(xt)) * h)
        return x + self.out(torch.stack(ys, dim=1))

class CausalAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int = 2):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads, self.dim = heads, width // heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        def shape(a):
            return a.view(b, t, self.heads, self.dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(shape(q), shape(k), shape(v), is_causal=True)
        return x + self.proj(y.transpose(1, 2).contiguous().view(b, t, w))

class TTTLinearMemory(nn.Module):
    """Online self-supervised fast-weight update; read-before-write is causal.

    W[t+1] = W[t] + eta*(v[t] - W[t]@k[t]) @ k[t]^T.
    This is an illustrative TTT-Linear-type fast state, not TTT-E2E.
    """
    def __init__(self, width: int, learning_rate: float = 0.5):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.query = nn.Linear(width, width)
        self.proj = nn.Linear(width, width)
        self.eta = float(learning_rate)
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError("learning_rate must lie in [0, 1]")

    def forward(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        b, length, width = x.shape
        z = self.norm(x)
        keys = F.normalize(self.key(z), dim=-1)
        queries = F.normalize(self.query(z), dim=-1)
        values = torch.tanh(self.value(z))
        state = x.new_zeros((b, width, width))
        reads = []
        for t in range(length):
            q = queries[:, t]
            reads.append(torch.bmm(state, q.unsqueeze(-1)).squeeze(-1))
            if update:
                k = keys[:, t]
                predicted = torch.bmm(state, k.unsqueeze(-1)).squeeze(-1)
                residual = values[:, t] - predicted
                state = state + self.eta * (residual.unsqueeze(-1) * k.unsqueeze(1))
        return x + self.proj(torch.stack(reads, 1))

class CausalDepthRouter(nn.Module):
    """Token-wise hard threshold routing with sparse FFN evaluation.

    This is MoD-*inspired*, not paper-exact top-k MoD; utilization is not
    guaranteed and hardware speedups are not guaranteed. Routing is causal.
    """
    def __init__(self, width: int, multiplier: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.router = nn.Linear(width, 1)
        self.ffn = nn.Sequential(nn.Linear(width, width * multiplier), nn.GELU(), nn.Linear(width * multiplier, width))
        self.last_utilization = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        p = torch.sigmoid(self.router(z))
        decision = (p >= 0.5).squeeze(-1)
        self.last_utilization = float(decision.float().mean().detach().cpu())
        out = torch.zeros_like(x)
        if decision.any():
            selected = z[decision]
            routed = self.ffn(selected)
            # STE updates selected tokens' router scores; unselected tokens
            # have no surrogate gradient and therefore may never activate.
            g = p[decision]
            ste = 1.0 + g - g.detach()
            out[decision] = routed * ste
        return x + out

@dataclass(frozen=True)
class Configuration:
    width: int = 24
    heads: int = 2
    ssm: bool = True
    attention: bool = True
    ttt: bool = True
    depth: bool = True

class HybridPilot(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        self.stem = nn.Linear(input_dim, cfg.width)
        self.ssm = SelectiveStateBlock(cfg.width) if cfg.ssm else None
        self.attn = CausalAttentionBlock(cfg.width, cfg.heads) if cfg.attention else None
        self.ttt = TTTLinearMemory(cfg.width) if cfg.ttt else None
        self.depth = CausalDepthRouter(cfg.width) if cfg.depth else None
        self.head = nn.Linear(cfg.width, output_dim)

    def forward(self, x: torch.Tensor, ttt_update: bool = True) -> torch.Tensor:
        z = self.stem(x)
        if self.ssm is not None:
            z = self.ssm(z)
        if self.attn is not None:
            z = self.attn(z)
        if self.ttt is not None:
            z = self.ttt(z, update=ttt_update)
        if self.depth is not None:
            z = self.depth(z)
        return self.head(z)
