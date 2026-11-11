"""P88 engineering-only TTT confirmed-write adapter.

Exact native TTT key/query/value update rule from model.py. With all writes
enabled, the result is identical to the original implementation. With
confirmed-fact mask, the 256-float fast state is only updated by observations.
This is NOT a claimed trained model improvement; see P88 negative receipt.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def confirmed_mask(length: int) -> torch.Tensor:
    if length != 97:
        raise ValueError("P88_EXPECTED_97_INTERLEAVED_TOKENS")
    return (torch.arange(length) % 2) == 1


def native_ttt_confirmed(ttt, x: torch.Tensor, mask: torch.Tensor):
    """Causal read-before-write, with exactly the original learnable weights."""
    if x.ndim != 3 or mask.shape != (x.shape[1],) or mask.dtype != torch.bool:
        raise ValueError("P88_INVALID_FACT_CONFIRMATION_MASK")
    b, length, width = x.shape
    z = ttt.norm(x)
    keys = F.normalize(ttt.key(z), dim=-1)
    queries = F.normalize(ttt.query(z), dim=-1)
    values = torch.tanh(ttt.value(z))
    state = x.new_zeros((b, width, width))
    reads = []
    for t in range(length):
        q = queries[:, t]
        reads.append(torch.bmm(state, q.unsqueeze(-1)).squeeze(-1))
        if bool(mask[t]):
            key = keys[:, t]
            predicted = torch.bmm(state, key.unsqueeze(-1)).squeeze(-1)
            state = state + ttt.eta * ((values[:, t] - predicted).unsqueeze(-1) * key.unsqueeze(1))
    return x + ttt.proj(torch.stack(reads, dim=1)), state


def paired_stream(history: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Q0,O0,...,Q47,O47,Q48; no query contains its own future effect."""
    if history.ndim != 3 or history.shape[1:] != (48, 32) or query.shape != (len(history), 32):
        raise ValueError("P88_INVALID_CAUSAL_WORLD")
    if torch.count_nonzero(query[:, 16:]):
        raise ValueError("P88_FUTURE_EFFECT_NOT_HIDDEN")
    out = history.new_zeros((len(history), 97, 32))
    out[:, ::2, :16] = torch.cat((history[:, :, :16], query[:, None, :16]), dim=1)
    out[:, 1::2] = history
    return out
