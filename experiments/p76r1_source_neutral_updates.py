"""P76-R1 source-neutral update-rule interface; no data generation or training.

The original P76 study and model stay unchanged. All examples exercising this
module must be fabricated in-memory interface fixtures, not P76 FIT/VAL/OOD.
Future independent source first-use and any scientific comparison are separate.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from experiments.p76_memory_repair import D, STATE_FLOATS, _facts, _query

SCHEMA = "p76.action-effect16.v1"
FEATURES = {
    "full5": (0, 1, 2, 3, 4),
    "no_contradiction4": (0, 1, 2, 3),
    "direction2": (0, 3),
    "magnitude3": (0, 1, 2),
}
FIXED_ETAS = (0.2, 0.35, 0.5, 0.65, 0.8, 1.0)


class SourceNeutralUpdate(nn.Module):
    """Factual 16x16 memory; static learned gate has no hidden per-world state.

    Caller must supply FIT-derived theta later; this class never chooses a
    training seed or a checkpoint and cannot instantiate a source environment.
    """

    def __init__(self, method: str, *, theta: torch.Tensor | None = None,
                 fixed_eta: float | None = None):
        super().__init__()
        if method == "fixed":
            if theta is not None or type(fixed_eta) is not float or fixed_eta not in FIXED_ETAS:
                raise ValueError("P76R1_FIXED_RATE_NOT_PREDECLARED")
            self.eta = fixed_eta
        elif method in FEATURES:
            if fixed_eta is not None or not isinstance(theta, torch.Tensor):
                raise ValueError("P76R1_THETA_REQUIRED")
            if (theta.shape != (len(FEATURES[method]),) or
                    theta.dtype != torch.float32 or
                    not bool(torch.isfinite(theta).all())):
                raise ValueError("P76R1_THETA_INVALID")
            self.theta = nn.Parameter(theta.detach().clone())
            self.eta = None
        else:
            raise ValueError("P76R1_UNKNOWN_METHOD")
        self.method = method

    @property
    def persistent_floats(self) -> int:
        return STATE_FLOATS

    @property
    def learned_static_floats(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode(self, facts: dict, *, schema: str, carry: torch.Tensor | None = None) -> torch.Tensor:
        if schema != SCHEMA:
            raise ValueError("P76R1_UNAPPROVED_UNITS_OR_SOURCE_SCHEMA")
        a, y = _facts(facts)
        if a.dtype != torch.float32:
            raise ValueError("P76R1_ONLY_FLOAT32")
        if not bool(torch.all((a.norm(dim=-1) - 1).abs() < 1e-4)):
            raise ValueError("P76R1_ACTIONS_NOT_UNIT_NORMALIZED")
        b = a.shape[0]
        if carry is None:
            w = a.new_zeros((b, D, D))
        else:
            if (not isinstance(carry, torch.Tensor) or carry.shape != (b, D, D) or
                    carry.dtype != a.dtype or carry.device != a.device or
                    not bool(torch.isfinite(carry).all())):
                raise ValueError("P76R1_INVALID_COLD_STATE")
            w = carry.clone()
        for t in range(a.shape[1]):
            key, observed = a[:, t], y[:, t]
            prediction = (w @ key[:, :, None]).squeeze(-1)
            residual = observed - prediction
            if self.method == "fixed":
                gain = a.new_full((b,), self.eta)
            else:
                cosine = F.cosine_similarity(prediction, observed, dim=-1, eps=1e-6)
                confidence = prediction.norm(dim=-1) / (observed.norm(dim=-1) + 1e-6)
                features = torch.stack((
                    torch.ones_like(cosine),
                    torch.log(residual.square().mean(-1) + 1e-6),
                    torch.log(prediction.square().mean(-1) + 1e-6),
                    cosine,
                    torch.relu(-cosine) * confidence.clamp(0, 1),
                ), dim=-1)
                gain = torch.sigmoid(features[:, FEATURES[self.method]] @ self.theta)
            w = w + gain[:, None, None] * residual[:, :, None] * key[:, None, :]
        if not bool(torch.isfinite(w).all()):
            raise ValueError("P76R1_NONFINITE_MEMORY")
        return w

    def read(self, state: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        if (not isinstance(state, torch.Tensor) or state.ndim != 3 or
                state.shape[1:] != (D, D) or state.dtype != torch.float32 or
                not bool(torch.isfinite(state).all())):
            raise ValueError("P76R1_INVALID_COLD_STATE")
        q = _query(query, len(state), state.dtype, state.device)
        return (state @ q[:, :, None]).squeeze(-1)

    def forward(self, facts: dict, query: torch.Tensor, *, schema: str,
                carry: torch.Tensor | None = None) -> torch.Tensor:
        return self.read(self.encode(facts, schema=schema, carry=carry), query)
