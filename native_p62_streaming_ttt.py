"""P62 opt-in factual streaming state for the original TTTLinearMemory.

Engineering reference only. The caller, not this module, authenticates that
committed inputs represent completed real observations. Candidate previews
read existing fast state and cannot write it. Does not stream SSM or attention.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from model import TTTLinearMemory


@dataclass(frozen=True)
class TTTFactualState:
    fast_weights: torch.Tensor  # [batch, width, width]
    confirmed_count: torch.Tensor  # [batch] int64


class StreamingTTT:
    """Explicit, bounded read-before-write TTT state, zero new parameters."""
    def __init__(self, core: TTTLinearMemory, max_confirmed_events: int = 128):
        if not isinstance(core, TTTLinearMemory):
            raise TypeError("P62_REQUIRES_ORIGINAL_TTT_CORE")
        if type(max_confirmed_events) is not int or not 1 <= max_confirmed_events <= 4096:
            raise ValueError("P62_INVALID_BOUNDED_EVENT_LIMIT")
        self.core = core
        self.max_confirmed_events = max_confirmed_events
        self.width = core.key.out_features

    def initial_state(self, template: torch.Tensor) -> TTTFactualState:
        if not isinstance(template, torch.Tensor) or template.ndim != 2 or template.shape[0] < 1 or template.shape[1] != self.width:
            raise ValueError("P62_BAD_INITIAL_TEMPLATE")
        if template.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(template).all()):
            raise ValueError("P62_NONFINITE_OR_UNSUPPORTED_DTYPE")
        if template.dtype != self.core.key.weight.dtype or template.device != self.core.key.weight.device:
            raise ValueError("P62_CORE_DEVICE_DTYPE_MISMATCH")
        return TTTFactualState(template.new_zeros((template.shape[0], self.width, self.width)),
                               torch.zeros(template.shape[0], device=template.device, dtype=torch.long))

    def _validate(self, x: torch.Tensor, state: TTTFactualState) -> None:
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 1 or x.shape[2] != self.width:
            raise ValueError("P62_BAD_SEQUENCE_SHAPE")
        if x.dtype not in (torch.float32, torch.float64) or x.dtype != self.core.key.weight.dtype or x.device != self.core.key.weight.device:
            raise ValueError("P62_CORE_DEVICE_DTYPE_MISMATCH")
        if not isinstance(state, TTTFactualState) or not isinstance(state.fast_weights, torch.Tensor) or not isinstance(state.confirmed_count, torch.Tensor):
            raise ValueError("P62_BAD_COLD_STATE")
        if (state.fast_weights.shape != (x.shape[0], self.width, self.width) or
                state.confirmed_count.shape != (x.shape[0],) or
                state.fast_weights.device != x.device or state.fast_weights.dtype != x.dtype or
                state.confirmed_count.dtype != torch.long or state.confirmed_count.device != x.device):
            raise ValueError("P62_STATE_SHAPE_DEVICE_DTYPE_MISMATCH")
        if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(state.fast_weights).all()):
            raise ValueError("P62_NONFINITE_FACT_OR_STATE")
        if not bool(((state.confirmed_count >= 0) & (state.confirmed_count <= self.max_confirmed_events)).all()):
            raise ValueError("P62_BAD_OBSERVATION_COUNT")

    def _features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.core.norm(x)
        return (F.normalize(self.core.key(z), dim=-1),
                F.normalize(self.core.query(z), dim=-1),
                torch.tanh(self.core.value(z)))

    def ingest_confirmed(self, factual: torch.Tensor, state: TTTFactualState,
                         confirmed: torch.Tensor) -> tuple[torch.Tensor, TTTFactualState]:
        """Only fully confirmed completed facts may write; no partial batch commits."""
        self._validate(factual, state)
        if (not isinstance(confirmed, torch.Tensor) or confirmed.shape != factual.shape[:2]
                or confirmed.device != factual.device or confirmed.dtype != torch.bool
                or not bool(confirmed.all())):
            raise ValueError("P62_UNCONFIRMED_FACT_MUST_NOT_COMMIT")
        if not bool((state.confirmed_count + factual.shape[1] <= self.max_confirmed_events).all()):
            raise ValueError("P62_BOUNDED_FACTUAL_MEMORY_FULL")
        k, q, v = self._features(factual)
        fast = state.fast_weights
        reads = []
        for t in range(factual.shape[1]):
            read = torch.bmm(fast, q[:, t, :, None]).squeeze(-1)
            reads.append(read)
            estimated = torch.bmm(fast, k[:, t, :, None]).squeeze(-1)
            fast = fast + self.core.eta * ((v[:, t] - estimated)[:, :, None] * k[:, t, None, :])
        output = factual + self.core.proj(torch.stack(reads, dim=1))
        next_state = TTTFactualState(fast, state.confirmed_count + factual.shape[1])
        return output, next_state

    def preview(self, candidates: torch.Tensor, state: TTTFactualState) -> torch.Tensor:
        """Each candidate independently reads the exact same pre-action fast state."""
        self._validate(candidates, state)
        z = self.core.norm(candidates)
        queries = F.normalize(self.core.query(z), dim=-1)
        read = torch.einsum("bij,bkj->bki", state.fast_weights, queries)
        return candidates + self.core.proj(read)
