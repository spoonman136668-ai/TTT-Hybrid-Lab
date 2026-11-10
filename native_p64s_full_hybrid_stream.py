"""P64-S DEV: opt-in complete original-Hybrid causal preview with confirmed TTT carry.

Engineering reference only. Caller confirmation is NOT physical source attestation.
The original SSM and attention are recomputed on the confirmed prefix: this is
functionally causal but NOT a fused, constant-memory, or fast streaming model.
Hypothetical queries never enter factual history, fast weights, or telemetry.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from model import HybridPilot
from native_p62_streaming_ttt import StreamingTTT, TTTFactualState


@dataclass(frozen=True)
class FullHybridFactualState:
    history: torch.Tensor  # [batch, facts, input_dim] caller-confirmed analog events
    ttt: TTTFactualState  # original fast matrix after exactly those events


class FullHybridFactualStream:
    """Original full SSM/attention/TTT/depth wrapper; no new weights."""
    def __init__(self, core: HybridPilot, max_confirmed: int = 16):
        if not isinstance(core, HybridPilot) or any(
                getattr(core, name) is None for name in ('ssm', 'attn', 'ttt', 'depth')):
            raise ValueError('P64S_REQUIRES_ALL_ORIGINAL_HYBRID_COMPONENTS')
        if type(max_confirmed) is not int or not 1 <= max_confirmed <= 64:
            raise ValueError('P64S_INVALID_HISTORY_CAPACITY')
        self.core = core
        self.input_dim = core.stem.in_features
        self.max_confirmed = max_confirmed
        self.ttt = StreamingTTT(core.ttt, max_confirmed_events=max_confirmed)

    def _analog(self, x: torch.Tensor, *, seq_min: int = 1, seq_max: int | None = None):
        p = self.core.stem.weight
        if (not isinstance(x, torch.Tensor) or x.ndim != 3
                or not 1 <= x.shape[0] <= 16 or x.shape[2] != self.input_dim
                or not seq_min <= x.shape[1] <= (seq_max if seq_max is not None else self.max_confirmed)
                or x.dtype not in (torch.float32, torch.float64)
                or x.dtype != p.dtype or x.device != p.device
                or not bool(torch.isfinite(x).all())):
            raise ValueError('P64S_INVALID_ANALOG_FACT_OR_QUERY')

    def initial_state(self, template: torch.Tensor) -> FullHybridFactualState:
        if not isinstance(template, torch.Tensor) or template.ndim != 2:
            raise ValueError('P64S_INVALID_TEMPLATE')
        self._analog(template[:, None, :], seq_max=1)
        projected = self.core.stem(template)
        ttt = self.ttt.initial_state(projected)
        return FullHybridFactualState(template[:, None, :][:, :0, :].clone(), ttt)

    def _state(self, x: torch.Tensor, state: FullHybridFactualState):
        if not isinstance(state, FullHybridFactualState):
            raise ValueError('P64S_MISSING_FACTUAL_STATE')
        history = state.history
        if (not isinstance(history, torch.Tensor) or history.ndim != 3
                or tuple(history.shape[::2]) != (x.shape[0], self.input_dim)
                or not 0 <= history.shape[1] <= self.max_confirmed
                or history.dtype != x.dtype or history.device != x.device
                or not bool(torch.isfinite(history).all())):
            raise ValueError('P64S_CORRUPTED_HISTORY')
        ttt = state.ttt
        if not isinstance(ttt, TTTFactualState):
            raise ValueError('P64S_CORRUPTED_TTT_STATE')
        if (ttt.confirmed_count.shape != (x.shape[0],)
                or not bool((ttt.confirmed_count == history.shape[1]).all())):
            raise ValueError('P64S_COUNT_MISMATCH')
        # Existing TTT validator checks fast matrix shape, dtype, device and finiteness.
        self.ttt._validate(self.core.stem(x[:, :1, :]), ttt)

    def _before_ttt(self, history: torch.Tensor) -> torch.Tensor:
        z = self.core.stem(history)
        return self.core.attn(self.core.ssm(z))

    def _pure_depth(self, x: torch.Tensor) -> torch.Tensor:
        """Precisely original depth math without the mutable utilization counter."""
        depth = self.core.depth
        z = depth.norm(x)
        p = torch.sigmoid(depth.router(z))
        decision = (p >= 0.5).squeeze(-1)
        out = torch.zeros_like(x)
        if bool(decision.any()):
            g = p[decision]
            ste = 1.0 + g - g.detach()
            out[decision] = depth.ffn(z[decision]) * ste
        return x + out

    def ingest_confirmed(self, facts: torch.Tensor, state: FullHybridFactualState,
                         confirmed: torch.Tensor) -> tuple[torch.Tensor, FullHybridFactualState]:
        """Append only caller-confirmed completed action/outcome analog events."""
        self._analog(facts)
        self._state(facts, state)
        if (not isinstance(confirmed, torch.Tensor) or confirmed.shape != facts.shape[:2]
                or confirmed.dtype != torch.bool or confirmed.device != facts.device
                or not bool(confirmed.all())):
            raise ValueError('P64S_UNCONFIRMED_EVENT')
        if state.history.shape[1] + facts.shape[1] > self.max_confirmed:
            raise ValueError('P64S_FACTUAL_HISTORY_FULL')
        extended = torch.cat((state.history, facts), dim=1)
        z = self._before_ttt(extended)[:, -facts.shape[1]:, :]
        transformed, next_fast = self.ttt.ingest_confirmed(z, state.ttt, confirmed)
        output = self.core.head(self.core.depth(transformed))
        if not bool(torch.isfinite(output).all()):
            raise ValueError('P64S_NONFINITE_CONFIRMED_OUTPUT')
        return output, FullHybridFactualState(extended, next_fast)

    def preview(self, queries: torch.Tensor, state: FullHybridFactualState) -> torch.Tensor:
        """Pure per-candidate causal continuation, WITHOUT any factual write."""
        self._analog(queries, seq_max=8)
        self._state(queries, state)
        if state.history.shape[1] == 0:
            raise ValueError('P64S_NO_CONFIRMED_HISTORY')
        b, k, d = queries.shape
        prefix = state.history[:, None, :, :].expand(-1, k, -1, -1)
        full = torch.cat((prefix, queries[:, :, None, :]), dim=2)
        flat = full.reshape(b * k, full.shape[2], d)
        zlast = self._before_ttt(flat)[:, -1:, :]
        fast = TTTFactualState(state.ttt.fast_weights.repeat_interleave(k, dim=0),
                               state.ttt.confirmed_count.repeat_interleave(k, dim=0))
        result = self.ttt.preview(zlast, fast)
        # Unlike core.depth(...), this path does not mutate `last_utilization`.
        output = self.core.head(self._pure_depth(result))[:, 0].reshape(b, k, -1)
        if not bool(torch.isfinite(output).all()):
            raise ValueError('P64S_NONFINITE_PREVIEW')
        return output
