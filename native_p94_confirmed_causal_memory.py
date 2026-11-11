"""P94: confirmed action/consequence fast-state boundary for event-vector Hybrid.

ENGINEERING ONLY: exact P86 W256 recurrence, not an improvement or science result.
No source first-use, OOD access, model replacement or original model edit.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FactualState:
    fast_weights: torch.Tensor  # [batch,16,16]; 256 predictive float32/world
    confirmed_count: torch.Tensor  # [batch] control/receipt int64, disclosed

    def snapshot(self) -> "FactualState":
        return FactualState(self.fast_weights.detach().clone(), self.confirmed_count.clone())


class ConfirmedCausalMemory(nn.Module):
    """Exact P86 CAL-locked raw W256 update, with externally confirmed facts."""
    def __init__(self, width: int = 16):
        super().__init__()
        if type(width) is not int or width <= 0:
            raise ValueError("P94_INVALID_WIDTH")
        self.width = width
        self.eta = 1.0
        self.decay = 0.99

    def _check_actions(self, x: torch.Tensor, rank: int) -> None:
        if (not isinstance(x, torch.Tensor) or x.ndim != rank or
            x.shape[-1] != self.width or x.dtype != torch.float32 or
            not bool(torch.isfinite(x).all())):
            raise ValueError("P94_INVALID_ACTIONS")

    def _check_state(self, state: FactualState, actions: torch.Tensor) -> None:
        if (not isinstance(state, FactualState) or
            state.fast_weights.shape != (len(actions), self.width, self.width) or
            state.fast_weights.dtype != torch.float32 or
            state.fast_weights.device != actions.device or
            state.confirmed_count.shape != (len(actions),) or
            state.confirmed_count.dtype != torch.int64 or
            state.confirmed_count.device != actions.device or
            not bool(torch.isfinite(state.fast_weights).all()) or
            not bool((state.confirmed_count >= 0).all())):
            raise ValueError("P94_INVALID_STATE")

    def initial_state(self, actions: torch.Tensor) -> FactualState:
        self._check_actions(actions, 2)
        return FactualState(actions.new_zeros((len(actions), self.width, self.width)),
                            torch.zeros(len(actions), device=actions.device, dtype=torch.int64))

    def read_only(self, query_action: torch.Tensor, state: FactualState) -> torch.Tensor:
        self._check_actions(query_action, 2)
        self._check_state(state, query_action)
        return torch.bmm(state.fast_weights, F.normalize(query_action,dim=-1).unsqueeze(-1)).squeeze(-1)

    def ingest_confirmed(self, action: torch.Tensor, effect: torch.Tensor,
                         confirmed: torch.Tensor, state: FactualState) -> FactualState:
        """Unconfirmed effects, including NaN sentinels, cannot affect predictive state."""
        self._check_actions(action, 3)
        self._check_state(state, action)
        batch, steps, _ = action.shape
        if (not isinstance(effect, torch.Tensor) or effect.shape != action.shape or
            effect.dtype != torch.float32 or effect.device != action.device or
            not isinstance(confirmed, torch.Tensor) or
            confirmed.shape != (batch,steps) or confirmed.dtype != torch.bool or
            confirmed.device != action.device):
            raise ValueError("P94_INVALID_FACTS")
        safe_effect = torch.where(confirmed.unsqueeze(-1), effect, torch.zeros_like(effect))
        if not bool(torch.isfinite(safe_effect).all()):
            raise ValueError("P94_NONFINITE_CONFIRMED_EFFECT")
        weights, count = state.fast_weights, state.confirmed_count
        for t in range(steps):
            key = F.normalize(action[:,t],dim=-1)
            predicted = torch.bmm(weights,key.unsqueeze(-1)).squeeze(-1)
            err = safe_effect[:,t]-predicted
            candidate = self.decay*weights+self.eta*err.unsqueeze(-1)*key.unsqueeze(1)
            selected = confirmed[:,t]
            weights = torch.where(selected[:,None,None],candidate,weights)
            count = count+selected.to(dtype=torch.int64)
        return FactualState(weights,count)

    def forward_episode(self, history: torch.Tensor, query_action: torch.Tensor,
                        confirmed: torch.Tensor | None = None,
                        initial: FactualState | None = None) -> tuple[torch.Tensor,FactualState]:
        self._check_actions(query_action,2)
        if (not isinstance(history, torch.Tensor) or history.ndim != 3 or
            history.shape[-1] != 2*self.width or history.dtype != torch.float32 or
            history.device != query_action.device or len(history) != len(query_action)):
            raise ValueError("P94_INVALID_HISTORY")
        if confirmed is None:
            confirmed = torch.ones(history.shape[:2],dtype=torch.bool,device=history.device)
        state = self.initial_state(query_action) if initial is None else initial
        observed = self.ingest_confirmed(history[...,:self.width],
                                         history[...,self.width:],confirmed,state)
        return self.read_only(query_action,observed), observed


class AnchoredHybridPilot(nn.Module):
    """Original HybridPilot remains unchanged; its optional residual is zero-gated."""
    def __init__(self,width: int = 16):
        super().__init__()
        from model import Configuration, HybridPilot
        self.factual = ConfirmedCausalMemory(width)
        self.original = HybridPilot(2*width,width,Configuration(width=width,heads=2))
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(self, history: torch.Tensor, query_action: torch.Tensor,
                confirmed: torch.Tensor | None = None, *,
                use_residual: bool = False) -> torch.Tensor:
        baseline,_ = self.factual.forward_episode(history,query_action,confirmed)
        if not use_residual:
            return baseline
        if confirmed is None:
            confirmed = torch.ones(history.shape[:2],dtype=torch.bool,device=history.device)
        safe_effect = torch.where(confirmed.unsqueeze(-1),
                                  history[...,self.factual.width:],
                                  torch.zeros_like(history[...,self.factual.width:]))
        facts = torch.cat((history[...,:self.factual.width],safe_effect),dim=-1)
        q = torch.cat((query_action,torch.zeros_like(query_action)),dim=-1)
        seq = torch.cat((facts,q[:,None,:]),dim=1)
        return baseline+self.residual_scale*self.original(seq)[:,-1]
