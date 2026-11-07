"""S25 numeric TTT prototype: separately trained conventional GRU and fast-weight pilot.

No simulator, pretrained checkpoint, optimizer run, or first-use science is
performed by importing this module. All observations are continuous numerical
sensor-action histories; genuine G5 TTT-LM is a separate token-based model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import torch
from torch import Tensor, nn
from torch.nn import functional as F

import g7_s25_contradiction_gate as policies

HIDDEN = 32
FAST_WIDTH = 16
FAST_LR = 0.001
EPISODE_STEPS = 32
ACTION_PAIR = (-0.8, 0.8)


class ConventionalGRU(nn.Module):
    """Conventional trained recurrent baseline structure; constructor alone proves nothing."""
    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        if hidden != HIDDEN:
            raise ValueError("FROZEN_GRU_WIDTH_DRIFT")
        self.history = nn.GRU(4, hidden, batch_first=True)
        self.query = nn.Sequential(nn.Linear(hidden + 3, hidden),
                                   nn.Tanh(), nn.Linear(hidden, 1))

    def encode(self, history: Tensor) -> Tensor:
        if history.ndim != 3 or history.shape[-1] != 4 or history.shape[1] < 1:
            raise ValueError("INVALID_NUMERIC_HISTORY")
        if not torch.isfinite(history).all():
            raise ValueError("NONFINITE_HISTORY")
        _, last = self.history(history)
        return last.squeeze(0)

    def predict(self, history: Tensor, queries: Tensor) -> Tensor:
        self._validate_queries(history, queries)
        latent = self.encode(history)
        summary = latent[:, None, :].expand(-1, queries.shape[1], -1)
        return self.query(torch.cat((summary, queries), dim=-1)).squeeze(-1)

    @staticmethod
    def _validate_queries(history: Tensor, queries: Tensor) -> None:
        if queries.ndim != 3 or queries.shape[-1] != 3 or history.shape[0] != queries.shape[0]:
            raise ValueError("INVALID_NUMERIC_QUERY")
        if not torch.isfinite(queries).all():
            raise ValueError("NONFINITE_QUERY")


class NumericFastWeightTTT(nn.Module):
    """Shared slow encoder and per-episode numerical fast-weight residual.

    All gate arms must restore identical frozen slow weights and initial
    zero fast-state. Only fast state is written during online adaptation.
    """
    def __init__(self, slow_model: ConventionalGRU, *, fast_width: int = FAST_WIDTH,
                 learning_rate: float = FAST_LR):
        super().__init__()
        if fast_width != FAST_WIDTH or learning_rate != FAST_LR:
            raise ValueError("FROZEN_FAST_WEIGHT_CAPACITY_OR_LR_DRIFT")
        self.slow = slow_model
        self.features = nn.Sequential(nn.Linear(HIDDEN + 3, fast_width),
                                      nn.Tanh())
        self.learning_rate = float(learning_rate)

    def new_state(self, batch: int, *, reference: Tensor) -> Tensor:
        if batch < 1:
            raise ValueError("INVALID_BATCH")
        return reference.new_zeros((batch, FAST_WIDTH))

    def prequential_predict(self, history: Tensor, queries: Tensor,
                            state: Tensor) -> tuple[Tensor, Tensor]:
        self.slow._validate_queries(history, queries)
        if state.shape != (queries.shape[0], FAST_WIDTH) or state.device != queries.device:
            raise ValueError("INVALID_FAST_STATE")
        if not torch.isfinite(state).all():
            raise ValueError("NONFINITE_FAST_STATE")
        context = self.slow.encode(history)
        joint = torch.cat((context[:, None].expand(-1, queries.shape[1], -1),
                           queries), dim=-1)
        features = F.normalize(self.features(joint), dim=-1)
        base = self.slow.query(joint).squeeze(-1)
        correction = (features * state[:, None, :]).sum(dim=-1)
        return base + correction, features

    @torch.no_grad()
    def update_after_observation(self, state: Tensor, features: Tensor,
                                 preupdate_prediction: Tensor, targets: Tensor,
                                 update_mask: Tensor) -> Tensor:
        """SGD on 0.5 mean squared pair error. Read happens before update."""
        if features.shape != (state.shape[0], 2, FAST_WIDTH) or \
           preupdate_prediction.shape != targets.shape or targets.shape != (state.shape[0], 2):
            raise ValueError("INVALID_PAIRED_UPDATE_SHAPE")
        if update_mask.shape != (state.shape[0],) or update_mask.dtype != torch.bool:
            raise ValueError("INVALID_UPDATE_MASK")
        if not torch.isfinite(targets).all():
            raise ValueError("NONFINITE_OBSERVED_TARGETS")
        # Compute update gradients ONLY for admitted worlds. Multiplication
        # by an all-false mask after computing every gradient was not sparse
        # and could leak NaNs from inactive rows (NaN * 0 == NaN).
        if update_mask.device != state.device:
            raise ValueError("INVALID_UPDATE_MASK_DEVICE")
        active = torch.nonzero(update_mask, as_tuple=True)[0]
        updated = state.detach().clone()
        if active.numel() == 0:
            return updated
        residual = targets.index_select(0, active) - preupdate_prediction.index_select(0, active)
        selected_features = features.index_select(0, active)
        direction = (residual[:, :, None] * selected_features).mean(dim=1)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1.0)
        updated.index_add_(0, active, self.learning_rate * direction)
        return updated.detach()


@dataclass(frozen=True)
class WorldStep:
    """Caller-owned prequential interface; no simulator truth or future observations."""
    history: Tensor
    query_pair: Tensor
    observed_pair: Tensor


class PrequentialTTTEpisode:
    """Existing preregistered policy controls actual per-world fast-state updates."""
    def __init__(self, network: NumericFastWeightTTT, *, mode: str, seed: int,
                 batch: int, reference: Tensor):
        self.model = network
        self.state = network.new_state(batch, reference=reference)
        self.gates = [policies.PrequentialGate(mode, engineering_seed=seed + 101 * j)
                      for j in range(batch)]
        self.steps = 0
        self.predictions: list[Tensor] = []
        self.update_masks: list[Tensor] = []

    @torch.no_grad()
    def step(self, incoming: WorldStep) -> Tensor:
        if self.steps >= EPISODE_STEPS:
            raise ValueError("EPISODE_ALREADY_COMPLETE")
        queries = incoming.query_pair
        if queries.shape != (len(self.gates), 2, 3) or \
           incoming.observed_pair.shape != (len(self.gates), 2):
            raise ValueError("INVALID_OBSERVED_PAIR_SHAPE")
        if not torch.allclose(queries[:, 0, 0], queries[:, 1, 0]) or \
           not torch.allclose(queries[:, 0, 2], queries[:, 1, 2]) or \
           not torch.allclose(queries[:, :, 1],
                              queries.new_tensor(ACTION_PAIR).expand_as(queries[:, :, 1])):
            raise ValueError("UNMATCHED_ACTION_INTERVENTIONS")
        predicted, features = self.model.prequential_predict(incoming.history,
                                                              queries, self.state)
        selected = []
        for j, gate in enumerate(self.gates):
            decision = gate.observe(
                self.steps,
                predicted_minus=float(predicted[j, 0].item()),
                predicted_plus=float(predicted[j, 1].item()),
                observed_minus=float(incoming.observed_pair[j, 0].item()),
                observed_plus=float(incoming.observed_pair[j, 1].item()),
                primary_residual=float((incoming.observed_pair[j, 1]
                                        - predicted[j, 1]).item()),
            )
            selected.append(decision.update)
        mask = torch.tensor(selected, device=self.state.device, dtype=torch.bool)
        self.state = self.model.update_after_observation(
            self.state, features, predicted, incoming.observed_pair, mask)
        self.predictions.append(predicted.detach().clone())
        self.update_masks.append(mask.clone())
        self.steps += 1
        return predicted

    def close(self) -> dict[str, int]:
        if self.steps != EPISODE_STEPS:
            raise ValueError("EPISODE_INCOMPLETE")
        counts = [gate.close() for gate in self.gates]
        if counts != [counts[0]] * len(counts):
            raise ValueError("UNMATCHED_WORLD_UPDATE_BUDGET")
        return {"steps": self.steps, "worlds": len(counts),
                "updates_per_world": counts[0]}


@dataclass(frozen=True)
class BaselineTrainingReceipt:
    updates: int
    validation_mse: tuple[float, ...]
    converged: bool
    status: str


def fit_conventional_gru(
    model: ConventionalGRU,
    training_batch: Callable[[int], tuple[Tensor, Tensor, Tensor]],
    validation_batch: Callable[[], tuple[Tensor, Tensor, Tensor]],
    *, learning_rate: float = 0.001, max_updates: int = 1600,
) -> BaselineTrainingReceipt:
    """Trainer defined but NOT invoked by engineering CI.

    Frozen validation rule: >=640 steps and 5 checks 64 steps apart with
    <0.1% relative variation. Source identity/split custody remains external.
    """
    if max_updates != 1600 or learning_rate != 0.001:
        raise ValueError("BASELINE_FROZEN_TRAINING_BUDGET_DRIFT")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=0.01)
    validation: list[float] = []
    for step in range(1, max_updates + 1):
        history, queries, targets = training_batch(step)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model.predict(history, queries)
        if output.shape != targets.shape or not torch.isfinite(targets).all():
            raise ValueError("INVALID_BASELINE_TRAINING_BATCH")
        loss = F.mse_loss(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step >= 640 and step % 64 == 0:
            model.eval()
            with torch.no_grad():
                vh, vq, vy = validation_batch()
                vmse = float(F.mse_loss(model.predict(vh, vq), vy).item())
            validation.append(vmse)
            if len(validation) >= 5:
                last = validation[-5:]
                change = (max(last) - min(last)) / max(min(last), 1e-12)
                if change < 0.001:
                    return BaselineTrainingReceipt(step, tuple(validation), True,
                                                  "BASELINE_VALIDATION_PLATEAU")
    return BaselineTrainingReceipt(max_updates, tuple(validation), False,
                                  "BASELINE_NOT_CONVERGED_SCIENCE_INVALID")


class CalibratedRBFKernelRidge:
    """Real-valued kernel control, calibration on train/validation worlds only."""
    GAMMAS = (0.5, 2.0, 6.0)
    RIDGES = (0.01, 0.08)

    def __init__(self) -> None:
        self.mean: Tensor | None = None
        self.scale: Tensor | None = None
        self.train_vectors: Tensor | None = None
        self.weights: Tensor | None = None
        self.gamma: float | None = None
        self.ridge: float | None = None
        self.validation_mse: float | None = None

    @staticmethod
    def numeric_vectors(history: Tensor, queries: Tensor) -> Tensor:
        if history.ndim != 3 or tuple(history.shape[1:]) != (32, 4):
            raise ValueError("KERNEL_REQUIRES_32_OBSERVED_TRANSITIONS")
        if queries.ndim != 3 or queries.shape[0] != history.shape[0] or queries.shape[2] != 3:
            raise ValueError("INVALID_KERNEL_QUERY_SHAPE")
        if not torch.isfinite(history).all() or not torch.isfinite(queries).all():
            raise ValueError("NONFINITE_KERNEL_INPUT")
        return torch.cat((history.flatten(start_dim=1)[:, None].expand(-1, queries.shape[1], -1),
                          queries), dim=-1).reshape(-1, 131)

    @staticmethod
    def _rbf(left: Tensor, right: Tensor, gamma: float) -> Tensor:
        return torch.exp(-gamma * torch.cdist(left, right).square() / left.shape[-1])

    def fit_calibrated(self, train: tuple[Tensor, Tensor, Tensor],
                       validation: tuple[Tensor, Tensor, Tensor]) -> dict[str, float]:
        """Training API; must NOT be called in design-only CI."""
        h, q, y = train
        vh, vq, vy = validation
        x = self.numeric_vectors(h, q).double()
        vx = self.numeric_vectors(vh, vq).double()
        labels = y.reshape(-1, 1).double()
        test_labels = vy.reshape(-1, 1).double()
        if y.shape != q.shape[:2] or vy.shape != vq.shape[:2]:
            raise ValueError("KERNEL_LABEL_SHAPE_MISMATCH")
        if x.shape[0] < 16 or vx.shape[0] < 16:
            raise ValueError("KERNEL_INSUFFICIENT_CALIBRATION_WORLD_EXPOSURES")
        if not torch.isfinite(labels).all() or not torch.isfinite(test_labels).all():
            raise ValueError("NONFINITE_KERNEL_TARGET")
        mean = x.mean(0)
        scale = x.std(0, unbiased=False).clamp_min(1e-4)
        standardized = (x - mean) / scale
        validation_std = (vx - mean) / scale
        eye = torch.eye(x.shape[0], dtype=x.dtype, device=x.device)
        best: tuple[float, float, float, Tensor] | None = None
        for gamma in self.GAMMAS:
            gram = self._rbf(standardized, standardized, gamma)
            probe = self._rbf(validation_std, standardized, gamma)
            for ridge in self.RIDGES:
                weights = torch.linalg.solve(gram + ridge * eye, labels)
                mse = float((probe @ weights - test_labels).square().mean())
                if best is None or mse < best[0]:
                    best = (mse, gamma, ridge, weights.detach().clone())
        assert best is not None
        self.validation_mse, self.gamma, self.ridge, self.weights = best
        self.train_vectors = standardized.detach()
        self.mean = mean.detach()
        self.scale = scale.detach()
        return {"validation_mse": self.validation_mse, "gamma": self.gamma,
                "ridge": self.ridge, "train_query_samples": int(x.shape[0])}

    @torch.no_grad()
    def predict(self, history: Tensor, queries: Tensor) -> Tensor:
        if (self.train_vectors is None or self.weights is None or
                self.mean is None or self.scale is None or self.gamma is None):
            raise ValueError("UNTRAINED_KERNEL_BASELINE")
        vectors = self.numeric_vectors(history, queries).double()
        matrix = self._rbf((vectors - self.mean) / self.scale,
                           self.train_vectors, self.gamma)
        return (matrix @ self.weights).reshape(queries.shape[:2]).to(queries.dtype)
