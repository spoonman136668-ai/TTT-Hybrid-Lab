"""G7-S24 engineering-only numeric JEPA components: no training or evaluation.

Two *independent* encoders learn intra-modality good-view invariances. The
loss does not compare sensor and event modalities to one another. A train-only
alignment/Procrustes stage, action prediction and frozen science are separate.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
import g7_s24_views as views

WIDTH = 16


@dataclass(frozen=True)
class ModalityPairs:
    sensor_a: torch.Tensor
    sensor_b: torch.Tensor
    graph_a_nodes: torch.Tensor
    graph_a_edges: torch.Tensor
    graph_b_nodes: torch.Tensor
    graph_b_edges: torch.Tensor


def development_modality_pairs(seed: int, batch: int, regime: str) -> ModalityPairs:
    """No world or target labels cross the numerical representation boundary.

    One underlying initial physical world is cloned four times. All four
    rollouts use independent action schedules and sensor noise. This is NOT
    a scientific or heldout-data API; seeds are explicitly development-only.
    """
    if seed not in (5101, 5113, 5129):
        raise ValueError("S24_PHASE1_DEVELOPMENT_SEED_ONLY")
    family, ood = views._regime(regime)
    old, new, x, z, _ = views._seed_initial(seed, batch, ood)

    def rollout(salt: int) -> torch.Tensor:
        gen = torch.Generator(device="cpu").manual_seed(views._view_seed(seed, salt))
        return views._trace(old, new, x.clone(), z.clone(), gen, family)

    sensor_a = rollout(101)
    sensor_b = rollout(103)
    graph_a_nodes, graph_a_edges = views._graph(rollout(211), views._view_seed(seed, 307))
    graph_b_nodes, graph_b_edges = views._graph(rollout(223), views._view_seed(seed, 331))
    if torch.equal(sensor_a[:, :, 1:3], sensor_b[:, :, 1:3]):
        raise RuntimeError("SENSOR_ACTIONS_NOT_INDEPENDENT")
    if torch.equal(graph_a_nodes, graph_b_nodes):
        raise RuntimeError("GRAPH_VIEWS_NOT_INDEPENDENT")
    return ModalityPairs(sensor_a, sensor_b, graph_a_nodes, graph_a_edges,
                         graph_b_nodes, graph_b_edges)


class SensorGRU(nn.Module):
    """Numerical, untokenized sequence encoder, width frozen to 16."""
    def __init__(self):
        super().__init__()
        self.recurrent = nn.GRU(input_size=4, hidden_size=WIDTH, batch_first=True)
        self.projection = nn.Linear(WIDTH, WIDTH)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or tuple(history.shape[1:]) != (32, 4):
            raise ValueError("S24_INVALID_SENSOR_SHAPE")
        _, h = self.recurrent(history)
        return self.projection(h[-1])


class EventMessagePassing(nn.Module):
    """Two-round numeric graph encoder with permutation-invariant readout."""
    def __init__(self):
        super().__init__()
        self.node_input = nn.Linear(6, WIDTH)
        self.rounds = nn.ModuleList([nn.Linear(2 * WIDTH, WIDTH) for _ in range(2)])
        self.projection = nn.Linear(WIDTH, WIDTH)

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        if nodes.ndim != 3 or tuple(nodes.shape[1:]) != (32, 6) or \
           tuple(edges.shape) != (2, 62) or edges.dtype != torch.int64:
            raise ValueError("S24_INVALID_GRAPH_SHAPE")
        if bool((edges < 0).any() or (edges >= 32).any()):
            raise ValueError("S24_INVALID_GRAPH_EDGE_INDEX")
        h = F.gelu(self.node_input(nodes))
        src, dst = edges.unbind(0)
        for layer in self.rounds:
            messages = h.index_select(1, src)
            agg = torch.zeros_like(h).index_add(1, dst, messages)
            degree = torch.bincount(dst, minlength=32).to(h.dtype).view(1, 32, 1).clamp_min(1)
            h = F.gelu(layer(torch.cat((h, agg / degree), dim=-1)))
        return self.projection(h.mean(dim=1))


def within_modality_jepa(z_a: torch.Tensor, z_b: torch.Tensor) -> dict[str, torch.Tensor]:
    """Centered/whitened aligned views plus frozen variance/covariance penalties.

    This is a *within*-encoder positive-pair loss. Cross-modal matching is
    prohibited during fitting and must later use train-only rotation fitting.
    """
    if z_a.ndim != 2 or z_a.shape != z_b.shape or z_a.shape[1] != WIDTH or z_a.shape[0] < 2:
        raise ValueError("S24_INVALID_LATENT_SHAPE")
    mean_a, mean_b = z_a.mean(dim=0), z_b.mean(dim=0)
    ca, cb = z_a - mean_a, z_b - mean_b
    va, vb = ca.square().mean(dim=0), cb.square().mean(dim=0)
    wa = ca / torch.sqrt(va + 1e-6)
    wb = cb / torch.sqrt(vb + 1e-6)
    alignment = (wa - wb).square().mean()
    variance = (F.relu(0.05 - va).mean() + F.relu(0.05 - vb).mean()) * 0.5
    cov_a = wa.T @ wa / z_a.shape[0]
    cov_b = wb.T @ wb / z_a.shape[0]
    mask = 1 - torch.eye(WIDTH, dtype=wa.dtype, device=wa.device)
    covariance = (((cov_a * mask).square() + (cov_b * mask).square()).sum()
                  / (2 * WIDTH * (WIDTH - 1)))
    total = alignment + variance + 0.1 * covariance
    return {"loss": total, "alignment": alignment, "variance": variance,
            "covariance": covariance}


def independent_losses(sensor: SensorGRU, graph: EventMessagePassing,
                       inputs: ModalityPairs) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Two separate objectives; NEVER produce a cross-encoder gradient."""
    sensor_loss = within_modality_jepa(sensor(inputs.sensor_a), sensor(inputs.sensor_b))
    graph_loss = within_modality_jepa(
        graph(inputs.graph_a_nodes, inputs.graph_a_edges),
        graph(inputs.graph_b_nodes, inputs.graph_b_edges))
    return sensor_loss, graph_loss
