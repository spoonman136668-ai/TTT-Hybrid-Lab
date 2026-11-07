"""G7-S24 phase-0 pairing adapter; no fitting, model execution or science run.

A world identity is its initial (x,z) and BOTH frozen coefficient vectors.
The two sensors independently sample interventions and measurement noise.
The generator's private state never appears in encoder-facing tensors.
The physics step invokes the pinned G7-S20 rate function verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

import g7_s20_hidden_multicycle as s20

STEPS = 32


@dataclass(frozen=True)
class PairedViews:
    sensor_sequence: torch.Tensor   # (B,32,4): y,u,dt,next_y
    event_nodes: torch.Tensor       # (B,32,6): y,u,dt,next_y,delta_y,t
    event_edges: torch.Tensor       # (2,62): bidirectional time adjacency


def _regime(regime: str) -> tuple[str, bool]:
    choices = {
        "source_train": ("source", False),
        "sine_train": ("sine", False),
        "rational_train": ("rational", False),
        "source_ood": ("source", True),
        "sine_ood": ("sine", True),
        "rational_ood": ("rational", True),
    }
    if regime not in choices:
        raise ValueError("UNREGISTERED_S24_REGIME")
    return choices[regime]


def _seed_initial(seed: int, batch: int, ood: bool):
    if batch < 1 or seed < 0:
        raise ValueError("INVALID_WORLD_SEED_OR_BATCH")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    old = s20.coefficients(gen, batch, ood)
    new = s20.coefficients(gen, batch, ood)
    x = -1 + 2 * torch.rand((batch, 1), generator=gen)
    z = -.8 + 1.6 * torch.rand((batch, 1), generator=gen)
    return old, new, x, z, gen


def _trace(old, new, x, z, gen, family: str) -> torch.Tensor:
    """Same order and numerical operations as S20.partial_world."""
    observation = x + .25 * z + .002 * torch.randn((x.shape[0], 1), generator=gen)
    frames = []
    for step in range(STEPS):
        u = -1 + 2 * torch.rand((x.shape[0], 1), generator=gen)
        dt = .05 + .11 * torch.rand((x.shape[0], 1), generator=gen)
        hidden = new if step >= 20 else old
        dx = s20.rate(x, u, hidden, family) + .35 * z
        dz = -.85 * z + .25 * u + .15 * x
        xn = x + dt * dx
        zn = z + dt * dz
        observed_next = xn + .25 * zn + .002 * torch.randn(
            (x.shape[0], 1), generator=gen
        )
        frames.append(torch.cat((observation, u, dt, observed_next), dim=1))
        x, z, observation = xn, zn, observed_next
    return torch.stack(frames, dim=1)


def reference_equivalent_trace(seed: int, batch: int, regime: str) -> bool:
    """Engineering control: reproduction of S20 history using its original RNG."""
    family, ood = _regime(regime)
    old, new, x, z, gen = _seed_initial(seed, batch, ood)
    regenerated = _trace(old, new, x, z, gen, family)
    original, _, _ = s20.partial_world(seed, batch, regime)
    return bool(torch.equal(regenerated, original))


def _graph(trace: torch.Tensor, permutation_seed: int):
    if trace.ndim != 3 or tuple(trace.shape[1:]) != (STEPS, 4):
        raise ValueError("INVALID_EVENT_TRACE_SHAPE")
    t = torch.arange(STEPS, dtype=trace.dtype).view(1, STEPS, 1)
    t = t.expand(trace.shape[0], -1, -1) / (STEPS - 1)
    nodes = torch.cat((trace, trace[:, :, 3:4] - trace[:, :, 0:1], t), dim=-1)
    g = torch.Generator(device="cpu").manual_seed(permutation_seed)
    perm = torch.randperm(STEPS, generator=g)
    inverse = torch.argsort(perm)
    chronological = torch.arange(STEPS - 1, dtype=torch.long)
    edges = torch.stack((
        torch.cat((inverse[chronological], inverse[chronological + 1])),
        torch.cat((inverse[chronological + 1], inverse[chronological])),
    ))
    return nodes[:, perm, :], edges


def _view_seed(seed: int, salt: int) -> int:
    return seed * 1009 + salt


def independent_views(seed: int, batch: int, regime: str) -> PairedViews:
    """Same initial physical world, different sensor noise and action history.

    Pair IDs, true states, physical coefficients and target outcomes are NEVER
    returned; the modality adapters have numeric tensors only.
    """
    family, ood = _regime(regime)
    old, new, x, z, _ = _seed_initial(seed, batch, ood)
    sensor_gen = torch.Generator(device="cpu").manual_seed(_view_seed(seed, 101))
    event_gen = torch.Generator(device="cpu").manual_seed(_view_seed(seed, 211))
    sensor = _trace(old, new, x.clone(), z.clone(), sensor_gen, family)
    events = _trace(old, new, x.clone(), z.clone(), event_gen, family)
    nodes, edges = _graph(events, _view_seed(seed, 307))
    if torch.equal(sensor[:, :, 1:3], events[:, :, 1:3]):
        raise RuntimeError("INDEPENDENT_ACTION_SCHEDULE_COLLISION")
    if not bool(torch.isfinite(sensor).all() and torch.isfinite(nodes).all()):
        raise RuntimeError("NONFINITE_OBSERVATION")
    return PairedViews(sensor, nodes, edges)


def same_trace_bad_views(seed: int, batch: int, regime: str) -> PairedViews:
    """S24 matched negative: both modalities originate from ONE trajectory."""
    family, ood = _regime(regime)
    old, new, x, z, _ = _seed_initial(seed, batch, ood)
    gen = torch.Generator(device="cpu").manual_seed(_view_seed(seed, 101))
    sensor = _trace(old, new, x, z, gen, family)
    nodes, edges = _graph(sensor, _view_seed(seed, 307))
    return PairedViews(sensor, nodes, edges)
