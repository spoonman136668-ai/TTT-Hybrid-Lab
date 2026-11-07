"""S25 causal-prefix numerical supervision contract for GRU, KRR and TTT.

This is a source-only data interface; its exposed generator accepts solely
the 3 DEVELOPMENT seeds, and does not call model.fit, optimizer.step, evaluate
heldout seeds or run S25 scientific first-use. Independent science custody must
supply a separately approved generator; this module cannot authorize it.

All model input streams consist only of the same 32-slot right-padded numeric
past, the current pair of action queries, and the pair of observed numeric
targets. The targets are NEVER part of past or queried input features.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import torch
from torch import Tensor

from g7_s25_numeric_ttt import (
    ConventionalGRU, CalibratedRBFKernelRidge, WorldStep, causal_history32,
)
from g7_s25_paired_interventions import (
    ENGINEERING_SEEDS, MAX_STEPS, PairedS20World,
)

@dataclass(frozen=True)
class CausalSupervision:
    # Only these three tensors may be passed to a trained predictor.
    history: Tensor
    queries: Tensor
    targets: Tensor

    def model_inputs(self) -> tuple[Tensor, Tensor]:
        return self.history, self.queries

    def training_triplet(self) -> tuple[Tensor, Tensor, Tensor]:
        return self.history, self.queries, self.targets


def causal_supervision(step: WorldStep) -> CausalSupervision:
    """Transform one already-revealed paired probe without touching the oracle."""
    if not isinstance(step, WorldStep):
        raise ValueError("S25_SUPERVISION_REQUIRES_WORLDSTEP")
    history = causal_history32(step.history)
    queries = step.query_pair
    targets = step.observed_pair
    if (queries.shape != (history.shape[0], 2, 3)
            or targets.shape != (history.shape[0], 2)):
        raise ValueError("S25_SUPERVISION_PAIRED_SHAPE")
    if (queries.device != history.device
            or targets.device != history.device
            or not torch.isfinite(queries).all()
            or not torch.isfinite(targets).all()):
        raise ValueError("S25_SUPERVISION_NONFINITE_OR_DEVICE_DRIFT")
    if not torch.equal(queries[:, 0, 0], queries[:, 1, 0]) or not torch.equal(
            queries[:, 0, 2], queries[:, 1, 2]):
        raise ValueError("S25_SUPERVISION_UNMATCHED_INTERVENTION_STATE")
    expected = queries.new_tensor((-0.8, 0.8)).expand(
        history.shape[0], -1)
    if not torch.allclose(queries[:, :, 1], expected, atol=0, rtol=0):
        raise ValueError("S25_SUPERVISION_ACTION_DRIFT")
    return CausalSupervision(history.clone(), queries.clone(),
                             targets.clone())


def engineering_only_prefixes(
    *, seed: int, batch: int, regime: str = "source_stable"
) -> tuple[CausalSupervision, ...]:
    """Capture all 32 causal prefixes; never consumes any frozen science seed."""
    if seed not in ENGINEERING_SEEDS:
        raise ValueError("S25_ENGINEERING_SEEDS_ONLY")
    if regime != "source_stable":
        raise ValueError("S25_DEV_SOURCE_ONLY_BASELINE_INPUTS")
    world = PairedS20World(seed, batch, regime)
    snapshots = []
    for _ in range(MAX_STEPS):
        snapshots.append(causal_supervision(world.emit()))
    receipt = world.close()
    if (len(snapshots) != MAX_STEPS or
            receipt["paired_observations"] != 2 * MAX_STEPS * batch):
        raise RuntimeError("S25_ENGINEERING_SAMPLE_COMPLETENESS")
    return tuple(snapshots)


def source_paired_baseline_inputs(
    snapshots: tuple[CausalSupervision, ...]
) -> tuple[tuple[Tensor, Tensor, Tensor], ...]:
    """Identical inputs for conventional GRU and kernel; no model fitting."""
    if (len(snapshots) != MAX_STEPS or
            any(not isinstance(v, CausalSupervision) for v in snapshots)):
        raise ValueError("S25_INVALID_CAUSAL_SOURCE_STREAM")
    return tuple(row.training_triplet() for row in snapshots)


def stream_fingerprint(
    rows: tuple[CausalSupervision, ...], *, include_targets: bool
) -> str:
    """Stable CPU-only stream digest of the exact feature/label exposures."""
    if not rows:
        raise ValueError("S25_EMPTY_INPUT_STREAM")
    sha = hashlib.sha256()
    for step, row in enumerate(rows):
        sha.update(step.to_bytes(2, "big"))
        for name in ("history", "queries", "targets"):
            if name == "targets" and not include_targets:
                continue
            value = getattr(row, name)
            if value.device.type != "cpu" or not torch.isfinite(value).all():
                raise ValueError("S25_STREAM_REQUIRES_FINITE_CPU_DATA")
            data = value.detach().contiguous()
            sha.update(name.encode("ascii"))
            sha.update(json.dumps((list(data.shape), str(data.dtype)),
                                   separators=(",", ":")).encode("ascii"))
            sha.update(data.numpy().tobytes())
    return sha.hexdigest()


def admit_cross_architecture_sample(sample: CausalSupervision) -> dict:
    """Shape admission only. Never fits or qualifies either baseline."""
    history, query, target = sample.training_triplet()
    if history.ndim != 3 or tuple(history.shape[1:]) != (32, 4):
        raise ValueError("S25_INVALID_TRAINING_HISTORY")
    if query.shape != (history.shape[0], 2, 3) or target.shape != (
            history.shape[0], 2):
        raise ValueError("S25_MISMATCHED_BASELINE_TARGETS")
    # Reuse actual KRR feature contract, not an invented proxy.
    features = CalibratedRBFKernelRidge.numeric_vectors(history, query)
    if features.shape != (history.shape[0] * 2, 131):
        raise ValueError("S25_KERNEL_BASELINE_FEATURE_SHAPE_DRIFT")
    # GRU expects the same 32 numerical frames; its learnable weights are
    # never constructed or trained by this function.
    return {
        "sample_worlds": history.shape[0],
        "numeric_history_features": history.shape[2],
        "numeric_history_steps": history.shape[1],
        "query_actions_per_world": query.shape[1],
        "kernel_flat_features": features.shape[1],
        "same_observed_source_for_gru_and_kernel": True,
        "fitted_baselines": False,
        "scientific_first_use": "NOT_RUN",
    }
