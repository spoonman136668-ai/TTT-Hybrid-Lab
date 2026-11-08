"""G7-S25 observed-work ledger for *untrained engineering* paired-action episodes.

This module does not train, fit, open frozen seeds, query a simulator, or run
an experiment. It audits an already existing PrequentialTTTEpisode and records
the *actual* world update-mask count and measured process CPU/wall time. Only
explicit analytic lower bounds are reported for operation counts; no profiler
or full end-to-end FLOP equality is implied by those bounds.
"""
from __future__ import annotations

import hashlib
import json
import time

import torch

from g7_s25_numeric_ttt import (
    ACTION_PAIR, EPISODE_STEPS, FAST_WIDTH, PrequentialTTTEpisode, WorldStep,
)
from g7_process_peak_memory import read_peak_memory

MATCHED = frozenset(("periodic", "random", "surprise", "contradiction"))
ARMS = MATCHED | {"full"}
PROBES_PER_WORLD_STEP = 2
# Fast-state read: per probe, F dot state uses W multiplies + W-1 adds.
FAST_READ_MIN_ARITHMETIC_PER_WORLD_STEP = 2 * (2 * FAST_WIDTH - 1)
# Fast-state write: 2*W residual-feature mul + W pair sums +
# W state scale + W state adds. Excludes norm, divide, masks, setup.
FAST_UPDATE_MIN_ARITHMETIC_PER_SELECTED_WORLD = 5 * FAST_WIDTH


def _hash_tensor(h, label: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise ValueError("LEDGER_CPU_NUMERIC_TENSOR_REQUIRED")
    if not torch.isfinite(value).all():
        raise ValueError("LEDGER_NONFINITE_TENSOR")
    data = value.detach().contiguous()
    h.update(label.encode("ascii"))
    h.update(json.dumps({"dtype": str(data.dtype), "shape": list(data.shape)},
                        sort_keys=True, separators=(",", ":")).encode("ascii"))
    h.update(data.numpy().tobytes(order="C"))


def _checkpoint_fingerprint(net: torch.nn.Module) -> tuple[str, int]:
    sha = hashlib.sha256()
    size_bytes = 0
    for name, tensor in net.state_dict().items():
        _hash_tensor(sha, name, tensor)
        size_bytes += tensor.numel() * tensor.element_size()
    return sha.hexdigest(), size_bytes


class EpisodeWorkLedger:
    """Lifecycle-coupled actual-work counts, never scientific comparison."""

    def __init__(self, episode: PrequentialTTTEpisode, *, source_seed: int):
        if not isinstance(episode, PrequentialTTTEpisode):
            raise ValueError("LEDGER_REQUIRES_REAL_NUMERIC_TTT")
        if source_seed not in (3101, 3109, 3119):
            raise ValueError("LEDGER_ENGINEERING_SEED_ONLY")
        if not episode.gates or any(g.mode != episode.gates[0].mode
                                    for g in episode.gates):
            raise ValueError("LEDGER_MISMATCHED_ARM")
        if episode.steps != 0 or episode.update_masks:
            raise ValueError("LEDGER_REQUIRES_NEW_EPISODE")
        self.episode = episode
        self.mode = episode.gates[0].mode
        if self.mode not in ARMS:
            raise ValueError("LEDGER_UNKNOWN_MODE")
        self.batch = len(episode.gates)
        self.source_seed = source_seed
        self._expected_updates = EPISODE_STEPS if self.mode == "full" else 8
        self._input_hash = hashlib.sha256()
        self._step = 0
        self._fast_update_worlds = 0
        self._forward_queries = 0
        self._observed_action_probes = 0
        self._realized_state_transitions = 0
        self._cpu_ns = 0
        self._wall_ns = 0
        self._initial_model_sha, self._model_checkpoint_bytes = (
            _checkpoint_fingerprint(episode.model))
        self._closed = False

    def step(self, world: WorldStep) -> torch.Tensor:
        if self._closed or self._step >= EPISODE_STEPS:
            raise ValueError("LEDGER_ALREADY_COMPLETE")
        if not isinstance(world, WorldStep):
            raise ValueError("LEDGER_INVALID_WORLD_STEP")
        if (tuple(world.query_pair.shape) != (self.batch, 2, 3)
                or tuple(world.observed_pair.shape) != (self.batch, 2)
                or world.history.ndim != 3
                or world.history.shape[0] != self.batch
                or world.history.shape[2] != 4):
            raise ValueError("LEDGER_WORLD_SHAPE")
        if world.query_pair.device.type != "cpu":
            raise ValueError("LEDGER_CPU_ONLY")
        choice = world.realized_action_index
        if (not isinstance(choice, torch.Tensor) or
                choice.shape != (self.batch,) or
                choice.dtype != torch.int64 or choice.device.type != "cpu" or
                not bool(((choice == 0) | (choice == 1)).all())):
            raise ValueError("LEDGER_INVALID_OBSERVABLE_REALIZED_ACTION")
        digest_fields = ("history", "query_pair", "observed_pair",
                         "realized_action_index")
        # Hash exactly the information all policy arms receive. Never hash
        # oracle x/z, physics laws, seed IDs, predictions or training labels.
        for name in digest_fields:
            _hash_tensor(self._input_hash, name, getattr(world, name))
        start_wall = time.perf_counter_ns()
        start_cpu = time.process_time_ns()
        predictions = self.episode.step(world)
        self._cpu_ns += time.process_time_ns() - start_cpu
        self._wall_ns += time.perf_counter_ns() - start_wall
        mask = self.episode.update_masks[-1]
        if mask.shape != (self.batch,) or mask.dtype != torch.bool:
            raise ValueError("LEDGER_INVALID_ACTUAL_UPDATE_MASK")
        if not torch.isfinite(predictions).all():
            raise ValueError("LEDGER_NONFINITE_MODEL_PREDICTION")
        self._fast_update_worlds += int(mask.sum().item())
        self._forward_queries += int(predictions.numel())
        self._observed_action_probes += int(world.observed_pair.numel())
        self._realized_state_transitions += self.batch
        self._step += 1
        return predictions

    def close(self) -> dict:
        if self._closed:
            raise ValueError("LEDGER_ALREADY_CLOSED")
        gates = self.episode.close()
        required = self.batch * self._expected_updates
        if self._fast_update_worlds != required:
            raise ValueError("LEDGER_UNMATCHED_REAL_UPDATES")
        if self._forward_queries != self.batch * 2 * EPISODE_STEPS:
            raise ValueError("LEDGER_FORWARD_EXPOSURE_MISMATCH")
        if self._observed_action_probes != self._forward_queries:
            raise ValueError("LEDGER_UNMATCHED_PROBES")
        if gates["updates_per_world"] != self._expected_updates:
            raise ValueError("LEDGER_GATE_RECEIPT_MISMATCH")
        after_sha, size_after = _checkpoint_fingerprint(self.episode.model)
        if after_sha != self._initial_model_sha or size_after != self._model_checkpoint_bytes:
            raise ValueError("LEDGER_SLOW_MODEL_CHECKPOINT_DRIFT")
        # Different OS metrics stay different. Linux retains the original
        # ru_maxrss semantics. Windows reports native peak working-set bytes
        # only, without inventing an equivalent Linux RSS measurement.
        memory=read_peak_memory()
        self._closed = True
        return {
            "schema": "ttt-hybrid.g7-s25.observed-work-ledger.v1",
            "status": "ENGINEERING_WORK_COUNTS_ONLY_NOT_SCIENTIFIC_PASS",
            "mode": self.mode,
            "development_seed": self.source_seed,
            "steps": self._step,
            "worlds": self.batch,
            "input_stream_sha256": self._input_hash.hexdigest(),
            "model_checkpoint_sha256": self._initial_model_sha,
            "slow_checkpoint_bytes": self._model_checkpoint_bytes,
            "fast_state_bytes": self.episode.state.numel()
                                * self.episode.state.element_size(),
            "realized_transitions": self._realized_state_transitions,
            "paired_intervention_observations": self._observed_action_probes,
            "read_forward_query_predictions": self._forward_queries,
            "real_fast_weight_update_worlds": self._fast_update_worlds,
            "skipped_fast_gradient_worlds": (
                self.batch * EPISODE_STEPS - self._fast_update_worlds),
            "updates_per_world": gates["updates_per_world"],
            "fast_state_read_flops_lower_bound": (
                self.batch * self._step * FAST_READ_MIN_ARITHMETIC_PER_WORLD_STEP),
            "fast_state_update_flops_lower_bound": (
                self._fast_update_worlds * FAST_UPDATE_MIN_ARITHMETIC_PER_SELECTED_WORLD),
            "estimated_flops_are_total_model_flops": False,
            "forward_or_gate_flops_exactly_profiled": False,
            "episode_process_cpu_seconds": round(self._cpu_ns / 1e9, 9),
            "episode_step_wall_seconds": round(self._wall_ns / 1e9, 9),
            "process_peak_rss_bytes": (
                memory.peak_bytes
                if memory.comparable_to_linux_ru_maxrss else None),
            "process_peak_working_set_bytes": (
                memory.peak_bytes
                if memory.metric == "windows_psapi_peak_working_set_bytes"
                else None),
            "process_peak_memory_metric": memory.metric,
            "process_peak_memory_comparable_to_linux_ru_maxrss": (
                memory.comparable_to_linux_ru_maxrss),
            "peak_rss_is_isolated_per_arm": False,
            "whole_process_training_and_validation_cpu_seconds": None,
            "trained_baselines_qualified": False,
            "whole_process_matched_compute_qualified": False,
            "gpu": "NOT_USED",
            "scientific_first_use": "NOT_RUN",
        }


def compare_equal_budget_receipts(receipts: list[dict]) -> dict:
    """Strong negative admission; never call this a scientific measurement."""
    names = {r.get("mode") for r in receipts}
    if names != MATCHED or len(receipts) != len(MATCHED):
        raise ValueError("LEDGER_MISSING_EQUAL_BUDGET_CONTROL")
    invariants = (
        "steps", "worlds", "input_stream_sha256", "model_checkpoint_sha256",
        "paired_intervention_observations", "read_forward_query_predictions",
        "realized_transitions", "real_fast_weight_update_worlds",
        "fast_state_read_flops_lower_bound",
        "fast_state_update_flops_lower_bound", "slow_checkpoint_bytes",
        "fast_state_bytes", "skipped_fast_gradient_worlds",
    )
    reference = receipts[0]
    for candidate in receipts:
        if any(candidate.get(k) != reference.get(k) for k in invariants):
            raise ValueError("LEDGER_UNMATCHED_INFORMATION_OR_UPDATE_WORK")
        if candidate.get("status") != (
                "ENGINEERING_WORK_COUNTS_ONLY_NOT_SCIENTIFIC_PASS"):
            raise ValueError("LEDGER_UNQUALIFIED_RECEIPT")
        if candidate.get("trained_baselines_qualified") is not False:
            raise ValueError("LEDGER_PREMATURE_BASELINE_ACCEPTANCE")
    return {
        "status": "PASS_EQUAL_INFORMATION_AND_FAST_UPDATE_WORK_ONLY",
        "arms": sorted(names),
        "shared_stream_sha256": reference["input_stream_sha256"],
        "worlds": reference["worlds"],
        "paired_observations_each": reference["paired_intervention_observations"],
        "real_fast_updates_each": reference["real_fast_weight_update_worlds"],
        "full_model_training_and_resource_parity": "UNVERIFIED",
        "scientific_comparison": "NOT_RUN",
    }
