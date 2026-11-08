"""C2 inactive transactional *candidate* checkpoint restore, not model promotion.

P2's load_checkpoint mutates the supplied model and optimizer before RNG
restoration finishes. Never point it at a retained champion. This adapter
first verifies the source/lineage, constructs separate trainable objects,
restores only into those objects, and returns nothing on failure. Ambient RNG
is returned to its pre-call state on success AND failure.

The model/optimizer factory is trusted project code, not a sandbox. This is
not a substitute for issuer authentication, accepted-ref authority, or actual
scientific evaluation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import random
import re
from pathlib import Path
from typing import Callable, Mapping

import torch

from native_training_checkpoint import load_checkpoint, validate_chain

HEX64 = re.compile(r"^[0-9a-f]{64}$")


class CandidateRestoreRejected(ValueError):
    pass


@dataclass(frozen=True)
class PreparedCandidate:
    """Unpromoted, isolated checkpoint state; does not activate an owner loop."""
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    native_state: dict
    sampler: dict
    checkpoint_sha256: str
    steps: int
    consumed_examples: int
    restored_rng: dict
    status: str = "ISOLATED_CANDIDATE_NOT_ACCEPTED"


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise CandidateRestoreRejected(reason)


def _rng_snapshot() -> dict:
    try:
        import numpy as np
    except ImportError:
        numpy_state = None
    else:
        numpy_state = deepcopy(np.random.get_state())
    return {
        "python": deepcopy(random.getstate()),
        "torch_cpu": torch.get_rng_state().clone(),
        "cuda": [x.clone() for x in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_initialized() else None,
        "numpy": numpy_state,
    }


def _rng_restore(state: dict) -> None:
    # Restoring global state is required even when a candidate fails validation.
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    if state["numpy"] is not None:
        import numpy as np
        np.random.set_state(state["numpy"])


def _assert_isolated(model, optimizer, protected_model=None,
                     protected_optimizer=None) -> None:
    _require(isinstance(model, torch.nn.Module)
             and isinstance(optimizer, torch.optim.Optimizer),
             "CANDIDATE_FACTORY_TYPES")
    _require(model is not protected_model and optimizer is not protected_optimizer,
             "CANDIDATE_IS_PROTECTED_INSTANCE")
    model_params = list(model.parameters())
    _require(bool(model_params)
             and len({id(p) for p in model_params}) == len(model_params),
             "CANDIDATE_MODEL_PARAMETERS_INVALID")
    own = {id(p) for p in model_params}
    optimizer_params = [p for g in optimizer.param_groups for p in g["params"]]
    _require(bool(optimizer_params)
             and all(id(p) in own for p in optimizer_params)
             and len({id(p) for p in optimizer_params}) == len(optimizer_params),
             "CANDIDATE_OPTIMIZER_REFERENCES_OTHER_MODEL")
    if protected_model is not None:
        _require(isinstance(protected_model, torch.nn.Module),
                 "PROTECTED_MODEL_INVALID")
        protected_params = list(protected_model.parameters())
        protected_ids = {id(p) for p in protected_params}
        _require(own.isdisjoint(protected_ids), "CANDIDATE_SHARED_PARAMETERS")
        # A factory may create distinct Parameter wrappers around shared
        # storage. Reject such aliases as well as identical objects.
        protected_storages = {p.untyped_storage().data_ptr()
                              for p in protected_params if p.numel()}
        protected_storages.update(b.untyped_storage().data_ptr()
                                  for b in protected_model.buffers() if b.numel())
        all_candidate = list(model.parameters()) + list(model.buffers())
        _require(all(p.untyped_storage().data_ptr() not in protected_storages
                     for p in all_candidate if p.numel()),
                 "CANDIDATE_SHARED_TENSOR_STORAGE")


def prepare_isolated_candidate(
    directory: str | Path, *, tip_sha256: str, identity: Mapping,
    make_fresh_pair: Callable[[], tuple], expected_position: int,
    minimum_steps: int = 1, minimum_consumed_examples: int = 1,
    expected_parent_sha256: str | None = None,
    protected_model: torch.nn.Module | None = None,
    protected_optimizer: torch.optim.Optimizer | None = None,
) -> PreparedCandidate:
    """Prepare, but never activate or promote, a prior trained checkpoint.

    Only a new candidate object receives the potentially partial state load.
    The caller's model/optimizer and ambient torch/Python/NumPy RNG remain
    unchanged. Returned restored_rng is an inactive saved snapshot; applying
    it to a live owner requires a separate authorized continuation decision.
    """
    _require(type(tip_sha256) is str and HEX64.fullmatch(tip_sha256) is not None,
             "CANDIDATE_TIP_NOT_IMMUTABLE")
    _require(type(expected_position) is int and expected_position >= 1
             and type(minimum_steps) is int and minimum_steps >= 1
             and type(minimum_consumed_examples) is int
             and minimum_consumed_examples >= 1,
             "CANDIDATE_RESTART_OR_CURSOR_NOT_ALLOWED")
    if expected_parent_sha256 is not None:
        _require(type(expected_parent_sha256) is str
                 and HEX64.fullmatch(expected_parent_sha256) is not None,
                 "CANDIDATE_PARENT_ID_INVALID")
    manifest = validate_chain(directory, tip_sha256, identity)
    _require(manifest["steps"] >= minimum_steps
             and manifest["consumed_examples"] >= minimum_consumed_examples,
             "CANDIDATE_RESETS_TRAINING_PROGRESS")
    if expected_parent_sha256 is not None:
        _require(manifest["parent_manifest_sha256"] == expected_parent_sha256,
                 "CANDIDATE_PARENT_MISMATCH")

    before = _rng_snapshot()
    try:
        candidate = make_fresh_pair()
        _require(type(candidate) is tuple and len(candidate) == 2,
                 "CANDIDATE_FACTORY_CONTRACT")
        model, optimizer = candidate
        _assert_isolated(model, optimizer, protected_model, protected_optimizer)
        native, cursor = load_checkpoint(
            directory, tip_sha256, model=model, optimizer=optimizer,
            identity=identity)
        _require(type(cursor) is dict
                 and type(cursor.get("position")) is int
                 and cursor["position"] == expected_position
                 and manifest["steps"] == expected_position,
                 "CANDIDATE_CURSOR_MISMATCH")
        _assert_isolated(model, optimizer, protected_model, protected_optimizer)
        saved_rng = _rng_snapshot()
        return PreparedCandidate(
            model=model, optimizer=optimizer, native_state=native,
            sampler=cursor, checkpoint_sha256=tip_sha256,
            steps=manifest["steps"],
            consumed_examples=manifest["consumed_examples"],
            restored_rng=saved_rng)
    finally:
        _rng_restore(before)
