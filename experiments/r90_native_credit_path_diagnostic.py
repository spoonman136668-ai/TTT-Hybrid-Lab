"""R90: TEST-only native causal-credit gradient and tiny-fit diagnostic.

This is NOT an independent scientific experiment. It may only inspect R89's
previously defined TEST source; it cannot admit any R89 FIT/DEV/OOD or historical
sealed cohort. R89 negative adjudication remains intact and immutable.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import r85_delayed_composition as r85
from experiments import r89_native_event_access as r89

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/r90_native_credit_path_freeze.json"
FREEZE_BLOB = "577f7b99670cbec335c91600ba282baf2e6afd77"
PARENT = "3b75351cc09979a505c7c1ea5e95f779afbb22c1"
BRANCH = "research/r90-native-credit-path-diagnostic-20261115"
SEED = 1000211
STEPS = 64
ARMS = ("HARD_ROUTER", "NO_DEPTH", "SOFT_ROUTER")
SCHEMA = "ttt-hybrid.r90-native-credit-path-diagnostic-result.v1"


def frozen() -> dict:
    spec = json.loads(FREEZE.read_text(encoding="utf-8"))
    if r85.git_blob(FREEZE.read_bytes()) != FREEZE_BLOB:
        raise RuntimeError("R90_PROSPECTIVE_DIAGNOSTIC_FREEZE_DRIFT")
    r89.frozen()
    if (spec["schema"] != "ttt-hybrid.r90-native-credit-path-diagnostic-freeze.v1"
        or spec["stage"] != "FROZEN_BEFORE_R90_DIAGNOSTIC_IMPLEMENTATION_OR_RUN"
        or spec["parent"]["R89_candidate_sha"] != PARENT
        or spec["parent"]["R89_freeze_blob"] != r89.FREEZE_BLOB
        or spec["scope"] != "ENGINEERING_DEV_DIAGNOSTIC_ONLY"
        or spec["source"]["only"] != "R89_EXISTING_TEST_WORLD_GENERATOR_25319991"
        or spec["source"]["indexes"] != [0, 1]
        or not spec["source"]["no_original_frozen_fit_validation_dev_ood"]
        or not spec["source"]["no_new_salt_or_protected_first_use"]
        or not spec["source"]["no_evaluator_only_law_or_oracle_labels"]
        or not spec["source"]["no_prior_trained_weights"]
        or spec["design"]["native_reference_widths"] != [32, 1024]
        or spec["design"]["diagnostic_arms"] != list(ARMS)
        or spec["design"]["training_steps_width32"] != STEPS
        or spec["design"]["training_steps_width1024"] != 0
        or spec["design"]["learning_rate"] != .001
        or not spec["authority"]["dev_cpu_test_only"]
        or any(v for k, v in spec["authority"].items() if k != "dev_cpu_test_only")):
        raise PermissionError("R90_SOURCE_OR_AUTHORITY_DRIFT")
    return spec


def test_only_factual_examples() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two separately realized actions; no alternate-action return or hidden law."""
    frozen()
    xs, raw, rewards = [], [], []
    for index, action in ((0, r85.CALIBRATE), (1, r85.EXPLOIT_POS)):
        world = r89._test_world(index)
        view, selected_return = r89._view(world, action)
        xs.append(r89._slots(view))
        raw.append(view)
        rewards.append(selected_return)
    return (torch.from_numpy(np.stack(xs).astype(np.float32)),
            torch.from_numpy(np.stack(raw).astype(np.float32)),
            torch.tensor(rewards, dtype=torch.float32))


def _native(width: int) -> nn.Module:
    if width not in (32, 1024):
        raise ValueError("R90_UNFROZEN_NATIVE_WIDTH")
    torch.manual_seed(SEED)
    model = r89.r74.make_model("HybridPilot_FULL", width)
    old = model.ttt
    replacement = r89.r75.TriangularFastWeightMemory(width, learning_rate=old.eta)
    replacement.load_state_dict(old.state_dict(), strict=True)
    model.ttt = replacement
    return model


class SoftDiagnosticRouter(nn.Module):
    """Engineering ablation: replace sparse hard selection by dense soft routing.

    Parameter tensors are cloned from the native router. This is not a
    promoted architecture and is not a matched-compute comparison.
    """
    def __init__(self, source: nn.Module):
        super().__init__()
        self.norm = copy.deepcopy(source.norm)
        self.router = copy.deepcopy(source.router)
        self.ffn = copy.deepcopy(source.ffn)
        self.last_utilization = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        probabilities = torch.sigmoid(self.router(z))
        self.last_utilization = float(probabilities.mean().detach())
        return x + self.ffn(z) * probabilities


def _arm(base: nn.Module, name: str) -> nn.Module:
    if name not in ARMS:
        raise ValueError("R90_UNFROZEN_DIAGNOSTIC_ARM")
    model = copy.deepcopy(base)
    if name == "NO_DEPTH":
        model.depth = None
    elif name == "SOFT_ROUTER":
        model.depth = SoftDiagnosticRouter(model.depth)
    return model


def _group(name: str) -> str:
    return name.split(".", 1)[0] if not name.startswith("depth.") else (
        "depth.router" if name.startswith("depth.router.") else
        "depth.ffn" if name.startswith("depth.ffn.") else "depth.norm")


def gradient_receipt(model: nn.Module, x: torch.Tensor,
                     target: torch.Tensor) -> dict:
    model.zero_grad(set_to_none=True)
    pred = model(x)[:, -1, 0]
    loss = F.mse_loss(pred, target)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("R90_NONFINITE_LOSS")
    loss.backward()
    groups: dict[str, dict] = {}
    for name, param in model.named_parameters():
        group = _group(name)
        entry = groups.setdefault(group, {"tensors": 0, "missing": 0,
                                          "zero": 0, "grad_l2": 0.0})
        entry["tensors"] += 1
        if param.grad is None:
            entry["missing"] += 1
            continue
        if not bool(torch.isfinite(param.grad).all()):
            raise RuntimeError("R90_NONFINITE_GRADIENT:" + name)
        norm = float(torch.linalg.vector_norm(param.grad.detach().float()))
        if norm == 0.0:
            entry["zero"] += 1
        entry["grad_l2"] += norm * norm
    for entry in groups.values():
        entry["grad_l2"] = math.sqrt(entry["grad_l2"])
    return {"loss": float(loss.detach()),
            "parameters": sum(p.numel() for p in model.parameters()),
            "router_utilization": model.depth.last_utilization if model.depth else None,
            "gradients": groups}


def _probe(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
           updates: int) -> dict:
    prior = gradient_receipt(model, x, y)
    # No checkpoint is ever persisted. Width1024 is a first-derivative-only
    # measurement, not more training on the same spent/independent science.
    if updates:
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001,
                                      weight_decay=.0001)
        for _ in range(updates):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x)[:, -1, 0], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
    with torch.no_grad():
        final = F.mse_loss(model(x)[:, -1, 0], y).item()
    return {"initial": prior, "updates": updates,
            "final_mse": float(final),
            "final_over_initial": float(final / max(prior["loss"], 1.e-12))}


def _reader_probe(raw: torch.Tensor, y: torch.Tensor) -> dict:
    reader = r89.NativeLearnedRoleEventAccess(_native(32))
    reader.zero_grad(set_to_none=True)
    loss = F.mse_loss(reader(raw)[:, -1, 0], y)
    loss.backward()
    named = dict(reader.named_parameters())
    chosen = ("query", "role_strength_raw", "key.0.weight", "key.2.weight",
              "native.stem.weight", "native.head.weight")
    out = {}
    for n in chosen:
        grad = named[n].grad
        if grad is not None and not bool(torch.isfinite(grad).all()):
            raise RuntimeError("R90_READER_NONFINITE:" + n)
        out[n] = None if grad is None else float(torch.linalg.vector_norm(grad))
    return {"loss": float(loss.detach()), "gradient_norms": out}


def diagnostic() -> dict:
    frozen()
    torch.set_num_threads(1)
    started = time.perf_counter()
    x, raw, target = test_only_factual_examples()
    result = {"schema": SCHEMA, "freeze_blob": FREEZE_BLOB,
              "parent_R89_candidate": PARENT,
              "branch": BRANCH,
              "scope": "ENGINEERING_TEST_ONLY_NO_INDEPENDENT_SCIENCE",
              "only_existing_R89_TEST_world_indexes": [0, 1],
              "device": "cpu", "platform": platform.platform(),
              "torch_version": str(torch.__version__),
              "source_reference_model": "R74 original and R75 triangular TTT",
              "diagnostics": {}}
    for width in (32, 1024):
        base = _native(width)
        if width == 1024 and sum(p.numel() for p in base.parameters()) != 16818178:
            raise RuntimeError("R90_MAX_WIDTH_NATIVE_PARAMETER_DRIFT")
        width_receipts = {}
        common_initial = dict(base.named_parameters())
        for arm in ARMS:
            model = _arm(base, arm)
            params = dict(model.named_parameters())
            if any(not torch.equal(p, params[n]) for n, p in common_initial.items()
                   if n in params):
                raise RuntimeError("R90_ABLATION_PARAMETER_INITIALIZATION_DRIFT")
            width_receipts[arm] = _probe(model, x, target,
                                         STEPS if width == 32 else 0)
            del model
        result["diagnostics"][str(width)] = width_receipts
        del base
    result["public_reader_width32"] = _reader_probe(raw, target)
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    result["status"] = "R90_ENGINEERING_DIAGNOSTIC_OBSERVED_NOT_SCIENCE"
    # A reproducible receipt is placed where the existing G4 full-regression
    # workflow already collects JSONs; no new job or workflow is introduced.
    out = ROOT / "local_runs/r90_native_credit_path_diagnostic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)
                   + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(diagnostic(), sort_keys=True, allow_nan=False))
