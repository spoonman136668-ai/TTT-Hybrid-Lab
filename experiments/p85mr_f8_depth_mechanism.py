"""F8 source-neutral original native depth-gradient mechanism falsifier.

The original four-block HybridPilot and F7 raw-action bypass are imported
unchanged. This creates paired *parameter-identical* hard/soft depth readouts
and proves reachability in deliberately fabricated action/effect histories.
No CartPoleWind or original F7 spent FIT/CAL/OOD source access occurs here.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from model import CausalDepthRouter
from experiments import p85mr_f5_native_vs_controls as f5
from experiments import p85mr_f7_native_candidate as f7
from experiments import p85mr_f7_protocol as p7

ROOT = Path(__file__).resolve().parents[1]
FREEZE_BLOB = "7fcff2117dde14bafb766506ea4c93ee6e36da2a"
F7_PARENT = "21c8df5b00e7b83a10b37f9b9ca3b3b9de362b5e"
MODEL_BLOB = "5d9ea57b827aa1796548447868fa5ee14c251b49"
MODES = ("hard_original", "surrogate_hard_forward", "soft_dense", "always_on")
SEEDS = p7.MODEL_SEEDS


def locked():
    raw = (ROOT / "experiments/p85mr_f8_depth_mechanism_freeze.json").read_bytes()
    if p7.git_blob(raw) != FREEZE_BLOB:
        raise ValueError("F8_FROZEN_PROTOCOL_DRIFT")
    frozen = json.loads(raw)
    study = frozen["study"]
    future = frozen["future_F8_external_trial_PROPOSED_ONLY"]
    authority = frozen["authority"]
    if (
        frozen.get("schema") != "ttt-hybrid.f8-native-depth-gradient-mechanism-freeze.v1"
        or frozen["parent"]["exact_F7_sha"] != F7_PARENT
        or frozen["parent"]["original_F7_model_blob"] != MODEL_BLOB
        or study["mode_order"] != list(MODES)
        or study["model_seeds"] != list(SEEDS)
        or study["inactive_probe_router_bias"] != -0.25
        or study["fixed_threshold"] != 0.5
        or study["training_epoch_limit_this_gate"] != 0
        or future["status"] != "NOT_AUTHORIZED_NOT_FROZEN_FOR_FIRSTUSE"
        or future["firstuse_scoped_approval"] is not False
        or future["CAL_OOD_approval"] is not False
        or authority["F8_real_simulator_effects"] != 0
        or authority["F8_authentic_source_training"] is not False
        or authority["no_new_workflow_or_scheduler"] is not True
        or authority["scientific_PASS"] is not False
    ):
        raise ValueError("F8_SCOPE_OR_SOURCE_AUTHORITY_DRIFT")
    if p7.git_blob((ROOT / "model.py").read_bytes()) != MODEL_BLOB:
        raise ValueError("F8_ORIGINAL_NATIVE_MODEL_DRIFT")
    p7.frozen()
    return frozen


class GradientCapableDepth(CausalDepthRouter):
    """Parameter-identical native depth; registered routing mode is fixed.

    surrogate_hard_forward: identical hard-mask forward values at the same
    weights, but the dense sigmoid times FFN expression contributes to the
    backward graph even for inactive tokens. This adds compute and may harm
    scientific efficacy; it does NOT promise a better trained model.
    """

    def __init__(self, width: int, mode: str):
        if mode not in MODES[1:]:
            raise ValueError("F8_UNREGISTERED_DEPTH_MODE")
        super().__init__(width)
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(x, torch.Tensor)
            or x.ndim != 3
            or x.shape[-1] != self.norm.normalized_shape[0]
            or not x.is_floating_point()
            or not bool(torch.isfinite(x).all())
        ):
            raise ValueError("F8_INVALID_NATIVE_DEPTH_STATE")
        if self.mode == "surrogate_hard_forward":
            # Preserve EXACT original hard forward, including its selected
            # sparse matrix-multiply shape. Add a second, dense path whose
            # values cancel but whose gradient reaches inactive router/FFN.
            # This has an explicit COMPUTE cost; no speed or science claim.
            reference = super().forward(x)
            z = self.norm(x)
            p = torch.sigmoid(self.router(z))
            soft = p * self.ffn(z)
            result = reference + (soft - soft.detach())
        elif self.mode == "always_on":
            self.last_utilization = 1.0
            result = x + self.ffn(self.norm(x))
        elif self.mode == "soft_dense":
            z = self.norm(x)
            p = torch.sigmoid(self.router(z))
            hard = (p >= 0.5).to(x.dtype)
            self.last_utilization = float(hard.mean().detach().cpu())
            result = x + p * self.ffn(z)
        else:
            raise ValueError("F8_UNKNOWN_DEPTH_MODE")
        if not bool(torch.isfinite(result).all()):
            raise ValueError("F8_NONFINITE_NATIVE_READOUT")
        return result


def _depth(model, bypass: bool):
    return model.original.depth if bypass else model.depth


def model_for(mode: str, seed: int, *, bypass: bool = False):
    locked()
    if mode not in MODES or type(seed) is not int or seed not in SEEDS or type(bypass) is not bool:
        raise ValueError("F8_UNREGISTERED_MODE_SEED_OR_BYPASS")
    arm = f7.K_BYPASS if bypass else f7.K_NATIVE
    model = f7.model_from_arm(arm, seed).cpu()
    if mode == "hard_original":
        return model
    original = _depth(model, bypass)
    if type(original) is not CausalDepthRouter:
        raise ValueError("F8_ORIGINAL_HARD_ROUTER_REQUIRED")
    replacement = GradientCapableDepth(original.norm.normalized_shape[0], mode)
    replacement.load_state_dict(original.state_dict(), strict=True)
    if bypass:
        model.original.depth = replacement
    else:
        model.depth = replacement
    if (
        dict(model.named_parameters()).keys() !=
        dict(f7.model_from_arm(arm, seed).named_parameters()).keys()
    ):
        raise ValueError("F8_PARAMETER_NAME_OR_CAPACITY_CHANGED")
    return model


def paired_forward_receipt(mode: str, seed: int, *, bypass: bool = False):
    """No source; report parameter count, no scientific result."""
    if mode == "hard_original":
        raise ValueError("F8_PAIR_REQUIRES_A_MECHANISM")
    ref = model_for("hard_original", seed, bypass=bypass)
    other = model_for(mode, seed, bypass=bypass)
    names = list(ref.state_dict())
    if names != list(other.state_dict()) or any(
        not torch.equal(ref.state_dict()[k], other.state_dict()[k]) for k in names
    ):
        raise ValueError("F8_INITIAL_NATIVE_PARAMETER_MISMATCH")
    return {
        "mode": mode, "seed": seed, "bypass": bypass,
        "static_parameters": sum(p.numel() for p in other.parameters()),
        "original_static_parameters": sum(p.numel() for p in ref.parameters()),
        "original_weight_hash": f5.hash_weights(ref),
        "candidate_weight_hash": f5.hash_weights(other),
        "source_effects_executed": 0,
        "native_weight_training": False,
        "independent_CAL_OOD": False,
        "scientific_pass": False,
        "deployment": False,
    }


def refuse_external_source(*args, **kwargs):
    raise PermissionError("F8_ORIGINAL_SOURCE_FIRSTUSE_NOT_AUTHORIZED")
