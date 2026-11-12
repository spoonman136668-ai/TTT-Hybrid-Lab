"""F8 development: isolate original native depth-gradient reachability.

No causal-gym import, source FIT/CAL access, model training loop, owner,
dispatch, science first-use, source replay, or protected model mutation.
The prospectively frozen F8 mechanism DEV test operates ONLY on constructed
local tensors. It never claims downstream reward, retention or transfer.
"""
from __future__ import annotations

import torch
from model import CausalDepthRouter
from experiments.p85mr_f5_native_vs_controls import original_native

ORIGINAL = "original_hard_depth"
SURROGATE = "inactive_surrogate_depth"
ALWAYS = "always_active_depth"
WITHOUT = "no_depth"
ARMS = (ORIGINAL, SURROGATE, ALWAYS, WITHOUT)


class InactiveSurrogateDepth(CausalDepthRouter):
    """Retain original hard forward gate but provide gradient on inactive tokens.

    The surrogate computes the FFN for all tokens, including inactive tokens:
    there is NO sparse compute or speedup claim. When hard=0, router gradients
    can be nonzero, while FFN parameter gradients remain zero until routed.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        p = torch.sigmoid(self.router(z))
        hard = (p >= 0.5).to(dtype=p.dtype)
        self.last_utilization = float(hard.mean().detach().cpu())
        straight_through = (hard - p).detach() + p
        return x + self.ffn(z) * straight_through


class AlwaysActiveDepth(CausalDepthRouter):
    """Compute every original FFN path, for an explicit gradient positive.

    Retains unused original router parameters for structural comparison;
    effective active parameter count differs and must be separately reported.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        self.last_utilization = 1.0
        return x + self.ffn(z)


def model_from_arm(seed: int, arm: str):
    """Construct an original native F5 seed model and change ONLY depth."""

    if type(arm) is not str or arm not in ARMS:
        raise ValueError("F8_UNREGISTERED_DEPTH_ARM")
    model = original_native(seed)
    if arm == ORIGINAL:
        return model
    if arm == WITHOUT:
        model.depth = None
        return model
    original = model.depth
    if original is None or not isinstance(original, CausalDepthRouter):
        raise ValueError("F8_ORIGINAL_DEPTH_SOURCE_MISSING")
    width = original.router.in_features
    multiplier = original.ffn[0].out_features // width
    variant = (InactiveSurrogateDepth if arm == SURROGATE
               else AlwaysActiveDepth)(width, multiplier=multiplier)
    variant.load_state_dict(original.state_dict(), strict=True)
    model.depth = variant
    return model


def parameter_receipt(model):
    """Report static size; never imply parameter/compute-matched SSM science."""
    if not hasattr(model, "parameters"):
        raise TypeError("F8_UNQUALIFIED_MODEL")
    return {"static_parameters": sum(p.numel() for p in model.parameters()),
            "depth_parameters": (sum(p.numel() for p in model.depth.parameters())
                                 if model.depth is not None else 0),
            "scientific_pass": False, "original_effects_consumed": 0}


def require_external_firstuse(*_args, **_kwargs):
    raise PermissionError("F8_ORIGINAL_SOURCE_FIRSTUSE_NOT_AUTHORIZED")
