"""Standalone GPU smoke adapter for exact-source Mamba2 and official TTTLinear.

No upstream vendoring or modifications. The MoD-inspired depth router here is a
causal *threshold* router, not paper-exact sequence-global top-k MoD. The
sequence use is development-only; this is not a quality/speed demonstration.
"""
from __future__ import annotations
from pathlib import Path
import inspect
import torch
from torch import nn
from model import CausalAttentionBlock, CausalDepthRouter


def load_official_classes(mamba_root: Path, ttt_root: Path):
    """Require classes to come from the preflight-validated source files."""
    from mamba_ssm.modules.mamba2 import Mamba2
    from ttt import TTTConfig, TTTLinear
    paths = {
        "mamba2": (Path(inspect.getfile(Mamba2)).resolve(),
                   (mamba_root / "mamba_ssm/modules/mamba2.py").resolve()),
        "ttt_linear": (Path(inspect.getfile(TTTLinear)).resolve(),
                       (ttt_root / "ttt.py").resolve()),
        "ttt_config": (Path(inspect.getfile(TTTConfig)).resolve(),
                       (ttt_root / "ttt.py").resolve()),
    }
    for label, (actual, expected) in paths.items():
        if actual != expected:
            raise RuntimeError(f"UPSTREAM_CLASS_ORIGIN_MISMATCH:{label}")
    return Mamba2, TTTConfig, TTTLinear


class AuthenticHybrid(nn.Module):
    """Mamba2 -> causal attention -> TTTLinear -> threshold depth.

    `official_classes` is only for unit-test doubles; production runner never
    supplies it. TTT passes no cache and therefore resets at each forward.
    """
    def __init__(self, *, width=64, heads=4, mini_batch=8,
                 use_ttt=True, use_depth=True, official_classes=None):
        super().__init__()
        if width % heads or width % 64 or mini_batch < 1:
            raise ValueError("width must be divisible by heads and 64; mini_batch >=1")
        if official_classes is None:
            raise ValueError("Official classes must be supplied from validated source check")
        mamba_cls, config_cls, ttt_cls = official_classes
        self.width = width
        self.mini_batch = mini_batch
        self.norm_in = nn.LayerNorm(width)
        self.mamba = mamba_cls(d_model=width, d_state=64, headdim=64,
                               chunk_size=64, use_mem_eff_path=True)
        self.attn = CausalAttentionBlock(width, heads)
        self.norm_ttt = nn.LayerNorm(width)
        if use_ttt:
            cfg = config_cls(vocab_size=256, hidden_size=width,
                             intermediate_size=2*width,
                             num_hidden_layers=1, num_attention_heads=heads,
                             mini_batch_size=mini_batch, use_cache=False,
                             max_position_embeddings=4096, pre_conv=False,
                             ttt_layer_type="linear")
            self.ttt = ttt_cls(cfg, layer_idx=0)
        else:
            self.ttt = None
        self.depth = CausalDepthRouter(width) if use_depth else None
        self.output = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[2] != self.width or x.shape[1] < 1:
            raise ValueError("expected nonempty [batch,time,width]")
        b, t, _ = x.shape
        h = x + self.mamba(self.norm_in(x))
        h = self.attn(h)
        if self.ttt is not None:
            positions = torch.arange(t, device=x.device).unsqueeze(0).expand(b, -1)
            h = h + self.ttt(self.norm_ttt(h), position_ids=positions,
                             cache_params=None)
        if self.depth is not None:
            h = self.depth(h)
        return self.output(h)


def integration_checks(model: nn.Module, *, device: torch.device,
                       dtype: torch.dtype = torch.float32, seed: int = 90210):
    """Deterministic engineering checks; never claim scientific qualification."""
    torch.manual_seed(seed)
    model = model.to(device=device, dtype=dtype)
    if model.depth is not None:
        # Force some depth computation to exercise the FFN in this smoke.
        # Not representative of learned dynamic utilization.
        with torch.no_grad():
            model.depth.router.weight.zero_()
            model.depth.router.bias.fill_(1.0)
    model.eval()
    x = torch.randn(2, 16, model.width, device=device, dtype=dtype)
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x)
        prefix = model(x[:, :8])
        perturbed = x.clone()
        perturbed[:, 8:] += 7.0
        changed = model(perturbed)
    def require(test, code):
        if not test:
            raise AssertionError(code)
    require(y1.shape == x.shape, "BAD_OUTPUT_SHAPE")
    require(bool(torch.isfinite(y1).all()), "NONFINITE_FORWARD")
    require(bool(torch.allclose(y1, y2, rtol=3e-4, atol=3e-4)),
            "EPISODE_RESET_NOT_DETERMINISTIC")
    require(bool(torch.allclose(y1[:, :8], changed[:, :8], rtol=3e-4, atol=3e-4)),
            "SUFFIX_LEAKAGE")
    require(bool(torch.allclose(y1[:, :8], prefix, rtol=3e-4, atol=3e-4)),
            "PREFIX_INVARIANCE_FAILED")

    model.train()
    model.zero_grad(set_to_none=True)
    y = model(x)
    loss = y.square().mean()
    require(bool(torch.isfinite(loss)), "NONFINITE_LOSS")
    loss.backward()
    parts = {"mamba":model.mamba, "attention":model.attn,
             "ttt":model.ttt, "depth":model.depth, "output":model.output}
    nonzero = {}
    for name, part in parts.items():
        if part is None:
            nonzero[name] = "OMITTED_ABLATION"
            continue
        params = list(part.parameters())
        if not params:
            raise AssertionError(f"NO_PARAMETERS_{name}")
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
            raise AssertionError(f"NONFINITE_GRADIENT_{name}")
        nonzero[name] = any(p.grad is not None and bool((p.grad.abs()>0).any()) for p in params)
        require(nonzero[name], f"NO_NONZERO_GRADIENT_{name}")
    return {"forward_finite": True, "shape": list(y1.shape),
            "episode_reset_deterministic": True,
            "prefix_invariance": True, "suffix_negative_control": True,
            "backward_finite_nonzero": nonzero,
            "smoke_loss": float(loss.detach().cpu()),
            "depth_utilization_forced": model.depth is not None,
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "interpretation":"development engineering checks only"}
