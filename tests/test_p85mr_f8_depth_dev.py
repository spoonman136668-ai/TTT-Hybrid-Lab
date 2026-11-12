"""Exact original-depth vs differentiable F8 alternative, no source effects."""
from pathlib import Path
import json
import inspect

import pytest
import torch

from model import CausalDepthRouter
from experiments import p85mr_f8_depth_dev as f8


def _inactive(depth):
    with torch.no_grad():
        depth.router.weight.zero_()
        depth.router.bias.fill_(-8.0)


def test_inactive_original_is_identical_but_cannot_update_router():
    torch.manual_seed(41)
    # An entirely skipped hard branch is the identity; give the input its
    # own graph, so backward can verify missing ROUTER grads (not fail).
    x = torch.randn(2, 5, 16, requires_grad=True)
    old = CausalDepthRouter(16)
    _inactive(old)
    got = old(x)
    assert torch.equal(got, x)
    assert old.last_utilization == 0.0
    got.square().mean().backward()
    assert old.router.weight.grad is None
    assert old.router.bias.grad is None


def test_inactive_surrogate_preserves_hard_forward_and_reaches_router():
    torch.manual_seed(41)
    x = torch.randn(2, 5, 16)
    old = CausalDepthRouter(16)
    _inactive(old)
    new = f8.InactiveSurrogateDepth(16)
    new.load_state_dict(old.state_dict(), strict=True)
    assert torch.equal(new(x), old(x))
    assert new.last_utilization == old.last_utilization == 0.0
    new(x).square().mean().backward()
    for grad in (new.router.weight.grad, new.router.bias.grad):
        assert grad is not None
        assert bool(torch.isfinite(grad).all())
        assert bool((grad.abs() > 0).any())
    # With hard=0, the FFN still gets no actual parameter update.
    for p in new.ffn.parameters():
        assert p.grad is not None
        assert torch.count_nonzero(p.grad) == 0


def test_always_active_activates_original_ffn_gradient():
    torch.manual_seed(42)
    x = torch.randn(2, 5, 16)
    old = CausalDepthRouter(16)
    _inactive(old)
    candidate = f8.AlwaysActiveDepth(16)
    candidate.load_state_dict(old.state_dict(), strict=True)
    y = candidate(x)
    assert y.shape == x.shape
    assert not torch.equal(y, x)
    assert candidate.last_utilization == 1.0
    y.square().mean().backward()
    assert any(p.grad is not None and bool((p.grad.abs() > 0).any())
               for p in candidate.ffn.parameters())


def test_full_original_hybrid_constructor_keeps_native_identity_and_shape():
    # No original pretrained model or external source touched.
    seed = 985201
    native = f8.model_from_arm(seed, f8.ORIGINAL)
    surrogate = f8.model_from_arm(seed, f8.SURROGATE)
    always = f8.model_from_arm(seed, f8.ALWAYS)
    without = f8.model_from_arm(seed, f8.WITHOUT)
    assert type(native.depth) is CausalDepthRouter
    assert type(surrogate.depth) is f8.InactiveSurrogateDepth
    assert type(always.depth) is f8.AlwaysActiveDepth
    assert without.depth is None
    assert f8.parameter_receipt(native)["static_parameters"] == (
        f8.parameter_receipt(surrogate)["static_parameters"])
    assert f8.parameter_receipt(native)["static_parameters"] == (
        f8.parameter_receipt(always)["static_parameters"])
    assert f8.parameter_receipt(without)["static_parameters"] < (
        f8.parameter_receipt(native)["static_parameters"])
    seq = torch.zeros(1, 2, 32)
    for m in (native, surrogate, always, without):
        assert m(seq).shape == (1, 2, 16)


def test_no_source_or_training_authority_emerges_from_dev_mechanism():
    freeze = (Path(__file__).resolve().parents[1]
              / "experiments" / "p85mr_f8_depth_gradient_dev_freeze.json")
    data = json.loads(freeze.read_text(encoding="utf-8"))
    assert data["stage"] == "PREIMPLEMENTATION_DEV_ONLY_NO_SCIENCE_FIRSTUSE"
    assert not data["separate_scientific_f8"]["cal_or_ood_firstuse_approved"]
    assert not data["separate_scientific_f8"]["frozen_source_cohort"]
    with pytest.raises(PermissionError, match="F8_ORIGINAL_SOURCE_FIRSTUSE_NOT_AUTHORIZED"):
        f8.require_external_firstuse()
    with pytest.raises(ValueError, match="F8_UNREGISTERED_DEPTH_ARM"):
        f8.model_from_arm(985201, "surprise_unapproved")
    src = inspect.getsource(f8)
    for forbidden in ("import causal_gym", "from causal_gym", "env.do(",
                      "env.reset(", "workflow_dispatch(", "torch.load(",
                      "create_runner(", "subprocess.run("):
        assert forbidden not in src
