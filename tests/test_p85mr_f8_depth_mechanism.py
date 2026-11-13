"""F8 original hard-depth causal mechanism: gradient and no-source controls."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from experiments import p85mr_f5_native_vs_controls as f5
from experiments import p85mr_f7_native_candidate as f7
from experiments import p85mr_f8_depth_mechanism as f8
from model import CausalDepthRouter

torch.set_num_threads(1)


def _force_off(router):
    with torch.no_grad():
        router.router.weight.zero_()
        router.router.bias.fill_(-.25)


def _force_on(router):
    with torch.no_grad():
        router.router.weight.zero_()
        router.router.bias.fill_(.25)


def _depth(model, bypass):
    return model.original.depth if bypass else model.depth


def _grad_norm(parameter):
    return 0. if parameter.grad is None else float(parameter.grad.abs().sum().item())


def _toy_facts():
    a = torch.zeros(5, 16)
    a[:, 0] = .7071
    a[:, 1] = .5
    a[:, 5] = torch.tensor([1., 0., 1., 0., 1.])
    a[:, 6] = 1. - a[:, 5]
    a = a / a.norm(dim=-1, keepdim=True)
    e = torch.zeros(5, 16)
    e[:, 0] = torch.arange(5)*.004
    e[:, 2] = torch.tensor([.002, .003, .004, .005, .006])
    e[:, 4] = .05
    e[:, 5] = torch.tensor([0., 0., 0., 0., 1.])
    return {"action": a.unsqueeze(0), "effect": e.unsqueeze(0)}


@pytest.mark.parametrize("seed", f8.SEEDS)
def test_f8_inactive_original_depth_is_unreachable_but_surrogate_fully_trains(seed):
    f8.locked()
    original = CausalDepthRouter(16)
    _force_off(original)
    surrogate = f8.GradientCapableDepth(16, "surrogate_hard_forward")
    surrogate.load_state_dict(original.state_dict(), strict=True)
    soft = f8.GradientCapableDepth(16, "soft_dense")
    soft.load_state_dict(original.state_dict(), strict=True)
    always = f8.GradientCapableDepth(16, "always_on")
    always.load_state_dict(original.state_dict(), strict=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        x = torch.randn(2, 5, 16, requires_grad=True)
    for depth in (original, surrogate, soft, always):
        depth.train()
        depth.zero_grad(set_to_none=True)
        probe = x.detach().clone().requires_grad_(True)
        y = depth(probe)
        assert y.shape == probe.shape and torch.isfinite(y).all()
        if depth is surrogate:
            assert torch.equal(y.detach(), original(x.detach()).detach())
        if depth is original:
            assert depth.last_utilization == 0.
        y.square().mean().backward()
        if depth is original:
            assert depth.router.weight.grad is None
            assert depth.ffn[0].weight.grad is None
        elif depth is always:
            assert depth.router.weight.grad is None  # explicit router ablation
            assert _grad_norm(depth.ffn[0].weight) > 0.
        else:
            assert depth.last_utilization == 0. if depth is surrogate else True
            assert _grad_norm(depth.router.bias) > 1e-10
            assert _grad_norm(depth.ffn[0].weight) > 1e-10
            assert bool(torch.isfinite(depth.router.bias.grad).all())
            assert bool(torch.isfinite(depth.ffn[0].weight.grad).all())


@pytest.mark.parametrize("seed", f8.SEEDS)
@pytest.mark.parametrize("bypass", [False, True])
def test_f8_original_whole_model_exact_forward_before_updates_and_causal_prefix(seed,bypass):
    arm = f7.K_BYPASS if bypass else f7.K_NATIVE
    ref = f8.model_for("hard_original",seed,bypass=bypass).eval()
    st = f8.model_for("surrogate_hard_forward",seed,bypass=bypass).eval()
    info = f8.paired_forward_receipt("surrogate_hard_forward",seed,bypass=bypass)
    assert info["static_parameters"] == info["original_static_parameters"]
    assert info["original_weight_hash"] == info["candidate_weight_hash"]
    assert info["source_effects_executed"]==0 and not info["scientific_pass"]
    facts = _toy_facts()
    for force in (_force_off,_force_on):
        force(_depth(ref,bypass)); force(_depth(st,bypass))
        with torch.no_grad():
            a = f7.prequential(arm,ref,facts)
            b = f7.prequential(arm,st,facts)
            assert torch.equal(a,b)
            assert _depth(ref,bypass).last_utilization in (0.,1.)
            changed={k:v.clone() for k,v in facts.items()}
            changed["effect"][:,2,:7] += .5
            prediction=f7.prequential(arm,st,changed)
            assert torch.equal(prediction[:,:3], b[:,:3])
        restored=f8.model_for("surrogate_hard_forward",seed,bypass=bypass).eval()
        restored.load_state_dict(deepcopy(st.state_dict()),strict=True)
        with torch.no_grad():
            assert torch.equal(f7.prequential(arm,restored,facts),b)


def test_f8_single_fictional_optimizer_update_reaches_unselected_router_ffn():
    facts=_toy_facts()
    mask=torch.zeros((1,5),dtype=torch.bool)
    mask[:,1:]=True
    for bypass in (False,True):
        arm=f7.K_BYPASS if bypass else f7.K_NATIVE
        hard=f8.model_for("hard_original",985201,bypass=bypass).train()
        fix=f8.model_for("surrogate_hard_forward",985201,bypass=bypass).train()
        _force_off(_depth(hard,bypass))
        _force_off(_depth(fix,bypass))
        hard.zero_grad(set_to_none=True)
        fix.zero_grad(set_to_none=True)
        lh=f7.loss_for_arm(arm,hard,facts,mask)
        lf=f7.loss_for_arm(arm,fix,facts,mask)
        assert lh.item()==pytest.approx(lf.item(),abs=1e-7)
        lh.backward();lf.backward()
        dh=_depth(hard,bypass);d=_depth(fix,bypass)
        assert dh.last_utilization==d.last_utilization==0.
        assert dh.router.weight.grad is None and dh.ffn[0].weight.grad is None
        assert _grad_norm(d.router.bias)>1e-10
        assert _grad_norm(d.ffn[0].weight)>1e-10
        before=d.router.bias.detach().clone()
        ff_before=d.ffn[0].weight.detach().clone()
        opt=torch.optim.AdamW(fix.parameters(),lr=.002,weight_decay=.01)
        torch.nn.utils.clip_grad_norm_(fix.parameters(),1.)
        opt.step()
        assert not torch.equal(d.router.bias,before)
        assert not torch.equal(d.ffn[0].weight,ff_before)


def test_f8_unknown_or_nonfinite_inputs_and_external_firstuse_fail_closed():
    with pytest.raises(ValueError,match="F8_UNREGISTERED_MODE_SEED_OR_BYPASS"):
        f8.model_for("new_control",985201)
    with pytest.raises(ValueError,match="F8_UNREGISTERED_MODE_SEED_OR_BYPASS"):
        f8.model_for("surrogate_hard_forward",0)
    with pytest.raises(ValueError,match="F8_UNREGISTERED_MODE_SEED_OR_BYPASS"):
        f8.model_for("soft_dense",985201,bypass=1)
    d=f8.GradientCapableDepth(16,"soft_dense")
    with pytest.raises(ValueError,match="F8_INVALID_NATIVE_DEPTH_STATE"):
        d(torch.full((1,3,16),float("nan")))
    with pytest.raises(ValueError,match="F8_INVALID_NATIVE_DEPTH_STATE"):
        d(torch.zeros((1,3,15)))
    with pytest.raises(PermissionError,match="F8_ORIGINAL_SOURCE_FIRSTUSE_NOT_AUTHORIZED"):
        f8.refuse_external_source(seed=1823000000)
    f=f8.locked()
    assert f["future_F8_external_trial_PROPOSED_ONLY"]["status"]=="NOT_AUTHORIZED_NOT_FROZEN_FOR_FIRSTUSE"
    assert f["authority"]["F8_real_simulator_effects"]==0
    body=Path(f8.__file__).read_text(encoding="utf-8")
    for bad in ("import causal_gym","from causal_gym","env.do(","env.reset(",
                "requests.get(","subprocess.run(","torch.load(","workflow_dispatch("):
        assert bad not in body
