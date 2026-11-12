"""Fabricated microfixtures only. No P76/F3/F4 source or checkpoint access."""
from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from experiments.p76_memory_repair import FactualRepair
from experiments.p76r1_source_neutral_updates import (
    SCHEMA, FEATURES, FIXED_ETAS, SourceNeutralUpdate,
)


def toy(batch: int = 3, steps: int = 9):
    generator = torch.Generator().manual_seed(44433)
    a = F.normalize(torch.randn(batch, steps, 16, generator=generator), dim=-1)
    y = torch.randn(batch, steps, 16, generator=generator)
    q = F.normalize(torch.randn(batch, 16, generator=generator), dim=-1)
    return {"action": a, "effect": y}, q


@pytest.mark.parametrize("seed", (976201, 976203, 976207))
def test_full5_bitwise_original_p76_parity_without_new_training(seed):
    facts, query = toy()
    original = FactualRepair("learned_contradiction_repair", seed)
    control = SourceNeutralUpdate("full5", theta=original.theta.detach())
    with torch.no_grad():
        original_state = original.encode(facts)
        state = control.encode(facts, schema=SCHEMA)
        assert torch.equal(state, original_state)
        assert torch.equal(control.read(state, query), original.read(original_state, query))
    assert control.persistent_floats == 256 and control.learned_static_floats == 5


@pytest.mark.parametrize("eta,name", ((0.5, "all_write_eta_0.5"), (1.0, "all_write_eta_1.0")))
def test_fixed_reference_exact_parity(eta, name):
    facts, query = toy()
    control = SourceNeutralUpdate("fixed", fixed_eta=eta)
    original = FactualRepair(name)
    with torch.no_grad():
        assert torch.equal(control.encode(facts, schema=SCHEMA), original.encode(facts))
    assert control.learned_static_floats == 0


@pytest.mark.parametrize("method", tuple(FEATURES))
def test_independent_feature_definitions_and_gradients(method):
    facts, query = toy()
    model = SourceNeutralUpdate(method, theta=torch.zeros(len(FEATURES[method])))
    assert model.persistent_floats == 256
    assert model.learned_static_floats == len(FEATURES[method])
    loss = model(facts, query, schema=SCHEMA).square().mean()
    loss.backward()
    assert model.theta.grad is not None
    assert bool(torch.isfinite(model.theta.grad).all())
    assert model.theta.grad.abs().sum().item() > 0


@pytest.mark.parametrize("method", tuple(FEATURES))
def test_bitwise_split_and_cold_resume(method):
    facts, query = toy()
    model = SourceNeutralUpdate(method, theta=torch.zeros(len(FEATURES[method])))
    first = {k: v[:, :4] for k, v in facts.items()}
    second = {k: v[:, 4:] for k, v in facts.items()}
    with torch.no_grad():
        one = model.encode(facts, schema=SCHEMA)
        part = model.encode(first, schema=SCHEMA)
        saved = part.clone()
        resumed = model.encode(second, schema=SCHEMA, carry=saved)
        assert torch.equal(one, resumed)
        assert torch.equal(part, saved)
        a = model.read(one, query)
        assert torch.equal(a, model.read(one, query))


def test_prefix_no_future_effect_and_wrong_credit_sensitivity():
    facts, query = toy()
    model = SourceNeutralUpdate("direction2", theta=torch.tensor([0.1, 0.4]))
    earlier = {k: v[:, :5] for k, v in facts.items()}
    tampered = {k: v.clone() for k, v in facts.items()}
    tampered["effect"][:, 5:] += 17
    altered = {k: v[:, :5] for k, v in tampered.items()}
    with torch.no_grad():
        assert torch.equal(model.encode(earlier, schema=SCHEMA), model.encode(altered, schema=SCHEMA))
        wrong = {"action": facts["action"].flip(1), "effect": facts["effect"]}
        assert not torch.equal(model.encode(facts, schema=SCHEMA), model.encode(wrong, schema=SCHEMA))


def test_schema_unit_shape_and_unauthorized_features_fail_closed():
    facts, query = toy()
    model = SourceNeutralUpdate("fixed", fixed_eta=0.5)
    with pytest.raises(ValueError, match="UNAPPROVED_UNITS"):
        model.encode(facts, schema="f4.tanh7.v1")
    with pytest.raises(ValueError, match="P76_FACTUAL_HISTORY_ONLY"):
        model.encode({**facts, "future_effect": query}, schema=SCHEMA)
    with pytest.raises(ValueError, match="NOT_UNIT_NORMALIZED"):
        model.encode({**facts, "action": facts["action"] * 2}, schema=SCHEMA)
    with pytest.raises(ValueError, match="P76_BAD_FACTS"):
        model.encode({"action": facts["action"], "effect": facts["effect"][..., :7]}, schema=SCHEMA)
    bad = {k: v.clone() for k, v in facts.items()}
    bad["effect"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="P76_NONFINITE_FACTS"):
        model.encode(bad, schema=SCHEMA)
    with pytest.raises(ValueError, match="P76R1_INVALID_COLD_STATE"):
        model.encode(facts, schema=SCHEMA, carry=torch.zeros(3, 7, 7))
    assert 0.5 in FIXED_ETAS
    with pytest.raises(ValueError, match="FIXED_RATE_NOT_PREDECLARED"):
        SourceNeutralUpdate("fixed", fixed_eta=0.91)
    with pytest.raises(ValueError, match="P76R1_THETA_REQUIRED"):
        SourceNeutralUpdate("full5")
    with pytest.raises(ValueError, match="P76R1_UNKNOWN_METHOD"):
        SourceNeutralUpdate("future_oracle", theta=torch.zeros(5))
