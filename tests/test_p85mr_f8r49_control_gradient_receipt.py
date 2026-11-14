"""R49 negative-control no-signal is an explicit null, not fake learning."""
import pytest
import torch

from experiments import p85mr_f8r49_public_event_selection as r49


def test_null_control_gradient_is_reported_without_masking_invalid_numerics():
    assert r49.productive_gradient([None, torch.zeros(2)]) is False
    assert r49.productive_gradient([torch.zeros(1), torch.tensor([0.,-1.])])
    with pytest.raises(ValueError, match="F8R49_NONFINITE_GRADIENT"):
        r49.productive_gradient([torch.tensor([float("nan")])])
    with pytest.raises(ValueError, match="F8R49_NONFINITE_GRADIENT"):
        r49.productive_gradient([torch.tensor([float("inf")])])


def test_frozen_main_model_has_actual_context_slope_gradient_on_fit():
    facts, labels=r49.factual_fit()
    offset=r49.calibration(facts)
    model=r49.Policy("event_context_selector",r49.SEEDS[0],offset)
    a,e,y=r49.batch(facts,labels,list(range(r49.BATCH)))
    torch.nn.functional.cross_entropy(-model(a,e),y).backward()
    grads=[p.grad for p in model.parameters() if p.requires_grad]
    assert r49.productive_gradient(grads)
    assert model.slopes.grad is not None
    assert bool(torch.isfinite(model.slopes.grad).all())
