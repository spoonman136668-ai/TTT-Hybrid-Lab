"""R49 Native adapter retains real selected long history without R34 mutation."""
import pytest
import torch
from experiments import p85mr_f8r34_learned_factual_memory as r34
from experiments import p85mr_f8r49_public_event_selection as r49


def test_r49_native_preserves_shared_short_history_exact_output():
    original=r34.FactualMemory("native_factual_memory",r49.SEEDS[0])
    adapter=r49.R49LongHistoryFactualMemory("native_factual_memory",r49.SEEDS[0])
    adapter.load_state_dict(original.state_dict(),strict=True)
    for panel in ("fit",):
        w=r49.world(panel,0,0,1,1)
        actions,effects=r49.candidates(w)
        for ablate in (False,True):
            old=original(actions,effects,ablate_backbone=ablate)
            new=adapter(actions,effects,ablate_backbone=ablate)
            assert torch.equal(old,new)


def test_r49_native_long_history_keeps_all_observations_and_gradients():
    model=r49.Policy("event_context_native",r49.SEEDS[0],0.0726)
    for panel,length in (("cal",18),("ood",38)):
        w=r49.world(panel,0,1,-1,1)
        a,e=r49.candidates(w)
        assert a.shape==e.shape==(2,length,16)
        for ablate in (False,True):
            result=model(a,e,zero_native=ablate)
            assert result.shape==(1,2)
            assert bool(torch.isfinite(result).all())
        out=model(a,e)
        loss=out.square().sum()
        loss.backward()
        grads=[p.grad for p in model.native.parameters()]
        assert any(g is not None and bool(torch.isfinite(g).all())
                   and bool(torch.any(g!=0)) for g in grads)
        model.zero_grad(set_to_none=True)


def test_r49_native_rejects_unsupported_history_instead_of_silent_truncation():
    model=r49.R49LongHistoryFactualMemory("native_factual_memory",r49.SEEDS[0])
    with pytest.raises(ValueError,match="UNREGISTERED_HISTORY_LENGTH"):
        model(torch.zeros((2,19,16)),torch.zeros((2,19,16)))
    with pytest.raises(ValueError,match="INVALID_ACTION_EFFECT"):
        model(torch.zeros((2,18,16)),torch.zeros((2,17,16)))
