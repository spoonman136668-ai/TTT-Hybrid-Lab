"""Stable public-event ordering must not use platform-specific tied topk."""
import pytest
import torch

from experiments import p85mr_f8r49_public_event_selection as r49


def test_public_ranking_uses_observed_values_then_first_seen_order():
    tags=torch.tensor([[0.,0.,0.,0.],
                       [.42,.18,.03,0.],
                       [.42,.42,.03,0.],
                       [.03,.03,.42,.18]],dtype=torch.float32)
    assert r49.stable_public_top_two(tags).tolist()==[
        [0,1],[0,1],[0,1],[2,3]]
    assert r49.stable_public_top_two(tags).tolist()==[
        [0,1],[0,1],[0,1],[2,3]]
    with pytest.raises(ValueError,match="INVALID_PUBLIC_EVENT_RANKING"):
        r49.stable_public_top_two(torch.ones((2,1)))
    with pytest.raises(ValueError,match="INVALID_PUBLIC_EVENT_RANKING"):
        r49.stable_public_top_two(torch.tensor([[float("nan"),0.]]))


def test_tag_erasure_ties_are_deterministic_and_no_unseen_cue_is_used():
    # Different hidden worlds receive the same deterministic rule, based
    # solely on the selected public record order after all tags are erased.
    for panel in ("fit","cal","ood"):
        for context in (0,1):
            for gate in (-1,1):
                w=r49.world(panel,0,context,1,gate)
                a,e=r49.candidates(w,erase_tag=True)
                tags,delta,fifth,ctx,observed=r49.components(
                    a,e,torch.tensor(.0726))
                assert torch.all(tags==0)
                assert r49.stable_public_top_two(tags).tolist()==[[0,1]]
                assert ctx.item()==context
                assert torch.isfinite(delta).all()
                assert torch.isfinite(fifth).all()


def test_non_erased_unique_top_two_indices_match_original_selector():
    for panel in ("fit","cal","ood"):
        for context in (0,1):
            w=r49.world(panel,0,context,1,1)
            a,e=r49.candidates(w)
            tags,_,_,_,_=r49.components(a,e,torch.tensor(.0726))
            stable_ids=r49.stable_public_top_two(tags)
            original_ids=torch.topk(tags,2,dim=-1).indices
            assert sorted(stable_ids[0].tolist())==sorted(
                original_ids[0].tolist())
