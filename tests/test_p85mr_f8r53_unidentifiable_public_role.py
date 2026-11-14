"""R53 tag-erasure is an exact information tie, not an OS-specific ULP vote."""
import torch

from experiments import p85mr_f8r53_repeated_intervention_stability as r53


def test_all_public_tag_roles_erased_gives_exact_neutral_preaction_costs():
    r53.locked()
    for seed in r53.SEEDS:
        for arm in ("learned_repeatability","native_repeatability"):
            # Explicit nonzero trained-like parameters prevent a trivial
            # zero-initialization from masking the precision problem.
            policy=r53.Policy(arm,seed,.46)
            with torch.no_grad():
                policy.slopes.copy_(torch.tensor([2.57,-2.61]))
                policy.beta.fill_(1.96)
            for panel in ("fit","cal","ood"):
                for ctx in (0,1):
                    for gate in (-1,1):
                        w=r53.world(panel,0,ctx,1,gate)
                        a,e=r53.candidates(w,erase_tag=True)
                        tags,_,_,_,_,_,_,_,_=r53.public_repeated_features(
                            a,e,policy.fit_offset)
                        assert torch.count_nonzero(tags)==0
                        with torch.no_grad():
                            cost=policy(a,e)
                        assert cost.shape==(1,2)
                        assert torch.isfinite(cost).all()
                        assert torch.count_nonzero(cost)==0
                        # In the original, learned context and selected
                        # consequence remain accessible for normal views.
                        normal_a,normal_e=r53.candidates(w)
                        normal_tag,_,_,_,_,_,_,_,_=r53.public_repeated_features(
                            normal_a,normal_e,policy.fit_offset)
                        assert torch.count_nonzero(normal_tag)>0


def test_exact_tie_guard_does_not_rewrite_partial_public_tag_information():
    model=r53.Policy("learned_repeatability",r53.SEEDS[0],.46)
    with torch.no_grad():
        model.slopes.copy_(torch.tensor([2.6,-2.6]))
        model.beta.fill_(2.0)
    w=r53.world("ood",0,1,1,-1)
    a,e=r53.candidates(w)
    with torch.no_grad():
        actual=model(a,e)
        zero_effect=model(a,e,zero_signal=True)
    assert actual.shape==(1,2)
    assert torch.isfinite(actual).all()
    assert torch.isfinite(zero_effect).all()
    assert torch.count_nonzero(actual)>0
    assert torch.count_nonzero(zero_effect)==0
