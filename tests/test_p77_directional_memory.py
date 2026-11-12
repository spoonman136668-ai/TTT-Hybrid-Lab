import sys
from pathlib import Path
import pytest
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from p77_write_policy import WritePolicy, MODES, MODEL_SEEDS, N_FEATURES, ETA_GRID
from p77_factual_source import make_world, make_batch, factual, audit
from p76_memory_repair import FactualRepair


def toy():
    g=torch.Generator(device='cpu').manual_seed(778802)
    x=F.normalize(torch.randn(2,10,16,generator=g),dim=-1)
    y=torch.randn(2,10,16,generator=g)
    q=F.normalize(torch.randn(2,16,generator=g),dim=-1)
    return {'action':x,'effect':y},q


@pytest.mark.parametrize('mode',MODES)
def test_state_and_cold_restore(mode):
    facts,q=toy()
    m=WritePolicy(mode,seed=977201 if mode in N_FEATURES else None,eta=0.5 if mode=='fixed_eta' else None)
    with torch.no_grad():
        full=m.encode(facts)
        first=m.encode({k:v[:,:6] for k,v in facts.items()})
        clone=first.clone()
        resumed=m.encode({k:v[:,6:] for k,v in facts.items()},carry=first)
        assert torch.equal(full,resumed)
        assert torch.equal(first,clone)
        assert torch.equal(m.read(full,q),m.read(resumed,q))
    assert full.shape==(2,16,16) and full.numel()==2*256


def test_original_full5_matches_original_p76_policy_for_same_theta():
    f,q=toy()
    m=WritePolicy('full5',977201)
    old=FactualRepair('learned_contradiction_repair',976201)
    with torch.no_grad():
        old.theta.copy_(m.theta)
    assert torch.allclose(m(f,q),old(f,q),atol=0,rtol=0)


def test_initial_features_paired_and_independent():
    g=torch.get_rng_state().clone()
    full=WritePolicy('full5',977201)
    cosine=WritePolicy('cosine4',977201)
    no_dir=WritePolicy('non_directional3',977201)
    p70=WritePolicy('p70_four_param',977201)
    assert torch.equal(full.theta[:4],cosine.theta)
    assert torch.equal(cosine.theta[:3],no_dir.theta)
    assert torch.equal(cosine.theta,p70.theta)
    assert not torch.equal(full.theta,WritePolicy('full5',977203).theta)
    assert torch.equal(g,torch.get_rng_state())


def test_strict_bad_input_and_unregistered_modes():
    f,q=toy()
    model=WritePolicy('full5',977201)
    for key in ('future_query','query_changed','phase','role'):
        with pytest.raises(ValueError,match='P76_FACTUAL_HISTORY_ONLY'):
            model.encode(dict(f,**{key:q}))
    with pytest.raises(ValueError,match='P77_BAD_ETA'):
        WritePolicy('fixed_eta',eta=.17)
    with pytest.raises(ValueError,match='P77_UNREGISTERED_SEED'):
        WritePolicy('full5',976201)
    with pytest.raises(ValueError,match='P77_BAD_MODE'):
        WritePolicy('oracle')
    with pytest.raises(ValueError,match='P77_BAD_CARRY'):
        model.encode(f,carry=torch.zeros(2,16,15))


def test_gradients_and_write_limit():
    f,q=toy()
    for mode in N_FEATURES:
        m=WritePolicy(mode,977201)
        loss=m(f,q).square().mean()
        loss.backward()
        assert torch.isfinite(m.theta.grad).all()
        assert m.theta.grad.abs().sum()>0
        assert len(m.theta)==N_FEATURES[mode]


def test_fixed_controls_and_contradiction_correction():
    f,q=toy()
    assert torch.allclose(WritePolicy('fixed_eta',eta=.5)(f,q),FactualRepair('all_write_eta_0.5')(f,q),atol=0,rtol=0)
    assert torch.allclose(WritePolicy('fixed_contradiction')(f,q),FactualRepair('fixed_contradiction_repair')(f,q),atol=0,rtol=0)
    assert len(ETA_GRID)==10


def test_source_independent_factual_and_holdout_rejected():
    w=make_world('fit',0)
    assert w['action'].shape==(96,16)
    assert set(factual(w))=={'action','effect'}
    for key in ('ood_0','ood_1','ood_2','negative'):
        with pytest.raises(ValueError,match='P77_HOLDOUT_NOT_ADMITTED'):
            make_world(key,0)
    assert not torch.equal(make_batch('fit',8)['action'],make_batch('val',8)['action'])
