"""P52 prospective experimental and physical-intervention negative controls."""
import hashlib
import inspect
from pathlib import Path
import numpy as np
import pytest
import torch
from scripts import native_p52_probe_value_dev as p52

SOURCE_SHA={
    820101:'93306831376565daed8ad1836d1e0d3bc0270625d76a9005bf47e6a1854995dc',
    820103:'970609fada88c9316f2ff7afe1c153892bc445d532336dfe4be3d0f7430fbec5',
    820107:'bd06e7f0912e67eb0b66f8e046f2b2005d4940e7a8a88e7bcec9af342be27361',
}

def test_frozen_source_cohorts_and_true_out_of_range_noise():
    for seed,digest in SOURCE_SHA.items():
        d=p52.worlds(seed)
        assert d['sha256']==digest
        n=89600 if seed==820101 else (1024 if seed==820103 else 2048)
        assert d['n']==n and d['prior'].shape==(n,6)
        assert d['theta'].shape==(n,)
        assert bool(torch.isfinite(d['prior']).all())
        if seed==820107:
            assert bool((d['prior'][:,2]>=.95).all())
            assert bool((d['prior'][:,4]>=.95).all())
            assert bool((d['prior'][:,5]<=.65).all())
    with pytest.raises(ValueError,match='P52_SOURCE_NOT_FROZEN'):
        p52.worlds(790101)


def test_parameter_budget_and_actual_probe_information_is_masked():
    ca=p52.FactorizedPolicy();gr=p52.ConventionalGRUPolicy()
    assert p52.count_params(ca)==902
    assert p52.count_params(gr)==954
    assert p52.count_params(gr)/p52.count_params(ca)<1.10
    data=p52.take_batch(p52.worlds(820103),slice(0,4))
    never=torch.zeros(4)
    assert torch.equal(p52.physical_feedback(data,never),torch.zeros(4))
    attempted=torch.ones(4)
    measured=p52.physical_feedback(data,attempted)
    assert torch.equal(measured,data['theta']*data['prior'][:,3]+data['eps_probe']*data['prior'][:,4])
    with pytest.raises(ValueError,match='P52_BAD_PHYSICAL_ACTION'):
        p52.physical_feedback(data,torch.full((4,),.5))
    c=p52.signed_reward(data,never,torch.ones(4))
    assert torch.equal(c,data['theta'])


def test_no_unexecuted_outcomes_are_policy_inputs_or_training_targets():
    sig=inspect.getsource(p52.FactorizedPolicy)+inspect.getsource(p52.ConventionalGRUPolicy)
    for forbidden in ("['theta']","['eps_probe']","random_action","source_sha256"):
        assert forbidden not in sig
    assert "o['reward'].detach()" in inspect.getsource(p52.train)
    assert "factual.roll(1)" in inspect.getsource(p52.train)
    d=p52.worlds(820107)
    obs0=d['prior'][:16].clone()
    assert obs0.shape==(16,6)
    m=p52.FactorizedPolicy().eval()
    with torch.no_grad():
        v=m.first(obs0)
    d['theta'].neg_()
    d['eps_probe'].mul_(2.0)
    with torch.no_grad():
        w=m.first(d['prior'][:16])
    assert torch.equal(v[0],w[0])


def test_fair_physical_bayes_reference_is_above_always_and_never():
    data=p52.worlds(820107)
    bayes=p52.bayes_reference(data,'bayes')
    always=p52.bayes_reference(data,'always')
    never=p52.bayes_reference(data,'never')
    simple=p52.bayes_reference(data,'simple')
    assert bayes['probe_rate']==pytest.approx(.267578125,abs=1e-9)
    assert bayes['net_reward']>never['net_reward']>always['net_reward']
    assert bayes['net_reward']>simple['net_reward']


def test_strict_frozen_negative_and_cold_state_remain_visible():
    result=p52.run()
    assert not result['all_predeclared_seed_gates']
    assert [row['seed'] for row in result['records']]==[8211,8223,8231]
    assert result['cohort_sha256']=={str(k):v for k,v in SOURCE_SHA.items()}
    assert result['parameters']=={'candidate':902,'gru':954,'wrong_credit':902}
    for row in result['records']:
        assert all(v['cold_exact'] for v in row['arms'].values())
        assert not row['gates']['beats_trained_gru_and_simple_by_point02']
        assert row['gates']['beats_corrupt_reward_credit_point05']
        assert not row['gates']['learns_probe_cost_gradient']
        assert row['arms']['candidate']['ood']['probe_rate']==0.0
        assert row['arms']['gru']['ood']['probe_rate']==0.0
