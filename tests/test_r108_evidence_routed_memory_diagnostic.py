"""R108 source/matched-learner CPU unit tests. DO NOT open heldout here."""
import numpy as np
import pytest
import torch

from experiments import r85_delayed_composition as r85
from experiments import r108_evidence_routed_memory_diagnostic as r108


def test_R108_freeze_authority_and_novelty_before_any_unit_unseen():
    assert r108.r108.audit_freeze()["scientific_pass"] is False
    f=r108.frozen()
    assert len(r108.TRAINABLE)==10 and len(r108.ARMS)==11
    assert f["training"]["updates_per_trainable_arm"]==512
    assert f["source"]["episodes_per_seed"]["UNIT_UNSEEN"]==128
    assert not f["authority"]["protected_original_scientific_first_use"]


def test_R108_actual_physics_two_visible_contexts_fit_only():
    record=r108.episode("UNIT_FIT",r108.SEEDS[0],12)
    p,e=record["public"],record["evaluator_only"]
    assert p["support_slots"].shape==(16,5,13)
    assert p["support_context"].shape==(16,)
    assert np.count_nonzero(p["support_context"]==0) in (6,8,10)
    assert len(p["query"])==2 and p["target"].shape==(2,)
    assert "query0" not in p and "query2" not in p
    assert "old_private_mask" not in p and "new_private_mask" not in p
    assert e["old_private_mask"] in r108.TRAIN
    assert e["new_private_mask"] in r108.TRAIN
    for phase,mask in ((0,e["old_private_mask"]),(1,e["new_private_mask"])):
        assert mask in r108.TRAIN
        assert int(np.sum(p["support_context"]==phase))>=6


def test_R108_exclusive_original_source_and_unregistered_seeds():
    with pytest.raises(PermissionError):
        r108.source("ORIGINAL_SCIENTIFIC_UNSEEN",r108.SEEDS[0])
    with pytest.raises(PermissionError):
        r108.episode("UNIT_FIT", 1006011, 0)
    with pytest.raises(PermissionError):
        r108._unit_world("FIT",r108.SEEDS[0],0,0,0,r108.TRAIN[0])
    with pytest.raises(TimeoutError):
        r108.episode("UNIT_FIT",r108.SEEDS[0],0,deadline=-1)


def test_R108_same_init_all_native_arms_with_fully_trained_control_models():
    models,widths=r108.make_models(r108.SEEDS[0])
    assert set(models)==set(r108.TRAINABLE)
    ref=dict(models[r108.NATIVE].named_parameters())
    for name in (r108.SINGLE,r108.TWO,r108.STATIC,r108.NO_CONTEXT,
                 r108.SHUFFLE,r108.ZERO,r108.NO_ROLES):
        for k,v in models[name].named_parameters():
            assert torch.equal(ref[k],v)
    assert widths[r108.GRU]["hidden_width"] in (32,64,96)
    assert widths[r108.MLP]["hidden_width"] in (32,64,96)


def test_R108_models_receive_same_physical_facts_and_public_phase():
    torch.set_num_threads(1)
    p=r108.episode("UNIT_FIT",r108.SEEDS[0],1)["public"]
    data={k:np.stack([v]) for k,v in p.items()}
    models,_=r108.make_models(r108.SEEDS[0])
    for name,model in models.items():
        for phase in (0,1):
            args=r108._inputs(data,np.asarray([0]),name,phase)
            assert args[0].shape==(1,16,5,13)
            assert args[1].shape==(1,16)
            assert args[2].shape==(1,16)
            assert args[3].shape==(1,5,13)
            assert args[4].shape==(1,)
            assert args[5].shape==(1,16)
            output=model(*args)
            assert output.shape==(1,) and bool(torch.isfinite(output).all())
    with pytest.raises(ValueError):
        r108._inputs(data,np.asarray([0]),"SOME_UNKNOWN_ARM",0)


def test_R108_phase_ablations_act_only_on_public_fields():
    p=r108.episode("UNIT_FIT",r108.SEEDS[0],5)["public"]
    data={k:np.stack([v]) for k,v in p.items()}
    ix=np.asarray([0])
    native=r108._inputs(data,ix,r108.NATIVE,1)
    contextless=r108._inputs(data,ix,r108.NO_CONTEXT,1)
    no_feedback=r108._inputs(data,ix,r108.ZERO,1)
    missing_roles=r108._inputs(data,ix,r108.NO_ROLES,1)
    assert torch.all(contextless[2]==0) and torch.all(contextless[4]==0)
    assert torch.all(no_feedback[1]==0)
    assert torch.all(missing_roles[0][:,:,:4,8]==0)
    assert torch.any(native[2]!=0) and torch.any(native[4]!=0)
    assert not torch.equal(native[1],no_feedback[1])


def test_R108_frozen_checks_map_exactly_and_cannot_report_scientific_pass():
    base={arm:{"old_before_accuracy":.5,"old_after_accuracy":.5,
               "new_after_accuracy":.5,"new_after_oracle_regret":1.0,
               "changed_law_new_after_accuracy":.5,
               "unchanged_law_new_after_accuracy":.5} for arm in r108.ARMS}
    result=r108.frozen_checks(base,0)
    assert set(result["checks"])==set(
        r108.frozen()["evaluation"]["frozen_checks_all_seeds"])
    assert result["scientific_pass"] is False
    assert not result["all_unit_only_checks_pass"]


def test_R108_isolated_cold_reload_on_FIT_public_fixture_only():
    torch.set_num_threads(1)
    public=r108.episode("UNIT_FIT",r108.SEEDS[0],7)["public"]
    q0=public["query"].copy()
    q2=public["query"].copy()
    q0[:,-1,0]=1.0
    q0[:,-1,2]=0.0
    q2[:,-1,0]=0.0
    q2[:,-1,2]=1.0
    fixture={k:np.stack([public[k]]) for k in
             ("support_slots","support_rewards","support_context")}
    fixture["query0"]=q0[None]
    fixture["query2"]=q2[None]
    models,_=r108.make_models(r108.SEEDS[0])
    assert r108.cold_prediction_parity(models[r108.NATIVE],fixture,None)==0


def test_R108_router_genuine_trainable_gradient_and_fixed_gate_control():
    torch.set_num_threads(1)
    public=r108.episode("UNIT_FIT",r108.SEEDS[0],8)["public"]
    data={k:np.stack([public[k]]) for k in public}
    models,_=r108.make_models(r108.SEEDS[0])
    args=r108._inputs(data,np.array([0]),r108.NATIVE,1)
    native=models[r108.NATIVE]
    loss=native(*args).square().sum()
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in native.router.parameters())
    assert native.last_route_gate.shape==(1,1)
    assert 0.0<float(native.last_route_gate.item())<1.0
    static=models[r108.STATIC]
    static(*args)
    assert static.last_route_gate.item()==.5
    single=models[r108.SINGLE]
    single(*args)
    assert single.last_route_gate.item()==0.0
    two=models[r108.TWO]
    two(*args)
    assert two.last_route_gate.item()==1.0


def test_R108_holdout_never_available_to_fitting_inputs():
    fit=r108.source("UNIT_FIT",r108.SEEDS[0])
    assert "old_private_mask" not in fit and "new_private_mask" not in fit
    assert "query0" not in fit and "query2" not in fit
    assert "reward0" not in fit and "reward2" not in fit
    assert r108.frozen()["evaluation"]["science_acceptance_from_unit_only"] is False
