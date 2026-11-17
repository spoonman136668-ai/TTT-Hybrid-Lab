"""R107 source/matched-learner CPU unit tests. DO NOT open heldout here."""
import numpy as np
import pytest
import torch

from experiments import r85_delayed_composition as r85
from experiments import r107_phase_correction_retention_diagnostic as r107


def test_R107_freeze_authority_and_novelty_before_any_unit_unseen():
    assert r107.r107.audit_freeze()["scientific_pass"] is False
    f=r107.frozen()
    assert len(r107.TRAINABLE)==8 and len(r107.ARMS)==9
    assert f["training"]["updates_per_trainable_arm"]==512
    assert f["source"]["episodes_per_seed"]["UNIT_UNSEEN"]==128
    assert not f["authority"]["protected_original_scientific_first_use"]


def test_R107_actual_physics_two_visible_contexts_fit_only():
    record=r107.episode("UNIT_FIT",r107.SEEDS[0],12)
    p,e=record["public"],record["evaluator_only"]
    assert p["support_slots"].shape==(16,5,13)
    assert p["support_context"].shape==(16,)
    assert np.count_nonzero(p["support_context"]==0) in (6,8,10)
    assert len(p["query"])==2 and p["target"].shape==(2,)
    assert "query0" not in p and "query2" not in p
    assert "old_private_mask" not in p and "new_private_mask" not in p
    assert e["old_private_mask"] in r107.TRAIN
    assert e["new_private_mask"] in r107.TRAIN
    for phase,mask in ((0,e["old_private_mask"]),(1,e["new_private_mask"])):
        assert mask in r107.TRAIN
        assert int(np.sum(p["support_context"]==phase))>=6


def test_R107_exclusive_original_source_and_unregistered_seeds():
    with pytest.raises(PermissionError):
        r107.source("ORIGINAL_SCIENTIFIC_UNSEEN",r107.SEEDS[0])
    with pytest.raises(PermissionError):
        r107.episode("UNIT_FIT", 1006011, 0)
    with pytest.raises(PermissionError):
        r107._unit_world("FIT",r107.SEEDS[0],0,0,0,r107.TRAIN[0])
    with pytest.raises(TimeoutError):
        r107.episode("UNIT_FIT",r107.SEEDS[0],0,deadline=-1)


def test_R107_same_init_all_native_arms_with_fully_trained_control_models():
    models,widths=r107.make_models(r107.SEEDS[0])
    assert set(models)==set(r107.TRAINABLE)
    ref=dict(models[r107.NATIVE].named_parameters())
    for name in (r107.SINGLE,r107.NO_CONTEXT,r107.SHUFFLE,
                 r107.ZERO,r107.NO_ROLES):
        for k,v in models[name].named_parameters():
            assert torch.equal(ref[k],v)
    assert widths[r107.GRU]["hidden_width"] in (32,64,96)
    assert widths[r107.MLP]["hidden_width"] in (32,64,96)


def test_R107_models_receive_same_physical_facts_and_public_phase():
    torch.set_num_threads(1)
    p=r107.episode("UNIT_FIT",r107.SEEDS[0],1)["public"]
    data={k:np.stack([v]) for k,v in p.items()}
    models,_=r107.make_models(r107.SEEDS[0])
    for name,model in models.items():
        for phase in (0,1):
            args=r107._inputs(data,np.asarray([0]),name,phase)
            assert args[0].shape==(1,16,5,13)
            assert args[1].shape==(1,16)
            assert args[2].shape==(1,16)
            assert args[3].shape==(1,5,13)
            assert args[4].shape==(1,)
            assert args[5].shape==(1,16)
            output=model(*args)
            assert output.shape==(1,) and bool(torch.isfinite(output).all())
    with pytest.raises(ValueError):
        r107._inputs(data,np.asarray([0]),"SOME_UNKNOWN_ARM",0)


def test_R107_phase_ablations_act_only_on_public_fields():
    p=r107.episode("UNIT_FIT",r107.SEEDS[0],5)["public"]
    data={k:np.stack([v]) for k,v in p.items()}
    ix=np.asarray([0])
    native=r107._inputs(data,ix,r107.NATIVE,1)
    contextless=r107._inputs(data,ix,r107.NO_CONTEXT,1)
    no_feedback=r107._inputs(data,ix,r107.ZERO,1)
    missing_roles=r107._inputs(data,ix,r107.NO_ROLES,1)
    assert torch.all(contextless[2]==0) and torch.all(contextless[4]==0)
    assert torch.all(no_feedback[1]==0)
    assert torch.all(missing_roles[0][:,:,:4,8]==0)
    assert torch.any(native[2]!=0) and torch.any(native[4]!=0)
    assert not torch.equal(native[1],no_feedback[1])


def test_R107_frozen_checks_map_exactly_and_cannot_report_scientific_pass():
    base={arm:{"old_before_accuracy":.5,"old_after_accuracy":.5,
               "new_after_accuracy":.5} for arm in r107.ARMS}
    result=r107.frozen_checks(base,0)
    assert set(result["checks"])==set(
        r107.frozen()["evaluation"]["frozen_checks_all_seeds"])
    assert result["scientific_pass"] is False
    assert not result["all_unit_only_checks_pass"]


def test_R107_isolated_cold_reload_on_FIT_public_fixture_only():
    torch.set_num_threads(1)
    public=r107.episode("UNIT_FIT",r107.SEEDS[0],7)["public"]
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
    models,_=r107.make_models(r107.SEEDS[0])
    assert r107.cold_prediction_parity(models[r107.NATIVE],fixture,None)==0
