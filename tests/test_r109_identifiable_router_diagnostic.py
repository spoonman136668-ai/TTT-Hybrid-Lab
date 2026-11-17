"""R109 public FIT-only test fixtures; never generate UNIT_UNSEEN here."""
import numpy as np
import pytest
import torch

from experiments import r85_delayed_composition as r85
from experiments import r109_public_custody as custody
from experiments import r109_identifiable_router_diagnostic as r109


def test_R109_frozen_novelty_authority_and_spent_panels(monkeypatch):
    f=r109.frozen()
    a=custody.audit_freeze()
    assert a["seeds"]==[1009011,1009023,1009037]
    assert len(a["fresh_unseen_masks"])==8
    assert not a["protected_firstuse"] and not a["scientific_pass"]
    assert f["evaluation"]["protected_scientific_pass_from_unit_only"] is False
    assert f["authority"]["no_new_runner_controller_or_scheduled_task"]
    assert len(r109.TRAINABLE)==11 and len(r109.ARMS)==12
    monkeypatch.setattr(custody,"FREEZE_BLOB","0"*40)
    with pytest.raises(RuntimeError,match="FREEZE_BLOB_DRIFT"):
        custody.audit_freeze()


def test_R109_physical_public_factual_only_and_negative_source():
    rec=r109.episode("UNIT_FIT",r109.SEEDS[0],9)
    p,secret=rec["public"],rec["evaluator_only"]
    assert p["support_slots"].shape==(16,5,13)
    assert p["support_context"].shape==(16,)
    assert np.count_nonzero(p["support_context"]==0) in (6,8,10)
    assert p["query"].shape==(2,5,13) and p["target"].shape==(2,)
    assert all(k not in p for k in ("old_private_mask","new_private_mask",
                                      "reward0","reward2","query0","query2"))
    assert secret["old_private_mask"] in r109.TRAIN
    with pytest.raises(PermissionError):
        r109.source("PROTECTED_ORIGINAL_FIRST_USE",r109.SEEDS[0])
    with pytest.raises(PermissionError):
        r109.episode("UNIT_FIT",1008011,9)
    with pytest.raises(TimeoutError):
        r109.episode("UNIT_FIT",r109.SEEDS[0],9,deadline=-1)


def test_R109_all_native_arms_exact_same_initial_parameters_and_controls():
    models,widths=r109.make_models(r109.SEEDS[0])
    base=dict(models[r109.NATIVE].named_parameters())
    for name in (r109.SINGLE,r109.TWO,r109.STATIC,r109.PHASE_ONLY,
                 r109.NO_CONTEXT,r109.SHUFFLE,r109.ZERO,r109.NO_ROLES):
        for k,p in models[name].named_parameters():
            assert torch.equal(base[k],p),name
    assert widths[r109.GRU]["hidden_width"] in (32,64,96)
    assert widths[r109.MLP]["hidden_width"] in (32,64,96)


def _fit():
    public=r109.episode("UNIT_FIT",r109.SEEDS[0],12)["public"]
    return {k:np.stack([v]) for k,v in public.items()}


def test_R109_frozen_gate_is_exclusive_action_value_path_and_has_gradient():
    torch.set_num_threads(1)
    data=_fit()
    models,_=r109.make_models(r109.SEEDS[0])
    args=r109._inputs(data,np.asarray([0]),r109.NATIVE,1)
    model=models[r109.NATIVE]
    model.train()
    y=model(*args)
    assert y.shape==(1,) and bool(torch.isfinite(y).all())
    y.square().mean().backward()
    assert any(p.grad is not None and bool(torch.isfinite(p.grad).all()) and
               bool(torch.any(p.grad!=0)) for p in model.router.parameters())
    with torch.no_grad():
        y0=model(*args,forced_gate=0.)
        y1=model(*args,forced_gate=1.)
        y05=model(*args,forced_gate=.5)
        learned=model(*args)
        gate=model.last_route_gate[:,0]
    assert torch.allclose(y05,.5*(y0+y1),atol=1e-6)
    assert torch.allclose(learned,gate*y1+(1-gate)*y0,atol=1e-6)
    with pytest.raises(PermissionError):
        model(*args,forced_gate=1)


def test_R109_phase_only_router_does_not_read_factual_feedback():
    torch.set_num_threads(1)
    models,_=r109.make_models(r109.SEEDS[0])
    data=_fit()
    ix=np.asarray([0])
    native_args=r109._inputs(data,ix,r109.NATIVE,1)
    zero_args=r109._inputs(data,ix,r109.ZERO,1)
    missing_args=r109._inputs(data,ix,r109.NO_ROLES,1)
    context_args=r109._inputs(data,ix,r109.NO_CONTEXT,1)
    assert torch.all(zero_args[1]==0)
    assert torch.all(missing_args[0][:,:,:4,8]==0)
    assert torch.all(context_args[2]==0) and torch.all(context_args[4]==0)
    phase=models[r109.PHASE_ONLY]
    phase(*native_args)
    a=phase.last_route_gate.clone()
    altered=list(native_args)
    altered[1]=torch.roll(altered[1],3,dims=1)
    phase(*altered)
    b=phase.last_route_gate
    assert torch.equal(a,b), "phase-only gate must not access reward evidence"
    fixed=((r109.SINGLE,0.),(r109.TWO,1.),(r109.STATIC,.5))
    for name,value in fixed:
        m=models[name]
        m(*native_args)
        assert m.last_route_gate.item()==value


def test_R109_all_fresh_arm_public_inputs_have_finite_action_values():
    torch.set_num_threads(1)
    data=_fit()
    models,_=r109.make_models(r109.SEEDS[0])
    for arm,model in models.items():
        for context in (0,1):
            args=r109._inputs(data,np.asarray([0]),arm,context)
            assert args[0].shape==(1,16,5,13)
            assert args[1].shape==(1,16)
            assert args[3].shape==(1,5,13)
            result=model(*args)
            assert result.shape==(1,) and bool(torch.isfinite(result).all())
    with pytest.raises(ValueError):
        r109._inputs(data,np.asarray([0]),"UNREGISTERED",0)


def test_R109_frozen_checks_map_and_no_science_promotion():
    base={arm:{"old_before_accuracy":.5,"old_after_accuracy":.5,
               "new_after_accuracy":.5,"new_after_oracle_regret":1.,
               "changed_law_new_after_accuracy":.5,
               "unchanged_law_new_after_accuracy":.5} for arm in r109.ARMS}
    r=r109.frozen_checks(base,0,{"feedback_action_flip_count":0})
    assert set(r["checks"])==set(
        r109.frozen()["evaluation"]["frozen_checks_all_seeds"])
    assert not r["scientific_pass"] and not r["all_unit_only_checks_pass"]


def test_R109_cold_restore_independent_child_on_FIT_only():
    torch.set_num_threads(1)
    data=_fit()
    public=r109.episode("UNIT_FIT",r109.SEEDS[0],12)["public"]
    q0=public["query"].copy()
    q2=public["query"].copy()
    q0[:,-1,0]=1.
    q0[:,-1,2]=0.
    q2[:,-1,0]=0.
    q2[:,-1,2]=1.
    data["query0"]=q0[None]
    data["query2"]=q2[None]
    model=r109.make_models(r109.SEEDS[0])[0][r109.NATIVE]
    assert r109.cold_prediction_parity(model,data,None)==0
