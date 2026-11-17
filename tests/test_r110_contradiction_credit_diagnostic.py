"""R110 public FIT-only test fixtures; never generate UNIT_UNSEEN here."""
import numpy as np
import pytest
import torch

from experiments import r85_delayed_composition as r85
from experiments import r110_public_custody as custody
from experiments import r110_public_contradiction_credit as credit
from experiments import r110_contradiction_credit_diagnostic as r110


def test_R110_frozen_novelty_authority_and_spent_panels(monkeypatch):
    f=r110.frozen()
    a=custody.audit_freeze()
    assert a["seeds"]==[1010011,1010023,1010037]
    assert len(a["fresh_unseen_masks"])==8
    assert not a["protected_firstuse"] and not a["scientific_pass"]
    assert f["evaluation"]["protected_scientific_pass_from_unit_only"] is False
    assert f["authority"]["no_new_runner_controller_or_scheduled_task"]
    assert len(r110.TRAINABLE)==13 and len(r110.ARMS)==14
    monkeypatch.setattr(custody,"FREEZE_BLOB","0"*40)
    with pytest.raises(RuntimeError,match="FREEZE_BLOB_DRIFT"):
        custody.audit_freeze()


def test_R110_physical_public_factual_only_and_negative_source():
    rec=r110.episode("UNIT_FIT",r110.SEEDS[0],9)
    p,secret=rec["public"],rec["evaluator_only"]
    assert p["support_slots"].shape==(16,5,13)
    assert p["support_context"].shape==(16,)
    assert np.count_nonzero(p["support_context"]==0) in (6,8,10)
    assert p["query"].shape==(2,5,13) and p["target"].shape==(2,)
    assert all(k not in p for k in ("old_private_mask","new_private_mask",
                                      "reward0","reward2","query0","query2"))
    assert secret["old_private_mask"] in r110.TRAIN
    with pytest.raises(PermissionError):
        r110.source("PROTECTED_ORIGINAL_FIRST_USE",r110.SEEDS[0])
    with pytest.raises(PermissionError):
        r110.episode("UNIT_FIT",1009011,9)
    with pytest.raises(TimeoutError):
        r110.episode("UNIT_FIT",r110.SEEDS[0],9,deadline=-1)


def test_R110_all_native_arms_exact_same_initial_parameters_and_controls():
    models,widths=r110.make_models(r110.SEEDS[0])
    base=dict(models[r110.NATIVE].named_parameters())
    for name in (r110.NO_AUX,r110.PLACEBO,r110.SINGLE,r110.TWO,r110.STATIC,r110.PHASE_ONLY,
                 r110.NO_CONTEXT,r110.SHUFFLE,r110.ZERO,r110.NO_ROLES):
        for k,p in models[name].named_parameters():
            assert torch.equal(base[k],p),name
    assert widths[r110.GRU]["hidden_width"] in (32,64,96)
    assert widths[r110.MLP]["hidden_width"] in (32,64,96)


def _fit():
    public=r110.episode("UNIT_FIT",r110.SEEDS[0],12)["public"]
    return {k:np.stack([v]) for k,v in public.items()}


def test_R110_frozen_gate_is_exclusive_action_value_path_and_has_gradient():
    torch.set_num_threads(1)
    data=_fit()
    models,_=r110.make_models(r110.SEEDS[0])
    args=r110._inputs(data,np.asarray([0]),r110.NATIVE,1)
    model=models[r110.NATIVE]
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


def test_R110_phase_only_router_does_not_read_factual_feedback():
    torch.set_num_threads(1)
    models,_=r110.make_models(r110.SEEDS[0])
    data=_fit()
    ix=np.asarray([0])
    native_args=r110._inputs(data,ix,r110.NATIVE,1)
    zero_args=r110._inputs(data,ix,r110.ZERO,1)
    missing_args=r110._inputs(data,ix,r110.NO_ROLES,1)
    context_args=r110._inputs(data,ix,r110.NO_CONTEXT,1)
    assert torch.all(zero_args[1]==0)
    assert torch.all(missing_args[0][:,:,:4,8]==0)
    assert torch.all(context_args[2]==0) and torch.all(context_args[4]==0)
    phase=models[r110.PHASE_ONLY]
    phase(*native_args)
    a=phase.last_route_gate.clone()
    altered=list(native_args)
    altered[1]=torch.roll(altered[1],3,dims=1)
    phase(*altered)
    b=phase.last_route_gate
    assert torch.equal(a,b), "phase-only gate must not access reward evidence"
    fixed=((r110.SINGLE,0.),(r110.TWO,1.),(r110.STATIC,.5))
    for name,value in fixed:
        m=models[name]
        m(*native_args)
        assert m.last_route_gate.item()==value


def test_R110_all_fresh_arm_public_inputs_have_finite_action_values():
    torch.set_num_threads(1)
    data=_fit()
    models,_=r110.make_models(r110.SEEDS[0])
    for arm,model in models.items():
        for context in (0,1):
            args=r110._inputs(data,np.asarray([0]),arm,context)
            assert args[0].shape==(1,16,5,13)
            assert args[1].shape==(1,16)
            assert args[3].shape==(1,5,13)
            result=model(*args)
            assert result.shape==(1,) and bool(torch.isfinite(result).all())
    with pytest.raises(ValueError):
        r110._inputs(data,np.asarray([0]),"UNREGISTERED",0)


def test_R110_frozen_checks_map_and_no_science_promotion():
    base={arm:{"old_before_accuracy":.5,"old_after_accuracy":.5,
               "new_after_accuracy":.5,"new_after_oracle_regret":1.,
               "changed_law_new_after_accuracy":.5,
               "unchanged_law_new_after_accuracy":.5} for arm in r110.ARMS}
    r=r110.frozen_checks(base,0,{"feedback_action_flip_count":0})
    assert set(r["checks"])==set(
        r110.frozen()["evaluation"]["frozen_checks_all_seeds"])
    assert not r["scientific_pass"] and not r["all_unit_only_checks_pass"]


def test_R110_cold_restore_independent_child_on_FIT_only():
    torch.set_num_threads(1)
    data=_fit()
    public=r110.episode("UNIT_FIT",r110.SEEDS[0],12)["public"]
    q0=public["query"].copy()
    q2=public["query"].copy()
    q0[:,-1,0]=1.
    q0[:,-1,2]=0.
    q2[:,-1,0]=0.
    q2[:,-1,2]=1.
    data["query0"]=q0[None]
    data["query2"]=q2[None]
    model=r110.make_models(r110.SEEDS[0])[0][r110.NATIVE]
    assert r110.cold_prediction_parity(model,data,None)==0


def test_R110_public_credits_are_bounded_factual_and_change_on_evidence():
    p=r110.episode("UNIT_FIT",r110.SEEDS[0],14)["public"]
    value=credit.factual_credit(p["support_slots"],p["support_rewards"],
                                p["support_actions"],p["support_context"])
    assert .05<=value<=.95
    changed=p["support_rewards"].copy()
    changed[p["support_context"]==1]*=-1
    altered=credit.factual_credit(p["support_slots"],changed,
                                  p["support_actions"],p["support_context"])
    assert .05<=altered<=.95
    assert abs(value-altered)>1e-9
    malformed=p["support_actions"].copy()
    malformed[0]=1
    with pytest.raises(ValueError):
        credit.factual_credit(p["support_slots"],p["support_rewards"],
                              malformed,p["support_context"])


def test_R110_credit_only_accepts_public_FIT_never_evaluator_or_original():
    public=r110.source("UNIT_FIT",r110.SEEDS[0])
    targets=credit.fit_credit(public)
    assert targets.shape==(176,)
    assert np.isfinite(targets).all()
    assert np.all((targets>=.05)&(targets<=.95))
    illegal=dict(public)
    illegal["private_switched"]=np.zeros(176,bool)
    with pytest.raises(PermissionError):
        credit.fit_credit(illegal)
    with pytest.raises(PermissionError):
        credit.fit_credit({"source":"UNIT_UNSEEN"})


def test_R110_gate_receives_usable_credit_gradient_without_oracle():
    torch.set_num_threads(1)
    data=_fit()
    models,_=r110.make_models(r110.SEEDS[0])
    m=models[r110.NATIVE]
    out=m(*r110._inputs(data,np.asarray([0]),r110.NATIVE,1))
    gate=m.route_for_credit_loss
    assert gate.shape==(1,) and gate.requires_grad
    target=torch.tensor([.75])
    torch.nn.functional.binary_cross_entropy(gate,target).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               and bool(torch.any(p.grad!=0)) for p in m.router.parameters())
    assert r110.credit_weight()==.25
