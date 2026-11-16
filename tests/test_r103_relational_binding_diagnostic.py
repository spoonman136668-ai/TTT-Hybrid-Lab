"""R103 prospective unit-only engineering and source-identity negative controls."""
import copy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from experiments import r85_delayed_composition as r85
from experiments import r101_unknown_operator_diagnostic as r101
from experiments import r102_contradiction_evidence_diagnostic as r102
from experiments import r103_relational_binding_diagnostic as r103


def test_r103_immutable_preimplementation_freeze_and_no_science_authority():
    f=r103.frozen()
    assert r85.git_blob(r103.FREEZE.read_bytes()) == r103.FREEZE_BLOB
    assert f["stage"]=="PROSPECTIVE_FREEZE_ONLY_BEFORE_IMPLEMENTATION_AND_FIRST_USE"
    assert f["lineage"]["base_parent_R102_sealed_SHA"]==r103.PARENT
    assert f["lineage"]["prior_R102_original_scored_SHA"]==r103.R102_SPENT
    assert f["lineage"]["prior_R102_scored_cohorts_consumed"] is True
    assert not f["lineage"]["prior_R102_scored_source_or_weights_reused"]
    assert set(r103.SEEDS).isdisjoint(set(r101.SEEDS+r102.SEEDS))
    assert set(f["source"]["salts"].values()).isdisjoint(
        set(r102.frozen()["source"]["salts"].values()))
    assert len(r103.ARMS)==11
    assert f["training"]["steps_per_arm"]==3072
    assert f["training"]["batch_size"]==16
    assert f["source"]["support_interventions_total"]==16
    assert not f["authority"]["protected_scored_first_use"]
    assert not f["authority"]["GPU_first_use"]
    assert not f["authority"]["FORGE_integration"]


@pytest.mark.parametrize("panel",tuple(r103.PANEL_COUNTS))
def test_every_original_scored_and_training_source_remains_fenced(panel):
    with pytest.raises(PermissionError,match="R103_PROTECTED_ORIGINAL_PHYSICAL_SOURCE_NOT_ADMITTED"):
        r103._world(panel,r103.SEEDS[0],0,"a"*40)
    with pytest.raises(PermissionError,match="R103_PROTECTED_ORIGINAL_FIRST_USE_NOT_AUTHORIZED"):
        r103.original_ci(r103.SEEDS[0])


@pytest.mark.parametrize("panel",("R102_DEV_UNKNOWN","R101_FIT","UNIT_ONLY","DEV","SOURCE"))
def test_prior_spent_and_unregistered_physical_source_never_accessible(panel):
    with pytest.raises(PermissionError,match="R103_PRIOR_SPENT_OR_UNKNOWN_PANEL_DENIED"):
        r103._world(panel,r103.SEEDS[0],0)
    with pytest.raises(PermissionError,match="R103_PROTECTED_ORIGINAL_FIRST_USE_NOT_AUTHORIZED"):
        r103._admit(seed=r103.SEEDS[0])


@pytest.mark.parametrize("family",r103.KNOWN+r103.UNSEEN)
def test_operator_truth_evaluator_only_deterministic_and_valid(family):
    a=np.asarray([-1,1,-1,1],dtype=np.int64)
    x=r103._operator_truth(a,(0,2,3),family)
    assert x in (-1,1)
    assert x==r103._operator_truth(a.copy(),(0,2,3),family)
    with pytest.raises(ValueError,match="R103_INVALID_PRIVATE_PHYSICS"):
        r103._operator_truth(np.asarray([1,0,1,-1]),(0,1,2),family)


def test_new_unknown_families_are_distinct_from_all_prior_familiar_families():
    audit=r103.non_scoring_design_audit()
    assert audit["original_scored_source_opened"] is False
    assert audit["prior_scored_cohorts_reused"] is False
    assert audit["independent_scientific_pass"] is False
    assert audit["nearest_known_truth_hamming"]=={
        "A_XNOR_B_AND_C":2,"A_XNOR_B_OR_C":2}
    assert audit["unseen_pair_hamming"]==4


@pytest.mark.parametrize("panel",("DEV_UNKNOWN","OOD_SHIFT","DELAYED_COLD","REVERSAL"))
def test_unseen_family_cannot_be_deduced_from_public_pivot_or_query_action(panel):
    audit=r103.non_scoring_design_audit()
    for seed in r103.SEEDS:
        cells=audit["public_pivot_action_family_balance"][panel][str(seed)]
        assert len(cells)==(2 if panel=="DELAYED_COLD" else 4)
        for group,distribution in cells.items():
            assert set(distribution)==set(r103.UNSEEN)
            assert len(set(distribution.values()))==1
            assert min(distribution.values())>0
    if panel=="REVERSAL":
        for index in (0,1,2,3,127):
            assert (r103.family_for_panel(panel,index,0)!=
                    r103.family_for_panel(panel,index,16))


def test_original_hidden_rule_public_pivot_leak_is_rejected(monkeypatch):
    original=r103.family_for_panel
    def leaky(panel,index,event=16):
        if panel=="DEV_UNKNOWN":
            return r103.UNSEEN[index%2]
        return original(panel,index,event)
    monkeypatch.setattr(r103,"family_for_panel",leaky)
    with pytest.raises(RuntimeError,match="R103_PUBLIC_TIME_ACTION_OPERATOR_IDENTITY_LEAK"):
        r103.non_scoring_design_audit()


def test_non_scoring_design_audit_cannot_open_any_protected_physical_world(monkeypatch):
    def no_physics(*args,**kwargs):
        raise AssertionError("R103_SCIENTIFIC_FIRST_USE_OPENED")
    monkeypatch.setattr(r103,"_world",no_physics)
    report=r103.non_scoring_design_audit()
    assert report["original_scored_source_opened"] is False


@pytest.mark.parametrize("index",(0,7,9,10))
def test_fresh_unit_physics_has_sixteen_actual_factual_returns_and_no_private_law(index):
    fixture=r103.unit_fixture(index)
    public,evaluator=fixture["public"],fixture["evaluator_only"]
    assert public["support_slots"].shape==(16,5,13)
    assert public["support_rewards"].shape==(16,)
    assert public["support_actions"].tolist()==[
        r85.CALIBRATE if j%2==0 else r85.EXPLOIT_POS for j in range(16)]
    assert public["query0"].shape==public["query2"].shape==(5,13)
    assert np.isfinite(public["support_rewards"]).all()
    np.testing.assert_array_equal(public["query0"][:4],public["query2"][:4])
    assert "private_law" not in public and "operator" not in public
    assert evaluator["operator"]==(r103.KNOWN+r103.UNSEEN)[index]
    assert evaluator["private_law"] in (-1,1)
    again=r103.unit_fixture(index)
    np.testing.assert_array_equal(public["support_rewards"],
                                  again["public"]["support_rewards"])
    np.testing.assert_array_equal(public["support_slots"],
                                  again["public"]["support_slots"])


def test_public_action_time_stratified_outcome_shuffle_is_reproducible():
    public=r103.unit_fixture(10)["public"]
    a=r103.public_shuffled_order(public["support_actions"],
                                  public["support_slots"],10)
    b=r103.public_shuffled_order(public["support_actions"],
                                  public["support_slots"],10)
    np.testing.assert_array_equal(a,b)
    assert len(a)==16 and not np.any(a==np.arange(16))
    np.testing.assert_array_equal(public["support_actions"][a],
                                  public["support_actions"])
    np.testing.assert_array_equal(public["support_slots"][a,-1,12],
                                  public["support_slots"][:,-1,12])


def test_native_relational_online_write_and_query_read_have_actual_gradients():
    torch.set_num_threads(1)
    from experiments import r95_learned_relation_diagnostic as r95
    fixture=r103.unit_fixture(9)
    p=fixture["public"]
    core=r95._native(r103.SEEDS[0])
    torch.manual_seed(r103.SEEDS[0]+103)
    model=r103.NativeRelationalValue(core,"gated")
    x=torch.from_numpy(p["support_slots"][None])
    r=torch.from_numpy(p["support_rewards"][None])
    q=torch.from_numpy(p["query0"][None])
    value=model(x,r,q)
    assert value.shape==(1,) and torch.isfinite(value).all()
    loss=F.mse_loss(value,torch.tensor([fixture["evaluator_only"]["return0"]]))
    loss.backward()
    for module in (model.memory.public,model.memory.key,model.memory.value,
                   model.memory.pred,model.memory.gate,model.projection):
        assert any(t.grad is not None and bool(torch.isfinite(t.grad).all())
                   and float(t.grad.abs().sum())>0
                   for t in module.parameters()),type(module)


def test_same_initial_native_core_and_equal_relational_adapter_capacity():
    torch.set_num_threads(1)
    models=r103.make_models(r103.SEEDS[0])
    assert set(models)==set(r103.ARMS)
    adapter=lambda m:sum(p.numel() for n,p in m.named_parameters()
                         if p.requires_grad and not n.startswith("core."))
    n=adapter(models[r103.NATIVE]); e=adapter(models[r103.EXTERNAL])
    assert abs(n-e)<=.1*max(n,e)
    for arm in (r103.UNGATED,r103.SHUFFLED,r103.ZERO,r103.RESET):
        assert adapter(models[arm])==n


def test_scalar_and_all_conventional_models_keep_factual_unit_only_interface():
    torch.set_num_threads(1)
    f=r103.unit_fixture(10)
    models=r103.make_models(r103.SEEDS[0])
    for arm in r103.ARMS:
        val=r103.unit_value(models[arm],f)
        assert val.shape==(1,),arm
        assert bool(torch.isfinite(val).all()),arm
    with torch.no_grad():
        static=models[r103.STATIC]
        p=f["public"]
        slots=torch.from_numpy(p["support_slots"][None])
        rewards=torch.from_numpy(p["support_rewards"][None])
        q=torch.from_numpy(p["query0"][None])
        v=static(slots,rewards,q)
        # Static truly cannot use factual support outcome changes.
        assert torch.equal(v,static(slots,rewards*0+1000,q))
