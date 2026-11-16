"""R102 UNIT_ONLY negative controls. Absolutely no original FIT/DEV first use."""
import numpy as np
import pytest
import torch

from experiments import r85_delayed_composition as r85
from experiments import r101_unknown_operator_diagnostic as r101
from experiments import r102_contradiction_evidence_diagnostic as r102


def test_r102_freeze_first_and_distinct_from_R101_spent_cohort():
    f=r102.frozen()
    assert r85.git_blob(r102.FREEZE.read_bytes())==r102.FREEZE_BLOB
    assert f["stage"]=="PROSPECTIVE_FREEZE_ONLY_BEFORE_IMPLEMENTATION_AND_FIRST_USE"
    assert f["lineage"]["parent_R101_sealed_SHA"]==r102.PARENT
    assert f["lineage"]["prior_R101_original_scored_SHA"]==r102.R101_SPENT
    assert not f["lineage"]["prior_scored_sources_or_weights_reused"]
    assert set(f["source"]["seeds"]).isdisjoint(r101.SEEDS)
    assert set(f["source"]["salts"].values()).isdisjoint(
        r101.frozen()["source"]["salts"].values())
    assert set(r102.UNSEEN).isdisjoint(set(r101.KNOWN+r101.UNSEEN))
    assert len(r102.ARMS)==11
    assert f["source"]["support_interventions_total"]==16
    assert f["training"]["steps_per_arm"]==3072
    assert f["training"]["batch_size"]==16
    assert not f["authority"]["protected_scored_first_use"]
    assert not f["authority"]["GPU_first_use"]
    assert not f["authority"]["merge"]
    assert not f["authority"]["FORGE_integration"]
    assert f["evaluation"]["independent_scientific_pass"] is False


@pytest.mark.parametrize("panel",tuple(r102.PANEL_COUNTS))
def test_all_original_fresh_source_panels_fenced_even_plausible_SHA(panel):
    with pytest.raises(PermissionError,match="R102_PROTECTED_SCORED_PHYSICAL_SOURCE_NOT_ADMITTED"):
        r102._world(panel,r102.SEEDS[0],0,"a"*40)


@pytest.mark.parametrize("panel",("R101_DEV_UNKNOWN","R100_SCORING","PROTECTED","UNIT_ONLY"))
def test_prior_or_unregistered_scored_physics_never_reused(panel):
    with pytest.raises(PermissionError,match="R102_OLD_SPENT_OR_NONREGISTERED_SOURCE_DENIED"):
        r102._world(panel,r102.SEEDS[0],0)
    with pytest.raises(PermissionError,match="R102_PROTECTED_ORIGINAL_FIRST_USE_NOT_AUTHORIZED"):
        r102.original_ci(seed=r102.SEEDS[0])


def test_registered_source_seeds_ranges_and_first_use_refused():
    with pytest.raises(PermissionError,match="R102_UNREGISTERED_SEED_OR_INDEX"):
        r102._world("FIT",42,0)
    with pytest.raises(PermissionError,match="R102_UNREGISTERED_SEED_OR_INDEX"):
        r102._world("DEV_UNKNOWN",r102.SEEDS[0],100000)
    with pytest.raises(PermissionError,match="R102_PROTECTED_ORIGINAL_FIRST_USE_NOT_AUTHORIZED"):
        r102._admit("original",seed=r102.SEEDS[0])


@pytest.mark.parametrize("family",r102.KNOWN+r102.UNSEEN)
def test_evaluator_only_operators_are_deterministic_and_bounded(family):
    factor=np.array([1,-1,1,-1],np.int64)
    assert r102._operator_truth(factor,(0,1,2),family) in (-1,1)
    assert r102._operator_truth(factor,(0,1,2),family)==(
        r102._operator_truth(factor.copy(),(0,1,2),family))
    with pytest.raises(ValueError,match="R102_INVALID_PRIVATE_PHYSICAL_LAW"):
        r102._operator_truth(np.array([0,1,1,-1]),(0,1,2),family)


def test_new_private_boolean_families_are_not_old_unseen_families():
    for fam in r102.UNSEEN:
        outputs=[r102._operator_truth(
            np.array([2*((j>>2)&1)-1,2*((j>>1)&1)-1,2*(j&1)-1,1]),
            (0,1,2),fam) for j in range(8)]
        assert len(set(outputs))==2
        assert any(outputs!=[
            r102._operator_truth(np.array([2*((j>>2)&1)-1,2*((j>>1)&1)-1,
                                          2*(j&1)-1,1]),(0,1,2),old)
            for j in range(8)] for old in r102.KNOWN)


@pytest.mark.parametrize("panel",("DEV_UNKNOWN","OOD_SHIFT","DELAYED_COLD","REVERSAL"))
def test_heldout_families_balanced_in_joint_public_pivot_and_action_cells(panel):
    report=r102.non_scoring_design_audit()
    assert report["original_scored_source_opened"] is False
    assert report["independent_scientific_pass"] is False
    assert report["R101_scored_cohort_reuse"] is False
    for seed in r102.SEEDS:
        cells=report["public_pivot_action_family_cell_balance"][panel][str(seed)]
        assert len(cells)==(2 if panel=="DELAYED_COLD" else 4)
        for counts in cells.values():
            assert set(counts)==set(r102.UNSEEN)
            assert len(set(counts.values()))==1
            assert min(counts.values())>0
    if panel=="REVERSAL":
        for j in (0,1,2,3,127):
            assert r102.family_for_panel(panel,j,0)!=r102.family_for_panel(panel,j,16)


def test_original_public_family_shortcut_is_rejected_in_negative_audit(monkeypatch):
    original=r102.family_for_panel
    def malicious_parity_family(panel,index,event=16):
        if panel=="DEV_UNKNOWN":
            return r102.UNSEEN[index%2] # public EARLY/COLD
        return original(panel,index,event)
    monkeypatch.setattr(r102,"family_for_panel",malicious_parity_family)
    with pytest.raises(RuntimeError,match="R102_PRIVATE_FAMILY_PUBLIC_TIME_ACTION_LEAK"):
        r102.non_scoring_design_audit()


def test_non_scoring_ideal_feedback_identifiability_increases_6_to_16(monkeypatch):
    monkeypatch.setattr(r102,"_world",lambda *_a,**_k:
                        (_ for _ in ()).throw(AssertionError("REAL_FIT_DEV_OPENED")))
    report=r102.non_scoring_design_audit()
    assert report["original_scored_source_opened"] is False
    for fam,diagnostic in report["combinatorial_family_distinguishability"].items():
        assert fam in r102.UNSEEN
        assert 0 < diagnostic["nearest_known_hamming_distance_of_8"] <= 8
        assert diagnostic["sixteen_iid_uniform_ideal_feedback_ambiguity"] <= (
            diagnostic["six_iid_uniform_ideal_feedback_ambiguity"])
    assert not report["independent_scientific_pass"]


def test_unit_only_factual_physics_and_true_six_subset():
    f=r102.unit_fixture(0)
    p=f["public"]
    e=f["evaluator_only"]
    assert p["support_slots"].shape==(16,5,13)
    assert p["support_rewards"].shape==(16,)
    assert p["support_actions"].tolist()==[
        r85.CALIBRATE if j%2==0 else r85.EXPLOIT_POS for j in range(16)]
    assert (p["support_actions"][:6]==np.array([
        r85.CALIBRATE,r85.EXPLOIT_POS]*3)).all()
    assert p["query0"].shape==(5,13) and p["query2"].shape==(5,13)
    assert np.isfinite(p["support_rewards"]).all()
    np.testing.assert_array_equal(p["query0"][:4],p["query2"][:4])
    assert isinstance(e["private_law"],int)
    assert "private_law" not in p and "private_family" not in p
    assert set(e)>= {"return0","return2","private_law"}
    again=r102.unit_fixture(0)
    np.testing.assert_array_equal(p["support_slots"],again["public"]["support_slots"])
    np.testing.assert_array_equal(p["support_rewards"],again["public"]["support_rewards"])


def test_unit_only_outcome_shuffle_changes_pairings_but_not_public_action_time():
    p=r102.unit_fixture(1)["public"]
    a=r102.public_support_shuffle(p["support_actions"],p["support_slots"],1)
    b=r102.public_support_shuffle(p["support_actions"],p["support_slots"],1)
    np.testing.assert_array_equal(a,b)
    assert not np.any(a==np.arange(16))
    np.testing.assert_array_equal(p["support_actions"][a],p["support_actions"])
    np.testing.assert_array_equal(p["support_slots"][a,-1,12],
                                  p["support_slots"][:,-1,12])
    assert np.any(p["support_rewards"][a]!=p["support_rewards"])


def test_native_contradiction_memory_value_and_differentiable_factual_update():
    torch.set_num_threads(1)
    f=r102.unit_fixture(2)
    p=f["public"]
    core=r95_native_for_test(r102.SEEDS[0])
    torch.manual_seed(r102.SEEDS[0]+102)
    m=r102.NativeEvidenceValue(core,"gated")
    slots=torch.from_numpy(p["support_slots"][None])
    rewards=torch.from_numpy(p["support_rewards"][None])
    query=torch.from_numpy(p["query0"][None])
    target=torch.tensor([f["evaluator_only"]["return0"]],dtype=torch.float32)
    value=m(slots,rewards,query)
    assert value.shape==(1,)
    assert bool(torch.isfinite(value).all())
    loss=torch.nn.functional.mse_loss(value,target)
    loss.backward()
    assert any(param.grad is not None and
               param.grad.abs().sum()>0
               for param in m.support.gate.parameters())
    assert any(param.grad is not None and
               param.grad.abs().sum()>0
               for param in m.support.prediction.parameters())
    assert any(param.grad is not None and
               param.grad.abs().sum()>0
               for param in m.support.public.parameters())


def r95_native_for_test(seed):
    from experiments import r95_learned_relation_diagnostic as r95
    return r95._native(seed)


def test_16_and_6_native_modes_match_first_six_factual_events_not_synthetic_future():
    torch.set_num_threads(1)
    f=r102.unit_fixture(3)
    p=f["public"]
    x=torch.from_numpy(p["support_slots"][None])
    y=torch.from_numpy(p["support_rewards"][None])
    q=torch.from_numpy(p["query2"][None])
    core=r95_native_for_test(r102.SEEDS[0])
    torch.manual_seed(r102.SEEDS[0]+102)
    six=r102.NativeEvidenceValue(core,"short")
    with torch.no_grad():
        baseline=six(x,y,q)
        changed=y.clone()
        changed[:,6:]=10000.
        assert torch.allclose(six(x,changed,q),baseline)
        assert not torch.allclose(six(x[:,:6],y[:,:6],q),torch.zeros_like(baseline))
    assert baseline.shape==(1,)


def test_all_11_fresh_source_models_equal_core_and_adapter_capacity():
    torch.set_num_threads(1)
    models=r102.make_models(r102.SEEDS[0])
    assert set(models)==set(r102.ARMS)
    adap=lambda m:sum(p.numel() for name,p in m.named_parameters()
                     if p.requires_grad and not name.startswith("core."))
    na=adap(models[r102.NATIVE]);ea=adap(models[r102.EXTERNAL])
    assert abs(na-ea) <=.1*max(na,ea)
    for name in (r102.NATIVE,r102.GATE_BLIND,r102.SHUFFLED,r102.NO_OUTCOME,
                 r102.RESET,r102.SHORT):
        assert adap(models[name])==na
    assert models[r102.NATIVE].mode=="gated"
    assert models[r102.GATE_BLIND].mode=="blind"
    assert models[r102.SHORT].mode=="short"
    assert models[r102.EXTERNAL].core is not models[r102.NATIVE].core


@pytest.mark.parametrize("mode",("gru","set","flat","static"))
def test_new_16_factual_history_conventional_models_trainable_unit_only(mode):
    torch.set_num_threads(1)
    f=r102.unit_fixture(0)
    p=f["public"]
    m=r102.ConventionalEvidenceValue(mode)
    x=torch.from_numpy(p["support_slots"][None])
    y=torch.from_numpy(p["support_rewards"][None])
    q=torch.from_numpy(p["query0"][None])
    target=torch.tensor([f["evaluator_only"]["return0"]],dtype=torch.float32)
    opt=torch.optim.AdamW(m.parameters(),lr=.001)
    init={name:param.detach().clone() for name,param in m.named_parameters()}
    pred=m(x,y,q)
    assert pred.shape==(1,)
    loss=torch.nn.functional.mse_loss(pred,target)
    opt.zero_grad(set_to_none=True);loss.backward()
    assert all(param.grad is not None and torch.isfinite(param.grad).all()
               for param in m.parameters())
    opt.step()
    assert any(not torch.equal(init[name],param)
               for name,param in m.named_parameters())
