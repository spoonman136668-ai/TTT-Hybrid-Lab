"""R106 frozen eight-slot causal memory: public physics UNIT_ONLY tests."""
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from experiments import r85_delayed_composition as r85
from experiments import r104_operator_identifiability_diagnostic as r104
from experiments import r105_native_role_binding_diagnostic as r105
from experiments import r106_eight_slot_causal_memory_diagnostic as r106


def test_r106_preimplementation_freeze_and_no_scored_science_authority():
    frozen=r106.frozen()
    assert r85.git_blob(r106.FREEZE.read_bytes())==r106.FREEZE_BLOB
    assert frozen["stage"]=="PREIMPLEMENTATION_FROZEN_UNIT_ONLY_DEV_NO_PROTECTED_FIRST_USE"
    assert frozen["authority"]["protected_original_scientific_first_use"] is False
    assert frozen["authority"]["GPU_first_use"] is False
    assert frozen["authority"]["merge"] is False
    assert frozen["authority"]["FORGE_integration"] is False
    assert len(r106.ARMS)==9 and len(r106.TRAINABLE)==8
    assert frozen["training"]["steps_per_trainable_arm"]==512
    assert frozen["training"]["seed_max_CPU_wall_seconds"]==240


def test_r106_eight_heldout_boolean_laws_are_fresh_and_separate():
    d=r106.static_audit()
    assert len(d["new_heldout_masks"])==8
    assert min(d["min_hamming_from_training"].values())>=2
    assert min(d["min_heldout_pairwise_hamming"].values())>=4
    assert not set(r106.UNSEEN)&set(r106.TRAIN)
    assert not set(r106.UNSEEN)&set(r104.MASKS)
    assert not set(r106.UNSEEN)&set(r105.UNSEEN)
    assert not set(r106.SEEDS)&set(r105.SEEDS)
    assert not d["prior_scored_cohorts_reused"]
    assert not d["science_pass"]


@pytest.mark.parametrize("panel",tuple(r106.COUNTS))
def test_private_rule_balances_public_pivot_within_family(panel):
    families=r106.UNSEEN if panel=="UNIT_UNSEEN" else r106.TRAIN
    counts={(mask,pivot):0 for mask in families for pivot in (0,1)}
    for i in range(r106.COUNTS[panel]):
        counts[(families[i%len(families)],(i//len(families))%2)]+=1
    assert set(counts.values())=={r106.COUNTS[panel]//(len(families)*2)}


@pytest.mark.parametrize("panel",tuple(r106.COUNTS))
def test_r106_public_factual_evaluator_isolation(panel):
    result=r106.episode(panel,r106.SEEDS[0],0)
    p,e=result["public"],result["evaluator_only"]
    assert p["support_slots"].shape==(16,5,13)
    assert p["support_rewards"].shape==(16,)
    assert p["support_rewards_shuffled"].shape==(16,)
    assert p["support_actions"].tolist()==[
        r85.CALIBRATE if i%2==0 else r85.EXPLOIT_POS for i in range(16)]
    assert "private_mask" not in p and "private_law" not in p
    assert e["private_mask"] in (r106.UNSEEN if panel=="UNIT_UNSEEN" else r106.TRAIN)
    assert e["optimal_action"]==(
        r85.CALIBRATE if e["private_law"]>0 else r85.EXPLOIT_POS)
    if panel=="UNIT_UNSEEN":
        assert "query0" in p and "query2" in p and "target" not in p
        np.testing.assert_array_equal(p["query0"][:4],p["query2"][:4])
    else:
        assert "query" in p and "target" in p
        assert "query0" not in p and "query2" not in p


def test_private_law_only_changes_actual_reward_not_preaction_public_cues():
    a=r106._unit_world("UNIT_UNSEEN",r106.SEEDS[0],0,0,r106.UNSEEN[0])
    b=dict(a)
    b["law_a"]=-a["law_a"]
    for action in (r85.CALIBRATE,r85.EXPLOIT_POS):
        x,ra=r106._view(a,action)
        y,rb=r106._view(b,action)
        np.testing.assert_array_equal(x,y)
        assert abs(ra-rb)>1.


def test_no_prior_R103_R104_R105_original_or_unit_source_replay(monkeypatch):
    from experiments import r103_relational_binding_diagnostic as r103
    def denied(*_a,**_kw):
        raise AssertionError("R106_OLD_OR_PROTECTED_PHYSICAL_SOURCE_REOPENED")
    monkeypatch.setattr(r103,"_world",denied)
    monkeypatch.setattr(r104,"_unit_world",denied)
    monkeypatch.setattr(r105,"_unit_world",denied)
    assert r106.episode("UNIT_UNSEEN",r106.SEEDS[0],1)["evaluator_only"]["private_mask"]==r106.UNSEEN[1]
    with pytest.raises(PermissionError,match="R106_PROTECTED_OR_UNREGISTERED_PHYSICAL_SOURCE_DENIED"):
        r106._unit_world("DEV_UNKNOWN",r106.SEEDS[0],0,0,r106.UNSEEN[0])
    with pytest.raises(PermissionError,match="R106_NONFROZEN_ORIGINAL_EPISODE_DENIED"):
        r106.episode("FIT",r106.SEEDS[0],0)


def test_frozen_240second_budget_stops_before_physics_or_fit():
    with pytest.raises(TimeoutError,match="R106_FROZEN_240S_UNIT_ONLY_SEED_BUDGET_EXCEEDED"):
        r106.episode("UNIT_FIT",r106.SEEDS[0],0,deadline=-1)
    with pytest.raises(TimeoutError,match="R106_FROZEN_240S_UNIT_ONLY_SEED_BUDGET_EXCEEDED"):
        r106.source("UNIT_FIT",r106.SEEDS[0],deadline=-1)


def test_8_slot_soft_addresses_normalized_and_public_query_only():
    torch.set_num_threads(1)
    p=r106.episode("UNIT_FIT",r106.SEEDS[0],0)["public"]
    model=r106.NativeEightSlotValue()
    x=torch.from_numpy(p["support_slots"][None])
    r=torch.from_numpy(p["support_rewards"][None])
    q=torch.from_numpy(p["query"][None])
    signs=model.bind(x)
    query=model.bind(q.unsqueeze(1)).squeeze(1)
    memory=model.addresses(signs)
    qloc=model.addresses(query)
    assert memory.shape==(1,16,8) and qloc.shape==(1,8)
    assert torch.allclose(memory.sum(-1),torch.ones(1,16),atol=1e-6)
    assert torch.allclose(qloc.sum(-1),torch.ones(1),atol=1e-6)
    out=model(x,r,q)
    assert out.shape==(1,) and bool(torch.isfinite(out).all())
    loss=F.mse_loss(out,torch.tensor([p["target"]],dtype=torch.float32))
    loss.backward()
    for component in (model.local_gate,model.evidence,model.readout):
        assert any(z.grad is not None and bool(torch.isfinite(z.grad).all())
                   and float(z.grad.abs().sum())>0
                   for z in component.parameters())


def test_full_native_and_negative_controls_share_exact_fresh_weights():
    torch.set_num_threads(1)
    models=r106.make_models(r106.SEEDS[0])
    assert set(models)==set(r106.TRAINABLE)
    params=dict(models[r106.NATIVE].named_parameters())
    for name in (r106.BLIND,r106.SHUFFLED,r106.ZERO,r106.NO_ROLE):
        for k,v in models[name].named_parameters():
            assert torch.equal(v,params[k])
    assert models[r106.BLIND].mode=="blind"


def test_all_eight_arms_trainable_on_same_public_factual_unit_fixture():
    torch.set_num_threads(1)
    p=r106.episode("UNIT_FIT",r106.SEEDS[0],3)["public"]
    data={k:np.asarray([v]) for k,v in p.items()}
    models=r106.make_models(r106.SEEDS[0])
    for name,model in models.items():
        x,r,q=r106._inputs(data,np.asarray([0]),name)
        assert x.shape==(1,16,5,13)
        assert r.shape==(1,16) and q.shape==(1,5,13)
        if name==r106.ZERO:
            assert not torch.any(r)
        if name==r106.NO_ROLE:
            assert not torch.any(x[:,:,:4,8])
        v=model(x,r,q)
        assert v.shape==(1,) and bool(torch.isfinite(v).all())


def test_prespecified_R106_check_mapping_and_no_science_upgrade():
    f=r106.frozen()
    d={arm:{"unknown_action_accuracy":0.5} for arm in r106.ARMS}
    result=r106.frozen_checks(d)
    assert set(result["checks"])==set(
        f["evaluation"]["thresholds_all_three_seeds_required_for_DEV_only"])
    assert not result["all_unit_only_checks_pass"]
    assert not result["scientific_pass"]
    d[r106.NATIVE]["unknown_action_accuracy"]=.95
    d[r106.BAYES]["unknown_action_accuracy"]=.91
    result=r106.frozen_checks(d)
    assert all(result["checks"].values())
    assert result["all_unit_only_checks_pass"]
    assert not result["scientific_pass"]


def test_train_fails_closed_on_privileged_and_partial_data():
    torch.set_num_threads(1)
    model=r106.make_models(r106.SEEDS[0])[r106.GRU]
    bad={"target":np.zeros(r106.COUNTS["UNIT_FIT"],np.float32),
         "private_mask":np.ones(r106.COUNTS["UNIT_FIT"])}
    with pytest.raises(PermissionError,match="R106_UNIT_FIT_UNFROZEN_OR_PRIVILEGED"):
        r106.fit_arm(model,r106.GRU,bad,bad,
                     np.zeros((512,16),np.int64),deadline=None)


def test_source_deny_old_science_seed_and_nonunit_panel():
    with pytest.raises(PermissionError,match="R106_ORIGINAL_PROTECTED_SOURCE_DENIED"):
        r106.source("DEV_UNKNOWN",r106.SEEDS[0])
    with pytest.raises(PermissionError,match="R106_ORIGINAL_PROTECTED_SOURCE_DENIED"):
        r106.source("UNIT_FIT",r105.SEEDS[0])
    with pytest.raises(PermissionError,match="R106_UNREGISTERED_FRESH_SEED"):
        r106.make_models(r105.SEEDS[0])
