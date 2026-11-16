"""R105 source-isolated native role memory: UNIT_ONLY engineering controls."""
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from experiments import r85_delayed_composition as r85
from experiments import r103_relational_binding_diagnostic as r103
from experiments import r104_operator_identifiability_diagnostic as r104
from experiments import r105_native_role_binding_diagnostic as r105


def test_r105_immutable_preimplementation_freeze_and_eight_arm_scope():
    f=r105.frozen()
    assert r85.git_blob(r105.FREEZE.read_bytes())==r105.FREEZE_BLOB
    assert f["stage"]=="PROSPECTIVELY_FROZEN_BEFORE_IMPLEMENTATION_OR_ANY_UNIT_ONLY_MODEL_FIT"
    assert f["authority"]["protected_original_scientific_first_use"] is False
    assert f["authority"]["GPU_first_use"] is False
    assert f["authority"]["merge"] is False
    assert f["authority"]["FORGE_integration"] is False
    assert len(r105.ARMS)==8 and len(r105.TRAINABLE)==7
    assert f["training"]["steps_per_trainable_arm"]==512
    assert f["data"]["episode_support_count"]==16
    assert len(set(r105.SEEDS)&set(r103.SEEDS))==0
    assert len(set(r105.SEEDS)&set(r104.SEEDS))==0


def test_r105_all_11_familiar_and_eight_unseen_rules_novelty():
    out=r105.static_source_audit()
    assert out["train_rule_count"]==11
    assert out["unseen_rule_count"]==8
    assert min(out["nearest_known_Hamming"].values())>=2
    assert min(out["nearest_unseen_Hamming"].values())>=4
    assert out["original_scored_opened"] is False
    assert not out["scientific_pass"]


@pytest.mark.parametrize("panel",tuple(r105.PANELS))
def test_private_mask_balanced_within_each_public_pivot_stratum(panel):
    fam=r105.UNSEEN if panel=="UNIT_UNSEEN" else r105.TRAIN
    groups={(m,p):0 for m in fam for p in (0,1)}
    for j in range(r105.PANELS[panel]):
        groups[(fam[j%len(fam)],(j//len(fam))%2)]+=1
    assert set(groups.values())=={(r105.PANELS[panel]//len(fam))//2}


@pytest.mark.parametrize("panel",("UNIT_FIT","UNIT_VALID","UNIT_UNSEEN"))
def test_unit_physics_produces_actual_physical_factual_consequence(panel):
    seed=r105.SEEDS[0]
    out=r105.episode(panel,seed,0)
    p,e=out["public"],out["evaluator_only"]
    assert p["support_slots"].shape==(16,5,13)
    assert p["support_rewards"].shape==(16,)
    assert p["support_actions"].tolist()==[
        r85.CALIBRATE if i%2==0 else r85.EXPLOIT_POS for i in range(16)]
    assert p["support_rewards_shuffled"].shape==(16,)
    assert "private_law" not in p and "mask" not in p
    assert e["mask"] in (r105.UNSEEN if panel=="UNIT_UNSEEN" else r105.TRAIN)
    assert e["optimal_action"]==(
        r85.CALIBRATE if e["private_law"]>0 else r85.EXPLOIT_POS)
    if panel=="UNIT_UNSEEN":
        assert "query0" in p and "query2" in p and "target" not in p
        np.testing.assert_array_equal(p["query0"][:4],p["query2"][:4])
    else:
        assert "query" in p and "target" in p and "query0" not in p


def test_r105_simulator_is_own_new_unit_source_never_old_scored(monkeypatch):
    def old_source(*args,**kwargs):
        raise AssertionError("PROTECTED_OR_OLD_SOURCE_REOPENED")
    monkeypatch.setattr(r103,"_world",old_source)
    monkeypatch.setattr(r104,"_unit_world",old_source)
    out=r105.episode("UNIT_UNSEEN",r105.SEEDS[0],9)
    assert out["evaluator_only"]["mask"]==r105.UNSEEN[1]
    with pytest.raises(PermissionError,match="R105_ORIGINAL_OR_UNREGISTERED_WORLD_DENIED"):
        r105._unit_world("DEV_UNKNOWN",r105.SEEDS[0],0,0,r105.UNSEEN[0])
    with pytest.raises(PermissionError,match="R105_PROTECTED_OR_UNFROZEN_EPISODE_DENIED"):
        r105.episode("FIT",r105.SEEDS[0],0)


def test_hidden_truth_effects_only_reward_not_preaction_public_observation():
    a=r105._unit_world("UNIT_UNSEEN",r105.SEEDS[0],0,0,r105.UNSEEN[0])
    b=dict(a)
    b["law_a"]=-a["law_a"]
    for choice in (r85.CALIBRATE,r85.EXPLOIT_POS):
        xa,ra=r105._view(a,choice)
        xb,rb=r105._view(b,choice)
        np.testing.assert_array_equal(xa,xb)
        assert abs(ra-rb)>1.


def test_native_and_role_aware_conventional_observe_same_factor_roles():
    torch.set_num_threads(1)
    fixture=r105.episode("UNIT_FIT",r105.SEEDS[0],0)["public"]
    slots=torch.from_numpy(fixture["support_slots"][None])
    rewards=torch.from_numpy(fixture["support_rewards"][None])
    query=torch.from_numpy(fixture["query"][None])
    native=r105.NativeRoleKeyValue()
    gru=r105.RoleHistoryValue("gru")
    mlp=r105.RoleHistoryValue("mlp")
    for model in (native,gru,mlp):
        out=model(slots,rewards,query)
        assert out.shape==(1,) and bool(torch.isfinite(out).all())
        loss=F.mse_loss(out,torch.tensor([fixture["target"]]))
        loss.backward()
        assert any(p.grad is not None and bool(torch.isfinite(p.grad).all())
                   and float(p.grad.abs().sum())>0 for p in model.parameters())


def test_all_seven_fresh_models_have_matching_native_role_initialization():
    torch.set_num_threads(1)
    models=r105.make_models(r105.SEEDS[0])
    assert set(models)==set(r105.TRAINABLE)
    base=dict(models[r105.NATIVE].named_parameters())
    for arm in (r105.SHUFFLED,r105.ZERO,r105.NO_ROLE):
        for k,v in models[arm].named_parameters():
            assert torch.equal(v,base[k])


def test_all_models_genuine_unit_only_forward_and_ablation_preserves_shape():
    torch.set_num_threads(1)
    out=r105.episode("UNIT_FIT",r105.SEEDS[0],1)["public"]
    data={k:np.asarray([v]) for k,v in out.items()}
    models=r105.make_models(r105.SEEDS[0])
    for name,model in models.items():
        x,y,q=r105._model_input(data,np.array([0]),name)
        assert x.shape==(1,16,5,13)
        assert q.shape==(1,5,13)
        assert y.shape==(1,16)
        if name==r105.ZERO:assert not torch.any(y)
        if name==r105.NO_ROLE:
            assert not torch.any(x[:,:,:4,8]) and not torch.any(q[:,:4,8])
        val=model(x,y,q)
        assert val.shape==(1,) and bool(torch.isfinite(val).all())


def test_frozen_thresholds_exact_mapping_not_post_result_tuned():
    f=r105.frozen()
    mock={arm:{"unknown_action_accuracy":.5,"realized_reward_mean":.2,
               "oracle_reward_gap":.3} for arm in r105.ARMS}
    criteria=r105.checks(mock)
    assert set(criteria["checks"])==set(f["evaluation"]["checks"])
    assert criteria["all_unit_only_checks_pass"] is False
    assert criteria["scientific_pass"] is False
    mock[r105.NATIVE]["unknown_action_accuracy"]=.95
    mock[r105.BAYES]["unknown_action_accuracy"]=.90
    criteria=r105.checks(mock)
    assert all(criteria["checks"].values()) and criteria["all_unit_only_checks_pass"]
    assert not criteria["scientific_pass"]


def test_original_science_and_protected_gpu_never_authorized():
    with pytest.raises(PermissionError,match="R105_ORIGINAL_PROTECTED_SOURCE_DENIED"):
        r105.source("FIT",r105.SEEDS[0])
    with pytest.raises(PermissionError,match="R105_UNREGISTERED_NEW_MODEL_SEED"):
        r105.make_models(1003011)
    with pytest.raises(TimeoutError,match="R105_FROZEN_UNIT_ONLY_240S_SEED_CAP_EXCEEDED"):
        r105.source("UNIT_FIT",r105.SEEDS[0],deadline=-1)
