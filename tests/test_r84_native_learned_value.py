"""R84 native full-width factual intervention learning source and authority fences.

All pytest tests access only NEW FIT or FIT_VALIDATION factual logged worlds.
DEV/OOD_SHIFT, old R83 CALIBRATION and original reserved R83 QUAL/DEV/OOD
remain untouched until both seed-isolated original first-use jobs finish
ALL required training arms and freeze their optimizer weights.
"""
import copy

import numpy as np
import pytest
import torch

from experiments import r84_native_learned_value as r84


def test_original_precommitted_frozen_science_and_provenance():
    f=r84.frozen()
    assert r84.r83.git_blob(r84.FREEZE.read_bytes())==r84.FREEZE_BLOB
    assert f["stage"]=="IMMUTABLE_PROSPECTIVE_FREEZE_BEFORE_R84_SOURCE_TESTS_CI_OR_FIRST_USE"
    assert f["parent"]["R83_exact_SHA"]==r84.PARENT
    assert f["parent"]["R83_freeze_git_blob"]==r84.r83.FREEZE_BLOB
    assert f["parent"]["R83_RESERVED_salts_still_sealed"]==[25250211,25250311,25250411]
    assert f["source"]["new_independent_salts"]==r84.SALT
    assert set(r84.SALT.values()).isdisjoint({25250111,25250211,25250311,25250411})
    assert f["source"]["world_counts"]==r84.COUNT
    assert f["models"]["arms"]==list(r84.ARMS)
    assert f["models"]["native_parameter_count"]==16818178
    assert f["models"]["both_native_train_updates"]==1024
    assert f["models"]["grus_and_mlp_train_updates"]==4096
    assert f["evaluation"]["thresholds"]["both_seeds_required"]
    assert f["evaluation"]["thresholds"]["independent_scientific_pass"] is False
    assert all(v is False for v in f["authority"].values())


def test_fresh_fit_exposes_only_one_logged_actual_action_return():
    for seed in r84.SEEDS:
        sample=r84._factual_data("FIT",seed,count_override=12)
        again=r84._factual_data("FIT",seed,count_override=12)
        assert set(sample)=={"input","target","pivot","logged_action","count","panel","only_one_factual_return"}
        assert sample["count"]==12
        assert sample["only_one_factual_return"] is True
        assert sample["input"].shape==(12,49,13)
        assert sample["target"].shape==(12,)
        assert sample["input"].dtype==np.float32
        assert sample["target"].dtype==np.float32
        np.testing.assert_array_equal(sample["input"],again["input"])
        np.testing.assert_array_equal(sample["target"],again["target"])
        for i in range(12):
            p=int(sample["pivot"][i])
            action=int(sample["logged_action"][i])
            assert action in (0,2)
            assert p in (16,48)
            assert sample["input"][i,p,action]==1
            assert sample["input"][i,p,10]!=0
            assert sample["input"][i,p-1,1]==1
            assert np.all(sample["input"][i,:p+1,:4].sum(-1)==1)
            assert np.all(sample["input"][i,p+1:]==0)
            assert np.all(sample["input"][i,0,4:12]==0)
            # Only first factual outcome used; never generate another branch.
            world,logged=r84._world("FIT",seed,i)
            factual,ret,_=r84.r83._public_rollout(world,logged)
            assert logged==action
            assert sample["target"][i]==pytest.approx(ret,abs=1e-6)
            np.testing.assert_array_equal(sample["input"][i,:p+1],factual)
        assert np.any(sample["logged_action"]==0)
        assert np.any(sample["logged_action"]==2)


def test_current_reward_and_hidden_state_not_accessible_before_intervention():
    sample=r84._factual_data("FIT_VALIDATION",r84.SEEDS[0],count_override=4)
    for i in range(4):
        world,action=r84._world("FIT_VALIDATION",r84.SEEDS[0],i)
        p=int(sample["pivot"][i])
        x=sample["input"][i]
        assert x[p,10]==pytest.approx(
            r84.r83._public_rollout(world,action)[0][p,10])
        assert set(world.keys())=={
            "initial","start","restore","pivot","actions","innovations"}
        assert x.shape[1]==13
        assert x[p,action]==1.0
        assert not np.shares_memory(x,sample["target"])
        other=copy.deepcopy(world)
        other["innovations"][p+3:p+9]+=27.
        a,ret,_=r84.r83._public_rollout(world,action)
        b,ret_changed,_=r84.r83._public_rollout(other,action)
        np.testing.assert_array_equal(a,b)
        assert ret!=ret_changed


def test_max_width_identical_native_parameters_and_strict_prefix():
    torch.set_num_threads(1)
    models=r84.models(r84.SEEDS[0])
    assert len(models)==5
    native=models["NATIVE_FULL_VALUE"]
    blind=models["NATIVE_OBSERVATION_BLIND"]
    one=dict(native.named_parameters())
    two=dict(blind.named_parameters())
    assert one.keys()==two.keys()
    assert all(torch.equal(one[k],two[k]) for k in one)
    assert sum(p.numel() for p in native.parameters())==16818178
    assert sum(p.numel() for p in blind.parameters())==16818178
    sample=r84._factual_data("FIT",r84.SEEDS[0],count_override=2)
    input_array=sample["input"]
    blinded=r84._blind(input_array)
    assert np.all(blinded[:,:,4:12]==0)
    np.testing.assert_array_equal(blinded[:,:,:4],input_array[:,:,:4])
    np.testing.assert_array_equal(blinded[:,:,12],input_array[:,:,12])
    for arm,model in models.items():
        i=blinded if arm==r84.BLIND else input_array
        model.eval()
        with torch.no_grad():
            x=torch.from_numpy(i[:1])
            result=model(x)
            pivot=int(sample["pivot"][0])
            truncated=model(x[:,:pivot+1])
            changed=x.clone()
            changed[:,pivot+1:,:]+=50
            alt=model(changed)
        torch.testing.assert_close(
            result[:,:pivot+1],truncated,atol=1e-6,rtol=0)
        torch.testing.assert_close(
            result[:,:pivot+1],alt[:,:pivot+1],atol=1e-6,rtol=0)


def test_original_protected_splits_and_unapproved_host_fail_closed(monkeypatch):
    for forbidden in ("R83_CALIBRATION","R83_QUALIFICATION","R83_DEV_SEALED",
                      "R83_OOD_SEALED","R82_DEV","R81_DEV","R84_OLD_SCIENCE"):
        with pytest.raises(PermissionError,match="R84_OLD_SPENT_OR_R83_SEALED_SOURCE_DENIED"):
            r84._world(forbidden,r84.SEEDS[0],0)
        with pytest.raises(PermissionError,match="R84_ONLY_FIT_SOURCE_IN_TESTS_AND_TRAINING"):
            r84._factual_data(forbidden,r84.SEEDS[0],count_override=1)
    with pytest.raises(ValueError,match="R84_NONFROZEN_WORLD_INDEX_OR_SEED"):
        r84._world("FIT",r84.SEEDS[0],1024)
    with pytest.raises(ValueError,match="R84_NONFROZEN_FIT_SAMPLE_COUNT"):
        r84._factual_data("FIT",r84.SEEDS[0],1025)
    for key in ("GITHUB_ACTIONS","GITHUB_EVENT_NAME","GITHUB_HEAD_REF",
                "GITHUB_REPOSITORY","RUNNER_OS","RUNNER_ENVIRONMENT",
                "GITHUB_RUN_ATTEMPT","GITHUB_REF","R84_SEED","EXPECTED_CANDIDATE_SHA"):
        monkeypatch.delenv(key,raising=False)
    with pytest.raises(PermissionError,match="R84_FIRST_HOSTED_PR_CPU_ONLY_ADMISSION_DENIED"):
        r84.admit(r84.SEEDS[0])


def test_strict_thresholds_precommitted_before_first_use():
    t=r84.frozen()["evaluation"]["thresholds"]
    assert t["DEV_reward_native_minus_blind_min"]==.35
    assert t["DEV_reward_native_minus_strongest_trained_conventional_min"]==.20
    assert t["DEV_reward_native_minus_best_fixed_min"]==.75
    assert t["DEV_reward_native_minus_shuffled_rule_min"]==.75
    assert t["DEV_public_rule_agreement_min"]==.85
    assert t["DEV_native_reward_as_fraction_of_rule_gain_over_best_fixed_min"]==.70
    assert t["OOD_reward_native_minus_best_conventional_min"]==.10
    assert t["OOD_cold_reward_native_minus_best_conventional_min"]==.10
    assert t["DEV_oracle_regret_native_max"]==1.25
    assert t["on_negative"]=="DEV_LEARNED_INTERVENTION_NO_GO"
    assert t["on_incomplete"]=="INCOMPLETE_ENGINEERING_OR_RESOURCE_NOT_SCIENTIFIC_NO_GO"
