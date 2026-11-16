"""R90 prospective tiny factual optimizer test; never score any protected world."""
import copy
import os
import json

import numpy as np
import pytest
import torch

from experiments import r90_native_fit_diagnostic as r90


def test_frozen_source_and_architecture_scope():
    f=r90.frozen()
    assert r90.r74.git_blob(r90.FREEZE.read_bytes())==r90.FREEZE_BLOB
    assert f["schema"]==r90.SCHEMA
    assert f["parent"]["R89_exact_candidate"]==r90.PARENT
    assert f["parent"]["R89_freeze_blob"]==r90.r89.FREEZE_BLOB
    assert f["parent"]["R89_original_result"]=="R89_NATIVE_ROLE_ACCESS_NO_GO_TWO_SEEDS_TWO_OF_THIRTEEN"
    assert f["seeds"]==list(r90.SEEDS)
    assert f["source"]["R90_test_fixture_salt"]==r90.SALT
    assert f["source"]["kind"]=="TEST_ONLY_DEVELOPMENT_FIXTURE_NOT_PROTECTED_SCIENCE"
    assert r90.SALT not in set(r90.r89.SALT.values())
    assert f["arms"]==list(r90.ARMS)
    assert f["training"]["steps"]==r90.STEPS
    assert f["training"]["checkpoint_indices"]==list(r90.CHECKPOINTS)
    assert f["control_geometry"]["native_core_parameter_count"]==16818178
    assert f["thresholds"]["per_arm_basic_fit_final_to_initial_MSE_ratio_max"]==.25
    assert f["thresholds"]["per_arm_basic_fit_final_MSE_absolute_max"]==4.
    assert f["thresholds"]["independent_scientific_pass"] is False
    assert all(v is False for v in f["authority"].values())


def test_original_protected_dataset_still_closed_and_full_g4_custody_denied():
    with pytest.raises(PermissionError, match="R90_NONFROZEN_TEST_SEED"):
        r90.test_only_factual(100)
    for panel in ("FIT","FIT_VALIDATION","DEV","OOD_SHIFT"):
        with pytest.raises(PermissionError, match="R89_ORIGINAL_FIRST_USE_SOURCE_NOT_ADMITTED"):
            r90.r89._world(panel,r90.r89.SEEDS[0],0)
    for old in ("R88_DEV","R85_CALIBRATION","R89_DEV"):
        with pytest.raises(PermissionError,match="R89_OLD_SPENT_OR_SEALED_PANEL_DENIED"):
            r90.r89._world(old,r90.r89.SEEDS[0],0)
    if os.getenv("GITHUB_HEAD_REF")!=r90.BRANCH:
        with pytest.raises(PermissionError,match="R90_ORIGINAL_EXISTING_G4_ONLY"):
            r90._custody()


def test_four_true_factual_worlds_use_no_hidden_sign_or_unobserved_action():
    for seed in r90.SEEDS:
        first=r90.test_only_factual(seed)
        second=r90.test_only_factual(seed)
        assert first["input"].shape==(4,5,13)
        assert first["factual_return"].shape==(4,)
        assert first["input"].dtype==torch.float32
        assert first["logged_actions"]==[r90.r85.CALIBRATE,r90.r85.EXPLOIT_POS]*2
        assert first["TEST_ONLY"] is True
        assert first["no_private_law_or_unchosen_rewards"] is True
        torch.testing.assert_close(first["input"],second["input"],rtol=0,atol=0)
        torch.testing.assert_close(first["factual_return"],second["factual_return"],rtol=0,atol=0)
        assert bool(torch.isfinite(first["factual_return"]).all())
        for i in range(4):
            row=first["input"][i]
            assert row[-1,first["logged_actions"][i]]==1
            assert bool(torch.all(row[:4,5]==1)) and bool(torch.all(row[:4,11]==1))
            assert set((row[:4,9]*4).round().to(torch.int64).tolist())=={1,2,3,4}
            assert bool(torch.all(row[:4,12]<row[-1,12]))


def test_future_physical_effect_never_changes_legal_prior_observation():
    seed=r90.SEEDS[0]
    for i in range(4):
        rng=np.random.default_rng(np.random.SeedSequence([r90.SALT,seed,i]))
        w=r90.r85._make_synthetic_world(i%r90.r85.N,rng)
        alt=copy.deepcopy(w)
        p=w["pivot"]
        alt["innovations"][p+1:p+9,0]+=50
        action=r90.r85.CALIBRATE if i%2==0 else r90.r85.EXPLOIT_POS
        a,ra=r90.r89._view(w,action)
        b,rb=r90.r89._view(alt,action)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(r90.r88._public_slots(a),r90.r88._public_slots(b))
        assert abs(ra-rb)>1


def test_predeclared_diagnostic_not_proof_of_native_reasoning_or_promotion():
    f=r90.frozen()
    assert "not evidence" in f["null"]
    assert "SCIENCE" in f["thresholds"]["on_original_hard_fit_both_seeds"] or (
        f["thresholds"]["independent_scientific_pass"] is False)
    assert f["execution"]["existing_G4_only"]
    assert f["execution"]["no_new_jobs_workflows_schedules_controllers_runners"]
    assert f["execution"]["no_gpu_firstuse"]
    assert f["source"]["never_call_R85_R89_protected_world_or_pairing"]


def test_original_test_only_full_width_diagnostic_under_existing_G4():
    if os.getenv("GITHUB_HEAD_REF")!=r90.BRANCH or os.getenv("GITHUB_ACTIONS")!="true":
        pytest.skip("R90 16.8M full width one-time diagnostic only on scoped existing G4")
    result=r90.original_g4_diagnostic()
    assert result["schema"]==r90.SCHEMA
    assert result["frozen_blob"]==r90.FREEZE_BLOB
    assert result["R89_parent_SHA"]==r90.PARENT
    assert result["TEST_ONLY_no_protected_DEV_OOD_or_source"]
    assert not result["scientific_pass"]
    assert set(result["seeds"])==set(map(str,r90.SEEDS))
    assert result["classification"] in (
        "R90_REFERENCE_HARD_TINY_FIT_POSSIBLE_ONLY",
        "R90_ABLATION_ASSISTED_TINY_FIT_DIAGNOSTIC_ONLY",
        "R90_NATIVE_TINY_FIT_NO_GO_DIAGNOSTIC_ONLY")
    for seed in r90.SEEDS:
        v=result["seeds"][str(seed)]
        assert set(v["arms"])==set(r90.ARMS)
        assert v["initial_hard_surrogate_identity_probe"]["forward_identical"]
        assert v["initial_hard_surrogate_identity_probe"]["surrogate_off_router_grad_positive"]
        for arm,metric in v["arms"].items():
            assert metric["steps"]==64
            assert set(metric["train_MSE"])==set(map(str,r90.CHECKPOINTS))
            assert metric["selected_factual_worlds"]==4
            assert metric["parameters"]>0
            assert metric["final_over_initial_MSE"]>=0
            assert all(np.isfinite(list(metric["train_MSE"].values())))
