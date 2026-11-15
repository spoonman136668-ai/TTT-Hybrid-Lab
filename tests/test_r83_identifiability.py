"""R83 information-identifiability: only fresh CALIBRATION is accessible here.

The original R82 first-scored DEV/OOD and R83 future qualification panels
must never be used by an R83 unit test. No PyTorch model trained.
"""
import copy

import numpy as np
import pytest

from experiments import r83_identifiability as r83


def test_prospective_original_freeze_keeps_next_science_sealed():
    z=r83.frozen()
    assert r83.git_blob(r83.FREEZE.read_bytes())==r83.FREEZE_BLOB
    assert z["stage"]=="PROSPECTIVE_FREEZE_AFTER_R82_NO_GO_BEFORE_NEW_SOURCE_OR_EXECUTION"
    assert z["parent"]["R82_SHA"]==r83.PARENT
    assert z["parent"]["R82_both_seed_result"]=="DEV_ACTIVE_POLICY_NO_GO"
    assert z["source"]["used_now"].startswith("CALIBRATION ONLY")
    assert z["source"]["seeds"]==list(r83.SEEDS)
    assert z["source"]["worlds_per_seed"]==512
    assert z["gates"]["independent_scientific_pass"] is False
    assert not any(z["authority"].values())


def test_physical_two_actual_interventions_share_all_public_history_before_decision():
    for idx in (0,1,2,3):
        case=r83.calibration_world(r83.SEEDS[0],idx)
        a=case["public_prefix_calibrate"]
        b=case["public_prefix_exploit"]
        pivot=case["pivot"]
        assert a.shape==b.shape==(pivot+1,13)
        np.testing.assert_array_equal(a[:pivot],b[:pivot])
        np.testing.assert_array_equal(a[pivot,4:13],b[pivot,4:13])
        assert a[pivot,0]==1
        assert b[pivot,2]==1
        assert a[pivot,10]==pytest.approx(case["cue"])
        assert b[pivot,10]==pytest.approx(case["cue"])
        assert a[pivot-1,1]==1
        assert np.all(a[:,:4].sum(-1)==1)
        assert np.all(b[:,:4].sum(-1)==1)
        assert case["true_law_private"] in (-1,1)
        assert np.isfinite(case["r0_evaluator_only"])
        assert np.isfinite(case["r2_evaluator_only"])


def test_public_information_does_not_change_with_future_physical_perturbation():
    f=r83.frozen()
    rng=np.random.default_rng(321)
    pivot=r83.EARLY
    w={
        "initial":1,
        "start":27,
        "restore":40,
        "pivot":pivot,
        "actions":rng.choice([1,2,3],size=64),
        "innovations":rng.standard_normal((64,4))}
    w["actions"][pivot-1]=1
    w["actions"][pivot+1:pivot+9]=2
    a,ret,_=r83._public_rollout(w,0)
    changed=copy.deepcopy(w)
    changed["innovations"][pivot+3:pivot+9,:]+=50.
    b,ret_changed,_=r83._public_rollout(changed,0)
    np.testing.assert_array_equal(a,b)
    assert ret!=ret_changed


def test_cold_and_early_public_probe_policy_uses_last_realized_observation():
    for index in (0,1):
        x=r83.calibration_world(r83.SEEDS[1],index)
        assert x["pivot"]==(r83.EARLY if index==0 else r83.COLD)
        assert x["public_prefix_calibrate"][x["pivot"],10]==pytest.approx(x["cue"])
        assert x["public_prefix_exploit"][x["pivot"],10]==pytest.approx(x["cue"])


def test_source_and_host_authority_fail_closed(monkeypatch):
    for key in ("GITHUB_ACTIONS","GITHUB_EVENT_NAME","GITHUB_HEAD_REF",
                "GITHUB_REPOSITORY","RUNNER_OS","RUNNER_ENVIRONMENT",
                "GITHUB_RUN_ATTEMPT","GITHUB_REF","EXPECTED_CANDIDATE_SHA"):
        monkeypatch.delenv(key,raising=False)
    with pytest.raises(PermissionError,match="R83_NOT_QUALIFIED_G4_FIRST_ATTEMPT_HOSTED_CPU"):
        r83.admit()
    with pytest.raises(PermissionError,match="R83_SEED_OR_SOURCE_COUNT_NOT_PRECOMMITTED"):
        r83.calibration_world(10,0)
    with pytest.raises(PermissionError,match="R83_SEED_OR_SOURCE_COUNT_NOT_PRECOMMITTED"):
        r83.calibration_world(r83.SEEDS[0],512)
    with pytest.raises(ValueError,match="R83_UNREGISTERED_ACTION"):
        r83._public_rollout({"pivot":16},99)


def test_frozen_benchmarks_are_predeclared_not_post_result_redefined():
    f=r83.frozen()
    assert f["gates"]["two_seeds_required"]
    assert f["gates"]["public_probe_sign_accuracy_min"]==.8
    assert f["gates"]["public_rule_reward_minus_best_fixed_min"]==1.5
    assert f["gates"]["public_rule_reward_minus_shuffled_min"]==1.5
    assert f["gates"]["public_rule_oracle_regret_max"]==.75
    assert f["gates"]["paired_bootstrap_resamples"]==2000
    assert f["gates"]["early_and_cold_reward_improvement_over_fixed_min"]==.8
    assert f["gates"]["never_generate_sealed_panels"]
    assert f["gates"]["classification_fail"]=="ENGINEERING_BENCHMARK_INFORMATION_NO_GO"
