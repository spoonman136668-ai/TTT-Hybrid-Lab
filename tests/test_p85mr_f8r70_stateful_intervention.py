"""Prospectively frozen R70 stateful causal intervention task qualification."""
import hashlib
import json

import numpy as np
import pytest

from experiments import p85mr_f8r70_stateful_intervention as r70


def test_freeze_parent_no_spent_source_and_protected_firstuse():
    f=r70.locked()
    assert r70.git_blob(r70.FREEZE.read_bytes())==r70.FREEZE_BLOB
    assert f["parent"]["R66_SHA"]==r70.PARENT
    assert f["parent"]["R69_result"]=="TASK_ADMISSION_NO_GO"
    assert f["parent"]["no_spent_R64_through_R69_scored_cohorts"]
    assert f["metrics"]["never_tune_on_scored_DEV"]
    assert all(v is False for v in f["authority"].values())
    assert r70.PROBES==(0,16,32)
    for panel in ("R69_DEV","R68_DEV","R67_DEV","R66_DEV","OOD","PROXY_NEGATIVE"):
        with pytest.raises(PermissionError,match="F8R70_OLD_OR_PROTECTED_SOURCE_DENIED"):
            r70.worlds(panel)
    with pytest.raises(PermissionError,match="F8R70_PROTECTED_ORIGINAL_FIRSTUSE_DENIED"):
        r70.deny_protected_firstuse()


def test_new_source_balances_causal_worlds_and_preserves_unchosen_boundaries():
    sources={panel:r70.worlds(panel) for panel in r70.COUNTS}
    digest=lambda w:hashlib.sha256(
        w["potential_innovations_evaluator_only"].tobytes()).hexdigest()
    assert set(map(digest,sources["FIT"])).isdisjoint(map(digest,sources["DEV"]))
    for panel,worlds in sources.items():
        assert len(worlds)==r70.COUNTS[panel]
        assert sum(w["switched_evaluator_only"] for w in worlds)==len(worlds)//2
        assert sum(w["initial_sign_evaluator_only"]==1 for w in worlds)==len(worlds)//2
        for world in worlds:
            assert world["panel"]==panel
            assert world["latent_law_evaluator_only"].shape==(40,)
            assert world["potential_innovations_evaluator_only"].shape==(40,4,4)
            assert world["unobserved_nuisance_proxy"].shape==(40,4)
            assert np.isfinite(world["potential_innovations_evaluator_only"]).all()
            assert 12<=world["B_start_evaluator_only"]<=15
            assert 28<=world["A_return_start_evaluator_only"]<=31
            expected=(world["initial_sign_evaluator_only"],
                      (-world["initial_sign_evaluator_only"] if
                       world["switched_evaluator_only"] else
                       world["initial_sign_evaluator_only"]),
                      world["initial_sign_evaluator_only"])
            assert tuple(int(world["latent_law_evaluator_only"][t])
                         for t in r70.PROBES)==expected
    negative=r70.negative_proxy(sources["DEV"])
    for original,changed in zip(sources["DEV"],negative):
        assert np.array_equal(original["latent_law_evaluator_only"],
                              changed["latent_law_evaluator_only"])
        assert np.array_equal(original["potential_innovations_evaluator_only"],
                              changed["potential_innovations_evaluator_only"])
        assert not np.array_equal(original["unobserved_nuisance_proxy"],
                                  changed["unobserved_nuisance_proxy"])
    with pytest.raises(PermissionError,match="F8R70_PROXY_REQUIRES_ORIGINAL_DEV"):
        r70.negative_proxy(sources["FIT"])


def test_calibration_changes_real_future_consequence_not_just_information():
    worlds=r70.worlds("FIT")[:4]
    contrast=r70.transition_contrast(worlds)
    assert contrast["controlled_worlds"]==4
    assert contrast["mean_future_reward_do_calibrate_minus_do_sham"]==pytest.approx(.8)
    assert contrast["min_future_reward_contrast"]==pytest.approx(.8)
    for world in worlds:
        measured,cal_state=r70.transition(world,0,0,0)
        _,sham_state=r70.transition(world,0,1,0)
        assert cal_state==int(world["latent_law_evaluator_only"][0])
        assert sham_state==0
        assert measured[0]<0  # Costly physical intervention.
        assert measured.shape==(4,)
        law=int(world["latent_law_evaluator_only"][1])
        common=2 if law>0 else 3
        after_cal,_=r70.transition(world,1,common,cal_state)
        after_sham,_=r70.transition(world,1,common,sham_state)
        assert after_cal[0]-after_sham[0]==pytest.approx(.8)


def test_strict_selected_history_and_private_law_isolation():
    rng=np.random.default_rng(0)
    for name in r70.POLICIES[:-1]:
        with pytest.raises(PermissionError,match="F8R70_NONORACLE_PRIVATE_LAW_DENIED"):
            r70.choose_action(name,0,[],rng,private_law=1)
    with pytest.raises(PermissionError,match="F8R70_ORACLE_REQUIRES_PRIVATE_LAW"):
        r70.choose_action("ORACLE_DYNAMIC_PRIVILEGED",0,[],rng)
    with pytest.raises(ValueError,match="F8R70_FACTUAL_HISTORY_LENGTH_MISMATCH"):
        r70.choose_action("RANDOM",1,[],rng)
    with pytest.raises(ValueError,match="F8R70_UNREGISTERED_POLICY"):
        r70.choose_action("INVALID_ORACLE",0,[],rng)
    world=r70.worlds("FIT")[0]
    for name in r70.POLICIES:
        row=r70.run_policy(name,world)
        assert row["action_count"]==40
        assert len(row["selected_actions"])==40
        assert len(row["selected_observation_sha256"])==64
        assert len(row["physical_alignment_trajectory_sha256"])==64
        assert np.isfinite(row["mean_episode_reward"])
        if name in ("CALIBRATE_CAUSAL","BLIND_CALIBRATE_REACTIVE"):
            assert row["calibration_count"]==3
            assert all(row["selected_actions"][t]==0 for t in r70.PROBES)
        elif name=="NO_CALIBRATE_REACTIVE":
            assert row["calibration_count"]==0
        elif name=="SHAM_PROXY":
            assert row["sham_count"]==3


def test_unobserved_future_selected_effect_cannot_change_earlier_choices():
    original=r70.worlds("FIT")[0]
    altered={**original,
             "potential_innovations_evaluator_only":
             original["potential_innovations_evaluator_only"].copy()}
    altered["potential_innovations_evaluator_only"][39,:,0]+=50.
    for name in r70.POLICIES:
        before=r70.run_policy(name,original)
        after=r70.run_policy(name,altered)
        assert before["selected_actions"]==after["selected_actions"]


def test_full_frozen_R70_task_admission_records_any_negative():
    result=r70.run()
    assert result["schema"]==r70.SCHEMA
    assert result["freeze_blob"]==r70.FREEZE_BLOB
    assert result["parent"]==r70.PARENT
    assert result["source"]=={
        "FIT":64,"DEV":96,"switched_DEV":48,"protected_external_OOD":0}
    assert set(result["scores"])==set(r70.POLICIES)
    assert result["controlled_physical_transition"]["controlled_worlds"]==96
    for arm,score in result["scores"].items():
        assert score["total_selected_action_count"]==96*40
        assert np.isfinite(score["mean_episode_reward"])
        assert np.isfinite(score["mean_switched_postreturn_reward"])
        if arm in ("CALIBRATE_CAUSAL","BLIND_CALIBRATE_REACTIVE"):
            assert score["mean_calibration_count"]==3
        if arm=="SHAM_PROXY":
            assert score["mean_sham_count"]==3
    checks=result["judgment"]["checks"]
    assert len(checks)==13
    assert checks["exact_selected_action_budget"]
    assert checks["fresh_source_nonoverlap_balanced"]
    assert checks["calibration_guaranteed_all_three_regimes"]
    assert checks["physical_intervention_future_effect"]
    assert checks["independent_nuisance_proxy_invariance"]
    assert checks["exact_cold_replay"]
    assert (result["admission"]=="TASK_ADMISSION_PASS_SYNTHETIC_ONLY")==all(checks.values())
    assert result["independent_scientific_pass"] is False
    assert result["trained_native_model_gain"] is False
    assert result["external_scientific_firstuse"] is False
    assert result["GPU_firstuse"] is False
    assert result["deployment"]=="NONE"
    json.dumps(result,allow_nan=False)
    r70.retain_ci_receipt(result)
