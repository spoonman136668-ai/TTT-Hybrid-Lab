"""F8-R69 preregistered consequence-aware causal probe task admission.

Checks critical source custody, hidden-law access isolation, real action costs,
paired control budgets and full frozen results (without asserting a win).
"""
import hashlib
import json

import numpy as np
import pytest

from experiments import p85mr_f8r69_causal_probe_task as r69


def test_source_freeze_and_first_use_boundary():
    f=r69.locked()
    assert r69.git_blob(r69.FREEZE.read_bytes())==r69.FREEZE_BLOB
    assert f["parent"]["R66_SHA"]==r69.PARENT
    assert f["parent"]["R67_result"]=="DEV_NO_GO"
    assert f["parent"]["R68_result"]=="DEV_NO_GO"
    assert f["metrics"]["no_post_result_tuning"]
    assert f["policies"]["no_parametric_model_training"]
    assert all(v is False for v in f["authority"].values())
    for panel in ("R68_DEV","R67_DEV","R66_DEV","OOD","PROXY_NEGATIVE"):
        with pytest.raises(PermissionError,match="F8R69_PROTECTED_OR_SPENT_SOURCE_DENIED"):
            r69.worlds(panel)
    with pytest.raises(PermissionError,match="F8R69_ORIGINAL_EXTERNAL_FIRST_USE_DENIED"):
        r69.deny_external_scientific_first_use()


def test_fresh_balanced_causal_source_and_costly_selected_interventions():
    panels={panel:r69.worlds(panel) for panel in r69.COUNTS}
    for panel,worlds in panels.items():
        assert len(worlds)==r69.COUNTS[panel]
        assert sum(w["switched_evaluator_only"] for w in worlds)==len(worlds)//2
        assert sum(w["initial_sign_evaluator_only"]==1 for w in worlds)==len(worlds)//2
        for w in worlds:
            assert w["panel"]==panel
            assert w["potential_evaluator_only"].shape==(36,4,4)
            assert np.isfinite(w["potential_evaluator_only"]).all()
            assert w["latent_law_evaluator_only"].shape==(36,)
            assert 10<=w["B_start_evaluator_only"]<=13
            assert 23<=w["A_return_start_evaluator_only"]<=26
            for t in r69.PROBES:
                truth=(w["initial_sign_evaluator_only"]
                       if t!=14 or not w["switched_evaluator_only"]
                       else -w["initial_sign_evaluator_only"])
                assert w["latent_law_evaluator_only"][t]==truth
    digest=lambda w:hashlib.sha256(w["potential_evaluator_only"].tobytes()).hexdigest()
    assert set(map(digest,panels["FIT"])).isdisjoint(map(digest,panels["DEV"]))
    neg=r69.negative_proxy(panels["DEV"])
    for original,changed in zip(panels["DEV"],neg):
        assert np.array_equal(original["potential_evaluator_only"],
                              changed["potential_evaluator_only"])
        assert not np.array_equal(original["unobserved_proxy"],
                                  changed["unobserved_proxy"])
    with pytest.raises(PermissionError,match="F8R69_PROXY_REQUIRES_ORIGINAL_DEV"):
        r69.negative_proxy(panels["FIT"])


def test_no_private_law_inputs_to_nonoracle_and_selected_history_only():
    source=r69.worlds("FIT")[0]
    for policy in r69.POLICIES:
        row=r69.run_policy(policy,source)
        assert row["action_count"]==36
        assert len(row["selected_actions"])==36
        assert len(row["selected_digest"])==64
        assert np.isfinite(row["sum_goal_reward"])
        if policy in ("CAUSAL_PROBE","SPURIOUS_PROBE"):
            assert row["probe_count"]==3
            assert len(row["probe_correct"])==3
        elif policy in ("REWARD_REACTIVE","ORACLE_UPPER"):
            assert row["probe_count"]==0
    rng=np.random.default_rng(0)
    for policy in r69.POLICIES[1:]:
        with pytest.raises(PermissionError,match="F8R69_POLICY_CANNOT_RECEIVE_PRIVATE_LAW"):
            r69.choose_action(policy,0,[],rng,private_latent=1)
    with pytest.raises(PermissionError,match="F8R69_ORACLE_PRIVATE_SIGN_REQUIRED"):
        r69.choose_action("ORACLE_UPPER",0,[],rng)
    with pytest.raises(ValueError,match="F8R69_OBSERVATION_ORDER_VIOLATION"):
        r69.choose_action("RANDOM",1,[],rng)
    with pytest.raises(ValueError,match="F8R69_UNREGISTERED_POLICY"):
        r69.choose_action("FAKE_ORACLE",0,[],rng)


def test_future_selected_outcome_never_affects_prior_actions():
    source=r69.worlds("FIT")[0]
    shifted={**source,"potential_evaluator_only":
             source["potential_evaluator_only"].copy()}
    shifted["potential_evaluator_only"][35,:,0]+=50
    for arm in r69.POLICIES:
        a=r69.run_policy(arm,source)
        b=r69.run_policy(arm,shifted)
        assert a["selected_actions"][:35]==b["selected_actions"][:35]
        assert a["selected_actions"][35]==b["selected_actions"][35]


def test_full_frozen_task_admission_preserves_negative_outcomes():
    report=r69.run()
    assert report["schema"]==r69.SCHEMA
    assert report["freeze_blob"]==r69.FREEZE_BLOB
    assert report["parent"]==r69.PARENT
    assert report["source"]=={
        "FIT":64,"DEV":96,"switched_DEV":48,"protected_OOD_generated":0}
    assert set(report["scores"])==set(r69.POLICIES)
    for name,score in report["scores"].items():
        assert score["selected_action_count_total"]==96*36
        assert np.isfinite(score["mean_episode_reward"])
        assert np.isfinite(score["mean_switched_cold_return_reward"])
        assert name!="ORACLE_UPPER" or score["mean_probe_actions"]==0
    checks=report["judgment"]["checks"]
    assert len(checks)==12
    assert checks["full_action_budget_and_selected_only"]
    assert checks["original_fit_dev_nonoverlap"]
    assert checks["balanced_switches_signs"]
    assert checks["actual_hidden_phase_guaranteed_at_probe_times"]
    assert checks["proxy_negative_invariance"]
    assert checks["cold_replay_exact"]
    assert report["admission"] in (
        "TASK_ADMISSION_NO_GO","TASK_ADMISSION_PASS_SYNTHETIC_ONLY")
    assert report["admission"]=="TASK_ADMISSION_PASS_SYNTHETIC_ONLY" == (
        all(checks.values()))
    assert report["independent_scientific_pass"] is False
    assert report["trained_model_improvement"] is False
    assert report["GPU_first_use"] is False
    assert report["external_first_use"] is False
    assert report["deployment"]=="NONE"
    json.dumps(report,allow_nan=False)
    r69.retain_ci_receipt(report)
