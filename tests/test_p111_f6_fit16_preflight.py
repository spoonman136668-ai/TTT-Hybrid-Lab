"""P111 16-world FIT block2 selection and full SSM-receipt negative controls."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from experiments import p109_f6_strong_control_preflight as p109
from experiments import p111_f6_fit16_preflight as p111


def make_good_fixture(steps=None, predictor_mses=None):
    survival={"guarded_voi":62,"a_opt":42,"greedy_goal":41,
              "physical_balance":21,"ssm256_model_based":53}
    survival.update(steps or {})
    episodes={
        arm:[p111.make_fixture_episode(
            arm,i,survival[arm],float(survival[arm])*.75,
            1. if arm=="ssm256_model_based" else 2.,3.)
            for i in range(16)]
        for arm in p111.ARMS
    }
    mse=predictor_mses or [1.0,1.25,1.75]
    checkpoints={}
    for seed,m in zip(p111.SEEDS,mse):
        data=f"synthetic-test-checkpoint-{seed}".encode("utf-8")
        checkpoints[seed]={
            "bytes":data,
            "manifest":p111.make_fixture_checkpoint(seed,m,2.0,data),
        }
    return episodes,checkpoints


def test_p111_exact_freeze_P110_parent_protection_and_no_F6_authority():
    f=p111.locked_protocol()
    assert p111.FREEZE_COMMIT=="3eacee07265e93d5356f080d9e4a513a979ab7b5"
    assert p111.FREEZE_BLOB=="fef0bf267904de982916f25acf2c77c2582691ee"
    assert f["parent"]["exact_precode_sha"]=="72e8c56b084903327085daa88bb21cf147cdf16b"
    assert f["static_interface"]["FIT_selection_block"]==2
    assert f["static_interface"]["worlds_per_arm"]==16
    assert f["static_interface"]["incumbent_arms"]==list(p111.CONVENTIONAL)
    assert f["synthetic_checkpoint_guard"]["train_blocks"]==[0,1]
    assert f["synthetic_checkpoint_guard"]["preheldout_lock_required"] is True
    assert f["synthetic_checkpoint_guard"]["plan_conflict"].startswith("THIS RULE REQUIRES")
    assert f["future_owner_gaps"]["real_F6_firstuse_approved"] is False
    assert f["classification"]["real_F6_effects"]==0
    assert f["classification"]["scientific_pass"] is False
    assert f["engineering"]["no_owner_dispatch"] is True
    with pytest.raises(PermissionError,match="P111_NO_REAL_F6_OR_CAL_FIRSTUSE_AUTHORITY"):
        p111.refuse_owner_cal_or_source("FIT",action=0)


def test_p111_complete_16_world_selection_is_FIT_only_and_deterministic():
    episodes,checkpoints=make_good_fixture()
    r=p111.check_fit_block2(episodes,checkpoints)
    assert r["selected_conventional_fixture"]=="ssm256_model_based"
    assert r["selected_ssm_seed_fixture_only"]==p111.SEEDS[0]
    assert r["ssm_fixture_predictor_competent"] is True
    assert r["can_claim_fixture_eligibility"] is True
    assert r["strongest_conventional_survival_fixture"]==53.
    assert r["per_arm_results"]["physical_balance"]["complete_factual_actions"]==21*16
    assert r["per_arm_results"]["ssm256_model_based"]["complete_factual_actions"]==53*16
    assert r["equal_FIT_block2_episode_opportunities"] is True
    assert r["equal_actual_factual_effects"] is False
    assert r["per_arm_results"]["ssm256_model_based"]["mean_realized_reward"]==53.*.75
    assert r["P110_owner_protocol_revision_required"] is True
    assert r["trained_real_ssm"] is False
    assert r["actual_source_attempts"]==0
    assert r["real_CAL_admitted"] is False
    assert r["scientific_pass"] is False
    assert r==p111.check_fit_block2(episodes,checkpoints)


def test_p111_p109_48_world_admission_is_not_reused_for_16_worlds():
    episodes,checkpoints=make_good_fixture()
    assert p111.check_fit_block2(episodes,checkpoints)["can_claim_fixture_eligibility"]
    with pytest.raises(ValueError,match="P109_INCOMPLETE_FIT_EPISODES"):
        p109.fit_incumbent_fixture(
            {arm:episodes[arm] for arm in p111.CONVENTIONAL},
            {},b"fixture:not-real")


def test_p111_ties_and_low_conventional_competence_fail_closed():
    episodes,checkpoints=make_good_fixture(
        {"a_opt":53,"greedy_goal":53,"ssm256_model_based":53})
    r=p111.check_fit_block2(episodes,checkpoints)
    assert r["selected_conventional_fixture"]=="ssm256_model_based"  # lower loss
    for arm in p111.ARMS:
        episodes[arm]=[p111.make_fixture_episode(arm,i,53,40.,2.,3.)
                       for i in range(16)]
    r=p111.check_fit_block2(episodes,checkpoints)
    assert r["selected_conventional_fixture"]=="a_opt"  # fixed order
    episodes,checkpoints=make_good_fixture({arm:12 for arm in p111.ARMS})
    r=p111.check_fit_block2(episodes,checkpoints)
    assert r["strongest_conventional_minimum_met_fixture"] is False
    assert r["can_claim_fixture_eligibility"] is False


def test_p111_all_three_seed_models_required_and_learned_competence_explicit():
    episodes,checkpoints=make_good_fixture()
    without=copy.deepcopy(checkpoints)
    del without[p111.SEEDS[1]]
    with pytest.raises(ValueError,match="P111_THREE_TRAIN_SEED_RECEIPTS_REQUIRED"):
        p111.check_fit_block2(episodes,without)
    wrong=copy.deepcopy(checkpoints)
    wrong[p111.SEEDS[1]]["manifest"]["seed"]=111
    with pytest.raises(ValueError,match="P111_WRONG_TRAINED_SSM_SEED"):
        p111.check_fit_block2(episodes,wrong)
    no_competence=make_good_fixture(predictor_mses=[1.7,1.8,1.9])
    r=p111.check_fit_block2(*no_competence)
    assert r["ssm_fixture_predictor_competent"] is False
    assert r["can_claim_fixture_eligibility"] is False
    assert r["selected_ssm_seed_fixture_only"]==p111.SEEDS[0]


def test_p111_no_survival_filtering_no_extra_or_CAL_worlds():
    episodes,checkpoints=make_good_fixture()
    for arm in p111.ARMS:
        episodes[arm][0]=p111.make_fixture_episode(arm,0,1,0.,0.,1.,terminated=True)
    r=p111.check_fit_block2(episodes,checkpoints)
    assert r["per_arm_results"]["physical_balance"]["complete_factual_actions"]==1+21*15
    missing=copy.deepcopy(episodes)
    missing["greedy_goal"].pop()
    with pytest.raises(ValueError,match="P111_REQUIRE_EXACTLY_16_HELDOUT_FIT_WORLDS"):
        p111.check_fit_block2(missing,checkpoints)
    dupe=copy.deepcopy(episodes)
    dupe["a_opt"][2]=copy.deepcopy(dupe["a_opt"][1])
    with pytest.raises(ValueError,match="P111_DUPLICATE_OR_UNKNOWN_HELDOUT_WORLD"):
        p111.check_fit_block2(dupe,checkpoints)
    extra=copy.deepcopy(episodes)
    extra["a_opt"].append(copy.deepcopy(extra["a_opt"][0]))
    with pytest.raises(ValueError,match="P111_REQUIRE_EXACTLY_16_HELDOUT_FIT_WORLDS"):
        p111.check_fit_block2(extra,checkpoints)
    wrong=copy.deepcopy(episodes)
    wrong["guarded_voi"][0]["split"]="CAL"
    with pytest.raises(ValueError,match="P111_FIT_BLOCK2_ONLY"):
        p111.check_fit_block2(wrong,checkpoints)
    wrong=copy.deepcopy(episodes)
    wrong["guarded_voi"][0]["block"]=1
    with pytest.raises(ValueError,match="P111_FIT_BLOCK2_ONLY"):
        p111.check_fit_block2(wrong,checkpoints)


def test_p111_wrong_credit_extra_oracle_nan_tampered_receipt_rejected():
    base,ckpt=make_good_fixture()
    bads=[
        ("action_count",0,"P111_FALSE_FACTUAL_ACTION_COUNT"),
        ("survival_steps",0,"P111_FALSE_FACTUAL_ACTION_COUNT"),
        ("terminated",False,"P111_DROPPED_EARLY_TERMINATION"),
        ("reward_total",float("nan"),"P111_INVALID_REALIZED_REWARD"),
        ("physical_loss",-1.,"P111_INVALID_PHYSICAL_LOSS"),
        ("wall_ms",0.,"P111_INVALID_WALL_COST"),
        ("source_session_sha256","0"*64,"P111_UNKNOWN_WORLD_SESSION"),
        ("receipt_ref","fixture:spoof","P111_CHANGED_EPISODE_RECEIPT"),
    ]
    for field,value,code in bads:
        rows=copy.deepcopy(base)
        rows["a_opt"][0][field]=value
        with pytest.raises(ValueError,match=code):
            p111.check_fit_block2(rows,ckpt)
    changed=copy.deepcopy(base)
    changed["a_opt"][0]["unselected_effect"]="fake_future_reward"
    with pytest.raises(ValueError,match="P111_UNKNOWN_EPISODE_FIELDS"):
        p111.check_fit_block2(changed,ckpt)
    changed=copy.deepcopy(base)
    changed["a_opt"][0]["receipt_ref"]="ghrun:unapproved"
    with pytest.raises(ValueError,match="P111_CHANGED_EPISODE_RECEIPT"):
        p111.check_fit_block2(changed,ckpt)


def test_p111_checkpoint_source_optimizer_data_and_future_split_denials():
    episode,good=make_good_fixture()
    edits=[
        ("model_source_blob","0"*40,"P111_UNTRAINED_OR_UNFROZEN_SSM"),
        ("fit_only",False,"P111_UNTRAINED_OR_UNFROZEN_SSM"),
        ("cal_or_ood_used",True,"P111_UNTRAINED_OR_UNFROZEN_SSM"),
        ("train_blocks",[1,2],"P111_UNTRAINED_OR_UNFROZEN_SSM"),
        ("train_worlds",48,"P111_UNTRAINED_OR_UNFROZEN_SSM"),
        ("train_updates",399,"P111_UNTRAINED_OR_UNFROZEN_SSM"),
        ("optimizer_steps",799,"P111_UNTRAINED_OR_UNFROZEN_SSM"),
        ("crossfit_scope","FIT_BLOCK2","P111_UNTRAINED_OR_UNFROZEN_SSM"),
        ("preheldout_locked",False,"P111_UNTRAINED_OR_UNFROZEN_SSM"),
        ("persistent_floats",512,"P111_UNTRAINED_OR_UNFROZEN_SSM"),
        ("checkpoint_sha256","0"*64,"P111_UNTRAINED_OR_UNFROZEN_SSM"),
        ("train_dataset_sha256","0"*64,"P111_UNTRAINED_OR_UNFROZEN_SSM"),
    ]
    for field,value,code in edits:
        records=copy.deepcopy(good)
        records[p111.SEEDS[0]]["manifest"][field]=value
        records[p111.SEEDS[0]]["manifest"]["receipt_ref"]=p111._checkpoint_receipt(records[p111.SEEDS[0]]["manifest"])
        with pytest.raises(ValueError,match=code):
            p111.check_fit_block2(episode,records)
    not_bound=copy.deepcopy(good)
    not_bound[p111.SEEDS[0]]["bytes"]=b"fake-different-checkpoint"
    with pytest.raises(ValueError,match="P111_UNTRAINED_OR_UNFROZEN_SSM"):
        p111.check_fit_block2(episode,not_bound)
    not_bound=copy.deepcopy(good)
    not_bound[p111.SEEDS[0]]["manifest"]["receipt_ref"]="fixture:bad"
    with pytest.raises(ValueError,match="P111_CHANGED_SSM_RECEIPT"):
        p111.check_fit_block2(episode,not_bound)


def test_p111_predictor_zero_reference_must_be_positive_and_finite():
    episode,good=make_good_fixture()
    for field,value in [
        ("zero_crossfit_mse",0.),
        ("zero_crossfit_mse",float("nan")),
        ("train_crossfit_mse",float("-inf")),
        ("train_crossfit_mse",-1.),
    ]:
        records=copy.deepcopy(good)
        records[p111.SEEDS[0]]["manifest"][field]=value
        if math.isfinite(value):
            records[p111.SEEDS[0]]["manifest"]["receipt_ref"]=p111._checkpoint_receipt(records[p111.SEEDS[0]]["manifest"])
        with pytest.raises(ValueError,match="P111_BAD_PREDICTOR_COMPETENCE"):
            p111.check_fit_block2(episode,records)


def test_p111_no_external_source_training_owner_CAL_or_promotion():
    source=Path(p111.__file__).read_text("utf-8")
    for forbidden in (
        "import causal_gym","from causal_gym","import requests",
        "import socket","subprocess","import urllib",
        "env.do(","env.reset(","torch.load(","optimizer.step(",
        "dispatch_workflow(","run_owner(","firstuse_authorize(",
    ):
        assert forbidden not in source
    assert p111.locked_protocol()["future_owner_gaps"]["real_F6_CAL_firstuse_approved"] is False
    with pytest.raises(PermissionError,match="P111_NO_REAL_F6_OR_CAL_FIRSTUSE_AUTHORITY"):
        p111.refuse_owner_cal_or_source("CAL",action=0)
