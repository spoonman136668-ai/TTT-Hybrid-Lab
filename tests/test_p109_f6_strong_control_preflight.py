"""P109 source-neutral complete FIT strong-control and fixture SSM custody tests."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from experiments import p106_f6_source_neutral_policies as p106
from experiments import p109_f6_strong_control_preflight as p109


CHECKPOINT_BYTES=b"fixture-ssm256-600-FIT-only-not-real-model-weights"

def cohorts(steps=None):
    base={"a_opt":29,"greedy_goal":28,"physical_balance":23,
          "ssm256_model_based":35}
    if steps is not None:
        base.update(steps)
    return {
        arm:[
            p109.make_fixture_row(arm,b,i,base[arm],
                                  float(base[arm])*.8,
                                  1. if arm=="ssm256_model_based" else 2.,
                                  .1*base[arm])
            for b in range(3) for i in range(16)]
        for arm in p109.CONTROLS
    }


def checkpoint(data,updates=600):
    rows=data[p109.CHECKPOINT_ARM]
    return {
        "schema":"p109.synthetic-SSM-checkpoint-receipt.v1",
        "model_kind":"ssm256","model_source_blob":p109.SSM_BLOB,
        "checkpoint_format":"fixture-bytes-only",
        "checkpoint_sha256":hashlib.sha256(CHECKPOINT_BYTES).hexdigest(),
        "training_seed":985201,"fit_only":True,"trained_worlds":48,
        "confirmed_factual_effects":sum(r["action_count"] for r in rows),
        "train_updates":updates,"optimizer":"AdamW",
        "optimizer_steps":updates,"model_static_parameters":8192,
        "persistent_floats":256,"cal_or_ood_used":False,
        "fit_dataset_sha256":p109._fit_receipt_sha(rows),
    }


def test_p109_protocol_frozen_before_code_and_no_external_firstuse():
    f=p109.freeze()
    assert p109.FREEZE_COMMIT=="7e882b348a55dfed644455de2422873cd1b047f9"
    assert f["source"]["canonical_pr"]==233
    assert f["source"]["parent_sha"]=="60fab256da20ae4808722d145682d0718d3c17b6"
    assert f["source"]["F6_protocol_fork"]=="UNRESOLVED_NO_F6_ACTIVATION"
    assert f["source"]["new_cartpolewind_firstuse"] is False
    assert f["science"]["authentic_F6_effects"]==0
    assert f["science"]["scientific_pass"] is False
    assert f["science"]["real_SSM_checkpoint_qualified"] is False
    assert f["science"]["real_CAL_admitted"] is False
    assert f["fixture_contract"]["worlds_per_arm"]==48
    assert f["fixture_contract"]["policy_arms"]==list(p109.CONTROLS)
    assert f["engineering"]["no_owner_dispatched"] is True
    assert p106._git_blob((Path(__file__).resolve().parents[1]/
        "experiments/p109_f6_strong_control_freeze.json").read_bytes())==p109.FREEZE_BLOB
    with pytest.raises(PermissionError,match="P109_F6_REAL_TRAINING_FIRSTUSE_AND_CAL_NOT_AUTHORIZED"):
        p109.refuse_real_source_or_CAL("CAL",action=1)


def test_p109_full_fit_train_receipt_and_strongest_conventional_fixture_only():
    data=cohorts()
    manifest=checkpoint(data)
    report=p109.fit_incumbent_fixture(data,manifest,CHECKPOINT_BYTES)
    assert report["incumbent_fixture_only"]=="ssm256_model_based"
    assert report["competence_fixture_only"] is True
    assert report["source_neutral_four_arm_fairness"]["ssm256_model_based"]["mean_survival"]==35.
    assert report["source_neutral_four_arm_fairness"]["ssm256_model_based"]["complete_factual_effect_count"]==35*48
    assert report["all_arms_equal_world_opportunities"] is True
    assert report["all_arms_have_equal_factual_effect_counts"] is False
    assert report["trained_ssm_fixture_custody"]["train_steps_fixture"]==600
    assert report["trained_ssm_fixture_custody"]["complete_FIT_fixture_effects"]==35*48
    assert report["trained_ssm_fixture_custody"]["real_owner_checkpoint_authenticated"] is False
    assert report["trained_real_SSM"] is False
    assert report["real_FIT_baseline_qualified"] is False
    assert report["real_CAL_admitted"] is False
    assert report["external_F6_effects"]==0
    assert report["scientific_pass"] is False
    assert report["no_oracle_effects_used"] is True
    assert p109.fit_incumbent_fixture(data,manifest,CHECKPOINT_BYTES)==report


def test_p109_incumbent_can_lose_to_goal_aware_conventional_and_tie_is_stable():
    rows=cohorts({"a_opt":55,"ssm256_model_based":42})
    result=p109.fit_incumbent_fixture(rows,checkpoint(rows),CHECKPOINT_BYTES)
    assert result["incumbent_fixture_only"]=="a_opt"
    ties=cohorts({"a_opt":42,"greedy_goal":42,"ssm256_model_based":42})
    # Cost 1 for SSM, 2 for A-opt/greedy, so SSM wins equal survival.
    result=p109.fit_incumbent_fixture(ties,checkpoint(ties),CHECKPOINT_BYTES)
    assert result["incumbent_fixture_only"]=="ssm256_model_based"
    all_equal=cohorts({"a_opt":42,"greedy_goal":42,"physical_balance":42,"ssm256_model_based":42})
    for arm in p109.CONTROLS:
        all_equal[arm]=[p109.make_fixture_row(arm,r["block"],r["index"],42,30.,2.,1.)
                        for r in all_equal[arm]]
    # Same physical loss => fixed deterministic arm order.
    result=p109.fit_incumbent_fixture(all_equal,checkpoint(all_equal),CHECKPOINT_BYTES)
    assert result["incumbent_fixture_only"]=="a_opt"


def test_p109_low_competence_blocks_even_a_well_formed_fixture_checkpoint():
    rows=cohorts({arm:12 for arm in p109.CONTROLS})
    result=p109.fit_incumbent_fixture(rows,checkpoint(rows),CHECKPOINT_BYTES)
    assert result["competence_fixture_only"] is False
    assert result["real_FIT_baseline_qualified"] is False
    assert result["real_CAL_admitted"] is False


def test_p109_missing_or_duplicate_fit_world_and_spent_CAL_born_fails_closed():
    data=cohorts()
    with pytest.raises(ValueError,match="P109_MISSING_STRONG_TRAINED_CONVENTIONAL_ARM"):
        p109.fit_incumbent_fixture(
            {k:v for k,v in data.items() if k!="ssm256_model_based"},
            checkpoint(data),CHECKPOINT_BYTES)
    missing=copy.deepcopy(data)
    missing["a_opt"].pop()
    with pytest.raises(ValueError,match="P109_INCOMPLETE_FIT_EPISODES"):
        p109.fit_incumbent_fixture(missing,checkpoint(missing),CHECKPOINT_BYTES)
    duplicate=copy.deepcopy(data)
    duplicate["a_opt"][1]=copy.deepcopy(duplicate["a_opt"][0])
    with pytest.raises(ValueError,match="P109_MISSING_DUPLICATE_OR_WRONG_FIT_WORLD"):
        p109.fit_incumbent_fixture(duplicate,checkpoint(duplicate),CHECKPOINT_BYTES)
    cal=copy.deepcopy(data)
    cal["a_opt"][0]["source_split"]="CAL"
    with pytest.raises(ValueError,match="P109_NOT_EXACT_FIT_EPISODE_FIELDS"):
        p109.fit_incumbent_fixture(cal,checkpoint(cal),CHECKPOINT_BYTES)
    with pytest.raises(ValueError,match="P109_UNQUALIFIED_FIT_SESSION"):
        p109.make_fixture_row("a_opt",4,0,24,10.,1.,1.)
    with pytest.raises(ValueError,match="P109_UNQUALIFIED_FIT_SESSION"):
        p109.make_fixture_row("a_opt",0,16,24,10.,1.,1.)


def test_p109_observed_action_count_terminal_and_hash_negative_controls():
    base=cohorts()
    changes=[
        ("action_count",0,"P109_UNMATCHED_FACTUAL_ACTION_BUDGET"),
        ("terminated",False,"P109_UNTRUTHFUL_TERMINATION"),
        ("truncated",True,"P109_UNTRUTHFUL_TERMINATION"),
        ("reward_total",float("nan"),"P109_BAD_REALIZED_REWARD"),
        ("physical_loss",-1.,"P109_BAD_PHYSICAL_COST"),
        ("wall_ms",0.,"P109_BAD_WALL_COST"),
        ("source_session_sha256","0"*64,"P109_WRONG_FIT_SESSION_ID"),
        ("receipt_ref","fixture:not-valid","P109_TAMPERED_FIXTURE_RECEIPT"),
    ]
    for key,value,reason in changes:
        rows=copy.deepcopy(base)
        rows["a_opt"][0][key]=value
        with pytest.raises(ValueError,match=reason):
            p109.fit_incumbent_fixture(rows,checkpoint(rows),CHECKPOINT_BYTES)
    # An actual full-horizon record is not required to fake early terminal.
    valid=cohorts({"a_opt":96})
    assert p109.fit_incumbent_fixture(valid,checkpoint(valid),CHECKPOINT_BYTES)["all_arms_equal_world_opportunities"]


def test_p109_checkpoint_hash_architecture_and_cal_leakage_are_not_waived():
    data=cohorts()
    m=checkpoint(data)
    assert p109.validate_checkpoint(m,CHECKPOINT_BYTES,data["ssm256_model_based"])["checkpoint_fixture_digest_matched"]
    with pytest.raises(ValueError,match="P109_INVALID_OR_WRONG_SOURCE_CHECKPOINT"):
        p109.validate_checkpoint(m,b"counterfeit",data["ssm256_model_based"])
    for key,bad,reason in [
        ("model_source_blob","0"*40,"P109_INVALID_OR_WRONG_SOURCE_CHECKPOINT"),
        ("checkpoint_format","pickle","P109_INVALID_OR_WRONG_SOURCE_CHECKPOINT"),
        ("fit_dataset_sha256","0"*64,"P109_INVALID_OR_WRONG_SOURCE_CHECKPOINT"),
        ("train_updates",0,"P109_UNTRAINED_OR_CAL_CONTAMINATED_SSM"),
        ("train_updates",399,"P109_UNTRAINED_OR_CAL_CONTAMINATED_SSM"),
        ("train_updates",801,"P109_UNTRAINED_OR_CAL_CONTAMINATED_SSM"),
        ("fit_only",False,"P109_UNTRAINED_OR_CAL_CONTAMINATED_SSM"),
        ("cal_or_ood_used",True,"P109_UNTRAINED_OR_CAL_CONTAMINATED_SSM"),
        ("optimizer_steps",999,"P109_UNTRAINED_OR_CAL_CONTAMINATED_SSM"),
        ("confirmed_factual_effects",1,"P109_UNTRAINED_OR_CAL_CONTAMINATED_SSM"),
        ("persistent_floats",512,"P109_UNTRAINED_OR_CAL_CONTAMINATED_SSM"),
    ]:
        altered=copy.deepcopy(m);altered[key]=bad
        with pytest.raises(ValueError,match=reason):
            p109.validate_checkpoint(altered,CHECKPOINT_BYTES,
                                     data["ssm256_model_based"])
    with pytest.raises(ValueError,match="P109_INVALID_FIXTURE_CHECKPOINT_FORMAT"):
        p109.validate_checkpoint(m,"not bytes",data["ssm256_model_based"])


def test_p109_never_calls_owner_source_or_old_f4_training():
    text=Path(p109.__file__).read_text("utf-8")
    for marker in ("import causal_gym","from causal_gym","from p85mr_dynamics_baselines",
                   "from p85mr_f4_train_controls","import subprocess",
                   "import requests","env.reset(","env.do(","dispatch_workflow(",
                   "optimizer.step(","torch.load(","pickle.load("):
        assert marker not in text
    assert p109.freeze()["science"]["real_FIT_selection_qualified"] is False
