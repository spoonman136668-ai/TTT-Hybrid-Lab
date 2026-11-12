"""P113 exact P80/F4 units, full factual fixture trajectories, no source birth."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from experiments import p112_f6_holdout_preflight as p112
from experiments import p113_f6_normalized_ssm_preflight as p113


def inputs():
    train=[p112.fixture_train_row(b,i,4) for b,i in p112.TRAIN_IDS]
    worlds=[]
    for i in range(8,16):
        a=(2.0+.3*i,.2,.1,-.35)
        b=(a[0]+.5,.4,.15,-.25)
        c=(b[0]-.1,.5,.16,-.3)
        events=[
          p113.fixture_event(1,i%2,(0.,0.,0.,0.),a,0.1),
          p113.fixture_event(2,(i+1)%2,a,b,.75),
          p113.fixture_event(3,i%2,b,c,1.2,terminated=True),
        ]
        worlds.append(p113.fixture_world(i,events))
    return train,worlds


def models(train,worlds):
    result={}
    for seed,factor in zip(p113.SEEDS,(.55,.8,0.)):
        preds=[]
        for world in sorted(worlds,key=lambda w:w["index"]):
            rows=[]
            for j,e in enumerate(world["events"]):
                lab=p113.label_from_fact(
                    e["state_before"],e["state_after"],e["reward"],
                    e["terminated"],e["truncated"])
                q=[float(v)*factor for v in lab]
                # Recurrent F4 head is unconstrained: do not falsely demand
                # zero predicted pad. First output is not used by F4 loss.
                q[7:]=[17. if seed==p113.SEEDS[1] else 0.]*9
                if j==0:
                    q[:7]=[10000.]*7
                rows.append(q)
            preds.append(rows)
        raw=f"test-fixture-NOT-trained-ssm:{seed}".encode()
        result[seed]={"bytes":raw,"predictions":preds,
          "manifest":p113.fixture_manifest(seed,raw,train,worlds,preds)}
    return result


def reseal(entry,train,worlds,seed):
    entry["manifest"]=p113.fixture_manifest(
        seed,entry["bytes"],train,worlds,entry["predictions"])


def test_p113_true_git_source_and_original_F4_forward_shape_pinned():
    f=p113.frozen()
    assert p113.FREEZE_COMMIT=="a24c1534efe238928b09781a0eafe64a649b3ad6"
    assert p113.FREEZE_BLOB=="72e3dc57ee078e884150b501850660fec5679b37"
    assert f["source"]["parent_sha"]=="d7c13bd08ea37d067443d366702916a2e493e1c2"
    assert f["source"]["p85_f4_dynamics_blob"]==p113.PINNED[
        "experiments/p85mr_dynamics_baselines.py"]
    assert f["controls"]["original_p69_readout_blob"]==p113.PINNED[
        "experiments/p69_trainable_competitors.py"]
    assert f["physical"]["P80_observed_channels"]==7
    assert f["physical"]["padding_channels"]==9
    assert "forecast padding channels may be nonzero" in f["physical"]["prediction"]
    assert f["controls"]["real_F6_firstuse_authorized"] is False
    assert f["controls"]["trained_SSM_checkpoint_qualified"] is False
    assert f["controls"]["real_F6_effects"]==0
    with pytest.raises(PermissionError,match="P113_F6_SCIENCE_FIRSTUSE_NOT_AUTHORIZED"):
        p113.deny_original_source("FIT",action=1)


def test_p113_normalized_label_matches_original_P80_seven_channels():
    before=(-.5,1.0,.1,2.0)
    after=(.7,1.2,.25,2.7)
    got=p113.label_from_fact(before,after,2.,True,False)
    assert len(got)==16
    expected=[math.tanh((y-x)/scale) for x,y,scale in
              zip(before,after,p113.SCALES)]
    expected += [math.tanh(2.),1.,0.]+[0.]*9
    assert got==expected
    assert all(x==0 for x in got[7:])
    assert not math.isclose(got[2],after[2]-before[2],abs_tol=.01)
    with pytest.raises(ValueError,match="P113_BAD_REALIZED_REWARD"):
        p113.label_from_fact(before,after,float("nan"),False,False)
    with pytest.raises(ValueError,match="P113_BAD_PHYSICAL_STATE"):
        p113.label_from_fact((float("inf"),0,0,0),after,.5,False,False)


def test_p113_all_eight_complete_trajectories_exact_count_and_seed_selection():
    train,worlds=inputs()
    audit=p113.validate_complete_fixture(train,worlds)
    assert audit["event_count"]==24
    assert audit["scored_count"]==16
    r=p113.score_p80_f4_fixture(train,worlds,models(train,worlds))
    assert r["validation_complete_worlds_fixture"]==8
    assert r["validation_actual_factual_events_fixture"]==24
    assert r["prequential_scored_events_after_first_fixture"]==16
    assert r["selected_seed_fixture_only"]==p113.SEEDS[1]
    scores=r["all_seed_seven_channel_metrics_fixture"]
    assert scores[p113.SEEDS[1]]["normalized_delta_mse_fixture"]>0.
    assert scores[p113.SEEDS[1]]["delta_vs_zero_ratio_fixture"]==pytest.approx(.04,abs=1e-12)
    assert scores[p113.SEEDS[0]]["delta_vs_zero_ratio_fixture"]==pytest.approx(.2025,abs=1e-12)
    assert scores[p113.SEEDS[2]]["delta_vs_zero_ratio_fixture"]==pytest.approx(1.,abs=1e-12)
    assert scores[p113.SEEDS[1]]["f4_weighted_objective_fixture"]>0.
    assert r["first_event_excluded_from_F4_loss"] is True
    assert r["forecast_padding_may_be_nonzero_and_is_excluded_from_F4_loss"] is True
    assert r["validation_used_for_seed_selection_not_independent_science"] is True
    assert r["original_F4_trained_weights_loaded"] is False
    assert r["original_F6_actions_executed"]==0
    assert r["real_F6_training_executed"] is False
    assert r["native_SSM_readout_authenticated"] is False
    assert r["scientific_pass"] is False
    assert r==p113.score_p80_f4_fixture(train,worlds,models(train,worlds))


def test_p113_first_step_and_unconstrained_padding_excluded_but_reward_matters():
    train,worlds=inputs()
    base=models(train,worlds)
    value=p113.score_p80_f4_fixture(train,worlds,base)
    changed=copy.deepcopy(base)
    seed=p113.SEEDS[1]
    changed[seed]["predictions"][0][0][:7]=[-1e6]*7
    changed[seed]["predictions"][0][2][7:]=[1e6]*9
    reseal(changed[seed],train,worlds,seed)
    r=p113.score_p80_f4_fixture(train,worlds,changed)
    assert r["all_seed_seven_channel_metrics_fixture"]==value["all_seed_seven_channel_metrics_fixture"]
    changed[seed]["predictions"][0][2][4]+=10.
    reseal(changed[seed],train,worlds,seed)
    r=p113.score_p80_f4_fixture(train,worlds,changed)
    assert r["all_seed_seven_channel_metrics_fixture"][seed]["reward_mse_fixture"] > value[
        "all_seed_seven_channel_metrics_fixture"][seed]["reward_mse_fixture"]


def test_p113_complete_episode_guards_reject_oracle_and_effect_replay():
    train,worlds=inputs()
    cases=[
       ("short","P113_INCOMPLETE_PREQUENTIAL_EPISODE"),
       ("missing_end","P113_EARLY_TERMINAL_DROP_OR_MISSING_END"),
       ("after_terminal","P113_EARLY_TERMINAL_DROP_OR_MISSING_END"),
       ("wrong_mode","P113_NOT_CONFIRMED_ACTUAL_BINARY_ACTION"),
       ("wrong_action","P113_NOT_CONFIRMED_ACTUAL_BINARY_ACTION"),
       ("bad_event_id","P113_NOT_CONFIRMED_ACTUAL_BINARY_ACTION"),
       ("broken_chain","P113_BROKEN_PHYSICAL_TRAJECTORY"),
       ("unknown_oracle","P113_UNCHOSEN_EFFECT_OR_STEP_SCHEMA"),
       ("nan","P113_BAD_PHYSICAL_STATE"),
       ("effect_receipt","P113_TAMPERED_SELECTED_EFFECT_RECEIPT"),
       ("world_receipt","P113_TAMPERED_WORLD_RECEIPT"),
       ("cal","P113_HELDOUT_FIT_SOURCE_SCOPE"),
       ("wrong_policy","P113_HELDOUT_FIT_SOURCE_SCOPE"),
    ]
    for change,reason in cases:
        rows=copy.deepcopy(worlds)
        w=rows[0];events=w["events"]
        if change=="short":events.pop()
        if change=="missing_end":events[-1]["terminated"]=False
        if change=="after_terminal":events[0]["terminated"]=True
        if change=="wrong_mode":events[0]["mode"]="observe"
        if change=="wrong_action":events[0]["executed_action"]=2
        if change=="bad_event_id":events[0]["event_id"]=3
        if change=="broken_chain":events[1]["state_before"][0]+=1.
        if change=="unknown_oracle":events[0]["unchosen_effect"]=[0.]*16
        if change=="nan":events[0]["state_after"][0]=float("nan")
        if change=="effect_receipt":events[0]["receipt_ref"]="fixture:fake"
        if change=="world_receipt":w["receipt_ref"]="fixture:fake"
        if change=="cal":w["split"]="CAL"
        if change=="wrong_policy":w["policy"]="a_opt"
        if change=="short":events.pop()   # one event, fails
        with pytest.raises(ValueError,match=reason):
            p113.validate_complete_fixture(train,rows)


def test_p113_no_world_omission_duplicate_or_train_validation_overlap():
    train,worlds=inputs()
    with pytest.raises(ValueError,match="P113_REQUIRE_EIGHT_COMPLETE_WORLDS"):
        p113.validate_complete_fixture(train,worlds[:-1])
    wrong=copy.deepcopy(worlds)
    wrong[1]=copy.deepcopy(wrong[0])
    with pytest.raises(ValueError,match="P113_DUPLICATE_HELDOUT_WORLD"):
        p113.validate_complete_fixture(train,wrong)
    wrong=copy.deepcopy(worlds)
    wrong[0]["index"]=7
    with pytest.raises(ValueError,match="P113_HELDOUT_FIT_SOURCE_SCOPE"):
        p113.validate_complete_fixture(train,wrong)
    more=copy.deepcopy(train)
    more[0]["block"],more[0]["index"]=1,8
    with pytest.raises(ValueError,match="P112_TRAIN_WORLD_ID_OR_SPLIT_DRIFT"):
        p113.validate_complete_fixture(more,worlds)


def test_p113_model_unit_shape_and_sealed_prediction_source_negatives():
    train,worlds=inputs()
    base=models(train,worlds)
    cases=[
        ("unknown_units","P113_WRONG_MODEL_OR_OUTPUT_UNITS"),
        ("wrong_F4","P113_WRONG_MODEL_OR_OUTPUT_UNITS"),
        ("wrong_P69","P113_WRONG_MODEL_OR_OUTPUT_UNITS"),
        ("bad_checkpoint","P113_WRONG_MODEL_OR_OUTPUT_UNITS"),
        ("FIT_leak","P113_WRONG_MODEL_OR_OUTPUT_UNITS"),
        ("future_label","P113_WRONG_MODEL_OR_OUTPUT_UNITS"),
        ("before_write_false","P113_WRONG_MODEL_OR_OUTPUT_UNITS"),
        ("bad_train_hash","P113_WRONG_MODEL_OR_OUTPUT_UNITS"),
        ("bad_world_hash","P113_WRONG_MODEL_OR_OUTPUT_UNITS"),
        ("bad_optimizer","P113_WRONG_MODEL_OR_OUTPUT_UNITS"),
        ("4D_raw","P113_REQUIRE_FULL_16D_SSM_FORECAST"),
        ("missing_event","P113_INCOMPLETE_PREQUENTIAL_PREDICTIONS"),
        ("nan_forecast","P113_NONFINITE_SSM_FORECAST"),
        ("changed_forecast","P113_TAMPERED_FORECAST_OR_CHECKPOINT_RECEIPT"),
        ("changed_receipt","P113_TAMPERED_FORECAST_OR_CHECKPOINT_RECEIPT"),
    ]
    for change,why in cases:
        x=copy.deepcopy(base);seed=p113.SEEDS[0]
        m=x[seed]["manifest"]
        p=x[seed]["predictions"]
        if change=="unknown_units":m["prediction_units"]="RAW_PHYSICAL_DELTA"
        if change=="wrong_F4":m["original_f4_blob"]="0"*40
        if change=="wrong_P69":m["original_p69_blob"]="0"*40
        if change=="bad_checkpoint":x[seed]["bytes"]=b"not checkpoint"
        if change=="FIT_leak":m["cal_or_ood_used"]=True
        if change=="future_label":m["fit_only"]=False
        if change=="before_write_false":m["read_before_current_effect"]=False
        if change=="bad_train_hash":m["train_receipts_sha256"]="0"*64
        if change=="bad_world_hash":m["validation_worlds_sha256"]="0"*64
        if change=="bad_optimizer":m["optimizer_steps_declared"]=799
        if change=="4D_raw":p[0][1]=[.1,.2,.3,.4]
        if change=="missing_event":p[0].pop()
        if change=="nan_forecast":p[0][1][0]=float("nan")
        if change=="changed_forecast":p[0][1][0]=99.
        if change=="changed_receipt":m["receipt_ref"]="fixture:wrong"
        with pytest.raises(ValueError,match=why):
            p113.score_p80_f4_fixture(train,worlds,x)
    x=copy.deepcopy(base)
    x.pop(p113.SEEDS[1])
    with pytest.raises(ValueError,match="P113_THREE_SSM_SEEDS_REQUIRED"):
        p113.score_p80_f4_fixture(train,worlds,x)


def test_p113_no_native_model_training_no_source_workflow_or_firstuse():
    s=Path(p113.__file__).read_text("utf-8")
    for forbidden in ("from causal_gym","import causal_gym",
                      "from p69_trainable_competitors","from p80mr_causalgym",
                      "import requests","import subprocess",
                      "import socket","env.do(","env.reset(",
                      "torch.load(","optimizer.step(","dispatch_workflow(",
                      "run_owner(","firstuse_authorize("):
        assert forbidden not in s
    with pytest.raises(PermissionError,match="P113_F6_SCIENCE_FIRSTUSE_NOT_AUTHORIZED"):
        p113.deny_original_source("CAL",source="old-F3")
