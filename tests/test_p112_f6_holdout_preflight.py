"""P112 disjoint TRAIN24/VAL8 and independently recomputed factual fixture MSE."""
from __future__ import annotations
import copy
import math
from pathlib import Path
import pytest

from experiments import p112_f6_holdout_preflight as p112


def sample():
    train=[p112.fixture_train_row(b,i,25)
           for b,i in p112.TRAIN_IDS]
    val=[p112.fixture_validation_row(i,i%2,
                  (0.,0.,0.,0.),(.1*(i-6),.05,.01,.2))
         for _,i in p112.VAL_IDS]
    truth=[[b-a for a,b in zip(row["state_before"],row["state_after"])]
           for row in val]
    predictions={seed:[[v*scale for v in y] for y in truth]
                 for seed,scale in zip(p112.SEEDS,(.4,.8,0.))}
    entries={}
    for seed in p112.SEEDS:
        byt=f"p112-fixture-not-trained-SSM:{seed}".encode()
        preds=predictions[seed]
        m=p112.fixture_checkpoint(seed,byt,train,val,preds)
        entries[seed]={"manifest":m,"bytes":byt,"predictions":preds}
    return train,val,entries


def test_p112_immutable_freeze_source_pins_zero_effect_and_no_training():
    f=p112.frozen()
    assert p112.FREEZE_COMMIT=="27e68423f1807574d0370e4eed9d9fb3341a3f4b"
    assert p112.FREEZE_BLOB=="76f46135a8f134e1d638615aafad2d1d8f5805c3"
    assert f["parent"]["exact_source_sha"]=="85f9dcdad0ad78dfb18a807bef64ebebef1ccbdc"
    assert f["data"]["train_worlds_n"]==24 and f["data"]["validation_worlds_n"]==8
    assert f["data"]["train_scoring_overlap_allowed"] is False
    assert f["data"]["extra_reset_do_calls"]==0
    assert f["data"]["checkpoint_selection"].startswith("Select seed")
    assert f["result"]["F6_owner_firstuse_authorized"] is False
    assert f["result"]["P110_protocol_changed"] is False
    assert f["result"]["real_source_FIT_runs"]==0
    assert f["result"]["P108_crossworld_ledger_qualified"] is False
    assert f["result"]["science_pass"] is False
    assert f["engineering"]["no_source_effect_firstuse"] is True
    with pytest.raises(PermissionError,match="P112_NO_F6_SOURCE_OR_CAL_FIRSTUSE_AUTHORITY"):
        p112.deny_external_trial("FIT",action=0)


def test_p112_24_8_no_world_overlap_and_three_seed_actual_MSE_recompute():
    train,val,models=sample()
    a=p112.validate_worlds(train,val)
    assert len(a["train_ids"])==24 and len(a["validation_ids"])==8
    assert not (a["train_ids"] & a["validation_ids"])
    assert a["confirmed_train_effects_fixture"]==24*25
    r=p112.score_fixture_holdout(train,val,models)
    assert r["selected_seed_fixture_only"]==p112.SEEDS[1]
    assert r["training_worlds_fixture"]==24 and r["validation_worlds_fixture"]==8
    ratios=r["all_three_seed_validation_metrics_fixture"]
    assert math.isclose(ratios[p112.SEEDS[0]]["ratio_to_zero_fixture"],.36,abs_tol=1e-12)
    assert math.isclose(ratios[p112.SEEDS[1]]["ratio_to_zero_fixture"],.04,abs_tol=1e-12)
    assert math.isclose(ratios[p112.SEEDS[2]]["ratio_to_zero_fixture"],1.,abs_tol=1e-12)
    assert r["development_fixture_threshold_met"] is True
    assert r["competence_score_is_selection_biased"] is True
    assert r["training_optimizer_steps_executed"]==0
    assert r["source_effects_executed"]==0
    assert r["trained_real_ssm"] is False
    assert r["real_checkpoint_authenticated"] is False
    assert r["P110_protocol_owner_revision_required"] is True
    assert r["real_CAL_admitted"] is False and r["scientific_pass"] is False
    assert p112.score_fixture_holdout(train,val,models)==r


def test_p112_same_holdout_is_not_double_counted_as_independent_competence():
    train,val,models=sample()
    r=p112.score_fixture_holdout(train,val,models)
    assert r["competence_score_is_selection_biased"] is True
    assert r["classification"]=="ENGINEERING_FIXTURE_ONLY_NOT_GENUINE_CROSS_FIT"
    assert r["deployment"] is False and r["F6_real_firstuse_authorized"] is False


def test_p112_exact_train_worlds_and_policy_no_duplicate_or_val_leakage():
    train,val,models=sample()
    cases=[
        ("remove",None,"P112_REQUIRE_24_TRAIN_WORLDS"),
        ("duplicate",None,"P112_TRAIN_WORLD_ID_OR_SPLIT_DRIFT"),
        ("cal",None,"P112_TRAIN_WORLD_ID_OR_SPLIT_DRIFT"),
        ("wrong_policy",None,"P112_TRAIN_WORLD_ID_OR_SPLIT_DRIFT"),
        ("val_world",None,"P112_TRAIN_WORLD_ID_OR_SPLIT_DRIFT"),
        ("fake_action",None,"P112_TRAIN_FACTUAL_ACTION_OR_TERMINAL_DRIFT"),
        ("fake_receipt",None,"P112_NOT_EXACT_FIXTURE_RECEIPT"),
        ("oracle",None,"P112_TRAIN_SCHEMA_OR_ORACLE_DRIFT"),
    ]
    for name,_,reason in cases:
        x=copy.deepcopy(train)
        if name=="remove":x.pop()
        elif name=="duplicate":x[0]=x[1].copy()
        elif name=="cal":x[0]["split"]="CAL"
        elif name=="wrong_policy":x[0]["policy"]="greedy_goal"
        elif name=="val_world":x[0]["block"],x[0]["index"]=(1,8)
        elif name=="fake_action":x[0]["action_count"]=0
        elif name=="fake_receipt":x[0]["receipt_ref"]="fixture:spoof"
        elif name=="oracle":x[0]["unchosen_effect"]="never observed"
        with pytest.raises(ValueError,match=reason):
            p112.validate_worlds(x,val)


def test_p112_heldout_worlds_not_train_and_only_factual_binary_actions():
    train,val,models=sample()
    cases=[
        ("remove","P112_REQUIRE_8_HELDOUT_WORLDS"),
        ("duplicate","P112_VALIDATION_TRAIN_OVERLAP_OR_SPLIT_DRIFT"),
        ("wrong_block","P112_VALIDATION_TRAIN_OVERLAP_OR_SPLIT_DRIFT"),
        ("cal","P112_VALIDATION_TRAIN_OVERLAP_OR_SPLIT_DRIFT"),
        ("action2","P112_UNQUALIFIED_FACTUAL_ACTION"),
        ("unconfirmed","P112_NOT_EXACT_FIXTURE_RECEIPT"),
        ("oracle","P112_VALIDATION_SCHEMA_OR_ORACLE_DRIFT"),
        ("nan","P112_INVALID_PHYSICAL_STATE"),
    ]
    for name,reason in cases:
        x=copy.deepcopy(val)
        if name=="remove":x.pop()
        elif name=="duplicate":x[1]=x[0].copy()
        elif name=="wrong_block":x[0]["block"]=0
        elif name=="cal":x[0]["split"]="CAL"
        elif name=="action2":x[0]["executed_action"]=2
        elif name=="unconfirmed":x[0]["receipt_ref"]="fixture:wrong"
        elif name=="oracle":x[0]["unexecuted_action_effect"]=[1,2,3,4]
        elif name=="nan":x[0]["state_after"][0]=float("nan")
        with pytest.raises(ValueError,match=reason):
            p112.validate_worlds(train,x)


def test_p112_checkpoint_model_hash_train_receipt_fit_only_and_budget_denials():
    train,val,models=sample()
    bads=[
        ("model_source_blob","0"*40),
        ("checkpoint_format","pickle"),
        ("checkpoint_sha256","0"*64),
        ("train_receipts_sha256","0"*64),
        ("train_world_ids",[[0,0]]),
        ("validation_world_ids",[[0,0]]),
        ("optimizer","SGD"),
        ("optimizer_steps",801),
        ("fit_only",False),
        ("cal_or_ood_used",True),
    ]
    for field,value in bads:
        x=copy.deepcopy(models)
        x[p112.SEEDS[0]]["manifest"][field]=value
        with pytest.raises(ValueError,match="P112_UNTRAINED_OR_CONTAMINATED_CHECKPOINT_FIXTURE"):
            p112.score_fixture_holdout(train,val,x)
    x=copy.deepcopy(models)
    x[p112.SEEDS[0]]["bytes"]=b"other weights"
    with pytest.raises(ValueError,match="P112_UNTRAINED_OR_CONTAMINATED_CHECKPOINT_FIXTURE"):
        p112.score_fixture_holdout(train,val,x)


def test_p112_no_scalar_crossfit_forgery_missing_seed_or_forecast_tamper():
    train,val,models=sample()
    x=copy.deepcopy(models);x.pop(p112.SEEDS[1])
    with pytest.raises(ValueError,match="P112_REQUIRE_ALL_THREE_SEED_MODELS"):
        p112.score_fixture_holdout(train,val,x)
    x=copy.deepcopy(models)
    x[p112.SEEDS[1]]["predictions"][0][0]+=100.
    with pytest.raises(ValueError,match="P112_TAMPERED_PREDICTION_OR_CHECKPOINT_RECEIPT"):
        p112.score_fixture_holdout(train,val,x)
    x=copy.deepcopy(models)
    x[p112.SEEDS[1]]["predictions"].pop()
    with pytest.raises(ValueError,match="P112_INCOMPLETE_EIGHT_WORLD_PREDICTIONS"):
        p112.score_fixture_holdout(train,val,x)
    x=copy.deepcopy(models)
    x[p112.SEEDS[1]]["manifest"]["train_crossfit_mse"]=0.
    with pytest.raises(ValueError,match="P112_BAD_CHECKPOINT_FIELDS"):
        p112.score_fixture_holdout(train,val,x)


def test_p112_zero_reference_invalid_prediction_and_cost_fail_closed():
    train,val,models=sample()
    all_zero=copy.deepcopy(val)
    for row in all_zero:
        row["state_after"]=row["state_before"][:]
        row["receipt_ref"]="fixture:"+p112._sha({k:v for k,v in row.items() if k!="receipt_ref"})
    new_models={}
    for seed,e in models.items():
        pred=[[0.,0.,0.,0.] for _ in all_zero]
        raw=e["bytes"]
        new_models[seed]={"bytes":raw,"predictions":pred,
           "manifest":p112.fixture_checkpoint(seed,raw,train,all_zero,pred)}
    with pytest.raises(ValueError,match="P112_UNINFORMATIVE_ZERO_PREDICTOR"):
        p112.score_fixture_holdout(train,all_zero,new_models)
    x=copy.deepcopy(models);x[p112.SEEDS[0]]["manifest"]["validation_wall_ms"]=0.
    with pytest.raises(ValueError,match="P112_BAD_EVALUATION_COST"):
        p112.score_fixture_holdout(train,val,x)
    x=copy.deepcopy(models);x[p112.SEEDS[0]]["predictions"][0][0]=float("nan")
    with pytest.raises(ValueError):
        p112.score_fixture_holdout(train,val,x)


def test_p112_no_external_owner_import_training_or_science_promotion():
    s=Path(p112.__file__).read_text("utf-8")
    for forbidden in ("from causal_gym","import causal_gym","import requests",
                      "import subprocess","import socket","env.reset(",
                      "env.do(","optimizer.step(","torch.load(",
                      "dispatch_workflow(","run_owner(","firstuse_authorize("):
        assert forbidden not in s
    f=p112.frozen()
    assert f["engineering"]["no_model_training"] is True
    assert f["engineering"]["no_new_workflow"] is True
    assert f["engineering"]["no_new_runner_or_schedule"] is True
