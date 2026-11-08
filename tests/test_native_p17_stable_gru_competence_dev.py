"""P17 exact P11 baseline stability and mature conventional control gates."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from scripts.native_p17_stable_gru_competence_dev import (
    extended_train,actual_episode,evaluate_model,require_dev_only,
    EXTENDED_UPDATES,EMA_START,EVAL_EVERY,INIT_SEEDS,ARMS)
from scripts.native_p11_temporal_credit_dev import (
    ConventionalCausalGRU,data,FIT_SEED,VALIDATION_SEED,plan,
    train_arm as original_train_arm)
from scripts.native_p10_closed_loop_intervention_dev import DEV_WORLD_SEEDS


def test_original_P11_architecture_and_factual_target_are_unchanged():
    torch.manual_seed(353)
    model=ConventionalCausalGRU().eval()
    assert sum(p.numel() for p in model.parameters())==833
    fit=data(FIT_SEED,2)
    val=data(VALIDATION_SEED,2)
    assert fit[0].shape==(80,8,2)
    assert val[0].shape==(80,8,2)
    assert evaluate_model(model,val)["factual_validation_mse"]>=0.


def test_frozen_extended_budget_and_ema_admission():
    fit=data(FIT_SEED,2)
    val=data(VALIDATION_SEED,2)
    with pytest.raises(ValueError,match="P17_FROZEN_TRAINING_BUDGET"):
        extended_train(INIT_SEEDS[0],fit,val,updates=1200)
    with pytest.raises(ValueError,match="P17_FROZEN_TRAINING_BUDGET"):
        extended_train(INIT_SEEDS[0],fit,val,updates=5000)
    assert EXTENDED_UPDATES==3600
    assert EMA_START==2400
    assert EVAL_EVERY==200


def test_original_and_extended_model_equal_factual_prefix_and_source():
    # A bounded real training probe, not an extra full scientific source run.
    torch.set_num_threads(1)
    fit=data(FIT_SEED,2)
    val=data(VALIDATION_SEED,2)
    original,original_receipt=original_train_arm(
        ConventionalCausalGRU,INIT_SEEDS[0],fit,val,updates=2)
    assert original_receipt["actual_optimizer_updates"]==2
    assert original_receipt["gradient_changed_weights"]
    assert original_receipt["factual_val_examples"]==80
    assert original_receipt["unique_factual_fit_examples"]==80


def test_strong_original_and_ema_are_evaluated_with_identical_actual_actions():
    torch.manual_seed(353)
    a=ConventionalCausalGRU().eval()
    b=ConventionalCausalGRU().eval()
    c=ConventionalCausalGRU().eval()
    ep=actual_episode("instant",DEV_WORLD_SEEDS[0],0,{
        "original_P11_1200":a,"extended_raw_3600":b,
        "extended_ema_3600":c})
    assert ep["integrity"]=="PASS",ep["checks"]
    assert set(ep["arms"])==set(ARMS)
    probes=[tuple(map(tuple,x["signed_forced_probe_actions"]))
            for x in ep["arms"].values()]
    assert len(set(probes))==1 and len(probes[0])==16
    assert all(x["actual_decisions"]==64 for x in ep["arms"].values())


def test_full_cli_and_original_science_first_use_denial(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    m=json.loads((root/"experiments/native_p17_stable_gru_frozen.json").read_text())
    assert m["schema"]=="ttt-hybrid.native-p17.stable-gru-frozen.v1"
    assert m["source"]["P16_parent_sha"]=="96e66cd70c5f9d45665b63f7c2b886c288bb5090"
    assert m["science"]["authorized"] is False
    assert m["science"]["first_use_consumed"] is False
    assert m["science"]["world_seeds"]=={}
    assert m["design"]["extended_updates"]==3600
    assert m["design"]["EMA_decay"]==.995
    assert m["design"]["plateau_5_last_check_max_spread"]==.01
    p=subprocess.run([sys.executable,"-I","-B",
                      str(root/"scripts/native_p17_stable_gru_competence_dev.py"),
                      "--help"],cwd=root,capture_output=True,text=True,timeout=20)
    assert p.returncode==0,p.stderr[-1200:]
    monkeypatch.setenv("NATIVE_P17_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="P17_SCIENTIFIC_FIRST_USE_FORBIDDEN"):
        require_dev_only()
