"""P8 DEV-only strict parity, timing, schedule and scientific fences."""
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from scripts.native_p8_schedule_dev import (
    SCHEMA, ARMS, SCHEDULES, INIT_SEEDS, UPDATES,
    BATCH,INITIAL_LR,MIN_LR,schedule_lr,train_pair_arm,run
)
from scripts.native_p7_learned_router_dev import (
    synthetic_dev,TRAIN_DEV_SEED,VALIDATION_DEV_SEED
)


def test_frozen_schedule_endpoints_and_bounds():
    assert schedule_lr("fixed",0)==INITIAL_LR
    assert schedule_lr("fixed",UPDATES)==INITIAL_LR
    assert math.isclose(schedule_lr("cosine_decay",0),INITIAL_LR)
    assert math.isclose(schedule_lr("cosine_decay",UPDATES),MIN_LR,abs_tol=1e-10)
    assert MIN_LR<schedule_lr("cosine_decay",UPDATES//2)<INITIAL_LR
    with pytest.raises(ValueError,match="UNREGISTERED_SCHEDULE"):
        schedule_lr("adaptive_validation",500)
    with pytest.raises(ValueError,match="UNREGISTERED_SCHEDULE"):
        schedule_lr("fixed",-1)


def test_unregistered_science_seeds_and_model_arms_rejected():
    train=synthetic_dev(TRAIN_DEV_SEED,1)
    val=synthetic_dev(VALIDATION_DEV_SEED,1)
    shuffled=torch.arange(len(val[0]))
    for architecture,schedule,seed in (
        ("LearnedRouter","fixed",7207),
        ("NovelOracle","fixed",INIT_SEEDS[0]),
        ("LearnedRouter","unknown",INIT_SEEDS[0]),
    ):
        with pytest.raises(ValueError,match="UNREGISTERED_DEV_ARM_OR_INIT"):
            train_pair_arm(architecture,schedule,seed,train,val,shuffled)


def test_actual_matched_training_and_preserved_plateau_failures():
    result=run()
    assert result["status"]=="PASS_FIXED_SCHEDULE_COMPARISON_ENGINEERING_ONLY"
    assert result["scientific_first_use"]=="NOT_RUN"
    assert result["scientific_world_seeds_consumed"]==[]
    assert result["native_substrate_changed"] is False
    assert result["worlds_constructed"]==0
    assert result["model_count"]==len(INIT_SEEDS)*len(ARMS)*len(SCHEDULES)==12
    assert result["fit_examples"]==6000
    assert result["dev_validation_examples"]==1920
    assert result["all_model_optimizer_steps"]==12*UPDATES
    assert result["all_model_examples_drawn"]==12*UPDATES*BATCH
    assert all(all(seed_checks.values()) for seed_checks in
               result["engineering_integrity_checks"].values())
    assert all(len(r["validation_curve"])==12 for r in result["records"])
    assert all(r["model_weights_changed"] for r in result["records"])
    assert all(r["training_wall_seconds"]>0 for r in result["records"])
    assert all(isinstance(r["strict_one_percent_plateau"],bool)
               for r in result["records"])
    assert all(0<=v<=3 for v in result["hypothesis_support_of_three"].values())


def test_protocol_is_dev_and_negative_outcomes_not_promoted(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/"experiments/native_p8_schedule_dev_frozen.json").read_text())
    assert manifest["schema"]=="ttt-hybrid.native-p8.schedule-stability-dev.v1"
    assert manifest["source"]["parent_pr62_sha"]=="7658be2cd4ff66afb3fb46519b516d8071641a36"
    assert manifest["science"]["authorized"] is False
    assert manifest["science"]["seeds"]=={}
    assert manifest["science"]["first_use_consumed"] is False
    assert manifest["design"]["steps_per_training_arm"]==UPDATES
    assert manifest["design"]["schedules"]==list(SCHEDULES)
    monkeypatch.setenv("NATIVE_P8_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="SCIENTIFIC_FIRST_USE_FORBIDDEN"):
        run()


def test_actual_isolated_cli_entrypoint_loads_local_p7_without_pythonpath():
    root=Path(__file__).resolve().parents[1]
    source=root/"scripts/native_p8_schedule_dev.py"
    result=subprocess.run([sys.executable,"-I","-B",str(source),"--help"],
                          cwd=root,
                          capture_output=True,text=True,timeout=20,check=False)
    assert result.returncode==0, result.stderr[-1000:]
    assert "--receipt" in result.stdout
