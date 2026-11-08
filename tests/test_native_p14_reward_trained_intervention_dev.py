"""P14 learned intervention: frozen real reward credit, state and controls."""
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest
import torch
from scripts.native_p14_reward_trained_intervention_dev import (
    CausalFeaturePolicy,ConventionalGRUPolicy,TrainingWorld,
    completed_history,world_batch,reward_alignment_negative,
    train_model,evaluate_episode,INIT_SEEDS,FAMILIES,ARMS,only_dev,
    TRAIN_HORIZON,BATCH_WORLDS,TRAIN_UPDATES
)


def test_models_equal_information_nearly_equal_trainable_params():
    a=CausalFeaturePolicy()
    b=ConventionalGRUPolicy()
    assert sum(p.numel() for p in a.parameters())==441
    assert sum(p.numel() for p in b.parameters())==431
    x=torch.linspace(-.5,.5,32).reshape(2,8,2)
    assert a(x).shape==b(x).shape==(2,)
    with pytest.raises(ValueError,match="CAUSAL_INPUT_SHAPE"):
        a(x[:,:7,:])
    with pytest.raises(ValueError,match="GRU_INPUT_SHAPE"):
        b(x[:,:7,:])


def test_actual_training_world_no_hidden_labels_and_no_future_return_in_input():
    worlds,histories=world_batch(0,INIT_SEEDS[0])
    assert len(worlds)==BATCH_WORLDS==12
    assert completed_history(histories).shape==(BATCH_WORLDS,8,2)
    for world,history in zip(worlds,histories):
        assert len(history)==8 and len(world.factual)==8
        y=world.realize(1)
        assert isinstance(y,float) and len(world.factual)==9
    with pytest.raises(ValueError,match="INCOMPLETE_OBSERVATION"):
        completed_history([[(1,.1)]])


def test_reward_alignment_negative_is_actually_different():
    r=torch.arange(TRAIN_HORIZON*BATCH_WORLDS,dtype=torch.float32).reshape(
        TRAIN_HORIZON,BATCH_WORLDS)
    broken=reward_alignment_negative(r)
    assert not torch.equal(broken,r)
    assert torch.equal(broken.sort().values.flatten().sort().values,
                       r.sort().values.flatten().sort().values)


def test_actual_policy_gradient_weights_change_and_cold_object_restoration():
    torch.set_num_threads(1)
    a,r=train_model("causal",INIT_SEEDS[0],updates=2)
    assert r["actual_policy_gradient_updates"]==2
    assert r["gradient_changed_weights"]
    assert r["in_process_frozen_object_restoration_bit_identical"]
    b,q=train_model("gru",INIT_SEEDS[0],updates=2)
    assert q["gradient_changed_weights"]
    assert r["training_world_cohort_digest"]==q["training_world_cohort_digest"]


def test_old_factual_world_and_no_effect_sham_with_no_training_future():
    torch.manual_seed(INIT_SEEDS[0])
    a=CausalFeaturePolicy().eval()
    b=ConventionalGRUPolicy().eval()
    models={"trained_causal":a,"trained_gru":b,"reward_misaligned":a}
    for fam,seed in (("instant",88031),("lagged",88031),
                     ("delay3",88421),("sham_no_action",88421)):
        result=evaluate_episode(fam,seed,INIT_SEEDS[0],models)
        assert result["integrity"]=="PASS",result["checks"]
        assert all(result["checks"].values())
        assert set(result["policies"])==set(ARMS)
        assert all(v["scored_actions"]==64 for v in result["policies"].values())
        if fam=="sham_no_action":
            assert result["checks"]["sham_exact_identical_reward_all_actions"]


def test_standalone_cli_and_original_science_fence(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    p=subprocess.run([sys.executable,"-I","-B",
                      str(root/"scripts/native_p14_reward_trained_intervention_dev.py"),
                      "--help"],capture_output=True,text=True,cwd=root,timeout=20)
    assert p.returncode==0,p.stderr[-1000:]
    d=json.loads((root/"experiments/native_p14_reward_policy_frozen.json").read_text())
    assert d["schema"]=="ttt-hybrid.native-p14.reward-policy-frozen.v1"
    assert d["source"]["parent_P13_sha"]=="065de52f2a6a698508e15db9c9bdb1e021201441"
    assert d["science"]["authorized"] is False
    assert d["science"]["world_seeds"]=={}
    assert d["science"]["first_use_consumed"] is False
    assert d["design"]["actual_policy_gradient_updates"]==TRAIN_UPDATES==96
    assert d["design"]["same_number_eval_probe_slots"]==16
    monkeypatch.setenv("NATIVE_P14_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="P14_SCIENCE_FIRST_USE_DENIED"):
        only_dev()
