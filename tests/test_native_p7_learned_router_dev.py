"""P7 gradient-trained factual-history mechanism (synthetic DEV, no science)."""
import ast
import json
from pathlib import Path
import pytest
import torch

from scripts.native_p7_learned_router_dev import (
    HISTORY,INIT_SEEDS,TRAIN_DEV_SEED,VALIDATION_DEV_SEED,
    TRAIN_EPISODES,VALID_EPISODES,UPDATES,BATCH,
    SCHEMA,LearnedRouter,ConventionalMLP,synthetic_dev,
    handcoded_prior_pair,train_one,run)


def test_history_has_only_past_factual_events_and_no_current_labels():
    train=synthetic_dev(TRAIN_DEV_SEED,TRAIN_EPISODES)
    val=synthetic_dev(VALIDATION_DEV_SEED,VALID_EPISODES)
    assert train[0].shape==(6000,HISTORY*2)
    assert val[0].shape==(1920,HISTORY*2)
    assert train[1].shape==(6000,1)
    assert train[2].shape==(6000,1)
    assert val[3].shape==(1920,)
    assert not torch.equal(train[0][:100],val[0][:100])
    assert set(val[3].tolist())=={-1,1}
    assert set(val[1].flatten().tolist())=={-1.,1.}
    with pytest.raises(ValueError,match="UNREGISTERED_DEV_SOURCE"):
        synthetic_dev(7207,1)


def test_two_trainable_arms_almost_same_parameter_budget():
    a=LearnedRouter()
    b=ConventionalMLP()
    na=sum(p.numel() for p in a.parameters())
    nb=sum(p.numel() for p in b.parameters())
    assert na==828
    assert nb==849
    assert abs(na-nb)/nb<.05
    h=torch.zeros(2,HISTORY*2)
    x=torch.tensor([[1.],[-1.]])
    pa,route=a(h,x)
    pb,unused=b(h,x)
    assert pa.shape==pb.shape==(2,1)
    assert route.shape==(2,1)
    assert unused is None


def test_prior_pair_negative_does_not_read_current_target():
    data=synthetic_dev(TRAIN_DEV_SEED,2)
    pred,choice=handcoded_prior_pair(data[0],data[1])
    assert pred.shape==data[2].shape
    assert choice.shape==data[3].shape
    same_pred,_=handcoded_prior_pair(data[0],data[1])
    assert torch.equal(pred,same_pred)


def test_real_gradient_training_factual_only_and_negative_retention():
    result=run()
    assert result["status"]=="PASS_TRAINABLE_DEV_INTEGRITY_NOT_SCIENCE"
    assert result["scientific_first_use"]=="NOT_RUN"
    assert result["scientific_seeds_consumed"]==[]
    assert result["worlds_constructed"]==0
    assert result["context_identity_used_for_training"]=="NEVER"
    assert result["training_example_count"]==6000
    assert result["independent_dev_validation_example_count"]==1920
    assert len(result["records"])==len(INIT_SEEDS)==3
    assert all(all(r["checks"].values()) for r in result["records"])
    for r in result["records"]:
        assert r["learned"]["training_updates"]==UPDATES
        assert r["trained_conventional"]["training_updates"]==UPDATES
        assert r["learned"]["model_weights_changed_by_real_gradients"]
        assert r["trained_conventional"]["model_weights_changed_by_real_gradients"]
        assert len(r["learned"]["validation_checkpoints"])==6
        assert r["diagnostics"]["learned_better_than_its_untrained_initialization"]
        assert r["diagnostics"]["learned_better_than_shuffled_history"]
        assert r["diagnostics"]["learned_better_than_no_history"]
        # Never force superiority over a competent trained conventional control,
        # or silently redefine plateau success after observing results.
        assert isinstance(r["diagnostics"]["learned_better_than_fully_trained_mlp"],bool)
        assert isinstance(r["diagnostics"]["strict_plateau"],bool)


def test_frozen_scope_manifest_and_science_denial(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    p=json.loads((root/"experiments/native_p7_learned_router_dev_frozen.json").read_text())
    assert p["schema"]=="ttt-hybrid.native-p7.learned-router-dev-frozen.v1"
    assert p["source"]["parent_pr60_sha"]=="1123a8aee15572f7f2602bce135a0f38c7b9909a"
    assert p["science"]["authorized"] is False
    assert p["science"]["world_seeds"]=={}
    assert p["science"]["consumed"] is False
    monkeypatch.setenv("NATIVE_P7_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="SCIENCE_FIRST_USE_FORBIDDEN"):
        run()
    source=(root/"scripts/native_p7_learned_router_dev.py").read_text()
    imports=[node.module or "" for node in ast.walk(ast.parse(source))
             if isinstance(node,ast.ImportFrom)]
    assert not any(tag in module for module in imports for tag in
                   ("g7_s20","g7_s25","g7_s26","g7_s27"))


def test_arm_and_seed_admission_rejects_invalid():
    val=synthetic_dev(VALIDATION_DEV_SEED,1)
    train=synthetic_dev(TRAIN_DEV_SEED,1)
    with pytest.raises(ValueError,match="UNREGISTERED_ARM_OR_INIT"):
        train_one(LearnedRouter,7207,train,val)
    with pytest.raises(ValueError,match="UNREGISTERED_ARM_OR_INIT"):
        train_one(int,INIT_SEEDS[0],train,val)
