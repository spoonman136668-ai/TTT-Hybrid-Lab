"""P114 tests run original native F4 optimizer only on self-generated toy facts."""
from __future__ import annotations

import math

import pytest
import torch

from experiments import p114_f6_native_optimizer_smoke as p114

torch.set_num_threads(1)


def test_p114_original_native_three_seed_train_and_cold_checkpoint_smoke():
    reports = [p114.smoke_seed(seed) for seed in p114.SEEDS]
    assert len({r["checkpoint_npz_sha256"] for r in reports}) == 3
    assert len({r["trained_toy_weight_sha256"] for r in reports}) == 3
    for seed, result in zip(p114.SEEDS, reports):
        assert result["seed"] == seed
        assert result["adamw_updates_fake"] == 8
        assert result["train_worlds_fake"] == 24
        assert result["validation_worlds_fake"] == 8
        assert result["initial_weight_sha256"] != result["trained_toy_weight_sha256"]
        assert result["checkpoint_cold_prediction_exact"] is True
        assert all(math.isfinite(result[k]) for k in (
            "initial_train_f4_loss_fake", "postupdate_train_f4_loss_fake",
            "initial_validation_f4_loss_fake", "postupdate_validation_f4_loss_fake"))
        assert result["trained_source_checkpoint_authenticated"] is False
        assert result["real_F6_training"] is False
        assert result["real_F6_source_sessions"] == 0
        assert result["real_F6_effects"] == 0
        assert result["scientific_pass"] is False


def test_p114_same_seed_toy_training_and_serialization_are_deterministic():
    a, b = p114.smoke_seed(p114.SEEDS[0]), p114.smoke_seed(p114.SEEDS[0])
    assert a == b


def test_p114_no_real_source_no_unfrozen_seed_or_protocol_promotion():
    freeze = p114.locked()
    assert freeze["source"]["original_external_source_used"] is False
    assert freeze["toy"]["trained_on_validation_data"] is False
    assert freeze["toy"]["full_future_owner_training_updates_not_executed"] == 800
    assert freeze["authority"]["P110_A5_owner_approved"] is False
    assert freeze["authority"]["protocol_P106A_B_resolved"] is False
    assert freeze["authority"]["scientific_pass"] is False
    with pytest.raises(ValueError, match="P114_UNREGISTERED_SEED"):
        p114.smoke_seed(0)
    with pytest.raises(PermissionError, match="P114_F6_EXTERNAL_FIRST_USE_NOT_AUTHORIZED"):
        p114.refuse_source("FIT", "CartPoleWind")
    source = p114.__file__
    from pathlib import Path
    body = Path(source).read_text("utf-8")
    for forbidden in ("from causal_gym", "import causal_gym", "env.do(",
                      "env.reset(", "import requests", "import socket",
                      "dispatch_workflow(", "torch.load(", "pickle.load("):
        assert forbidden not in body
