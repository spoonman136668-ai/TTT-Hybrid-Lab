"""P19: external factual teacher supervision, frozen negative and source admission."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from scripts.native_p19_factual_expert_imit_dev import (
    LinearCurrentController,history_tensor,scaled_history,logits,
    training_model,shuffled_label_negative,require_dev,
    INIT_SEEDS,TRAIN_RESET_SEEDS,VAL_RESET_SEEDS,DEV_EVAL_SEEDS,
    TRAIN_STEPS,BATCH,REFERENCE_MIN_REWARD,EVAL_ARMS)


def test_small_factual_actors_shape_and_no_privileged_engine_fields():
    torch.manual_seed(8)
    x=scaled_history([[.1,.2,.03,.04,0.,0.]])
    assert x.shape==(1,8,6)
    assert len(INIT_SEEDS)==3
    for k in ("linear","history_mlp","history_gru","misattributed_linear"):
        model=training_model(k,INIT_SEEDS[0])
        scores=logits(model,k,x)
        assert scores.shape==(1,2)
        assert all(torch.isfinite(v).all() for v in (scores,))
    with pytest.raises(ValueError,match="P19_INPUT_NOT_COMPLETE_FACTUAL_HISTORY"):
        LinearCurrentController()(x[:,:7,:])


def test_action_label_negative_really_breaks_attribution_preserves_histogram():
    y=torch.tensor([0,0,1,1]*32,dtype=torch.long)
    bad,sha=shuffled_label_negative(y)
    assert not torch.equal(y,bad)
    assert torch.equal(y.sort().values,bad.sort().values)
    assert len(sha)==64
    assert all(TRAIN_STEPS[k]>0 for k in TRAIN_STEPS)


def test_only_prior_factual_rows_and_current_visible_sensors():
    orig=[[.03,.04,.05,.06,1.,1.]]
    first=scaled_history(orig)
    assert first.shape==(1,8,6)
    assert torch.equal(first[:,:7,:],torch.zeros_like(first[:,:7,:]))
    assert first[0,-1,2].item()>first[0,-1,0].item()


def test_frozen_protocol_and_forbidden_firstuse(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    m=json.loads((root/"experiments/native_p19_factual_expert_frozen.json").read_text())
    assert m["schema"]=="ttt-hybrid.native-p19.real-factual-expert-bc-frozen.v1"
    assert m["source"]["P18_parent_sha"]=="675ef40f80c43814cc320d6b9950d37a9395c0f8"
    assert m["source"]["S29_owner_sha"]=="30c27ca4865e16b7574ee2b562879108db2b3796"
    assert m["science"]["authorized"] is False
    assert m["science"]["first_use_consumed"] is False
    assert m["science"]["reserved_world_seeds"]=={}
    assert m["design"]["train_world_resets"]==list(TRAIN_RESET_SEEDS)
    assert m["design"]["validation_resets"]==list(VAL_RESET_SEEDS)
    assert m["design"]["eval_world_resets"]==list(DEV_EVAL_SEEDS)
    assert m["design"]["train_updates"]==TRAIN_STEPS
    assert m["design"]["evaluation_arms"]==list(EVAL_ARMS)
    assert m["design"]["conventional_learned_competence_min_reward"]==REFERENCE_MIN_REWARD==175
    monkeypatch.setenv("NATIVE_P19_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="PROTECTED_SCIENCE_FIRST_USE_DENIED"):
        require_dev()


def test_source_only_cli_has_no_install_permission():
    root=Path(__file__).resolve().parents[1]
    p=subprocess.run([sys.executable,"-I","-B",
           str(root/"scripts/native_p19_factual_expert_imit_dev.py"),
           "--help"],capture_output=True,text=True,cwd=root,timeout=20)
    assert p.returncode==0,p.stderr[-1500:]
    s=(root/"scripts/native_p19_factual_expert_imit_dev.py").read_text()
    assert "pip install" not in s
    assert "source_identity" not in s
    assert "admit_gymnasium" in s
    assert "not_reward_only_autonomous_learning" in s
