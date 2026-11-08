"""P18 external real observation-to-action actor comparisons: source-safe."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from scripts.native_p18_real_cartpole_factual_dev import (
    HistoryGRU,FullHistoryMLP,history_tensor,factual_row,handcoded_action,
    require_dev,TRAIN_EPISODES,TRAIN_INITS,EVAL_SEEDS,EVAL_ARMS,
    TRAIN_ARMS,MAX_PROCESS_SECONDS
)

def test_near_equal_models_get_identical_factual_complete_history():
    g=HistoryGRU()
    m=FullHistoryMLP()
    gs=sum(p.numel() for p in g.parameters())
    ms=sum(p.numel() for p in m.parameters())
    assert gs==489 and ms==471
    assert (max(gs,ms)-min(gs,ms))/max(gs,ms)<.05
    x=history_tensor([factual_row([0.,.1,.03,.05],0,0.)])
    assert x.shape==(1,8,6)
    assert g(x)[0].shape==(1,2)
    assert m(x)[0].shape==(1,2)
    with pytest.raises(ValueError,match="P18_HISTORY_SHAPE"):
        g(x[:,:7,:])
    with pytest.raises(ValueError,match="P18_HISTORY_SHAPE"):
        m(x[:,:7,:])

def test_factual_event_admission_and_observation_only_handcoded_reference():
    assert handcoded_action([0.,0.,.1,0.])==1
    assert handcoded_action([0.,0.,-.1,0.])==0
    with pytest.raises(ValueError,match="P18_PREVIOUS_ACTION_INVALID"):
        factual_row([0.,0.,0.,0.],9,1.)
    with pytest.raises(ValueError,match="P18_NONFINITE_SENSOR"):
        factual_row([0.,float("nan"),0.,0.],1,1.)
    with pytest.raises(ValueError,match="P18_UNQUALIFIED_FACTUAL_HISTORY"):
        history_tensor([[0.]*4])

def test_mit_source_owned_no_scientific_first_use_and_locked_world_budget(monkeypatch):
    r=Path(__file__).resolve().parents[1]
    d=json.loads((r/"experiments/native_p18_real_cartpole_factual_frozen.json").read_text())
    assert d["schema"]=="ttt-hybrid.native-p18.real-cartpole-frozen.v1"
    assert d["source"]["P17_parent_sha"]=="70d75944534dbbf5e2af42a5cd7038ae7f07b871"
    assert d["source"]["S29_external_owner_sha"]=="30c27ca4865e16b7574ee2b562879108db2b3796"
    assert d["science"]["authorized"] is False
    assert d["science"]["first_use_consumed"] is False
    assert d["science"]["reserved_seeds"]=={}
    assert d["design"]["training_episodes_per_model"]==TRAIN_EPISODES==64
    assert d["design"]["train_initializations"]==list(TRAIN_INITS)
    assert d["design"]["development_eval_seeds"]==list(EVAL_SEEDS)
    assert d["design"]["evaluation_arms"]==list(EVAL_ARMS)
    assert d["design"]["training_arms"]==list(TRAIN_ARMS)
    assert d["design"]["max_process_seconds"]==MAX_PROCESS_SECONDS==220
    monkeypatch.setenv("NATIVE_P18_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="SCIENTIFIC_FIRST_USE_NOT_AUTHORIZED"):
        require_dev()

def test_self_contained_help_no_install_or_live_science():
    r=Path(__file__).resolve().parents[1]
    v=subprocess.run([sys.executable,"-I","-B",
        str(r/"scripts/native_p18_real_cartpole_factual_dev.py"),
        "--help"],cwd=r,capture_output=True,text=True,timeout=20)
    assert v.returncode==0,v.stderr[-1000:]
    txt=(r/"scripts/native_p18_real_cartpole_factual_dev.py").read_text()
    assert "pip install" not in txt
    assert "source_identity" in txt
    assert "cloudpickle_source_identity" in txt
    assert "scientific_superiority_established" in txt
