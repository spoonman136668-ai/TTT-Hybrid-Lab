"""P15 frozen stronger-control chronological DEV comparison and negative gates."""
import json
import math
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from scripts.native_p15_strong_control_dev import (
    ARMS,common_probe_rng,select_arm_action,evaluate,require_dev,
    P14_INIT,P11_INIT,FAMILIES,OLD_SEEDS,LAG_SEEDS,PROBE_COUNT)
from scripts.native_p14_reward_trained_intervention_dev import (
    CausalFeaturePolicy)
from scripts.native_p11_temporal_credit_dev import (
    ConventionalCausalGRU,LearnedTemporalCredit)


def models_fixture():
    torch.manual_seed(7)
    return {"p14_reward_causal":CausalFeaturePolicy().eval(),
            "p14_credit_corrupted":CausalFeaturePolicy().eval(),
            "p11_gru":ConventionalCausalGRU().eval(),
            "p11_structured":LearnedTemporalCredit().eval()}


def test_frozen_forced_probe_signs_match_every_neural_policy():
    models=models_fixture()
    q=evaluate("lagged",OLD_SEEDS[0],0,models)
    assert q["integrity"]=="PASS",q["checks"]
    assert all(q["checks"].values())
    probes=[x["fixed_probe_decisions"] for x in q["arms"].values()]
    assert len({tuple(tuple(x) for x in seq) for seq in probes})==1
    assert len(probes[0])==PROBE_COUNT==16
    assert set(q["arms"])==set(ARMS)
    assert all(x["scored_action_count"]==64 for x in q["arms"].values())


def test_same_source_sham_has_no_action_consequence():
    q=evaluate("sham_no_action",LAG_SEEDS[0],0,models_fixture())
    assert q["integrity"]=="PASS"
    assert q["checks"]["zero_action_sham_all_arms_same_exact_reward_sequence"]
    assert len({tuple(x["factual_outcomes"]) for x in q["arms"].values()})==1


def test_context_guard_and_both_P11_planners_see_only_real_history():
    models=models_fixture()
    history=[(1 if x%2 else -1,float(x)*.01)
             for x in range(8)]
    for arm in ARMS:
        rng=common_probe_rng(OLD_SEEDS[0],0)
        result=select_arm_action(
            arm,models,history,[a for a,_ in history],1,rng,None
            if arm!="p13_passive_online" else
            __import__("scripts.native_p12_online_causal_delay_dev",
                       fromlist=["FactualLagEstimator"]).FactualLagEstimator())
        assert result in (-1,1)
    with pytest.raises(ValueError,match="INCOMPLETE_FACTUAL_CONTEXT"):
        select_arm_action("p11_conventional_gru_twostep",models,[(1,.5)],
                          [1],1,common_probe_rng(OLD_SEEDS[0],0),None)


def test_exact_original_sources_science_denial_and_cli(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    d=json.loads((root/"experiments/native_p15_strong_control_frozen.json").read_text())
    assert d["schema"]=="ttt-hybrid.native-p15.strong-control-frozen.v1"
    assert d["source"]["P14_parent_sha"]=="40329f552eec61ddd498772100380020eb6f1489"
    assert d["science"]["authorized"] is False
    assert d["science"]["first_use_consumed"] is False
    assert d["science"]["world_seeds"]=={}
    assert d["design"]["forced_probe_slots"]==16
    assert d["design"]["required_margin"]==.05
    assert d["design"]["old_capability_max_deficit"]==.02
    p=subprocess.run([sys.executable,"-I","-B",
                      str(root/"scripts/native_p15_strong_control_dev.py"),
                      "--help"],cwd=root,capture_output=True,text=True,timeout=20)
    assert p.returncode==0,p.stderr[-1200:]
    monkeypatch.setenv("NATIVE_P15_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="FIRST_USE_FORBIDDEN"):
        require_dev()
