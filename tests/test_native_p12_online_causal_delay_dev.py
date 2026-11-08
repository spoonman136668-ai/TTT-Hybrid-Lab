"""Native P12 chronological observations, lag identification and sham controls."""
import json
import math
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from scripts.native_p12_online_causal_delay_dev import (
    ARMS,FAMILIES,DEV_SEEDS,OBS_PREFIX,DECISIONS,RIDGE,
    CausalDelayWorld,FactualLagEstimator,features,score_episode,dev_only)
from scripts.native_p11_temporal_credit_dev import (
    LearnedTemporalCredit,ConventionalCausalGRU)


def test_unknown_delays_are_real_and_sham_has_no_action_effect():
    for delay_name,delay in (("delay2",2),("delay3",3)):
        a=CausalDelayWorld(delay_name,DEV_SEEDS[0])
        b=CausalDelayWorld(delay_name,DEV_SEEDS[0])
        for t in range(OBS_PREFIX+DECISIONS):
            # A unilateral change at index k must affect the future factual
            # consequence at t=k+delay, not time k.
            i=11
            a_act=1
            b_act=-1 if t==i else 1
            ya=a.realize(t,a_act)
            yb=b.realize(t,b_act)
            if t<i+delay:
                assert ya==yb
            if t==i+delay:
                assert abs(ya-yb)>.7
    a=CausalDelayWorld("sham_no_action",DEV_SEEDS[1])
    b=CausalDelayWorld("sham_no_action",DEV_SEEDS[1])
    for t in range(OBS_PREFIX+DECISIONS):
        assert a.realize(t,1)==b.realize(t,-1)
    with pytest.raises(ValueError,match="UNREGISTERED_DEVELOPMENT_WORLD"):
        CausalDelayWorld("delay4",DEV_SEEDS[0])


def test_current_only_and_full_same_shape_same_updates_never_future():
    full=FactualLagEstimator("full")
    current=FactualLagEstimator("current")
    history=[]
    for t in range(OBS_PREFIX+4):
        a=1 if t%2 else -1
        y=.4*(history[-2] if len(history)>=2 else 0)+.03
        for policy in (full,current):
            policy.observe(a,history,y,scored=t>=OBS_PREFIX)
        history.append(a)
    assert full.count==current.count==OBS_PREFIX+4
    assert full.scored_updates==current.scored_updates==4
    assert full.gram.shape==current.gram.shape==(5,5)
    assert float(full.coefficients()[3])!=float(current.coefficients()[3])
    assert torch.equal(features(1,[1,-1,1],"current")[2:],
                       torch.zeros(3,dtype=torch.float64))
    with pytest.raises(ValueError,match="P12_UNVERIFIED_FACTUAL_FEEDBACK"):
        full.observe(1,history,float("nan"),scored=True)


def test_real_closed_loop_uses_own_factual_feedback_and_equal_action_budget():
    torch.manual_seed(353)
    models={"structured":LearnedTemporalCredit().eval(),
            "gru":ConventionalCausalGRU().eval()}
    for family in FAMILIES:
        e=score_episode(family,DEV_SEEDS[0],models)
        assert e["same_initial_factual_history"]
        assert e["all_policies_equal_decision_budget"]
        assert e["all_consequences_finite"]
        assert set(e["policies"])==set(ARMS)
        assert e["policies"]["online_full"]["online_feedback_updates_during_scoring"]==64
        assert e["policies"]["online_current"]["online_feedback_updates_during_scoring"]==64
        assert e["policies"]["online_no_adapt"]["online_feedback_updates_during_scoring"]==0
        assert all(len(r["trace_sha256"])==64 for r in e["policies"].values())
        if family=="sham_no_action":
            assert e["causal_action_effect_absent_on_sham"] is True


def test_isolated_cli_frozen_protocol_no_scientific_firstuse(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    r=subprocess.run([sys.executable,"-I","-B",
                      str(root/"scripts/native_p12_online_causal_delay_dev.py"),
                      "--help"],cwd=root,capture_output=True,text=True,timeout=20)
    assert r.returncode==0,r.stderr[-1000:]
    m=json.loads((root/"experiments/native_p12_online_causal_delay_frozen.json").read_text())
    assert m["schema"]=="ttt-hybrid.native-p12.online-delay-frozen.v1"
    assert m["source"]["parent_P11_sha"]=="bcd8ae3b2af1b54f3d93a3d2e467d6962fd7dd4f"
    assert m["science"]["authorized"] is False
    assert m["science"]["first_use_consumed"] is False
    assert m["science"]["world_seeds"]=={}
    assert m["design"]["allowed_delays"]==[0,1,2,3]
    assert m["design"]["online_feedback_updates"]==64
    monkeypatch.setenv("NATIVE_P12_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="P12_SCIENCE_FIRST_USE_FORBIDDEN"):
        dev_only()
