"""P13 real intervention ordering, matched probes, corrected lag negative."""
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import pytest
import torch
from scripts.native_p13_active_causal_probe_dev import (
    ARMS,PROBE_INTERVAL,PROBES_PER_EPISODE,
    FactualLagEstimator,information_scores,action_from_model,
    update_with_actual_feedback,evaluate,guard_dev)
from scripts.native_p12_online_causal_delay_dev import (
    CausalDelayWorld,DEV_SEEDS,FAMILIES)
from scripts.native_p11_temporal_credit_dev import (
    ConventionalCausalGRU,INIT_SEEDS)


def test_d_optimal_probe_uses_prior_gram_not_an_unseen_target():
    learner=FactualLagEstimator("full")
    history=[-1,1,-1,1,1,-1,1,-1]
    scores=information_scores(learner,history)
    assert set(scores)=={-1,1}
    assert all(math.isfinite(s) and s>=0 for s in scores.values())
    rng=random.Random(19)
    action,probing,record=action_from_model(
        learner,rng,history,0,policy="active")
    assert probing and record==scores and action in (-1,1)
    passive_rng=random.Random(19)
    other,passive_probe,passive_scores=action_from_model(
        learner,passive_rng,history,0,policy="passive")
    assert passive_probe and passive_scores==scores and other in (-1,1)
    assert action==max(scores,key=lambda a:(scores[a],a)) or (
        scores[1]==scores[-1] and action in (-1,1))
    for step in range(64):
        _,flag,_=action_from_model(
            learner,random.Random(step),history,step,policy="active")
        assert flag==(step%PROBE_INTERVAL==0)
    with pytest.raises(ValueError,match="INVALID_POLICY"):
        action_from_model(learner,rng,history,0,policy="privileged")


def test_scrambled_temporal_features_not_column_renaming():
    history=[1,1,1,-1,-1,1,1,-1,1,1,-1,1]
    good=FactualLagEstimator("full")
    corrupted=FactualLagEstimator("full")
    # Intentionally pick a seed that demonstrably breaks last-three history
    # alignment, unlike original P12's column-permutation negative.
    baseline=[history[-k] for k in (1,2,3)]
    chosen=None
    for seed in range(1,30):
        copy=list(history)
        random.Random(seed).shuffle(copy)
        if [copy[-k] for k in (1,2,3)]!=baseline:
            chosen=seed
            break
    assert chosen is not None
    unchanged=update_with_actual_feedback(
        good,1,history,.7,scramble_rng=None)
    shifted=update_with_actual_feedback(
        corrupted,1,history,.7,
        scramble_rng=random.Random(chosen))
    assert unchanged is False and shifted is True
    assert not torch.equal(good.gram,corrupted.gram)
    assert good.count==corrupted.count==1


def test_sham_action_effect_and_delay_are_observed_not_labels():
    seed=DEV_SEEDS[0]
    for family,lag in (("delay2",2),("delay3",3)):
        a=CausalDelayWorld(family,seed)
        b=CausalDelayWorld(family,seed)
        k=11
        for t in range(72):
            y1=a.realize(t,1)
            y2=b.realize(t,-1 if t==k else 1)
            if t<k+lag:assert y1==y2
            if t==k+lag:assert abs(y1-y2)>.7
    sham_a=CausalDelayWorld("sham_no_action",seed)
    sham_b=CausalDelayWorld("sham_no_action",seed)
    for t in range(72):
        assert sham_a.realize(t,1)==sham_b.realize(t,-1)


def test_equal_factual_online_update_budget_vs_adaptive_gru():
    torch.set_num_threads(1)
    torch.manual_seed(INIT_SEEDS[0])
    net=ConventionalCausalGRU().eval()
    q=evaluate("delay2",DEV_SEEDS[0],INIT_SEEDS[0],net)
    assert q["integrity"]=="PASS"
    assert all(q["checks"].values())
    assert set(q["arms"])==set(ARMS)
    for arm in ("active_probe","passive_random_probe",
                "scrambled_feedback_lags","adaptive_gru"):
        r=q["arms"][arm]
        assert r["online_actual_feedback_updates"]==64
        assert r["fixed_probe_opportunities_used"]==PROBES_PER_EPISODE
        assert len(r["trace_sha256"])==64
    assert q["arms"]["no_probe"]["online_actual_feedback_updates"]==64
    assert q["arms"]["no_probe"]["fixed_probe_opportunities_used"]==0
    assert q["arms"]["scrambled_feedback_lags"][
        "history_alignments_deliberately_broken"]>0
    assert q["arms"]["frozen_gru"]["online_actual_feedback_updates"]==0


def test_exact_source_protocol_and_fail_closed_science(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    p=subprocess.run([sys.executable,"-I","-B",
                      str(root/"scripts/native_p13_active_causal_probe_dev.py"),
                      "--help"],cwd=root,text=True,capture_output=True,
                     timeout=20,check=False)
    assert p.returncode==0,p.stderr[-900:]
    m=json.loads((root/"experiments/native_p13_active_causal_probe_frozen.json").read_text())
    assert m["schema"]=="ttt-hybrid.native-p13.active-probe-frozen.v1"
    assert m["source"]["parent_P12_sha"]=="e2d5ea4ff4650f7f3b6d6686c0575885c3e23e48"
    assert m["science"]["authorized"] is False
    assert m["science"]["world_seeds"]=={}
    assert m["science"]["first_use_consumed"] is False
    assert m["design"]["fixed_probe_slots"]==16
    assert m["design"]["online_feedback_updates"]==64
    monkeypatch.setenv("NATIVE_P13_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="SCIENTIFIC_FIRST_USE_FORBIDDEN"):
        guard_dev()
