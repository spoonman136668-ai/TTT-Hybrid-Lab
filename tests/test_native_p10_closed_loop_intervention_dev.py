"""P10: actual factual intervention, causal-order and equal-control tests."""
import json
import math
from pathlib import Path
import subprocess
import sys
import pytest
import torch

from scripts.native_p10_closed_loop_intervention_dev import (
    World,FAMILIES,DEV_WORLD_SEEDS,ARMS,WARMUP,DECISIONS,INIT_SEEDS,
    select_action,run_episode,dev_guard
)
from scripts.native_p7_learned_router_dev import LearnedRouter,ConventionalMLP


def test_world_counterfactual_is_diagnostic_only_and_lag_really_delayed():
    seed=DEV_WORLD_SEEDS[0]
    a=World("instant",seed)
    b=World("instant",seed)
    t=12
    difference=a.realize(t,1,-1)-b.realize(t,-1,-1)
    assert abs(difference)>0.7
    a=World("lagged",seed)
    b=World("lagged",seed)
    assert a.realize(t,1,-1)==b.realize(t,-1,-1)
    c=World("confounded",seed)
    d=World("confounded",seed)
    assert abs(c.realize(t,1,1)-d.realize(t,1,-1)-.4)<1e-10
    assert World("instant",seed).regime_sign(12)==World("instant",seed).regime_sign(60)
    assert World("instant",seed).regime_sign(12)==-World("instant",seed).regime_sign(40)
    with pytest.raises(ValueError,match="UNREGISTERED_WORLD_SOURCE"):
        World("instant",7207)


def test_select_action_has_no_outcome_or_regime_input():
    torch.manual_seed(303)
    models={"learned_router":LearnedRouter().eval(),
            "trained_mlp":ConventionalMLP().eval()}
    hist=[((-1 if i%2==0 else 1),float(i%2)*.4)
          for i in range(WARMUP)]
    import random
    rng=random.Random(19)
    for arm in ARMS[:-1]:
        action,prediction=select_action(arm,hist,models,rng,8)
        assert action in (-1,1)
        if arm in ("learned_router","trained_mlp"):
            assert prediction is not None and math.isfinite(prediction)
    with pytest.raises(ValueError,match="COMPLETED_FACTUAL_HISTORY_REQUIRED"):
        select_action("learned_router",[(1,.1)],models,rng,8)


def test_full_episode_initial_history_same_and_every_factual_action_counted():
    torch.manual_seed(303)
    models={"learned_router":LearnedRouter().eval(),
            "trained_mlp":ConventionalMLP().eval()}
    p=run_episode("lagged",DEV_WORLD_SEEDS[0],models)
    assert p["same_initial_factual_history_for_every_policy"]
    assert p["equal_decision_opportunities_and_nonzero_actions"]
    assert p["no_counterfactual_result_used_by_policy"]
    assert set(p["data"])==set(ARMS)
    for value in p["data"].values():
        assert value["factual_interventions_including_shared_prefix"]==WARMUP+DECISIONS
        assert value["decision_budget"]==DECISIONS
        assert len(value["factual_outcomes"])==DECISIONS
        assert len(value["trace_sha256"])==64


def test_isolated_entrypoint_manifest_and_science_denial(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    p=subprocess.run([sys.executable,"-I","-B",
                      str(root/"scripts/native_p10_closed_loop_intervention_dev.py"),
                      "--help"],cwd=root,text=True,capture_output=True,timeout=20)
    assert p.returncode==0,p.stderr[-1000:]
    m=json.loads((root/"experiments/native_p10_closed_loop_dev_frozen.json").read_text())
    assert m["schema"]=="ttt-hybrid.native-p10.intervention-dev-frozen.v1"
    assert m["source"]["parent_p8_sha"]=="a0d749d3ec28cbc15ff212a6fdd9752799cc8578"
    assert m["science"]["authorized"] is False
    assert m["science"]["seeds"]=={}
    assert m["science"]["first_use_consumed"] is False
    assert m["design"]["original_worlds_synthetically_known"] is True
    monkeypatch.setenv("NATIVE_P10_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="ORIGINAL_SCIENTIFIC_FIRST_USE_FORBIDDEN"):
        dev_guard()
