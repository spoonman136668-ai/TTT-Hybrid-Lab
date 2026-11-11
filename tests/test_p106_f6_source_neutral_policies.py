"""P106 strict source-neutral future-F6 readiness: NO new simulator effects."""
from __future__ import annotations

import inspect
import math
from pathlib import Path

import pytest
from experiments import p106_f6_source_neutral_policies as p106


def matrices():
    weights=[[0. for _ in range(16)] for _ in range(16)]
    precision=[[0. for _ in range(16)] for _ in range(16)]
    for i in range(16):
        precision[i][i]=1.
    return weights,precision


def test_p106_exact_source_first_freeze_and_unchanged_parent_contract():
    f=p106.locked_protocol()
    assert p106.FREEZE_COMMIT=="389bf0a3c6e023309d63a5d01a7f353ed0ffe0ec"
    assert f["parent"]["sha"]=="955d032ac21b71b9d5e02061ae9727600205d569"
    assert f["authority"]["fresh_F6_external_first_use_authorized"] is False
    assert f["source"]["source_firstuse_now"] is False
    assert f["outcome"]["scientific_pass"] is False
    assert f["outcome"]["external_source_executed"] is False
    assert f["source"]["sha"]=="c94e9a4cd3641588b6b8507e71ccee5c27032bdf"
    assert f["selector"]["epsilon_grid_FIT_only"]==[.05,.15,.30]
    assert f["source"]["upstream_blobs"]["causal_gym/envs/cartpole_wind.py"]==(
        "d65a6a31941709dc9201259d80dd4ebbcf48b4fa")


def test_p106_source_free_proposed_fit_cal_seeds_are_new_and_strictly_scoped():
    receipt=p106.proposed_partition()
    assert receipt["fit_seeds"]==48 and receipt["cal_seeds"]==48
    assert receipt["unique_future_fit_cal"]==96 and receipt["overlap"]==0
    assert receipt["source_events_created"] is False
    assert p106.future_seed_schedule("FIT",0,0)==1820000000
    assert p106.future_seed_schedule("FIT",2,15)==1820002015
    assert p106.future_seed_schedule("CAL",0,0)==1820100000
    assert p106.future_seed_schedule("CAL",2,15)==1820102015
    for args in (("OOD",0,0),("NEGATIVE",0,0),
                 ("DEV",0,0),("FIT",3,0),("FIT",-1,0),
                 ("FIT",0,16),("CAL",0,-1),("FIT",True,0)):
        with pytest.raises(ValueError,match="P106_SEALED_OR_UNKNOWN_SOURCE"):
            p106.future_seed_schedule(*args)


def test_p106_pure_16d_state_conditional_p80_key_matches_independent_math():
    for action in (0,1):
        s=(.1,-.2,.03,-.4)
        result=p106.action_key(s,action)
        basis=[1.]+[math.tanh(v/sc) for v,sc in zip(s,p106.SCALES)]
        basis += [float(action==0),float(action==1)]+[0.]*9
        denom=math.sqrt(sum(v*v for v in basis))
        assert len(result)==16
        assert math.isclose(sum(x*x for x in result),1.,rel_tol=1e-12)
        assert all(math.isclose(a,b/denom,abs_tol=1e-15)
                   for a,b in zip(result,basis))
        assert all(x==0. for x in result[7:])
    for state,action in (((0,0,0,0),2),((0,0,0,0),True),
                         ((0,0,math.nan,0),0),((0,0,0),0)):
        with pytest.raises(ValueError):
            p106.action_key(state,action)


def test_p106_binary_policies_only_observed_state_and_estimated_effect():
    weights,precision=matrices()
    precision[5][5]=50.
    precision[6][6]=1.
    weights[2][5]=.1
    weights[2][6]=-.1
    observed=(.0,.0,.04,.0)
    aopt=p106.select_policy("a_opt",observed,weights,precision)
    greedy=p106.select_policy("greedy_goal",observed,weights,precision)
    guarded=p106.select_policy("guarded_voi",observed,weights,precision,
                               epsilon=.05)
    balanced=p106.select_policy("physical_balance",observed,weights,precision)
    assert aopt==0
    assert greedy==1
    assert guarded==greedy
    assert balanced==1
    assert p106.select_policy("guarded_voi",observed,weights,precision,
                              epsilon=.05)==guarded
    assert all(a in (0,1) for a in (aopt,greedy,guarded,balanced))
    signature=inspect.signature(p106.select_policy)
    assert set(signature.parameters)=={
        "policy","state4","weights16","inverse_precision16","epsilon"}
    assert not any(x in signature.parameters for x in (
        "oracle","effect","future_state","target_effect","source"))


def test_p106_rejects_unknown_policy_nonfinite_weights_and_bad_precision():
    W,P=matrices()
    for name,obs,wt,prec,epsilon in (
        ("evil",(0.,0.,0.,0.),W,P,.05),
        ("guarded_voi",(0.,0.,0.,0.),W,P,.66),
        ("a_opt",(0.,0.,0.,math.inf),W,P,.05),
        ("a_opt",(0.,0.,0.,0.),[[1.]],P,.05),
    ):
        with pytest.raises(ValueError):
            p106.select_policy(name,obs,wt,prec,epsilon=epsilon)
    W[0][0]=float("nan")
    with pytest.raises(ValueError,match="P106_NONFINITE_OR_NONNUMERIC"):
        p106.select_policy("a_opt",(0,0,0,0),W,P)
    W,P=matrices()
    P[2][3]=.3
    with pytest.raises(ValueError,match="P106_NON_SYMMETRIC_PRECISION"):
        p106.select_policy("greedy_goal",(0,0,0,0),W,P)
    W,P=matrices()
    for i in range(16):
        P[i][i]=-10.
    with pytest.raises(ValueError,match="P106_NEGATIVE_POSTERIOR_VARIANCE"):
        p106.select_policy("a_opt",(0,0,0,0),W,P)


def test_p106_f6_firstuse_and_counterfactuals_inaccessible():
    with pytest.raises(PermissionError,match="P106_F6_EXTERNAL_FIRSTUSE_NOT_AUTHORIZED"):
        p106.refuse_unapproved_source_firstuse(
            "CartPoleWind",requested_action=1,session="unqualified")
    source=Path(p106.__file__).read_text(encoding="utf-8")
    for banned in (
        "from causal_gym", "import causal_gym", "import subprocess",
        "import requests", "import socket", "env.do(", "env.reset(",
        "run_owner(", "dispatch_workflow(", "actions/checkout@",
    ):
        assert banned not in source
    assert not any(x.startswith("collect_external") or
                   x.startswith("run_source") for x in dir(p106))
    receipt=p106.engineering_receipt()
    assert receipt["status"]=="PREFLIGHT_ONLY_NOT_SCIENCE"
    assert receipt["science_pass"] is False
    assert receipt["new_source_effects_executed"] is False
    assert receipt["new_source_firstuse_authorized"] is False
    assert receipt["runner_dispatched"] is False
    assert receipt["owner_workflow_created"] is False
