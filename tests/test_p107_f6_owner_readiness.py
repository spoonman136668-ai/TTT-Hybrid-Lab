"""P107: source-neutral F6 owner contract, no external source sessions or actions."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from experiments import p107_f6_owner_readiness as p107


def fit_fixture(survivals=None):
    records={}
    for name in p107.CONTROL_SET:
        steps=30 if name=="a_opt" else 28 if name=="greedy_goal" else 23
        if survivals and name in survivals:
            steps=survivals[name]
        records[name]=[{
            "block":block,"index":idx,"survival_steps":steps,
            "physical_loss":0.6 if name=="a_opt" else .5,
            "terminated":True,"truncated":False,
            "receipt_ref":f"fixture:{name}:{block}:{idx}",
        } for block in range(3) for idx in range(16)]
    return records


def test_p107_precise_source_first_freeze_and_explicit_protocol_fork():
    f=p107.frozen()
    assert p107.FREEZE_COMMIT=="fb74b8d245e7c9d1e13403cc35ee36487f0a781b"
    assert p107.FREEZE_BLOB=="95c5d4fe840287c1c86aacd54b6eadd60f2e0d0f"
    assert f["parent"]["pr"]==233
    assert f["parent"]["sha"]=="9368dd4015b7e9151ff525f88d5a64e831f1713b"
    assert f["dual_protocol_conflict"]["first"]["freeze_blob"]==(
        "3a2cee9454edf2e2f9937e35492d618e30f79de0")
    assert f["dual_protocol_conflict"]["other"]["freeze_blob"]==(
        "42db0228c1c4561333058a3b9d849b8c3db1183c")
    assert f["dual_protocol_conflict"]["first"]["FIT"]!=(
        f["dual_protocol_conflict"]["other"]["FIT_seed_bases"])
    assert f["dual_protocol_conflict"]["status"]=="UNRESOLVED_NO_F6_ACTIVATION"
    assert f["dual_protocol_conflict"]["must_not_implicitly_pick_one"] is True
    assert f["owner"]["run_owner_scoped_authorization"]=="REQUIRED_NOT_GRANTED"
    assert f["future_science"]["F6_effect_attempts"]==0
    assert f["future_science"]["scientific_pass"] is False
    assert f["future_science"]["OOD_unopened"] is True


def test_p107_only_source_neutral_dryrun_fit_and_no_cal_seeds(tmp_path):
    ledger=p107.FixtureLedger(tmp_path)
    proposed=p107.engineering_receipt()
    assert proposed["status"]=="SOURCE_NEUTRAL_ENG_ONLY_BLOCK_F6"
    assert proposed["proposed_sessions_if_authorized"]==384
    assert proposed["proposed_max_effects_if_authorized"]==36864
    assert proposed["real_effects"]==0
    assert proposed["cal_unlocked"] is False
    assert proposed["owner_firstuse_authorized"] is False
    first=ledger.claim("FIT",0,0,"guarded_voi")
    assert first=="fixture-FIT-1820000000-guarded_voi"
    with pytest.raises(FileExistsError):
        p107.FixtureLedger(tmp_path).claim("FIT",0,0,"guarded_voi")
    for args in (("CAL",0,0,"guarded_voi"),
                 ("OOD",0,0,"a_opt"),("FIT",3,0,"a_opt"),
                 ("FIT",0,16,"greedy_goal"),
                 ("FIT",0,0,"unqualified"),
                 ("FIT",True,0,"a_opt")):
        with pytest.raises(PermissionError,match="P107_CAL_OOD_AND_UNKNOWN_SOURCE_DENIED"):
            ledger.claim(*args)


def test_p107_pre_effect_exclusive_intent_and_same_action_receipt(tmp_path):
    ledger=p107.FixtureLedger(tmp_path)
    name=ledger.claim("FIT",1,2,"a_opt")
    before=(.0,.1,.02,-.1)
    prep=ledger.intend(name,0,1,before)
    assert prep["action"]==1 and len(prep["intent_sha256"])==64
    # A second action cannot be started without factual confirmation.
    with pytest.raises(FileNotFoundError):
        ledger.intend(name,1,0,before)
    with pytest.raises(FileExistsError):
        ledger.intend(name,0,1,before)
    with pytest.raises(ValueError,match="P107_WRONG_OR_UNCOMMITTED_ACTION_CREDIT"):
        ledger.confirm_fixture(name,0,prep["intent_sha256"],0,
                               before,1.,False,False,"fixture:bad-action")
    with pytest.raises(PermissionError,match="P107_ONLY_FIXTURE_RECEIPT_PERMITTED"):
        ledger.confirm_fixture(name,0,prep["intent_sha256"],1,
                               before,1.,False,False,"ghrun:real-f6")
    with pytest.raises(ValueError,match="P107_WRONG_OR_UNCOMMITTED_ACTION_CREDIT"):
        ledger.confirm_fixture(name,0,"0"*64,1,
                               before,1.,False,False,"fixture:bad-hash")
    confirmed=ledger.confirm_fixture(name,0,prep["intent_sha256"],1,
                       (.02,.08,.03,-.11),.5,False,False,"fixture:step0")
    assert confirmed["step"]==0 and confirmed["not_external_science"] is True
    with pytest.raises(FileExistsError):
        ledger.confirm_fixture(name,0,prep["intent_sha256"],1,
                       (.02,.08,.03,-.11),.5,False,False,"fixture:duplicate")
    second=ledger.intend(name,1,0,(.02,.08,.03,-.11))
    assert second["action"]==0
    ledger.confirm_fixture(name,1,second["intent_sha256"],0,
                           (.2,.0,.04,0),.2,True,False,"fixture:terminal")
    with pytest.raises(PermissionError,match="P107_POST_TERMINAL_ATTEMPT_DENIED"):
        ledger.intend(name,2,0,(.2,.0,.04,0.))


def test_p107_unknown_effect_is_quarantined_across_new_ledger_object(tmp_path):
    ledger=p107.FixtureLedger(tmp_path)
    name=ledger.claim("FIT",2,15,"physical_balance")
    first=ledger.intend(name,0,0,(.02,.03,-.04,0.))
    ledger.mark_unknown(name,0)
    reopened=p107.FixtureLedger(tmp_path)
    with pytest.raises(PermissionError,match="P107_UNKNOWN_EFFECT_REPLAY_DENIED"):
        reopened.confirm_fixture(name,0,first["intent_sha256"],0,
                                 (0,0,0,0),0.,True,False,"fixture:after-crash")
    with pytest.raises(PermissionError,match="P107_UNKNOWN_EFFECT_REPLAY_DENIED"):
        reopened.intend(name,1,1,(.02,.03,-.04,0.))
    with pytest.raises(FileExistsError):
        reopened.claim("FIT",2,15,"physical_balance")


def test_p107_invalid_observation_nan_action_tampering_and_unknown_session(tmp_path):
    ledger=p107.FixtureLedger(tmp_path)
    name=ledger.claim("FIT",0,1,"greedy_goal")
    for action,obs in ((2,[0]*4),(True,[0]*4),(0,[0,0,float("nan"),0]),
                       (0,[0,0,0]),(0,[0,0,0,True])):
        with pytest.raises(ValueError):
            ledger.intend(name,0,action,obs)
    p=ledger.intend(name,0,0,(0,0,0,0))
    for after,reward in (([0,0,float("inf"),0],0),([0,0,0,0],float("nan"))):
        with pytest.raises(ValueError):
            ledger.confirm_fixture(name,0,p["intent_sha256"],0,
                                   after,reward,False,False,"fixture:negative")
    with pytest.raises(ValueError,match="P107_BAD_TERMINAL_STATUS"):
        ledger.confirm_fixture(name,0,p["intent_sha256"],0,
                               (0,0,0,0),0,0,False,"fixture:bool")
    with pytest.raises(ValueError):
        ledger.intend("../malicious",0,0,(0,0,0,0))


def test_p107_mock_fit_only_best_conventional_frozen_not_science():
    fake=fit_fixture()
    result=p107.fit_select_fixture(fake)
    assert result["selected"]=="a_opt"
    assert result["competence_fixture_only"]
    assert result["real_FIT_train_checkpoint"]=="UNVERIFIED"
    assert result["real_CAL_admission"] is False
    assert result["science_pass"] is False
    tie=fit_fixture({"a_opt":30,"greedy_goal":30,"physical_balance":23})
    assert p107.fit_select_fixture(tie)["selected"]=="greedy_goal"
    weak=fit_fixture({"a_opt":14,"greedy_goal":15,"physical_balance":16})
    s=p107.fit_select_fixture(weak)
    assert s["selected"]=="physical_balance"
    assert s["competence_fixture_only"] is False
    assert s["real_CAL_admission"] is False
    with pytest.raises(ValueError,match="P107_INCOMPLETE_STRONG_CONVENTIONAL_ARMS"):
        p107.fit_select_fixture({"a_opt":fake["a_opt"]})
    missing=copy.deepcopy(fake)
    missing["a_opt"].pop()
    with pytest.raises(ValueError,match="P107_INCOMPLETE_FIT_EPISODES"):
        p107.fit_select_fixture(missing)
    duplicate=copy.deepcopy(fake)
    duplicate["a_opt"][1]["index"]=duplicate["a_opt"][0]["index"]
    with pytest.raises(ValueError,match="P107_MISSING_OR_DUPLICATE_FIT_WORLD"):
        p107.fit_select_fixture(duplicate)
    fabricated=copy.deepcopy(fake)
    fabricated["greedy_goal"][0]["receipt_ref"]="ghrun:unqualified"
    with pytest.raises(ValueError,match="P107_MISSING_OR_FORGED_FIT_OUTCOME"):
        p107.fit_select_fixture(fabricated)


def test_p107_no_effectful_firstuse_api_or_external_imports():
    with pytest.raises(PermissionError,match="P107_EXTERNAL_F6_FIRSTUSE_UNAUTHORIZED_PROTOCOL_CONFLICT"):
        p107.deny_external_firstuse(
            scope="F6",action=0,runner="TTT-HYBRID-GPU-WIN")
    src=Path(p107.__file__).read_text("utf-8")
    for forbidden in ("import causal_gym","from causal_gym","import subprocess",
                      "import requests","import socket",
                      "env.do(","env.reset(","dispatch_owner(","run_external("):
        assert forbidden not in src
    c=p107.engineering_receipt()
    assert c["dual_protocol_fork_unresolved"]
    assert c["runner_dispatched"] is False
    assert c["new_workflows"] is False
    assert c["real_source_sessions"]==0
    assert c["scientific_pass"] is False
