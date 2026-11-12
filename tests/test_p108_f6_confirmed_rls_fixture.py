"""P108 frozen confirmed-effect fixture RLS: strict full SPD and credit custody."""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import pytest
import torch

from experiments import p106_f6_source_neutral_policies as p106
from experiments import p107_f6_owner_readiness as p107
from experiments import p108_f6_confirmed_rls_fixture as p108


def _setup(tmp_path,policy="a_opt"):
    ledger=p107.FixtureLedger(tmp_path)
    session=ledger.claim("FIT",0,0,policy)
    model=p108.initial()
    before=(.03,-.04,.012,.09)
    prop=p108.propose(ledger,session,model,policy,before,0)
    return ledger,session,model,before,prop


def _receipt(ledger,session,prop,after=(.08,-.06,.009,.06),
             reward=.7,terminated=False,truncated=False):
    return ledger.confirm_fixture(
        session,prop.step,prop.intent_sha256,prop.action,
        after,reward,terminated,truncated,
        f"fixture:p108:{prop.step}")


def test_p108_prospective_pinned_custody_no_F6_authorization():
    f=p108.frozen()
    assert p108.FREEZE_COMMIT=="122bbf7201e62cb4c2dcdfcd2089af5f963ed5c5"
    assert f["source_identity"]["parent_head"]=="4704a7a7c0beec44c5ca4e4fe86ed3f55703fe7d"
    assert f["science"]["scientific_pass"] is False
    assert f["science"]["external_F6_effects"]==0
    assert f["science"]["source_protocol_fork_unresolved"] is True
    assert f["science"]["trained_SSM_control_unverified"] is True
    assert f["science"]["owner_firstuse_authorized"] is False
    assert f["qualification"]["mandatory_G4_Windows_Ubuntu"] is True
    assert f["qualification"]["no_external_firstuse"] is True
    root=Path(__file__).resolve().parents[1]
    assert p106._git_blob((root/"experiments/p108_f6_confirmed_rls_freeze.json").read_bytes())==p108.FREEZE_BLOB
    for path,blob in p108.BLOBS.items():
        assert p106._git_blob((root/path).read_bytes())==blob


def test_p108_full_spectrum_SPD_rejects_invisible_corruption():
    good=p108.initial()
    assert p108.validate(good) is good
    assert good.count==0
    assert torch.equal(good.W,torch.zeros((16,16),dtype=torch.float64))
    assert torch.equal(good.P,torch.eye(16,dtype=torch.float64)*2)
    before=(.1,.2,.02,-.05)
    bad=good.P.clone()
    bad[7,7]=-1.
    # P80 action keys have zeros in indices >=7, so P106 alone cannot
    # detect this negative eigenvalue. P108's full spectrum MUST reject.
    assert bool((p108._key(before,0)[7:]==0).all())
    assert bool((p108._key(before,1)[7:]==0).all())
    assert p106.select_policy("a_opt",before,good.W.tolist(),bad.tolist()) in (0,1)
    with pytest.raises(ValueError,match="P108_NOT_FULL_16D_SPD"):
        p108.validate(replace(good,P=bad))
    with pytest.raises(ValueError,match="P108_NOT_FULL_16D_SPD"):
        p108.validate(replace(good,P=torch.zeros_like(good.P)))
    asym=good.P.clone()
    asym[0,1]=.5
    with pytest.raises(ValueError,match="P108_NONSYMMETRIC_FULL_PRECISION"):
        p108.validate(replace(good,P=asym))
    nan=good.W.clone()
    nan[0,0]=float("nan")
    with pytest.raises(ValueError,match="P108_BAD_MODEL_STATE"):
        p108.validate(replace(good,W=nan))
    with pytest.raises(ValueError,match="P108_BAD_MODEL_STATE"):
        p108.validate(replace(good,count=True))


def test_p108_exact_public_P80_preaction_key_and_no_oracle():
    for before in ((0.,0.,0.,0.),(.3,-.14,.07,1.2)):
        for action in (0,1):
            values=[1.]+[math.tanh(x/scale) for x,scale in zip(before,p106.SCALES)]
            values += [float(action==0),float(action==1)]+[0.]*9
            denom=math.sqrt(math.fsum(x*x for x in values))
            expected=torch.tensor([v/denom for v in values],dtype=torch.float64)
            assert torch.equal(p108._key(before,action),expected)
    with pytest.raises(ValueError,match="P108_ONLY_AUTHENTIC_BINARY_ACTIONS"):
        p108._key((0.,0.,0.,0.),2)
    with pytest.raises(ValueError,match="P107_NONFINITE_OR_NONNUMERIC"):
        p108._key((float("nan"),0,0,0),0)
    # No oracle, unexecuted effect or future target arguments.
    model=p108.initial()
    with pytest.raises(TypeError):
        p108.read(model,(0,0,0,0),0,unexecuted_effect=torch.ones(16))


def test_p108_only_confirmed_selected_action_updates_exact_P101_RLS(tmp_path):
    ledger,session,model,before,prop=_setup(tmp_path,"guarded_voi")
    assert model.count==0 and prop.step==0
    assert prop.action in (0,1)
    assert prop.model_sha256==p108.model_sha(model)
    assert len(prop.seal_sha256)==64
    intent=p107._load(tmp_path/session/"000-intent.json")
    assert intent["action"]==prop.action
    assert tuple(intent["before"])==before
    # No RLS state update until a confirmed factual fixture outcome exists.
    with pytest.raises(FileNotFoundError):
        p108.confirm_fixture(ledger,session,model,prop,outcome_sha256="0"*64)
    assert torch.equal(model.W,p108.initial().W)
    saved=p108.snapshot(model)
    after=(.08,-.06,.009,.06)
    reward=.7
    receipt=_receipt(ledger,session,prop,after,reward)
    deltas=[a-b for b,a in zip(before,after)]
    expected_effect=[math.tanh(d/scale) for d,scale in zip(deltas,p106.SCALES)]
    expected_effect += [math.tanh(reward),0.,0.]+[0.]*9
    factual=torch.tensor(expected_effect,dtype=torch.float64)
    key=torch.tensor(p106.action_key(before,prop.action),dtype=torch.float64)
    g=model.P@key
    den=1.+key@g
    expectedW=model.W+torch.outer(factual-model.W@key,g)/den
    expectedP=model.P-torch.outer(g,g)/den
    updated=p108.confirm_fixture(ledger,session,model,prop,
                                  outcome_sha256=receipt["outcome_sha256"])
    assert torch.equal(updated.W,expectedW)
    assert torch.equal(updated.P,expectedP)
    assert updated.count==1
    assert updated.journal_sha256!=model.journal_sha256
    assert torch.equal(saved.W,model.W)
    assert torch.equal(saved.P,model.P)
    assert p108.model_sha(updated)!=p108.model_sha(model)
    assert torch.equal(p108.read(updated,before,prop.action),expectedW@key)
    with pytest.raises(ValueError,match="P108_MISMATCHED_OR_TAMPERED_PROPOSAL"):
        p108.confirm_fixture(ledger,session,updated,prop,
                             outcome_sha256=receipt["outcome_sha256"])


def test_p108_wrong_credit_unknown_effect_receipt_digest_and_tamper(tmp_path):
    ledger,session,model,before,prop=_setup(tmp_path,"a_opt")
    other=replace(prop,action=1-prop.action)
    with pytest.raises(ValueError,match="P108_MISMATCHED_OR_TAMPERED_PROPOSAL"):
        p108.confirm_fixture(ledger,session,model,other,outcome_sha256="0"*64)
    wrong_session=replace(prop,session="fixture-FIT-1820000000-greedy_goal")
    with pytest.raises(ValueError,match="P108_MISMATCHED_OR_TAMPERED_PROPOSAL"):
        p108.confirm_fixture(ledger,session,model,wrong_session,outcome_sha256="0"*64)
    with pytest.raises(PermissionError,match="P107_ONLY_FIXTURE_RECEIPT_PERMITTED"):
        ledger.confirm_fixture(session,0,prop.intent_sha256,prop.action,
                               before,.2,False,False,"ghrun:external")
    with pytest.raises(ValueError,match="P107_WRONG_OR_UNCOMMITTED_ACTION_CREDIT"):
        ledger.confirm_fixture(session,0,prop.intent_sha256,1-prop.action,
                               before,.2,False,False,"fixture:wrong")
    confirm=_receipt(ledger,session,prop)
    with pytest.raises(ValueError,match="P108_UNAUTHENTICATED_FIXTURE_OUTCOME"):
        p108.confirm_fixture(ledger,session,model,prop,outcome_sha256="0"*64)
    # A changed but still canonical JSON outcome cannot be used with the
    # original receipt digest: immutable confirmed outcome is enforced.
    path=tmp_path/session/"000-outcome.json"
    data=json.loads(path.read_bytes())
    data["reward"]=data["reward"]+1.
    path.write_bytes(p107._canonical(data)+b"\n")
    with pytest.raises(ValueError,match="P108_UNAUTHENTICATED_FIXTURE_OUTCOME"):
        p108.confirm_fixture(ledger,session,model,prop,
                             outcome_sha256=confirm["outcome_sha256"])
    assert torch.equal(model.W,p108.initial().W)


def test_p108_unknown_effect_quarantine_terminal_and_state_chain(tmp_path):
    ledger,session,model,before,prop=_setup(tmp_path,"a_opt")
    ledger.mark_unknown(session,0)
    with pytest.raises(PermissionError,match="P108_UNKNOWN_EFFECT_REPLAY_DENIED"):
        p108.confirm_fixture(ledger,session,model,prop,outcome_sha256="0"*64)
    with pytest.raises(PermissionError,match="P107_UNKNOWN_EFFECT_REPLAY_DENIED"):
        p108.propose(ledger,session,model,"a_opt",before,0)
    ledger2=p107.FixtureLedger(tmp_path)
    with pytest.raises(PermissionError,match="P108_UNKNOWN_EFFECT_REPLAY_DENIED"):
        p108.confirm_fixture(ledger2,session,model,prop,outcome_sha256="0"*64)
    different=ledger2.claim("FIT",0,1,"physical_balance")
    pre=(0.,0.,.03,.0)
    proposal=p108.propose(ledger2,different,p108.initial(),"physical_balance",pre,0)
    result=ledger2.confirm_fixture(different,0,proposal.intent_sha256,proposal.action,
                               (.1,0.,.02,0.),.4,False,False,"fixture:fact")
    nextmodel=p108.confirm_fixture(ledger2,different,p108.initial(),proposal,
                                   outcome_sha256=result["outcome_sha256"])
    with pytest.raises(PermissionError,match="P108_STATE_OR_TERMINAL_CHAIN_DRIFT"):
        p108.propose(ledger2,different,nextmodel,"physical_balance",(999.,0,0,0),1)
    # Exact physical-state chaining allows a new action after the first
    # actual *fixture* outcome, never before.
    follow=p108.propose(ledger2,different,nextmodel,"physical_balance",(.1,0.,.02,0.),1)
    assert follow.step==1


def test_p108_cold_model_restoration_read_only_after_independent_task(tmp_path):
    ledger,session,model,before,prop=_setup(tmp_path,"greedy_goal")
    first=_receipt(ledger,session,prop)
    trained=p108.confirm_fixture(ledger,session,model,prop,
                                 outcome_sha256=first["outcome_sha256"])
    cold=p108.snapshot(trained)
    old_read=p108.read(cold,before,0).clone()
    other=ledger.claim("FIT",1,0,"physical_balance")
    second=p108.initial()
    trial=p108.propose(ledger,other,second,"physical_balance",
                       (.2,.3,.04,.6),0)
    observed=ledger.confirm_fixture(other,0,trial.intent_sha256,trial.action,
                           (.25,.31,.02,.5),.3,True,False,"fixture:other")
    _=p108.confirm_fixture(ledger,other,second,trial,
                            outcome_sha256=observed["outcome_sha256"])
    assert torch.equal(old_read,p108.read(cold,before,0))
    assert torch.equal(cold.W,trained.W)
    assert torch.equal(cold.P,trained.P)
    assert p108.model_sha(cold)==p108.model_sha(trained)


def test_p108_does_not_dispatch_source_or_train_native_hybrid():
    with pytest.raises(PermissionError,match="P108_F6_EXTERNAL_SCIENCE_FIRSTUSE_DENIED"):
        p108.refuse_external_firstuse("F6",action=0)
    source=Path(p108.__file__).read_text("utf-8")
    for bad in ("from causal_gym","import causal_gym","import requests",
                "import socket","import subprocess","env.do(","env.reset(",
                "dispatch_workflow(","run_owner(","firstuse_authorize("):
        assert bad not in source
    # No real SSM/native Hybrid training is imported or claimed.
    assert "mamba_ssm" not in source
    assert "import ttt" not in source
