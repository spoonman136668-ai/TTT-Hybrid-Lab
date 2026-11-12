"""P115 exact frozen choice matrix and POSIX dry-run recovery controls."""
from __future__ import annotations

import copy
import json
import os

import pytest

from experiments import p115_f6_owner_scope_preflight as p115


def test_p115_protocol_matrix_does_not_select_between_original_forks():
    r=p115.protocol_matrix()
    assert r["legacy_A4_FIT_sessions"]==192
    assert r["alternative_B4_FIT_sessions"]==384
    assert r["proposed_A5_FIT_sessions"]==240
    assert r["proposed_A5_FIT_max_effects"]==23040
    assert r["SSM_training_worlds"]==24
    assert r["SSM_seed_selection_worlds"]==8
    assert r["FIT_conventional_selection_worlds_per_arm"]==16
    assert r["protocol_selected"] is False
    assert r["owner_native_storage_qualified"] is False
    assert r["real_FIT_source_authorized"] is False
    assert r["science_pass"] is False
    frozen=p115.locked()
    assert frozen["proposed_only"]["not_authorized_revision"] is True
    assert frozen["proposed_only"]["original_P110_P112_unchanged"] is True
    with pytest.raises(PermissionError,match="P115_F6_FIRST_USE_AND_PROTOCOL_CHOICE_NOT_AUTHORIZED"):
        p115.deny_source(split="FIT",policy="guarded_voi")


def test_p115_posix_claim_intent_before_outcome_and_cold_restart(tmp_path):
    if os.name!="posix":
        with pytest.raises(PermissionError,match="P115_WSL_OWNER_DURABILITY_UNVERIFIED_WINDOWS"):
            p115.FixtureJournal(tmp_path)
        return
    ledger=p115.FixtureJournal(tmp_path)
    ident=ledger.claim(0,0,"guarded_voi","a"*64)
    with pytest.raises(FileExistsError):
        ledger.claim(0,0,"guarded_voi","a"*64)
    p=ledger.intend(ident,0,1,[0.,0.,0.,0.],"a"*64)
    recovered=p115.FixtureJournal(tmp_path)
    assert recovered.audit(ident)["status"]=="UNCONFIRMED_INTENT_BLOCKED"
    with pytest.raises(FileExistsError):
        recovered.intend(ident,0,1,[0.,0.,0.,0.],"a"*64)
    with pytest.raises(ValueError,match="P115_ACTION_INTENT_MISMATCH"):
        recovered.confirm_fixture(ident,0,p,0,[1.,0.,0.,0.],1.,False,False,"fixture:fake")
    with pytest.raises(PermissionError,match="P115_NOT_A_FIXTURE_RECEIPT"):
        recovered.confirm_fixture(ident,0,p,1,[1.,0.,0.,0.],1.,False,False,"ghrun:unqualified")
    recovered.confirm_fixture(ident,0,p,1,[.01,0.,0.,0.],1.,False,False,"fixture:one")
    assert recovered.audit(ident)["status"]=="CLOSED_FIXTURE_ONLY"
    q=recovered.intend(ident,1,0,[.01,0.,0.,0.],"a"*64)
    recovered.quarantine(ident,1)
    assert p115.FixtureJournal(tmp_path).audit(ident)["status"]=="UNKNOWN_BLOCKED"
    with pytest.raises(PermissionError,match="P115_UNKNOWN_EFFECT_NO_REPLAY"):
        recovered.confirm_fixture(ident,1,q,0,[.02,0.,0.,0.],1.,True,False,"fixture:two")
    with pytest.raises(PermissionError,match="P115_UNKNOWN_EFFECT_NO_REPLAY"):
        recovered.intend(ident,2,1,[.02,0.,0.,0.],"a"*64)


def test_p115_invalid_scope_model_digest_symlinks_and_terminal(tmp_path):
    if os.name!="posix":
        pytest.skip("POSIX directory-fsync required, unsupported on native Windows")
    ledger=p115.FixtureJournal(tmp_path)
    for args in ((0,0,"hash_passive","b"*64),
                 (0,16,"guarded_voi","b"*64),
                 (0,0,"guarded_voi","bad")):
        with pytest.raises((ValueError,PermissionError)):
            ledger.claim(*args)
    session=ledger.claim(2,15,"ssm256_model_based","b"*64)
    with pytest.raises(ValueError,match="P115_INVALID_INTENT_OR_MODEL"):
        ledger.intend(session,0,1,[0,0,0,0],"c"*64)
    intent=ledger.intend(session,0,0,[0,0,0,0],"b"*64)
    ledger.confirm_fixture(session,0,intent,0,[0,.01,0,0],1,True,False,"fixture:terminal")
    with pytest.raises(PermissionError,match="P115_AFTER_TERMINAL"):
        ledger.intend(session,1,1,[0,.01,0,0],"b"*64)
    assert ledger.audit(session)["real_source_actions"]==0
    with pytest.raises(ValueError,match="P115_BAD_SESSION"):
        ledger.audit("../etc/passwd")
    alias=tmp_path/"shortcut"
    alias.symlink_to(tmp_path/session,target_is_directory=True)
    with pytest.raises(ValueError,match="P115_BAD_SESSION"):
        ledger.audit("shortcut")
    with pytest.raises(ValueError,match="P115_REQUIRED_UNLINKED_DIRECTORY"):
        p115.FixtureJournal(alias)


def test_p115_tamper_and_crash_controls(tmp_path):
    if os.name!="posix":
        pytest.skip("POSIX filesystem rehearsal")
    ledger=p115.FixtureJournal(tmp_path)
    name=ledger.claim(1,0,"a_opt","e"*64)
    prior=ledger.intend(name,0,1,[.0,.0,.0,.0],"e"*64)
    ledger.confirm_fixture(name,0,prior,1,[0,0,0,.01],1,False,False,"fixture:receipt")
    with pytest.raises(ValueError,match="P115_BROKEN_PREVIOUS_STATE"):
        ledger.intend(name,1,0,[1,1,1,1],"e"*64)
    path=tmp_path/name/"000-outcome.json"
    row=json.loads(path.read_text())
    row["external_source"]=True
    path.write_text(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n")
    with pytest.raises(ValueError,match="P115_TAMPERED_OUTCOME"):
        p115.FixtureJournal(tmp_path).audit(name)


def test_p115_no_external_source_or_new_workflow_entry_point():
    source=open(p115.__file__,encoding="utf-8").read()
    for prohibited in ("import causal_gym","from causal_gym","env.do(",
                       "env.reset(","subprocess.run(","requests.get(",
                       "dispatch_workflow(","owner_execute(","torch.load(",
                       "os.system("):
        assert prohibited not in source
