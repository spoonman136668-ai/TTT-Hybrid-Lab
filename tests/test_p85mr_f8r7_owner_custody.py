"""F8-R7: pure DEV-only fsynced ledger; no owner effects or simulator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments import p85mr_f8r7_owner_custody as j


def first():
    return next(x for x in j.r6.planned_sessions()
                if x["phase"] == "calibration")


def make(tmp_path):
    return j.DevCustodyJournal(tmp_path)


def ready(journal, identity):
    journal.claim(identity)
    journal.append(identity, "constructor_intent")
    journal.append(identity, "reset_intent")


def sample_effect():
    return hashlib.sha256(b"FABRICATED_NOT_A_REAL_SOURCE_EFFECT").hexdigest()


def test_frozen_contract_exact_prior_source_and_firstuse_denial():
    contract = j.locked()
    assert contract["proposed_limits"]["sessions"] == 480
    assert contract["proposed_limits"]["real_source_max_effects"] == 46080
    assert contract["authority"]["real_source_firstuse"] is False
    assert contract["authority"]["owner_runner_execution"] is False
    with pytest.raises(PermissionError, match="NOT_AUTHORIZED"):
        j.refuse_real_source_firstuse()
    with pytest.raises(PermissionError, match="NOT_AUTHORIZED"):
        j.DevCustodyJournal(Path("."), classification="REAL_OWNER")


def test_exact_proposed_plan_matches_r6_and_static_accounting():
    f = j.locked()
    records = j.r6.planned_sessions()
    assert len(records) == 480
    assert len({r["identity"] for r in records}) == 480
    assert sum(r["maximum_factual_actions"] for r in records) == 46080
    assert [sum(x["phase"] == phase for x in records)
            for phase in j.PHASES] == list(j.COUNTS)
    assert f["proposed_limits"]["test_unexecuted_predictions_ceiling"] == 67584


def test_atomic_claim_reserves_96_effects_and_prevents_replay(tmp_path):
    identity = first()["identity"]
    journal = make(tmp_path)
    claim = journal.claim(identity)
    assert claim["reserved_effects"] == 96
    result = journal.recover_readonly(identity)
    assert result["reserved_factual_effects"] == 96
    assert result["source_effects_executed"] == 0
    assert not result["may_resume"]
    assert result["parent_directory_fsync_qualified"] is (__import__("os").name != "nt")
    with pytest.raises(FileExistsError):
        journal.claim(identity)
    # Even another development run cannot reclaim source identity.
    second = j.DevCustodyJournal(tmp_path, run_id="DEV-RUN-2")
    assert second.recover_readonly(identity)["global_attempt_reopen_is_read_only"]
    with pytest.raises(PermissionError, match="GLOBAL_ATTEMPT"):
        second.claim(identity)
    with pytest.raises(PermissionError, match="RESTART"):
        second.append(identity, "constructor_intent")


def test_before_effect_intents_wrong_action_receipt_and_no_replay(tmp_path):
    identity = first()["identity"]
    journal = make(tmp_path)
    ready(journal, identity)
    journal.append(identity, "action_intent", step=1, action=0)
    with pytest.raises(ValueError, match="REPLAY_OR_ORDER"):
        journal.append(identity, "action_intent", step=1, action=0)
    with pytest.raises(ValueError, match="WRONG_ACTION_RECEIPT"):
        journal.append(identity, "action_observed", step=1, action=1,
                       effect_sha256=sample_effect())
    with pytest.raises(ValueError, match="WRONG_ACTION_RECEIPT"):
        journal.append(identity, "action_observed", step=1, action=0,
                       effect_sha256="fake")
    journal.append(identity, "action_observed", step=1, action=0,
                   effect_sha256=sample_effect())
    journal.append(identity, "action_intent", step=2, action=1)
    state = journal.recover_readonly(identity)
    assert state["recorded_attempted_actions"] == 2
    assert state["unknown_or_aborted_consumes_reservation"]
    assert state["reserved_factual_effects"] == 96
    # A simulated crash cannot recover a token to replay unknown step 2.
    reopened = j.DevCustodyJournal(tmp_path)
    assert reopened.recover_readonly(identity)["may_resume"] is False
    with pytest.raises(PermissionError, match="RESTART"):
        reopened.append(identity, "action_observed", step=2, action=1,
                        effect_sha256=sample_effect())


def test_torn_or_corrupted_record_fail_closed(tmp_path):
    identity = first()["identity"]
    journal = make(tmp_path)
    ready(journal, identity)
    path = journal._events_path(identity) / "000002.json"
    path.write_bytes(b"{damaged")
    with pytest.raises(ValueError, match="TORN_OR"):
        journal.recover_readonly(identity)
    with pytest.raises(ValueError, match="TORN_OR"):
        journal.append(identity, "action_intent", step=1, action=0)


def test_constructor_reset_order_and_terminal_no_effect_rejected(tmp_path):
    identity = first()["identity"]
    journal = make(tmp_path)
    journal.claim(identity)
    with pytest.raises(ValueError, match="RESET_ORDER"):
        journal.append(identity, "reset_intent")
    journal.append(identity, "constructor_intent")
    with pytest.raises(ValueError, match="UNQUALIFIED_SESSION_COMPLETION"):
        journal.append(identity, "session_terminal", outcome="completed")
    journal.append(identity, "reset_intent")
    journal.append(identity, "session_terminal", outcome="unknown")
    with pytest.raises(ValueError, match="EVENT_AFTER_TERMINAL"):
        journal.append(identity, "action_intent", step=1, action=0)


def test_unknown_calibration_blocks_future_phase(tmp_path):
    journal = make(tmp_path)
    calibration = first()["identity"]
    ready(journal, calibration)
    journal.append(calibration, "session_terminal", outcome="unknown")
    fit_id = next(x["identity"] for x in j.r6.planned_sessions()
                  if x["phase"] == "fit")
    with pytest.raises(PermissionError, match="PREVIOUS_PHASE"):
        journal.claim(fit_id)


def test_model_query_requires_test_phase_and_distinct_unexecuted_actions(tmp_path):
    identity = first()["identity"]
    journal = make(tmp_path)
    ready(journal, identity)
    with pytest.raises(ValueError, match="MODEL_QUERY"):
        journal.append(identity, "model_query_intent", step=1, action=0)
    # Direct pure validator exercises proposed test-phase semantics without
    # creating any real trial claim or bypassing phase approvals.
    def item(i, kind, step=None, action=None, effect=None):
        return {"index":i, "session_identity":"fiction",
                "classification":j.PROTOTYPE,"previous_sha256":"GENESIS",
                "kind":kind, "step":step, "action":action,
                "effect_sha256":effect}
    events = [item(1, "constructor_intent"), item(2, "reset_intent"),
              item(3, "model_query_intent", step=1, action=0),
              item(4, "model_query_intent", step=1, action=1)]
    j.DevCustodyJournal._validate_sequence(events, phase="test")
    with pytest.raises(ValueError, match="MODEL_QUERY"):
        j.DevCustodyJournal._validate_sequence(
            events+[item(5, "model_query_intent", step=1, action=1)],
            phase="test")


def test_source_neutral_no_actions_and_no_owner_authority():
    import inspect
    source = inspect.getsource(j)
    for disallowed in ("env.do(", "env.reset(", "import causal_gym",
                       "from causal_gym", "workflow_dispatch(",
                       "subprocess.run(", "torch.save(", "requests.get(",
                       "urlopen(", "GH_TOKEN"):
        assert disallowed not in source
    assert j.locked()["authority"]["external_effects"] == 0

def test_global_marker_enforces_one_writer_across_different_sessions(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    (tmp_path / "claims").mkdir()
    (tmp_path / "events").mkdir()
    rows = j.r6.planned_sessions()
    def contender(n):
        try:
            obj = j.DevCustodyJournal(tmp_path, run_id=f"DEV-RUN-{n}")
            obj.claim(rows[n-1]["identity"])
            return "writer"
        except (PermissionError, ValueError) as exc:
            assert any(k in str(exc) for k in (
                "GLOBAL_ATTEMPT", "TORN_OR", "INVALID_EXISTING_GLOBAL"))
            return "denied"
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(contender, (1, 2)))
    assert sorted(outcomes) == ["denied", "writer"]
    # Any restart, including same DEV run label, must not claim another id.
    cold = j.DevCustodyJournal(tmp_path)
    with pytest.raises(PermissionError, match="GLOBAL_ATTEMPT"):
        cold.claim(rows[2]["identity"])
    assert cold._writer is False


def test_global_marker_survives_post_fsync_crash_fault(tmp_path, monkeypatch):
    original = j._exclusive
    def injected_after_durable_marker(path, record):
        original(path, record)
        if path.name == j.ATTEMPT_FILE:
            raise OSError("INJECTED_AFTER_MARKER_FSYNC")
    with monkeypatch.context() as scoped:
        scoped.setattr(j, "_exclusive", injected_after_durable_marker)
        with pytest.raises(OSError, match="INJECTED_AFTER"):
            j.DevCustodyJournal(tmp_path)
    assert (tmp_path / j.ATTEMPT_FILE).exists()
    reopened = j.DevCustodyJournal(tmp_path)
    with pytest.raises(PermissionError, match="GLOBAL_ATTEMPT"):
        reopened.claim(first()["identity"])


def test_forged_session_run_and_global_marker_tamper_fail_closed(tmp_path):
    obj = j.DevCustodyJournal(tmp_path)
    identity = first()["identity"]
    obj.claim(identity)
    claim_file = obj._claim_path(identity)
    row = json.loads(claim_file.read_bytes())
    row["run_id"] = "DEV-RUN-99"
    claim_file.write_bytes((json.dumps(
        row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
    with pytest.raises(ValueError, match="INVALID_OR_TAMPERED"):
        obj.recover_readonly(identity)
    (tmp_path / j.ATTEMPT_FILE).write_bytes(b"{torn")
    with pytest.raises(ValueError, match="TORN_OR"):
        obj.claim(j.r6.planned_sessions()[1]["identity"])
    with pytest.raises(ValueError, match="TORN_OR"):
        j.DevCustodyJournal(tmp_path)


def test_f8r8_freeze_precedes_code_and_denies_owner_execution():
    f = j.r8_locked()
    assert f["design"]["root_marker"] == j.ATTEMPT_FILE
    assert f["limits"]["real_source_effects"] == 0
    assert f["limits"]["owner_execution"] is False
    assert f["limits"]["independent_scientific_pass"] is False
