"""P82-MR fake-only reset/do tests. No real Causal-Gymnasium environment."""
import copy
import math
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
from p82mr_owner_session_driver import drive_toy_session
from p80mr_causalgym_trace_adapter import adapt_capture
from p79mr_toy_transfer_replay import toy_replay


from p82mr_owner_session_driver import InMemoryToySession as FakeSession


def run(session=None,**override):
    if session is None:
        session=FakeSession()
    args={
        "source_kind":"toy_engineering","split":"fit","episode_index":3,
        "source_id":"toy:test-p82", "producer_id":"toy:engineer",
        "custody_ref":"toy:local-no-witness", "world_id":"toy-p82-world",
        "session":session,
    }
    args.update(override)
    return drive_toy_session(**args)


def test_three_realized_toy_steps_and_p80_p79_trace_custody():
    session=FakeSession()
    outcome=run(session)
    assert outcome["episode_seed"]==811131
    assert session.reset_calls==1 and session.do_calls==3
    assert session.close_calls==1 and session.policy_calls==3
    assert session.reset_seed==811131
    assert outcome["observed_events"]==3
    assert outcome["status"]=="TOY_SESSION_ENGINEERING_ONLY"
    assert outcome["independent_source_attested"] is False
    assert outcome["external_first_use"] is False
    assert outcome["scientific_pass"] is False
    cap=outcome["capture"]
    rows=cap["worlds"][0]["transitions"]
    assert [row["event_id"] for row in rows]==[1,2,3]
    assert cap["worlds"][0]["probe_event_ids"]==[2,3]
    assert all(row["mode"]=="do" for row in rows)
    assert all("secret_phase" not in row and "wind" not in row for row in rows)
    adapted=adapt_capture(cap)
    assert outcome["captured_digest"]==adapted["custody"]["capture_sha256"]
    with pytest.raises(ValueError,match="P79MR_OWNER_FIRST_USE_REQUIRED"):
        toy_replay(adapted["manifest"],adapted["worlds"])


@pytest.mark.parametrize("kind",("external_simulator","physical","unknown",None))
def test_real_first_use_rejected_before_session_access(kind):
    class Poison:
        def __getattribute__(self,_):
            raise AssertionError("SOURCE_TOUCHED_BEFORE_AUTHORITY")
    with pytest.raises(PermissionError,match="P82MR_OWNER_EXTERNAL_FIRST_USE_REQUIRED"):
        run(Poison(),source_kind=kind)


@pytest.mark.parametrize("split",("dev_external","science_ood_0","science_ood_1","science_ood_2"))
def test_heldout_split_rejected_before_any_session_effect(split):
    session=FakeSession()
    with pytest.raises(PermissionError,match="P82MR_SPLIT_FIRST_USE_REQUIRED"):
        run(session,split=split)
    assert session.reset_calls==session.do_calls==session.close_calls==0


def test_invalid_id_and_episode_rejected_before_any_effect():
    s=FakeSession()
    with pytest.raises(ValueError,match="P81MR_BAD_CUSTODY_ID"):
        run(s,source_id="contains white space")
    assert s.reset_calls==0
    with pytest.raises(ValueError,match="P81MR_UNREGISTERED_EPISODE"):
        run(s,episode_index=999)
    assert s.reset_calls==0


def test_initial_and_subsequent_numpy_like_observation_are_pure_data():
    s=FakeSession(arraylike=True)
    out=run(s)
    assert out["observed_events"]==3
    assert s.reset_calls==1 and s.do_calls==3


def test_short_terminated_episode_is_not_padded_or_retried():
    s=FakeSession(limit=1)
    with pytest.raises(RuntimeError,match="P81MR_TOO_SHORT_NO_PADDING"):
        run(s)
    assert s.reset_calls==1 and s.do_calls==1 and s.close_calls==1


def test_uncertain_effect_never_replayed_or_compensated():
    s=FakeSession(behavior="uncertain_do")
    with pytest.raises(RuntimeError,match="P82MR_DO_OUTCOME_UNCERTAIN_NO_REPLAY"):
        run(s)
    assert s.reset_calls==1 and s.do_calls==1 and s.close_calls==1


@pytest.mark.parametrize("behavior",("wrong_visible","callback_twice","bad_return"))
def test_callback_and_shape_failures_are_uncertain_and_never_retried(behavior):
    s=FakeSession(behavior=behavior)
    with pytest.raises(RuntimeError,match="P82MR_DO_OUTCOME_UNCERTAIN_NO_REPLAY"):
        run(s)
    assert s.reset_calls==1 and s.do_calls==1 and s.close_calls==1


def test_do_that_did_not_consult_policy_fails_without_next_step():
    s=FakeSession(behavior="never_calls_policy")
    with pytest.raises(RuntimeError,match="P82MR_POLICY_CALLBACK_NOT_ONCE"):
        run(s)
    assert s.reset_calls==1 and s.do_calls==1 and s.close_calls==1


@pytest.mark.parametrize("behavior",("wrong_info","no_info","bad_reward"))
def test_returned_actual_action_or_reward_invalid_stops_all_further_actions(behavior):
    s=FakeSession(behavior=behavior)
    with pytest.raises(ValueError,match="P81MR_APPLIED_ACTION|P81MR_BAD_REALIZED_REWARD"):
        run(s)
    assert s.reset_calls==1 and s.do_calls==1 and s.close_calls==1


@pytest.mark.parametrize("behavior,pattern",[
    ("reset_exception","injected reset unavailable"),
    ("bad_reset","P82MR_BAD_RESET_RESULT"),
])
def test_reset_failed_once_and_cleanup_no_retry(behavior,pattern):
    s=FakeSession(behavior=behavior)
    with pytest.raises((ValueError,RuntimeError),match=pattern):
        run(s)
    assert s.reset_calls==1 and s.do_calls==0 and s.close_calls==1


def test_full_96_steps_retained_without_inventing_terminal_effect():
    s=FakeSession(limit=999)
    result=run(s)
    assert result["observed_events"]==96
    assert s.reset_calls==1 and s.do_calls==96 and s.close_calls==1
    rows=result["capture"]["worlds"][0]["transitions"]
    assert all(not e["terminated"] for e in rows)


def test_deterministic_action_schedule_independent_of_future_outcomes():
    a=run(FakeSession(limit=4),episode_index=3)
    b=run(FakeSession(limit=4),episode_index=3)
    assert [e["executed_action"] for e in a["capture"]["worlds"][0]["transitions"]]==[
        e["executed_action"] for e in b["capture"]["worlds"][0]["transitions"]]
    assert a["captured_digest"]==b["captured_digest"]


def test_code_never_imports_real_environment_or_makes_owner_calls():
    code=(Path(__file__).resolve().parents[1]/"experiments"/"p82mr_owner_session_driver.py").read_text()
    for forbidden in ("import causal_gym","import gymnasium","subprocess",
                      "requests.","urllib","torch","if __name__",
                      "os.system","schedule.create","runner.dispatch"):
        assert forbidden not in code


@pytest.mark.parametrize("label", ("toy_engineering",))
def test_mislabelled_external_like_session_rejected_before_any_attribute_access(label):
    class ExternalLike:
        def __getattribute__(self, key):
            raise AssertionError("P83MR_EXTERNAL_SOURCE_OBJECT_TOUCHED")
    with pytest.raises(PermissionError, match="P82MR_UNTRUSTED_SESSION_CLASS"):
        run(ExternalLike(), source_kind=label)


def test_non_substitutable_source_even_if_it_implements_reset_do_methods():
    class RealLike:
        def __init__(self):
            self.side_effect_calls = 0

        def reset(self, *, seed):
            self.side_effect_calls += 1
            raise RuntimeError("SHOULD_NEVER_EXECUTE")

        def do(self, policy):
            self.side_effect_calls += 1
            raise RuntimeError("SHOULD_NEVER_EXECUTE")
    session = RealLike()
    with pytest.raises(PermissionError, match="P82MR_UNTRUSTED_SESSION_CLASS"):
        run(session, source_kind="toy_engineering")
    assert session.side_effect_calls == 0


def test_subclass_and_instance_method_replacement_denied():
    class AttackSubclass(FakeSession):
        def do(self, policy):
            raise RuntimeError("SHOULD_NEVER_EXECUTE")
    with pytest.raises(PermissionError, match="P82MR_UNTRUSTED_SESSION_CLASS"):
        run(AttackSubclass(), source_kind="toy_engineering")
    session = FakeSession()
    with pytest.raises(AttributeError):
        session.do = lambda policy: (_ for _ in ()).throw(
            RuntimeError("SHOULD_NEVER_EXECUTE"))
    assert run(session)["observed_events"] == 3


def test_reject_null_and_all_non_fixture_builtin_objects():
    for source in (None, object(), {}, {"reset": "fake", "do": "fake"}):
        with pytest.raises(PermissionError, match="P82MR_UNTRUSTED_SESSION_CLASS"):
            run(source, source_kind="toy_engineering")
