"""P105 source-neutral F3 observation-support refusal; NO external source use."""
import copy
import math
from pathlib import Path

import pytest

from experiments import p105_f3_action_support_preflight as p105


def fixture_fit():
    return {
        "schema":p105.CAPTURE_SCHEMA,
        "upstream_repo":p105.UPSTREAM_REPO,
        "upstream_commit":p105.UPSTREAM_SHA,
        "upstream_path":p105.UPSTREAM_PATH,
        "upstream_blob":p105.UPSTREAM_BLOB,
        "source_id":"p85mr-fit-fixture",
        "producer_id":"unit-test-fake-producer",
        "custody_ref":"unit-test-no-owner-run",
        "worlds":[{
            "world_id":"fit-fixture-000",
            "episode_seed":7,
            "probe_event_ids":[2],
            "transitions":[{
                "event_id":1,"state_before":[0.,0.,0.,0.],
                "executed_action":0,
                "state_after":[.1,0.,0.,0.],
                "reward":.1,"terminated":False,"truncated":False,
                "mode":"do","receipt_ref":"unit-fixture:one"
            },{
                "event_id":2,"state_before":[.1,0.,0.,0.],
                "executed_action":1,
                "state_after":[.2,0.,0.,0.],
                "reward":.2,"terminated":True,"truncated":False,
                "mode":"do","receipt_ref":"unit-fixture:two",
            }],
        }],
    }


def test_p105_frozen_authentic_source_identity_and_no_new_firstuse():
    f=p105.locked_protocol()
    assert f["stage"]=="FROZEN_NON_EFFECTFUL_FEASIBILITY_ONLY_NOT_F6_FIRST_USE"
    assert p105.FREEZE_COMMIT=="2b687182b90a844a5af210c39502a9a2f4bcc9ae"
    assert f["exact_parent"]["pr"]==224
    assert f["exact_parent"]["f3_original_artifact_id"]==13878284321
    assert f["exact_parent"]["f3_original_artifact_zip_sha256"]==(
        "a04697bdfba1eb93548244329f4f76fb807489aa7766f4a343de897a56caceb5")
    assert f["exact_parent"]["f3_fit_capture_sha256"]==(
        "b495d6c41fd54c6955fd83ead7569282146af45bc68b3a3dba2a63808e1d6f7b")
    assert f["authority"]["new_owner_effect_execution_authorized"] is False
    assert f["authority"]["distinct_firstuse_requires_exact_separate_approval"] is True
    assert f["proposed_F6_future_protocol_only"]["not_yet_authorized_or_executed"] is True
    assert f["frozen_conclusion"].startswith("NO_GO_DIRECT_OFFLINE")
    assert p105._git_blob(Path(__file__).resolve().parents[1].joinpath(
        "experiments/p105_authentic_action_support_freeze.json").read_bytes())==p105.FREEZE_BLOB


def test_p105_factual_fixture_has_no_direct_offpolicy_counterfactual():
    s=p105.observational_support(fixture_fit())
    assert s["worlds"]==1
    assert s["confirmed_effects"]==2
    assert s["action_0"]==1 and s["action_1"]==1
    assert s["unique_exact_states"]==2
    assert s["exact_state_with_both_actions"]==0
    assert s["direct_offpolicy_counterfactual_effect_qualified"] is False
    assert s["science_pass"] is False
    assert s["authentic_source_effects_executed"] is False
    assert s["new_F6_source_authorized_or_generated"] is False


def test_p105_even_paired_identical_states_do_not_prove_factual_counterfactual():
    f=fixture_fit()
    # Pretending identical observations occurred with both actions does not
    # prove the same exogenous noise or true potential outcomes were observed.
    f["worlds"][0]["transitions"][1]["state_before"]=[0.,0.,0.,0.]
    s=p105.observational_support(f)
    assert s["exact_state_with_both_actions"]==1
    assert s["unique_exact_states"]==1
    assert s["repeated_exact_state_events"]==1
    assert s["direct_offpolicy_counterfactual_effect_qualified"] is False


def test_p105_fail_closed_on_any_cal_or_unpinned_source_or_broken_effect():
    orig=fixture_fit()
    def bad(fn, expected):
        f=copy.deepcopy(orig)
        fn(f)
        with pytest.raises(ValueError,match=expected):
            p105.observational_support(f)

    bad(lambda f:f.__setitem__("source_id","p85mr-cal-fixture"),
        "P105_ONLY_CONSUMED_FIT_PERMITTED")
    bad(lambda f:f.__setitem__("upstream_commit","different"),
        "P105_WRONG_EXTERNAL_SOURCE")
    bad(lambda f:f["worlds"][0]["transitions"][0].__setitem__("executed_action",2),
        "P105_NOT_CONFIRMED_BINARY_DO")
    bad(lambda f:f["worlds"][0]["transitions"][0].__setitem__("mode","observe"),
        "P105_NOT_CONFIRMED_BINARY_DO")
    bad(lambda f:f["worlds"][0]["transitions"][0].__setitem__("reward",float("nan")),
        "P105_INVALID_F3_REWARD")
    bad(lambda f:f["worlds"][0]["transitions"][0].__setitem__("state_after",[float("nan")]*4),
        "P105_INVALID_F3_STATE")
    bad(lambda f:f["worlds"][0]["transitions"][0].__setitem__("logging_propensity",.5),
        "P105_BAD_EVENT_FIELDS")
    bad(lambda f:f["worlds"][0].__setitem__("probe_event_ids",[1,2]),
        "P105_PROBE_RECEIPT_DRIFT")
    with pytest.raises(ValueError,match="P105_UNBOUND_RAW_CAPTURE"):
        p105.observational_support(orig,original_bytes=b"some historical file")
    with pytest.raises(ValueError,match="P105_CAPTURE_CONTENT_SHA_DRIFT"):
        p105.observational_support(orig,expected_raw_sha256="0"*64,original_bytes=b"not exact")


def test_p105_no_source_read_replay_or_unknown_action_effect_api():
    with pytest.raises(PermissionError,match="P105_UNOBSERVED_COUNTERFACTUAL"):
        p105.refuse_unobserved_action_effect(
            state=[0.,0.,0.,0.],unobserved_action=1)
    source=Path(p105.__file__).read_text(encoding="utf-8")
    for forbidden in ("import causal_gym", "from causal_gym",
                      "import requests", "import subprocess", "import socket",
                      "asyncio.run(", "env.do(", "env.reset("):
        assert forbidden not in source
    assert not any(k.startswith("run_real") or k.startswith("collect_real")
                   for k in dir(p105))


def test_p105_historical_fit_summary_matches_readonly_custody_not_science():
    f=p105.locked_protocol()
    x=f["f3_fit_observation_only"]
    s={
        "worlds":x["exact_worlds"],
        "confirmed_effects":x["confirmed_transition_count"],
        "action_0":x["action_0"],"action_1":x["action_1"],
        "unique_exact_states":x["unique_exact_4D_state_before"],
        "repeated_exact_state_events":x["duplicate_exact_state_count"],
        "exact_state_with_both_actions":x["exact_state_with_both_actions_count"],
        "terminated":x["source_terminal_count"],"truncated":x["truncated_count"],
        "action_probability_field_logged":False,
        "direct_offpolicy_counterfactual_effect_qualified":False,
        "new_F6_source_authorized_or_generated":False
    }
    assert p105.validate_historical_summary(s)
    tampered=copy.deepcopy(s)
    tampered["exact_state_with_both_actions"]=1
    with pytest.raises(ValueError,match="P105_HISTORICAL_FIT_SUPPORT_MISMATCH"):
        p105.validate_historical_summary(tampered)
