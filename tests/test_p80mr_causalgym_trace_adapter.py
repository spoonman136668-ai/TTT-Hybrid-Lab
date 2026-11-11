"""P80-MR external CartPoleWind integration tests: fictional transitions ONLY.

These tests neither import Causal-Gymnasium, install its dependencies, call a
simulator, use an external trace, nor perform an external scientific first-use.
"""
import copy
import math
import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from p79mr_trace_admission import validate_trace, canonical_sha256
from p79mr_toy_transfer_replay import toy_replay
from p80mr_causalgym_trace_adapter import (
    adapt_capture, CAPTURE_SCHEMA, UPSTREAM_REPO, UPSTREAM_SHA,
    UPSTREAM_PATH, UPSTREAM_BLOB,
)


def raw_capture():
    states = [
        [0.0, 0.0, 0.0, 0.0],
        [0.1, 0.2, 0.001, 0.01],
        [0.11, 0.3, 0.003, 0.02],
        [0.12, 0.31, 0.005, 0.025],
    ]
    steps = [
        {"event_id": i+1, "state_before": states[i],
         "executed_action": (i%2), "state_after": states[i+1],
         "reward": 1.0, "terminated": (i==2), "truncated": False,
         "mode": "do", "receipt_ref": f"witness:event-{i+1}"}
        for i in range(3)
    ]
    return {
        "schema": CAPTURE_SCHEMA,
        "upstream_repo": UPSTREAM_REPO,
        "upstream_commit": UPSTREAM_SHA,
        "upstream_path": UPSTREAM_PATH,
        "upstream_blob": UPSTREAM_BLOB,
        "source_id": "fictional-local-test",
        "producer_id": "unit-test",
        "custody_ref": "toy:fictional",
        "worlds": [{"world_id": "fictional-episode", "episode_seed": 7001,
                    "transitions": steps, "probe_event_ids": [2,3]}],
    }


def test_output_validates_but_independence_is_unverified():
    x = adapt_capture(raw_capture())
    result = validate_trace(x["manifest"], x["worlds"])
    assert result["admission"] == "SCHEMA_PASS_SCIENCE_UNVERIFIED"
    assert result["worlds"] == 1 and result["probes"] == 2
    assert x["manifest"]["source_kind"] == "external_simulator"
    assert x["custody"]["scientific_pass"] is False
    assert x["custody"]["independent_origin"] == "UNVERIFIED"
    assert x["custody"]["upstream_commit"] == UPSTREAM_SHA
    assert x["custody"]["adapted_worlds_sha256"] == canonical_sha256(x["worlds"])


def test_no_external_inference_allowed_through_existing_toy_replay():
    x = adapt_capture(raw_capture())
    with pytest.raises(ValueError, match="P79MR_OWNER_FIRST_USE_REQUIRED"):
        toy_replay(x["manifest"], x["worlds"])


def test_fixed_state_action_embedding_is_unit_and_16d():
    x = adapt_capture(raw_capture())
    for item in x["worlds"][0]["events"]:
        action, effect = item["action"], item["effect"]
        assert len(action) == len(effect) == 16
        assert abs(sum(v*v for v in action)-1.0) < 1e-7
        assert action[0] > 0
        assert sum(action[5:7]) > 0
        assert effect[7:] == [0.0]*9
        assert all(math.isfinite(v) for v in action+effect)


def test_known_effects_and_actual_read_before_write_probes():
    capture = raw_capture()
    x = adapt_capture(capture)
    w = x["worlds"][0]
    assert [p["after_events"] for p in w["probes"]] == [1,2]
    for p in w["probes"]:
        i = p["after_events"]
        assert p["action"] == w["events"][i]["action"]
        assert p["realized_effect"] == w["events"][i]["effect"]
        assert p["receipt_ref"] == w["events"][i]["receipt_ref"]
        assert p["group"] == "unknown"
    # Delta x=0.01, reward=1, terminating in final event.
    effect = w["events"][-1]["effect"]
    assert effect[0] == pytest.approx(math.tanh(0.01/2.4), rel=1e-6)
    assert effect[4] == pytest.approx(math.tanh(1.0))
    assert effect[5] == 1.0 and effect[6] == 0.0


def test_same_capture_produces_identical_hash_and_no_science_claim():
    a = adapt_capture(raw_capture())
    b = adapt_capture(raw_capture())
    assert a == b
    assert a["custody"]["capture_sha256"] == canonical_sha256(raw_capture())
    assert a["custody"]["project_owner_first_use"] == "NOT_AUTHORIZED"


@pytest.mark.parametrize("edit,code", [
    ("upstream_sha","P80MR_UNPINNED_UPSTREAM"),
    ("mode_see","P80MR_INTERVENTION_NOT_EXECUTED"),
    ("mode_counterfactual","P80MR_INTERVENTION_NOT_EXECUTED"),
    ("hidden_wind","P80MR_BAD_TRANSITION"),
    ("episode_extra","P80MR_BAD_WORLD"),
    ("action_2","P80MR_BAD_EXECUTED_ACTION"),
    ("action_bool","P80MR_BAD_EXECUTED_ACTION"),
    ("step_skip","P80MR_EVENT_SEQUENCE"),
    ("not_contiguous","P80MR_BROKEN_TRAJECTORY"),
    ("nonfinite","P80MR_BAD_STATE"),
    ("nonunit_reward","P80MR_BAD_REWARD"),
    ("early_done","P80MR_EVENTS_AFTER_TERMINATION"),
    ("probe_first","P80MR_BAD_PROBE_IDS"),
    ("duplicate_probe","P80MR_BAD_PROBE_IDS"),
    ("event_receipt_empty","P80MR_BAD_RECEIPT_OR_ID"),
])
def test_malformed_or_unauthorized_capture_fails_closed(edit,code):
    capture = raw_capture()
    ep = capture["worlds"][0]
    steps = ep["transitions"]
    if edit == "upstream_sha":
        capture["upstream_commit"] = "0"*40
    elif edit == "mode_see":
        steps[0]["mode"] = "see"
    elif edit == "mode_counterfactual":
        steps[0]["mode"] = "ctf_do"
    elif edit == "hidden_wind":
        steps[0]["hidden_wind"] = 0.03
    elif edit == "episode_extra":
        ep["oracle_changepoint"] = 2
    elif edit == "action_2":
        steps[0]["executed_action"] = 2
    elif edit == "action_bool":
        steps[0]["executed_action"] = True
    elif edit == "step_skip":
        steps[1]["event_id"] = 7
    elif edit == "not_contiguous":
        # Make a distinct, inconsistent next-observation copy. In the toy
        # setup state_after[i] aliases state_before[i+1] unless rebound.
        steps[1]["state_before"] = [0.2] + steps[1]["state_before"][1:]
    elif edit == "nonfinite":
        steps[0]["state_after"][0] = float("nan")
    elif edit == "nonunit_reward":
        steps[0]["reward"] = float("nan")
    elif edit == "early_done":
        steps[0]["terminated"] = True
    elif edit == "probe_first":
        ep["probe_event_ids"] = [1]
    elif edit == "duplicate_probe":
        ep["probe_event_ids"] = [2,2]
    elif edit == "event_receipt_empty":
        steps[1]["receipt_ref"] = ""
    with pytest.raises(ValueError, match=code):
        adapt_capture(capture)


def test_earlier_probes_do_not_change_when_future_transition_changes():
    x = raw_capture()
    first = adapt_capture(x)["worlds"][0]["probes"][0]
    x["worlds"][0]["transitions"][2]["reward"] = -1.0
    second = adapt_capture(x)["worlds"][0]["probes"][0]
    assert first["action"] == second["action"]
    assert first["realized_effect"] == second["realized_effect"]


def test_terminated_trace_cannot_have_following_events():
    x = raw_capture()
    x["worlds"][0]["transitions"][1]["truncated"] = True
    with pytest.raises(ValueError,match="P80MR_EVENTS_AFTER_TERMINATION"):
        adapt_capture(x)


def test_independent_world_ids_and_source_id_checked_by_trace_schema():
    x = raw_capture()
    x["worlds"].append(copy.deepcopy(x["worlds"][0]))
    with pytest.raises(ValueError,match="P79MR_DUPLICATE_WORLD"):
        adapt_capture(x)
    y = raw_capture()
    y["source_id"] = "invalid source ID"
    with pytest.raises(ValueError,match="P79MR_BAD_MANIFEST"):
        adapt_capture(y)


def test_no_upstream_execution_or_hidden_scoring_entrypoint():
    text = (EXPERIMENTS/"p80mr_causalgym_trace_adapter.py").read_text()
    assert "import causal_gym" not in text
    assert "gymnasium" not in text
    assert "if __name__" not in text
    assert "torch" not in text
    assert "run_experiment(" not in text
    assert ".do(" not in text
    assert "score(" not in text


def test_state_delta_overflow_is_explicit_failure_not_silent_tanh_clamp():
    capture = raw_capture()
    capture["worlds"][0]["transitions"][0]["state_before"][0] = 1e308
    capture["worlds"][0]["transitions"][0]["state_after"][0] = -1e308
    with pytest.raises(ValueError, match="P80MR_BAD_EFFECT"):
        adapt_capture(capture)
