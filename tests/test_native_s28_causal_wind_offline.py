"""S28: source-locked contract and real factual-codec fixture negatives.

No gymnasium, upstream code, simulator, sealed world, optimizer or torch import.
"""
import hashlib
import inspect
import json
from pathlib import Path
import sys

import pytest

from scripts import native_s28_causal_wind_offline as s28


class SyntheticCartPoleFixture:
    """Intentionally NOT the third-party CartPoleWind world."""

    def __init__(self):
        self.current = (0., 0., .02, 0.)
        self.actions = []

    def reset(self):
        return self.current, {"public_note": "ignored"}

    def step(self, action):
        self.actions.append(action)
        self.current = (
            self.current[0] + .01, self.current[1] - .03,
            self.current[2] + .005, self.current[3] + .015
        )
        return self.current, 1.0, False, len(self.actions) >= 3, {
            "public_note": "never forwarded"
        }


def test_frozen_upstream_source_identity_authority_and_license_hold():
    p = s28.frozen()
    up = p["upstream"]
    assert up["repo"] == "CausalAILab/Causal-Gymnasium"
    assert up["commit"] == "c94e9a4cd3641588b6b8507e71ccee5c27032bdf"
    assert len(up["required_blobs"]) == 11
    assert up["recorded_license_marker"] == "@@ -154,49 +194,23 @@"
    assert p["qualification"]["science_first_use_authorized"] is False
    assert p["qualification"]["upstream_source_license_approved"] is False
    assert p["adapter"]["no_upstream_module_import"] is True
    assert p["adapter"]["can_execute_gym_environment"] is False
    assert s28.git_blob_sha(b"hello\n") == hashlib.sha1(b"blob 6\x00hello\n").hexdigest()


def test_real_codec_consumes_completed_fixture_actions_but_no_env_info():
    env = SyntheticCartPoleFixture()
    codec = s28.FactualWindTransitionLedger()
    initial, info = env.reset()
    assert codec.start(initial, info) == initial
    records = []
    for action in (0, 1, 0):
        transition = env.step(action)
        records.append(codec.add_completed_transition(action, transition))
    assert env.actions == [0,1,0]
    receipt = codec.receipt()
    assert receipt["events"] == 3
    assert records[1]["executed_action"] == 1
    assert records[1]["previous_observation"] == records[0]["next_observation"]
    assert records[-1]["truncated"] is True
    assert receipt["sha256"] == hashlib.sha256(
        s28.canonical_json(records)).hexdigest()
    raw = json.dumps(receipt)
    assert not any(s in raw for s in (
        "public_note", "current_wind", "counterfactual",
        "oracle_label", "natural_action", "policy_intuition"))
    assert receipt["science_first_use"] == "NOT_RUN"
    with pytest.raises(s28.S28Blocked, match="AFTER_TERMINATION"):
        codec.add_completed_transition(0, env.step(0))


def test_observation_action_reward_and_completion_fail_closed():
    codec = s28.FactualWindTransitionLedger()
    with pytest.raises(s28.S28Blocked, match="STEP_BEFORE_RESET"):
        codec.add_completed_transition(0, ((0,0,0,0), 1.,False,False,{}))
    with pytest.raises(s28.S28Blocked, match="OBSERVATION"):
        codec.start((0.,0.,0.), {})
    with pytest.raises(s28.S28Blocked, match="ENV_INFO_CONTAINS_PRIVILEGED_FIELD"):
        codec.start((0.,0.,0.,0.), {"u_wind": .1})
    codec.start((0.,0.,0.,0.), {})
    good = ((0.,0.,.01,0.), 1.,False,False,{})
    for bad in (True,1.,-1,2, "0"):
        with pytest.raises(s28.S28Blocked, match="ACTION"):
            codec.add_completed_transition(bad,good)
    for bad in (
        ((0.,0.,float("nan"),0.),1.,False,False,{}),
        ((0.,0.,.01,0.),float("inf"),False,False,{}),
        ((0.,0.,.01,0.),True,False,False,{}),
        ((0.,0.,.01,0.),1.,0,False,{}),
        ((0.,0.,.01,0.),1.,False,False,{"hidden_state": [1]}),
        ((0.,0.,.01,0.),1.,False,False,{"counterfactual_result": 7}),
        ((0.,0.,.01,0.),1.,False,False,{"causal_graph": {}}),
        ((0.,0.,.01,0.),1.,False,False,[]),
    ):
        with pytest.raises(s28.S28Blocked):
            codec.add_completed_transition(0,bad)
    assert codec.receipt()["events"] == 0
    codec.add_completed_transition(0,good)
    assert codec.receipt()["events"] == 1
    with pytest.raises(s28.S28Blocked, match="EPISODE_ALREADY_STARTED"):
        codec.start((0.,0.,0.,0.), {})


def test_readonly_source_path_and_no_checkout_failure(tmp_path):
    with pytest.raises(s28.S28Blocked, match="UPSTREAM_CHECKOUT_MISSING"):
        s28.audit_exact_upstream_checkout(tmp_path/"absent")
    with pytest.raises(s28.S28Blocked, match="OFFLINE_GIT_CHECK_FAILURE"):
        s28.audit_exact_upstream_checkout(tmp_path)
    dep = s28.dependency_metadata_only()
    assert dep["runtime_ready"] is False
    assert dep["package_install_attempted"] is False
    assert dep["runtime_import_or_instantiation_attempted"] is False
    assert "gymnasium" in dep["installed_distributions"]


def test_module_cannot_import_or_instantiate_external_world():
    source = Path(s28.__file__).read_text("utf-8")
    assert "import gymnasium" not in source
    assert "from causal_gym" not in source
    assert "pip install" not in source
    assert list(inspect.signature(
        s28.FactualWindTransitionLedger.add_completed_transition
    ).parameters) == ["self", "action", "transition"]
    assert "causal_gym" not in sys.modules
    assert "gymnasium" not in sys.modules
