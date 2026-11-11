"""P81-MR contract-only toys. No real Causal-Gymnasium data or actions."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "experiments"))
from p81mr_collection_contract import (
    ToyCaptureRecorder, planned_action, frozen_episode_seed,
    MAX_STEPS, SPLIT_SEEDS, NAMESPACE,
)
from p80mr_causalgym_trace_adapter import adapt_capture
from p79mr_toy_transfer_replay import toy_replay


def new_recorder(*, initial=None, **overrides):
    opts = {
        "source_kind": "toy_engineering", "split": "fit",
        "episode_index": 3, "initial_observation": [0.,0.,0.,0.],
        "source_id": "toy:fixture-p81", "producer_id": "toy:in-memory",
        "custody_ref": "toy:unwitnessed", "world_id": "toy-world-3"
    }
    if initial is not None:
        opts["initial_observation"] = initial
    opts.update(overrides)
    return ToyCaptureRecorder(**opts)


def finish_two(obs1=None, obs2=None, *, extras=None):
    r = new_recorder()
    x = r.prepare_action()
    if obs1 is None:
        obs1 = [0.1,0.2,0.01,0.02]
    if obs2 is None:
        obs2 = [0.2,0.3,0.012,0.025]
    info = {"action":x["intended_action"], "hidden_wind":123.0}
    if extras:
        info.update(extras)
    r.record_observed_step(info=info,next_observation=obs1,reward=1.,
                           terminated=False,truncated=False,receipt_ref="toy:step-1")
    y = r.prepare_action()
    r.record_observed_step(info={"action":y["intended_action"],"secret_oracle":True},
                           next_observation=obs2,reward=1.,terminated=True,
                           truncated=False,receipt_ref="toy:step-2")
    return r, (x,y)


def test_frozen_seed_namespace_and_unrelated_global_rng():
    assert NAMESPACE == "ttt-hybrid/p81mr/collection-prospective-v1"
    assert frozen_episode_seed("fit",3)==811131
    assert frozen_episode_seed("cal",3)==811133
    a = [planned_action("fit",3,i) for i in range(1,MAX_STEPS+1)]
    b = [planned_action("fit",3,i) for i in range(1,MAX_STEPS+1)]
    assert a == b and set(a) == {0,1}
    assert all(type(x) is int for x in a)
    assert len(set(SPLIT_SEEDS.values())) == len(SPLIT_SEEDS)


@pytest.mark.parametrize("kind",("external_simulator","physical","unknown",None))
def test_external_first_use_rejected_before_touching_source(kind):
    poison=object()
    with pytest.raises(PermissionError,match="P81MR_EXTERNAL_FIRST_USE_NOT_AUTHORIZED"):
        new_recorder(source_kind=kind,initial=poison)


@pytest.mark.parametrize("split",("dev_external","science_ood_0","science_ood_1","science_ood_2"))
def test_no_heldout_source_born_even_on_toy_fixture(split):
    with pytest.raises(PermissionError,match="P81MR_SPLIT_FIRST_USE_REQUIRED"):
        new_recorder(split=split)
    with pytest.raises(ValueError,match="P81MR_SPLIT_FIRST_USE_REQUIRED"):
        planned_action(split,0,1)


def test_complete_fictional_capture_p80_admission_no_external_inference():
    recorder, planned = finish_two()
    result = recorder.finalize()
    assert result["classification"] == "TOY_ENGINEERING_ONLY"
    assert result["external_environment_executed"] is False
    assert result["science_pass"] is False
    cap = result["capture"]
    assert cap["worlds"][0]["episode_seed"] == frozen_episode_seed("fit",3)
    assert [x["event_id"] for x in cap["worlds"][0]["transitions"]] == [1,2]
    assert [x["executed_action"] for x in cap["worlds"][0]["transitions"]] == [
        planned[0]["intended_action"], planned[1]["intended_action"]]
    assert cap["worlds"][0]["probe_event_ids"] == [2]
    for row in cap["worlds"][0]["transitions"]:
        assert set(row) == {"event_id", "state_before", "executed_action",
                            "state_after", "reward", "terminated", "truncated",
                            "mode", "receipt_ref"}
        assert row["mode"] == "do"
        assert "hidden_wind" not in row
        assert "secret_oracle" not in row
    adapted = adapt_capture(cap)
    assert adapted["custody"]["scientific_pass"] is False
    assert result["schema_only_sha256"] == adapted["custody"]["capture_sha256"]
    with pytest.raises(ValueError,match="P79MR_OWNER_FIRST_USE_REQUIRED"):
        toy_replay(adapted["manifest"],adapted["worlds"])


def test_read_before_write_event_index_and_no_future_output_copy():
    recorder,_=finish_two()
    cap=recorder.finalize()["capture"]
    steps=cap["worlds"][0]["transitions"]
    assert steps[0]["state_after"] == steps[1]["state_before"]
    after=adapt_capture(cap)["worlds"][0]
    probe=after["probes"][0]
    assert probe["after_events"]==1
    assert probe["action"] == after["events"][1]["action"]
    assert probe["realized_effect"] == after["events"][1]["effect"]
    assert probe["realized_effect"] != after["events"][0]["effect"] or steps[1]["reward"] == 1.


def test_trajectory_determinism_not_conditioned_on_future_effects():
    x=new_recorder()
    y=new_recorder()
    x.prepare_action()
    y.prepare_action()
    for i,(rec,post) in enumerate(((x,[0.1,0.2,0.01,0.02]),
                                   (y,[-0.1,0.2,0.01,0.02]))):
        rec.record_observed_step(info={"action":planned_action("fit",3,1)},
                                next_observation=post,reward=float(i),
                                terminated=False,truncated=False,
                                receipt_ref=f"toy:step-{i}")
    assert x.prepare_action() == y.prepare_action()
    assert x.event_count == y.event_count == 1


def test_no_double_prepare_or_uncertain_action_replay_after_mismatch():
    r=new_recorder()
    wanted=r.prepare_action()["intended_action"]
    with pytest.raises(RuntimeError,match="P81MR_EPISODE_NOT_PREPARABLE"):
        r.prepare_action()
    with pytest.raises(ValueError,match="P81MR_APPLIED_ACTION_MISMATCH"):
        r.record_observed_step(info={"action":1-wanted},
                               next_observation=[0.1,0.,0.,0.],
                               reward=0.,terminated=False,truncated=False,
                               receipt_ref="toy:step")
    assert r.event_count == 0
    with pytest.raises(RuntimeError,match="P81MR_EPISODE_NOT_PREPARABLE"):
        r.prepare_action()
    with pytest.raises(RuntimeError,match="P81MR_UNFINALIZABLE"):
        r.finalize()


@pytest.mark.parametrize("bad_info",({},{"action":True},{"action":2},{"action":"0"}))
def test_missing_or_malformed_applied_action_is_abort(bad_info):
    r=new_recorder()
    r.prepare_action()
    with pytest.raises(ValueError,match="P81MR_APPLIED_ACTION"):
        r.record_observed_step(info=bad_info,next_observation=[0.1,0.,0.,0.],
                               reward=1.,terminated=False,truncated=False,
                               receipt_ref="toy:step")
    with pytest.raises(RuntimeError,match="P81MR_NO_PENDING_EXECUTED_ACTION"):
        r.record_observed_step(info={"action":0},
                               next_observation=[0.2,0.,0.,0.],
                               reward=1.,terminated=False,truncated=False,
                               receipt_ref="toy:replay")


def test_no_synthetic_padding_or_action_after_early_terminal():
    r=new_recorder()
    intent=r.prepare_action()["intended_action"]
    r.record_observed_step(info={"action":intent},
                           next_observation=[0.1,0.,0.,0.],
                           reward=0.,terminated=True,truncated=False,
                           receipt_ref="toy:early-end")
    with pytest.raises(RuntimeError,match="P81MR_EPISODE_NOT_PREPARABLE"):
        r.prepare_action()
    with pytest.raises(RuntimeError,match="P81MR_TOO_SHORT_NO_PADDING"):
        r.finalize()


def test_no_capture_completion_without_source_terminal_or_budget():
    r=new_recorder()
    for i in range(2):
        a=r.prepare_action()["intended_action"]
        r.record_observed_step(info={"action":a},next_observation=[0.1*i,0.,0.,0.],
                               reward=1.,terminated=False,truncated=False,
                               receipt_ref=f"toy:event-{i}")
    with pytest.raises(RuntimeError,match="P81MR_INCOMPLETE_EPISODE"):
        r.finalize()


@pytest.mark.parametrize("case,pattern",[
    ("bad_state","P81MR_BAD_OBSERVATION"),
    ("bad_reward","P81MR_BAD_REALIZED_REWARD"),
    ("bad_flag","P81MR_BAD_TERMINAL_FLAG"),
    ("bad_receipt","P81MR_BAD_CUSTODY_ID"),
])
def test_bad_observation_after_pending_is_abort(case,pattern):
    r=new_recorder()
    a=r.prepare_action()["intended_action"]
    kwargs={"info":{"action":a}, "next_observation":[0.1,0.,0.,0.],
            "reward":1.0,"terminated":False,"truncated":False,"receipt_ref":"toy:okay"}
    if case=="bad_state": kwargs["next_observation"]=[float("nan"),0,0,0]
    if case=="bad_reward": kwargs["reward"]=float("inf")
    if case=="bad_flag": kwargs["terminated"]=1
    if case=="bad_receipt": kwargs["receipt_ref"]=""
    with pytest.raises(ValueError,match=pattern):
        r.record_observed_step(**kwargs)
    with pytest.raises(RuntimeError,match="P81MR_EPISODE_NOT_PREPARABLE"):
        r.prepare_action()


def test_finished_episode_cannot_be_consumed_twice():
    r,_=finish_two()
    good=r.finalize()
    assert len(good["capture"]["worlds"][0]["transitions"])==2
    with pytest.raises(RuntimeError,match="P81MR_UNFINALIZABLE"):
        r.finalize()
    with pytest.raises(RuntimeError,match="P81MR_EPISODE_NOT_PREPARABLE"):
        r.prepare_action()


def test_capture_has_no_active_source_access_or_science_entrypoint():
    source=(Path(__file__).resolve().parents[1]/"experiments"/"p81mr_collection_contract.py").read_text()
    for forbidden in ("import causal_gym", "import gymnasium",
                      "env.reset(", "env.do(", "env.step(", "subprocess", "requests.",
                      "torch", "if __name__", "run_science(", "torch.save"):
        assert forbidden not in source, forbidden
