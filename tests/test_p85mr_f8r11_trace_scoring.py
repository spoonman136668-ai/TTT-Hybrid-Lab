"""F8-R11: synthetic physical trajectories verified before R10 metrics."""
from __future__ import annotations

import copy
import inspect
import math

import pytest

from experiments import p85mr_f8r11_trace_scoring as s
from experiments import p85mr_f8r10_paired_science as r10


def episode(arm="original_hard_depth_HybridPilot", seed=985201,
            world=0, *, n=16, angle=.04, angvel=0.):
    key = r10.session_id(arm, seed, world)
    state = [0.0, 0.0, angle, angvel]
    events = []
    for step in range(1, n+1):
        action = step % 2
        events.append({
            "event_id": step,
            "state_before": list(state),
            "intent_selected_action": action,
            "selected_action": action,
            "state_after": list(state),
            "reward": 1.,
            "terminated": step == n and n < 96,
            "truncated": step == 96,
            "hypothetical_action_queries": (
                [] if seed == "fixed" else [0, 1]),
        })
    return {
        "classification": s.CLASSIFICATION,
        "session_id": key,
        "policy_arm": arm,
        "model_seed": seed,
        "world_index": world,
        "world_seed": r10.WORLD_SEED_BASE + world,
        "events": events,
    }


def fabricated_panel(*, n=16):
    return [
        episode(arm, seed, world, n=n,
                angle=.04 if arm in r10.CANDIDATES else .50)
        for arm, seed in r10.policy_instances()
        for world in range(r10.WORLD_COUNT)
    ]


def test_original_metric_and_scope_frozen_before_implementation():
    freeze = s.locked()
    assert freeze["classification"] == "FABRICATED_DEV_ONLY"
    assert freeze["metric"]["horizon_steps"] == 96
    assert freeze["metric"]["missing_step_penalty"] == 2.
    assert freeze["metric"]["angle_index"] == 2
    assert freeze["authority"]["external_effects"] is False
    assert freeze["authority"]["owner_first_use"] is False
    with pytest.raises(PermissionError, match="NOT_AUTHORIZED"):
        s.refuse_original_source_receipt()


def test_three_physical_actions_score_and_93_penalized_missing_steps():
    trace = episode(n=3, angle=.2095, angvel=3.5)
    result = s.score_fabricated_trace(trace)
    expected = (3*(1.+.1)+93*2.)/96
    assert result["mean_stability_cost"] == pytest.approx(expected, abs=1e-12)
    assert result["selected_actions"] == 3
    assert result["survival_steps"] == 3
    assert result["hypothetical_queries"] == 6
    assert "events" not in result


def test_full_horizon_has_no_tail_penalty_and_dual_terminal_at_horizon_allowed():
    trace = episode(n=96, angle=.2095, angvel=3.5)
    trace["events"][-1]["terminated"] = True
    actual = s.score_fabricated_trace(trace)
    assert actual["mean_stability_cost"] == pytest.approx(1.1)
    assert actual["selected_actions"] == 96
    assert actual["hypothetical_queries"] == 192
    trace["events"][-1]["truncated"] = False
    with pytest.raises(ValueError, match="HORIZON_TRUNCATION"):
        s.score_fabricated_trace(trace)


def test_fixed_scripted_action_has_no_fake_counterfactual_query():
    trace = episode("frozen_handwritten_feedback_policy", "fixed", 0,
                    n=4, angle=.1)
    score = s.score_fabricated_trace(trace)
    assert score["hypothetical_queries"] == 0
    trace["events"][0]["hypothetical_action_queries"] = [0,1]
    with pytest.raises(ValueError, match="HYPOTHETICAL_QUERY"):
        s.score_fabricated_trace(trace)


def test_full_352_synthetic_traces_recomputed_then_original_r10_admitted():
    traces = fabricated_panel()
    rows, counts = s.compile_fabricated_panel(traces)
    assert len(rows) == 352
    assert counts["independent_paired_worlds"] == 16
    assert counts["fictional_selected_actions"] == 352*16
    assert counts["fictional_model_queries"] == 21*16*16*2
    assert counts["physically_executed_external_actions"] == 0
    assert counts["scientific_pass"] is False
    report = s.evaluate_fabricated_trajectories(traces)
    assert report["scientific_pass"] is False
    assert report["trace_recomputation"]["input_episodes"] == 352
    assert report["prospective_scientific_acceptance"].startswith("BLOCKED")
    # Only a hypothetical fixture passes the preregistered numeric threshold.
    assert all(x["hypothetical_2_of_3_gate_pass"]
               for x in report["candidate_results"].values())


def test_cannot_supply_self_reported_metric_or_signed_real_owner_record():
    trace = episode(n=3)
    trace["mean_stability_cost"] = 0.0
    with pytest.raises(ValueError, match="UNTRUSTED_TRACE"):
        s.score_fabricated_trace(trace)
    trace.pop("mean_stability_cost")
    trace["classification"] = "ORIGINAL_OWNER_FIRSTUSE_RECEIPT"
    with pytest.raises(ValueError, match="UNREGISTERED_POLICY_OR_WORLD"):
        s.score_fabricated_trace(trace)


@pytest.mark.parametrize("tamper, expected", [
    (lambda t: t["events"][0].update(selected_action=3), "ACTION_CREDIT"),
    (lambda t: t["events"][0].update(intent_selected_action=1-t["events"][0]["selected_action"]), "ACTION_CREDIT"),
    (lambda t: t["events"][0].update(intent_selected_action=True), "ACTION_CREDIT"),
    (lambda t: t["events"][0].update(hypothetical_action_queries=[0,0]), "HYPOTHETICAL_QUERY"),
    (lambda t: t["events"][0].update(hypothetical_action_queries=[0,1,0]), "HYPOTHETICAL_QUERY"),
    (lambda t: t["events"][0].update(event_id=2), "EVENT_GAP"),
    (lambda t: t["events"][1].update(state_before=[0,0,.5,0]), "BROKEN_PHYSICAL"),
    (lambda t: t["events"][0].update(state_after=[0,0,float("nan"),0]), "NONFINITE"),
    (lambda t: t["events"][0].update(state_after=[0,0,True,0]), "NONFINITE"),
    (lambda t: t["events"][0].update(reward=float("inf")), "NONFINITE"),
    (lambda t: t["events"][0].update(terminated=True), "AFTER_TERMINATION"),
    (lambda t: t["events"][0].update(truncated=True), "AFTER_TERMINATION"),
    (lambda t: t["events"][-1].update(terminated=False), "EARLY_STOP_FLAGS"),
    (lambda t: t["events"][-1].update(truncated=True), "EARLY_STOP_FLAGS"),
    (lambda t: t["events"][-1].update(state_after=[0,0]), "STATE_SHAPE"),
    (lambda t: t["events"][0].update(source_receipt="forged"), "EFFECT_FIELDS"),
])
def test_tampering_single_factual_effect_is_denied(tamper, expected):
    trace = episode(n=3)
    tamper(trace)
    with pytest.raises(ValueError, match=expected):
        s.score_fabricated_trace(trace)


def test_out_of_horizon_and_missing_worlds_and_duplicate_physical_traces():
    with pytest.raises(ValueError, match="UNBOUNDED_EFFECT"):
        s.score_fabricated_trace(episode(n=97))
    good = fabricated_panel(n=1)
    with pytest.raises(ValueError, match="INCOMPLETE_FABRICATED"):
        s.compile_fabricated_panel(good[:-1])
    dupe = list(good)
    dupe[-1] = copy.deepcopy(dupe[-2])
    with pytest.raises(ValueError, match="DUPLICATED"):
        s.compile_fabricated_panel(dupe)
    altered = list(good)
    altered[-1] = dict(altered[-1], world_seed=1824300500)
    with pytest.raises(ValueError, match="UNREGISTERED"):
        s.compile_fabricated_panel(altered)


def test_no_owner_source_or_new_scheduled_execution():
    text = inspect.getsource(s)
    for invalid in ("from causal_gym", "import causal_gym",
                    "env.reset(", "env.do(", "workflow_dispatch(",
                    "subprocess.run(", "requests.get(", "urlopen(",
                    "torch.save(", "optimizer.step("):
        assert invalid not in text
    assert s.locked()["authority"]["scheduler"] is False
