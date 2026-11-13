"""F8-R12: falsify unfounded self-reported model-control action credit.

All traces use fabricated states; no original owner/source first-use permitted.
"""
from __future__ import annotations

import copy
import inspect

import pytest

from experiments import p85mr_f8r12_preaction_decision as decision
from experiments import p85mr_f8r11_trace_scoring as r11
from experiments import p85mr_f8r10_paired_science as r10


def trace(*, arm="original_hard_depth_HybridPilot", seed=985201,
          world=0, n=3, angle=0.01):
    scripted = seed == "fixed"
    before = [0.0, 0.0, angle, 0.0]
    episodes = []
    for step in range(1, n + 1):
        predictions = [] if scripted else [
            {"hypothetical_action": 0,
             "query_state_before": list(before),
             "predicted_state_after": [0.0, 0.0, 0.01, 0.0]},
            {"hypothetical_action": 1,
             "query_state_before": list(before),
             "predicted_state_after": [0.0, 0.0, 0.5, 0.0]},
        ]
        episodes.append({
            "event_id": step, "state_before": list(before),
            "intent_selected_action": 0, "selected_action": 0,
            "state_after": list(before), "reward": 1.0,
            "terminated": step == n and n < 96,
            "truncated": step == 96,
            "hypothetical_action_queries": [] if scripted else [0, 1],
            "pre_action_predictions": predictions,
        })
    return {
        "classification": decision.DEV,
        "session_id": r10.session_id(arm, seed, world),
        "policy_arm": arm, "model_seed": seed,
        "world_index": world, "world_seed": r10.WORLD_SEED_BASE + world,
        "events": episodes,
    }


def panel(*, n=12, candidate_angle=0.0, baseline_angle=0.5):
    return [
        trace(arm=arm, seed=seed, world=world, n=n,
              angle=candidate_angle if arm in r10.CANDIDATES
              else baseline_angle)
        for arm, seed in r10.policy_instances()
        for world in range(r10.WORLD_COUNT)
    ]


def test_locked_precommitted_contract_and_source_denial():
    f = decision.locked()
    assert f["stage"] == "FROZEN_BEFORE_IMPLEMENTATION_OR_F8_NEW_EFFECTS"
    assert f["authority"]["owner_firstuse"] is False
    assert f["authority"]["science_pass"] is False
    assert f["decision"]["action_order"] == [0, 1]
    with pytest.raises(PermissionError, match="NOT_AUTHORIZED"):
        decision.refuse_original_model_or_source()


def test_model_argmin_checked_and_factual_score_is_unchanged():
    t = trace(n=3)
    clean, report = decision.check_and_strip_fabricated_trace(t)
    assert len(clean["events"]) == 3
    assert all("pre_action_predictions" not in e for e in clean["events"])
    assert report["decision_steps_checked"] == 3
    assert not report["owner_verified_inference"]
    assert report["real_source_effects"] == 0
    assert report["scientific_pass"] is False
    score = r11.score_fabricated_trace(clean)
    assert score["survival_steps"] == 3
    assert score["mean_stability_cost"] > 0


def test_original_r11_could_not_detect_unearned_action_but_r12_denies_it():
    t = trace()
    t["events"][0]["selected_action"] = 1
    t["events"][0]["intent_selected_action"] = 1
    # R11 sees matching chosen-action intent and selected physical action
    # but cannot inspect imaginary model outputs it did not receive.
    old_trace = copy.deepcopy(t)
    for event in old_trace["events"]:
        del event["pre_action_predictions"]
    r11.score_fabricated_trace(old_trace)
    with pytest.raises(ValueError, match="NOT_MODEL_ARGMIN"):
        decision.check_and_strip_fabricated_trace(t)


def test_tie_exactly_chooses_action_zero_and_second_predicted_action_can_win():
    t = trace()
    t["events"][0]["pre_action_predictions"][1]["predicted_state_after"] = [
        0.0, 0.0, 0.01, 0.0]
    decision.check_and_strip_fabricated_trace(t)
    t["events"][0]["selected_action"] = 1
    t["events"][0]["intent_selected_action"] = 1
    with pytest.raises(ValueError, match="NOT_MODEL_ARGMIN"):
        decision.check_and_strip_fabricated_trace(t)
    t = trace()
    t["events"][0]["pre_action_predictions"][0]["predicted_state_after"] = [
        0.0, 0.0, 0.5, 0.0]
    t["events"][0]["pre_action_predictions"][1]["predicted_state_after"] = [
        0.0, 0.0, 0.01, 0.0]
    t["events"][0]["selected_action"] = 1
    t["events"][0]["intent_selected_action"] = 1
    decision.check_and_strip_fabricated_trace(t)


def test_scripted_control_cannot_falsely_claim_model_queries():
    t = trace(arm="frozen_handwritten_feedback_policy", seed="fixed")
    clean, report = decision.check_and_strip_fabricated_trace(t)
    assert report["scripted_control"]
    assert report["decision_steps_checked"] == 0
    assert r11.score_fabricated_trace(clean)["hypothetical_queries"] == 0
    t["events"][0]["pre_action_predictions"] = [
        {"hypothetical_action": 0, "query_state_before": [0.,0.,0.,0.],
         "predicted_state_after": [0.,0.,0.,0.]},
    ]
    with pytest.raises(ValueError, match="SCRIPTED_POLICY"):
        decision.check_and_strip_fabricated_trace(t)


@pytest.mark.parametrize("change,error", [
    (lambda t: t["events"][0].update(pre_action_predictions=[]), "MISSING_OR_EXTRA"),
    (lambda t: t["events"][0]["pre_action_predictions"].append(
        {"hypothetical_action":0,"query_state_before":[0.,0.,0.,0.],
         "predicted_state_after":[0.,0.,0.,0.]}), "MISSING_OR_EXTRA"),
    (lambda t: t["events"][0]["pre_action_predictions"].reverse(), "WRONG_HYPOTHETICAL"),
    (lambda t: t["events"][0]["pre_action_predictions"][0].update(
        hypothetical_action=True), "WRONG_HYPOTHETICAL"),
    (lambda t: t["events"][0]["pre_action_predictions"][0].update(
        query_state_before=[0.,0.,999.,0.]), "WRONG_HYPOTHETICAL"),
    (lambda t: t["events"][0]["pre_action_predictions"][0].update(
        query_state_before=t["events"][0]["state_after"][:3]), "INVALID_PREDICTION"),
    (lambda t: t["events"][0]["pre_action_predictions"][0].update(
        predicted_state_after=[0.,0.,float("nan"),0.]), "INVALID_PREDICTION"),
    (lambda t: t["events"][0]["pre_action_predictions"][0].update(
        predicted_state_after=[0.,0.,True,0.]), "INVALID_PREDICTION"),
    (lambda t: t["events"][0]["pre_action_predictions"][0].update(
        source_effect_receipt="fake"), "WRONG_HYPOTHETICAL"),
    (lambda t: t["events"][0].update(pre_action_predictions="made-up"), "INVALID_PREACTION"),
    (lambda t: t.update(classification="AUTHENTIC_OWNER_RECEIPT"), "REAL_OWNER_RECEIPT"),
    (lambda t: t["events"][0].update(hypothetical_action_queries=[0,0]), "HYPOTHETICAL_QUERY"),
    (lambda t: t["events"][0].update(intent_selected_action=True), "NOT_MODEL_ARGMIN"),
])
def test_false_model_and_action_provenance_denied(change, error):
    t = trace()
    change(t)
    with pytest.raises(ValueError, match=error):
        decision.check_and_strip_fabricated_trace(t)


def test_all_352_paired_world_fictional_decisions_flow_through_r11_and_r10():
    result = decision.evaluate_fabricated_decision_panel(panel())
    d = result["preaction_decision_verification"]
    assert d["decision_steps_checked"] == 21*16*12
    assert d["scripted_sessions"] == 16
    assert d["independent_world_clusters"] == 16
    assert not d["signed_original_owner_inference"]
    assert not d["real_source_firstuse"]
    assert result["trace_recomputation"]["input_episodes"] == 352
    assert result["trace_recomputation"]["physically_executed_external_actions"] == 0
    assert result["scientific_pass"] is False
    assert all(z["hypothetical_2_of_3_gate_pass"]
               for z in result["candidate_results"].values())


def test_one_unearned_action_breaks_whole_hypothetical_352_panel():
    traces = panel()
    wrong = traces[1]["events"][0]
    wrong["selected_action"] = 1
    wrong["intent_selected_action"] = 1
    with pytest.raises(ValueError, match="NOT_MODEL_ARGMIN"):
        decision.evaluate_fabricated_decision_panel(traces)


def test_no_run_source_import_or_external_model_and_no_science_claim():
    code = inspect.getsource(decision)
    for bad in ("import causal_gym", "from causal_gym", "env.reset(",
                "env.do(", "workflow_dispatch(", "subprocess.run(",
                "requests.get(", "urlopen(", "torch.save(",
                "optimizer.step(", "create_runner("):
        assert bad not in code
    assert decision.locked()["authority"]["new_source_effects"] is False
