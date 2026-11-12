"""F8-R10: fabricated paired action panels; never original F8 source."""
from __future__ import annotations
import copy
import inspect
import pytest

from experiments import p85mr_f8r10_paired_science as sci


def sample(*, candidate_cost=.7, baseline_cost=1., steps=80):
    records = []
    for arm, seed in sci.policy_instances():
        for world in range(sci.WORLD_COUNT):
            if arm in sci.CANDIDATES:
                cost = candidate_cost
            elif arm == "28_coefficient_linear7_ridge":
                cost = baseline_cost
            else:
                cost = baseline_cost + .2
            records.append({
                "classification": sci.EXAMPLE_ONLY,
                "session_id": sci.session_id(arm, seed, world),
                "policy_arm": arm,
                "model_seed": seed,
                "world_index": world,
                "world_seed": sci.WORLD_SEED_BASE + world,
                "mean_stability_cost": cost,
                "survival_steps": steps,
                "selected_actions": steps,
                "hypothetical_queries": 0 if seed == "fixed" else 2*steps,
            })
    return records


def mutate(rows, arm, seed, world, **changes):
    record = next(z for z in rows
                  if z["policy_arm"] == arm and z["model_seed"] == seed
                  and z["world_index"] == world)
    record.update(changes)


def test_exact_fixed_protocol_and_no_real_source_acceptance():
    f = sci.locked()
    assert f["stage"] == (
        "FROZEN_BEFORE_IMPLEMENTATION_OR_F8_SCIENTIFIC_SOURCE_FIRST_USE")
    assert f["permissions"]["owner_firstuse"] is False
    assert f["permissions"]["scientific_pass"] is False
    assert f["test_plan"]["paired_worlds"] == 16
    assert f["test_plan"]["models_per_world"] == 22
    assert f["test_plan"]["physical_trial_sessions"] == 352
    with pytest.raises(PermissionError, match="NOT_AUTHORIZED"):
        sci.refuse_real_firstuse()


def test_full_22_by_16_panel_counts_only_sixteen_independent_worlds():
    r = sci.audit_fabricated_panel(sample(steps=80))
    assert r["policy_sessions"] == 352
    assert r["independent_world_clusters"] == 16
    assert r["fictional_selected_effects"] == 352*80
    assert r["fictional_hypothetical_queries"] == 21*16*80*2
    assert r["actual_source_effects"] == 0
    assert r["owner_source_firstuse_authorized"] is False
    assert r["scientific_pass"] is False


def test_large_uniform_hypothetical_gain_passes_nominal_dev_gate_only():
    result = sci.evaluate_fabricated_panel(sample(candidate_cost=.75))
    assert result["strongest_baseline"]["arm"] == "28_coefficient_linear7_ridge"
    assert result["bootstrap"]["cluster_count"] == 16
    assert result["bootstrap"]["independent_policy_sessions"] is False
    assert all(x["hypothetical_2_of_3_gate_pass"]
               for x in result["candidate_results"].values())
    assert all(s["paired_bootstrap_2p5_difference"] > 0
               for arm in result["candidate_results"].values()
               for s in arm["seeds"])
    assert result["scientific_pass"] is False
    assert result["prospective_scientific_acceptance"].startswith("BLOCKED")


def test_null_gain_and_survival_degradation_cannot_pass():
    null = sci.evaluate_fabricated_panel(sample(candidate_cost=1.))
    assert not any(z["hypothetical_2_of_3_gate_pass"]
                   for z in null["candidate_results"].values())
    records = sample(candidate_cost=.7, steps=80)
    for z in records:
        if z["policy_arm"] in sci.CANDIDATES:
            z["selected_actions"] = 75
            z["survival_steps"] = 75
            z["hypothetical_queries"] = 150
    degraded = sci.evaluate_fabricated_panel(records)
    assert not any(z["hypothetical_2_of_3_gate_pass"]
                   for z in degraded["candidate_results"].values())


def test_impressive_mean_does_not_override_negative_paired_uncertainty():
    # 12 good worlds and 4 very bad worlds: 20% mean improvement,
    # but only 16 independent clusters and the lower bootstrap bound
    # crosses zero. This diagnostic is not an empirical F8 result.
    records = sample(candidate_cost=.4)
    for z in records:
        if z["policy_arm"] in sci.CANDIDATES and z["world_index"] >= 12:
            z["mean_stability_cost"] = 2.0
    report = sci.evaluate_fabricated_panel(records)
    for arm in report["candidate_results"].values():
        for seed in arm["seeds"]:
            assert seed["relative_gain_vs_strongest"] > .10
            assert seed["paired_bootstrap_2p5_difference"] <= 0
            assert not seed["hypothetical_seed_gate_pass"]
        assert not arm["hypothetical_2_of_3_gate_pass"]


def test_missing_duplicate_wrong_seed_or_cross_world_credit_denied():
    rows = sample()
    with pytest.raises(ValueError, match="INCOMPLETE"):
        sci.audit_fabricated_panel(rows[:-1])
    fake = copy.deepcopy(rows)
    fake[-1] = dict(fake[-2])
    with pytest.raises(ValueError, match="DUPLICATED"):
        sci.audit_fabricated_panel(fake)
    fake = copy.deepcopy(rows)
    fake[0]["world_seed"] += 10
    with pytest.raises(ValueError, match="ACCOUNTING"):
        sci.audit_fabricated_panel(fake)
    fake = copy.deepcopy(rows)
    fake[0]["session_id"] = "another-arm-session"
    with pytest.raises(ValueError, match="ACCOUNTING"):
        sci.audit_fabricated_panel(fake)


@pytest.mark.parametrize("changes", [
    {"classification":"REAL_OWNER_RECEIPT"},
    {"mean_stability_cost":float("nan")},
    {"mean_stability_cost":float("-inf")},
    {"mean_stability_cost":-1.},
    {"survival_steps":0},
    {"survival_steps":97},
    {"survival_steps":True},
    {"selected_actions":True},
    {"selected_actions":97},
    {"hypothetical_queries":999},
    {"hypothetical_queries":True},
    {"world_seed":True},
])
def test_poisoned_or_fake_external_owner_receipt_is_rejected(changes):
    rows = sample()
    rows[0].update(changes)
    with pytest.raises(ValueError):
        sci.audit_fabricated_panel(rows)


def test_strongest_conventional_zero_cost_blocks_spurious_relative_win():
    rows = sample(candidate_cost=0., baseline_cost=0.)
    with pytest.raises(ValueError, match="ZERO_COST"):
        sci.evaluate_fabricated_panel(rows)


def test_no_external_source_workflow_or_training_and_evidence_stays_repo():
    code = inspect.getsource(sci)
    for forbidden in ("import causal_gym", "from causal_gym",
                      "env.reset(", "env.do(", "workflow_dispatch(",
                      "subprocess.run(", "requests.get(", "urlopen(",
                      "torch.save(", "optimizer.step(", "create_runner("):
        assert forbidden not in code
    assert sci.locked()["qualification"]["scientific"] == (
        "NEVER SCIENTIFIC_PASS_FROM_THIS_MODULE")
