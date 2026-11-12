"""R2 microfixture-only: no simulator, research data, or coefficient updates."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.p78mr_r2_noise_shift_audit import (
    CHECKPOINTS, D, FOLLOW, FREEZE, FREEZE_BLOB, PREFIX, SCENARIOS, TOTAL,
    SEEDS, audit, frozen_model, simulate, toy_facts,
)
from experiments.p78mr_gain_posthoc_audit import read_locked


def test_precommitted_freeze_and_all_three_unchanged_coefficients():
    frozen = json.loads(read_locked(FREEZE, FREEZE_BLOB))
    assert frozen["stage"] == "FROZEN_BEFORE_TOY_DIAGNOSTIC_IMPLEMENTATION_OR_OBSERVATION"
    assert frozen["source_class"] == "HAND_AUTHORED_DETERMINISTIC_TOY_ONLY_NOT_INDEPENDENT_SCIENCE"
    result = audit()
    assert tuple(x["seed"] for x in result["seeds"]) == SEEDS
    assert not result["scientific_pass"]
    assert not result["source_birth"]
    assert not result["model_training"]
    assert not result["holdout_scoring"]
    assert result["decision"] in ("TOY_JOINT_PASS", "TOY_JOINT_FAIL")


def test_source_free_bounded_fixture_no_hidden_labels_and_no_future_query():
    assert TOTAL == 45 and PREFIX == 32 and FOLLOW == 12
    for case in SCENARIOS:
        facts = toy_facts(case)
        assert set(facts) == {"action", "effect"}
        assert facts["action"].shape == facts["effect"].shape == (1, 45, 16)
        assert torch.equal(facts["action"].norm(dim=-1), torch.ones((1, 45)))
        assert bool((facts["effect"][:, :32, 0] == 2).all())
    assert torch.equal(toy_facts("outlier")["effect"][:, :33],
                       toy_facts("reversal")["effect"][:, :33])
    assert not torch.equal(toy_facts("outlier")["effect"][:, 33:],
                           toy_facts("reversal")["effect"][:, 33:])


def test_toy_no_lookahead_and_exact_cold_restoration():
    row = audit()["seeds"][0]
    model = frozen_model(row["seed"], tuple(row["theta"]))
    facts = toy_facts("reversal")
    with torch.no_grad():
        full = model.encode(facts)
        first = {k: v[:, :32] for k, v in facts.items()}
        tail = {k: v[:, 32:] for k, v in facts.items()}
        prefix = model.encode(first)
        checkpoint = prefix.clone()
        cold = model.encode(tail, carry=checkpoint)
    assert torch.equal(full, cold)
    assert torch.equal(prefix, checkpoint)
    assert full.shape == (1, 16, 16)
    # Mutating all later effects must never change a prior prefix state.
    corrupted = {k: v.clone() for k, v in facts.items()}
    corrupted["effect"][:, 32:] *= -1
    with torch.no_grad():
        assert torch.equal(prefix, model.encode({k: v[:, :32] for k, v in corrupted.items()}))


def test_wrong_action_credit_disturbs_factual_memory():
    row = audit()["seeds"][0]
    model = frozen_model(row["seed"], tuple(row["theta"]))
    facts = toy_facts("reversal")
    swapped = {k: v.clone() for k, v in facts.items()}
    swapped["action"][:, 32, 0] = 0
    swapped["action"][:, 32, 1] = 1
    with torch.no_grad():
        assert not torch.equal(model.encode(facts), model.encode(swapped))


def test_original_fixed_rate_reference_and_full_metric_trace():
    result = audit()
    assert result["fixed_cal_selected_gain"] == 0.4
    assert len(result["seeds"]) == 3
    assert len(CHECKPOINTS) == 6
    for case in SCENARIOS:
        fixed = result["fixed_controls"][case]
        assert fixed["events"] == 45
        assert len(fixed["trace_prewrite_predictions"]) == TOTAL
        assert len(fixed["trace_prewrite_errors"]) == TOTAL
        assert set(fixed["after_disruption_and_followup_predictions"]) == {
            str(i) for i in CHECKPOINTS
        }
        for seed in result["seeds"]:
            record = seed["frozen_learned"][case]
            assert len(record["trace_prewrite_predictions"]) == TOTAL
            assert len(record["trace_prewrite_errors"]) == TOTAL


def test_frozen_joint_verdict_is_recomputed_from_metrics():
    result = audit()
    for row in result["seeds"]:
        a = row["frozen_learned"]["outlier"]
        b = row["frozen_learned"]["reversal"]
        f_a = result["fixed_controls"]["outlier"]
        f_b = result["fixed_controls"]["reversal"]
        noise = a["first_followup_mse"] <= f_a["first_followup_mse"]
        shift = (b["first_four_followup_prewrite_mse"] <=
                 f_b["first_four_followup_prewrite_mse"]
                 and b["end12_mse"] <= f_b["end12_mse"])
        assert row["h_noise"] is noise
        assert row["h_shift"] is shift
        assert row["joint_pass"] is (noise and shift)
    assert result["joint_all_seed_pass"] is all(x["joint_pass"] for x in result["seeds"])


def test_diagnostic_stores_complete_metrics_in_existing_g4_artifact():
    result = audit()
    target = Path(__file__).resolve().parents[1] / "local_runs" / "p78mr_r2_toy_result.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert json.loads(target.read_text())["decision"] == result["decision"]


@pytest.mark.parametrize("case", ("sealed_ood", "external", "nightmare"))
def test_only_immutable_fictional_scenarios_allowed(case):
    with pytest.raises(ValueError, match="P78MR_R2_UNREGISTERED_CASE"):
        toy_facts(case)


def test_no_external_science_or_training_entrypoint():
    text = (Path(__file__).resolve().parents[1] /
            "experiments" / "p78mr_r2_noise_shift_audit.py").read_text()
    for forbidden in ("make_batch(", "make_world(", "env.reset(", "env.do(",
                      "optimizer.step(", "requests.get(", "urlopen(", "torch.save("):
        assert forbidden not in text
