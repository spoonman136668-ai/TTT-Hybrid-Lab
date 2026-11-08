"""C9: prequential missing-feedback stress, only prior factual inputs to router."""
from copy import deepcopy
import hashlib
import inspect

import pytest
import torch

from scripts import native_c9_observation_robustness as c9
from scripts import native_p3_continuity as factual
from scripts import native_p7_learned_router_dev as p7
from scripts import native_c8_trained_archive_selection as c8


def _test_fixture():
    torch.set_num_threads(1)
    cfg = c9._frozen()
    model, _ = factual.new_model_and_optimizer(factual.MODEL_INITIALIZATIONS[0])
    init = model.initial_state(torch.zeros((1, 1)))
    new = {"recurrent": init.recurrent.detach().clone(),
           "fast_weights": init.fast_weights.detach().clone(),
           "history": init.history.detach().clone()}
    old = deepcopy(new)
    old["fast_weights"].add_(.01)
    learned = p7.LearnedRouter().eval()
    conventional = p7.ConventionalMLP().eval()
    return cfg, model, old, new, learned, conventional


def test_three_frozen_stress_conditions_have_deterministic_distinct_masks():
    cfg = c9._frozen()
    assert cfg["parent_c8_sha"] == "dc9288552d91c1f56051b0d348ca8ca8e98e2bb4"
    assert cfg["p7_original_source_blob"] == c8.P7_ORIGINAL_BLOB
    assert cfg["qualification"]["hidden_scientific_first_use"] is False
    assert cfg["training"]["p7_real_adamw_updates"] == p7.UPDATES == 600
    traces = {}
    for condition in cfg["stress"]["conditions"]:
        a = c9.make_trace(71931, 46111, condition, cfg)
        b = c9.make_trace(71931, 46111, condition, cfg)
        assert a == b and len(a) == 96
        assert all(row["action"] in (-1, 1) for row in a)
        assert all(row["latent_regime_evaluator_only"] in ("old", "recent")
                   for row in a)
        traces[condition["id"]] = a
    assert all(x["reveal_to_all_after_prediction"] for x in traces["balanced"])
    assert all(x["reveal_to_all_after_prediction"]
               for x in traces["action_imbalanced"])
    assert any(not x["reveal_to_all_after_prediction"]
               for x in traces["feedback_sparse"])
    assert any(x["action"] < 0 for x in traces["action_imbalanced"])


def test_unrevealed_factual_outcome_cannot_change_future_routes():
    cfg, model, old, recent, trained, conv = _test_fixture()
    condition = next(x for x in cfg["stress"]["conditions"]
                     if x["id"] == "feedback_sparse")
    plan = c9.make_trace(71947, 46111, condition, cfg)
    baseline = c9.evaluate_trace(model, old, recent, trained, conv, plan, cfg)
    changed = deepcopy(plan)
    i = next(i for i, row in enumerate(changed)
             if not row["reveal_to_all_after_prediction"])
    changed[i]["noise_evaluator_only"] += 0.73
    perturbed = c9.evaluate_trace(model, old, recent, trained, conv, changed, cfg)
    assert baseline["revealed_factual_samples"] == perturbed["revealed_factual_samples"]
    assert baseline["ledger_sha256"] != perturbed["ledger_sha256"]
    assert baseline["ledger"][i]["factual_consequence_evaluator_only"] != (
        perturbed["ledger"][i]["factual_consequence_evaluator_only"])
    for arm in c9.ARMS:
        assert (baseline["measurements"][arm]["pre_outcome_choices"]
                == perturbed["measurements"][arm]["pre_outcome_choices"])
    assert baseline["unrevealed_factual_outcomes_seen_by_router"] is False
    assert baseline["model_and_native_fast_writes"] == 0


def test_read_only_trace_uses_equal_actions_and_exact_factual_exposure():
    cfg, model, old, recent, trained, conv = _test_fixture()
    condition = cfg["stress"]["conditions"][0]
    plan = c9.make_trace(71931, 46111, condition, cfg)
    frozen = [deepcopy(x.state_dict()) for x in [model, trained, conv]]
    native_old, native_recent = deepcopy(old), deepcopy(recent)
    receipt = c9.evaluate_trace(model, old, recent, trained, conv, plan, cfg)
    assert receipt["status"] == "PASS_OBSERVATIONAL_TRACE_INTEGRITY_DEV"
    assert receipt["events"] == 96
    assert receipt["revealed_factual_samples"] == 96
    assert receipt["model_and_native_fast_writes"] == 0
    assert receipt["trained_model_gradient_updates_during_eval"] == 0
    assert receipt["current_outcome_seen_by_router"] is False
    assert receipt["ledger_sha256"] == hashlib.sha256(
        c9._canonical(receipt["ledger"])).hexdigest()
    assert len(receipt["hypotheses"]) == 5
    assert all(type(v) is bool for v in receipt["hypotheses"].values())
    for arm in c9.ARMS:
        scores = receipt["measurements"][arm]
        assert len(scores["raw_squared_errors"]) == 96
        assert len(scores["pre_outcome_choices"]) == 96
        assert scores["mature_samples"] + scores["immediate_samples"] == 96
    assert all(factual._exact_equal(m.state_dict(), prev)
               for m, prev in zip([model, trained, conv], frozen))
    assert factual._exact_equal(old, native_old)
    assert factual._exact_equal(recent, native_recent)
    assert list(inspect.signature(c8.signed_pair_route).parameters) == [
        "model", "completed_history", "history_mode"]


def test_scientific_first_use_denied(monkeypatch):
    for flag in ("NATIVE_C9_ENGINEERING_ONLY",
                 "NATIVE_P3_ENGINEERING_ONLY", "NATIVE_P7_DEV_ONLY"):
        monkeypatch.setenv(flag, "0")
        with pytest.raises(RuntimeError, match="SCIENCE_FIRST_USE_FORBIDDEN"):
            c9.run()
        monkeypatch.delenv(flag)
