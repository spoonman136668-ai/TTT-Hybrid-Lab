"""C10 DEV tests: unseen continuous action-law shifts without world first-use."""
from copy import deepcopy
import hashlib
import inspect
import math

import pytest
import torch

from scripts import native_c10_action_law_transfer_dev as c10
from scripts import native_p3_continuity as factual
from scripts import native_p7_learned_router_dev as p7
from scripts import native_c8_trained_archive_selection as c8


def _fixture():
    torch.set_num_threads(1)
    cfg = c10._frozen()
    model, _ = factual.new_model_and_optimizer(factual.MODEL_INITIALIZATIONS[0])
    s = model.initial_state(torch.zeros(1, 1))
    latest = {
        "recurrent": s.recurrent.detach().clone(),
        "fast_weights": s.fast_weights.detach().clone(),
        "history": s.history.detach().clone(),
    }
    prior = deepcopy(latest)
    prior["fast_weights"].add_(.02)
    learned = p7.LearnedRouter().eval()
    mlp = p7.ConventionalMLP().eval()
    return cfg, model, prior, latest, learned, mlp


def test_new_laws_are_declared_and_not_identical_to_training_law():
    cfg = c10._frozen()
    assert cfg["original_science_first_use_authorized"] is False
    assert cfg["training"]["original_p7_adamw_updates"] == 600
    assert cfg["transfer"]["conditions"] == [
        "linear_magnitude", "saturating", "deadzone"]
    for a in (-1., 1.):
        baseline = c10._law_outcome("linear_magnitude", a, "old", 0.)
        saturation = c10._law_outcome("saturating", a, "old", 0.)
        dead = c10._law_outcome("deadzone", a, "old", 0.)
        assert baseline == pytest.approx(saturation, abs=1e-12)
        assert baseline == pytest.approx(dead, abs=1e-12)
    x = .25
    shifted = [
        c10._law_outcome(c, x, "old", 0.)
        for c in cfg["transfer"]["conditions"]
    ]
    assert len(set(round(t,6) for t in shifted)) == 3
    for c in cfg["transfer"]["conditions"]:
        o = c10._law_outcome(c,.65,"old",0.)
        n = c10._law_outcome(c,.65,"recent",0.)
        assert o > n
    with pytest.raises(ValueError, match="UNKNOWN"):
        c10._law_outcome("unregistered", 1., "old", 0.)
    with pytest.raises(ValueError, match="NONFINITE"):
        c10._law_outcome("deadzone",float("nan"),"old",0.)


def test_deterministic_frozen_dev_trace_and_information_matched_ridge():
    cfg = c10._frozen()
    for law in cfg["transfer"]["conditions"]:
        a = c10.make_trace(123041,46111,law,cfg)
        b = c10.make_trace(123041,46111,law,cfg)
        assert a == b and len(a) == 96
        assert any(abs(e["action"]) < 1 for e in a)
        assert all(e["action"] in cfg["transfer"]["actions"] for e in a)
    assert list(inspect.signature(c10.past_only_ridge_choice).parameters) == [
        "history", "ridge", "floor"]
    old_hist = ((-1.,-.41),(1.,.49),(-.65,-.25),(.65,.33))
    new_hist = ((-1.,.49),(1.,-.41),(-.65,.33),(.65,-.25))
    assert c10.past_only_ridge_choice(old_hist) == 1
    assert c10.past_only_ridge_choice(new_hist) == -1
    assert c10.past_only_ridge_choice(()) == 1
    with pytest.raises(ValueError, match="TUPLE"):
        c10.past_only_ridge_choice(list(old_hist))
    with pytest.raises(ValueError, match="NONFINITE"):
        c10.past_only_ridge_choice(((1.,float("nan")),(1.,.5)))


def test_read_only_factual_ledger_and_current_outcome_order():
    cfg,model,old,new,learned,mlp = _fixture()
    cfg = dict(cfg)
    cfg["condition"] = "saturating"
    cfg["stream_seed"] = 123041
    plan = c10.make_trace(123041,46111,cfg["condition"],cfg)
    checkpoint = [deepcopy(x.state_dict()) for x in (model,learned,mlp)]
    original_old,original_new = deepcopy(old),deepcopy(new)
    first = c10._eval_trace(model,old,new,learned,mlp,plan,cfg)
    assert first["events"] == 96
    assert first["revealed_after_all_predictions"] is True
    assert first["native_optimizer_and_fast_memory_updates"] == 0
    assert len(first["diagnostic_hypotheses"]) == 6
    assert all(type(x) is bool for x in first["diagnostic_hypotheses"].values())
    assert first["ledger_sha256"] == hashlib.sha256(
        c10._canon(first["raw_prequential_ledger"])).hexdigest()
    assert all(x["number_of_completed_facts_before_route"] == i for i,x
               in enumerate(first["raw_prequential_ledger"]))
    assert all(first["raw_prequential_ledger"][i]["route_chosen_before_current_result"][
               "trained_p7"] == 1 for i in range(8))
    assert all(factual._exact_equal(model_.state_dict(),sd)
               for model_,sd in zip((model,learned,mlp),checkpoint))
    assert factual._exact_equal(old,original_old)
    assert factual._exact_equal(new,original_new)
    for name in c10.ARMS:
        assert len(first["metrics"][name]["squared_errors"]) == 96
        assert len(first["metrics"][name]["routes_before_current_outcome"]) == 96
    # Current outcome can change this event's loss but not its routing.
    altered=deepcopy(plan)
    altered[20]["noise_evaluator_only"]+=1.3
    second=c10._eval_trace(model,old,new,learned,mlp,altered,cfg)
    for arm in c10.ARMS:
        assert (first["metrics"][arm]["routes_before_current_outcome"][:21]
                == second["metrics"][arm]["routes_before_current_outcome"][:21])
    assert first["raw_prequential_ledger"][20]["squared_errors"] != (
        second["raw_prequential_ledger"][20]["squared_errors"])


def test_scientific_first_use_denied_before_training(monkeypatch):
    for flag in ("NATIVE_C10_ENGINEERING_ONLY",
                 "NATIVE_P3_ENGINEERING_ONLY", "NATIVE_P7_DEV_ONLY"):
        monkeypatch.setenv(flag,"0")
        with pytest.raises(RuntimeError,match="FIRST_SCIENTIFIC_USE_FORBIDDEN"):
            c10.run()
        monkeypatch.delenv(flag)
