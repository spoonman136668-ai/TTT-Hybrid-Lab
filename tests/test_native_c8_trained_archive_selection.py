"""C8 DEV: source-reused P7 factual-history router versus fully trained MLP."""
from copy import deepcopy
import hashlib
import inspect
import json

import pytest
import torch

from scripts import native_p7_learned_router_dev as p7
from scripts import native_c8_trained_archive_selection as c8
from scripts import native_c7_history_archive_selection as c7
from scripts import native_p3_continuity as factual


def test_original_p7_source_interfaces_exact_and_equal_sample_shape():
    cfg = c8._frozen()
    assert cfg["p7_original"]["pr"] == 62
    assert cfg["p7_original"]["script_git_blob"] == c8.P7_ORIGINAL_BLOB
    assert cfg["scientific"]["first_use_authorized"] is False
    assert p7.UPDATES == 600 and p7.BATCH == 256 and p7.HISTORY == 8
    assert sum(p.numel() for p in p7.LearnedRouter().parameters()) == 828
    assert sum(p.numel() for p in p7.ConventionalMLP().parameters()) == 849
    train = p7.synthetic_dev(p7.TRAIN_DEV_SEED, 1)
    assert train[0].shape == (40,16)
    assert train[1].shape == train[2].shape == (40,1)
    assert set(train[3].tolist()) == {-1,1} or set(train[3].tolist()).issubset({-1,1})
    assert c8._sampler_sha(303,6000) == c8._sampler_sha(303,6000)


def test_trained_pair_selection_only_takes_prior_completed_history():
    model = p7.LearnedRouter()
    conv = p7.ConventionalMLP()
    history = tuple((float(-1 if i % 2 == 0 else 1), float(i)/10)
                    for i in range(8))
    assert list(inspect.signature(c8.signed_pair_route).parameters) == [
        "model", "completed_history", "history_mode"]
    for m in [model, conv]:
        assert c8.signed_pair_route(m, ()) == 1
        assert c8.signed_pair_route(m, history) in (-1,1)
        assert c8.signed_pair_route(m, history, history_mode="reversed") in (-1,1)
        assert c8.signed_pair_route(m, history, history_mode="zero") in (-1,1)
        with pytest.raises(ValueError, match="HISTORY_ARGUMENT_INVALID"):
            c8.signed_pair_route(m, history, history_mode="oracle_label")
        with pytest.raises(TypeError):
            c8.signed_pair_route(m, history, "old_label")


def test_no_current_outcome_or_latent_label_in_actual_64_event_assay():
    torch.set_num_threads(1)
    model, _ = factual.new_model_and_optimizer(factual.MODEL_INITIALIZATIONS[0])
    base = model.initial_state(torch.zeros((1,1)))
    recent = {"recurrent": base.recurrent.clone(),
              "fast_weights": base.fast_weights.clone(),
              "history": base.history.clone()}
    old = deepcopy(recent)
    old["fast_weights"].add_(.01)
    learned = p7.LearnedRouter().eval()
    mlp = p7.ConventionalMLP().eval()
    frozen = [deepcopy(model.state_dict()), deepcopy(learned.state_dict()),
              deepcopy(mlp.state_dict())]
    cfg = c8._frozen()
    x = c8._read_only_route_eval(model, old, recent, learned, mlp,
                                  tuple(cfg["evaluation"]["orders"][0]), cfg)
    assert len(x["raw_event_ledger"]) == 64
    assert x["p6_controls_exact_source_parity"] is True
    assert x["current_factual_result_revealed_after_all_routes"] is True
    assert len(x["hypotheses"]) == 6
    assert all(type(h) is bool for h in x["hypotheses"].values())
    assert all(v["history_count_before_prediction"] == i
               for i,v in enumerate(x["raw_event_ledger"]))
    assert all(v["pre_outcome_route"]["p7_trained_router"] == 1
               and v["pre_outcome_route"]["trained_conventional_mlp"] == 1
               for v in x["raw_event_ledger"][:8])
    assert x["native_fast_memory_writes"] == x["native_optimizer_updates"] == 0
    assert all(factual._exact_equal(m.state_dict(), reference)
               for m,reference in zip([model,learned,mlp],frozen))
    for arm in c8.ARMS:
        assert len(x["measurements"][arm]["raw_squared_errors"]) == 64
        assert len(x["measurements"][arm]["pre_outcome_route_choices"]) == 64
        assert 0 <= x["measurements"][arm]["late_context_accuracy"] <= 1
    assert x["ledger_sha256"] == hashlib.sha256(
        c8._canonical(x["raw_event_ledger"])).hexdigest()


def test_c8_science_first_use_disabled_before_any_training(monkeypatch):
    for flag in ["NATIVE_C8_ENGINEERING_ONLY",
                 "NATIVE_P3_ENGINEERING_ONLY", "NATIVE_P7_DEV_ONLY"]:
        monkeypatch.setenv(flag, "0")
        with pytest.raises(RuntimeError, match="SCIENCE_FIRST_USE_FORBIDDEN"):
            c8.run()
        monkeypatch.delenv(flag)
