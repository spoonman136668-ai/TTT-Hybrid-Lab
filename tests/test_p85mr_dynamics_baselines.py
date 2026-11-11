"""P85-MR stronger domain controls: all data authored in memory, NO external source."""
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from p85mr_dynamics_baselines import (
    KINDS, SEEDS, StatelessDynamics, new_baseline, model_metadata,
    prequential, component_errors, factual_objective,
)
from p80mr_causalgym_trace_adapter import _action_key, _observed_effect

torch.set_num_threads(1)


def toy(batch=2, steps=5):
    """Deterministic fictional physical states, actions and factual effects."""
    keys, results = [], []
    for b in range(batch):
        world_keys, world_results = [], []
        state = [0.02*b, 0.04, 0.001, 0.01]
        for t in range(steps):
            act = (t + b) % 2
            after = [
                state[0] + (0.06 if act else -0.04),
                state[1] + (0.03 if act else -0.02),
                state[2] + (0.002 if act else -0.001),
                state[3] + (0.005 if act else -0.004),
            ]
            world_keys.append(_action_key(state, act))
            world_results.append(_observed_effect(
                state, after, 1.0 + .02*t, False, False))
            state = after
        keys.append(world_keys)
        results.append(world_results)
    return {
        "action": torch.tensor(keys, dtype=torch.float32),
        "effect": torch.tensor(results, dtype=torch.float32),
    }


@pytest.mark.parametrize("kind,params,state", [
    ("linear7", 56, 0), ("mlp7", 487, 0),
    ("gru256", None, 256), ("ssm256", None, 256),
])
def test_declared_architecture_capacity_and_input_and_output(kind, params, state):
    m = new_baseline(kind, SEEDS[0])
    info = model_metadata(m)
    assert info["kind"] == kind
    assert info["persistent_floats_per_world"] == state
    assert info["trainable_static_parameters"] == params if params else info["trainable_static_parameters"] > 1000
    assert info["external_science_qualified"] is False
    preds = prequential(m, toy())
    assert preds.shape == (2, 5, 16)
    assert torch.isfinite(preds).all()
    if state == 0:
        assert torch.equal(preds[..., 7:], torch.zeros_like(preds[..., 7:]))


@pytest.mark.parametrize("kind", KINDS)
def test_deterministic_seed_scoped_initialization_without_global_rng_drift(kind):
    previous = torch.random.get_rng_state().clone()
    m1 = new_baseline(kind, SEEDS[0])
    assert torch.equal(previous, torch.random.get_rng_state())
    m2 = new_baseline(kind, SEEDS[0])
    m3 = new_baseline(kind, SEEDS[1])
    assert all(torch.equal(a, b) for a, b in zip(m1.parameters(), m2.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(m1.parameters(), m3.parameters()))


@pytest.mark.parametrize("bad_kind,bad_seed", [
    ("oracle", SEEDS[0]), ("linear7", 981201),
    ("gru256", True), ("ssm256", None),
])
def test_no_unregistered_architecture_or_seeds(bad_kind,bad_seed):
    with pytest.raises(ValueError, match="P85MR_UNREGISTERED_MODEL_OR_SEED"):
        new_baseline(bad_kind,bad_seed)


@pytest.mark.parametrize("kind", KINDS)
def test_optimizable_factual_prequential_loss_on_toy(kind):
    m = new_baseline(kind, SEEDS[0])
    losses = factual_objective(m, toy())
    assert losses["first_history_index"] == 1
    assert losses["padding_in_primary_metric"] is False
    assert losses["science_pass"] is False
    assert losses["loss"].isfinite()
    losses["loss"].backward()
    gradients = [p.grad for p in m.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)


@pytest.mark.parametrize("kind", KINDS)
def test_current_outcome_is_not_used_for_current_prediction(kind):
    original = toy(2, 6)
    altered = {k: v.clone() for k, v in original.items()}
    altered["effect"][:, 3, 0] += 2.0
    m = new_baseline(kind, SEEDS[0])
    with torch.no_grad():
        a = prequential(m, original)
        b = prequential(m, altered)
    assert torch.equal(a[:, :4], b[:, :4])
    if kind in ("gru256", "ssm256"):
        assert not torch.equal(a[:, 4:], b[:, 4:])
    else:
        assert torch.equal(a, b)


@pytest.mark.parametrize("kind", ("gru256", "ssm256"))
def test_no_future_and_recurrent_prefix_readout_match(kind):
    facts = toy(2, 6)
    m = new_baseline(kind, SEEDS[0]).eval()
    with torch.no_grad():
        full_pred = prequential(m, facts)
        for t in (1, 2, 4, 5):
            prefix = {k: v[:, :t] for k, v in facts.items()}
            history = m.encode(prefix)
            assert history.shape == (2, 256)
            prediction = m.read(history, facts["action"][:, t])
            assert torch.allclose(full_pred[:, t], prediction, rtol=3e-5, atol=2e-6)
        entire = m.encode(facts)
        left = m.encode({k:v[:,:3] for k,v in facts.items()})
        right = m.encode({k:v[:,3:] for k,v in facts.items()}, left.clone())
        assert torch.allclose(entire, right, atol=2e-6, rtol=3e-5)


@pytest.mark.parametrize("kind", ("linear7", "mlp7"))
def test_toy_fit_can_reduce_observed_causal_dynamics_error(kind):
    facts = toy(4, 8)
    model = new_baseline(kind, SEEDS[0])
    opt = torch.optim.AdamW(model.parameters(),lr=.02,weight_decay=.0)
    initial=float(factual_objective(model,facts)["loss"])
    for _ in range(45):
        loss=factual_objective(model,facts)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        opt.step()
    final=float(factual_objective(model,facts)["loss"])
    assert final < .7*initial
    # Training on these hand-authored toys proves gradient plumbing only.


def test_primary_metric_never_counts_nine_padded_channels():
    truth = toy(2, 5)["effect"]
    pred = torch.zeros_like(truth)
    first = component_errors(pred,truth)["deltas"]
    pred[...,7:] = 999.0
    second = component_errors(pred,truth)["deltas"]
    assert torch.equal(first, second)
    assert torch.equal(component_errors(pred,truth)["reward"],
                       component_errors(torch.zeros_like(pred),truth)["reward"])


@pytest.mark.parametrize("mutation", ["extra_key","future_target","nonunit","padding","bad_effect","short"])
def test_factual_input_violations_fail_closed(mutation):
    f = toy(2,5)
    if mutation=="extra_key":
        f["oracle_changepoint"] = torch.ones(2,5)
    elif mutation=="future_target":
        f["query_action"] = f["action"][:, -1]
    elif mutation=="nonunit":
        f["action"][:,0] *= 0.5
    elif mutation=="padding":
        f["action"][:,:,7] = 0.1
    elif mutation=="bad_effect":
        f["effect"][:,:,12] = 0.1
    elif mutation=="short":
        f={k:v[:,:1] for k,v in f.items()}
    with pytest.raises(ValueError, match="P69_HISTORICAL_FACTS_ONLY|P85MR_INVALID_P80_EMBEDDING|P85MR_MIN_TWO_FACTUAL_EVENTS"):
        factual_objective(new_baseline("linear7",SEEDS[0]),f)


def test_no_environment_loading_training_entrypoint_or_heldout_access():
    s=(EXPERIMENTS/"p85mr_dynamics_baselines.py").read_text()
    for blocked in ("import causal_gym","import gymnasium",
                    "env.reset(","env.do(","subprocess","requests.",
                    "make_world(","make_batch(","science_ood",
                    "if __name__","torch.save","torch.load",
                    "optimizer.step(","fit_val_only(","runner.dispatch"):
        assert blocked not in s, blocked


def test_cannot_claim_scientific_pass_from_toy_losses():
    x=factual_objective(new_baseline("mlp7",SEEDS[2]),toy())
    assert x["science_pass"] is False
    assert model_metadata(new_baseline("gru256",SEEDS[2]))["external_science_qualified"] is False
