"""P85-MR source-free conventional CartPoleWind dynamics controls.

This is model and objective implementation ONLY. It has no dataset discovery,
external-source import, trainer main, saved checkpoints, OOD selector, or
autonomous execution. Toy gradient checks are not external science results.

Static source dimensions/equations are frozen in
experiments/p85mr_strong_baseline_freeze.json. Reuse exact P69 GRU/SSM and
P75 prequential-before-write models rather than duplicating architectures.
"""
import torch
from torch import nn
import torch.nn.functional as F

from p69_trainable_competitors import TrainedGRU, DiagonalContinuousSSM, _facts
from p75_prequential_baselines import before_write_predictions

SEEDS = (985201, 985203, 985207)
STATE = 256
KEY = 16
OBSERVED = 7
ACTIVE_KEY = 7
PAD = 9
KINDS = ("linear7", "mlp7", "gru256", "ssm256")
OBJECTIVE_WEIGHTS = (1.0, 0.2, 0.1, 0.1)


def _validate_factual(facts):
    """Only P80 factual action/effect tensors, no future or oracle fields."""
    a, y = _facts(facts)
    if a.shape[1] < 2:
        raise ValueError("P85MR_MIN_TWO_FACTUAL_EVENTS")
    if (not torch.allclose(a[..., ACTIVE_KEY:], torch.zeros_like(a[..., ACTIVE_KEY:]),
                           rtol=0, atol=1e-7) or
        not torch.allclose(y[..., OBSERVED:], torch.zeros_like(y[..., OBSERVED:]),
                           rtol=0, atol=1e-7) or
        not bool(((a.norm(dim=-1)-1).abs()<1e-4).all()) or
        not bool((a[..., 0]>0).all())):
        raise ValueError("P85MR_INVALID_P80_EMBEDDING")
    return a, y


class StatelessDynamics(nn.Module):
    """State/action-conditioned conventional controls with ZERO episodic memory.

    Their lack of a persistent recurrent state is a known model advantage for
    a fully observed Markov process, not a defect to hide in comparison.
    """
    def __init__(self, kind):
        super().__init__()
        if kind == "linear7":
            self.net = nn.Linear(ACTIVE_KEY, OBSERVED)
        elif kind == "mlp7":
            self.net = nn.Sequential(
                nn.Linear(ACTIVE_KEY, 32), nn.SiLU(), nn.Linear(32, OBSERVED))
        else:
            raise ValueError("P85MR_UNKNOWN_MODEL")
        self.kind = kind
        self.persistent_state_floats = 0

    def forward(self, actions):
        if (not isinstance(actions, torch.Tensor) or actions.ndim not in (2, 3)
                or actions.shape[-1] != KEY or not actions.is_floating_point()
                or not bool(torch.isfinite(actions).all())
                or not bool((actions[..., 0] > 0).all())
                or not bool(((actions.norm(dim=-1)-1).abs() < 1e-4).all())
                or not torch.allclose(actions[..., ACTIVE_KEY:],
                                      torch.zeros_like(actions[..., ACTIVE_KEY:]),
                                      atol=1e-7, rtol=0)):
            raise ValueError("P85MR_BAD_P80_ACTION_KEY")
        p = self.net(actions[..., :ACTIVE_KEY])
        return F.pad(p, (0, PAD))

    def read(self, action):
        return self.forward(action)


def new_baseline(kind, seed):
    """Exact architecture registered before science FIT data exists."""
    if kind not in KINDS or type(seed) is not int or seed not in SEEDS:
        raise ValueError("P85MR_UNREGISTERED_MODEL_OR_SEED")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if kind in ("linear7", "mlp7"):
            model = StatelessDynamics(kind)
        elif kind == "gru256":
            model = TrainedGRU()
        else:
            model = DiagonalContinuousSSM()
    return model


def model_metadata(model):
    if isinstance(model, StatelessDynamics):
        state = 0
        kind = model.kind
    elif isinstance(model, TrainedGRU):
        state = STATE
        kind = "gru256"
    elif isinstance(model, DiagonalContinuousSSM):
        state = STATE
        kind = "ssm256"
    else:
        raise ValueError("P85MR_UNREGISTERED_MODEL")
    return {
        "kind": kind,
        "persistent_floats_per_world": state,
        "trainable_static_parameters": sum(p.numel() for p in model.parameters()),
        "external_science_qualified": False,
        "uses_observed_effect_before_current_prediction": False,
    }


def prequential(model, facts):
    """Predict y[t] with only historical facts i<t and current action a[t].

    Historical factual y[i<t] is legitimate memory information. y[t] is
    evaluator/optimizer label *after* model prediction, not an input.
    """
    actions, _ = _validate_factual(facts)
    if isinstance(model, StatelessDynamics):
        pred = model(actions)
    elif isinstance(model, (TrainedGRU, DiagonalContinuousSSM)):
        pred, _ = before_write_predictions(model, facts)
    else:
        raise ValueError("P85MR_UNREGISTERED_MODEL")
    if pred.shape != actions.shape or not bool(torch.isfinite(pred).all()):
        raise ValueError("P85MR_BAD_PREDICTION")
    return pred


def component_errors(pred, observed):
    """Per-timestep registered physical losses; explicitly exclude 9 pads.

    This is NOT scientific acceptance scoring; it is a differentiable source-
    free objective on in-memory toy tensors. No source acquisition exists.
    """
    if (pred.shape != observed.shape or pred.ndim != 3 or pred.shape[-1] != KEY
            or not bool(torch.isfinite(pred).all())
            or not bool(torch.isfinite(observed).all())):
        raise ValueError("P85MR_INVALID_LOSS_TENSORS")
    squared = (pred - observed).square()
    return {
        "deltas": squared[..., :4].mean(dim=-1),
        "reward": squared[..., 4],
        "terminated": squared[..., 5],
        "truncated": squared[..., 6],
    }


def factual_objective(model, facts):
    """Frozen FIT objective: exclude first event with no historical prefix."""
    _, y = _validate_factual(facts)
    pred = prequential(model, facts)
    vals = component_errors(pred[:, 1:], y[:, 1:])
    loss = (vals["deltas"].mean() +
            .2 * vals["reward"].mean() +
            .1 * vals["terminated"].mean() +
            .1 * vals["truncated"].mean())
    if not bool(torch.isfinite(loss)):
        raise ValueError("P85MR_NONFINITE_TOY_OBJECTIVE")
    return {"loss": loss, "delta": vals["deltas"].mean(),
            "reward": vals["reward"].mean(),
            "terminated": vals["terminated"].mean(),
            "truncated": vals["truncated"].mean(),
            "first_history_index": 1,
            "padding_in_primary_metric": False,
            "science_pass": False}
