"""P40 exact-source negative-control DEV: exogenous drift is not an action effect.

No train/test random cohorts, no protected source first use, no parameter tuning.
Source mode labels and nuisance offsets are evaluator-only; P40 receives only
sensor, executed action, elapsed time and its corresponding realized outcome.
"""
import hashlib
import io
from pathlib import Path

import pytest
import torch

from native_hidden_mode_belief import FactualModeInference, HiddenModeBelief


SOURCE = Path(__file__).resolve().parents[1] / 'native_hidden_mode_belief.py'
SOURCE_BLOB = '8406c97361adc2b3caa13e3ef42b53044c75d0b0'
ACTIONS = (1.0, -1.0, 1.0)
BETA = .30


def original_blob(path):
    raw = path.read_bytes()
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\x00' + raw).hexdigest()


def model_and_initial():
    model = FactualModeInference(sensor_dim=1, capacity=2, novelty=.15,
                                noise=.05, stay_probability=.8, learning_rate=.25)
    sensor = torch.zeros((1, 1), dtype=torch.float32)
    dt = torch.ones_like(sensor)
    return model, sensor, dt, model.initial_state(sensor)


def factual_event(model, sensor, dt, state, *, action, mode, nuisance):
    assert action in (-1., 1.) and mode in (-1, 1)
    # The hidden mode and nuisance are used ONLY by the evaluator's real action step.
    physical_action = torch.tensor([[action]], dtype=sensor.dtype)
    realized = sensor + (BETA * mode * physical_action) + float(nuisance)
    return model.observe(sensor, physical_action, dt, realized, state), realized


def aba_estimate(completed):
    """Zero-fit conventional control: three equally timed, actual A,-A,A facts."""
    assert len(completed) == 3
    (a0, y0), (a1, y1), (a2, y2) = completed
    assert a0 == a2 == -a1
    return a0 * (0.5 * (y0 + y2) - y1) / 2.0


def test_exact_p40_original_source_and_clean_action_effect():
    assert original_blob(SOURCE) == SOURCE_BLOB
    model, sensor, dt, state = model_and_initial()
    for action in ACTIONS:
        state, _ = factual_event(model, sensor, dt, state,
                                 action=action, mode=1, nuisance=0.)
    assert state.factual_steps.item() == 3
    assert (state.observations > 0).sum().item() == 1
    assert state.effects[0, 0, 0].item() == pytest.approx(.3, abs=1e-6)


def test_zero_action_effect_sham_creates_opposite_false_causal_prototypes():
    model, sensor, dt, state = model_and_initial()
    # Evaluator's SHAM world is x_next=x+0.4 regardless of action.
    for action in (1., -1.):
        physical_action = torch.tensor([[action]])
        realized = sensor + .4
        state = model.observe(sensor, physical_action, dt, realized, state)
    assert state.factual_steps.item() == 2
    assert (state.observations > 0).sum().item() == 2
    torch.testing.assert_close(state.effects[0, :, 0],
                               torch.tensor([.4, -.4]), atol=1e-6, rtol=0)
    candidates = torch.tensor([[[-1.], [1.]]])
    speculative = model.preview(sensor, candidates, dt, state)
    assert (speculative[0, 0, 0] - speculative[0, 1, 0]).abs().item() > .6
    # Real evaluator would return .4 for BOTH factual actions, unlike preview.
    assert (sensor + .4).item() == pytest.approx(.4)
    assert state.factual_steps.item() == 2  # preview does not fake a source event


def test_one_stable_mode_affine_nuisance_exhausts_two_slots_before_third_fact():
    model, sensor, dt, state = model_and_initial()
    completed = []
    for action, nuisance in zip(ACTIONS[:2], (0., .4)):
        state, actual = factual_event(model, sensor, dt, state,
                                      action=action, mode=1, nuisance=nuisance)
        completed.append((action, float(actual.item())))
    assert state.factual_steps.item() == 2
    assert (state.observations > 0).sum().item() == 2
    effects_before = state.effects.clone()
    # Exactly the same underlying action law +0.30; only background drift grows.
    with pytest.raises(ValueError, match='P40_UNMODELED_DYNAMICS_CAPACITY'):
        factual_event(model, sensor, dt, state,
                      action=ACTIONS[2], mode=1, nuisance=.8)
    assert torch.equal(state.effects, effects_before)
    assert state.factual_steps.item() == 2
    # A zero-trained competitor cancels affine background after all 3 real facts.
    completed.append((ACTIONS[2], BETA * ACTIONS[2] + .8))
    assert aba_estimate(completed) == pytest.approx(BETA, abs=1e-6)
    # Unlike ABA, naive two-action contrast attributes time drift to causality.
    two_action_estimate = (completed[0][1] - completed[1][1]) / 2
    assert two_action_estimate == pytest.approx(.1, abs=1e-6)


def test_exact_cold_tensors_keep_capacity_failure_and_quadratic_drift_breaks_aba():
    model, sensor, dt, state = model_and_initial()
    for action, nuisance in zip(ACTIONS[:2], (0., .4)):
        state, _ = factual_event(model, sensor, dt, state,
                                 action=action, mode=1, nuisance=nuisance)
    store = io.BytesIO()
    torch.save({'effects':state.effects, 'observations':state.observations,
                'probabilities':state.probabilities, 'factual_steps':state.factual_steps},store)
    store.seek(0)
    restored = HiddenModeBelief(**torch.load(store, weights_only=True))
    assert all(torch.equal(getattr(state,name), getattr(restored,name))
               for name in ('effects','observations','probabilities','factual_steps'))
    with pytest.raises(ValueError, match='P40_UNMODELED_DYNAMICS_CAPACITY'):
        factual_event(model, sensor, dt, restored,
                      action=1., mode=1, nuisance=.8)
    # ABA assumes affine drift. Strong quadratic nuisance reverses its conclusion.
    curved = [(a, BETA*a - (t*t)) for t,a in enumerate(ACTIONS)]
    assert aba_estimate(curved) < 0  # explicit negative, not hidden generalization
