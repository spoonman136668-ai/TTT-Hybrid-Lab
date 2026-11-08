"""Synthetic engineering controls only: no project science seeds or training runs."""
import ast
import io
from pathlib import Path
import torch
import pytest
from native_continuous_state import NativeContinuousHybrid, ContinuousState


def fixture():
    torch.manual_seed(140)
    net = NativeContinuousHybrid(4, 2, width=12, history_limit=3).eval()
    sensor = torch.tensor([[.1, .2, .3, .4], [-.2, 0., .7, 1.]], dtype=torch.float32)
    action = torch.tensor([[.5, -.5], [.2, .3]], dtype=torch.float32)
    dt = torch.tensor([[.1], [2.]], dtype=torch.float32)
    return net, sensor, action, dt


def test_real_continuous_physical_input_and_gradients():
    net, obs, act, dt = fixture()
    obs.requires_grad_()
    pred, state = net.step(obs, act, dt)
    assert pred.shape == obs.shape
    assert torch.isfinite(pred).all()
    net.transition_loss(pred, obs * .9).backward()
    assert obs.grad is not None and torch.isfinite(obs.grad).all()
    assert net.dynamics.weight.grad is not None
    assert state.history.shape == (2, 1, 12)


def test_persistent_state_and_cold_restore_exact():
    net, obs, act, dt = fixture()
    with torch.no_grad():
        state = None
        for _ in range(8):
            _, state = net.step(obs, act, dt, state)
        assert state.history.shape == (2, 3, 12)
        first, _ = net.step(obs, act, dt, state)
        buffer = io.BytesIO()
        torch.save({'model': net.state_dict(), 'state': {
            'recurrent': state.recurrent, 'fast_weights': state.fast_weights,
            'history': state.history}}, buffer)
        buffer.seek(0)
        payload = torch.load(buffer, weights_only=True)
        restored = NativeContinuousHybrid(4, 2, width=12, history_limit=3).eval()
        restored.load_state_dict(payload['model'])
        cold = ContinuousState(**payload['state'])
        second, _ = restored.step(obs, act, dt, cold)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        fresh, _ = net.step(obs, act, dt)
        assert not torch.allclose(first, fresh)


def test_read_before_outcome_and_delayed_learning():
    net, obs, act, dt = fixture()
    with torch.no_grad():
        before, state = net.step(obs, act, dt)
        later = net.learn_outcome(state, act, before, before + .3)
        assert torch.count_nonzero(state.fast_weights) == 0
        changed, _ = net.step(obs, act, dt, later)
        original, _ = net.step(obs, act, dt, state)
        assert not torch.allclose(changed, original)
        torch.testing.assert_close(changed-original, torch.full_like(before, .3*net.fast_rate), atol=1e-6, rtol=0)


def test_bounded_history_no_cross_episode_contamination():
    net, obs, act, dt = fixture()
    with torch.no_grad():
        pred1, _ = net.step(obs, act, dt)
        other, _ = net.step(obs * 3., act, dt)
        pred2, _ = net.step(obs, act, dt)
        torch.testing.assert_close(pred1, pred2)
        assert not torch.allclose(pred1, other)


def test_action_intervention_and_nonpositive_dt_controls():
    net, obs, act, dt = fixture()
    with torch.no_grad():
        base, _ = net.step(obs, act, dt)
        changed, _ = net.step(obs, -act, dt)
        assert not torch.allclose(base, changed)  # sensitivity only, NOT causal learning
    with pytest.raises(ValueError, match='NONPOSITIVE'):
        net.step(obs, act, dt * 0)


def test_integer_lexical_token_and_nonfinite_rejected():
    net, obs, act, dt = fixture()
    with pytest.raises(ValueError, match='NONCONTINUOUS'):
        net.step(obs.long(), act, dt)
    with pytest.raises(ValueError, match='NONCONTINUOUS'):
        net.step(obs, act.long(), dt)
    bad = obs.clone(); bad[0, 0] = float('nan')
    with pytest.raises(ValueError, match='INVALID_SENSOR'):
        net.step(bad, act, dt)
    with pytest.raises(ValueError, match='DEVICE_OR_DTYPE_DRIFT'):
        net.step(obs.double(), act.double(), dt.double())
    with pytest.raises(ValueError, match='STATE_SHAPE_DRIFT'):
        net.step(obs, act, dt, ContinuousState(obs, obs, obs))


def test_no_tokenizer_embeddings_or_next_token_objective_in_native_path():
    src = Path(__file__).resolve().parents[1] / 'native_continuous_state.py'
    tree = ast.parse(src.read_text())
    attrs = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
    assert 'Embedding' not in attrs
    assert 'CrossEntropyLoss' not in attrs
    assert 'multinomial' not in attrs
    for name in ('tokenizer', 'vocabulary', 'token_ids', 'next_token_logits'):
        assert name not in [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)]
