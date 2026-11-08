"""Synthetic API negative controls; do not instantiate any scientific world."""
import json
from pathlib import Path

import pytest
import torch
from native_continuous_state import NativeContinuousHybrid
from scripts.native_prequential_gate import NativeCausalEpisode, PredictionTicket


def fixture():
    torch.manual_seed(461)
    model = NativeContinuousHybrid(sensor_dim=2, action_dim=1, width=8, history_limit=2)
    sensor = torch.tensor([[.25, -.75]], dtype=torch.float32)
    action = torch.tensor([[.7]], dtype=torch.float32)
    dt = torch.tensor([[.4]], dtype=torch.float32)
    return NativeCausalEpisode(model), sensor, action, dt


def test_prediction_before_observed_update():
    ep, s, a, dt = fixture()
    t = ep.predict(s, a, dt)
    assert torch.count_nonzero(ep.snapshot().fast_weights) == 0
    ep.observe(t, t.prediction + .25, factual=True)
    assert torch.count_nonzero(ep.snapshot().fast_weights) > 0


def test_second_prediction_cannot_overwrite_pending_factual_event():
    ep, s, a, dt = fixture()
    t = ep.predict(s, a, dt)
    with pytest.raises(RuntimeError, match='PENDING_OUTCOME'):
        ep.predict(s, -a, dt)
    ep.observe(t, t.prediction, factual=True)
    ep.predict(s, a, dt)


def test_stale_foreign_and_forged_tickets_rejected():
    ep, s, a, dt = fixture()
    t = ep.predict(s, a, dt)
    other, *_ = fixture()
    foreign = other.predict(s, a, dt)
    for invalid in (foreign, PredictionTicket(t.event_index, t.prediction)):
        with pytest.raises(RuntimeError, match='STALE_OR_FOREIGN'):
            ep.observe(invalid, t.prediction, factual=True)
    ep.observe(t, t.prediction, factual=True)
    with pytest.raises(RuntimeError, match='STALE_OR_FOREIGN'):
        ep.observe(t, t.prediction, factual=True)


def test_counterfactual_feedback_forbidden_without_consuming_ticket():
    ep, s, a, dt = fixture()
    t, alt = ep.predict_pair(s, a, -a, dt)
    assert not torch.allclose(t.prediction, alt)
    with pytest.raises(ValueError, match='COUNTERFACTUAL'):
        ep.observe(t, alt, factual=False)
    assert torch.count_nonzero(ep.snapshot().fast_weights) == 0
    ep.observe(t, t.prediction + .2, factual=True)
    assert torch.count_nonzero(ep.snapshot().fast_weights) > 0


def test_counterfactual_has_no_state_or_history_side_effect():
    ep, s, a, dt = fixture()
    paired, alt = ep.predict_pair(s, a, -a, dt)
    paired_state = ep.snapshot()
    ctrl, s2, a2, dt2 = fixture()
    factual = ctrl.predict(s2, a2, dt2)
    torch.testing.assert_close(paired.prediction, factual.prediction, atol=0, rtol=0)
    torch.testing.assert_close(paired_state.recurrent, ctrl.snapshot().recurrent, atol=0, rtol=0)
    torch.testing.assert_close(paired_state.history, ctrl.snapshot().history, atol=0, rtol=0)
    assert alt.shape == factual.prediction.shape


def test_invalid_outcome_does_not_consume_ticket():
    ep, s, a, dt = fixture()
    t = ep.predict(s, a, dt)
    bad = t.prediction.clone()
    bad[0, 0] = float('nan')
    with pytest.raises(ValueError, match='INVALID_OBSERVED_OUTCOME'):
        ep.observe(t, bad, factual=True)
    assert torch.count_nonzero(ep.snapshot().fast_weights) == 0
    ep.observe(t, t.prediction + .1, factual=True)


def test_no_external_mutation_of_ticket_or_action_affects_learning():
    ep, s, a, dt = fixture()
    t = ep.predict(s, a, dt)
    original = t.prediction.clone()
    a.fill_(999)
    t.prediction.fill_(999)
    ep.observe(t, original + .4, factual=True)
    corrected = ep.snapshot().fast_weights.clone()
    ctrl, s2, a2, dt2 = fixture()
    expected = ctrl.predict(s2, a2, dt2)
    ctrl.observe(expected, expected.prediction + .4, factual=True)
    torch.testing.assert_close(corrected, ctrl.snapshot().fast_weights, atol=0, rtol=0)


def test_snapshot_defensive_copy_and_reset_requires_consumed_outcome():
    ep, s, a, dt = fixture()
    t = ep.predict(s, a, dt)
    external = ep.snapshot()
    external.recurrent.fill_(999)
    assert torch.max(torch.abs(ep.snapshot().recurrent)) < 999
    with pytest.raises(RuntimeError, match='PENDING_OUTCOME'):
        ep.reset()
    ep.observe(t, t.prediction, factual=True)
    ep.reset()
    assert ep.snapshot() is None


def test_bad_time_and_noncontinuous_sensor_fail_before_ticket_creation():
    ep, s, a, dt = fixture()
    with pytest.raises(ValueError, match='NONPOSITIVE'):
        ep.predict(s, a, dt * 0)
    with pytest.raises(ValueError, match='NONCONTINUOUS'):
        ep.predict(s.long(), a, dt)
    t = ep.predict(s, a, dt)
    assert t.event_index == 1


def test_manifest_is_pre_firstuse_and_zero_science_seed_material():
    root = Path(__file__).resolve().parents[1]
    p = json.loads((root / 'experiments/native_prequential_p0_engineering_frozen.json').read_text())
    assert p['schema'] == 'ttt-hybrid.native-prequential-p0.engineering-frozen.v1'
    assert p['science_first_use']['authorized'] is False
    assert p['science_first_use']['seed_assignments'] == {}
    assert p['science_first_use']['world_data_consumed'] is False
    assert p['scope']['model_training'] is False
    assert p['scope']['research_loop_integration'] is False
    assert p['source_lineage']['native_v1_sha'] == '7df69c701dd80799b0d1ae49cc0b7c51536051df'
