"""Engineering only. No optimizer step, training, downstream metric or heldout seeds."""
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import g7_s24_encoders as e


def test_pair_generator_is_deterministic_and_unlabeled():
    p = e.development_modality_pairs(5101, 8, 'source_train')
    p2 = e.development_modality_pairs(5101, 8, 'source_train')
    assert set(p.__dataclass_fields__) == {
        'sensor_a','sensor_b','graph_a_nodes','graph_a_edges','graph_b_nodes','graph_b_edges'}
    assert all(torch.equal(getattr(p, x), getattr(p2, x)) for x in p.__dataclass_fields__)
    assert p.sensor_a.shape == p.sensor_b.shape == (8, 32, 4)
    assert p.graph_a_nodes.shape == p.graph_b_nodes.shape == (8, 32, 6)
    assert not torch.equal(p.sensor_a[:,:,1:3], p.sensor_b[:,:,1:3])


def test_non_development_seed_rejected():
    with pytest.raises(ValueError, match='S24_PHASE1_DEVELOPMENT_SEED_ONLY'):
        e.development_modality_pairs(5201, 8, 'source_train')


def test_numeric_shape_and_finite_losses():
    torch.manual_seed(190)
    p = e.development_modality_pairs(5113, 20, 'sine_train')
    sensor, graph = e.SensorGRU(), e.EventMessagePassing()
    with torch.no_grad():
        s,g = e.independent_losses(sensor, graph, p)
    assert set(s) == set(g) == {'loss', 'alignment', 'variance', 'covariance'}
    assert all(torch.isfinite(x).all() for x in list(s.values())+list(g.values()))


def test_gradient_isolation_by_modality():
    torch.manual_seed(191)
    p = e.development_modality_pairs(5129, 16, 'rational_train')
    sensor, graph = e.SensorGRU(), e.EventMessagePassing()
    s,g = e.independent_losses(sensor, graph, p)
    s['loss'].backward()
    assert any(x.grad is not None and torch.isfinite(x.grad).all() for x in sensor.parameters())
    assert all(x.grad is None for x in graph.parameters())
    sensor.zero_grad(set_to_none=True)
    g['loss'].backward()
    assert any(x.grad is not None and torch.isfinite(x.grad).all() for x in graph.parameters())
    assert all(x.grad is None for x in sensor.parameters())


def test_graph_permutation_invariance():
    torch.manual_seed(192)
    p = e.development_modality_pairs(5101, 8, 'source_train')
    graph = e.EventMessagePassing().eval()
    permutation = torch.randperm(32)
    inverse = torch.argsort(permutation)
    with torch.no_grad():
        original = graph(p.graph_a_nodes, p.graph_a_edges)
        changed = graph(p.graph_a_nodes[:, permutation], inverse[p.graph_a_edges])
    assert torch.allclose(original, changed, atol=1e-6, rtol=1e-6)


def test_collapse_penalty_and_invalid_shape():
    a = torch.zeros(20,16)
    loss = e.within_modality_jepa(a,a)
    assert loss['variance'] > 0
    assert torch.isfinite(loss['loss'])
    with pytest.raises(ValueError, match='S24_INVALID_LATENT_SHAPE'):
        e.within_modality_jepa(torch.zeros(20,15),torch.zeros(20,15))
