"""Source-only S24 validation: these tests do not train or score a model."""
from pathlib import Path
import sys
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import g7_s24_views as v


@pytest.mark.parametrize('regime', [
    'source_train', 'sine_train', 'rational_train',
    'source_ood', 'sine_ood', 'rational_ood',
])
def test_byte_equivalent_s20_original_history(regime):
    assert v.reference_equivalent_trace(5101, 4, regime)


@pytest.mark.parametrize('regime', ['source_train', 'sine_train', 'rational_train'])
def test_two_independent_views_have_only_numeric_encoder_inputs(regime):
    pair = v.independent_views(5101, 4, regime)
    repeat = v.independent_views(5101, 4, regime)
    assert tuple(pair.sensor_sequence.shape) == (4, 32, 4)
    assert tuple(pair.event_nodes.shape) == (4, 32, 6)
    assert tuple(pair.event_edges.shape) == (2, 62)
    assert all(torch.equal(getattr(pair, f), getattr(repeat, f)) for f in pair.__dataclass_fields__)
    assert set(pair.__dataclass_fields__) == {'sensor_sequence', 'event_nodes', 'event_edges'}
    chronological = torch.argsort(pair.event_nodes[0, :, 5])
    assert not torch.equal(pair.sensor_sequence[:, :, 1:3],
                           pair.event_nodes[:, chronological, 1:3])
    assert torch.isfinite(pair.sensor_sequence).all()
    assert torch.isfinite(pair.event_nodes).all()


def test_same_trace_control_does_not_use_independent_interventions():
    pair = v.same_trace_bad_views(5101, 4, 'source_train')
    # The graph's permutation must be inverted by its normalized time field.
    order = torch.argsort(pair.event_nodes[0, :, 5])
    assert torch.equal(pair.sensor_sequence[:, :, :4], pair.event_nodes[:, order, :4])
    good = v.independent_views(5101, 4, 'source_train')
    good_order = torch.argsort(good.event_nodes[0, :, 5])
    assert not torch.equal(good.sensor_sequence[:, :, 1:3], good.event_nodes[:, good_order, 1:3])


def test_graph_permutation_retains_chronological_adjacency():
    pair = v.independent_views(5113, 2, 'rational_train')
    node_time = pair.event_nodes[0, :, 5]
    src, dst = pair.event_edges
    assert torch.allclose((node_time[src] - node_time[dst]).abs(),
                          torch.full((62,), 1 / 31), atol=1e-6)
    assert pair.event_nodes.shape[-1] == 6


def test_forbid_invalid_regimes_before_world_generation():
    with pytest.raises(ValueError, match='UNREGISTERED_S24_REGIME'):
        v.independent_views(5101, 4, 'bad_regime')
