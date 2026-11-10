"""P67 focused engineering controls; not a frozen study or scientific first use."""
import importlib.util
from pathlib import Path
import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / 'experiments' / 'p67_counterbalanced_memory_dev.py'
spec = importlib.util.spec_from_file_location('p67', PATH)
p67 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p67)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def test_repeatable_source_and_clean_factual_view():
    world = p67.make_world('fit', 8)
    assert p67.source_hash(world) == p67.source_hash(p67.make_world('fit', 8))
    assert set(p67.factual(world)) == {'action', 'effect'}
    assert world['query_action'].shape == (8, 16)
    assert world['observed_query_effect'].shape == (8, 16)
    assert world['action'].shape == (8, 64, 16)
    assert world['role_audit'].sum().item() == 8 * 32


def test_randomized_role_counterbalance_and_index_shortcut():
    for split in ('fit', 'val', 'ood_dev', 'negative'):
        data = p67.make_world(split)
        audit = p67.static_controls(data)
        assert audit['no_phase_shortcut'], (split, audit)
        assert len(audit['relevant_fraction_quartiles']) == 4


def test_different_training_initializations_and_gradient():
    hashes = [p67.digest_tensor(p67.WriteGate(seed).theta) for seed in p67.MODEL_SEEDS]
    assert len(set(hashes)) == 3
    gate = p67.WriteGate(p67.MODEL_SEEDS[0])
    loss = p67.loss_gate(gate, p67.make_world('fit', 5))
    loss.backward()
    assert torch.isfinite(loss)
    assert gate.theta.grad is not None and torch.isfinite(gate.theta.grad).all()


def test_wrong_credit_and_cold_restore():
    world = p67.make_world('negative', 4)
    gate = p67.WriteGate(p67.MODEL_SEEDS[0])
    assert p67.restore_exact(gate, world)
    good = p67.loss_gate(gate, world)
    bad = p67.loss_gate(gate, world, wrong_credit=True)
    assert abs(float(good - bad)) > 1e-8


def test_input_validation_and_no_oracle_model_arm():
    gate = p67.WriteGate(970201)
    w = p67.make_world('fit', 4)
    with pytest.raises(ValueError, match='P67_BAD_RANGE'):
        gate(p67.factual(w), lo=-1)
    with pytest.raises(ValueError, match='P67_BAD_MODE'):
        gate(p67.factual(w), mode='oracle')
    with pytest.raises(ValueError, match='P67_NONFINITE_FACTS'):
        bad = p67.factual(w)
        bad['action'] = bad['action'].clone()
        bad['action'][0, 0, 0] = float('nan')
        gate(bad)


def test_short_training_paths_engineering_only():
    fit = p67.make_world('fit', 8)
    val = p67.make_world('val', 8)
    indices = torch.randint(8, (2, 4), generator=torch.Generator().manual_seed(10))
    for kind, model in [('gate', p67.WriteGate(970201)),
                        ('wrong', p67.WriteGate(971001)),
                        ('gru', p67.GRUBaseline(970201))]:
        result = p67.train(model, kind, fit, val, indices, max_updates=2)
        assert result['steps'] == 2
        assert not result['early_stopped']
        assert result['parameters'] > 0