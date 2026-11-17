"""R108 public-phase baseline invariants only; NOT a scored Hybrid experiment."""
import json
import pytest

from experiments import r108_public_phase_memory as r108


def test_r108_preregistered_novelty_and_dev_only_fence():
    evidence = r108.audit_freeze()
    assert evidence["new_seeds"] == [1008011, 1008023, 1008037]
    assert len(evidence["fresh_unseen_masks"]) == 8
    assert not evidence["protected_firstuse"] and not evidence["scientific_pass"]
    f = json.loads(r108.FREEZE.read_text(encoding="utf-8"))
    assert f["stage"] == "PROSPECTIVE_FROZEN_BEFORE_IMPLEMENTATION_UNIT_ONLY_DEV_NO_PROTECTED_FIRST_USE"
    assert not f["authority"]["protected_original_scientific_first_use"]
    assert not f["authority"]["GPU_training_authorized_by_this_freeze"]
    assert not f["authority"]["merge_authorized_by_this_freeze"]
    assert not f["authority"]["forge_integration_authorized"]
    assert not f["authority"]["deployment_authorized"]
    assert f["source"]["unseen_heldout_masks_hex"] == [
        "09", "db", "24", "4e", "ed", "b1", "e2", "d4"]


def test_public_factual_correction_does_not_erase_old_context():
    memory = r108.PublicPhaseMemory(update_weight=0.75)
    key = (1.2, -0.7, 0.8)
    memory.observe(0, key, 0, 4.0)
    old = memory.estimate(0, key, 0)
    assert old == 4.0
    memory.observe(1, key, 0, 4.0)
    memory.observe(1, key, 0, -4.0)
    assert memory.estimate(1, key, 0) == pytest.approx(-2.0)
    assert memory.estimate(0, key, 0) == old
    assert memory.estimate(0, key, 2) is None
    assert memory.counts.sum() == 3


def test_eight_address_public_mapping_and_no_hidden_law_labels():
    m = r108.PublicPhaseMemory()
    for index in range(8):
        factors = tuple(1.0 if (index >> shift) & 1 else -1.0
                        for shift in (2, 1, 0))
        m.observe(0, factors, 2, float(index))
    for index in range(8):
        factors = tuple(1.0 if (index >> shift) & 1 else -1.0
                        for shift in (2, 1, 0))
        assert m.estimate(0, factors, 2) == float(index)
        assert m.estimate(1, factors, 2) is None
    with pytest.raises(TypeError):
        m.observe(0, (1, 1, 1), 2, 1, private_operator_mask="01")


def test_public_context_and_physical_input_negative_controls():
    m = r108.PublicPhaseMemory()
    for context, role, action, reward in (
        (2, (1, -1, 1), 0, 1),
        (0, (1, 0, 1), 0, 1),
        (0, (1, -1), 0, 1),
        (0, (1, -1, 1), 1, 1),
        (0, (1, -1, 1), 0, float("nan")),
        (0, (1, -1, 1), 0, float("inf")),
        (0, (1, -1, 1), 0, 200)):
        with pytest.raises(ValueError):
            m.observe(context, role, action, reward)
    assert m.counts.sum() == 0


def test_cold_restore_keeps_exact_predictions_and_denies_tamper():
    key = (1, -1, 1)
    m = r108.PublicPhaseMemory()
    m.observe(0, key, 0, 4)
    m.observe(1, key, 0, -4)
    snapshot = m.snapshot()
    restored = r108.PublicPhaseMemory.cold_restore(snapshot)
    assert restored.snapshot() == snapshot
    for phase in (0, 1):
        for action in (0, 2):
            assert restored.estimate(phase, key, action) == m.estimate(phase, key, action)
    altered = json.loads(snapshot)
    altered["counts"][0][5][0] += 1
    with pytest.raises(ValueError):
        r108.PublicPhaseMemory.cold_restore(json.dumps(altered))
